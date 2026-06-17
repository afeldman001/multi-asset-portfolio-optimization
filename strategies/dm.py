# strategies/dm.py
#
# data-and-model (dm) strategy in the "risk-based model" framework using closed-form
# posterior and predictive moments (pástor and stambaugh, 1999)
#
# implementation:
# - expected returns and covariances are not treated as raw sample moments
# - instead, a pricing model provides a prior restriction on expected excess returns, 
#   described by the multivariate regression:
#     r_1,t ≈ alpha + B_i * r_2,t
#   with mispricing alpha shrunk toward zero by a normal prior (pástor and stambaugh, 1999, eq. 15, 19)
# - posterior moments are computed from the predictive return distribution (Eq. 14),
#   yielding closed-form predictive moments for risk-based models:
#     predictive mean:       mu* = E[r_{T+1} | R]
#     predictive covariance: V* = E[V | R] + Var(E | R)
#   (pástor and stambaugh, 1999, eq. 14; applied in demiguel, garlappi, and uppal, 2009)
#
# factor choice:
# - capm is implemented as the baseline single-factor case (k=1) consistent with the
#   thesis design decision to use a tradable market proxy (demiguel et al., 2009)
# - adding additional factors (e.g., ff3 or carhart4) is feasible in this framework but is
#   intentionally not implemented here because the thesis universes are multi-asset class,
#   sector, and international etf sets. Adding additional factors would be ad hoc and outside the 
#   the scope of this thesis
# - dm uses a capm factor with a global equity market proxy (MSCI World, acwi.o) to avoid a 
#   u.s.-centric market factor that would mechanically align with the u.s. equity sleeve (e.g., VTI) in the benchmark universe
#
# posterior covariance updating:
# - as data accumulates, the covariance is updated using the sample residual
#   covariance and a penalty for pricing-model misfit:
#     s_post = h + t * sigma_hat + a_hat' * q * a_hat  (pástor and stambaugh, 1999 (eq. A.19))  
#
# external data dependencies (see data_pipeline.py):
# - data/processed/mkt_us_daily.csv (not used in report)
# - data/processed/mkt_global_daily.csv
# - each file must contain exactly one column of daily returns aligned to the
#   same trading-day calendar as the asset returns


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from scipy.optimize import minimize

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# simple result container (matches eval_engine expectations)
# ------------------------------------------------------------

@dataclass(frozen=True)
class WeightResult:
    weights: pd.Series


# ------------------------------------------------------------
# configuration
# ------------------------------------------------------------

@dataclass(frozen=True)
class DMConfig:
    # sigma_alpha_ann: prior std dev of annualized expected mispricing (alpha)
    # 1% calibration aligns with demiguel et al. (2009)
    sigma_alpha_ann: float = 0.01

    # annualization factor used to map sigma_alpha_ann (annualized mean mispricing std dev)
    # to per-period units. since this is an expected-return quantity (not volatility),
    # the scaling is linear in time:
    # sigma_alpha_per = sigma_alpha_ann / periods_per_year
    periods_per_year: int = 252

    # inverse-wishart prior df parameter v (pástor and stambaugh, 1999, eq. 18)
    # this is a baseline value; the effective df depends on the universe size m:
    #   v_eff = max(v, m + 2)
    # this enforces v_eff > m + 1 so the posterior mean E[Sigma | R] exists
    iw_df_v: int = 15

    # market return series path (daily simple returns, single column)
    #market_path: str = "data/processed/mkt_us_daily.csv" # (not reported)
    market_path: str = "data/processed/mkt_global_daily.csv"


    # strategy name reported in outputs
    # usage:
    # - dm(dmconfig(name="dm_us", market_path="...mkt_us_daily.csv"))
    # - dm(dmconfig(name="dm_global", market_path="...mkt_global_daily.csv"))
    name: str = "dm"

    # long-only constraint flag 
    long_only: bool = True

    # risk aversion in the quadratic utility objective:
    #     max_w  w' mu - (gamma / 2) w' cov w
    # fixed across the thesis for comparability across windows and universes
    # gamma = 5 is a pre-specified moderate calibration for the long-only etf setting
    # pástor and stambaugh (1999) use A = 2.84 for a different normalization in a
    # margin-constrained spread-position problem: it is the value that allocates all
    # wealth to the market portfolio when that is the only risky position available,
    # that calibration is not carried over directly here
    gamma: float = 5.0 

    # minimal window length to avoid unstable posterior objects
    min_obs: int = 30

    # sanity checks for return scaling (optional, store flags only) 
    rf_abs_median_warn: float = 0.01
    mkt_abs_median_warn: float = 0.05


# ------------------------------------------------------------
# io helpers
# ------------------------------------------------------------

def _load_single_col_series(path: Path, name: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(
            f"missing required market return file: {path}. "
            "run data_pipeline.py to create the market proxy return series csv."
        )

    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if df.shape[1] != 1:
        raise ValueError(f"{path} must have exactly one column")

    s = df.iloc[:, 0].astype(float)
    s.name = name
    return s

# ------------------------------------------------------------
# numerical helpers
# ------------------------------------------------------------

def _mv_weights(mu: np.ndarray, cov: np.ndarray, long_only: bool, gamma: float = 5.0) -> np.ndarray:
    """
    quadratic-utility mean-variance optimizer using the pástor and stambaugh (1999) objective (eq. 7),
    solved under practical long-only, fully-invested constraints:

        maximize  w' mu - (gamma/2) w' cov w
        
        s.t.      sum(w) = 1
                  and (if long_only) w >= 0

    relation to pástor and stambaugh (1999):
     - objective matches eq. 7 exactly, with gamma interpreted as risk aversion
     - eq. 8 gives the unconstrained solution direction cov^{-1} mu, used here only as a warm start
     - constraints differ from ps (1999), who study margin/leverage constraints; here the constraint set is long-only and fully invested
    """

    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    n = int(mu.size)

    if n == 0:
        raise ValueError("empty universe: cannot optimize weights")

    if (not np.all(np.isfinite(mu))) or (not np.all(np.isfinite(cov))):
        raise ValueError("non-finite inputs in mu or cov")

    if (not np.isfinite(gamma)) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")

    # symmetrize for numerical stability
    cov = 0.5 * (cov + cov.T)

    # warm start motivated by the unconstrained mean-variance direction (pástor and stambaugh, 1999, eq. 8):
    # eq. 8 motivates using cov^{-1} mu as an initial direction before imposing constraints
    # used here as a numerical initialization for the constrained optimizer, not a modeling assumption 
    try:
        w_dir = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        w_dir = np.linalg.pinv(cov) @ mu

    # debug flag if NaN or infinite values appear (covariance singularity / near singularity)
    if not np.all(np.isfinite(w_dir)):
        raise ValueError("failed to compute cov^{-1} mu direction (non-finite)")

    if long_only:
        w0 = np.maximum(w_dir, 0.0)
        s0 = float(np.sum(w0))
        if (not np.isfinite(s0)) or s0 <= 0.0:
            # if the mean-variance direction is all negative, start from a feasible interior point
            w0 = np.ones(n, dtype=float) / float(n)
        else:
            w0 = w0 / s0
        bnds = [(0.0, None) for _ in range(n)]
    else:
        # unconstrained start: normalize to sum to 1
        s0 = float(np.sum(w_dir))
        if (not np.isfinite(s0)) or abs(s0) < 1e-12:
            w0 = np.ones(n, dtype=float) / float(n)
        else:
            w0 = w_dir / s0
        bnds = None

    # objective for minimization (pástor and stambaugh, 1999, eq. 7)
    def obj(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        return -(float(w @ mu) - 0.5 * float(gamma) * float(w @ cov @ w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if (not res.success) or (not np.all(np.isfinite(res.x))):
        # flag failure so the issue is visible during evaluation
        raise RuntimeError(f"mv optimizer failed: {res.message}")

    w = np.asarray(res.x, dtype=float)

    if long_only:
        # enforce feasibility and renormalize
        w = np.maximum(w, 0.0)

    s = float(np.sum(w))
    if (not np.isfinite(s)) or abs(s) < 1e-12:
        raise RuntimeError("mv optimizer returned invalid weights (sum not finite or ~0)")

    return w / s

def _median_abs(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    return float(np.median(np.abs(x.to_numpy(dtype=float))))


# ------------------------------------------------------------
# dm predictive moments (risk-based models, capm special case)
# ------------------------------------------------------------

def _check_inputs(asset_excess: pd.DataFrame, factor_excess: pd.DataFrame, min_obs: int) -> None:
    if asset_excess.shape[0] != factor_excess.shape[0]:
        raise ValueError("asset_excess and factor_excess must have identical row counts (aligned dates)")

    if asset_excess.isna().any().any() or factor_excess.isna().any().any():
        raise ValueError("inputs contain nan; align and drop missing rows before dm estimation")

    if factor_excess.shape[1] != 1:
        raise ValueError("capm dm requires factor_excess to have exactly one column (market excess return)")

    if int(asset_excess.shape[0]) < int(min_obs):
        raise ValueError("window too short for dm posterior moments; increase estimation window length")

def dm_predictive_moments_risk_based(
    asset_excess: pd.DataFrame,
    factor_excess: pd.DataFrame,
    cfg: DMConfig,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    """
    predictive moments for excess returns under the risk-based model.
    capm special case: k = 1 factor (market excess return).
    """
    _check_inputs(asset_excess, factor_excess, cfg.min_obs)

    y = np.asarray(asset_excess.values, dtype=float)          # t x m
    x = np.asarray(factor_excess.values, dtype=float)         # t x 1
    t, m = y.shape
    k = int(x.shape[1])

    # z = [1, x]
    z = np.hstack([np.ones((t, 1), dtype=float), x])          # t x (k+1)
    p = int(k + 1)

    # ols a_hat = (z'z)^{-1} z'y, where a_hat stacks intercept and betas 
    ztz = z.T @ z                                             # p x p
    zty = z.T @ y                                             # p x m
    try:
        a_hat = np.linalg.solve(ztz, zty)                     # p x m
    except np.linalg.LinAlgError:
        a_hat, _, _, _ = np.linalg.lstsq(z, y, rcond=None)    # p x m

    # residual covariance sigma_hat = u'u / t; 
    # standard sample estimator for the regression residual covariance (MLE-style scaling)
    u_hat = y - z @ a_hat
    sigma_hat = (u_hat.T @ u_hat) / float(t)                  # m x m
    sigma_hat = 0.5 * (sigma_hat + sigma_hat.T)

    # benchmark factor parameters are diffuse/noninformitive and unrestricted
    e2_hat = x.mean(axis=0).reshape(k, 1)                     # k x 1; factor mean; sample mean under diffuse prior (pástor and stambaugh, 1999, pgs. 11-13)
    x_demean = x - x.mean(axis=0, keepdims=True)
    v22_hat = (x_demean.T @ x_demean) / float(t)              # k x k; sample factor covariance (PS 1999, App. A)

    # alpha prior tightness: annualized mean mispricing -> per-period mean mispricing
    # expected returns scale linearly with time
    s2 = float(np.mean(np.diag(sigma_hat)))
    if not np.isfinite(s2) or s2 <= 0.0:
        raise ValueError("invalid s2 computed from sigma_hat diagonal")

    sigma_alpha_per = float(cfg.sigma_alpha_ann) / float(cfg.periods_per_year)
    if not np.isfinite(sigma_alpha_per) or sigma_alpha_per <= 0.0:
        raise ValueError("sigma_alpha_per must be positive and finite")

    # risk-based prior (pástor and stambaugh, 1999, eq. 19):
    # Var(alpha | Sigma) = (sigma_alpha_per^2 / s2) * Sigma
    # in matrix-normal form this implies V0^{-1}[0,0] = s2 / sigma_alpha_per^2
    d = np.zeros((p, p), dtype=float)
    d[0, 0] = s2 / (sigma_alpha_per ** 2)

    # implementation modeled after the multivariate regression framework in
    # pástor and stambaugh (1999), equations (15)–(17), with Bayesian updating
    # of (alpha, beta) under the risk-based prior
    f = d + ztz

    try:
        f_inv = np.linalg.inv(f)
    except np.linalg.LinAlgError:
        f_inv = np.linalg.pinv(f)

    # posterior mean of a: a_tilde = f^{-1} z'z a_hat (pástor and stambaugh, 1999, eq. A.17 and A.20)
    a_tilde = f_inv @ ztz @ a_hat                              # p x m
    c_tilde = a_tilde[0:1, :].T                                # m x 1 (alphas)
    b_tilde = a_tilde[1:, :].T                                 # m x k (betas)

    # implementation of inverse-Wishart prior on Sigma following pástor and stambaugh (1999, eq. 18)
    # enforce nu > m + 1 so that the posterior mean E[Sigma] is well-defined
    v_cfg = int(cfg.iw_df_v)
    v_eff = max(v_cfg, int(m) + 2)
    if v_eff <= int(m) + 1:
        raise ValueError("iw prior df must satisfy v_eff > m + 1")

    df_post = float(t + v_eff - k)
    denom_mean = float(df_post - float(m) - 1.0) # used to compute posterior covariance (see pástor and stambaugh, 1999, Appendix A, eq. A.21)
    if denom_mean <= 0.0:
        raise ValueError("posterior df too small for finite mean of sigma; increase window length or v")

    # empirical-bayes approach: h = s2 * (v_eff - m - 1) * i_m (pástor and stambaugh, 1999, pg. 11)
    h = s2 * float(v_eff - m - 1) * np.eye(m, dtype=float)

    # q = z'z - z'z f^{-1} z'z 
    # arises from integrating out regression coefficients a in the Gaussian model
    # it appears implicitly in the inverse-Wishart posterior scale matrix for Sigma
    # (pástor and stambaugh, 1999, Appendix A, eq. A.19)
    q = ztz - (ztz @ f_inv @ ztz)
    q = 0.5 * (q + q.T)

    # posterior mean of Sigma (residual covariance) in the risk-based model:
    # pástor and stambaugh (1999) Appendix A show that the inverse-wishart posterior scale is
    #   S_post = H + T * Sigma_hat + A_hat' Q A_hat  (eq. A.19),
    # and the posterior mean is
    #   Sigma_tilde = E[Sigma | R] = S_post / (T + nu - m - k - 1)  (eq. A.21)
    # numerical note: symmetrize to remove floating-point asymmetry
    s_post = h + float(t) * sigma_hat + (a_hat.T @ q @ a_hat)
    sigma_tilde = s_post / denom_mean 
    sigma_tilde = 0.5 * (sigma_tilde + sigma_tilde.T)

    # benchmark (factor) moments: posterior expectation of V22 and variance of E2
    # From Appendix A, V22^{-1} | R is Wishart with df (T - 1) and scale (T * V22_hat)^{-1} (eq. A.25),
    # implying E[V22 | R] = (T / (T - k - 2)) * V22_hat (eq. A.27) and Var(E2 | R) = V22_tilde / (T - k - 2) (eq. A.28)
    # require T > k + 2 so these moments are finite
    denom = float(t - k - 2)
    if denom <= 0.0:
        raise ValueError("window too short for benchmark posterior moments; need t > k + 2")
    v22_tilde = (float(t) / denom) * v22_hat 
    var_e2 = v22_tilde / denom

    # predictive mean of non-benchmark returns: E* = E(E | R) = a_tilde + B_tilde * E2_tilde
    # pástor and stambaugh (1999), appendix A, eq. A.29, using a_tilde and b_tilde from eq. A.20
    # and E2_tilde from eq. A.26
    mu_star = (c_tilde + b_tilde @ e2_hat).reshape(-1)

    # predictive covariance V* = E[V | R] + Var(E | R),
    # implemented in closed form following Pástor and Stambaugh (1999),
    # defined in eq. (14) and Appendix A, eqs. A.30–A.33
    f_inv_bb = f_inv[1:, 1:]                                   # k x k

    e_bv22bt = b_tilde @ v22_tilde @ b_tilde.T + sigma_tilde * float(np.trace(f_inv_bb @ v22_tilde))
    e_v11 = sigma_tilde + e_bv22bt

    var_term_e2 = b_tilde @ var_e2 @ b_tilde.T

    f00 = float(f_inv[0, 0])
    f0b = f_inv[0, 1:].reshape(1, -1)
    e2 = e2_hat.reshape(-1, 1)

    scalar_e = (
        f00
        + 2.0 * float(f0b @ e2)
        + float(np.trace(f_inv_bb @ var_e2))
        + float(e2.T @ f_inv_bb @ e2)
    )
    var_term_coef = sigma_tilde * scalar_e

    cov_star = e_v11 + var_term_coef + var_term_e2
    cov_star = 0.5 * (cov_star + cov_star.T)

    mu_star_s = pd.Series(mu_star, index=asset_excess.columns, name="mu_star")
    cov_star_df = pd.DataFrame(cov_star, index=asset_excess.columns, columns=asset_excess.columns)

    # shrinkage diagnostics for auditing and write-up transparency
    sigma_from_h = h / denom_mean
    sigma_from_data = (float(t) * sigma_hat) / denom_mean
    sigma_from_aqa = (a_hat.T @ q @ a_hat) / denom_mean

    info: Dict[str, Any] = {
        "t": int(t),
        "m": int(m),
        "k": int(k),
        "sigma_alpha_ann": float(cfg.sigma_alpha_ann),
        "sigma_alpha_per_period_mean": float(sigma_alpha_per),
        "iw_df_v_cfg": int(v_cfg),
        "iw_df_v_eff": int(v_eff),
        "df_post": float(df_post),
        "shrink_target_s2": float(s2),
        "sigma_mean_decomp": {
            "from_h_identity_prior": pd.DataFrame(
                sigma_from_h, index=asset_excess.columns, columns=asset_excess.columns
            ),
            "from_t_sigma_hat": pd.DataFrame(
                sigma_from_data, index=asset_excess.columns, columns=asset_excess.columns
            ),
            "from_aq_at": pd.DataFrame(
                sigma_from_aqa, index=asset_excess.columns, columns=asset_excess.columns
            ),
        },
    }

    return mu_star_s, cov_star_df, info


# ------------------------------------------------------------
# wrapper class (eval_engine compatible)
# ------------------------------------------------------------

class DM:
    
    """
    dm strategy wrapper integrating with eval_engine
    
    key behavior:
     - loads a market return series from disk (u.s. or global proxy)
     - computes market excess returns using the project convention:
         mkt_excess = mkt_simple - rf_simple
     - computes asset excess returns similarly:
         r_excess = r_simple - rf_simple
     - forms predictive moments from the risk-based dm model (pástor and stambaugh, 1999)
     - converts predictive moments to a long-only mean-variance portfolio
       by solving the constrained quadratic utility problem

    """
    def __init__(self, cfg: Optional[DMConfig] = None):
        self.cfg = cfg if cfg is not None else DMConfig()
        self._mkt_daily = _load_single_col_series(Path(self.cfg.market_path), name="mkt_daily")

        # last diagnostics are stored for optional inspection and debugging
        self.last_info: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return str(self.cfg.name)

    def get_weights(
        self,
        returns_window: pd.DataFrame,
        window_rf: Optional[pd.Series] = None,
        **_: Any,
    ) -> WeightResult:
        # returns_window: daily simple returns for investable assets
        # window_rf: daily simple rf returns aligned by eval_engine
        if window_rf is None:
            raise ValueError("dm requires window_rf to compute excess returns")

        # align all series to the window index and enforce complete-case rows
        df = returns_window.astype(float).copy()
        rf = window_rf.reindex(df.index).astype(float)
        mkt = self._mkt_daily.reindex(df.index).astype(float)

        joined = df.join(rf.rename("rf"), how="inner").join(mkt.rename("mkt"), how="inner")
        joined = joined.dropna(axis=0, how="any")

        # debug/validation: store minimal evidence that rf/mkt/assets are correctly used and aligned
#        debug: Dict[str, Any] = {}
#        if not joined.empty:
#            first_asset = str(df.columns[0]) if df.shape[1] > 0 else None
#            debug["debug_t_joined"] = int(joined.shape[0])
#            debug["debug_rf_head"] = joined["rf"].head(3).to_list() if "rf" in joined.columns else []
#            debug["debug_mkt_head"] = joined["mkt"].head(3).to_list() if "mkt" in joined.columns else []
#            if first_asset is not None and first_asset in joined.columns:
#                debug["debug_asset_head"] = joined[first_asset].head(3).to_list()
#            else:
#                debug["debug_asset_head"] = []

            # single extra diagnostic to validate alignment
            # corr(mkt, first_asset) on the exact joined slice dm uses
#            if first_asset is not None and first_asset in joined.columns:
#                try:
#                    corr = float(joined["mkt"].corr(joined[first_asset]))
#                except Exception:
#                    corr = np.nan
#                debug["debug_corr_mkt_first_asset"] = corr
#            else:
#                debug["debug_corr_mkt_first_asset"] = np.nan

        # diagnostics that must exist in both fallback and non-fallback paths
        t_raw = int(df.shape[0])
        t_joined = int(joined.shape[0])
        #dropped_rows = int(t_raw - t_joined)

        rf_abs_median = _median_abs(joined["rf"]) if "rf" in joined.columns else np.nan
        mkt_abs_median = _median_abs(joined["mkt"]) if "mkt" in joined.columns else np.nan
        #rf_suspicious = bool(np.isfinite(rf_abs_median) and rf_abs_median > float(self.cfg.rf_abs_median_warn))
        #mkt_suspicious = bool(np.isfinite(mkt_abs_median) and mkt_abs_median > float(self.cfg.mkt_abs_median_warn))

        if t_joined < int(self.cfg.min_obs):
            raise ValueError(
                f"dm requires at least {self.cfg.min_obs} observations after alignment; got {t_joined}"
            )

        # compute excess returns using the project simple-return convention
        asset_excess = joined[df.columns].sub(joined["rf"], axis=0)

        # capm factor is market excess return
        factor_excess = pd.DataFrame(
            {"mkt_excess": joined["mkt"] - joined["rf"]},
            index=joined.index,
        )

        mu_star, cov_star, info = dm_predictive_moments_risk_based(
            asset_excess=asset_excess,
            factor_excess=factor_excess,
            cfg=self.cfg,
        )

        # extra debug moments (commented out for clean output)
#        try:
#            mkt_ex = (joined["mkt"] - joined["rf"]).to_numpy(dtype=float)
#            info["debug_mkt_excess_mean"] = float(np.mean(mkt_ex))
#            info["debug_mkt_excess_std"] = float(np.std(mkt_ex, ddof=1)) if mkt_ex.size > 1 else np.nan
#        except Exception:
#            info["debug_mkt_excess_mean"] = np.nan
#            info["debug_mkt_excess_std"] = np.nan

#        try:
#            ae = asset_excess.to_numpy(dtype=float)
#            info["debug_asset_excess_mean_mean"] = float(np.mean(np.mean(ae, axis=0))) if ae.size > 0 else np.nan
#        except Exception:
#            info["debug_asset_excess_mean_mean"] = np.nan

#        try:
#            mu_np = mu_star.to_numpy(dtype=float)
#            info["debug_mu_star_mean"] = float(np.mean(mu_np))
#            info["debug_mu_star_min"] = float(np.min(mu_np))
#            info["debug_mu_star_max"] = float(np.max(mu_np))
#        except Exception:
#            info["debug_mu_star_mean"] = np.nan
#            info["debug_mu_star_min"] = np.nan
#            info["debug_mu_star_max"] = np.nan

#        try:
#            cov_np = cov_star.to_numpy(dtype=float)
#            diag = np.diag(cov_np)
#            info["debug_cov_diag_mean"] = float(np.mean(diag))
#            info["debug_cov_diag_min"] = float(np.min(diag))
#            info["debug_cov_diag_max"] = float(np.max(diag))
#        except Exception:
#            info["debug_cov_diag_mean"] = np.nan
#            info["debug_cov_diag_min"] = np.nan
#            info["debug_cov_diag_max"] = np.nan

        # append wrapper diagnostics for transparency and debugging
#        info["t_raw"] = t_raw
#        info["t_joined"] = t_joined
#        info["dropped_rows"] = dropped_rows
#        info["rf_abs_median"] = float(rf_abs_median) if np.isfinite(rf_abs_median) else np.nan
#        info["mkt_abs_median"] = float(mkt_abs_median) if np.isfinite(mkt_abs_median) else np.nan
#        info["rf_suspicious"] = rf_suspicious
#        info["mkt_suspicious"] = mkt_suspicious
#        info.update(debug)

        w = _mv_weights(
            mu_star.to_numpy(),
            cov_star.to_numpy(),
            long_only=bool(self.cfg.long_only),
            gamma=float(self.cfg.gamma),
        )

        info["opt_gamma"] = float(self.cfg.gamma)
        info["opt_long_only"] = bool(self.cfg.long_only)
        info["market_path"] = str(self.cfg.market_path)
        info["strategy_name"] = str(self.cfg.name)
        self.last_info = info

        return WeightResult(weights=pd.Series(w, index=df.columns, name="weight"))
