# strategies/erc.py
#
# equal risk contribution (erc) / risk parity portfolio under long-only, fully-invested constraints
#
# literature anchor:
# - maillard, roncalli, teiletche (2008), "the properties of equally weighted risk contribution portfolios"
#
# core object:
# - let x be portfolio weights (long-only, fully invested)
# - with covariance matrix Sigma, define the (volatility-based) total risk contribution component:
#       m_i(x) = x_i * (Sigma x)_i
#   (up to a common scaling by portfolio volatility, which does not affect equality across i)
#
# - the erc condition is equality of contributions:
#       m_i(x) = m_j(x) for all i, j
#
# numerical approach (maillard et al., 2008, section 3.3):
# - solve the constrained program (their eq. 6):
#       minimize    f(x)
#       subject to  1' x = 1
#                   0 <= x_i <= 1
#   where f(x) measures dispersion of m_i(x) across assets
#
# - eq. (6) proportional to:
#       sum_i sum_j (m_i(x) - m_j(x))^2
#   which is zero iff all m_i(x) are equal
#
# implementation:
# - Sigma estimated using ledoit-wolf shrinkage (sklearn LedoitWolf) for consistency
# - numerical stability: ridge ladder Sigma + lambda I
# - solver: scipy SLSQP, which is an sqp-type constrained nonlinear optimizer
#
# integration:
# - weights returned as pd.Series indexed by window_returns.columns
# - optional runtime diagnostics are exposed via self.last_info with keys prefixed "debug_"

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from strategies.base import BaseStrategy, StrategyResult


@dataclass(frozen=True)
class ERCConfig:
    # numerical conditioning: ridge ladder
    ridge_eps: float = 1e-10
    ridge_steps: int = 6  # total ladder points: ridge_eps * 10^k for k=0..ridge_steps-1

    # slsqp controls
    max_iter: int = 5_000
    ftol: float = 1e-12

    # initial guess restarts per ridge level
    n_restarts: int = 3

    # lower bound used in the log-barrier fallback only (strict positivity)
    # note: this is not a fallback to ew; it is a positivity floor for a different objective.
    w_min_pos: float = 1e-10

    # validation tolerances
    sum_tol: float = 1e-10
    neg_tol: float = 1e-14

    # if the solver returns a vector extremely close to 1/n, keep it only if it truly satisfies erc
    ew_linf_tol: float = 1e-10

    # erc validation: require near-equality of m_i(x) = x_i (Sigma x)_i
    # this guards against "solver success" returning a point on the simplex that is not actually erc.
    # tolerance is relative dispersion: std(m) / max(|mean(m)|, eps)
    rc_rel_tol: float = 1e-6
    rc_rel_eps: float = 1e-16


class EqualRiskContribution(BaseStrategy):
    name = "erc"

    def __init__(self, cfg: Optional[ERCConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else ERCConfig()
        self.last_info: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------------------
    # public api (called by eval_engine)
    # ----------------------------------------------------------------
    def get_weights(
        self,
        window_returns: pd.DataFrame,
        window_rf: Optional[pd.Series] = None,
    ) -> StrategyResult:
        # window_rf unused; accepted for interface consistency
        x = window_returns.dropna(how="any")
        if x.shape[0] < 5:
            raise ValueError("erc: insufficient observations in window after dropping nans")

        cols = x.columns
        n = int(len(cols))

        # estimate Sigma via ledoit-wolf shrinkage for stability/comparability
        lw = LedoitWolf().fit(x.values)
        sigma = lw.covariance_.astype(float)
        sigma = 0.5 * (sigma + sigma.T)

        # prepare initial guesses (initializers only, not fallbacks)
        x0_list = self._initial_guesses(sigma=sigma, n=n)

        # ridge ladder for conditioning
        ridge_list = [float(self.cfg.ridge_eps) * (10.0 ** k) for k in range(int(self.cfg.ridge_steps))]

        # try maillard eq. (6) dispersion objective first
        last_err = None
        for ridge in ridge_list:
            s = sigma + ridge * np.eye(n)
            for x0 in x0_list[: max(1, int(self.cfg.n_restarts))]:
                try:
                    w = self._solve_eq6_slsqp(sigma=s, x0=x0, cols=cols)
                    self.last_info = self._build_debug_info(w=w, sigma=s, ridge=ridge, tag="eq6")
                    return StrategyResult(weights=w)
                except Exception as e:
                    last_err = str(e)

        # standard log-barrier risk parity formulation
        # rationale:
        # - a common way to compute risk parity weights is to solve an unconstrained (or box-constrained)
        #   program that enforces strict positivity via a log term
        # - this is consistent with the erc first-order condition target (equalized contributions),
        #   but is not the same objective as eq. (6). this is used only as a numerical fallback
        for ridge in ridge_list:
            s = sigma + ridge * np.eye(n)
            for x0 in x0_list[: max(1, int(self.cfg.n_restarts))]:
                try:
                    w = self._solve_logbarrier_rp(sigma=s, x0=x0, cols=cols)
                    self.last_info = self._build_debug_info(w=w, sigma=s, ridge=ridge, tag="logbarrier")
                    return StrategyResult(weights=w)
                except Exception as e:
                    last_err = str(e)

        raise ValueError(f"erc: failed to compute valid weights. last_err={last_err}")

    # ----------------------------------------------------------------
    # maillard eq. (6): dispersion objective over m_i(x) = x_i (Sigma x)_i
    # ----------------------------------------------------------------
    @staticmethod
    def _m_contrib(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        sx = sigma @ x
        return x * sx

    @classmethod
    def _f_eq6(cls, x: np.ndarray, sigma: np.ndarray) -> float:
        """
        f(x) = sum_i sum_j (m_i - m_j)^2
        efficient identity:
          sum_i sum_j (m_i - m_j)^2 = 2n * sum_i m_i^2 - 2 (sum_i m_i)^2
        """
        m = cls._m_contrib(x, sigma)
        n = int(m.shape[0])
        s1 = float(np.sum(m))
        s2 = float(np.sum(m * m))
        val = 2.0 * n * s2 - 2.0 * (s1 * s1)
        if not np.isfinite(val):
            return float("inf")
        if val < 0.0:
            val = 0.0
        return float(val)

    def _solve_eq6_slsqp(self, sigma: np.ndarray, x0: np.ndarray, cols: pd.Index) -> pd.Series:
        n = int(sigma.shape[0])
        if n != len(cols):
            raise ValueError("erc(eq6): sigma dimension mismatch")

        x0 = self._sanitize_init_guess_simplex(x0, n=n)

        cons = [{"type": "eq", "fun": lambda z: float(np.sum(z) - 1.0)}]
        bnds = [(0.0, 1.0) for _ in range(n)]

        scale = 1e12  # numerical conditioning only; argmin unchanged

        res = minimize(
            fun=lambda z: scale * self._f_eq6(z, sigma),
            x0=x0,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": int(self.cfg.max_iter), "ftol": float(self.cfg.ftol), "disp": False},
        )
        if (not bool(res.success)) or res.x is None:
            raise ValueError(f"erc(eq6): slsqp failed: success={res.success}, status={res.status}, msg={res.message}")

        w = np.asarray(res.x, dtype=float).reshape(-1)
        self._validate_solution_simplex(w, n=n, label="erc(eq6)")

        # ensure the returned point actually satisfies erc to tolerance
        self._validate_erc_contributions(w, sigma=sigma, label="erc(eq6)")

        # guard against accidental "sanitize to equal weight" behavior:
        # keep ew only if it genuinely satisfies rc equality for this sigma.
        self._guard_against_accidental_ew(w, sigma=sigma)

        return pd.Series(w, index=cols, dtype=float)

    # ----------------------------------------------------------------
    # log-barrier risk parity (strictly positive), then normalize
    # ----------------------------------------------------------------
    def _solve_logbarrier_rp(self, sigma: np.ndarray, x0: np.ndarray, cols: pd.Index) -> pd.Series:
        """
        common risk parity objective variant:
          minimize  0.5 * x' Sigma x  -  sum_i log(x_i)
        subject to x_i >= w_min_pos (box constraint)

        this yields a strictly positive solution with an erc-type first order condition
        after solving, normalize to sum to one (fully invested)
        """
        n = int(sigma.shape[0])
        if n != len(cols):
            raise ValueError("erc(logbarrier): sigma dimension mismatch")

        w_min = float(self.cfg.w_min_pos)
        if not (w_min > 0.0):
            raise ValueError("erc(logbarrier): w_min_pos must be positive")

        x0 = self._sanitize_init_guess_positive(x0, n=n, w_min=w_min)

        def obj(z: np.ndarray) -> float:
            z = np.asarray(z, dtype=float).reshape(-1)
            q = 0.5 * float(z @ (sigma @ z))
            if np.any(z <= 0.0) or np.any(~np.isfinite(z)):
                return float("inf")
            barrier = -float(np.sum(np.log(z)))
            val = q + barrier
            return float(val) if np.isfinite(val) else float("inf")

        bnds = [(w_min, 1.0) for _ in range(n)]

        res = minimize(
            fun=obj,
            x0=x0,
            method="SLSQP",
            bounds=bnds,
            options={"maxiter": int(self.cfg.max_iter), "ftol": float(self.cfg.ftol), "disp": False},
        )
        if (not bool(res.success)) or res.x is None:
            raise ValueError(
                f"erc(logbarrier): slsqp failed: success={res.success}, status={res.status}, msg={res.message}"
            )

        y = np.asarray(res.x, dtype=float).reshape(-1)
        if np.any(~np.isfinite(y)) or np.any(y < w_min):
            raise ValueError("erc(logbarrier): invalid raw solution")

        s = float(np.sum(y))
        if (not np.isfinite(s)) or s <= 0.0:
            raise ValueError("erc(logbarrier): degenerate sum after solve")

        w = y / s
        self._validate_solution_simplex(w, n=n, label="erc(logbarrier)")

        # ensure the returned point actually satisfies erc to tolerance
        self._validate_erc_contributions(w, sigma=sigma, label="erc(logbarrier)")

        self._guard_against_accidental_ew(w, sigma=sigma)

        return pd.Series(w, index=cols, dtype=float)

    # ----------------------------------------------------------------
    # diagnostics
    # ----------------------------------------------------------------
    def _build_debug_info(self, w: pd.Series, sigma: np.ndarray, ridge: float, tag: str) -> Dict[str, Any]:
        ww = w.values.astype(float)
        n = int(ww.shape[0])
        m = self._m_contrib(ww, sigma)
        rc_disp = float(np.std(m, ddof=1)) if n > 1 else 0.0
        rc_mean = float(np.mean(m)) if n > 0 else np.nan

        ew = np.ones(n) / n
        l1_from_ew = float(np.sum(np.abs(ww - ew)))
        linf_from_ew = float(np.max(np.abs(ww - ew)))

        return {
            "debug_solver": str(tag),
            "debug_ridge": float(ridge),
            "debug_w_min": float(np.min(ww)),
            "debug_w_max": float(np.max(ww)),
            "debug_l1_from_ew": float(l1_from_ew),
            "debug_linf_from_ew": float(linf_from_ew),
            "debug_rc_disp": float(rc_disp),
            "debug_rc_mean": float(rc_mean),
            "debug_eq6_obj": float(self._f_eq6(ww, sigma)),
        }

    # ----------------------------------------------------------------
    # initialization
    # ----------------------------------------------------------------
    def _initial_guesses(self, sigma: np.ndarray, n: int) -> List[np.ndarray]:
        guesses: List[np.ndarray] = []

        # inverse volatility initializer: w_i ∝ 1 / sqrt(sigma_ii)
        diag = np.diag(sigma).astype(float)
        if np.all(np.isfinite(diag)) and np.all(diag > 0.0):
            iv = 1.0 / np.sqrt(diag)
            iv = iv / float(np.sum(iv))
            guesses.append(iv)

        # equal weight initializer (initializer only)
        guesses.append(np.ones(n) / n)

        # mild random jitter around ew (initializer only)
        rng = np.random.default_rng(123)
        for _ in range(2):
            z = np.ones(n) / n + 1e-3 * rng.standard_normal(n)
            z[z < 0.0] = 0.0
            s = float(np.sum(z))
            if np.isfinite(s) and s > 0.0:
                guesses.append(z / s)

        return guesses

    # ----------------------------------------------------------------
    # validation and guards
    # ----------------------------------------------------------------
    def _sanitize_init_guess_simplex(self, x0: np.ndarray, n: int) -> np.ndarray:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.shape[0] != n:
            x0 = np.ones(n) / n
            return x0

        x0[~np.isfinite(x0)] = 0.0
        x0[x0 < 0.0] = 0.0
        s = float(np.sum(x0))
        if (not np.isfinite(s)) or s <= 0.0:
            return np.ones(n) / n
        return x0 / s

    def _sanitize_init_guess_positive(self, x0: np.ndarray, n: int, w_min: float) -> np.ndarray:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.shape[0] != n:
            x0 = np.ones(n) / n

        x0[~np.isfinite(x0)] = w_min
        x0[x0 < w_min] = w_min
        s = float(np.sum(x0))
        if (not np.isfinite(s)) or s <= 0.0:
            return np.ones(n) / n
        return x0 / s

    def _validate_solution_simplex(self, w: np.ndarray, n: int, label: str) -> None:
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"{label}: wrong dimension")
        if np.any(~np.isfinite(w)):
            raise ValueError(f"{label}: non-finite weights")
        if np.any(w < -float(self.cfg.neg_tol)):
            raise ValueError(f"{label}: negative weights found")
        s = float(np.sum(w))
        if not np.isfinite(s) or abs(s - 1.0) > float(self.cfg.sum_tol):
            raise ValueError(f"{label}: weights do not sum to 1 (sum={s})")

    def _validate_erc_contributions(self, w: np.ndarray, sigma: np.ndarray, label: str) -> None:
        # ensure the solution actually equalizes m_i(x) = x_i (Sigma x)_i to tolerance
        w = np.asarray(w, dtype=float).reshape(-1)
        m = self._m_contrib(w, sigma)
        n = int(m.shape[0])

        if n <= 1:
            return

        rc_mean = float(np.mean(m))
        rc_std = float(np.std(m, ddof=1))
        scale = max(abs(rc_mean), float(self.cfg.rc_rel_eps))
        rel = rc_std / scale

        if (not np.isfinite(rel)) or rel > float(self.cfg.rc_rel_tol):
            raise ValueError(
                f"{label}: erc condition not met to tolerance "
                f"(rel_disp={rel:.3e} > rc_rel_tol={float(self.cfg.rc_rel_tol):.3e})"
            )

    def _guard_against_accidental_ew(self, w: np.ndarray, sigma: np.ndarray) -> None:
        # if w is extremely close to ew, verify it actually satisfies the erc condition
        # (otherwise this likely indicates a sanitize path collapsing to ew)
        w = np.asarray(w, dtype=float).reshape(-1)
        n = int(w.shape[0])
        ew = np.ones(n) / n
        linf = float(np.max(np.abs(w - ew)))

        if linf <= float(self.cfg.ew_linf_tol):
            m = self._m_contrib(w, sigma)
            # require very small dispersion relative to mean magnitude
            rc_mean = float(np.mean(m)) if n > 0 else np.nan
            rc_std = float(np.std(m, ddof=1)) if n > 1 else 0.0
            scale = max(abs(rc_mean), 1e-16)
            rel = rc_std / scale
            if not np.isfinite(rel) or rel > 1e-6:
                raise ValueError(
                    "erc: solution is ~equal-weight but does not satisfy erc contributions. "
                    "this suggests an unintended collapse to ew."
                )