# inference/lw_bootstrap.py
#
# purpose:
# - hypothesis tests for sharpe ratio differences using ledoit and wolf (2008):
#   (1) boot-ts: studentized time-series bootstrap for delta sharpe (section 3.2.2)
#   (2) algorithm 3.1: calibration-based block length selection (section 3.2.2, algorithm 3.1)
#
# integration:
# - takes realized out-of-sample daily net excess return series produced by eval_engine.py
# - weights are not re-estimated; inference is conditional on the realized return series
#
# literature mapping (lw, 2008):
# - parameter vector (y' in section 3.1): v = (mu_i, mu_n, gamma_i, gamma_n)' where gamma = E[r^2]
#   (uncentered second moment); psi is the long-run covariance of y_t
# - delta sharpe: f(v) = mu_i/sigma_i - mu_n/sigma_n with sigma^2 = gamma - mu^2 (sections 3.1–3.2)
# - y_t = (r_i - mu_i, r_n - mu_n, r_i^2 - gamma_i, r_n^2 - gamma_n)' (section 3.1)
# - heteroskedastic and autocorrelation robust (hac) long-run covariance estimator psi_hat:
#   quadratic-spectral (qs) kernel + and/or prewhitened qs kernel with plug-in bandwidth AR(1) approximation (andrews, 1991; andrews and monahan, 1992),
#   as used by lw in their hac and hac_pw methods (section 4.1)
# - boot-ts studentization uses psi_star computed from block sums of y_t* (section 3.2.2)
# - algorithm 3.1 uses:
#   b_grid = {1,2,4,6,8,10} in lw simulations (section 4.1),
#   p-hat implemented as var(1) + residual bootstrap with stationary bootstrap avg block size 5
#   (politis and romano, 1994) (section 4.1)
#
# practical note:
# - lw do not prescribe exact numeric values of M (final bootstrap reps) or m_inner (calibration inner reps) for all contexts
#   their applied/simulation sections use large values; this module exposes them in config so the thesis run can match the paper

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import sys
import time

# ============================================================
# status update helpers
# ============================================================

def _status_line(msg: str) -> None:
    # writes a single updating console line
    sys.stdout.write("\r" + msg[:200] + " " * 10)  # pad to clear tail
    sys.stdout.flush()

def _maybe_report(last_t: float, min_interval_s: float, msg: str) -> float:
    # rate-limited status updates to avoid excessive console spam
    now = time.time()
    if (now - last_t) >= min_interval_s:
        _status_line(msg)
        return now
    return last_t

# ============================================================
# configuration
# ============================================================

@dataclass(frozen=True)
class BootTSConfig:
    # lw (2008) boot-ts configuration
    alpha: float                 # two-sided nominal level
    m: int                       # number of bootstrap replications (lw denote M)
    seed: int = 123              # base seed if caller does not supply rng
    eps: float = 1e-12           # numeric guardrail
    max_invalid_frac: float = 0.02  # flag if too many invalid bootstrap draws

    # progress reporting (console-only; default off so baseline behavior is unchanged)
    show_progress: bool = False
    progress_min_interval_s: float = 0.5


@dataclass(frozen=True)
class Algo31Config:
    # lw (2008) algorithm 3.1 configuration
    b_grid: Tuple[int, ...]          # step (2): candidate block lengths
    k_pseudo: int                    # step (3): number of pseudo sequences (lw denote K)
    m_inner: int                     # step (3): inner boot-ts reps used to build each ci_{k,b}
    alpha: float                     # nominal level for calibration coverage (must match final alpha)

    # step (1): p-hat choice
    # lw section 4.1: var(1) used for p-hat in boot-ts experiments
    var_order: int = 1

    # lw section 4.1: stationary bootstrap avg block size 5 for residual bootstrap inside p-hat simulation
    resid_sb_avg_block: int = 5

    seed: int = 456
    eps: float = 1e-12

    # progress reporting (console-only; default off so baseline behavior is unchanged)
    show_progress: bool = False
    progress_min_interval_s: float = 0.5


@dataclass(frozen=True)
class Algo31Selection:
    # lw (2008) algorithm 3.1 output
    b_star: int
    g_by_b: Dict[int, float]
    abs_dev_by_b: Dict[int, float]
    k_pseudo: int
    m_inner: int
    alpha: float
    var_order: int


@dataclass(frozen=True)
class LWSharpeTestResult:
    # result for one (strategy, benchmark, b) test
    strategy: str
    benchmark: str
    t: int

    block_len: int

    sharpe_strategy: float
    sharpe_benchmark: float
    delta_sharpe: float

    se_delta: float
    p_value: float
    ci_low: float
    ci_high: float

    m: int
    invalid_frac: float
    flag_invalid_draws: bool


# ============================================================
# core moments and delta sharpe mapping (lw, 2008)
# ============================================================

def _moments_v(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    # v = (mu_i, mu_n, gamma_i, gamma_n)'
    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    g_x = float(np.mean(x * x))
    g_y = float(np.mean(y * y))
    return mu_x, mu_y, g_x, g_y


def _sigma_from_moments(mu: float, g: float, eps: float) -> float:
    # sigma^2 = gamma - mu^2
    v = g - mu * mu
    if not np.isfinite(v) or v <= eps:
        return np.nan
    return float(np.sqrt(v))


def _sharpe_from_moments(mu: float, g: float, eps: float) -> float:
    # sr = mu / sigma with sigma^2 = gamma - mu^2
    s = _sigma_from_moments(mu, g, eps)
    if not np.isfinite(s) or s <= eps:
        return np.nan
    return float(mu / s)


def _delta_sharpe_from_v(v: Tuple[float, float, float, float], eps: float) -> float:
    mu_x, mu_y, g_x, g_y = v
    sx = _sharpe_from_moments(mu_x, g_x, eps)
    sy = _sharpe_from_moments(mu_y, g_y, eps)
    if not np.isfinite(sx) or not np.isfinite(sy):
        return np.nan
    return float(sx - sy)


def _y_t_matrix(x: np.ndarray, y: np.ndarray, v: Tuple[float, float, float, float]) -> np.ndarray:
    # y_t = (r_i - mu_i, r_n - mu_n, r_i^2 - gamma_i, r_n^2 - gamma_n)'
    mu_x, mu_y, g_x, g_y = v
    yt = np.column_stack([
        x - mu_x,
        y - mu_y,
        x * x - g_x,
        y * y - g_y,
    ]).astype(float)
    return yt


def _grad_delta_sharpe(v: Tuple[float, float, float, float], eps: float) -> np.ndarray:
    # gradient of delta sharpe w.r.t v = (mu_x, mu_y, g_x, g_y)
    # mapping uses sr(mu,g) = mu / sqrt(g - mu^2)
    mu_x, mu_y, g_x, g_y = v

    def grad_sr(mu: float, g: float) -> Tuple[float, float]:
        # d/dmu and d/dg of sr(mu,g)
        s2 = g - mu * mu
        if not np.isfinite(s2) or s2 <= eps:
            return (np.nan, np.nan)
        s = np.sqrt(s2)

        # sr = mu * s^{-1}
        # d sr / d mu = s^{-1} + mu * d(s^{-1})/dmu
        # s^{-1} = s2^{-1/2}
        # d(s^{-1})/dmu = (-1/2) s2^{-3/2} * d s2/dmu = (-1/2) s2^{-3/2} * (-2mu) = mu * s2^{-3/2}
        d_mu = (1.0 / s) + mu * (mu / (s2 ** 1.5))

        # d sr / d g = mu * d(s^{-1})/dg with d(s^{-1})/dg = (-1/2) s2^{-3/2} * 1
        d_g = mu * (-(0.5) / (s2 ** 1.5))
        return (float(d_mu), float(d_g))

    dmu_x, dg_x = grad_sr(mu_x, g_x)
    dmu_y, dg_y = grad_sr(mu_y, g_y)
    if not all(np.isfinite([dmu_x, dg_x, dmu_y, dg_y])):
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

    # delta = sr_x - sr_y
    return np.array([dmu_x, -dmu_y, dg_x, -dg_y], dtype=float)


# ============================================================
# circular block bootstrap indices (lw, 2008)
# ============================================================

def _circular_block_indices(t: int, b: int, rng: np.random.Generator) -> np.ndarray:
    # lw use circular blocks to avoid edge effects (section 3.2.2)
    if b <= 0 or b > t:
        raise ValueError("invalid block length b")

    n_blocks = int(np.ceil(t / b))
    starts = rng.integers(0, t, size=n_blocks)

    # vectorized: build s..s+b-1 mod t for each start s
    offsets = np.arange(b, dtype=int)
    idx = (starts[:, None] + offsets[None, :]) % t
    return idx.ravel()[:t]


# ============================================================
# stationary bootstrap for residuals (politis-romano, 1994; used by lw section 4.1)
# ============================================================

def _stationary_bootstrap_indices(t: int, avg_block: int, rng: np.random.Generator) -> np.ndarray:
    # politis-romano stationary bootstrap:
    # - block lengths are geometric with mean avg_block
    # - next index continues with prob 1 - p, restarts uniform with prob p, where p = 1/avg_block
    if avg_block <= 0:
        raise ValueError("avg_block must be positive")
    p = 1.0 / float(avg_block)

    idx = np.empty(t, dtype=int)
    idx[0] = int(rng.integers(0, t))
    for i in range(1, t):
        if rng.random() < p:
            idx[i] = int(rng.integers(0, t))
        else:
            idx[i] = (idx[i - 1] + 1) % t
    return idx


# ============================================================
# hac long-run covariance: qs kernel with automatic bandwidth + optional prewhitening
# (andrews, 1991; andrews and monahan, 1992; referenced by lw section 3.1 and 4.1)
# ============================================================

def _qs_kernel(x: np.ndarray) -> np.ndarray:
    # quadratic spectral kernel k(x) (andrews, 1991)
    # k(0)=1 and decays smoothly
    out = np.empty_like(x, dtype=float)
    z = x.copy().astype(float)

    # handle zero separately for numerical stability
    out[z == 0.0] = 1.0

    nz = z != 0.0
    zz = z[nz]
    # k(z) = 25/(12*pi^2*z^2) * [sin(6*pi*z/5)/(6*pi*z/5) - cos(6*pi*z/5)]
    a = 6.0 * np.pi * zz / 5.0
    out[nz] = (25.0 / (12.0 * (np.pi ** 2) * (zz ** 2))) * (np.sin(a) / a - np.cos(a))
    return out


def _ar1_rho_hat(u: np.ndarray, eps: float) -> float:
    """
    ar(1) coefficient estimate used for andrews (1991) plug-in bandwidth
    this is standard: rho = sum u_t u_{t-1} / sum u_{t-1}^2
    """
    x = u[1:]
    xm1 = u[:-1]
    den = float(np.dot(xm1, xm1))
    if not np.isfinite(den) or den <= eps:
        return 0.0
    rho = float(np.dot(x, xm1) / den)
    # bound to keep stable
    return float(np.clip(rho, -0.98, 0.98))


def _andrews_qs_bandwidth(u: np.ndarray, eps: float) -> int:
    """
    automatic bandwidth selection for qs kernel (andrews, 1991)
    
    implementation follows the standard qs plug-in rule based on an ar(1) approximation
    lw cite andrews (1991) for automatic bandwidth selection
    
    note
     - andrews derives the optimal bandwidth for qs as a constant times T^{1/5} times a plug-in term
     - for an ar(1), the plug-in term can be expressed in terms of rho
     - this function implements the closed form for qs under ar(1) plug-in
    """
    t = int(u.shape[0])
    if t < 10:
        return 1

    rho = _ar1_rho_hat(u, eps)
    # plug-in factor for qs under ar(1)
    # g = ( (4*rho*rho) / ((1-rho)**4) )^{1/5} is a standard ar(1) specialization
    denom = (1.0 - rho)
    if abs(denom) <= 1e-6:
        denom = 1e-6
    g = ((4.0 * (rho ** 2)) / (denom ** 4)) ** 0.2

    # constant for qs kernel (andrews, 1991)
    c = 1.3221
    bw = int(np.floor(c * g * (t ** 0.2)))
    bw = int(max(1, min(bw, t - 1)))
    return bw


def _hac_qs_psi(y: np.ndarray, prewhiten: bool, eps: float) -> np.ndarray:
    # hac estimate of psi for y_t (t x k), using qs kernel and automatic bandwidth (andrews, 1991)
    # optional prewhitening (andrews and monahan, 1992)
    t, k = y.shape
    if t < 20:
        raise ValueError("sample too short for hac estimation")

    # demeaned series (lw define y_hat_t using estimated v in section 3.1)
    y0 = y - np.mean(y, axis=0, keepdims=True)

    # prewhitening: fit var(1) to y0, compute residuals e_t = y_t - a - A y_{t-1},
    # compute hac on e_t, then recolor with (I - A)^{-1} (andrews and monahan, 1992)
    if prewhiten:
        y_lag = y0[:-1, :]          # (t-1) x k
        y_now = y0[1:, :]           # (t-1) x k

        # fit var(1) without intercept because y0 is already demeaned
        B = np.linalg.lstsq(y_lag, y_now, rcond=None)[0]    # k x k (coeffs from lag to now)
        A = B.T                                             # k x k

        # residuals (already mean close to zero under mean-zero var)
        e0 = y_now - y_lag @ A.T   # (t-1) x k
        e0 = e0 - np.mean(e0, axis=0, keepdims=True)

        # bandwidth: conservative choice as max of component-wise andrews bw across dimensions
        bws = [_andrews_qs_bandwidth(e0[:, j], eps) for j in range(k)]
        S = int(max(bws))
        S = int(max(1, min(S, e0.shape[0] - 1)))

        psi_e = _hac_qs_no_prewhiten(e0, S, eps)

        # recolor: psi_y = (I - A)^{-1} psi_e (I - A)^{-T}
        M = np.linalg.inv(np.eye(k) - A)
        psi = M @ psi_e @ M.T
    else:
        bws = [_andrews_qs_bandwidth(y0[:, j], eps) for j in range(k)]
        S = int(max(bws))
        S = int(max(1, min(S, y0.shape[0] - 1)))

        psi = _hac_qs_no_prewhiten(y0, S, eps)

    # lw small-sample dof adjustment for dim(v)=4
    if t > 4:
        psi *= float(t) / float(t - 4)
    return psi

def _hac_qs_no_prewhiten(y0: np.ndarray, S: int, eps: float) -> np.ndarray:
    # qs kernel hac psi with fixed bandwidth S (andrews, 1991), using all lags 1..T-1
    t, k = y0.shape

    # gamma(0)
    g0 = (y0.T @ y0) / float(t)

    # kernel-weighted sum of autocovariances
    psi = g0.copy()
    # sum all feasible lags, with qs weights k(j/S) 
    for j in range(1, t):
        w = float(_qs_kernel(np.array([j / float(S)], dtype=float))[0])
        gj = (y0[j:, :].T @ y0[:-j, :]) / float(t)
        psi += w * (gj + gj.T)
    
    return psi


def _se_from_psi(psi: np.ndarray, grad: np.ndarray, t: int, eps: float) -> float:
    # lw eq (5): s(delta_hat) = sqrt( grad' psi_hat grad / T )
    v = float(grad.T @ psi @ grad)
    if not np.isfinite(v) or v <= eps:
        return np.nan
    return float(np.sqrt(v / float(t)))


# ============================================================
# boot-ts studentization psi_star (lw, 2008 section 3.2.2)
# ============================================================

def _psi_star_from_blocks(y_star: np.ndarray, b: int) -> np.ndarray:
    """
    lw (section 3.2.2):
     - let l = floor(T/b)
     - define f_j = (1/sqrt(b)) sum_{t=1}^b y^*_{(j-1)b + t}
     - define psi^* = (1/l) sum_{j=1}^l f_j f_j'
    """
    t, k = y_star.shape
    l = int(t // b)
    if l <= 1:
        return np.full((k, k), np.nan, dtype=float)

    y_use = y_star[: l * b, :]
    blocks = y_use.reshape(l, b, k)
    f = blocks.sum(axis=1) / float(np.sqrt(b))            # l x k
    psi = (f.T @ f) / float(l)                            # k x k
    return psi


# ============================================================
# ledoit-wolf boot-ts test (lw, 2008 section 3.2.2)
# ============================================================

def lw_boot_ts_sharpe_test(
    x: pd.Series,
    y: pd.Series,
    b: int,
    cfg: BootTSConfig,
    rng: Optional[np.random.Generator] = None,
    prewhiten_hac: bool = True,
) -> LWSharpeTestResult:
    # align series (lw bootstrap is on the paired bivariate series)
    df = pd.concat([x, y], axis=1).dropna()
    x0 = df.iloc[:, 0].to_numpy(dtype=float)
    y0 = df.iloc[:, 1].to_numpy(dtype=float)
    t = int(x0.shape[0])

    if b < 1 or b > t:
        raise ValueError("invalid block length b relative to sample length")

    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    # observed v_hat, delta_hat, sharpe components
    v_hat = _moments_v(x0, y0)
    delta_hat = _delta_sharpe_from_v(v_hat, cfg.eps)
    sr_x = _sharpe_from_moments(v_hat[0], v_hat[2], cfg.eps)
    sr_y = _sharpe_from_moments(v_hat[1], v_hat[3], cfg.eps)

    # observed y_hat_t and hac psi_hat (lw section 3.1; hacpw in section 4.1)
    y_hat_t = _y_t_matrix(x0, y0, v_hat)
    psi_hat = _hac_qs_psi(y_hat_t, prewhiten=prewhiten_hac, eps=cfg.eps)

    # observed se_hat via lw eq (5)
    grad_hat = _grad_delta_sharpe(v_hat, cfg.eps)
    se_hat = _se_from_psi(psi_hat, grad_hat, t, cfg.eps)

    # bootstrap studentized statistics
    stats = np.empty(cfg.m, dtype=float)
    invalid = 0

    # progress reporting for boot-ts (console-only; default off)
    if cfg.show_progress:
        last = time.time()
        t0 = last

    for m in range(cfg.m):
        # progress: inner bootstrap replications
        if cfg.show_progress:
            last = _maybe_report(
                last,
                cfg.progress_min_interval_s,
                f"boot-ts | {x.name} vs {y.name} | b={b} | m={m+1}/{cfg.m} "
                f"({100.0*(m+1)/cfg.m:.1f}%) | invalid={invalid} | elapsed={time.time()-t0:.1f}s"
            )

        idx = _circular_block_indices(t, b, rng)

        xb = x0[idx]
        yb = y0[idx]

        # bootstrap v_star and delta_star
        v_star = _moments_v(xb, yb)
        delta_star = _delta_sharpe_from_v(v_star, cfg.eps)

        # bootstrap y_star,t using v_star (lw section 3.2.2)
        y_star_t = _y_t_matrix(xb, yb, v_star)

        # bootstrap psi_star from block sums (lw section 3.2.2)
        psi_star = _psi_star_from_blocks(y_star_t, b)

        # bootstrap se_star using the same delta-method gradient at v_star
        grad_star = _grad_delta_sharpe(v_star, cfg.eps)
        se_star = _se_from_psi(psi_star, grad_star, t, cfg.eps)

        if (not np.isfinite(delta_star)) or (not np.isfinite(se_star)) or se_star <= cfg.eps:
            invalid += 1
            stats[m] = np.nan
            continue

        # lw studentized statistic: |delta* - delta_hat| / s(delta*)
        stats[m] = abs(delta_star - delta_hat) / se_star

    # finalize progress line with a newline so subsequent prints do not overwrite it
    if cfg.show_progress:
        print()

    stats_v = stats[np.isfinite(stats)]
    invalid_frac = 1.0 - float(stats_v.size) / float(cfg.m)
    flag_invalid = bool(invalid_frac > cfg.max_invalid_frac)

    if stats_v.size == 0 or (not np.isfinite(se_hat)) or se_hat <= cfg.eps:
        return LWSharpeTestResult(
            strategy=str(x.name),
            benchmark=str(y.name),
            t=t,
            block_len=int(b),
            sharpe_strategy=float(sr_x),
            sharpe_benchmark=float(sr_y),
            delta_sharpe=float(delta_hat),
            se_delta=float(se_hat),
            p_value=float(np.nan),
            ci_low=float(np.nan),
            ci_high=float(np.nan),
            m=int(cfg.m),
            invalid_frac=float(invalid_frac),
            flag_invalid_draws=flag_invalid,
        )

    # lw symmetric studentized ci: delta_hat ± q_{1-alpha} * se_hat
    q = float(np.quantile(stats_v, 1.0 - cfg.alpha))
    ci_low = delta_hat - q * se_hat
    ci_high = delta_hat + q * se_hat

    # lw p-value shortcut (eq. 9): p = (#{t_m* >= |delta_hat|/se_hat} + 1) / (M_valid + 1)
    d = abs(delta_hat) / se_hat
    p_val = (float(np.sum(stats_v >= d)) + 1.0) / (float(stats_v.size) + 1.0)

    return LWSharpeTestResult(
        strategy=str(x.name),
        benchmark=str(y.name),
        t=t,
        block_len=int(b),
        sharpe_strategy=float(sr_x),
        sharpe_benchmark=float(sr_y),
        delta_sharpe=float(delta_hat),
        se_delta=float(se_hat),
        p_value=float(p_val),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        m=int(cfg.m),
        invalid_frac=float(invalid_frac),
        flag_invalid_draws=flag_invalid,
    )


# ============================================================
# p-hat for algorithm 3.1: var(1) + stationary bootstrap residuals (lw section 4.1)
# ============================================================

def _fit_var1_ols(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # z_t = c + A z_{t-1} + u_t, fit by ols
    t, k = z.shape
    if k != 2:
        raise ValueError("z must be bivariate (t,2)")
    if t < 10:
        raise ValueError("too few observations for var(1)")

    y = z[1:, :]
    x = np.column_stack([np.ones((t - 1, 1), dtype=float), z[:-1, :]])
    b = np.linalg.lstsq(x, y, rcond=None)[0]          # (1+2) x 2

    c = b[0, :]                                       # (2,)
    A = b[1:, :].T                                    # 2 x 2
    u = y - x @ b                                     # (t-1) x 2
    return c, A, u


def _simulate_var1_from_phat(
    c: np.ndarray,
    A: np.ndarray,
    u: np.ndarray,
    z0: np.ndarray,
    t: int,
    avg_block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # simulate z_star via var(1) recursion with stationary-bootstrap innovations
    # lw section 4.1: residuals bootstrapped using stationary bootstrap with average block size 5
    if z0.shape != (2,):
        raise ValueError("z0 must be shape (2,)")

    z_star = np.empty((t, 2), dtype=float)
    z_star[0, :] = z0

    # stationary bootstrap indices on residuals u_t
    u_idx = _stationary_bootstrap_indices(t - 1, avg_block=avg_block, rng=rng)

    for tt in range(1, t):
        z_star[tt, :] = c + A @ z_star[tt - 1, :] + u[u_idx[tt - 1], :]

    return z_star


# ============================================================
# algorithm 3.1 block length selection (lw, 2008)
# ============================================================

def lw_algo31_select_block_length(
    x: pd.Series,
    y: pd.Series,
    cfg: Algo31Config,
    prewhiten_hac: bool = True,
) -> Algo31Selection:
    # align observed bivariate series z_t = (x_t, y_t)
    df = pd.concat([x, y], axis=1).dropna()
    x0 = df.iloc[:, 0].to_numpy(dtype=float)
    y0 = df.iloc[:, 1].to_numpy(dtype=float)
    t = int(x0.shape[0])

    z = np.column_stack([x0, y0])

    # lw algorithm 3.1 step (1): fit p-hat
    # lw section 4.1: var(1) is used for boot-ts experiments
    if cfg.var_order != 1:
        raise ValueError("lw section 4.1 uses var(1) for p-hat; set var_order=1 for strict replication")

    c_hat, A_hat, u_hat = _fit_var1_ols(z)

    # lw algorithm 3.1: substitute unknown true parameter with pseudo parameter from observed sample
    v_hat = _moments_v(x0, y0)
    delta_pseudo = _delta_sharpe_from_v(v_hat, cfg.eps)

    # inner boot-ts config for constructing ci_{k,b} in step (3)
    inner_cfg = BootTSConfig(
        alpha=cfg.alpha,
        m=cfg.m_inner,
        seed=cfg.seed + 999,
        eps=cfg.eps,
        max_invalid_frac=1.0,

        # progress reporting is disabled for the inner calibration bootstraps to keep output usable
        show_progress=False,
        progress_min_interval_s=cfg.progress_min_interval_s,
    )

    rng = np.random.default_rng(cfg.seed)

    contain_counts: Dict[int, int] = {int(b): 0 for b in cfg.b_grid}

    last = time.time()
    t0 = last

    # lw algorithm 3.1 step (3): loop over pseudo sequences
    for k in range(cfg.k_pseudo):
        # progress: outer loop (pseudo sequences)
        if cfg.show_progress:
            last = _maybe_report(
                last,
                cfg.progress_min_interval_s,
                f"algo3.1 | {x.name} vs {y.name} | k={k+1}/{cfg.k_pseudo} "
                f"({100.0*(k+1)/cfg.k_pseudo:.1f}%) | elapsed={time.time()-t0:.1f}s"
            )

        z_star = _simulate_var1_from_phat(
            c=c_hat,
            A=A_hat,
            u=u_hat,
            z0=z[0, :],
            t=t,
            avg_block=int(cfg.resid_sb_avg_block),
            rng=rng,
        )

        xs = pd.Series(z_star[:, 0], name="x_star")
        ys = pd.Series(z_star[:, 1], name="y_star")

        # for each b in grid, compute ci_{k,b} and check coverage of delta_pseudo
        for b in cfg.b_grid:
            # unique rng stream per (k,b) for reproducibility without stream reuse
            rng_ci = np.random.default_rng(cfg.seed + 10_000_000 * k + 10_000 * int(b) + 777)
            res = lw_boot_ts_sharpe_test(
                xs,
                ys,
                int(b),
                inner_cfg,
                rng=rng_ci,
                prewhiten_hac=prewhiten_hac,
            )

            if np.isfinite(res.ci_low) and np.isfinite(res.ci_high):
                if (delta_pseudo >= res.ci_low) and (delta_pseudo <= res.ci_high):
                    contain_counts[int(b)] += 1

    # finalize progress line with a newline so subsequent prints do not overwrite it
    if cfg.show_progress:
        print()

    # lw algorithm 3.1 step (4) and (5)
    target = 1.0 - float(cfg.alpha)
    g_by_b: Dict[int, float] = {}
    abs_dev_by_b: Dict[int, float] = {}

    for b in cfg.b_grid:
        gb = float(contain_counts[int(b)]) / float(cfg.k_pseudo)
        g_by_b[int(b)] = gb
        abs_dev_by_b[int(b)] = abs(gb - target)

    b_star = int(min(cfg.b_grid, key=lambda bb: abs_dev_by_b[int(bb)]))

    return Algo31Selection(
        b_star=b_star,
        g_by_b=g_by_b,
        abs_dev_by_b=abs_dev_by_b,
        k_pseudo=int(cfg.k_pseudo),
        m_inner=int(cfg.m_inner),
        alpha=float(cfg.alpha),
        var_order=int(cfg.var_order),
    )


# ============================================================
# suite runner for one universe
# ============================================================

def run_lw_suite_for_universe(
    excess_rets_daily: pd.DataFrame,
    benchmark_col: str,
    strategies: Optional[Iterable[str]],
    algo31_cfg: Algo31Config,
    final_cfg: BootTSConfig,
    report_all_b: bool = True,
    prewhiten_hac: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Algo31Selection]]:
    # prepare strategy list
    if benchmark_col not in excess_rets_daily.columns:
        raise ValueError("benchmark_col not found in excess_rets_daily")

    if strategies is None:
        strat_list = [c for c in excess_rets_daily.columns if c != benchmark_col]
    else:
        strat_list = [c for c in strategies if c != benchmark_col]

    bench = excess_rets_daily[benchmark_col].rename(benchmark_col)

    rows: List[Dict[str, object]] = []
    selections: Dict[str, Algo31Selection] = {}

    # loop strategies
    for i_strat, strat in enumerate(strat_list):
        x = excess_rets_daily[strat].rename(strat)
        y = bench

        # algorithm 3.1 selects b_star for this (x,y) pair
        sel = lw_algo31_select_block_length(
            x,
            y,
            algo31_cfg,
            prewhiten_hac=prewhiten_hac,
        )
        selections[strat] = sel

        # decide which b values to run for final inference
        if report_all_b:
            b_values = list(algo31_cfg.b_grid)
        else:
            b_values = [sel.b_star]
        if sel.b_star not in b_values:
            b_values.append(sel.b_star)

        # final boot-ts at each b
        for i_b, b in enumerate(b_values):
            # unique rng per (strategy, b) to prevent shared bootstrap streams across tests
            rng_final = np.random.default_rng(final_cfg.seed + 1_000_000 * i_strat + 10_000 * i_b + int(b))

            res = lw_boot_ts_sharpe_test(
                x,
                y,
                int(b),
                final_cfg,
                rng=rng_final,
                prewhiten_hac=prewhiten_hac,
            )

            rows.append({
                "strategy": res.strategy,
                "benchmark": res.benchmark,
                "t": res.t,
                "block_len": res.block_len,
                "block_len_selected": int(sel.b_star),
                "g_of_b": float(sel.g_by_b.get(int(b), np.nan)),
                "abs_dev_of_b": float(sel.abs_dev_by_b.get(int(b), np.nan)),
                "sharpe_strategy": res.sharpe_strategy,
                "sharpe_benchmark": res.sharpe_benchmark,
                "delta_sharpe": res.delta_sharpe,
                "se_delta": res.se_delta,
                "p_value": res.p_value,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "m": res.m,
                "invalid_frac": res.invalid_frac,
                "flag_invalid_draws": res.flag_invalid_draws,
                "algo31_k_pseudo": sel.k_pseudo,
                "algo31_m_inner": sel.m_inner,
                "algo31_alpha": sel.alpha,
                "algo31_var_order": sel.var_order,
                "algo31_resid_sb_avg_block": int(algo31_cfg.resid_sb_avg_block),
                "hac_prewhiten_qs": bool(prewhiten_hac),
            })

    out = pd.DataFrame(rows).sort_values(["strategy", "block_len"]).reset_index(drop=True)
    return out, selections