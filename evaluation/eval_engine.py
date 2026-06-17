# evaluation/eval_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# configuration
# ------------------------------------------------------------

@dataclass(frozen=True)
class EvalConfig:
    # proportional transaction cost per unit traded (e.g. 0.005 for 50 bps)
    tc_cost: float = 0.005

    # numeric guardrails
    eps: float = 1e-12


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _safe_normalize(w: pd.Series, eps: float) -> pd.Series:
    # normalize weights to sum to 1.0 and error on invalid vectors
    s = float(w.sum())
    if not np.isfinite(s) or abs(s) <= eps:
        raise ValueError("invalid weight vector: sum is zero or non-finite")
    return w / s


def _compute_drifted_weights(w_t: np.ndarray, r_t: np.ndarray, eps: float) -> np.ndarray:
    """
    compute drifted pre-rebalance weights at the rebalance instant, denoted w_{t^+} in demiguel et al.

        w_{t^+} = (w_t * (1 + r_t)) / sum_j w_tj * (1 + r_{j,t})

    assumptions:
    - returns are simple returns (not log returns)
    - all gross returns (1 + r) must be positive
    """
    gross = 1.0 + r_t
    if np.any(~np.isfinite(gross)) or np.any(gross <= 0.0):
        raise ValueError("non-finite or non-positive gross return in drift computation")

    numer = w_t * gross
    denom = float(numer.sum())
    if not np.isfinite(denom) or denom <= eps:
        raise ValueError("invalid drift denominator in drift computation")

    return numer / denom


def _to_monthly_simple_returns(daily_simple_returns: pd.Series) -> pd.Series:
    """
    convert daily simple returns to monthly simple returns by compounding within each month:

        R_m = prod_{t in month} (1 + r_t) - 1

    index is month-end timestamps produced by pandas resample("ME").
    """
    r = daily_simple_returns.dropna()
    if r.empty:
        return r
    return (1.0 + r).resample("ME").prod() - 1.0


def compute_oos_moments(port_rets: pd.Series, ann_factor: int) -> Dict[str, float]:
    """
    compute out-of-sample moments and report both native-frequency and annualized metrics.

    outputs:
    - mu_hat: mean (native frequency)
    - sigma_hat: std dev (native frequency)
    - sr_hat: sharpe (native frequency)
    - mu_ann, sigma_ann, sr_ann: annualized versions using ann_factor
    """
    r = port_rets.dropna()
    if len(r) == 0:
        return {}

    # native frequency moments
    mu_hat = float(r.mean())
    sigma_hat = float(r.std(ddof=1))
    sr_hat = mu_hat / sigma_hat if sigma_hat > 0 else np.nan

    # annualized moments
    mu_ann = mu_hat * ann_factor
    sigma_ann = sigma_hat * np.sqrt(ann_factor)
    sr_ann = mu_ann / sigma_ann if sigma_ann > 0 else np.nan

    return {
        # native-frequency
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "sr_hat": sr_hat,
        "n_oos": int(len(r)),
        # annualized
        "mu_ann": mu_ann,
        "sigma_ann": sigma_ann,
        "sr_ann": sr_ann,
        "ann_factor": int(ann_factor),
    }


def compute_ceq(mu_hat: float, sigma_hat: float, gamma: float) -> float:
    """
    certainty equivalent return (equation 14).

        ceq_hat_k(gamma) = mu_hat_k - (gamma/2) * sigma_hat_k^2

    important:
    - this function is frequency-agnostic.
    - if you pass daily mu/sigma, you get daily ceq.
    - if you pass annualized mu/sigma, you get annualized ceq.
    """
    if not np.isfinite(mu_hat) or not np.isfinite(sigma_hat):
        return np.nan
    return mu_hat - 0.5 * gamma * (sigma_hat ** 2)


def compute_return_loss(mu_ew: float, sigma_ew: float, mu_k: float, sigma_k: float) -> float:
    """
    return-loss (equation 17).

        return-loss_k = (mu_ew / sigma_ew) * sigma_k - mu_k

    important:
    - this is frequency-agnostic. use consistent inputs (daily or annualized).
    """
    if not np.isfinite(mu_ew) or not np.isfinite(sigma_ew) or sigma_ew <= 0:
        return np.nan
    if not np.isfinite(mu_k) or not np.isfinite(sigma_k):
        return np.nan
    return (mu_ew / sigma_ew) * sigma_k - mu_k


# ------------------------------------------------------------
# backtest engine
# ------------------------------------------------------------

def run_backtest(
    returns: pd.DataFrame,
    strategy,
    window: int = 756,
    rebalance: int = 21,
    ann_factor: int = 252,  # annualization factor for daily debug stats only (monthly stats always use 12)
    rf_daily: Optional[pd.Series] = None,
    cfg: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    """
    rolling-window backtest that produces the objects needed for demiguel-style evaluation.

    key conventions:
    - returns are simple daily returns (not log returns)
    - complete-case policy is enforced here: drop any row with any nan in returns
    - risk-free (rf_daily), if provided, is aligned to daily returns
    - transaction costs are applied at the rebalance instant based on turnover

    stats conventions:
    - thesis summary stats are computed on monthly excess returns
    - daily excess and daily stats are returned for debugging only

    outputs (high level):
    - weights, drifted_weights
    - turnover (per rebalance), avg_turnover
    - daily portfolio returns (gross, net)
    - monthly portfolio returns (gross, net)
    - monthly excess returns (gross, net) used for stats
    - wealth series (daily, gross/net)
    - stats (monthly) and stats (daily debug)
    """
    if cfg is None:
        cfg = EvalConfig()

    # enforce chronological order and strict complete-case rows for asset returns
    rets = returns.sort_index().dropna().copy()

    t, n = rets.shape
    if t <= window:
        raise ValueError(f"not enough rows for window={window}: t={t}")
    if rebalance <= 0:
        raise ValueError("rebalance must be positive")

    # align risk-free (daily) if provided
    rf_aligned: Optional[pd.Series] = None
    if rf_daily is not None:
        rf_aligned = rf_daily.sort_index().reindex(rets.index).ffill()

        # if the very first values are still nan, backfill once
        if rf_aligned.isna().any():
            rf_aligned = rf_aligned.bfill()

        if rf_aligned.isna().any():
            raise ValueError("rf_daily alignment produced nan values; check rf coverage")

    # rebalance dates are every 'rebalance' days, starting at first oos day (window)
    rebal_dates = rets.index[window::rebalance]
    if len(rebal_dates) == 0:
        raise ValueError("no rebalance dates produced; check window and rebalance")

    # containers aligned to rebalance dates
    weights_hist: List[np.ndarray] = []
    drift_hist: List[np.ndarray] = []
    turnover_hist: List[float] = []
    weights_dates: List[pd.Timestamp] = []

    # per-day outputs aligned to the daily return index
    port_rets_gross = pd.Series(index=rets.index, dtype=float)
    port_rets_net = pd.Series(index=rets.index, dtype=float)
    wealth_gross = pd.Series(index=rets.index, dtype=float)
    wealth_net = pd.Series(index=rets.index, dtype=float)

    port_rets_gross.iloc[:] = np.nan
    port_rets_net.iloc[:] = np.nan
    wealth_gross.iloc[:] = np.nan
    wealth_net.iloc[:] = np.nan

    # state
    w_current: Optional[np.ndarray] = None
    last_rebal_loc: Optional[int] = None

    # initialize wealth at the day before the first oos day so daily ratios are defined
    first_oos_loc = window
    wealth_gross.iloc[first_oos_loc - 1] = 1.0
    wealth_net.iloc[first_oos_loc - 1] = 1.0

    # main rebalance loop
    for d in rebal_dates:
        loc = int(rets.index.get_loc(d))

        # compute new target weights using trailing window ending at loc-1
        window_slice = rets.iloc[loc - window: loc]

        # pass the aligned rf window slice to strategies that need excess return estimation
        window_rf_slice: Optional[pd.Series] = None
        if rf_aligned is not None:
            window_rf_slice = rf_aligned.iloc[loc - window: loc]

        res = strategy.get_weights(window_slice, window_rf=window_rf_slice)

        w = res.weights.reindex(rets.columns).astype(float)
        if w.isna().any():
            raise ValueError(f"strategy produced nan weights on {d}")

        w = _safe_normalize(w, cfg.eps)
        w_new = w.values.astype(float)

        # apply previous weights from last rebalance through the day before this rebalance date
        if w_current is not None and last_rebal_loc is not None:
            start = last_rebal_loc
            end = loc

            # walk day-by-day so wealth is well-defined
            for i in range(start, end):
                r_i = rets.iloc[i].values.astype(float)

                # daily gross portfolio return under current weights
                gross_ret = float(r_i @ w_current)
                port_rets_gross.iloc[i] = gross_ret

                # gross wealth update
                wg_prev = float(wealth_gross.iloc[i - 1])
                if not np.isfinite(wg_prev) or wg_prev <= 0.0:
                    raise ValueError("invalid wealth_gross state while updating daily returns")
                wealth_gross.iloc[i] = wg_prev * (1.0 + gross_ret)

                # net wealth update (no cost on non-rebalance days)
                wn_prev = float(wealth_net.iloc[i - 1])
                if not np.isfinite(wn_prev) or wn_prev <= 0.0:
                    raise ValueError("invalid wealth_net state while updating daily returns")
                wealth_net.iloc[i] = wn_prev * (1.0 + gross_ret)

                # net return implied from wealth ratio (equals gross_ret on non-rebalance days)
                port_rets_net.iloc[i] = (float(wealth_net.iloc[i]) / float(wealth_net.iloc[i - 1])) - 1.0

            # compute pre-trade weights at rebalance date (before trading), i.e. w_hat_{t+}
            # matches demiguel eq (15): weight differs due to price changes between rebalances
            block = rets.iloc[start:end].values.astype(float)        # rows: days held, cols: assets
            gross_vec = np.prod(1.0 + block, axis=0)                 # cumulative gross return per asset over holding period

            numer = w_current * gross_vec
            denom = float(numer.sum())
            if not np.isfinite(denom) or denom <= cfg.eps:
                raise ValueError("invalid drift denominator in turnover computation")

            w_t_plus = numer / denom                                 # w_hat_{t+} in demiguel notation

            turnover = float(np.abs(w_new - w_t_plus).sum())

            # transaction costs are charged at the rebalance close and reflected in same-day net returns
            if cfg.tc_cost > 0.0 and np.isfinite(turnover) and turnover > 0.0:
                cost_factor = 1.0 - (cfg.tc_cost * turnover)
                cost_factor = max(cost_factor, 0.0)

                wn_before = float(wealth_net.iloc[loc - 1])
                if not np.isfinite(wn_before) or wn_before <= 0.0:
                    raise ValueError("invalid wealth_net at loc-1 before transaction cost haircut")
                wealth_net.iloc[loc - 1] = wn_before * cost_factor

                # update the net return for day (loc-1) after the haircut
                wn_prev = float(wealth_net.iloc[loc - 2])
                if not np.isfinite(wn_prev) or wn_prev <= 0.0:
                    raise ValueError("invalid wealth_net at loc-2; cannot compute net return after tc haircut")
                port_rets_net.iloc[loc - 1] = (float(wealth_net.iloc[loc - 1]) / wn_prev) - 1.0

            drift_hist.append(w_t_plus)
            turnover_hist.append(turnover)

        else:
            # first rebalance: drift and turnover are undefined
            drift_hist.append(np.full(n, np.nan))
            turnover_hist.append(np.nan)

        # record target weights at this rebalance date
        weights_hist.append(w_new)
        weights_dates.append(d)

        # update state to new weights for next holding interval
        w_current = w_new
        last_rebal_loc = loc

    # apply final weights through the end of the sample
    if w_current is not None and last_rebal_loc is not None:
        start = last_rebal_loc
        end = t

        for i in range(start, end):
            r_i = rets.iloc[i].values.astype(float)

            gross_ret = float(r_i @ w_current)
            port_rets_gross.iloc[i] = gross_ret

            wg_prev = float(wealth_gross.iloc[i - 1])
            if not np.isfinite(wg_prev) or wg_prev <= 0.0:
                raise ValueError("invalid wealth_gross state while applying final weights")
            wealth_gross.iloc[i] = wg_prev * (1.0 + gross_ret)

            wn_prev = float(wealth_net.iloc[i - 1])
            if not np.isfinite(wn_prev) or wn_prev <= 0.0:
                raise ValueError("invalid wealth_net state while applying final weights")
            wealth_net.iloc[i] = wn_prev * (1.0 + gross_ret)

            port_rets_net.iloc[i] = (float(wealth_net.iloc[i]) / float(wealth_net.iloc[i - 1])) - 1.0

    # daily excess returns are for debugging only (stats use monthly excess below)
    if rf_aligned is not None:
        port_excess_gross_daily = port_rets_gross - rf_aligned
        port_excess_net_daily = port_rets_net - rf_aligned
    else:
        port_excess_gross_daily = port_rets_gross.copy()
        port_excess_net_daily = port_rets_net.copy()

    # ------------------------------------------------------------
    # monthly aggregation for evaluation (demiguel tables are monthly)
    # ------------------------------------------------------------
    # compound daily portfolio returns to monthly simple returns
    port_ret_gross_m = _to_monthly_simple_returns(port_rets_gross)
    port_ret_net_m = _to_monthly_simple_returns(port_rets_net)

    # compound daily rf to monthly simple returns (if provided)
    if rf_aligned is not None:
        rf_m = _to_monthly_simple_returns(rf_aligned)

        # drop the first month to avoid the partial-month artifact
        # oos starts mid-month (window boundary), so the first resampled month is not a full month
        if len(port_ret_net_m) > 0:
            first_month = port_ret_net_m.index.min()
            port_ret_gross_m = port_ret_gross_m.loc[port_ret_gross_m.index > first_month]
            port_ret_net_m = port_ret_net_m.loc[port_ret_net_m.index > first_month]
            rf_m = rf_m.loc[rf_m.index > first_month]

        # align monthly indices explicitly across gross, net, and rf
        idx_m = port_ret_gross_m.index.intersection(port_ret_net_m.index).intersection(rf_m.index)

        # monthly excess = monthly portfolio return minus monthly rf return
        port_excess_gross_m = port_ret_gross_m.reindex(idx_m) - rf_m.reindex(idx_m)
        port_excess_net_m = port_ret_net_m.reindex(idx_m) - rf_m.reindex(idx_m)
    else:
        port_excess_gross_m = port_ret_gross_m.copy()
        port_excess_net_m = port_ret_net_m.copy()

    # assemble outputs aligned to rebalance dates
    weights_df = pd.DataFrame(
        data=np.vstack(weights_hist),
        index=pd.DatetimeIndex(weights_dates),
        columns=rets.columns,
    )

    drifted_df = pd.DataFrame(
        data=np.vstack(drift_hist),
        index=pd.DatetimeIndex(weights_dates),
        columns=rets.columns,
    )

    turnover_ser = pd.Series(turnover_hist, index=pd.DatetimeIndex(weights_dates), name="turnover")
    avg_turnover = float(turnover_ser.dropna().mean()) if turnover_ser.dropna().shape[0] > 0 else np.nan

    # stats computed on monthly excess returns
    # use ann_factor=12 so mu_ann/sigma_ann/sr_ann correspond to annualized-from-monthly
    stats_gross = compute_oos_moments(port_excess_gross_m, ann_factor=12)
    stats_net = compute_oos_moments(port_excess_net_m, ann_factor=12)

    # daily moments for debugging only
    stats_gross_daily = compute_oos_moments(port_excess_gross_daily, ann_factor=ann_factor)
    stats_net_daily = compute_oos_moments(port_excess_net_daily, ann_factor=ann_factor)

    return {
        # weights and turnover (rebalance dates)
        "weights": weights_df,
        "drifted_weights": drifted_df,
        "turnover": turnover_ser,
        "avg_turnover": avg_turnover,
        # daily portfolio returns
        "portfolio_returns_gross_daily": port_rets_gross,
        "portfolio_returns_net_daily": port_rets_net,
        # monthly portfolio returns
        "portfolio_returns_gross_monthly": port_ret_gross_m,
        "portfolio_returns_net_monthly": port_ret_net_m,
        # daily excess (debug)
        "portfolio_excess_gross_daily": port_excess_gross_daily,
        "portfolio_excess_net_daily": port_excess_net_daily,
        # monthly excess (used for thesis stats)
        "portfolio_excess_gross_monthly": port_excess_gross_m,
        "portfolio_excess_net_monthly": port_excess_net_m,
        # wealth (daily)
        "wealth_gross": wealth_gross,
        "wealth_net": wealth_net,
        # moments
        "stats_gross": stats_gross,
        "stats_net": stats_net,
        "stats_gross_daily": stats_gross_daily,
        "stats_net_daily": stats_net_daily,
        # configuration metadata
        "config": {
            "window": int(window),
            "rebalance": int(rebalance),
            "stats_freq": "monthly",
            "stats_ann_factor": 12,
            "daily_ann_factor": int(ann_factor),
            "ann_factor": int(ann_factor),
            "tc_cost": float(cfg.tc_cost),
            "rf_used": bool(rf_daily is not None),
        },
    }


# ------------------------------------------------------------
# in-sample evaluation
# ------------------------------------------------------------

def run_insample_backtest(
    returns: pd.DataFrame,
    strategy,
    ann_factor: int = 252,  # annualization factor for daily debug stats only (monthly stats always use 12)
    rf_daily: Optional[pd.Series] = None,
    cfg: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    """
    in-sample evaluation:
    - estimate weights once using the full sample
    - apply fixed weights to the same sample
    - no rebalancing, no turnover, no transaction costs

    note:
    - outputs are shaped to match run_backtest() so the same saving + reporting code works.
    """
    if cfg is None:
        cfg = EvalConfig()

    # enforce chronological order and strict complete-case rows for asset returns
    rets = returns.sort_index().dropna().copy()
    t, n = rets.shape
    if t < 5:
        raise ValueError("not enough rows for in-sample evaluation")

    # align risk-free (daily) if provided
    rf_aligned: Optional[pd.Series] = None
    if rf_daily is not None:
        rf_aligned = rf_daily.sort_index().reindex(rets.index).ffill()

        if rf_aligned.isna().any():
            rf_aligned = rf_aligned.bfill()

        if rf_aligned.isna().any():
            raise ValueError("rf_daily alignment produced nan values; check rf coverage")

    # compute one fixed weight vector using the full sample window
    res = strategy.get_weights(rets, window_rf=rf_aligned)

    w = res.weights.reindex(rets.columns).astype(float)
    if w.isna().any():
        raise ValueError("strategy produced nan weights in in-sample evaluation")

    w = _safe_normalize(w, cfg.eps)
    w_vec = w.values.astype(float)

    # daily portfolio returns with fixed weights
    port_rets_gross = pd.Series(rets.values @ w_vec, index=rets.index, dtype=float)

    # no costs and no turnover in-sample
    port_rets_net = port_rets_gross.copy()

    # wealth indices (start at 1.0)
    wealth_gross = (1.0 + port_rets_gross).cumprod()
    wealth_net = (1.0 + port_rets_net).cumprod()

    # daily excess returns are for debugging only (stats use monthly excess below)
    if rf_aligned is not None:
        port_excess_gross_daily = port_rets_gross - rf_aligned
        port_excess_net_daily = port_rets_net - rf_aligned
    else:
        port_excess_gross_daily = port_rets_gross.copy()
        port_excess_net_daily = port_rets_net.copy()

    # ------------------------------------------------------------
    # monthly aggregation for evaluation (keep same convention)
    # ------------------------------------------------------------
    port_ret_gross_m = _to_monthly_simple_returns(port_rets_gross)
    port_ret_net_m = _to_monthly_simple_returns(port_rets_net)

    if rf_aligned is not None:
        rf_m = _to_monthly_simple_returns(rf_aligned)

        # drop the first month to avoid the partial-month artifact (same as run_backtest)
        if len(port_ret_net_m) > 0:
            first_month = port_ret_net_m.index.min()
            port_ret_gross_m = port_ret_gross_m.loc[port_ret_gross_m.index > first_month]
            port_ret_net_m = port_ret_net_m.loc[port_ret_net_m.index > first_month]
            rf_m = rf_m.loc[rf_m.index > first_month]

        idx_m = port_ret_gross_m.index.intersection(port_ret_net_m.index).intersection(rf_m.index)
        port_excess_gross_m = port_ret_gross_m.reindex(idx_m) - rf_m.reindex(idx_m)
        port_excess_net_m = port_ret_net_m.reindex(idx_m) - rf_m.reindex(idx_m)
    else:
        port_excess_gross_m = port_ret_gross_m.copy()
        port_excess_net_m = port_ret_net_m.copy()

    # weights artifacts:
    # - one row (single fixed weight vector)
    # - drift/turnover not defined for a non-rebalanced in-sample portfolio
    weights_df = pd.DataFrame(
        data=np.asarray(w_vec, dtype=float).reshape(1, -1),
        index=pd.DatetimeIndex([rets.index[0]]),
        columns=rets.columns,
    )

    drifted_df = pd.DataFrame(
        data=np.full((1, n), np.nan),
        index=weights_df.index,
        columns=rets.columns,
    )

    turnover_ser = pd.Series([np.nan], index=weights_df.index, name="turnover")
    avg_turnover = np.nan

    # stats computed on monthly excess returns
    stats_gross = compute_oos_moments(port_excess_gross_m, ann_factor=12)
    stats_net = compute_oos_moments(port_excess_net_m, ann_factor=12)

    # daily moments for debugging only
    stats_gross_daily = compute_oos_moments(port_excess_gross_daily, ann_factor=ann_factor)
    stats_net_daily = compute_oos_moments(port_excess_net_daily, ann_factor=ann_factor)

    return {
        # weights and turnover (single-row placeholder date)
        "weights": weights_df,
        "drifted_weights": drifted_df,
        "turnover": turnover_ser,
        "avg_turnover": avg_turnover,
        # daily portfolio returns
        "portfolio_returns_gross_daily": port_rets_gross,
        "portfolio_returns_net_daily": port_rets_net,
        # monthly portfolio returns
        "portfolio_returns_gross_monthly": port_ret_gross_m,
        "portfolio_returns_net_monthly": port_ret_net_m,
        # daily excess (debug)
        "portfolio_excess_gross_daily": port_excess_gross_daily,
        "portfolio_excess_net_daily": port_excess_net_daily,
        # monthly excess (used for thesis stats)
        "portfolio_excess_gross_monthly": port_excess_gross_m,
        "portfolio_excess_net_monthly": port_excess_net_m,
        # wealth (daily)
        "wealth_gross": wealth_gross,
        "wealth_net": wealth_net,
        # moments
        "stats_gross": stats_gross,
        "stats_net": stats_net,
        "stats_gross_daily": stats_gross_daily,
        "stats_net_daily": stats_net_daily,
        # configuration metadata
        "config": {
            "mode": "in_sample",
            "stats_freq": "monthly",
            "stats_ann_factor": 12,
            "daily_ann_factor": int(ann_factor),
            "ann_factor": int(ann_factor),
            "tc_cost": 0.0,
            "rf_used": bool(rf_daily is not None),
        },
    }
