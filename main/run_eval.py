# main/run_eval.py
#
# purpose:
# - run rolling-window out-of-sample backtests for multiple universes and strategies
# - save all artifacts needed for thesis tables/figures (weights, turnover, returns, wealth, stats)
# - compute demiguel-style evaluation add-ons:
#     * ceq (certainty equivalent) for multiple risk aversion parameters gamma
#     * return-loss relative to the ew benchmark
#
# key conventions (must match eval_engine.py):
# - inputs are daily simple returns (not log returns)
# - engine enforces strict complete-case rows for asset returns (drop any row with any nan)
# - evaluation statistics are computed on monthly excess returns (annualized from monthly with factor 12)
# - daily stats are also returned for debugging only
# - transaction costs are applied at rebalance instants based on turnover
#
# outputs:
# - results are written under: data/results/<strategy>/
# - daily and monthly series are saved separately with explicit suffixes

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# strategies
from strategies.ew import EqualWeight
from strategies.msr_is import MaxSharpeInSample
from strategies.mv import MinVariance
from strategies.msr import MaxSharpe
from strategies.dm import DM, DMConfig
from strategies.erc import EqualRiskContribution
from strategies.hrp import HierarchicalRiskParity
from strategies.nco import NestedClusterOptimization

# inference
from inference.lw_bootstrap import Algo31Config, BootTSConfig, run_lw_suite_for_universe

# evaluation engine + metrics
from evaluation.eval_engine import (
    run_backtest,
    run_insample_backtest,  # in-sample evaluation path
    EvalConfig,
    compute_ceq,
    compute_return_loss,
)


# ------------------------------------------------------------
# universes and evaluation settings
# ------------------------------------------------------------

# universes correspond to files created by data_pipeline.py:
# data/processed/rets_<universe>.csv
UNIVERSES = [
    "benchmark_12",
    "us_sectors_11",
    "intl_countries_15",
    "intl_countries_20",
    "intl_countries_30",
]

# stable universe labels used in filenames (keeps results navigable without universe subfolders)
UNIVERSE_LABELS = {
    "benchmark_12": "bm",
    "us_sectors_11": "sec",
    "intl_countries_15": "int15",
    "intl_countries_20": "int20",
    "intl_countries_30": "int30",
}

# rolling estimation window in trading days:
# - 756 ~= 3 years of daily data
WINDOW = 756

# rebalance frequency in trading days:
# - 21 ~= 1 month of daily data
REBALANCE = 21

# daily annualization factor:
# - used only for the debug daily stats in eval_engine.py
DAILY_ANN_FACTOR = 252

# demiguel risk aversion grid:
# - baseline gamma=1 and robustness gamma in {2,3,4,5,10}
GAMMAS = [1, 2, 3, 4, 5, 10]

# proportional transaction cost per unit traded:
# - applied at rebalance instants using turnover (sum |w_new - w_drifted|)
TC_COST = 0.005

# inference rng seed (kept separate from strategy seeds)
INFERENCE_SEED = 12345

# data directories
DATA_DIR = Path("data")
RETS_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# daily risk-free series produced by data_pipeline.py:
# - expected to contain a single column of daily simple rf returns
RF_PATH = RETS_DIR / "rf_1m_daily.csv"

# ------------------------------------------------------------
# io helpers
# ------------------------------------------------------------

def load_returns(universe: str) -> pd.DataFrame:
    """
    load a universe return matrix from csv
    
    expected file:
     data/processed/rets_<universe>.csv
    
    expected shape:
     (t x n) where columns are asset identifiers (tickers)
    
    expected contents:
     daily simple returns, already aligned across assets
    """
    path = RETS_DIR / f"rets_{universe}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def load_rf_daily() -> pd.Series:
    """
    load daily risk-free returns from csv
    
    expected file:
    data/processed/rf_1m_daily.csv
    
    expected contents:
     daily simple rf returns aligned to the daily trading calendar used for prices
    """
    if not RF_PATH.exists():
        raise FileNotFoundError(
            f"missing risk-free file: {RF_PATH}. "
            "run data_pipeline.py to create data/processed/rf_1m_daily.csv"
        )

    df = pd.read_csv(RF_PATH, index_col=0, parse_dates=True)

    # pipeline writes a single column; keep defensive logic anyway
    if df.shape[1] == 1:
        s = df.iloc[:, 0].astype(float)
        s.name = "rf_daily"
        return s.sort_index()

    # if multiple columns exist, attempt a reasonable match
    for c in ["rf_daily", "RF", "risk_free", "rf"]:
        if c in df.columns:
            s = df[c].astype(float)
            s.name = "rf_daily"
            return s.sort_index()

    raise ValueError(f"could not identify rf column in {RF_PATH}. columns={list(df.columns)}")


# cleanup helper to remove stale result files before writing new ones
def _cleanup_strategy_dir(strategy_name: str) -> None:
    """
    erase prior outputs for a given strategy so each run produces a clean, current results directory.

    behavior:
    - deletes only files with suffix .csv or .json inside data/results/<strategy_name>/
    - leaves directories intact
    """
    sdir = OUT_DIR / strategy_name
    if not sdir.exists():
        return

    for p in sdir.iterdir():
        if p.is_file() and p.suffix.lower() in {".csv", ".json"}:
            try:
                p.unlink()
            except Exception:
                # best-effort cleanup; any failure should not crash the run
                pass


def save_outputs(universe: str, strategy_name: str, results: Dict[str, Any]) -> None:
    """
    save all artifacts produced by eval_engine.run_backtest()
    
    directory convention:
    data/results/<strategy_name>/
    
    filename convention:
     <strategy>_<universe_label>_<artifact>.csv (or .json)
    
    saved objects:
     - weights and drifted weights on rebalance dates
     - turnover on rebalance dates
     - daily portfolio returns/excess and wealth series (debug + path plots)
     - monthly portfolio returns/excess (primary thesis evaluation series)
     - stats.json (compact metrics + config)
    """

    # remove universe subfolders; write into strategy folder
    ulabel = UNIVERSE_LABELS.get(universe, universe)
    sdir = OUT_DIR / strategy_name
    sdir.mkdir(parents=True, exist_ok=True)

    # strategy + universe prefix for all output files
    prefix = f"{strategy_name}_{ulabel}_"

    # ------------------------------------------------------------
    # weights and trading diagnostics
    # ------------------------------------------------------------

    # target weights at each rebalance date (n_rebals x n_assets)
    results["weights"].to_csv(sdir / f"{prefix}weights.csv")

    # drifted weights at the rebalance instant, w_{t^+} (n_rebals x n_assets)
    results["drifted_weights"].to_csv(sdir / f"{prefix}drifted_weights.csv")

    # turnover at each rebalance date (n_rebals x 1)
    results["turnover"].to_csv(sdir / f"{prefix}turnover.csv", header=["turnover"])

    # ------------------------------------------------------------
    # daily series (debug and path construction)
    # ------------------------------------------------------------

    # daily gross and net simple returns
    results["portfolio_returns_gross_daily"].to_csv(
        sdir / f"{prefix}portfolio_returns_gross_daily.csv", header=["ret"]
    )
    results["portfolio_returns_net_daily"].to_csv(
        sdir / f"{prefix}portfolio_returns_net_daily.csv", header=["ret"]
    )

    # daily excess returns (gross/net), if rf is used; otherwise equals raw daily returns
    results["portfolio_excess_gross_daily"].to_csv(
        sdir / f"{prefix}portfolio_excess_gross_daily.csv", header=["ret_excess"]
    )
    results["portfolio_excess_net_daily"].to_csv(
        sdir / f"{prefix}portfolio_excess_net_daily.csv", header=["ret_excess"]
    )

    # daily wealth indices (starting at 1 at the oos boundary)
    results["wealth_gross"].to_csv(sdir / f"{prefix}wealth_gross_daily.csv", header=["wealth"])
    results["wealth_net"].to_csv(sdir / f"{prefix}wealth_net_daily.csv", header=["wealth"])

    # ------------------------------------------------------------
    # monthly series (primary thesis evaluation series)
    # ------------------------------------------------------------

    # monthly gross/net returns computed by compounding daily returns within each month
    results["portfolio_returns_gross_monthly"].to_csv(
        sdir / f"{prefix}portfolio_returns_gross_monthly.csv", header=["ret"]
    )
    results["portfolio_returns_net_monthly"].to_csv(
        sdir / f"{prefix}portfolio_returns_net_monthly.csv", header=["ret"]
    )

    # monthly excess returns computed as:
    # excess_m = port_ret_m - rf_ret_m
    results["portfolio_excess_gross_monthly"].to_csv(
        sdir / f"{prefix}portfolio_excess_gross_monthly.csv", header=["ret_excess"]
    )
    results["portfolio_excess_net_monthly"].to_csv(
        sdir / f"{prefix}portfolio_excess_net_monthly.csv", header=["ret_excess"]
    )

    # ------------------------------------------------------------
    # json summary (thesis tables built from these)
    # ------------------------------------------------------------

    payload = {
        # includes window, rebalance, tc_cost, and other run metadata
        "config": results.get("config", {}),

        # average turnover across rebalances
        "avg_turnover": results.get("avg_turnover", None),

        # monthly stats (primary thesis stats): computed on monthly excess returns, ann_factor=12
        "stats_gross": results.get("stats_gross", {}),
        "stats_net": results.get("stats_net", {}),

        # daily stats (debug): computed on daily excess returns, ann_factor=252
        "stats_gross_daily": results.get("stats_gross_daily", {}),
        "stats_net_daily": results.get("stats_net_daily", {}),

        # evaluation add-ons computed in this script (annual units)
        "ceq_by_gamma_net": results.get("ceq_by_gamma_net", {}),
        "return_loss_net": results.get("return_loss_net", None),
    }

    with open(sdir / f"{prefix}stats.json", "w") as f:
        json.dump(payload, f, indent=2)

# ------------------------------------------------------------
# printing helpers (console output only)
# ------------------------------------------------------------

def _fmt(x: Any, nd: int = 4) -> str:
    # robust float formatting for console output
    # returns "nan" for non-finite values
    if x is None:
        return "nan"
    try:
        xf = float(x)
    except Exception:
        return "nan"
    if not np.isfinite(xf):
        return "nan"
    return f"{xf:.{nd}f}"

def _extract_ann_stats(stats: Dict[str, Any]) -> Dict[str, float]:
    """
    extract annualized stats from an eval_engine stats dict

    important:
     - eval_engine computes stats_net/stats_gross on monthly excess returns
     - those stats are annualized using ann_factor=12 inside the engine
    """
    return {
        "mu_ann": float(stats.get("mu_ann", np.nan)),
        "sigma_ann": float(stats.get("sigma_ann", np.nan)),
        "sr_ann": float(stats.get("sr_ann", np.nan)),
    }

def _compute_ceq_ann_dict(mu_ann: float, sigma_ann: float) -> Dict[str, float]:
    """
    compute annualized certainty equivalent return for each gamma

    definition (annual units):
    ceq(gamma) = mu - (gamma/2) * sigma^2

    inputs must be annualized (mu_ann, sigma_ann) for ceq to be annual
    """
    out: Dict[str, float] = {}
    for g in GAMMAS:
        out[str(g)] = float(compute_ceq(mu_ann, sigma_ann, gamma=float(g)))
    return out

def _print_strategy_summary(
    universe: str,
    strategy_name: str,
    res: Dict[str, Any],
    return_loss_ann: Optional[float],
    ceq_by_gamma_ann: Dict[str, float],
) -> None:
    """
    print one-line annualized performance summary plus ceq values
    
    note:
     - the mu/vol/sr displayed are annualized from monthly excess returns
     - return_loss_ann is in annual units
    """
    net = _extract_ann_stats(res.get("stats_net", {}))
    gross = _extract_ann_stats(res.get("stats_gross", {}))
    avg_to = res.get("avg_turnover", np.nan)

    rl_ann = float(return_loss_ann) if return_loss_ann is not None else np.nan

    print(
        f"{universe} | {strategy_name} | "
        f"net(mu={_fmt(net['mu_ann'], 4)}, vol={_fmt(net['sigma_ann'], 4)}, sr={_fmt(net['sr_ann'], 3)}) | "
        f"gross(mu={_fmt(gross['mu_ann'], 4)}, vol={_fmt(gross['sigma_ann'], 4)}, sr={_fmt(gross['sr_ann'], 3)}) | "
        f"avg_turnover={_fmt(avg_to, 6)} | "
        f"return_loss_ann={_fmt(rl_ann, 4)}"
    )

    # show ceq for two endpoints plus the full map
    ceq1 = ceq_by_gamma_ann.get("1", np.nan)
    ceq10 = ceq_by_gamma_ann.get("10", np.nan)
    print(f"  ceq_ann: gamma=1 -> {_fmt(ceq1, 4)} | gamma=10 -> {_fmt(ceq10, 4)}")

    ceq_ann_str = ", ".join([f"{k}:{_fmt(v, 4)}" for k, v in ceq_by_gamma_ann.items()])
    print(f"  ceq_ann_all: {ceq_ann_str}")

def _nan_report(universe: str, df: pd.DataFrame) -> None:
    """
    report missingness in the raw return matrix loaded from disk

    purely diagnostic:
     - eval_engine will enforce complete-case policy regardless
     - if missingness exists, document the degree of row dropping
    """
    total = int(df.size)
    n_nan = int(df.isna().sum().sum())
    rows_any = int(df.isna().any(axis=1).sum())
    pct = (100.0 * n_nan / total) if total > 0 else np.nan

    if n_nan == 0:
        print(f"{universe} | nan check: 0 nans found (no rows would be dropped by complete-case).")
        return

    print(
        f"{universe} | nan check: nans_found={n_nan} ({pct:.4f}% of cells), "
        f"rows_with_any_nan={rows_any} / {len(df)}"
    )

# ------------------------------------------------------------
# main evaluation loop
# ------------------------------------------------------------

def main() -> None:
    """
    strategy list is defined in one place so adding strategies later is easy

    important ordering convention:
     - ew must be first, since return-loss is computed relative to ew
    """
    strategies: List[Any] = [
        # add strategies
        EqualWeight(),        # naive benchmark strategy
        MaxSharpeInSample(),  # in-sample msr 
        MinVariance(),        # mv
        MaxSharpe(),          # msr
        # DM(DMConfig(name="dm_us", market_path="data/processed/mkt_us_daily.csv")),  # dm (us market proxy) -> do not report
        EqualRiskContribution(),  # erc
        HierarchicalRiskParity(), # hrp
        NestedClusterOptimization(base="mv"),  # nco with minimum-variance
        NestedClusterOptimization(base="msr"), # nco with Maximum Sharpe Ratio
   
    ]

    # dm (global market proxy)
    # dm_global sigma_alpha grid (annual units)
    # unique names are required because results are written under data/results/<strategy_name>/
    sigma_grid = [0.01, 0.02, 0.03, 0.04, 0.05]
    for s in sigma_grid:
        strategies.append(
            DM(DMConfig(
                name=f"dm_global_sa{int(round(100*s))}",  # dm_global_sa1 ... dm_global_sa5
                sigma_alpha_ann=s,
                market_path="data/processed/mkt_global_daily.csv",
            ))
        )

    # diagnostic: confirm instantiated strategy configs (especially dm market_path)
#    for s in strategies:
#        if hasattr(s, "cfg"):
#            print(s.name, "cfg:", getattr(s, "cfg"))
#        if hasattr(s, "cfg") and hasattr(getattr(s, "cfg"), "market_path"):
#            print(s.name, "market_path:", getattr(s.cfg, "market_path"))

    # clean strategy result dirs once per run so outputs are always "current run only"
    for s in strategies:
        _cleanup_strategy_dir(s.name)

    # eval configuration shared across strategies:
    # - tc_cost is the proportional cost per unit turnover
    cfg = EvalConfig(tc_cost=TC_COST)

    # load rf series once and reuse across universes
    rf_daily = load_rf_daily()

    # loop over universes and run all strategies
    for u in UNIVERSES:
        # load the raw daily return matrix for this universe
        rets_raw = load_returns(u)

        # print missingness diagnostics (engine still enforces complete-case)
        _nan_report(u, rets_raw)

        # basic console progress output
        print(f"running universe={u}  t={len(rets_raw)}  n={rets_raw.shape[1]}")

        # ------------------------------------------------------------
        # run ew benchmark first (reference strategy)
        # ------------------------------------------------------------

        ew = strategies[0]
        ew_res = run_backtest(
            returns=rets_raw,
            strategy=ew,
            window=WINDOW,
            rebalance=REBALANCE,
            # this controls only the annualization of debug daily stats
            ann_factor=DAILY_ANN_FACTOR,
            rf_daily=rf_daily,
            cfg=cfg,
        )

        # collect daily net excess returns for hypothesis testing (lw uses bivariate return series)
        excess_daily_map: Dict[str, pd.Series] = {}
        excess_daily_map[ew.name] = ew_res["portfolio_excess_net_daily"].rename(ew.name)

        # eval_engine stats are computed on monthly excess returns and annualized with factor 12
        # extract annualized mean and volatility to use as references
        ew_net_ann = _extract_ann_stats(ew_res.get("stats_net", {}))
        mu_ew_ann = ew_net_ann["mu_ann"]
        sigma_ew_ann = ew_net_ann["sigma_ann"]

        # compute annual ceq values for the ew benchmark
        # note: ceq is stored as annual units in the result dict
        ew_ceq_ann = _compute_ceq_ann_dict(mu_ew_ann, sigma_ew_ann)
        ew_res["ceq_by_gamma_net"] = ew_ceq_ann

        # by definition, ew return-loss relative to itself is zero
        ew_res["return_loss_net"] = (
            0.0 if np.isfinite(mu_ew_ann) and np.isfinite(sigma_ew_ann) and sigma_ew_ann > 0 else np.nan
        )

        # persist all ew artifacts for this universe
        save_outputs(u, ew.name, ew_res)

        # print ew summary to console
        _print_strategy_summary(
            universe=u,
            strategy_name=ew.name,
            res=ew_res,
            return_loss_ann=0.0,
            ceq_by_gamma_ann=ew_ceq_ann,
        )

        # ------------------------------------------------------------
        # run remaining strategies (optimized portfolios)
        # ------------------------------------------------------------

        for strat in strategies[1:]:
            # in-sample branch (single fixed weight vector, no rebalancing)
            if bool(getattr(strat, "is_insample", False)):
                res = run_insample_backtest(
                    returns=rets_raw,
                    strategy=strat,
                    ann_factor=DAILY_ANN_FACTOR,
                    rf_daily=rf_daily,
                    cfg=cfg,
                )
            else:
                # run out-of-sample rolling-window backtest
                res = run_backtest(
                    returns=rets_raw,
                    strategy=strat,
                    window=WINDOW,
                    rebalance=REBALANCE,
                    ann_factor=DAILY_ANN_FACTOR,
                    rf_daily=rf_daily,
                    cfg=cfg,
                )

            # collect daily net excess returns for lw inference
            if not bool(getattr(strat, "is_insample", False)):
                excess_daily_map[strat.name] = res["portfolio_excess_net_daily"].rename(strat.name)

            # diagnostic: for dm, print minimal runtime evidence of rf/mkt usage
            if hasattr(strat, "last_info") and getattr(strat, "last_info") is not None:
                li = getattr(strat, "last_info") or {}
                dbg = {k: li[k] for k in li if k.startswith("debug_")}
                if len(dbg) > 0:
                    print(strat.name, "last_info:", dbg)

            # extract annualized moments (from monthly excess stats) for ceq and return-loss
            k_net_ann = _extract_ann_stats(res.get("stats_net", {}))
            mu_k_ann = k_net_ann["mu_ann"]
            sigma_k_ann = k_net_ann["sigma_ann"]

            # compute annual ceq values for this strategy
            ceq_ann = _compute_ceq_ann_dict(mu_k_ann, sigma_k_ann)
            res["ceq_by_gamma_net"] = ceq_ann

            # compute return-loss relative to the ew benchmark using annual moments
            rl_ann = compute_return_loss(
                mu_ew=mu_ew_ann,
                sigma_ew=sigma_ew_ann,
                mu_k=mu_k_ann,
                sigma_k=sigma_k_ann,
            )
            res["return_loss_net"] = float(rl_ann) if rl_ann is not None else np.nan

            # save artifacts to disk
            save_outputs(u, strat.name, res)

            # print summary to console
            _print_strategy_summary(
                universe=u,
                strategy_name=strat.name,
                res=res,
                return_loss_ann=rl_ann,
                ceq_by_gamma_ann=ceq_ann,
            )

        # ------------------------------------------------------------
        # lw (2008) hypothesis testing: sharpe differences vs ew benchmark
        # ------------------------------------------------------------

        # align all strategy daily excess return series on common dates
        excess_df = pd.concat(excess_daily_map.values(), axis=1).dropna()

        # require enough overlap for stable VAR(p) estimation and LW block bootstrap
        # ~2 years of daily data (~500 obs) is imposed as a numerical stability guard
        min_t = 500  
        if excess_df.shape[0] < min_t:
            print(f"{u} | lw skipped: insufficient overlapping daily observations (t={excess_df.shape[0]})")
            ulabel = UNIVERSE_LABELS.get(u, u)

            # write a small placeholder artifact so downstream thesis code can rely on file existence
            pd.DataFrame([{
                "universe": u,
                "status": "skipped",
                "reason": "insufficient_overlap",
                "t_overlap": int(excess_df.shape[0]),
                "min_t": int(min_t),
            }]).to_csv(OUT_DIR / f"hyp_tests_{ulabel}.csv", index=False)

            print(f"done universe={u}")
            continue

        algo31_cfg = Algo31Config(
            b_grid=(1, 2, 4, 6, 8, 10),
            k_pseudo=5000,              # lw configuration: k=5000
            m_inner=199,      
            alpha=0.05,
            show_progress=True,
            progress_min_interval_s=0.5,
            var_order=1,                # lw section 4.1 uses var(1) for p-hat
            resid_sb_avg_block=5,       # lw section 4.1 uses stationary bootstrap avg block size 5
            seed=INFERENCE_SEED,
        )

        final_cfg = BootTSConfig(
            alpha=0.05,
            m=4999,                      # lw configuration: m=4999 in the empirical apllication 
            show_progress=False,
            progress_min_interval_s=0.5,
            seed=123,
        )

        # exclude in-sample strategies from hypothesis tests
        # keep only dm_global_sa5 among DM variants for the formal LW inference
        # this avoids running separate hypothesis tests for every sigma_alpha calibration
        # within the same strategy family and keeps the model comparison balanced
        # dm_global_sa5 is the least dogmatic pre-specified calibration in the tested grid
        test_names = []
        for s in strategies:
            if s.name == ew.name:
                continue
            if bool(getattr(s, "is_insample", False)):
                continue

            # keep only dm_global_sa5 among DM variants
            if s.name.startswith("dm_") and s.name != "dm_global_sa5":
                continue

            test_names.append(s.name)
            
        cols = [ew.name] + test_names
        missing = [c for c in cols if c not in excess_df.columns]
        if missing:
            raise ValueError(f"{u} | lw: missing required columns in excess_df: {missing}")

        excess_df = excess_df.loc[:, cols]

        tests_df, _ = run_lw_suite_for_universe(
            excess_rets_daily=excess_df,
            benchmark_col=ew.name,
            strategies=test_names,
            algo31_cfg=algo31_cfg,
            final_cfg=final_cfg,
            report_all_b=False,
            prewhiten_hac=True,
        )

        print(f"{u} | lw boot-ts sharpe tests vs {ew.name} (reported at b*)")
        for strat_name in sorted(tests_df["strategy"].unique()):
            sub = tests_df[
                (tests_df["strategy"] == strat_name)
                & (tests_df["block_len"] == tests_df["block_len_selected"])
            ]
            if sub.empty:
                continue
            row = sub.iloc[0]
            print(
                f"  {strat_name}: "
                f"delta_sr={row['delta_sharpe']:.4f}, "
                f"p={row['p_value']:.4f}, "
                f"ci=[{row['ci_low']:.4f}, {row['ci_high']:.4f}], "
                f"b*={int(row['block_len_selected'])}, "
                f"g(b*)={row['g_of_b']:.3f}, "
                f"invalid={row['invalid_frac']:.3f}"
            )

        ulabel = UNIVERSE_LABELS.get(u, u)
        tests_df.to_csv(OUT_DIR / f"hyp_tests_{ulabel}.csv", index=False)

        print(f"done universe={u}")

    print("done all")


if __name__ == "__main__":
    main()
