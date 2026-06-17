# main/sec_eval.py
#
# purpose:
# - produce the secondary robustness and presentation outputs needed for the thesis
# - restrict all outputs to the core multi-asset universe: benchmark_12
# - consume artifacts already written by main/run_eval.py
# - rerun window-length sensitivity with benchmark-only LW inference on the full window grid
# - rerun the benchmark LW inference layer using the exact frozen thesis configuration
#   from main/run_eval.py, so the benchmark inference summary is always reproducible
# - produce appendix-ready HRP matrix diagnostics for the benchmark universe using
#   the final estimation window and the current hrp implementation
#
# design principle:
# - this script is a benchmark-only reporting and robustness layer
# - it reads the stored backtest artifacts written by main/run_eval.py
# - it does not rerun the full multi-universe evaluation
# - it does rerun benchmark-only LW inference from stored artifacts so the frozen
#   thesis benchmark inference remains reproducible even if the raw file is missing
# - it also reruns benchmark-only LW inference across the window-length grid using
#   fresh benchmark backtests for the sensitivity analysis
# - it also reads the processed benchmark return panel directly to construct the
#   HRP covariance and correlation matrix figures used in the appendix
#
# outputs produced:
# - data/secondary_eval/robustness_summary_bm.csv
# - data/secondary_eval/inference_summary_bm.csv
# - data/secondary_eval/hyp_tests_bm_full.csv
# - data/secondary_eval/window_length_sensitivity_bm.csv
# - data/secondary_eval/window_inference_summary_bm.csv
# - data/secondary_eval/window_hyp_tests_bm_full.csv

# - family-specific benchmark figure files under data/results/figures/:
# - data/results/figures/wealth_usd_levels_bm_{family}.png
# - data/results/figures/wealth_diff_vs_ew_usd_bm_{family}.png
# - data/results/figures/ceq_vs_gamma_diff_bm_{family}.png
# - data/results/figures/effective_n_bm_{family}.png
# - data/results/figures/max_weight_bm_{family}.png
# - data/results/figures/turnover_bm_{family}.png

# HRP visualization figures:
# - data/results/figures/hrp_cov_before_after_bm.png
# - data/results/figures/hrp_corr_before_after_bm.png
#
# inputs expected from main/run_eval.py:
# - data/results/<strategy>/<strategy>_bm_stats.json
# - data/results/<strategy>/<strategy>_bm_weights.csv
# - data/results/<strategy>/<strategy>_bm_turnover.csv
# - data/results/<strategy>/<strategy>_bm_wealth_net_daily.csv
#
# additional direct benchmark input:
# - data/processed/rets_benchmark_12.csv
#
# benchmark-only inference configuration:
# - benchmark: ew
# - tested strategies: mv, msr, erc, hrp, nco_mv, nco_msr, dm_global_sa5
# - algo31 configuration:
#     b_grid=(1, 2, 4, 6, 8, 10)
#     k_pseudo=5000
#     m_inner=199
#     alpha=0.05
#     var_order=1
#     resid_sb_avg_block=5
#     seed=12345
# - final bootstrap configuration:
#     alpha=0.05
#     m=4999
#     seed=123
# - prewhiten_hac=True
# - report_all_b=False
#
# scope:
# - only the benchmark_12 universe is processed
# - only the strategies intended for thesis reporting are used
# - only the figures and tables plausibly usable in the main text or appendix are written


from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd

# benchmark-only backtest sensitivity reruns
from evaluation.eval_engine import EvalConfig, run_backtest

# shared loaders from the main evaluation script
from main.run_eval import load_returns, load_rf_daily

# strategies
from strategies.dm import DM, DMConfig
from strategies.erc import EqualRiskContribution
from strategies.ew import EqualWeight
from strategies.hrp import (
    HRPConfig,
    HierarchicalRiskParity,
    tree_clustering,
    _get_quasi_diag,
)
from strategies.msr import MaxSharpe
from strategies.mv import MinVariance
from strategies.nco import NestedClusterOptimization

# inference
from inference.lw_bootstrap import Algo31Config, BootTSConfig, run_lw_suite_for_universe


# ------------------------------------------------------------
# fixed thesis configuration
# ------------------------------------------------------------

# core universe used for the secondary analysis
UNIVERSE = "benchmark_12"
ULABEL = "bm"

# all strategies included in the artifact-based benchmark summary and figures
# these names must match the strategy folder names under data/results/
STRATEGIES = [
    "ew",
    "mv",
    "msr",
    "erc",
    "hrp",
    "nco_mv",
    "nco_msr",
    "dm_global_sa5",
]

DISPLAY_NAMES = {
    "ew": "EW",
    "mv": "MV",
    "msr": "MSR",
    "erc": "ERC",
    "hrp": "HRP",
    "nco_mv": "NCO-MV",
    "nco_msr": "NCO-MSR",
    "dm_global_sa5": "DM-CAPM",
}

MEAN_FAMILY = [
    "ew",
    "msr",
    "dm_global_sa5",
    "nco_msr",
]

COVARIANCE_FAMILY = [
    "ew",
    "mv",
    "erc",
    "hrp",
    "nco_mv",
]


# ew is the benchmark for relative wealth and ceq figures
BASELINE = "ew"

# initial capital used to convert the normalized wealth index into dollar wealth
INITIAL_CAPITAL = 1_000_000.0

# mandatory benchmark window-length sensitivity settings
WINDOWS = [504, 756, 1260]  # 2y, 3y, 5y in trading days
REBALANCE = 21              # ~ monthly rebalancing
TC_COST = 0.005             # 50 bps per unit turnover

# benchmark-only inference settings
INFERENCE_SEED = 12345
BOOTTS_SEED = 123

# final thesis benchmark inference strategy set
# these are the only strategies formally tested against ew in the main run
INFERENCE_TEST_STRATEGIES = [
    "mv",
    "msr",
    "erc",
    "hrp",
    "nco_mv",
    "nco_msr",
    "dm_global_sa5",
]

# plotting defaults
FIGSIZE_WIDE = (10.5, 6.0)
FIGSIZE_CEQ = (8.8, 5.6)
LINEWIDTH = 1.6
LEGEND_FONTSIZE = 10
#TITLE_FONTSIZE = 13
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 11

# hrp appendix figure settings
HRP_WINDOW = 756  # final benchmark estimation window used for the appendix matrices
HRP_LINKAGE_METHOD = "single"
HRP_FIGSIZE = (10.2, 12.2)
HRP_DPI = 350

# directory structure
DATA_DIR = Path("data")
RESULTS_DIR = DATA_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
OUT_DIR = DATA_DIR / "secondary_eval"

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# terminal status helpers
# ------------------------------------------------------------

def _status(msg: str) -> None:
    """Print a concise progress line for terminal monitoring"""
    print(f"[sec_eval] {msg}", flush=True)


def _elapsed_str(t0: float) -> str:
    """Format elapsed time in seconds or minutes for compact status reporting"""
    dt = time.time() - t0
    return f"{dt:.1f}s" if dt < 60 else f"{dt / 60:.1f}m"


# ------------------------------------------------------------
# file and artifact helpers
# ------------------------------------------------------------

def _artifact_path(strategy_name: str, artifact: str, ext: str) -> Path:
    """
    Build the path to one benchmark artifact produced by main/run_eval.py

    File convention:
        data/results/<strategy>/<strategy>_bm_<artifact>.<ext>

    Example:
        data/results/ew/ew_bm_stats.json
    """
    return RESULTS_DIR / strategy_name / f"{strategy_name}_{ULABEL}_{artifact}.{ext}"


def _read_df_csv(path: Path) -> pd.DataFrame:
    """Read a csv artifact as a dataframe with a parsed datetime index."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _read_series_csv(path: Path) -> pd.Series:
    """
    Read a single-column csv artifact as a pandas Series

    The stored backtest artifacts are generally single-column files
    If an empty file is encountered, return an empty float series
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.shape[1] == 0:
        return pd.Series(dtype=float)
    return df.iloc[:, 0].astype(float)


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a json artifact into a Python dictionary"""
    with open(path, "r") as f:
        return json.load(f)


def _exists_bundle(strategy: str) -> bool:
    """
    Check whether the minimum required benchmark artifact bundle exists for a strategy

    Required files:
    - stats json
    - weights csv
    - turnover csv
    - wealth_net_daily csv
    """
    return (
        _artifact_path(strategy, "stats", "json").exists()
        and _artifact_path(strategy, "weights", "csv").exists()
        and _artifact_path(strategy, "turnover", "csv").exists()
        and _artifact_path(strategy, "wealth_net_daily", "csv").exists()
    )


# ------------------------------------------------------------
# data transformation helpers
# ------------------------------------------------------------

def _effective_n(weights: pd.DataFrame) -> pd.Series:
    """
    Compute the effective number of assets over rebalance dates

    Formula:
        n_eff(t) = 1 / sum_i w_i(t)^2

    Interpretation:
    - close to 1: extreme concentration
    - close to N: near-equal allocation
    """
    w = weights.astype(float).copy()
    w = w.div(w.sum(axis=1), axis=0)
    out = 1.0 / (w.pow(2).sum(axis=1))
    out.name = "eff_n"
    return out


def _max_weight(weights: pd.DataFrame) -> pd.Series:
    """
    Compute the maximum single-asset portfolio weight over rebalance dates

    Formula:
        max_w(t) = max_i w_i(t)

    Interpretation:
    - larger values indicate stronger portfolio concentration
    """
    w = weights.astype(float).copy()
    w = w.div(w.sum(axis=1), axis=0)
    out = w.max(axis=1)
    out.name = "max_w"
    return out


def _wealth_usd(wealth: pd.Series, initial_capital: float) -> pd.Series:
    """
    Convert the normalized wealth index into a dollar wealth path

    The evaluation engine writes a normalized wealth series that starts at 1.0
    at the out-of-sample boundary
    
    Multiplying by initial capital yields a directly interpretable dollar wealth path
    """
    w = wealth.dropna().copy()
    out = w * float(initial_capital)
    out.name = "wealth_usd"
    return out


def _align_two(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Align two series to their common non-missing date intersection

    Used for relative-difference plots, ensuring both legs are evaluated on
    identical timestamps
    """
    idx = a.dropna().index.intersection(b.dropna().index)
    return a.reindex(idx), b.reindex(idx)


# ------------------------------------------------------------
# plotting helpers
# ------------------------------------------------------------

def _savefig(path: Path) -> None:
    """
    Save the active matplotlib figure with tight layout and close it

    All figures are written to the shared figure directory:
        data/results/figures/
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def _legend_right() -> None:
    """Place the legend outside the plotting area on the right-hand side"""
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        borderaxespad=0.0,
    )


# ------------------------------------------------------------
# figure generation
# ------------------------------------------------------------

def plot_wealth_usd_levels(
        wealth_map: Dict[str, pd.Series],
        strategies: List[str],
        out_name: str,
        legend_mode: str = "left",
) -> None:
    """
    Plot net wealth in USD over time for a selected strategy subset

    This is the main economic-magnitude figure for the benchmark universe
    """
    plt.figure(figsize=FIGSIZE_WIDE)

    for name in strategies:
        s = wealth_map.get(name)
        if s is None or s.dropna().empty:
            continue
        lw = 1.8 if name == "ew" else LINEWIDTH
        plt.plot(s.index, s.values / 1_000_000, label=DISPLAY_NAMES.get(name, name), linewidth=lw)               

#    plt.title("net wealth (usd) | benchmark_12", fontsize=20, pad=12)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.xlabel("date", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Wealth (USD millions)", fontsize=LABEL_FONTSIZE)
    if legend_mode == "right_outside":
        _legend_right()
    else:
        plt.legend(loc="upper left", frameon=True, fontsize=LEGEND_FONTSIZE)

    _savefig(FIG_DIR / out_name)


def plot_wealth_diff_vs_ew(
        wealth_map: Dict[str, pd.Series],
        strategies: List[str],
        out_name: str,
        legend_mode: str = "left",
) -> None:
    """
    Plot wealth difference in USD over time relative to EW for a selected strategy subset

    This figure shows whether a strategy's outperformance or underperformance
    relative to the naive benchmark is persistent or episodic
    """
    if BASELINE not in wealth_map or wealth_map[BASELINE].dropna().empty:
        return

    ew = wealth_map[BASELINE].dropna()

    plt.figure(figsize=FIGSIZE_WIDE)
    plt.axhline(0.0, linewidth=LINEWIDTH)

    for name in strategies:
        if name == BASELINE:
            continue
        s = wealth_map.get(name)
        if s is None or s.dropna().empty:
            continue

        s1, ew1 = _align_two(s, ew)
        if not s1.empty:
            plt.plot(s1.index, s1 - ew1, label=DISPLAY_NAMES.get(name, name), linewidth=LINEWIDTH)

#    plt.title("wealth difference vs ew (usd) | benchmark_12", fontsize=20, pad=12)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.xlabel("date", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Wealth difference relative to EW (USD)", fontsize=LABEL_FONTSIZE)
    if legend_mode == "right_outside":
        _legend_right()
    else:
        plt.legend(loc="upper left", frameon=True, fontsize=LEGEND_FONTSIZE)
    
    _savefig(FIG_DIR / out_name)


def plot_ceq_vs_gamma_diff(
        ceq_map: Dict[str, Dict[str, float]],
        strategies: List[str],
        out_name: str,
        legend_mode: str = "left",
    ) -> None:
    """
    Plot CEQ difference relative to EW across the risk-aversion grid
    for a selected strategy subset

    The CEQ values are read from the stats json files and are already stored
    in annual units
    """
    if BASELINE not in ceq_map:
        return

    base = ceq_map[BASELINE]
    gammas = sorted([float(k) for k in base.keys()])
    base_vec = np.array([float(base[str(int(g))]) for g in gammas])

    plt.figure(figsize=FIGSIZE_CEQ)
    plt.axhline(0.0, linewidth=1.0)

    for name in strategies:
        if name == BASELINE:
            continue
        if name not in ceq_map:
            continue

        m = ceq_map[name]
        vec = np.array([float(m[str(int(g))]) for g in gammas])

        lw = LINEWIDTH
        plt.plot(gammas, vec - base_vec, marker="o", markersize=4.5, label=DISPLAY_NAMES.get(name, name), linewidth=lw)

#    plt.title("ceq difference vs ew | benchmark_12", fontsize=20, pad=12)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.xlabel("Risk aversion parameter (γ)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("CEQ difference from EW", fontsize=LABEL_FONTSIZE)
    if legend_mode == "right_outside":
        _legend_right()
    else:
        plt.legend(loc="upper left", frameon=True, fontsize=LEGEND_FONTSIZE)
    _savefig(FIG_DIR / out_name)

def plot_effective_n(
        effn_map: Dict[str, pd.Series],
        strategies: List[str],
        out_name: str,
        legend_mode: str = "left",
) -> None:
    """
    Plot the effective number of assets over rebalance dates

    This is an appendix-style concentration diagnostic supporting the
    estimation-risk discussion
    """
    plt.figure(figsize=FIGSIZE_WIDE)

    for name in strategies:
        s = effn_map.get(name)
        if s is None or s.dropna().empty:
            continue
        lw = 1.8 if name == "ew" else LINEWIDTH
        plt.plot(s.index, s.values, label=DISPLAY_NAMES.get(name, name), linewidth=lw)

#    plt.title("effective number of assets | benchmark_12", fontsize=20, pad=12)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.xlabel("rebalance date", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Effective number of assets", fontsize=LABEL_FONTSIZE)
    if legend_mode == "right_outside":
        _legend_right()
    else:
        plt.legend(loc="upper left", frameon=True, fontsize=LEGEND_FONTSIZE)
    _savefig(FIG_DIR / out_name)


def plot_max_weight(
        maxw_map: Dict[str, pd.Series],
        strategies: List[str],
        out_name: str,
        legend_mode: str = "left",
) -> None:
    """
    Plot the maximum portfolio weight over rebalance dates

    This is a direct concentration diagnostic supporting the appendix
    discussion of portfolio instability
    """
    plt.figure(figsize=FIGSIZE_WIDE)

    for name in strategies:
        s = maxw_map.get(name)
        if s is None or s.dropna().empty:
            continue
        lw = 1.8 if name == "ew" else LINEWIDTH
        plt.plot(s.index, s.values, label=DISPLAY_NAMES.get(name, name), linewidth=lw)

#    plt.title("max weight over time | benchmark_12", fontsize=20, pad=12)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.xlabel("rebalance date", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Maximum portfolio weight", fontsize=LABEL_FONTSIZE)
    if legend_mode == "right_outside":
        _legend_right()
    else:
        plt.legend(loc="upper left", frameon=True, fontsize=LEGEND_FONTSIZE)

    _savefig(FIG_DIR / out_name)


def plot_turnover(
        turnover_map: Dict[str, pd.Series],
        strategies: List[str],
        out_name: str,
        legend_mode: str = "left",
) -> None:
    """
    Plot turnover per rebalance date

    This is an appendix-style trading-intensity diagnostic complementing
    the average-turnover summary statistic.
    """
    plt.figure(figsize=FIGSIZE_WIDE)

    for name in strategies:
        s = turnover_map.get(name)
        if s is None or s.dropna().empty:
            continue
        lw = 1.8 if name == "ew" else LINEWIDTH
        plt.plot(s.index, s.values, label=DISPLAY_NAMES.get(name, name), linewidth=lw)

#    plt.title("turnover per rebalance | benchmark_12", fontsize=20, pad=12)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.xlabel("rebalance date", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Turnover", fontsize=LABEL_FONTSIZE)
    if legend_mode == "right_outside":
        _legend_right()
    else:
        plt.legend(loc="upper left", frameon=True, fontsize=LEGEND_FONTSIZE)
        
    _savefig(FIG_DIR / out_name)


def _load_hrp_matrix_window() -> pd.DataFrame:
    """
    Load the benchmark return panel used for the appendix HRP matrix figures

    Important:
    - complete-case cleaning is applied before the final window is taken,
      matching the evaluation-layer convention that rows with any missing
      asset returns are dropped
    - the final HRP_WINDOW rows are then used as the benchmark end-of-sample
      estimation window for the appendix diagnostic figures
    """
    rets = load_returns(UNIVERSE).sort_index()
    rets_cc = rets.dropna(how="any")
    if rets_cc.shape[0] < HRP_WINDOW:
        raise ValueError(
            f"benchmark HRP figure requires at least {HRP_WINDOW} complete-case observations, "
            f"but only {rets_cc.shape[0]} are available"
        )
    return rets_cc.tail(HRP_WINDOW).copy()


def _plot_hrp_matrix_before_after(
    matrix_before: pd.DataFrame,
    matrix_after: pd.DataFrame,
    title_before: str,
    title_after: str,
    cmap: str,
    vmin: float,
    vmax: float,
    out_path: Path,
) -> None:
    """
    Plot a clean vertical before/after matrix figure for the appendix

    Design choices:
    - no figure-level title, since LaTeX will provide the caption and label
    - panel labels remain inside the image for direct visual reference
    - colorbars are kept because they are informative for appendix use
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
    })

    fig = plt.figure(figsize=HRP_FIGSIZE, facecolor="white")
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[28, 1.2],
        height_ratios=[1, 1],
        left=0.10,
        right=0.90,
        top=0.96,
        bottom=0.07,
        hspace=0.22,
        wspace=0.06,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    cax2 = fig.add_subplot(gs[1, 1])

    im1 = ax1.imshow(matrix_before.values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    im2 = ax2.imshow(matrix_after.values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax1.set_title(title_before, pad=8)
    ax2.set_title(title_after, pad=8)

    for ax, cols in [(ax1, list(matrix_before.columns)), (ax2, list(matrix_after.columns))]:
        n = len(cols)
        ax.set_xticks(range(n))
        ax.set_xticklabels(cols, rotation=90, fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels(cols, fontsize=10)
        ax.tick_params(axis="both", length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#444444")

    cb1 = fig.colorbar(im1, cax=cax1)
    cb2 = fig.colorbar(im2, cax=cax2)
    cb1.ax.tick_params(labelsize=9)
    cb2.ax.tick_params(labelsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=HRP_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_hrp_structure_figures() -> None:
    """
    Produce appendix-ready HRP covariance and correlation matrix figures

    The benchmark_12 return panel is clustered using the current thesis HRP
    implementation with single linkage. The figures compare the original and
    quasi-diagonalized matrix orderings over the final benchmark estimation window
    """
    x = _load_hrp_matrix_window()

    out = tree_clustering(x, cfg=HRPConfig(linkage_method=HRP_LINKAGE_METHOD))
    cov = out["cov"]
    corr = out["corr"]
    assets = list(corr.columns)

    sort_ix = _get_quasi_diag(out["link"])
    ordered_assets = [assets[i] for i in sort_ix]

    cov_ord = cov.loc[ordered_assets, ordered_assets]
    corr_ord = corr.loc[ordered_assets, ordered_assets]

    cov_vmin = float(min(cov.values.min(), cov_ord.values.min()))
    cov_vmax = float(max(cov.values.max(), cov_ord.values.max()))

    _plot_hrp_matrix_before_after(
        matrix_before=cov,
        matrix_after=cov_ord,
        title_before="(a) Covariance matrix before HRP reordering",
        title_after="(b) Covariance matrix after HRP quasi-diagonalization",
        cmap="viridis",
        vmin=cov_vmin,
        vmax=cov_vmax,
        out_path=FIG_DIR / "hrp_cov_before_after_bm.png",
    )

    _plot_hrp_matrix_before_after(
        matrix_before=corr,
        matrix_after=corr_ord,
        title_before="(a) Correlation matrix before HRP reordering",
        title_after="(b) Correlation matrix after HRP quasi-diagonalization",
        cmap="RdYlBu_r",
        vmin=-1.0,
        vmax=1.0,
        out_path=FIG_DIR / "hrp_corr_before_after_bm.png",
    )


# ------------------------------------------------------------
# artifact-based benchmark summary
# ------------------------------------------------------------

def build_robustness_summary() -> pd.DataFrame:
    """
    Build the compact benchmark robustness summary table from existing artifacts

    Output columns:
    - annualized mean, volatility, and Sharpe ratio
    - number of monthly out-of-sample observations
    - average turnover
    - concentration diagnostics: eff_n_p05 and max_w_p95
    - final wealth in USD
    - final wealth difference relative to EW in USD
    """
    rows = []

    wealth_map: Dict[str, pd.Series] = {}
    turnover_map: Dict[str, pd.Series] = {}
    effn_map: Dict[str, pd.Series] = {}
    maxw_map: Dict[str, pd.Series] = {}
    ceq_map: Dict[str, Dict[str, float]] = {}

    for strategy in STRATEGIES:
        if not _exists_bundle(strategy):
            continue

        stats = _read_json(_artifact_path(strategy, "stats", "json"))
        weights = _read_df_csv(_artifact_path(strategy, "weights", "csv"))
        turnover = _read_series_csv(_artifact_path(strategy, "turnover", "csv"))
        wealth = _read_series_csv(_artifact_path(strategy, "wealth_net_daily", "csv"))

        effn = _effective_n(weights)
        maxw = _max_weight(weights)
        wealth_dollars = _wealth_usd(wealth, INITIAL_CAPITAL)

        wealth_map[strategy] = wealth_dollars
        turnover_map[strategy] = turnover
        effn_map[strategy] = effn
        maxw_map[strategy] = maxw

        ceq_by_gamma = stats.get("ceq_by_gamma_net", {}) or {}
        if len(ceq_by_gamma) > 0:
            ceq_map[strategy] = {str(k): float(v) for k, v in ceq_by_gamma.items()}

        net = stats.get("stats_net", {}) or {}

        rows.append({
            "universe": UNIVERSE,
            "strategy": strategy,
            "mu_ann": float(net.get("mu_ann", np.nan)),
            "sigma_ann": float(net.get("sigma_ann", np.nan)),
            "sr_ann": float(net.get("sr_ann", np.nan)),
            "n_months_oos": int(net.get("n_oos", 0)),
            "avg_turnover": float(stats.get("avg_turnover", np.nan)),
            "eff_n_p05": float(np.nanpercentile(effn.values, 5)) if effn.dropna().shape[0] > 0 else np.nan,
            "max_w_p95": float(np.nanpercentile(maxw.values, 95)) if maxw.dropna().shape[0] > 0 else np.nan,
            "final_wealth_usd": float(wealth_dollars.dropna().iloc[-1]) if wealth_dollars.dropna().shape[0] > 0 else np.nan,
        })

    df = pd.DataFrame(rows).sort_values(["strategy"]).reset_index(drop=True)

    if not df.empty and BASELINE in set(df["strategy"]):
        ew_final = float(df.loc[df["strategy"] == BASELINE, "final_wealth_usd"].iloc[0])
        df["delta_wealth_vs_ew_usd"] = df["final_wealth_usd"] - ew_final
    else:
        df["delta_wealth_vs_ew_usd"] = np.nan

    # write only the figures that are actually intended for use
    plot_wealth_usd_levels(
        wealth_map=wealth_map,
        strategies=MEAN_FAMILY,
        out_name="wealth_usd_levels_bm_mean_family.png",
        legend_mode="right_outside",
    )

    plot_wealth_usd_levels(
        wealth_map=wealth_map,
        strategies=COVARIANCE_FAMILY,
        out_name="wealth_usd_levels_bm_cov_family.png",
        legend_mode="right_outside",
    )

    plot_wealth_diff_vs_ew(
        wealth_map=wealth_map,
        strategies=MEAN_FAMILY,
        out_name="wealth_diff_vs_ew_usd_bm_mean_family.png",
        legend_mode="right_outside",
    )

    plot_wealth_diff_vs_ew(
        wealth_map=wealth_map,
        strategies=COVARIANCE_FAMILY,
        out_name="wealth_diff_vs_ew_usd_bm_cov_family.png",
        legend_mode="right_outside",
    )

    plot_effective_n(
        effn_map=effn_map,
        strategies=MEAN_FAMILY,
        out_name="effective_n_bm_mean_family.png",
        legend_mode="right_outside",
    )

    plot_effective_n(
        effn_map=effn_map,
        strategies=COVARIANCE_FAMILY,
        out_name="effective_n_bm_cov_family.png",
        legend_mode="right_outside",
    )

    plot_max_weight(
        maxw_map=maxw_map,
        strategies=MEAN_FAMILY,
        out_name="max_weight_bm_mean_family.png",
        legend_mode="right_outside",
    )

    plot_max_weight(
        maxw_map=maxw_map,
        strategies=COVARIANCE_FAMILY,
        out_name="max_weight_bm_cov_family.png",
        legend_mode="right_outside",
    )

    plot_turnover(
        turnover_map=turnover_map,
        strategies=MEAN_FAMILY,
        out_name="turnover_bm_mean_family.png",
        legend_mode="right_outside",
    )

    plot_turnover(
        turnover_map=turnover_map,
        strategies=COVARIANCE_FAMILY,
        out_name="turnover_bm_cov_family.png",
        legend_mode="right_outside",
    )

    plot_ceq_vs_gamma_diff(
        ceq_map=ceq_map,
        strategies=MEAN_FAMILY,
        out_name="ceq_vs_gamma_diff_bm_mean_family.png",
        legend_mode="right_outside",
    )

    plot_ceq_vs_gamma_diff(
        ceq_map=ceq_map,
        strategies=COVARIANCE_FAMILY,
        out_name="ceq_vs_gamma_diff_bm_cov_family.png",
        legend_mode="right_outside",
    )

    plot_hrp_structure_figures()

    return df


# ------------------------------------------------------------
# benchmark-only inference rerun
# ------------------------------------------------------------

def _build_inference_excess_df() -> pd.DataFrame:
    """
    Rebuild the benchmark daily excess-return dataframe needed for LW inference

    This uses the stored benchmark daily net excess-return series written by
    main/run_eval.py for the benchmark and the formally tested strategy set
    """
    series_map: Dict[str, pd.Series] = {}

    # ew benchmark
    ew_path = _artifact_path(BASELINE, "portfolio_excess_net_daily", "csv")
    if not ew_path.exists():
        raise FileNotFoundError(f"missing benchmark excess-return file: {ew_path}")
    series_map[BASELINE] = _read_series_csv(ew_path).rename(BASELINE)

    # tested strategies
    for strategy in INFERENCE_TEST_STRATEGIES:
        path = _artifact_path(strategy, "portfolio_excess_net_daily", "csv")
        if not path.exists():
            raise FileNotFoundError(f"missing strategy excess-return file: {path}")
        series_map[strategy] = _read_series_csv(path).rename(strategy)

    excess_df = pd.concat(series_map.values(), axis=1).dropna()
    return excess_df

def _as_float_series(x: Any, name: str) -> pd.Series:
    """
    Coerce a returned backtest object into a named float Series
    """
    if isinstance(x, pd.Series):
        return x.astype(float).rename(name)
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"expected one-column dataframe for {name}, got shape={x.shape}")
        return x.iloc[:, 0].astype(float).rename(name)
    raise TypeError(f"cannot coerce object of type {type(x)} into a Series for {name}")


def _read_required_observed_se(df: pd.DataFrame) -> pd.Series:
    """
    read the required observed-sample standard error of the sharpe-ratio difference
    from the frozen lw output schema

    expected upstream column:
    - se_delta
    """
    col = "se_delta"
    if col not in df.columns:
        raise KeyError(
            f"expected column '{col}' in lw output, but columns were {list(df.columns)}"
        )
    return pd.to_numeric(df[col], errors="coerce")

def _strategy_ctor_map() -> Dict[str, Any]:
    """
    Shared strategy constructor map used by benchmark sensitivity reruns
    """
    return {
        "ew": lambda: EqualWeight(),
        "mv": lambda: MinVariance(),
        "msr": lambda: MaxSharpe(),
        "erc": lambda: EqualRiskContribution(),
        "hrp": lambda: HierarchicalRiskParity(),
        "nco_mv": lambda: NestedClusterOptimization(base="mv"),
        "nco_msr": lambda: NestedClusterOptimization(base="msr"),
        "dm_global_sa5": lambda: DM(DMConfig(
            name="dm_global_sa5",
            sigma_alpha_ann=0.05,
            market_path="data/processed/mkt_global_daily.csv",
        )),
    }

def run_benchmark_inference() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rerun the benchmark LW inference layer using the frozen final thesis configuration

    Returns:
    - tests_df_full: raw full hypothesis-test output from run_lw_suite_for_universe
    - summary_df: benchmark inference summary with the selected b* row per strategy
    """
    excess_df = _build_inference_excess_df()

    min_t = 500
    if excess_df.shape[0] < min_t:
        raise ValueError(
            f"benchmark inference requires at least {min_t} overlapping daily observations, "
            f"but only {excess_df.shape[0]} are available"
        )

    algo31_cfg = Algo31Config(
        b_grid=(1, 2, 4, 6, 8, 10),
        k_pseudo=5000,
        m_inner=199,
        alpha=0.05,
        show_progress=True,
        progress_min_interval_s=0.5,
        var_order=1,
        resid_sb_avg_block=5,
        seed=INFERENCE_SEED,
    )

    final_cfg = BootTSConfig(
        alpha=0.05,
        m=4999,
        show_progress=False,
        progress_min_interval_s=0.5,
        seed=BOOTTS_SEED,
    )

    tests_df, _ = run_lw_suite_for_universe(
        excess_rets_daily=excess_df,
        benchmark_col=BASELINE,
        strategies=INFERENCE_TEST_STRATEGIES,
        algo31_cfg=algo31_cfg,
        final_cfg=final_cfg,
        report_all_b=False,
        prewhiten_hac=True,
    )

    # select the row corresponding to the chosen block length for each strategy
    # use numeric coercion + isclose for robustness
    tests_df = tests_df.copy()
    tests_df["block_len"] = pd.to_numeric(tests_df["block_len"], errors="coerce")
    tests_df["block_len_selected"] = pd.to_numeric(tests_df["block_len_selected"], errors="coerce")

    mask = np.isclose(
        tests_df["block_len"].to_numpy(dtype=float),
        tests_df["block_len_selected"].to_numpy(dtype=float),
        equal_nan=False,
    )

    selected = tests_df.loc[mask].copy()
    selected["se_delta_sharpe"] = _read_required_observed_se(selected)

    summary_df = selected.loc[:, [
        "strategy",
        "delta_sharpe",
        "p_value",
        "ci_low",
        "ci_high",
        "se_delta_sharpe",
        "block_len_selected",
        "g_of_b",
        "invalid_frac",
    ]].copy()

    summary_df = summary_df.sort_values("strategy").reset_index(drop=True)
    return tests_df, summary_df


# ------------------------------------------------------------
# window-length sensitivity
# ------------------------------------------------------------

def run_window_sensitivity() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rerun the benchmark backtest across multiple estimation windows and
    run benchmark-only LW inference for each window

    Returns:
    - sens_df:
        point-estimate sensitivity table merged with selected LW inference results
    - tests_full_df:
        raw full LW output across all windows
    - inf_summary_df:
        selected b* row per strategy-window pair
    """
    rets = load_returns(UNIVERSE)
    rf_daily = load_rf_daily()
    cfg = EvalConfig(tc_cost=TC_COST)
    ctor = _strategy_ctor_map()

    total_jobs = len(WINDOWS) * len(STRATEGIES)
    done = 0
    t0 = time.time()

    point_rows: List[Dict[str, Any]] = []
    tests_full_all: List[pd.DataFrame] = []
    summary_all: List[pd.DataFrame] = []

    for w in WINDOWS:
        _status(f"start sensitivity window={w}")

        # ------------------------------------------------------------
        # 1. rerun all benchmark strategies for this window once
        # ------------------------------------------------------------
        series_map: Dict[str, pd.Series] = {}

        for strategy in STRATEGIES:
            strat = ctor[strategy]()
            res = run_backtest(
                returns=rets,
                strategy=strat,
                window=w,
                rebalance=REBALANCE,
                ann_factor=252,
                rf_daily=rf_daily,
                cfg=cfg,
            )

            net = res.get("stats_net", {}) or {}

            point_rows.append({
                "universe": UNIVERSE,
                "strategy": strategy,
                "window": int(w),
                "rebalance": int(REBALANCE),
                "tc_cost": float(TC_COST),
                "mu_ann": float(net.get("mu_ann", np.nan)),
                "sigma_ann": float(net.get("sigma_ann", np.nan)),
                "sr_ann": float(net.get("sr_ann", np.nan)),
                "n_months_oos": int(net.get("n_oos", 0)),
                "avg_turnover": float(res.get("avg_turnover", np.nan)),
            })

            # collect daily net excess returns needed for inference
            if strategy == BASELINE or strategy in INFERENCE_TEST_STRATEGIES:
                s = res.get("portfolio_excess_net_daily", None)
                if s is None:
                    raise KeyError(
                        f"run_backtest did not return 'portfolio_excess_net_daily' "
                        f"for strategy={strategy}, window={w}"
                    )
                series_map[strategy] = _as_float_series(s, strategy)

            done += 1
            _status(f"sensitivity backtest {done}/{total_jobs} | {strategy} | w={w} | elapsed={_elapsed_str(t0)}")

        # ------------------------------------------------------------
        # 2. run LW inference for this window on the rerun daily series
        # ------------------------------------------------------------
        excess_df = pd.concat(series_map.values(), axis=1).dropna()

        min_t = 500
        if excess_df.shape[0] < min_t:
            raise ValueError(
                f"window={w}: benchmark inference requires at least {min_t} overlapping daily observations, "
                f"but only {excess_df.shape[0]} are available"
            )

        algo31_cfg = Algo31Config(
            b_grid=(1, 2, 4, 6, 8, 10),
            k_pseudo=5000,
            m_inner=199,
            alpha=0.05,
            show_progress=True,
            progress_min_interval_s=0.5,
            var_order=1,
            resid_sb_avg_block=5,
            seed=INFERENCE_SEED,
        )

        final_cfg = BootTSConfig(
            alpha=0.05,
            m=4999,
            show_progress=False,
            progress_min_interval_s=0.5,
            seed=BOOTTS_SEED,
        )

        tests_df, _ = run_lw_suite_for_universe(
            excess_rets_daily=excess_df,
            benchmark_col=BASELINE,
            strategies=INFERENCE_TEST_STRATEGIES,
            algo31_cfg=algo31_cfg,
            final_cfg=final_cfg,
            report_all_b=False,
            prewhiten_hac=True,
        )

        tests_df = tests_df.copy()
        tests_df["window"] = int(w)

        tests_df["block_len"] = pd.to_numeric(tests_df["block_len"], errors="coerce")
        tests_df["block_len_selected"] = pd.to_numeric(tests_df["block_len_selected"], errors="coerce")

        mask = np.isclose(
            tests_df["block_len"].to_numpy(dtype=float),
            tests_df["block_len_selected"].to_numpy(dtype=float),
            equal_nan=False,
        )

        # read the required observed-sample SE from the frozen LW output schema
        # fail loudly if the expected column is missing
        selected = tests_df.loc[mask].copy()
        selected["se_delta_sharpe"] = _read_required_observed_se(selected)

        summary_df = selected.loc[:, [
            "window",
            "strategy",
            "delta_sharpe",
            "p_value",
            "ci_low",
            "ci_high",
            "se_delta_sharpe",
            "block_len_selected",
            "g_of_b",
            "invalid_frac",
        ]].copy()

        tests_full_all.append(tests_df)
        summary_all.append(summary_df)

        _status(f"completed LW inference for window={w} | elapsed={_elapsed_str(t0)}")

    sens_df = pd.DataFrame(point_rows).sort_values(["strategy", "window"]).reset_index(drop=True)

    tests_full_df = pd.concat(tests_full_all, axis=0, ignore_index=True) if tests_full_all else pd.DataFrame()
    inf_summary_df = pd.concat(summary_all, axis=0, ignore_index=True) if summary_all else pd.DataFrame()

    # merge the selected LW results into the main sensitivity table
    sens_df = sens_df.merge(
        inf_summary_df,
        on=["window", "strategy"],
        how="left",
    )

    sens_df = sens_df.sort_values(["strategy", "window"]).reset_index(drop=True)
    tests_full_df = tests_full_df.sort_values(["window", "strategy"]).reset_index(drop=True)
    inf_summary_df = inf_summary_df.sort_values(["window", "strategy"]).reset_index(drop=True)

    return sens_df, tests_full_df, inf_summary_df

# ------------------------------------------------------------
# main execution
# ------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full benchmark secondary evaluation.

    Steps:
    1. build the compact robustness summary from existing artifacts
    2. rerun the benchmark-only LW inference layer with the frozen final thesis configuration
    3. rerun benchmark window-length sensitivity with benchmark-only LW inference across the full window grid
    """
    t0 = time.time()
    _status(f"start | universe={UNIVERSE} | strategies={len(STRATEGIES)}")

    _status("building robustness summary + figures")
    summary = build_robustness_summary()
    summary_path = OUT_DIR / "robustness_summary_bm.csv"
    summary.to_csv(summary_path, index=False)
    _status(f"wrote {summary_path.name} | rows={len(summary)}")

    _status("rerunning benchmark-only inference with frozen thesis configuration")
    inf_t0 = time.time()
    tests_full, inf_summary = run_benchmark_inference()

    inf_full_path = OUT_DIR / "hyp_tests_bm_full.csv"
    inf_summary_path = OUT_DIR / "inference_summary_bm.csv"

    tests_full.to_csv(inf_full_path, index=False)
    inf_summary.to_csv(inf_summary_path, index=False)
    _status(f"wrote {inf_full_path.name} | rows={len(tests_full)}")
    _status(f"wrote {inf_summary_path.name} | rows={len(inf_summary)} | elapsed={_elapsed_str(inf_t0)}")

    _status(f"start sensitivity + window-grid inference | windows={WINDOWS} | rebalance={REBALANCE} | tc_cost={TC_COST:.4f}")
    sens_t0 = time.time()
    sens, win_tests_full, win_inf_summary = run_window_sensitivity()

    sens_path = OUT_DIR / "window_length_sensitivity_bm.csv"
    win_inf_summary_path = OUT_DIR / "window_inference_summary_bm.csv"
    win_tests_full_path = OUT_DIR / "window_hyp_tests_bm_full.csv"

    sens.to_csv(sens_path, index=False)
    win_inf_summary.to_csv(win_inf_summary_path, index=False)
    win_tests_full.to_csv(win_tests_full_path, index=False)

    _status(f"wrote {sens_path.name} | rows={len(sens)}")
    _status(f"wrote {win_inf_summary_path.name} | rows={len(win_inf_summary)}")
    _status(f"wrote {win_tests_full_path.name} | rows={len(win_tests_full)} | elapsed={_elapsed_str(sens_t0)}")

    _status(f"complete | total_elapsed={_elapsed_str(t0)}")
    print(f"- robustness table: {summary_path}")
    print(f"- inference table: {inf_summary_path}")
    print(f"- inference full output: {inf_full_path}")
    print(f"- window inference summary: {win_inf_summary_path}")
    print(f"- window inference full output: {win_tests_full_path}")
    print(f"- window sensitivity table: {sens_path}")
    print(f"- figures directory: {FIG_DIR}")


if __name__ == "__main__":
    main()