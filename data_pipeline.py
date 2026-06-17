# data_pipeline.py

import os
import time
import numpy as np
import pandas as pd
import eikon as ek


# ------------------------------------------------------------
# refinitiv session
# ------------------------------------------------------------

# api key is set in the environment (export EIKON_APP_KEY="...")
ek.set_app_key(os.getenv("EIKON_APP_KEY"))


# ------------------------------------------------------------
# configuration
# ------------------------------------------------------------

# use close prices only for now
# the call uses corax="adjusted", so this is adjusted close
FIELDS = ["CLOSE"]

# benchmark portfolio (n=12)
BENCHMARK_TICKERS = ["VTI", "VEA", "VWO", "IEF.O", "TLT.O", "BNDX.O", "LQD", "HYG", "VNQ", "TIP", "GLD", "DBC"]

# secondary portfolios

# us sector sleeves (n=11)
# xlc/xlre are avoided because they truncate the sample (late inception)
# iyz and vnq are used instead to preserve 2013–2024 coverage
US_SECTORS = [
    "XLB", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLU", "XLV", "XLY", "IYZ", "VNQ"
]

# international countries (n=15)
INTL_COUNTRIES_15 = [
    "EWA",  # australia
    "EWC",  # canada
    "EWG",  # germany
    "EWQ",  # france
    "EWI",  # italy
    "EWU",  # uk
    "EWJ",  # japan
    "EWY",  # south korea
    "EWT",  # taiwan
    "EWH",  # hong kong
    "EWS",  # singapore
    "EWZ",  # brazil
    "EWW",  # mexico
    "EZA",  # south africa
    "INDA.K"  # india (ric resolved explicitly to avoid remapping)
]

# international countries (n=20)
INTL_COUNTRIES_20 = [
    "EWA",    # australia
    "EWC",    # canada
    "EWG",    # germany
    "EWQ",    # france
    "EWI",    # italy
    "EWP",    # spain
    "EWL",    # switzerland
    "EWU",    # uk
    "EWJ",    # japan
    "EWY",    # south korea
    "EWT",    # taiwan
    "EWH",    # hong kong
    "EWS",    # singapore
    "EWZ",    # brazil
    "EWW",    # mexico
    "EZA",    # south africa
    "TUR.O",  # turkey
    "INDA.K", # india
    "EIDO.K", # indonesia
    "THD"     # thailand
]

# international countries (n=30)
INTL_COUNTRIES_30 = [
    "EWA",     # australia
    "EWC",     # canada
    "EWG",     # germany
    "EWQ",     # france
    "EWI",     # italy
    "EWP",     # spain
    "EWL",     # switzerland
    "EWU",     # uk
    "EWN",     # netherlands
    "EWD",     # sweden
    "EWO",     # austria
    "EWK",     # belgium
    "EIRL.K",  # ireland
    "EIS",     # israel
    "EWJ",     # japan
    "EWY",     # south korea
    "EWT",     # taiwan
    "EWH",     # hong kong
    "EWS",     # singapore
    "EWM",     # malaysia
    "EPU",     # peru
    "EPOL.K",  # poland
    "TUR.O",   # turkey
    "EZA",     # south africa
    "EWZ",     # brazil
    "EWW",     # mexico
    "INDA.K",  # india
    "EIDO.K",  # indonesia
    "THD",     # thailand
    "ECH"      # chile
]

# dm market proxies
# - us proxy is voo 
# - global proxy is single broad global equity etf
GLOBAL_MKT_CANDIDATES = [
    "ACWI.O",  # fixed: confirmed to exist in workspace
]

US_MKT_CANDIDATES = [
    "VOO",     # fixed: confirmed to exist in workspace
]

# sample start and end match the benchmark aligned sample
START_DATE = "2013-06-05"
END_DATE = "2024-12-31"

# retry behavior
MAX_RETRIES = 6
RETRY_SLEEP_SEC = 1.5

# fallback mapping for instruments that sometimes fail to resolve
# keys are labels useed in portfolios
# values are candidate identifiers to try in order
FALLBACKS = {
    # common issues: .O variants may fail depending on workspace mapping
    "IEF.O": ["IEF.O", "IEF"],
    "TLT.O": ["TLT.O", "TLT"],
    "BNDX.O": ["BNDX.O", "BNDX"],

    # turkish etf identifier variability
    "TUR.O": ["TUR.O", "TUR"],

    # keep explicit .K mapping first, but allow fallback if needed
    "INDA.K": ["INDA.K", "INDA"],
    "EIDO.K": ["EIDO.K", "EIDO"],
    "EIRL.K": ["EIRL.K", "EIRL"],
    "EPOL.K": ["EPOL.K", "EPOL"],
}


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def dedupe(tickers):
    # de-duplicate while preserving order
    # protects from accidental duplicates when constructing lists
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


US_SECTORS = dedupe(US_SECTORS)
INTL_COUNTRIES_15 = dedupe(INTL_COUNTRIES_15)
INTL_COUNTRIES_20 = dedupe(INTL_COUNTRIES_20)
INTL_COUNTRIES_30 = dedupe(INTL_COUNTRIES_30)


def _get_candidates(ticker):
    # always try the exact ticker first, then any configured fallbacks
    if ticker in FALLBACKS:
        return FALLBACKS[ticker]
    return [ticker]


def _fetch_timeseries_single(ticker, fields, start_date, end_date):
    """
    fetch a single series with retries and return (df, used_identifier)
    returns (None, used_identifier) if all candidates fail
    """
    candidates = _get_candidates(ticker)

    last_err = None

    for cand in candidates:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = ek.get_timeseries(
                    cand,
                    fields=fields,
                    calendar="tradingdays",
                    start_date=start_date,
                    end_date=end_date,
                    interval="daily",
                    corax="adjusted"
                )

                if df is not None and not df.empty:
                    return df, cand

                # treat none/empty as a failure worth retrying (transient issues happen)
                last_err = RuntimeError("returned none/empty dataframe")

            except Exception as e:
                last_err = e

            # sleep between retries
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SEC)

        # candidate failed after retries, move to next candidate
        continue

    return None, str(last_err) if last_err is not None else "unknown error"


def fetch_refinitiv_panel(tickers, start_date, end_date, fields):
    """
    fetch a daily adjusted close panel from refinitiv

    important: tickers are fetched one-by-one
    avoids timeseries limits observed when requesting many tickers at once
    """
    dfs = []
    failures = []
    resolutions = {}

    for t in tickers:
        df, used = _fetch_timeseries_single(t, fields, start_date, end_date)

        if df is None:
            failures.append((t, used))
            continue

        # record which identifier actually worked
        resolutions[t] = used

        # when only one field is requested, rename "CLOSE" to the ticker symbol
        # this ensures unique column names when concatenating
        if len(fields) == 1:
            if fields[0] not in df.columns:
                failures.append((t, f"missing expected field column {fields[0]} after fetch via {used}"))
                continue
            df = df.rename(columns={fields[0]: t})
        else:
            df.columns = pd.MultiIndex.from_product([[t], df.columns])

        dfs.append(df)

    if failures:
        msg_lines = ["refinitiv fetch failures:"]
        msg_lines += [f"- {t}: {reason}" for t, reason in failures]
        raise RuntimeError("\n".join(msg_lines))

    panel = pd.concat(dfs, axis=1).sort_index()

    # print resolution mapping for transparency
    changed = {k: v for k, v in resolutions.items() if k != v}
    if changed:
        print("\nidentifier resolution overrides (label -> fetched_as):")
        for k, v in changed.items():
            print(f"- {k} -> {v}")

    return panel


def fetch_risk_free_1m_yield(start_date, end_date):
    """
    fetch an annualized 1-month u.s. treasury yield series

    output: dataframe with one column named "rf_yield_ann"
    """
    candidates = [
        "US1MT=RR",
        "US1M=RR",
        "US1MTX=RR",
        "US1MTR=RR",
    ]

    last_err = None

    for ric in candidates:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = ek.get_timeseries(
                    ric,
                    fields=["CLOSE"],
                    calendar="tradingdays",
                    start_date=start_date,
                    end_date=end_date,
                    interval="daily",
                    corax="adjusted"
                )

                if df is not None and not df.empty:
                    df = df.rename(columns={"CLOSE": "rf_yield_ann"})
                    print(f"\nrisk-free series loaded from ric: {ric}")
                    return df

                last_err = RuntimeError("returned none/empty dataframe")

            except Exception as e:
                last_err = e

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SEC)

    raise RuntimeError(
        "failed to fetch 1m risk-free yield from all candidate rics. "
        "find the correct 1m u.s. treasury yield ric in the workspace and add it to candidates."
    ) from last_err


def yield_to_daily_rf_return(yield_ann_series, ann_factor=252):
    """
    convert annualized yield levels (in percent points) to daily risk-free returns

    output: series named "rf_daily"
    """
    y = yield_ann_series.copy().astype(float)

    # refinitiv 1m yield is in percent points (e.g. 5.25 means 5.25%)
    y = y / 100.0

    rf_daily = (1.0 + y).pow(1.0 / ann_factor) - 1.0
    rf_daily.name = "rf_daily"

    return rf_daily


def run_portfolio(name, tickers):
    """
    fetch adjusted close prices, compute arithmetic returns, and print coverage diagnostics
    """
    print("\n" + "=" * 80)
    print(f"portfolio: {name}  |  n_assets = {len(tickers)}")
    print("=" * 80)

    prices = fetch_refinitiv_panel(tickers, START_DATE, END_DATE, FIELDS)

    # compute arithmetic simple returns: r_t = P_t / P_{t-1} - 1
    rets = prices.pct_change().dropna()

    print("prices rows:", len(prices), "start:", prices.index.min(), "end:", prices.index.max())
    print("aligned returns rows:", len(rets), "start:", rets.index.min(), "end:", rets.index.max())

    coverage = prices.count().sort_values()
    print("\nper-ticker non-missing counts (ascending):")
    print(coverage)

    print("\nlimiting ticker:", coverage.index[0], "count:", int(coverage.iloc[0]))

    return prices, rets


def build_market_series_from_candidates(candidates, start_date, end_date):
    # try candidate tickers until one returns a valid daily return series
    last_err = None
    for c in candidates:
        df, used = _fetch_timeseries_single(c, ["CLOSE"], start_date, end_date)
        if df is None:
            last_err = used
            continue

        if "CLOSE" not in df.columns:
            last_err = f"missing CLOSE column for {used}"
            continue

        px = df.rename(columns={"CLOSE": "price"})
        rets = px["price"].pct_change().dropna().to_frame(name="mkt_daily")

        if rets.empty:
            last_err = f"empty return series for {used}"
            continue

        return rets, used

    raise RuntimeError(f"failed to build market series from candidates={candidates}. last_err={last_err}")


# ------------------------------------------------------------
# run all portfolios
# ------------------------------------------------------------

prices_benchmark, rets_benchmark = run_portfolio("benchmark_12", BENCHMARK_TICKERS)
prices_sectors, rets_sectors = run_portfolio("us_sectors_11", US_SECTORS)
prices_intl15, rets_intl15 = run_portfolio("intl_countries_15", INTL_COUNTRIES_15)
prices_intl20, rets_intl20 = run_portfolio("intl_countries_20", INTL_COUNTRIES_20)
prices_intl30, rets_intl30 = run_portfolio("intl_countries_30", INTL_COUNTRIES_30)

# ------------------------------------------------------------
# market series for dm (factor robustness)
# ------------------------------------------------------------

# dm requires a market return series even for universes that do not contain the market proxy
# build two series:
# - mkt_us_daily: voo daily return series from the benchmark universe
# - mkt_global_daily: msci world/acwi proxy from candidates, fetched directly

mkt_us_daily, mkt_us_used = build_market_series_from_candidates(
    candidates=US_MKT_CANDIDATES,
    start_date=START_DATE,
    end_date=END_DATE,
)

if mkt_us_daily.shape[1] != 1:
    raise RuntimeError("mkt_us_daily must be a single-column dataframe")

print("mkt_us source:", mkt_us_used)

mkt_global_daily, mkt_global_used = build_market_series_from_candidates(
    candidates=GLOBAL_MKT_CANDIDATES,
    start_date=START_DATE,
    end_date=END_DATE,
)

if mkt_global_daily.shape[1] != 1:
    raise RuntimeError("mkt_global_daily must be a single-column dataframe")

print("\n" + "=" * 80)
print("market series verification")
print("=" * 80)
print("mkt_us_daily rows:", len(mkt_us_daily), "start:", mkt_us_daily.index.min(), "end:", mkt_us_daily.index.max())
print("mkt_us_daily non-missing:", int(mkt_us_daily["mkt_daily"].count()))
print("mkt_global source:", mkt_global_used)
print("mkt_global_daily rows:", len(mkt_global_daily), "start:", mkt_global_daily.index.min(), "end:", mkt_global_daily.index.max())
print("mkt_global_daily non-missing:", int(mkt_global_daily["mkt_daily"].count()))

# ------------------------------------------------------------
# fetch and build risk-free series
# ------------------------------------------------------------

rf_yield_df = fetch_risk_free_1m_yield(START_DATE, END_DATE)

rf_yield_aligned = rf_yield_df.reindex(prices_benchmark.index).ffill()

rf_daily = yield_to_daily_rf_return(rf_yield_aligned["rf_yield_ann"], ann_factor=252)

rf_daily_aligned = rf_daily.reindex(rets_benchmark.index).ffill()

print("\n" + "=" * 80)
print("risk-free series verification")
print("=" * 80)
print("rf_yield rows:", len(rf_yield_aligned), "start:", rf_yield_aligned.index.min(), "end:", rf_yield_aligned.index.max())
print("rf_daily rows:", len(rf_daily_aligned), "start:", rf_daily_aligned.index.min(), "end:", rf_daily_aligned.index.max())
print("rf_yield non-missing:", int(rf_yield_aligned["rf_yield_ann"].count()))
print("rf_daily non-missing:", int(rf_daily_aligned.count()))

# ------------------------------------------------------------
# write outputs (csv)
# ------------------------------------------------------------

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

prices_benchmark.to_csv("data/raw/prices_benchmark_12.csv")
prices_sectors.to_csv("data/raw/prices_us_sectors_11.csv")
prices_intl15.to_csv("data/raw/prices_intl_countries_15.csv")
prices_intl20.to_csv("data/raw/prices_intl_countries_20.csv")
prices_intl30.to_csv("data/raw/prices_intl_countries_30.csv")

rets_benchmark.to_csv("data/processed/rets_benchmark_12.csv")
rets_sectors.to_csv("data/processed/rets_us_sectors_11.csv")
rets_intl15.to_csv("data/processed/rets_intl_countries_15.csv")
rets_intl20.to_csv("data/processed/rets_intl_countries_20.csv")
rets_intl30.to_csv("data/processed/rets_intl_countries_30.csv")

rf_yield_aligned.to_csv("data/processed/rf_1m_yield_daily.csv")
rf_daily_aligned.to_csv("data/processed/rf_1m_daily.csv", header=True)

# market series outputs (single-column, required by dm loader)
mkt_us_daily.to_csv("data/processed/mkt_us_daily.csv")
mkt_global_daily.to_csv("data/processed/mkt_global_daily.csv")
