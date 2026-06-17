# Multi-Asset Portfolio Optimization

This repository contains the Python code used for the master's thesis:

**Multi-Asset Portfolio Optimization: Evaluating Hierarchical and Cluster-Based Portfolio Optimizers**

The project compares hierarchical, cluster-based, and classical portfolio optimization strategies in long-only ETF portfolios. The implemented strategies include equal weighting, maximum Sharpe ratio optimization, minimum-variance optimization, equal risk contribution, a Bayesian Data-and-Model CAPM specification, hierarchical risk parity, and nested clustered optimization.

The thesis PDF is available as a downloadable asset under the repository's **Releases** section.

## Data availability

The original thesis analysis used market data sourced through the Refinitiv Eikon Data Library API.

This public repository does **not** include raw Refinitiv/Eikon data, processed datasets, generated result files, figures, or secondary evaluation outputs because the underlying data may be subject to license restrictions.

Users with valid Refinitiv/Eikon access can rebuild the required local data files using `data_pipeline.py`.

## Repository structure

```text
multi-asset-portfolio-optimization/
  main/
    run_eval.py
    sec_eval.py
  evaluation/
    __init__.py
    eval_engine.py
  inference/
    __init__.py
    lw_bootstrap.py
  strategies/
    __init__.py
    base.py
    dm.py
    erc.py
    ew.py
    hrp.py
    msr.py
    msr_is.py
    mv.py
    nco.py
  cov_denoising.py
  data_pipeline.py
  README.md
  requirements.txt
  LICENSE
```

The repository includes code only. Data folders and result folders are created locally when the scripts are run.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Required Python packages include:

```text
numpy
pandas
scipy
scikit-learn
cvxpy
matplotlib
eikon
```

## Refinitiv/Eikon API key

To rebuild the data, a valid Refinitiv/Eikon license and API key are required.

Set the API key in the environment before running the data pipeline:

```bash
export EIKON_APP_KEY="your_api_key_here"
```

Do not commit API keys, `.env` files, credentials, raw data, processed data, or generated output files to this repository.

## Replication order

The full replication workflow is:

```bash
python data_pipeline.py
python -m main.run_eval
python -m main.sec_eval
```

These commands should be run from the repository root.

## Step 1: Rebuild the data

Run:

```bash
python data_pipeline.py
```

This downloads adjusted close price data from Refinitiv/Eikon, computes daily simple returns, builds the risk-free return series and market proxy series, and writes the required local data files.

The script creates:

```text
data/raw/
data/processed/
```

The main processed files required by the evaluation scripts include:

```text
data/processed/rets_benchmark_12.csv
data/processed/rets_us_sectors_11.csv
data/processed/rets_intl_countries_15.csv
data/processed/rets_intl_countries_20.csv
data/processed/rets_intl_countries_30.csv
data/processed/rf_1m_daily.csv
data/processed/mkt_us_daily.csv
data/processed/mkt_global_daily.csv
```

## Step 2: Run the main evaluation

After rebuilding the data, run:

```bash
python -m main.run_eval
```

This runs the rolling-window out-of-sample backtests across the ETF universes and strategy set. It writes strategy-level artifacts under:

```text
data/results/<strategy>/
```

The outputs include portfolio weights, drifted weights, turnover, daily and monthly returns, daily and monthly excess returns, wealth series, strategy summary statistics, CEQ values, return-loss values, and Ledoit-Wolf bootstrap Sharpe-ratio inference outputs.

The main evaluation uses the baseline thesis configuration:

```text
rolling estimation window = 756 trading days
rebalance frequency = 21 trading days
transaction cost = 0.005 per unit turnover
```

## Step 3: Run the secondary evaluation

After `main.run_eval` has completed, run:

```bash
python -m main.sec_eval
```

This script consumes artifacts produced by `main.run_eval`. It generates benchmark-universe robustness tables, secondary inference outputs, window-length sensitivity results, and selected figures used for the thesis discussion and appendix.

It writes outputs under:

```text
data/secondary_eval/
data/results/figures/
```

`main.run_eval` must be run before `main.sec_eval`, because `main/sec_eval.py` reads artifacts written by `main/run_eval.py`.

## Runtime

The full program is computationally expensive. Under the baseline thesis calibration, the main evaluation can take approximately 24 hours.

Most of the runtime is driven by the Ledoit-Wolf inference layer, especially block-length calibration and bootstrap repetition counts.

The baseline inference settings used in the thesis are:

```text
k_pseudo = 5000
m_inner = 199
m = 4999
```

For a faster code check, these settings can be reduced inside the inference configuration blocks in `main/run_eval.py` and `main/sec_eval.py`, for example:

```text
k_pseudo = 100
m_inner = 49
m = 199
```

Reduced settings shorten runtime but will not reproduce the final thesis inference results.

## Generated files

The following folders are generated locally and are intentionally excluded from the public repository:

```text
data/raw/
data/processed/
data/results/
data/secondary_eval/
```

These folders may contain proprietary or derived market data and should not be committed to a public repository.

## License

The code in this repository is released under the MIT License.

The MIT License applies only to the code in this repository. It does not grant rights to redistribute proprietary market data from Refinitiv/Eikon or any other third-party data provider.
