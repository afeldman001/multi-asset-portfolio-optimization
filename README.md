# Multi-Asset Portfolio Optimization

This repository contains the Python code used for the master's thesis:

**Multi-Asset Portfolio Optimization: Evaluating Hierarchical and Cluster-Based Portfolio Optimizers**

The project compares hierarchical, cluster-based, and classical portfolio optimization strategies in long-only ETF portfolios. The implemented strategies include equal weighting, maximum Sharpe ratio optimization, minimum-variance optimization, equal risk contribution, a Bayesian Data-and-Model CAPM specification, hierarchical risk parity, and nested clustered optimization.

## Data availability

The original thesis analysis used market data sourced through the Refinitiv Eikon Data Library API.

The public repository does **not** include raw Refinitiv/Eikon data, processed datasets, or generated result files because the underlying data may be subject to license restrictions.

Users with valid Refinitiv/Eikon access can rebuild the data locally using `data_pipeline.py`.

## Repository structure

```text
multi-asset-portfolio-optimization/
  main/
    run_eval.py
    sec_eval.py
  evaluation/
    eval_engine.py
  inference/
    lw_bootstrap.py
  strategies/
    base.py
    dm.py
    erc.py
    ew.py
    hrp.py
    msr.py
    msr_is.py
    mv.py
    nco.py
  validation_tests/
  data_pipeline.py
  cov_denoising.py
  data/
    raw/
    processed/
    results/
  README.md
  requirements.txt
```

The `data/` folders are included only as placeholders. The public repository does not include proprietary data files or generated output files.

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

Do not commit API keys, `.env` files, or credentials to this repository.

## Rebuilding the data

Run:

```bash
python data_pipeline.py
```

This rebuilds the local raw and processed data files required by the evaluation scripts, provided that the user has valid Refinitiv/Eikon access.

## Main evaluation

After rebuilding the data, run the main evaluation with:

```bash
python -m main.run_eval
```

## Secondary evaluation

The secondary evaluation is run with:

```bash
python -m main.sec_eval
```

`main.run_eval` should be run before `main.sec_eval`, since `main/sec_eval.py` reads artifacts written by `main/run_eval.py`.

## Runtime

The full program is computationally expensive. Under the baseline thesis calibration, the main evaluation can take approximately 24 hours.

Most of the runtime is driven by the Ledoit-Wolf inference layer, especially block-length calibration and bootstrap repetition counts.

The baseline inference settings used in the thesis are:

```text
k_pseudo = 5000
m_inner = 199
m = 4999
```

For a faster code check, these settings can be reduced, for example:

```text
k_pseudo = 100
m_inner = 49
m = 199
```

Reduced settings shorten runtime but will not reproduce the final thesis inference results.

## License

The code in this repository is released under the MIT License.

The MIT License applies only to the code in this repository. It does not grant rights to redistribute proprietary market data from Refinitiv/Eikon or any other third-party data provider.
