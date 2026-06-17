# strategies/base.py
#
# purpose:
# - define a common interface for all portfolio strategies used in the thesis
# - enforce a consistent output format (weights indexed by asset names)
# - provide small utilities that strategy implementations can reuse
#
# design:
# - evaluation engine calls strategy.get_weights(window_returns) at each rebalance date
# - strategies return StrategyResult(weights=...), where weights is a pandas Series
# - this file does not implement any strategy logic itself (only structure + helpers)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional  # rf window is optional for strategies that need it
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# standardized strategy output container
# ------------------------------------------------------------

@dataclass
class StrategyResult:
    """
    portfolio weights for the next holding interval
    
    requirements:
     - type: pd.Series
     - index: asset identifiers (must match return matrix columns)
     - values: numeric weights (float), typically expected to sum to 1
    
    note:
     - the evaluation engine will reindex and normalize weights again,
       but strategies should still aim to return a clean, sensible vector
    """
    weights: pd.Series


# ------------------------------------------------------------
# base strategy interface
# ------------------------------------------------------------

class BaseStrategy:
    # short name used in output folders and logs.
    # subclasses should override this, e.g. name = "ew", "mv", "hrp"
    name: str = "base"

    def get_weights(
        self,
        window_returns: pd.DataFrame,
        window_rf: Optional[pd.Series] = None,  # needed for mean-variance / msr style strategies
    ) -> StrategyResult:
        """
        compute portfolio weights based on a rolling estimation window of returns
        
        inputs:
         - window_returns: dataframe shaped (t x n)
           * t = estimation window length (e.g. 756 daily observations)
           * n = number of assets in the universe
           * columns = asset identifiers (tickers)
           * values = daily simple returns

         - window_rf: optional series shaped (t,)
           * daily simple risk-free returns aligned to window_returns.index
           * only needed for strategies that require excess return estimation (e.g. msr)

        output:
         - StrategyResult(weights=...), where weights is a pd.Series indexed by asset names
        
        important:
         - this is the only method the evaluation engine needs from a strategy
         - the base class does not implement this; subclasses must override
        """
        raise NotImplementedError("strategy must implement get_weights()")


# ------------------------------------------------------------
# helper utilities (optional for strategies)
# ------------------------------------------------------------

def _normalize_long_only(w: pd.Series) -> pd.Series:
    """
    enforce long-only, fully-invested normalization on a weight vector
    
    behavior:
     1) copy the weights to avoid mutating the input
     2) clip negative weights to 0 (long-only constraint)
     3) renormalize so that sum(w) = 1
     4) if sum(w) is zero or non-positive after clipping, fall back to equal weights
    
    inputs:
     - w: pd.Series of raw weights (may include negatives, may not sum to 1)
    
    outputs:
     - pd.Series of cleaned weights, non-negative and summing to 1
    """
    w = w.copy()

    # remove short positions
    w[w < 0.0] = 0.0

    # renormalize to sum to one
    s = float(w.sum())

    # if everything got clipped to zero (or input was degenerate), fall back to 1/n
    if s <= 0.0:
        w[:] = 1.0 / len(w)
        return w

    return w / s


def _ensure_weights_series(w, columns) -> pd.Series:
    """
    coerce a weight vector into a pandas Series aligned to a specific column index
    
    why this exists:
     - some solvers return numpy arrays
     - sometimes a strategy constructs weights as a Series with missing assets
     - the evaluation engine expects a vector aligned to returns.columns
    
    behavior:
     - if w is already a pd.Series:
       * copy it (avoid mutating upstream)
       * reindex to the given columns (ensures the same order and includes all assets)
     - otherwise:
       * convert w to a numpy array and wrap it in a pd.Series with index=columns
    
    note:
     - reindexing a Series can introduce NaNs if w lacks some assets,
       strategies should handle that explicitly if they can output incomplete weights
    """
    if isinstance(w, pd.Series):
        out = w.copy()
        out = out.reindex(columns)
        return out

    return pd.Series(np.asarray(w, dtype=float), index=columns)
