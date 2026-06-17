# strategies/ew.py
#
# equal-weight (ew) strategy, naive benchmark
#
# role in thesis:
#   This is the naive 1/N benchmark portfolio used throughout the empirical analysis
#   All other strategies are evaluated relative to this benchmark, following
#   DeMiguel, Garlappi, and Uppal (2009)
#
# economic interpretation:
#   Capital is allocated equally across all available assets at each rebalance date,
#   regardless of return expectations, risk, or correlation structure
#
# properties:
#   - long-only
#   - fully invested
#   - no estimation error (no moments are estimated)
#   - extremely stable weights
#   - minimal turnover relative to optimized strategies
#
# This strategy serves as the baseline against which improvements from
# MV, MSR, ERC, BL, HRP, and NCO are assessed.

from typing import Optional  # interface consistency
import pandas as pd

from strategies.base import BaseStrategy, StrategyResult


class EqualWeight(BaseStrategy):
    # short name used in output folders, logs, and result summaries
    name = "ew"

    def get_weights(self, window_returns: pd.DataFrame, window_rf: Optional[pd.Series] = None) -> StrategyResult:
        """
        window_returns:
         - dataframe of simple returns with shape (t x n)
         - t is the rolling estimation window length
         - n is the number of assets in the universe
        
        note:
         - Ew does not use return history at all. The argument is accepted only
           for interface consistency with other strategies
         - window_rf is accepted for interface consistency, but is unused by ew
        """

        # extract asset identifiers from the return matrix
        cols = window_returns.columns
        n = len(cols)

        # assign equal weight to each asset
        # this enforces:
        #   - sum(w) = 1
        #   - w_i = 1 / n for all assets
        w = pd.Series(1.0 / n, index=cols)

        # wrap weights in a StrategyResult object expected by the evaluation engine
        return StrategyResult(weights=w)
