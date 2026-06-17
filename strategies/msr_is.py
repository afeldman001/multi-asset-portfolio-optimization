# strategies/msr_is.py
#
# objective:
# - provide an in-sample variant of the maximum-sharpe (msr) strategy 
# - reflects the in-sample upper bound under the same constraints and estimators as MSR, 
#   computed using the full sample (no rolling-window estimation error)
#
# design:
# - this class intentionally reuses the exact same estimation + optimization logic as MaxSharpe
# - the only change is the strategy name used for output labeling and folder naming
#
# important:
# - "in-sample" behavior is not implemented here
# - in-sample vs out-of-sample is controlled by the evaluation layer (eval_engine/run_eval),
#   which decides whether to:
#     (a) compute one fixed weight vector on the full sample (in-sample), or
#     (b) roll a window and rebalance repeatedly (out-of-sample)
#
# why:
# - avoids duplicating optimization code
# - ensures msr and msrIS differ by evaluation protocol, not by estimator settings

from __future__ import annotations

from strategies.msr import MaxSharpe


class MaxSharpeInSample(MaxSharpe):
    # short name used in output folders, logs, and result summaries
    # keeps artifacts distinct from the out-of-sample msr strategy
    name = "msrIS"

    # evaluation flag:
    # - run_eval.py uses this attribute to route msrIS through the in-sample engine path
    # - if missing or false, the strategy is treated as out-of-sample by default
    is_insample = True
