# strategies/mv.py
#
# minimum-variance (mv) portfolio under long-only, fully-invested constraints
#
# objective:
#   minimize    w' * Sigma * w
#   subject to  sum(w) = 1
#               w >= 0
#
# estimation:
#   Sigma is estimated using Ledoit-Wolf shrinkage on a rolling window of daily simple returns
#
# solver:
#   Quadratic program (qp) solved via cvxpy. OSQP is used by default, with SCS as a fallback

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import cvxpy as cp
from sklearn.covariance import LedoitWolf

from strategies.base import BaseStrategy, StrategyResult


class MinVariance(BaseStrategy):
    # short name used in output folders, logs, and result summaries
    name = "mv"

    def __init__(
        self,
        solver: str = "OSQP",
        solver_max_iter: int = 20_000,
        solver_eps: float = 1e-8,
        ridge_eps: float = 1e-10,
    ) -> None:
        # primary qp solver choice. OSQP is fast/stable for convex qps
        # if the primary solver fails, a single fallback attempt is made with SCS
        self.solver = str(solver)

        # solver controls, max_iter and eps settings are passed into cvxpy solver calls
        self.solver_max_iter = int(solver_max_iter)
        self.solver_eps = float(solver_eps)

        # tiny ridge added to the diagonal of Sigma for numerical stability
        # helps when Sigma is near-singular (common in financial return data),
        # makes the qp numerically more robust
        self.ridge_eps = float(ridge_eps)

    def get_weights(self, window_returns: pd.DataFrame, window_rf: Optional[pd.Series] = None) -> StrategyResult:
        """
        # window_returns:
        # - dataframe of simple returns (not log returns)
        # - shape: (t x n), where t is the rolling estimation window length
        #
        # mv uses only the covariance matrix. expected returns are not needed.

        # Enforce a strict complete-case policy for covariance estimation: drop any row with any nan.
        # Consistent with engine behavior, avoids pairwise covariance artifacts.
        """
        x = window_returns.dropna(how="any")

        # window_rf is accepted for interface consistency, but is unused by mv
        # if something upstream is broken (or a tiny sample is passed in),
        # avoid throwing and just fall back to equal weight; should never trigger in normal operation
        if x.shape[0] < 5:
            cols = window_returns.columns
            w = pd.Series(1.0 / len(cols), index=cols)
            return StrategyResult(weights=w)

        cols = x.columns
        n = len(cols)

        # ------------------------------------------------------------
        # covariance estimation (Ledoit-Wolf shrinkage)
        # ------------------------------------------------------------
        #
        # LedoitWolf().fit(X) expects X shaped (t x n), where columns are assets
        # .covariance_ returns the (n x n) covariance estimate
        #
        # used to reduce estimation error and to keep covariance estimation consistent
        # across all optimizers evaluated in the thesis

        lw = LedoitWolf().fit(x.values)
        sigma = lw.covariance_.astype(float)

        # numerical guard: enforce symmetry
        # even if the estimator is theoretically symmetric, floating point arithmetic can produce
        # tiny asymmetries that confuse solvers
        sigma = 0.5 * (sigma + sigma.T)

        # numerical guard: add a tiny ridge to stabilize the qp
        # this nudges Sigma toward positive definiteness and helps avoid numerical issues
        sigma = sigma + (self.ridge_eps * np.eye(n))

        # ------------------------------------------------------------
        # quadratic program definition
        # ------------------------------------------------------------
        #
        # decision variable: portfolio weights
        w_var = cp.Variable(n)

        # objective: minimize portfolio variance w' Sigma w
        # psd_wrap tells cvxpy to treat Sigma as psd even if tiny numerical eigenvalues go negative
        objective = cp.Minimize(cp.quad_form(w_var, cp.psd_wrap(sigma)))

        # constraints: fully invested and long-only
        constraints = [
            cp.sum(w_var) == 1.0,
            w_var >= 0.0,
        ]

        prob = cp.Problem(objective, constraints)

        # solve qp, with a primary solver and a fallback path
        w_sol = self._solve_qp(prob, w_var)

        # ------------------------------------------------------------
        # post-solve sanitation
        # ------------------------------------------------------------
        #
        # even if a solver returns a "valid" solution, numerical noise can create:
        # - tiny negative weights
        # - non-finite entries (rare, but possible on solver failure)
        # - sum(w) slightly different from 1
        #
        # sanitize aggressively to guarantee long-only and sum-to-one weights (expected by eval engine)
        # 
        w_sol = np.asarray(w_sol, dtype=float).reshape(-1)

        # replace non-finite entries with 0
        w_sol[~np.isfinite(w_sol)] = 0.0

        # clip negatives to 0 (long-only constraint)
        w_sol[w_sol < 0.0] = 0.0

        # renormalize to sum to 1, with an ew fallback if everything got zeroed
        s = float(w_sol.sum())
        if s <= 0.0 or not np.isfinite(s):
            w_sol[:] = 1.0 / n
        else:
            w_sol = w_sol / s

        return StrategyResult(weights=pd.Series(w_sol, index=cols))

    def get_weights_from_cov(self, cov: pd.DataFrame) -> StrategyResult:
        """
        solve mv directly from a supplied covariance matrix
        used by nco inter-cluster step (snippet 7.5 uses minVarPort(cov2))
        """
        if not isinstance(cov, pd.DataFrame):
            raise TypeError("mv.get_weights_from_cov expects cov as pd.DataFrame")

        cols = cov.index
        if not cols.equals(cov.columns):
            raise ValueError("mv.get_weights_from_cov expects cov with matching index/columns")

        n = int(cov.shape[0])
        if n == 0:
            raise ValueError("mv.get_weights_from_cov: empty covariance")

        sigma = cov.values.astype(float)
        sigma = 0.5 * (sigma + sigma.T)
        sigma = sigma + (self.ridge_eps * np.eye(n))

        w_var = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w_var, cp.psd_wrap(sigma)))
        constraints = [cp.sum(w_var) == 1.0, w_var >= 0.0]
        prob = cp.Problem(objective, constraints)

        w_sol = self._solve_qp(prob, w_var)

        w_sol = np.asarray(w_sol, dtype=float).reshape(-1)
        w_sol[~np.isfinite(w_sol)] = 0.0
        w_sol[w_sol < 0.0] = 0.0

        s = float(w_sol.sum())
        if s <= 0.0 or not np.isfinite(s):
            w_sol[:] = 1.0 / n
        else:
            w_sol = w_sol / s

        return StrategyResult(weights=pd.Series(w_sol, index=cols))

    def _solve_qp(self, prob: cp.Problem, w_var: cp.Variable) -> np.ndarray:
        """
        solve the convex qp and return weights
        
        behavior:
         1) try the user-selected primary solver
         2) if that fails or returns invalid values, try SCS once
         3) if still failing, return equal weights
        """

        # try primary solver
        try:
            self._solve(prob, solver=self.solver)
            if w_var.value is not None and np.all(np.isfinite(w_var.value)):
                return w_var.value
        except Exception:
            pass

        # fallback solver: SCS is slower but robust for many convex problems
        try:
            self._solve(prob, solver="SCS")
            if w_var.value is not None and np.all(np.isfinite(w_var.value)):
                return w_var.value
        except Exception:
            pass

        # last resort fallback: equal weights
        n = int(w_var.shape[0])
        return np.ones(n) / n

    def _solve(self, prob: cp.Problem, solver: str) -> None:
        """
        dispatch to the configured solver with stable default parameters

        note: cvxpy solver parameters differ by backend. only the parameters supported by each
        backend are passed here
        """

        s = solver.upper()

        if s == "OSQP":
            # OSQP settings:
            # - eps_abs / eps_rel: convergence tolerances
            # - max_iter: iteration cap
            prob.solve(
                solver=cp.OSQP,
                verbose=False,
                eps_abs=self.solver_eps,
                eps_rel=self.solver_eps,
                max_iter=self.solver_max_iter,
            )
            return

        if s == "SCS":
            # SCS settings:
            # - eps: convergence tolerance
            # - max_iters: iteration cap
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=self.solver_eps,
                max_iters=self.solver_max_iter,
            )
            return

        # allow any other cvxpy-supported solver identifier
        prob.solve(solver=solver, verbose=False)
