# strategies/msr.py
#
# maximum sharpe ratio portfolio (msrp) using palomar (2025) algorithm 7.1 (bisection):
#
# - palomar (2025) section 7.2, problem (7.7): msrp objective
# - palomar (2025) section 7.2.1, problem (7.8): convex feasibility problem
# - palomar (2025) algorithm 7.1: bisection method
#
# what is implemented:
# - algorithm 7.1 requires: choose [l, u] with l > 0 containing the optimal sharpe ratio, tolerance epsilon > 0
#   then repeat: t <- (l+u)/2; solve feasibility (7.8); if feasible set l <- t and keep w; else set u <- t
#   stop when u - l <= epsilon.
#
# feasibility problem (7.8):
#   find w
#   subject to:
#     t * sqrt(w' * sigma * w) <= w' * mu - r_f
#     1' * w = 1
#     w >= 0
#
# mapping to thesis codebase:
# - mu is estimated as the sample mean of simple returns over the estimation window (mu_raw)
# - sigma is estimated as ledoit-wolf shrinkage covariance on simple returns (sigma)
# - r_f in (7.8) is a scalar. window_rf is a time series, so r_f is set to the sample mean over the window (rf_bar)
#   this keeps (mu, sigma, r_f) in consistent "per-period" units on the estimation window
#
# edgecase handling:
#   - if a valid bracket cannot be constructed, raise with diagnostics
#   - if all conic solvers fail, raise with diagnostics
#   - if bisection cannot keep a feasible witness, raise with diagnostics
#
# solver robustness:
# - (7.8) is an socp. clarabel is preferred, but it can fail numerically in some windows
# - if solver="AUTO", a solver cascade is used: clarabel -> ecos -> scs (only among installed solvers)
# - if a solver throws a cvxpy SolverError, the next solver is tried
# - if all solvers fail, a single error is raised including the full solver error trace summary

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import warnings

import numpy as np
import pandas as pd

import cvxpy as cp
from cvxpy.error import SolverError
from sklearn.covariance import LedoitWolf

from strategies.base import BaseStrategy, StrategyResult


@dataclass(frozen=True)
class _Bracket:
    l: float
    u: float


class MaxSharpe(BaseStrategy):
    name = "msr"

    def __init__(
        self,
        solver: str = "auto",
        solver_max_iter: int = 100_000,
        solver_eps: float = 1e-6,
        ridge_eps: float = 1e-10,
        epsilon: float = 1e-7,                # palomar algorithm 7.1: "tolerance epsilon > 0", stop when u - l <= epsilon
        bracket_l_init: float = 1e-6,         # palomar algorithm 7.1: l > 0
        bracket_u_init: float = 1.0,          # initial u guess before expansion
        bracket_u_max: float = 1e3,           # safety cap for expansion (not part of algorithm 7.1)
        bracket_expand_factor: float = 2.0,   # expand u until (7.8) becomes infeasible
        suppress_warnings: bool = True,
        debug: bool = False,                  # include detailed diagnostics in raised exceptions (set to False to turn off)
        verbose_solver_on_fail: bool = False  # if true, re-solve the failing instance with verbose=True before raising
    ) -> None:
        self.solver = str(solver).upper()
        self.solver_max_iter = int(solver_max_iter)
        self.solver_eps = float(solver_eps)

        self.ridge_eps = float(ridge_eps)

        self.epsilon = float(epsilon)
        self.bracket_l_init = float(bracket_l_init)
        self.bracket_u_init = float(bracket_u_init)
        self.bracket_u_max = float(bracket_u_max)
        self.bracket_expand_factor = float(bracket_expand_factor)

        self.suppress_warnings = bool(suppress_warnings)
        self.debug = bool(debug)
        self.verbose_solver_on_fail = bool(verbose_solver_on_fail)

        self.last_info: Optional[Dict[str, Any]] = None

        if self.epsilon <= 0.0:
            raise ValueError("msr: epsilon must be > 0 (palomar algorithm 7.1)")

        if self.bracket_l_init <= 0.0:
            raise ValueError("msr: bracket_l_init must be > 0 (palomar algorithm 7.1 requires l > 0)")

        if self.bracket_expand_factor <= 1.0:
            raise ValueError("msr: bracket_expand_factor must be > 1 to expand u")

    def get_weights(
        self,
        window_returns: pd.DataFrame,
        window_rf: Optional[pd.Series] = None,
    ) -> StrategyResult:
        x = window_returns.dropna(how="any")
        cols = x.columns
        t_obs, n = x.shape

        self.last_info = None

        if t_obs < 5:
            raise RuntimeError("msr failed: insufficient return observations")

        if window_rf is None:
            raise RuntimeError("msr failed: window_rf is required for palomar (7.7) and (7.8)")

        rf = window_rf.reindex(x.index).astype(float)
        if rf.isna().any():
            rf = rf.ffill().bfill()
        if rf.isna().any():
            raise RuntimeError("msr failed: rf contains unresolved missing values after fill")

        # palomar (7.8) uses scalar r_f; thesis mapping: r_f := mean(rf_t) over the estimation window
        rf_bar = float(rf.mean())
        if not np.isfinite(rf_bar):
            raise RuntimeError("msr failed: rf_bar is non-finite")

        # palomar mu in (7.7) and (7.8) is expected return vector (raw, not excess)
        mu_raw = x.mean(axis=0).astype(float).values.reshape(-1)
        if mu_raw.shape[0] != n or not np.isfinite(mu_raw).all():
            raise RuntimeError("msr failed: non-finite mu_raw")

        # sigma is covariance of returns (psd); ledoit-wolf shrinkage on simple returns
        sigma = self._estimate_sigma_lw(x.values)

        # boundary case (palomar warning): if max(mu) <= rf, then (7.8) is infeasible for every t > 0
        # in that regime, the best achievable sharpe is <= 0; return the minimum-variance simplex portfolio
        mu_max = float(np.max(mu_raw))
        if mu_max <= rf_bar + 1e-12:
            self.last_info = {
                "debug_regime": "mu_max_le_rf",
                "debug_t": float(t_obs),
                "debug_n": float(n),
                "debug_mu_max": float(mu_max),
                "debug_rf_bar": float(rf_bar),
            }

            w = cp.Variable(n)
            obj = cp.Minimize(cp.quad_form(w, sigma))
            constraints = [cp.sum(w) == 1.0, w >= 0.0]
            prob = cp.Problem(obj, constraints)

            installed = {name.upper() for name in cp.installed_solvers()}
            mv_solvers = [s for s in ["OSQP", "SCS", "CLARABEL"] if s in installed]
            if not mv_solvers:
                mv_solvers = [s for s in ["ECOS"] if s in installed]

            if not mv_solvers:
                raise RuntimeError("msr failed: no solver available for boundary min-variance solve")

            mv_errors: List[Tuple[str, str]] = []
            solved = False
            for s in mv_solvers:
                try:
                    if s == "OSQP":
                        prob.solve(solver=cp.OSQP, verbose=False)
                    elif s == "SCS":
                        prob.solve(solver=cp.SCS, verbose=False, max_iters=self.solver_max_iter, eps=self.solver_eps)
                    elif s == "CLARABEL":
                        prob.solve(
                            solver=cp.CLARABEL,
                            verbose=False,
                            max_iter=self.solver_max_iter,
                            tol_feas=self.solver_eps,
                            tol_gap_abs=self.solver_eps,
                            tol_gap_rel=self.solver_eps,
                        )
                    else:
                        prob.solve(solver=s, verbose=False)
                    solved = True
                    break
                except Exception as e:
                    mv_errors.append((s, str(e)))
                    continue

            if (not solved) or (prob.status not in ("optimal", "optimal_inaccurate")) or (w.value is None):
                raise RuntimeError(
                    f"msr failed: boundary min-variance solve did not return a solution. "
                    f"status={prob.status} errors={mv_errors}"
                )

            w_mv = np.asarray(w.value, dtype=float).reshape(-1)
            w_mv[w_mv < 0.0] = 0.0
            ssum = float(np.sum(w_mv))
            if (not np.isfinite(ssum)) or (ssum <= 0.0):
                w_mv = np.full(n, 1.0 / n, dtype=float)
            else:
                w_mv = w_mv / ssum

            return StrategyResult(weights=pd.Series(w_mv, index=cols))

        # socp norm representation (as stated in the text under 7.2.1):
        # sqrt(w' sigma w) can be written as an l2 norm via a factorization sigma = L' L
        L = self._psd_factor(sigma)

        # algorithm 7.1: choose [l, u] with l > 0 that contains the optimal sharpe ratio
        # operationally enforce: feasible at l and infeasible at u for feasibility problem (7.8)
        bracket = self._construct_bracket(mu_raw=mu_raw, rf_bar=rf_bar, L=L)

        # algorithm 7.1 bisection loop with "keep solution w" when feasible
        l = bracket.l
        u = bracket.u

        w_best: Optional[np.ndarray] = None

        # verify initial feasibility at l and infeasibility at u (useful for debugging)
        # not an extra algorithmic step; validates the constructed bracket
        if self.debug:
            w_l = self._solve_feasibility_78(mu_raw=mu_raw, rf_bar=rf_bar, L=L, t=l)
            if w_l is None:
                raise RuntimeError(self._fmt_diag(
                    "msr failed: bracket validation failed, l is not feasible",
                    mu_raw=mu_raw, rf_bar=rf_bar, l=l, u=u, t=l
                ))
            w_u = self._solve_feasibility_78(mu_raw=mu_raw, rf_bar=rf_bar, L=L, t=u)
            if w_u is not None:
                raise RuntimeError(self._fmt_diag(
                    "msr failed: bracket validation failed, u is still feasible (optimum not enclosed)",
                    mu_raw=mu_raw, rf_bar=rf_bar, l=l, u=u, t=u
                ))

        while (u - l) > self.epsilon:
            t_mid = 0.5 * (l + u)

            w_mid = self._solve_feasibility_78(mu_raw=mu_raw, rf_bar=rf_bar, L=L, t=t_mid)

            if w_mid is not None:
                # algorithm 7.1 step 3: feasible -> l <- t and keep solution w
                l = t_mid
                w_best = w_mid
            else:
                # algorithm 7.1 step 3: infeasible -> u <- t
                u = t_mid

        if w_best is None:
            raise RuntimeError(self._fmt_diag(
                "msr failed: bisection ended without a feasible witness (unexpected if bracket is valid)",
                mu_raw=mu_raw, rf_bar=rf_bar, l=bracket.l, u=bracket.u, t=None
            ))

        w = np.asarray(w_best, dtype=float).reshape(-1)
        if w.shape[0] != n or not np.isfinite(w).all():
            raise RuntimeError("msr failed: non-finite weights returned from solver")

        # enforce simplex constraints numerically (solver tolerances can leave tiny violations)
        w[w < 0.0] = 0.0
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0.0:
            raise RuntimeError("msr failed: non-positive weight sum after cleanup")
        w = w / s

        self.last_info = {
            "debug_regime": "normal",
            "debug_t": float(t_obs),
            "debug_n": float(n),
            "debug_mu_max": float(mu_max),
            "debug_rf_bar": float(rf_bar),
            "debug_bracket_l": float(bracket.l),
            "debug_bracket_u": float(bracket.u),
        }

        return StrategyResult(weights=pd.Series(w, index=cols))

    def get_weights_from_moments(
        self,
        mu_raw: pd.Series,
        cov: pd.DataFrame,
        rf_bar: float,
    ) -> StrategyResult:
        """
        solve msr directly from supplied moments (mu_raw, cov, rf_bar).
        used by nco inter-cluster step on reduced moments (mu2, cov2).
        """
        if not isinstance(mu_raw, pd.Series):
            raise TypeError("msr.get_weights_from_moments expects mu_raw as pd.Series")
        if not isinstance(cov, pd.DataFrame):
            raise TypeError("msr.get_weights_from_moments expects cov as pd.DataFrame")

        cols = cov.index
        if not cols.equals(cov.columns):
            raise ValueError("msr.get_weights_from_moments expects cov with matching index/columns")

        mu_vec = mu_raw.reindex(cols).astype(float).values.reshape(-1)
        if mu_vec.shape[0] != len(cols) or not np.isfinite(mu_vec).all():
            raise RuntimeError("msr failed: non-finite mu_raw in get_weights_from_moments")

        rf_bar = float(rf_bar)
        if not np.isfinite(rf_bar):
            raise RuntimeError("msr failed: rf_bar non-finite in get_weights_from_moments")

        sig = cov.values.astype(float)
        sig = 0.5 * (sig + sig.T)
        n = int(sig.shape[0])
        sig = sig + (self.ridge_eps * np.eye(n))
        sig = 0.5 * (sig + sig.T)

        self.last_info = None

        mu_max = float(np.max(mu_vec))
        if mu_max <= rf_bar + 1e-12:
            # same boundary behavior as get_weights: return simplex min-variance
            w = cp.Variable(n)
            obj = cp.Minimize(cp.quad_form(w, sig))
            constraints = [cp.sum(w) == 1.0, w >= 0.0]
            prob = cp.Problem(obj, constraints)

            installed = {name.upper() for name in cp.installed_solvers()}
            mv_solvers = [s for s in ["OSQP", "SCS", "CLARABEL"] if s in installed]
            if not mv_solvers:
                mv_solvers = [s for s in ["ECOS"] if s in installed]
            if not mv_solvers:
                raise RuntimeError("msr failed: no solver available for boundary min-variance solve")

            solved = False
            for s in mv_solvers:
                try:
                    if s == "OSQP":
                        prob.solve(solver=cp.OSQP, verbose=False)
                    elif s == "SCS":
                        prob.solve(solver=cp.SCS, verbose=False, max_iters=self.solver_max_iter, eps=self.solver_eps)
                    elif s == "CLARABEL":
                        prob.solve(
                            solver=cp.CLARABEL,
                            verbose=False,
                            max_iter=self.solver_max_iter,
                            tol_feas=self.solver_eps,
                            tol_gap_abs=self.solver_eps,
                            tol_gap_rel=self.solver_eps,
                        )
                    else:
                        prob.solve(solver=s, verbose=False)
                    solved = True
                    break
                except Exception:
                    continue

            if (not solved) or (prob.status not in ("optimal", "optimal_inaccurate")) or (w.value is None):
                raise RuntimeError(f"msr failed: boundary min-variance solve did not return a solution. status={prob.status}")

            w_mv = np.asarray(w.value, dtype=float).reshape(-1)
            w_mv[w_mv < 0.0] = 0.0
            ssum = float(np.sum(w_mv))
            if (not np.isfinite(ssum)) or (ssum <= 0.0):
                w_mv = np.full(n, 1.0 / n, dtype=float)
            else:
                w_mv = w_mv / ssum

            self.last_info = {
                "debug_regime": "mu_max_le_rf",
                "debug_n": float(n),
                "debug_mu_max": float(mu_max),
                "debug_rf_bar": float(rf_bar),
            }
            return StrategyResult(weights=pd.Series(w_mv, index=cols))

        L = self._psd_factor(sig)
        bracket = self._construct_bracket(mu_raw=mu_vec, rf_bar=rf_bar, L=L)

        l = bracket.l
        u = bracket.u
        w_best: Optional[np.ndarray] = None

        while (u - l) > self.epsilon:
            t_mid = 0.5 * (l + u)
            w_mid = self._solve_feasibility_78(mu_raw=mu_vec, rf_bar=rf_bar, L=L, t=t_mid)
            if w_mid is not None:
                l = t_mid
                w_best = w_mid
            else:
                u = t_mid

        if w_best is None:
            raise RuntimeError("msr failed: bisection ended without a feasible witness")

        w = np.asarray(w_best, dtype=float).reshape(-1)
        if w.shape[0] != n or not np.isfinite(w).all():
            raise RuntimeError("msr failed: non-finite weights returned from solver")

        w[w < 0.0] = 0.0
        ssum = float(np.sum(w))
        if not np.isfinite(ssum) or ssum <= 0.0:
            raise RuntimeError("msr failed: non-positive weight sum after cleanup")
        w = w / ssum

        self.last_info = {
            "debug_regime": "normal",
            "debug_n": float(n),
            "debug_mu_max": float(mu_max),
            "debug_rf_bar": float(rf_bar),
            "debug_bracket_l": float(bracket.l),
            "debug_bracket_u": float(bracket.u),
        }

        return StrategyResult(weights=pd.Series(w, index=cols))

    # ---------------------------------------------------------------------
    # palomar algorithm 7.1: choose [l, u] with l > 0 containing the optimum
    # implemented by enforcing feasibility at l and infeasibility at u for (7.8)
    # ---------------------------------------------------------------------

    def _construct_bracket(self, mu_raw: np.ndarray, rf_bar: float, L: np.ndarray) -> _Bracket:
        l = float(self.bracket_l_init)
        if l <= 0.0:
            raise RuntimeError("msr failed: bracket_l_init must be > 0 (palomar algorithm 7.1)")

        # must be feasible at l
        w_l = self._solve_feasibility_78(mu_raw=mu_raw, rf_bar=rf_bar, L=L, t=l)
        if w_l is None:
            # this means even a tiny positive sharpe level cannot be supported by (7.8)
            # palomar notes infeasibility can occur in practice; fail loudly
            raise RuntimeError(self._fmt_diag(
                "msr failed: cannot construct bracket because (7.8) is infeasible at the required l > 0",
                mu_raw=mu_raw, rf_bar=rf_bar, l=l, u=None, t=l
            ))

        # expand u until infeasible
        u = float(self.bracket_u_init)
        if u <= l:
            u = l * 2.0

        while u <= self.bracket_u_max:
            w_u = self._solve_feasibility_78(mu_raw=mu_raw, rf_bar=rf_bar, L=L, t=u)
            if w_u is None:
                return _Bracket(l=l, u=u)
            u *= float(self.bracket_expand_factor)

        raise RuntimeError(self._fmt_diag(
            "msr failed: cannot construct bracket, (7.8) remained feasible up to bracket_u_max",
            mu_raw=mu_raw, rf_bar=rf_bar, l=l, u=self.bracket_u_max, t=None
        ))

    # ---------------------------------------------------------------------
    # palomar (7.8) feasibility problem used in algorithm 7.1 step 2
    # ---------------------------------------------------------------------

    def _solve_feasibility_78(self, mu_raw: np.ndarray, rf_bar: float, L: np.ndarray, t: float) -> Optional[np.ndarray]:
        """
        palomar (7.8):
           t * sqrt(w' sigma w) <= w' mu - r_f
           1' w = 1
           w >= 0
        
        implement sqrt(w' sigma w) as ||L w||_2 where sigma = L' L
        implement "find w" as minimize 0 subject to constraints
        """

        if (not np.isfinite(t)) or (t <= 0.0):
            return None

        n = int(mu_raw.shape[0])
        w = cp.Variable(n)

        lhs = float(t) * cp.norm(L @ w, 2)
        rhs = (mu_raw @ w) - float(rf_bar)

        constraints = [
            lhs <= rhs,
            cp.sum(w) == 1.0,
            w >= 0.0,
        ]

        prob = cp.Problem(cp.Minimize(0.0), constraints)

        solve_meta = {
            "t": float(t),
            "rf_bar": float(rf_bar),
            "mu_min": float(np.min(mu_raw)),
            "mu_max": float(np.max(mu_raw)),
            "mu_mean": float(np.mean(mu_raw)),
            "n": int(n),
        }

        try:
            self._solve_conic(prob, solve_meta=solve_meta)
        except SolverError as e:
            # solver failure is different from infeasibility
            # fail loudly, so raise with diagnostics
            raise RuntimeError(self._fmt_diag(
                f"msr failed: conic solver failure while solving (7.8) at t={t:.6g}",
                mu_raw=mu_raw, rf_bar=rf_bar, l=None, u=None, t=t, extra={"solve_meta": solve_meta, "solver_error": str(e)}
            )) from e

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return np.asarray(w.value, dtype=float)

        # infeasible or other non-solution statuses are treated as infeasible for bisection purposes
        return None

    # ---------------------------------------------------------------------
    # conic solver selection and execution with diagnostics
    # ---------------------------------------------------------------------

    def _solve_conic(self, prob: cp.Problem, solve_meta: Optional[Dict[str, Any]] = None) -> None:
        # choose solver list
        installed = {name.upper() for name in cp.installed_solvers()}

        if self.solver == "AUTO":
            # clarabel -> ecos -> scs (only if installed)
            candidates = [s for s in ["CLARABEL", "ECOS", "SCS"] if s in installed]
        else:
            candidates = [self.solver] if self.solver in installed else []

        if not candidates:
            raise SolverError(f"no requested/eligible conic solver installed; installed={sorted(installed)}")

        errors: List[Tuple[str, str]] = []

        for s in candidates:
            try:
                self._solve_with_solver(prob, solver_name=s)
                return
            except SolverError as e:
                errors.append((s, str(e)))
                continue

        # re-run verbosely on the last solver for more trace output in the console
        if self.verbose_solver_on_fail and candidates:
            try:
                self._solve_with_solver(prob, solver_name=candidates[-1], verbose=True)
            except Exception:
                pass

        msg = "all conic solvers failed for (7.8)."
        extra = {"attempted_solvers": candidates, "errors": errors, "solve_meta": solve_meta, "status": prob.status}
        raise SolverError(f"{msg} details={extra}")

    def _solve_with_solver(self, prob: cp.Problem, solver_name: str, verbose: bool = False) -> None:
        # suppress cvxpy's "solution may be inaccurate" noise unless requested otherwise
        if self.suppress_warnings:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self._solve_with_solver_inner(prob, solver_name=solver_name, verbose=verbose)
        else:
            self._solve_with_solver_inner(prob, solver_name=solver_name, verbose=verbose)

    def _solve_with_solver_inner(self, prob: cp.Problem, solver_name: str, verbose: bool) -> None:
        if solver_name == "CLARABEL":
            prob.solve(
                solver=cp.CLARABEL,
                verbose=bool(verbose),
                max_iter=self.solver_max_iter,
                tol_feas=self.solver_eps,
                tol_gap_abs=self.solver_eps,
                tol_gap_rel=self.solver_eps,
            )
            return

        if solver_name == "SCS":
            prob.solve(
                solver=cp.SCS,
                verbose=bool(verbose),
                max_iters=self.solver_max_iter,
                eps=self.solver_eps,
            )
            return

        if solver_name == "ECOS":
            # ecos is available for many socps; use tolerances as close as possible
            prob.solve(
                solver=cp.ECOS,
                verbose=bool(verbose),
                abstol=self.solver_eps,
                reltol=self.solver_eps,
                feastol=self.solver_eps,
                max_iters=self.solver_max_iter,
            )
            return

        # fallback: let cvxpy route to a specified solver
        prob.solve(solver=solver_name, verbose=bool(verbose))

    # ---------------------------------------------------------------------
    # estimation and helpers
    # ---------------------------------------------------------------------

    def _estimate_sigma_lw(self, x: np.ndarray) -> np.ndarray:
        lw = LedoitWolf().fit(x)
        sigma = lw.covariance_.astype(float)

        sigma = 0.5 * (sigma + sigma.T)
        n = int(sigma.shape[0])
        sigma = sigma + (self.ridge_eps * np.eye(n))
        sigma = 0.5 * (sigma + sigma.T)

        if not np.isfinite(sigma).all():
            raise RuntimeError("msr failed: non-finite covariance estimate")

        return sigma

    def _psd_factor(self, sigma: np.ndarray) -> np.ndarray:
        d, Q = np.linalg.eigh(0.5 * (sigma + sigma.T))
        if not np.isfinite(d).all() or not np.isfinite(Q).all():
            raise RuntimeError("msr failed: sigma eigendecomposition non-finite")

        d = np.clip(d, 0.0, None)
        L = (np.diag(np.sqrt(d)) @ Q.T).astype(float)

        if not np.isfinite(L).all():
            raise RuntimeError("msr failed: non-finite psd factor L")

        return L

    # ---------------------------------------------------------------------
    # diagnostics formatting
    # ---------------------------------------------------------------------

    def _fmt_diag(
        self,
        headline: str,
        mu_raw: Optional[np.ndarray],
        rf_bar: Optional[float],
        l: Optional[float],
        u: Optional[float],
        t: Optional[float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.debug:
            return headline

        d: Dict[str, Any] = {}
        if mu_raw is not None:
            d["mu_min"] = float(np.min(mu_raw))
            d["mu_max"] = float(np.max(mu_raw))
            d["mu_mean"] = float(np.mean(mu_raw))
            d["mu_std"] = float(np.std(mu_raw))
            d["mu_pos_frac"] = float(np.mean(mu_raw > 0.0))
        if rf_bar is not None:
            d["rf_bar"] = float(rf_bar)
        if l is not None:
            d["l"] = float(l)
        if u is not None:
            d["u"] = float(u)
        if t is not None:
            d["t"] = float(t)
        if extra:
            d["extra"] = extra

        return f"{headline} | diag={d}"
