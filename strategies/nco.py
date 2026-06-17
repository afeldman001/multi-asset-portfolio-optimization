# strategies/nco.py
#
# nested cluster optimization (nco), following marcos lopez de prado (2020),
# "machine learning for asset managers", chapter 7
#
# ----------------------------------------------------------------------
# project integration:
# - integrates into eval_engine.py:
#     * run_eval.py imports and appends this strategy
#     * eval_engine expects strategy.get_weights(...) to return an object
#       with a .weights attribute (pd.Series indexed by assets)
# - base="mv": use minimum variance optimizer
# - base="msr": use maximum sharpe optimizer
# - long-only constraint enforced via clipping and renormalization
#
# ----------------------------------------------------------------------
# algorithmic intent (de prado, ch. 7):
# nco is an overlay around a mean-variance optimizer. it reduces estimation
# error by:
# (1) clustering assets using correlations
# (2) computing optimal intra-cluster allocations
# (3) computing optimal inter-cluster allocations on reduced moments
# (4) combining intra- and inter-cluster weights into final asset weights
#
# ----------------------------------------------------------------------
# mapping to book snippets:
#
# snippet 4.1 (clusterKMeansBase):
# - implemented in clusterKMeansBase(...)
# - x = ((1 - corr0.fillna(0)) / 2) ** 0.5
# - loops over init and k, selecting clustering by mean(silh)/std(silh)
# - corr matrix reordered by cluster labels
# - returns corr1 (reordered), clstrs (cluster membership), silh
# - fails loudly if no valid clustering is found
# - note: corr1 reordering is returned for fidelity; clstrs drives allocations
#
# snippet 7.3 (implemented across get_weights and optPort_nco):
# - cov0 estimated from return window
# - q = t/n
# - cov1 = deNoiseCov(cov0, q, bWidth)
# - cov1 is constructed in get_weights, and clustering is executed inside optPort_nco via corr1 = cov2corr(cov1) and clusterKMeansBase
#
# snippet 7.4 (intracluster optimal allocations):
# - implemented inside optPort_nco(...)
# - for each cluster:
#     * solve base optimizer on cov submatrix (and mu subvector if msr)
# - build wIntra
# - compute reduced covariance:
#     cov2 = wIntra.T * cov * wIntra
#
# snippet 7.5 (intercluster optimal allocations):
# - implemented inside optPort_nco(...)
# - solve base optimizer on cov2 (and mu2 if msr)
# - combine:
#     wAllo = wIntra * wInter (column-wise multiplication, summed over clusters)
#
# snippet 7.6 (optPort_nco wrapper):
# - implemented as optPort_nco(...)
# - ties clustering, intra-, and inter-optimization together
# - get_weights(...) acts as thesis integration wrapper and calls optPort_nco
#
# ----------------------------------------------------------------------
# implementation design choices:
# - use denoised covariance (cov1) consistently throughout intra and inter steps
# - moments estimated once on the full window; intra-cluster problems use the corresponding submatrices/subvectors
# - enforce long-only constraint via clipping + normalization
# - deterministic seeding for reproducibility (does not alter algorithmic logic)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Any, Tuple
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from cov_denoising import denoise_cov_from_cov, cov2corr
from strategies.mv import MinVariance
from strategies.msr import MaxSharpe

BaseOpt = Literal["mv", "msr"]

@dataclass(frozen=True)
class NCOParams:
    # thesis switch: mv -> mu is none, msr -> mu is not none
    base: BaseOpt = "mv"

    # snippet 7.3: cov1 = deNoiseCov(cov0, q, bWidth=.01)
    bWidth: float = 0.01

    # snippet 4.1: clusterKMeansBase(..., n_init=10)
    n_init: int = 10

    # deterministic seeding for reproducibility
    random_state: int = 42

    # numeric guardrails
    eps: float = 1e-12

# ------------------------------------------------------------
# generic helpers (required for codebase)
# ------------------------------------------------------------

def _safe_normalize(w: pd.Series, eps: float) -> pd.Series:
    s = float(w.sum())
    if not np.isfinite(s) or abs(s) <= eps:
        raise RuntimeError("nco: normalization failed (degenerate weight vector).")
    return w / s

def _clip_long_only_and_normalize(w: pd.Series, eps: float) -> pd.Series:
    # thesis long-only constraint
    w = w.astype(float).clip(lower=0.0)
    return _safe_normalize(w, eps=eps)

def _estimate_cov(returns_window: pd.DataFrame) -> pd.DataFrame:
    # provides cov0 used in snippet 7.3 (must be computed from the window)
    cols = returns_window.columns
    x = returns_window.values.astype(float)
    x = x - x.mean(axis=0)
    t = x.shape[0]
    denom = float(max(t - 1, 1))
    cov = (x.T @ x) / denom
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=cols, columns=cols)

def _extract_weights(result: Any, assets: List[str]) -> pd.Series:
    if isinstance(result, pd.Series):
        return result.reindex(assets).astype(float)

    if isinstance(result, dict):
        if "weights" in result:
            return _extract_weights(result["weights"], assets)
        return pd.Series(result, dtype=float).reindex(assets)

    for attr in ["weights", "w", "portfolio_weights"]:
        if hasattr(result, attr):
            return _extract_weights(getattr(result, attr), assets)

    try:
        return _extract_weights(result["weights"], assets)
    except Exception:
        pass

    if isinstance(result, (list, tuple, np.ndarray)):
        arr = np.asarray(result, dtype=float).reshape(-1)
        if arr.shape[0] != len(assets):
            raise ValueError(f"base returned {arr.shape[0]} weights for {len(assets)} assets")
        return pd.Series(arr, index=assets, dtype=float)

    raise TypeError(f"could not extract weights from base output type={type(result)}")

# ------------------------------------------------------------
# snippet 4.1 base clustering: clusterKMeansBase
# ------------------------------------------------------------

def clusterKMeansBase(
    corr0: pd.DataFrame,
    maxNumClusters: int = 10,
    n_init: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, List[str]], pd.Series]:
    # snippet 4.1: x, silh = ((1-corr0.fillna(0))/2.)**.5, pd.Series()
    corr0 = corr0.fillna(0.0).astype(float)
    x = ((1.0 - corr0) / 2.0)
    x = x.clip(lower=0.0) ** 0.5  # numeric safety; book omits clip
    silh = pd.Series(dtype=float)

    n = int(corr0.shape[0])
    k_max = int(min(maxNumClusters, max(n - 1, 1)))
    if n <= 2 or k_max < 2:
        clstrs = {0: corr0.columns.tolist()}
        silh = pd.Series([np.nan] * n, index=corr0.index, dtype=float)
        return corr0.copy(), clstrs, silh

    kmeans_best: Optional[KMeans] = None

    # snippet 4.1: for init in range(n_init):
    for init in range(int(n_init)):
        rs = int(random_state) + init

        # snippet 4.1: for i in xrange(2, maxNumClusters+1):
        for i in range(2, k_max + 1):
            # snippet 4.1: kmeans_ = KMeans(n_clusters=i, n_jobs=1, n_init=1)
            kmeans_ = KMeans(n_clusters=i, n_init=1, random_state=rs)

            # snippet 4.1: kmeans_ = kmeans_.fit(x)
            kmeans_ = kmeans_.fit(x.values)

            # snippet 4.1: silh_ = silhouette_samples(x, kmeans_.labels_)
            silh_ = silhouette_samples(x.values, kmeans_.labels_)

            # snippet 4.1: stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            # match snippet 4.1 behavior:
            # - numpy std uses ddof=0
            # - pandas Series std uses ddof=1 by default
            sd0 = float(np.std(silh_, ddof=0))
            stat0 = float(np.mean(silh_) / sd0) if sd0 > 0.0 else np.nan

            if silh.size == 0:
                stat1 = np.nan
            else:
                sd1 = float(silh.std())  # pandas default ddof=1
                stat1 = float(silh.mean() / sd1) if sd1 > 0.0 else np.nan

            # snippet 4.1: if np.isnan(stat[1]) or stat[0] > stat[1]:
            if np.isnan(stat1) or (np.isfinite(stat0) and stat0 > stat1):
                # snippet 4.1: silh, kmeans = silh_, kmeans_
                silh = pd.Series(silh_, index=x.index, dtype=float)
                kmeans_best = kmeans_

    # snippet 4.1 assumes this exists; thesis version fails loudly if not
    if kmeans_best is None or silh.size == 0:
        raise RuntimeError("clusterKMeansBase failed: no valid clustering found for any k.")

    # snippet 4.1: newIdx = np.argsort(kmeans.labels_)
    newIdx = np.argsort(kmeans_best.labels_)

    # snippet 4.1: corr1 = corr0.iloc[newIdx]  # reorder rows
    corr1 = corr0.iloc[newIdx].copy()

    # snippet 4.1: corr1 = corr1.iloc[:, newIdx]  # reorder columns
    corr1 = corr1.iloc[:, newIdx].copy()

    # snippet 4.1: clstrs = {i: corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() ...}
    labels = kmeans_best.labels_
    clstrs: Dict[int, List[str]] = {
        int(i): corr0.columns[np.where(labels == i)[0]].tolist()
        for i in np.unique(labels)
    }

    # snippet 4.1: return corr1, clstrs, silh
    return corr1, clstrs, silh

# ------------------------------------------------------------
# snippet 7.6 function implementing the nco algorithm: optPort_nco
# ------------------------------------------------------------

def optPort_nco(
    cov: pd.DataFrame,
    mu: Optional[pd.Series] = None,
    rf_bar: Optional[float] = None,
    maxNumClusters: Optional[int] = None,
    eps: float = 1e-12,
    n_init: int = 10,
    random_state: int = 42,
) -> Tuple[pd.Series, Dict[int, List[str]], pd.Series]:
    # snippet 7.6: cov = pd.DataFrame(cov)
    cov = pd.DataFrame(cov)

    # snippet 7.6: if mu is not None: mu = pd.Series(mu[:,0])
    if mu is not None:
        mu = pd.Series(mu, index=cov.index, dtype=float)

    # snippet 7.6: corr1 = cov2corr(cov)
    corr1 = pd.DataFrame(cov2corr(cov.values), index=cov.index, columns=cov.columns)
    corr1 = 0.5 * (corr1 + corr1.T)

    # snippet 7.6: corr1, clstrs, _ = clusterKMeansBase(corr1, maxNumClusters, n_init=10)
    # maxNumClusters follows López de Prado’s choice of half the correlation-matrix dimension (Snippet 7.3), 
    # implemented as n//2 (floored division) and clamped to [2, n−1] to ensure a valid integer range for KMeans
    if maxNumClusters is None:
        maxNumClusters = int(max(2, min(corr1.shape[0] - 1, corr1.shape[0] // 2)))

    corr1, clstrs, silh = clusterKMeansBase(
        corr0=corr1,
        maxNumClusters=int(maxNumClusters),
        n_init=int(n_init),
        random_state=int(random_state),
    )

    # snippet 7.6: wIntra = pd.DataFrame(0, index=cov.index, columns=clstrs.keys())
    wIntra = pd.DataFrame(0.0, index=cov.index, columns=list(clstrs.keys()), dtype=float)

    mv = MinVariance()
    msr = MaxSharpe()

    # snippet 7.6: for i in clstrs:
    for i in clstrs:
        assets_i = clstrs[i]

        # snippet 7.6: cov_ = cov.loc[clstrs[i], clstrs[i]].values
        cov_ = cov.loc[assets_i, assets_i].copy()
        cov_ = 0.5 * (cov_ + cov_.T)

        # snippet 7.6: if mu is None: mu_ = None else: mu_ = mu.loc[clstrs[i]]...
        if mu is None:
            # snippet 7.6: wIntra.loc[clstrs[i], i] = optPort(cov_, mu_).flatten()
            out = mv.get_weights_from_cov(cov_)
            w_sub = _extract_weights(out, assets_i)
        else:
            if rf_bar is None or not np.isfinite(float(rf_bar)):
                raise RuntimeError("optPort_nco(msr) requires rf_bar to be provided and finite")
            mu_ = mu.reindex(assets_i).astype(float)
            out = msr.get_weights_from_moments(mu_raw=mu_, cov=cov_, rf_bar=float(rf_bar))
            w_sub = _extract_weights(out, assets_i)

        w_sub = _clip_long_only_and_normalize(w_sub, eps=eps)
        wIntra.loc[assets_i, i] = w_sub.values

    # snippet 7.6: cov_ = wIntra.T.dot(np.dot(cov, wIntra)); reduce covariance matrix
    cov_ = wIntra.T.dot(cov.dot(wIntra))
    cov_ = 0.5 * (cov_ + cov_.T)

    # snippet 7.6: mu_ = (None if mu is None else wIntra.T.dot(mu))
    mu_ = None if mu is None else wIntra.T.dot(mu)

    # snippet 7.6: wInter = pd.Series(optPort(cov_, mu_).flatten(), index=cov_.index)
    if mu_ is None:
        out_inter = mv.get_weights_from_cov(cov_)
        wInter = _extract_weights(out_inter, cov_.index.tolist())
    else:
        if rf_bar is None or not np.isfinite(float(rf_bar)):
            raise RuntimeError("optPort_nco(msr) requires rf_bar to be provided and finite")
        out_inter = msr.get_weights_from_moments(mu_raw=mu_, cov=cov_, rf_bar=float(rf_bar))
        wInter = _extract_weights(out_inter, cov_.index.tolist())

    wInter = _clip_long_only_and_normalize(wInter, eps=eps)

    # snippet 7.6: nco = wIntra.mul(wInter, axis=1).sum(axis=1).values.reshape(-1,1)
    nco = wIntra.mul(wInter, axis=1).sum(axis=1)
    nco = _clip_long_only_and_normalize(nco, eps=eps)

    # snippet 7.6: return nco
    return nco, clstrs, silh

# ------------------------------------------------------------
# strategy wrapper for eval_engine: get_weights(...)
# ------------------------------------------------------------

class NestedClusterOptimization:
    def __init__(
        self,
        base: BaseOpt = "mv",
        bWidth: float = 0.01,
        n_init: int = 10,
        random_state: int = 42,
        eps: float = 1e-12,
    ):
        self.params = NCOParams(
            base=base,
            bWidth=bWidth,
            n_init=n_init,
            random_state=random_state,
            eps=eps,
        )
        self.name = f"nco_{base}"
        self.last_info: Optional[Dict[str, float]] = None

    def get_weights(self, returns_window: pd.DataFrame, window_rf: Optional[pd.Series] = None) -> Any:
        if not isinstance(returns_window, pd.DataFrame):
            raise TypeError("nco.get_weights expects returns_window as pd.DataFrame")

        
        if returns_window.shape[1] == 0:
            raise ValueError("returns_window has no assets")

        if returns_window.shape[1] == 1:
            w_one = pd.Series([1.0], index=returns_window.columns, dtype=float)
            return SimpleNamespace(weights=w_one)

        # enforce canonical asset ordering so cov/denoising/mu are permutation-invariant
        assets = sorted(returns_window.columns.tolist())
        x = returns_window[assets]

        t, n = x.shape

        # chapter 7: q = t/n (rolling window ratio)
        q = float(t) / float(n)

        # snippet 7.3 requires cov0 as input
        cov0 = _estimate_cov(x)

        # snippet 7.3: cov1 = deNoiseCov(cov0, q, bWidth)
        cov1 = denoise_cov_from_cov(cov=cov0, q=float(q), bWidth=float(self.params.bWidth), method="const_resid")
        cov1 = pd.DataFrame(cov1.values, index=cov0.index, columns=cov0.columns)
        cov1 = 0.5 * (cov1 + cov1.T)

        # base switch matches snippet 7.6: mu none -> mv, mu not none -> msr
        if self.params.base == "mv":
            mu = None
            rf_bar = None
        else:
            x = x.astype(float)
            mu = pd.Series(x.mean(axis=0), index=x.columns, dtype=float)

            if window_rf is None:
                raise RuntimeError("nco_msr requires window_rf for rf_bar computation")
            rf = window_rf.reindex(x.index).astype(float)
            if rf.isna().any():
                rf = rf.ffill().bfill()
            if rf.isna().any():
                raise RuntimeError("nco_msr failed: rf contains unresolved missing values after fill")
            rf_bar = float(rf.mean())
            if not np.isfinite(rf_bar):
                raise RuntimeError("nco_msr failed: rf_bar is non-finite")

        # snippet 7.6: optPort_nco(cov, mu, maxNumClusters=None)
        # pass cov1 (denoised covariance) so the pipeline uses the cleaned matrix from snippet 7.3/7.4
        maxNumClusters = int(max(2, min(cov1.shape[0] - 1, cov1.shape[0] // 2)))

        w_all, clstrs, silh = optPort_nco(
            cov=cov1,
            mu=mu,
            rf_bar=rf_bar,
            maxNumClusters=maxNumClusters,
            eps=self.params.eps,
            n_init=self.params.n_init,
            random_state=self.params.random_state,
        )

        w_all = w_all.reindex(x.columns).astype(float)     # internal canonical
        w_all = w_all.reindex(returns_window.columns)      # return aligned to pipeline ordering
        w_all = w_all.astype(float)
        w_all = _clip_long_only_and_normalize(w_all, eps=self.params.eps)

        # diagnostics 
        self.last_info = {
            "debug_t": float(t),
            "debug_n": float(n),
            "debug_q": float(q),
            "debug_num_clusters": float(len(clstrs)),
            "debug_cluster_min_size": float(min(len(v) for v in clstrs.values())),
            "debug_cluster_max_size": float(max(len(v) for v in clstrs.values())),
            "debug_silh_mean": float(np.nanmean(silh.values)),
            "debug_silh_std": float(np.nanstd(silh.values)),
        }

        return SimpleNamespace(weights=w_all)
