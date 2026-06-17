# strategies/hrp.py
#
# hierarchical risk parity (hrp)
#
# implementation:
# - follow marcos lópez de prado (2018), advances in financial machine learning, chapter 16
# - replicate the three-stage hrp pipeline:
#   (1) tree clustering via correlation distance + hierarchical linkage
#   (2) quasi-diagonalization to obtain an ordered leaf list
#   (3) recursive bisection to allocate weights (inverse-variance within clusters)
#
# hrp pipeline:
# - stage (1) tree clustering
# - stage (2) quasi-diagonalization
# - stage (3) recursive bisection allocation
# integrates with the BaseStrategy interface via HierarchicalRiskParity.get_weights()
#
# references:
# - lópez de prado, m. (2018), advances in financial machine learning, chapter 16, snippet 16.1
# - lópez de prado, m. (2016/2018), "building diversified portfolios that outperform out of sample"
#   (ssrn 2708678), appendix code: correlDist + linkage usage
#
# note:
# - scipy.cluster.hierarchy.linkage does not accept an n x n "square" distance matrix
#   it expects a condensed distance vector (length n*(n-1)/2), typically from
#   scipy.spatial.distance.squareform. passing the square matrix directly can make scipy
#   interpret it as an observation matrix, producing incorrect clustering

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from strategies.base import BaseStrategy, StrategyResult, _normalize_long_only

# ------------------------------------------------------------
# configuration
# ------------------------------------------------------------

@dataclass(frozen=True)
class HRPConfig:
    # linkage method used in lópez de prado (2018) snippet 16.1
    # canonical hrp implementation uses single linkage in lópez de prado's code examples
    linkage_method: str = "single"

    # numeric guardrails
    eps: float = 1e-12

    # if an asset has near-zero variance in the window, corr can become nan
    # drop such assets during hrp fitting to keep distance matrix valid
    min_var: float = 1e-12

    # if true, clip correlation into [-1, 1] before distance transform
    clip_corr: bool = True


# ---------------------------------------------------------------------------------
# stage (1): tree clustering 
# (marcos lópez de prado, 2018, snippet 16.1; adjusted for scipy expected params)
# ---------------------------------------------------------------------------------

def _cov_corr(x: pd.DataFrame, cfg: HRPConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # x is expected to be returns, columns are assets
    if not isinstance(x, pd.DataFrame):
        raise TypeError("x must be a pandas dataframe of returns")

    if x.shape[1] < 2:
        raise ValueError("hrp requires at least two assets")

    # drop assets with effectively zero variance to avoid nan correlations
    var = x.var(ddof=1)
    keep = var[var > cfg.min_var].index
    if len(keep) < 2:
        raise ValueError("hrp: fewer than two assets remain after variance filtering")

    x2 = x[keep]

    cov = x2.cov()
    corr = x2.corr()

    if cfg.clip_corr:
        corr = corr.clip(-1.0, 1.0)

    return cov, corr


def _corr_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """
    marcos lópez de prado correlation distance matrix (snippet 16.4 -> 'correlDist()'):
    dist_ij = sqrt((1 - corr_ij) / 2)
    lópez de prado (2018), snippet 16.1 
    """
    # distance matrix (marcos lópez de prado, 2018, snippet 16.1 & 16.4)
    dist = np.sqrt((1.0 - corr) / 2.0) 

    # enforce exact diagonal zeros for hygiene
    np.fill_diagonal(dist.values, 0.0)

    return dist


def _linkage(dist: pd.DataFrame, method: str) -> np.ndarray:
    # scipy expects condensed distance vector, not square form
    dist_condensed = squareform(dist.values, checks=False)
    # linkage matrix (marcos lópez de prado, 2018, snippet 16.1 & 16.4)
    link = sch.linkage(dist_condensed, method=method) 
    return link


def tree_clustering(
    x: pd.DataFrame,
    cfg: Optional[HRPConfig] = None,
) -> Dict[str, Any]:
    """
    stage (1): compute cov, corr, distance, linkage
    
    returns a dict so later stages can reuse artifacts without recomputing
    efficient debugging and unit testing stage-by-stage
    """
    cfg = cfg or HRPConfig()

    cov, corr = _cov_corr(x, cfg)
    dist = _corr_distance(corr)
    link = _linkage(dist, cfg.linkage_method)

    # basic validity checks
    if not sch.is_valid_linkage(link):
        raise ValueError("hrp: scipy produced an invalid linkage matrix")

    return {
        "cov": cov,
        "corr": corr,
        "dist": dist,
        "link": link,
        "assets": list(corr.columns),
        "cfg": cfg,
    }


# ------------------------------------------------------------
# stage (2): quasi-diagonalization
# (marcos lópez de prado, 2018, snippet 16.2)
# ------------------------------------------------------------

def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """
    marcos lópez de prado quasi-diagonalization (2018, ch.16, 'getQuasiDiag()'):
    recursively expand final merge into leaf ordering so correlated items sit near each other

    marcos lópez de prado (2018), ch. 16, snippet 16.2 & 16.4
    """
    if link.ndim != 2 or link.shape[1] != 4:
        raise ValueError("link must be a scipy linkage matrix with shape (n-1, 4)")

    # scipy linkage uses floats; we only need integer cluster ids and counts
    link_i = link.copy()
    link_i[:, 0:2] = link_i[:, 0:2].astype(int)
    link_i[:, 3] = link_i[:, 3].astype(int)

    # start from last merge and sort clustered items by distance 
    sort_ix = pd.Series([int(link_i[-1, 0]), int(link_i[-1, 1])])
    num_items = int(link_i[-1, 3]) # number of original items (leaves) equals the count in the last row
    while sort_ix.max() >= num_items:
        # make space for inserting the two children for each cluster id
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)

        # find cluster ids that need expansion (>= num_items are non-leaf cluster ids)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = (df0.values - num_items).astype(int)  # linkage row indices

        # replace cluster id with its left child
        sort_ix.loc[i] = link_i[j, 0].astype(int) # item 1

        # insert right child
        df1 = pd.Series(link_i[j, 1].astype(int), index=i + 1)

        # pandas append is deprecated; concat is the correct replacement
        sort_ix = pd.concat([sort_ix, df1]).sort_index() # item 2, re-sort

        # re-index to 0..k-1
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()

# ------------------------------------------------------------
# stage (3): recursive bisection (python3-safe)
# (marcos lópez de prado, 2018, snippet 16.3 & 16.4)
# ------------------------------------------------------------

def _get_ivp(cov: pd.DataFrame) -> np.ndarray:
    # marcos lópez de prado (2018, ch.16, snippet 16.4 -> 'getIVP()')
    # compute inverse-variance portfolio weights within a cluster
    diag = np.diag(cov.values)
    
    # debug steps
    if np.any(~np.isfinite(diag)):
        raise ValueError("hrp: non-finite variances in covariance diagonal")
    if np.any(diag <= 0):
        # hrp assumes variances are positive; non-positive means broken input window
        raise ValueError("hrp: non-positive variance encountered in covariance diagonal")

    inv_diag = 1.0 / diag
    w = inv_diag / inv_diag.sum()
    
    return w.reshape(-1, 1) # shape adjustment for matrix algebra 


def _get_cluster_var(cov: pd.DataFrame, c_items: list[int], assets: list[str]) -> float:
    # compute variance per cluster using explicit asset name mapping
    # marcos lópez de prado (2018, ch.16, snippet 16.4 -> 'getClusterVar()')
    cluster_assets = [assets[i] for i in c_items] # index by asset name
    cov_slice = cov.loc[cluster_assets, cluster_assets] # matrix slice

    w = _get_ivp(cov_slice) 
    v = float(w.T @ cov_slice.values @ w)
    return v


def _get_rec_bipart(cov: pd.DataFrame, sort_ix: list[int], assets: list[str]) -> pd.Series:
    """
    recursive bisection allocation 
    - operate on the quasi-diagonal leaf ordering (sort_ix)
    - split into left/right halves recursively
    - allocate between halves using inverse cluster variances

    reference: marcos lópez de prado, 2018, snippet 16.3 & 16.4 -> 'getRecBipart()'
    """
    # compute HRP alloc
    w = pd.Series(1.0, index=sort_ix, dtype=float)
    c_items: List[List[int]] = [sort_ix[:] ]  # initialize all items in one cluster 

    while len(c_items) > 0:
        # bi-section step: expand each cluster into two halves if size > 1
        new_clusters: List[List[int]] = []
        for cluster in c_items:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2  # integer bisection of the ordered cluster (de prado, 2018)
            new_clusters.append(cluster[0:split])
            new_clusters.append(cluster[split:len(cluster)])
        c_items = new_clusters

        # parse in pairs
        for i in range(0, len(c_items), 2):
            c0 = c_items[i]       # cluster 1
            c1 = c_items[i + 1]   # cluster 2

            c_var0 = _get_cluster_var(cov, c0, assets)
            c_var1 = _get_cluster_var(cov, c1, assets)

            denom = c_var0 + c_var1
            if not np.isfinite(denom) or denom <= 0:
                raise ValueError("hrp: invalid cluster variance in recursive bisection")

            alpha = 1.0 - c_var0 / denom  # = c_var1 / (c_var0 + c_var1)

            w.loc[c0] *= alpha           # weight 1
            w.loc[c1] *= (1.0 - alpha)   # weight 2  

    return w

# ------------------------------------------------------------
# iterface integration
# ------------------------------------------------------------

def compute_weights(
    x: pd.DataFrame,
    cfg: Optional[HRPConfig] = None,
) -> pd.Series:
    # stage (1): tree clustering
    out = tree_clustering(x, cfg=cfg)

    cov: pd.DataFrame = out["cov"]
    corr: pd.DataFrame = out["corr"]
    link: np.ndarray = out["link"]
    assets: list[str] = list(corr.columns)

    # stage (2): quasi-diagonalization
    sort_ix = _get_quasi_diag(link)

    # stage (3): recursive bisection
    w_leaf = _get_rec_bipart(cov=cov, sort_ix=sort_ix, assets=assets)

    # map leaf-index weights back to asset labels
    w = pd.Series(index=assets, dtype=float)
    for leaf_idx, val in w_leaf.items():
        asset = assets[int(leaf_idx)]
        w.loc[asset] = float(val)

    # align to original universe and normalize
    w = w.reindex(x.columns).dropna()
    s = float(w.sum())
    if not np.isfinite(s) or s <= (cfg.eps if cfg else 1e-12):
        raise ValueError("hrp: invalid weight vector after allocation")
    w = _normalize_long_only(w)
    
    return w

# ------------------------------------------------------------
# strategy class integration (base interface)
# ------------------------------------------------------------

class HierarchicalRiskParity(BaseStrategy):
    # short name used in output folders and logs
    name: str = "hrp"

    def __init__(self, cfg: Optional[HRPConfig] = None):
        self.cfg = cfg or HRPConfig()

    def get_weights(
        self,
        window_returns: pd.DataFrame,
        window_rf: Optional[pd.Series] = None,  # unused by hrp
    ) -> StrategyResult:
        w = compute_weights(window_returns, cfg=self.cfg)
        return StrategyResult(weights=w)
