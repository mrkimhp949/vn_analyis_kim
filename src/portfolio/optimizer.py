"""Advanced portfolio optimization utilities."""

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from risk_metrics import load_returns_dataframe

logger = logging.getLogger(__name__)

try:  # Optional SciPy dependency
    from scipy.cluster.hierarchy import linkage  # type: ignore
    from scipy.spatial.distance import squareform  # type: ignore

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False
    logger.info("SciPy not available. HRP sẽ fallback sang Risk Budgeting.")


@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    method: str
    annualized_volatility: Optional[float] = None
    notes: Optional[str] = None


class PortfolioOptimizer:
    def __init__(self, lookback: int = 90):
        self.lookback = lookback

    def optimize_weights(
        self,
        symbols: Iterable[str],
        method: str = "hrp",
    ) -> Optional[OptimizationResult]:
        symbols = [s for s in symbols]
        if len(symbols) < 2:
            return None

        returns_df = load_returns_dataframe(symbols, lookback=self.lookback)
        if returns_df.empty:
            return None

        method = method.lower()
        if method == "hrp" and SCIPY_AVAILABLE:
            weights = self._hrp_allocation(returns_df)
            chosen_method = "HRP"
        else:
            weights = self._risk_budgeting_allocation(returns_df)
            chosen_method = (
                "RiskBudgeting" if method != "hrp" else "RiskBudgeting(Fallback)"
            )

        if weights is None or weights.empty:
            return None

        cov = returns_df.cov()
        annualized_vol = float(
            np.sqrt(np.dot(weights.values, cov @ weights.values)) * np.sqrt(252)
        )

        return OptimizationResult(
            weights=weights.to_dict(),
            method=chosen_method,
            annualized_volatility=annualized_vol,
        )

    @staticmethod
    def _risk_budgeting_allocation(returns_df: pd.DataFrame) -> pd.Series:
        vol = returns_df.std()
        vol = vol.replace(0, np.nan)
        inv_vol = 1.0 / vol
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
        if inv_vol.empty:
            return pd.Series(dtype=float)
        weights = inv_vol / inv_vol.sum()
        return weights.reindex(returns_df.columns).fillna(0.0)

    def _hrp_allocation(self, returns_df: pd.DataFrame) -> pd.Series:
        cov = returns_df.cov()
        corr = returns_df.corr()
        dist = np.sqrt(0.5 * (1 - corr))
        dist = dist.fillna(0)

        condensed = squareform(dist.values, checks=False)
        link = linkage(condensed, method="single")
        sort_ix = self._get_quasi_diag(link)
        ordered_symbols = [returns_df.columns[i] for i in sort_ix]
        return self._recursive_bisection(cov, ordered_symbols)

    @staticmethod
    def _get_quasi_diag(link) -> List[int]:
        link = link.astype(int)
        n = int(link[-1, 3])
        queue = [int(link[-1, 0]), int(link[-1, 1])]
        order: List[int] = []
        while queue:
            idx = queue.pop(0)
            if idx < n:
                order.append(idx)
            else:
                sub_idx = idx - n
                queue.insert(0, int(link[sub_idx, 1]))
                queue.insert(0, int(link[sub_idx, 0]))
        return order

    @staticmethod
    def _get_cluster_variance(cov: pd.DataFrame, cluster: List[str]) -> float:
        cov_slice = cov.loc[cluster, cluster]
        diag = np.diag(cov_slice.values)
        diag = np.where(diag <= 1e-8, 1e-8, diag)
        ivp = 1.0 / diag
        ivp = ivp / ivp.sum()
        w = ivp.reshape(-1, 1)
        variance = float(w.T @ cov_slice.values @ w)
        return variance

    def _recursive_bisection(
        self, cov: pd.DataFrame, ordered_symbols: List[str]
    ) -> pd.Series:
        weights = pd.Series(1.0, index=ordered_symbols)
        clusters = [ordered_symbols]

        while clusters:
            cluster = clusters.pop(0)
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]

            var_left = self._get_cluster_variance(cov, left)
            var_right = self._get_cluster_variance(cov, right)
            total = var_left + var_right
            if total == 0:
                alloc_left = alloc_right = 0.5
            else:
                alloc_left = 1 - var_left / total
                alloc_right = 1 - alloc_left

            weights[left] *= alloc_left
            weights[right] *= alloc_right

            clusters.append(left)
            clusters.append(right)

        weights = weights / weights.sum()
        return weights
