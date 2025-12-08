# -*- coding: utf-8 -*-
"""
Correlation Cache

Thread-safe LRU cache for correlation values.
"""

import logging
import threading
import time
from typing import Dict, Optional, Tuple

from .constants import PositionSizingConstants

logger = logging.getLogger(__name__)


class CorrelationCache:
    """Thread-safe LRU cache for correlation values."""

    def __init__(
        self,
        ttl: int = PositionSizingConstants.CORRELATION_CACHE_TTL,
        maxsize: int = PositionSizingConstants.CORRELATION_CACHE_MAXSIZE,
    ):
        self._cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def _make_key(self, symbol1: str, symbol2: str) -> Tuple[str, str]:
        """Create order-independent cache key."""
        return tuple(sorted([symbol1, symbol2]))

    def get(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Get cached correlation if valid."""
        key = self._make_key(symbol1, symbol2)

        with self._lock:
            if key in self._cache:
                corr, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    return corr
                # Expired - remove
                del self._cache[key]

            self._misses += 1
            return None

    def set(self, symbol1: str, symbol2: str, correlation: float) -> None:
        """Store correlation in cache."""
        key = self._make_key(symbol1, symbol2)

        with self._lock:
            self._cache[key] = (correlation, time.time())
            self._prune_if_needed()

    def _prune_if_needed(self) -> None:
        """Prune cache if over maxsize (must hold lock)."""
        if len(self._cache) <= self._maxsize:
            return

        current_time = time.time()

        # Remove expired first
        expired = [k for k, (_, ts) in self._cache.items() if current_time - ts > self._ttl]
        for k in expired:
            del self._cache[k]

        # If still over, remove oldest
        while len(self._cache) > self._maxsize:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
