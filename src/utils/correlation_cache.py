"""
Correlation Matrix Caching
Simple cache with TTL to avoid redundant correlation calculations
"""

import logging
import time
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationCache:
    """
    Simple TTL-based cache for correlation matrices

    Features:
    - Thread-safe operations
    - TTL-based expiration (default: 1 hour)
    - Automatic cleanup of expired entries
    - Cache key based on symbols + lookback period
    """

    def __init__(self, ttl_seconds: int = 3600):  # 1 hour default
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[pd.DataFrame, float]] = {}  # {key: (matrix, timestamp)}
        self._lock = RLock()
        logger.info(f"✅ Correlation cache initialized with TTL={ttl_seconds}s")

    def get(self, symbols: list[str], lookback: int) -> Optional[pd.DataFrame]:
        """
        Get cached correlation matrix

        Args:
            symbols: List of symbols
            lookback: Lookback period in days

        Returns:
            Cached correlation matrix or None if not found/expired
        """
        with self._lock:
            cache_key = self._make_key(symbols, lookback)

            if cache_key not in self.cache:
                return None

            matrix, timestamp = self.cache[cache_key]

            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                logger.debug(f"🗑️ Cache expired for {cache_key}")
                del self.cache[cache_key]
                return None

            logger.debug(f"✅ Cache hit for {cache_key}")
            return matrix

    def set(self, symbols: list[str], lookback: int, matrix: pd.DataFrame):
        """
        Cache correlation matrix

        Args:
            symbols: List of symbols
            lookback: Lookback period in days
            matrix: Correlation matrix to cache
        """
        with self._lock:
            cache_key = self._make_key(symbols, lookback)
            self.cache[cache_key] = (matrix, time.time())
            logger.debug(f"💾 Cached correlation matrix for {cache_key}")

            # Cleanup old entries (every 10 cache sets)
            if len(self.cache) % 10 == 0:
                self._cleanup_expired()

    def _make_key(self, symbols: list[str], lookback: int) -> str:
        """
        Create cache key from symbols and lookback

        Args:
            symbols: List of symbols
            lookback: Lookback period

        Returns:
            Cache key string
        """
        # Sort symbols for consistent key
        sorted_symbols = sorted(symbols)
        return f"{','.join(sorted_symbols)}_{lookback}"

    def _cleanup_expired(self):
        """Remove expired entries from cache"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, (_, timestamp) in self.cache.items()
                if current_time - timestamp > self.ttl_seconds
            ]

            for key in expired_keys:
                del self.cache[key]

            if expired_keys:
                logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"🧹 Cleared {count} cache entries")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            current_time = time.time()
            valid_entries = sum(
                1
                for _, timestamp in self.cache.values()
                if current_time - timestamp <= self.ttl_seconds
            )
            expired_entries = len(self.cache) - valid_entries

            return {
                "total_entries": len(self.cache),
                "valid_entries": valid_entries,
                "expired_entries": expired_entries,
                "ttl_seconds": self.ttl_seconds,
            }


# Global singleton
_correlation_cache = None


def get_correlation_cache(ttl_seconds: int = 3600) -> CorrelationCache:
    """Get singleton correlation cache instance"""
    global _correlation_cache
    if _correlation_cache is None:
        _correlation_cache = CorrelationCache(ttl_seconds=ttl_seconds)
    return _correlation_cache


# Test
if __name__ == "__main__":
    print("Testing Correlation Cache...")

    cache = CorrelationCache(ttl_seconds=2)  # 2 second TTL for testing

    # Test 1: Cache miss
    print("\n1️⃣ Test cache miss:")
    result = cache.get(["VNM", "HPG", "FPT"], 60)
    print(f"  Result: {result}")

    # Test 2: Cache set and hit
    print("\n2️⃣ Test cache set and hit:")
    test_matrix = pd.DataFrame([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    cache.set(["VNM", "HPG", "FPT"], 60, test_matrix)
    result = cache.get(["VNM", "HPG", "FPT"], 60)
    print(f"  Result shape: {result.shape if result is not None else None}")

    # Test 3: Cache expiration
    print("\n3️⃣ Test cache expiration:")
    print("  Waiting 3 seconds...")
    time.sleep(3)
    result = cache.get(["VNM", "HPG", "FPT"], 60)
    print(f"  Result after expiration: {result}")

    # Test 4: Cache stats
    print("\n4️⃣ Cache stats:")
    stats = cache.get_stats()
    print(f"  Stats: {stats}")

    print("\n✅ Test completed!")
