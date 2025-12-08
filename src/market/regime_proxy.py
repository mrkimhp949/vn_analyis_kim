# -*- coding: utf-8 -*-
"""
Market Regime Proxy - Caching wrapper for MarketRegimeDetector

Provides a singleton pattern with TTL-based caching to avoid
expensive regime calculations on every request.

Author: Trading Bot Team
Version: 1.1.0
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Type

import pandas as pd

logger = logging.getLogger(__name__)

# Fix encoding for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        os.environ["PYTHONIOENCODING"] = "utf-8"
    except AttributeError:
        pass


def safe_print(message: str) -> None:
    """Print safely handling Unicode encoding errors."""
    try:
        print(message)
    except UnicodeEncodeError:
        clean_message = "".join(char for char in message if ord(char) < 128)
        print(clean_message)


class SimpleCache:
    """
    Simple in-memory cache with TTL expiration.

    Attributes:
        default_ttl: Default time-to-live in seconds
    """

    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/missing
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() > entry["expires_at"]:
            # Expired - remove and return None
            del self._cache[key]
            return None

        return entry["value"]

    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> None:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            timeout: TTL in seconds (uses default if not specified)
        """
        ttl = timeout if timeout is not None else self.default_ttl
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "created_at": time.time(),
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def remove(self, key: str) -> bool:
        """
        Remove specific cache entry.

        Args:
            key: Cache key to remove

        Returns:
            True if entry was removed, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False


class ProxyMarketRegimeAnalyzer:
    """
    Proxy wrapper for MarketRegimeDetector with caching.

    Singleton pattern ensures only one instance exists.
    Caching reduces expensive regime calculations.
    """

    _instance: Optional["ProxyMarketRegimeAnalyzer"] = None
    _cache: SimpleCache = SimpleCache(default_ttl=3600)  # 1 hour default

    def __new__(cls, *args, **kwargs) -> "ProxyMarketRegimeAnalyzer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        analyzer_class: Optional[Type] = None,
        cache_ttl: int = 3600,
        **kwargs,
    ):
        """
        Initialize the proxy analyzer.

        Args:
            analyzer_class: Optional custom analyzer class
            cache_ttl: Cache time-to-live in seconds
            **kwargs: Arguments passed to analyzer
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = True
        self._cache = SimpleCache(default_ttl=cache_ttl)

        # Initialize the actual analyzer
        if analyzer_class:
            self.analyzer = analyzer_class(**kwargs)
        else:
            from src.market.regime_detector import MarketRegimeDetector

            self.analyzer = MarketRegimeDetector(**kwargs)
            logger.info("📊 Initialized MarketRegimeDetector")

        logger.info("✅ ProxyMarketRegimeAnalyzer initialized")

    def analyze_market_regime(self, vnindex_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyze market regime with caching.

        Args:
            vnindex_df: Optional pre-loaded VNINDEX data

        Returns:
            Dict with regime analysis results
        """
        cache_key = f"market_regime_{datetime.now().strftime('%Y-%m-%d_%H')}"

        # Check cache
        cached_data = self._cache.get(cache_key)
        if cached_data is not None:
            logger.debug("✅ Returning cached market regime")
            return cached_data

        logger.info("🔧 Analyzing market regime (cache miss)...")

        try:
            if self.analyzer is None:
                return self._default_regime("No analyzer available")

            # Call analyzer with vnindex_df if supported
            if hasattr(self.analyzer, "analyze_market_regime"):
                regime = self.analyzer.analyze_market_regime(vnindex_df=vnindex_df)
            else:
                regime = self._default_regime("Analyzer missing analyze_market_regime method")

            # Cache result
            self._cache.set(cache_key, regime)
            return regime

        except Exception as e:
            logger.error(f"Regime analysis failed: {e}", exc_info=True)
            return self._default_regime(f"Error: {str(e)}")

    def invalidate_cache(self) -> None:
        """Invalidate all cached regime data."""
        self._cache.clear()
        logger.info("🗑️ Regime cache invalidated")

    @staticmethod
    def _default_regime(message: str) -> Dict:
        """
        Return default cautious regime.

        Args:
            message: Reason for using default

        Returns:
            Cautious SIDEWAYS regime dict
        """
        return {
            "regime": "SIDEWAYS",
            "confidence": 20,
            "tradeable": False,
            "details": {"reason": message},
            "message": message,
        }

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance. Useful for testing."""
        cls._instance = None


def get_proxy_analyzer(**kwargs) -> ProxyMarketRegimeAnalyzer:
    """
    Get singleton instance of ProxyMarketRegimeAnalyzer.

    Args:
        **kwargs: Arguments passed to ProxyMarketRegimeAnalyzer

    Returns:
        ProxyMarketRegimeAnalyzer singleton
    """
    return ProxyMarketRegimeAnalyzer(**kwargs)


def main() -> None:
    """Demo/test the proxy analyzer."""
    analyzer = get_proxy_analyzer()
    result = analyzer.analyze_market_regime()
    safe_print(f"Regime: {result}")


if __name__ == "__main__":
    main()
