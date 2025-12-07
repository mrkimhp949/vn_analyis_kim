"""
VN30 Symbols Fetcher - IMPROVED v5.1

Fetches VN30 symbols from API instead of hardcoding.
VN30 index is rebalanced quarterly (January, April, July, October).

ADDRESSES RISK: Hardcoded VN30 symbols need manual update quarterly.

Author: Trading Bot Team
Version: 5.1.0
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Set

import requests

logger = logging.getLogger(__name__)

# Cache settings
VN30_CACHE_FILE = "data_cache/vn30_symbols.json"
VN30_CACHE_TTL_HOURS = 24  # Refresh daily
VN30_CACHE_MAX_AGE_DAYS = 7  # Force refresh after 7 days

# Fallback VN30 symbols (Q4 2024) - used if API fails
VN30_FALLBACK_SYMBOLS = {
    "ACB",
    "BCM",
    "BID",
    "BVH",
    "CTG",
    "FPT",
    "GAS",
    "GVR",
    "HDB",
    "HPG",
    "MBB",
    "MSN",
    "MWG",
    "PLX",
    "POW",
    "SAB",
    "SHB",
    "SSB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VNM",
    "VPB",
    "VRE",
}

# API endpoints for VN30 data
VN30_API_SOURCES = [
    {
        "name": "SSI",
        "url": "https://iboard.ssi.com.vn/dchart/api/1.1/defaultAllStocks",
        "parser": "_parse_ssi_response",
    },
    {
        "name": "TCBS",
        "url": "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/second-tc-price?tickers=VN30",
        "parser": "_parse_tcbs_response",
    },
    {
        "name": "VNDirect",
        "url": "https://finfo-api.vndirect.com.vn/v4/index_components?indexCode=VN30",
        "parser": "_parse_vndirect_response",
    },
]


class VN30Fetcher:
    """
    Fetches and caches VN30 symbols from multiple API sources.

    Features:
    - Multiple API sources with fallback
    - Local caching to reduce API calls
    - Automatic cache invalidation
    - Fallback to hardcoded list if all APIs fail
    """

    def __init__(
        self,
        cache_file: str = VN30_CACHE_FILE,
        cache_ttl_hours: int = VN30_CACHE_TTL_HOURS,
    ):
        """
        Initialize VN30 fetcher.

        Args:
            cache_file: Path to cache file
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.cache_file = cache_file
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self._symbols: Optional[Set[str]] = None
        self._last_fetch: Optional[datetime] = None

        # Ensure cache directory exists
        cache_dir = os.path.dirname(cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    def get_vn30_symbols(self, force_refresh: bool = False) -> Set[str]:
        """
        Get VN30 symbols with caching.

        Args:
            force_refresh: Force API refresh ignoring cache

        Returns:
            Set of VN30 symbol strings
        """
        # Check memory cache
        if not force_refresh and self._symbols and self._is_cache_valid():
            return self._symbols

        # Check file cache
        if not force_refresh:
            cached = self._load_from_cache()
            if cached:
                self._symbols = cached
                return cached

        # Fetch from API
        symbols = self._fetch_from_api()

        if symbols:
            self._symbols = symbols
            self._last_fetch = datetime.now()
            self._save_to_cache(symbols)
            logger.info(f"✅ VN30 symbols updated: {len(symbols)} symbols")
            return symbols

        # Fallback to hardcoded
        logger.warning("⚠️ Using fallback VN30 symbols (API unavailable)")
        self._symbols = VN30_FALLBACK_SYMBOLS.copy()
        return self._symbols

    def _is_cache_valid(self) -> bool:
        """Check if memory cache is still valid."""
        if not self._last_fetch:
            return False
        return datetime.now() - self._last_fetch < self.cache_ttl

    def _load_from_cache(self) -> Optional[Set[str]]:
        """Load symbols from file cache."""
        if not os.path.exists(self.cache_file):
            return None

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check cache age
            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            max_age = timedelta(days=VN30_CACHE_MAX_AGE_DAYS)

            if datetime.now() - cached_at > max_age:
                logger.info("📅 VN30 cache expired, will refresh")
                return None

            symbols = set(data.get("symbols", []))
            if len(symbols) >= 25:  # VN30 should have ~30 symbols
                self._last_fetch = cached_at
                logger.debug(f"📂 Loaded {len(symbols)} VN30 symbols from cache")
                return symbols

        except Exception as e:
            logger.warning(f"Failed to load VN30 cache: {e}")

        return None

    def _save_to_cache(self, symbols: Set[str]):
        """Save symbols to file cache."""
        try:
            data = {
                "symbols": sorted(list(symbols)),
                "count": len(symbols),
                "cached_at": datetime.now().isoformat(),
                "source": "api",
            }

            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"💾 Saved {len(symbols)} VN30 symbols to cache")

        except Exception as e:
            logger.warning(f"Failed to save VN30 cache: {e}")

    def _fetch_from_api(self) -> Optional[Set[str]]:
        """Fetch VN30 symbols from API sources."""
        for source in VN30_API_SOURCES:
            try:
                logger.debug(f"🔄 Fetching VN30 from {source['name']}...")

                response = requests.get(
                    source["url"],
                    timeout=10,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "application/json",
                    },
                )

                if response.status_code == 200:
                    parser = getattr(self, source["parser"], None)
                    if parser:
                        symbols = parser(response.json())
                        if symbols and len(symbols) >= 25:
                            logger.info(
                                f"✅ Fetched {len(symbols)} VN30 symbols from {source['name']}"
                            )
                            return symbols

            except requests.RequestException as e:
                logger.debug(f"API {source['name']} failed: {e}")
            except Exception as e:
                logger.debug(f"Parser {source['name']} failed: {e}")

        return None

    def _parse_ssi_response(self, data: dict) -> Optional[Set[str]]:
        """Parse SSI API response."""
        try:
            symbols = set()
            stocks = data.get("data", [])

            for stock in stocks:
                if stock.get("indexCode") == "VN30":
                    symbol = stock.get("stockSymbol", "").upper()
                    if symbol and len(symbol) == 3:
                        symbols.add(symbol)

            return symbols if symbols else None

        except Exception:
            return None

    def _parse_tcbs_response(self, data: dict) -> Optional[Set[str]]:
        """Parse TCBS API response."""
        try:
            symbols = set()

            # TCBS returns list of tickers
            if isinstance(data, list):
                for item in data:
                    symbol = item.get("ticker", "").upper()
                    if symbol and len(symbol) == 3:
                        symbols.add(symbol)

            return symbols if symbols else None

        except Exception:
            return None

    def _parse_vndirect_response(self, data: dict) -> Optional[Set[str]]:
        """Parse VNDirect API response."""
        try:
            symbols = set()
            components = data.get("data", [])

            for comp in components:
                symbol = comp.get("code", "").upper()
                if symbol and len(symbol) == 3:
                    symbols.add(symbol)

            return symbols if symbols else None

        except Exception:
            return None

    def is_vn30(self, symbol: str) -> bool:
        """
        Check if a symbol is in VN30 index.

        Args:
            symbol: Stock symbol to check

        Returns:
            True if symbol is in VN30
        """
        symbols = self.get_vn30_symbols()
        return symbol.upper() in symbols

    def get_vn30_changes(self) -> dict:
        """
        Detect changes in VN30 composition.

        Useful for quarterly rebalancing detection.

        Returns:
            Dict with added, removed, and current symbols
        """
        current = self.get_vn30_symbols(force_refresh=True)

        # Compare with fallback (last known composition)
        added = current - VN30_FALLBACK_SYMBOLS
        removed = VN30_FALLBACK_SYMBOLS - current

        return {
            "current": sorted(list(current)),
            "count": len(current),
            "added": sorted(list(added)) if added else [],
            "removed": sorted(list(removed)) if removed else [],
            "has_changes": bool(added or removed),
            "checked_at": datetime.now().isoformat(),
        }


# Singleton instance
_vn30_fetcher: Optional[VN30Fetcher] = None


def get_vn30_fetcher() -> VN30Fetcher:
    """Get singleton VN30 fetcher instance."""
    global _vn30_fetcher
    if _vn30_fetcher is None:
        _vn30_fetcher = VN30Fetcher()
    return _vn30_fetcher


def get_vn30_symbols(force_refresh: bool = False) -> Set[str]:
    """
    Convenience function to get VN30 symbols.

    Args:
        force_refresh: Force API refresh

    Returns:
        Set of VN30 symbols
    """
    return get_vn30_fetcher().get_vn30_symbols(force_refresh)


def is_vn30_symbol(symbol: str) -> bool:
    """
    Check if symbol is in VN30.

    Args:
        symbol: Stock symbol

    Returns:
        True if in VN30
    """
    return get_vn30_fetcher().is_vn30(symbol)


# CLI for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("Testing VN30 Fetcher...")

    fetcher = VN30Fetcher()

    # Get symbols
    symbols = fetcher.get_vn30_symbols()
    print(f"\n✅ VN30 Symbols ({len(symbols)}):")
    print(", ".join(sorted(symbols)))

    # Check changes
    changes = fetcher.get_vn30_changes()
    print(f"\n📊 VN30 Changes:")
    print(f"  Added: {changes['added']}")
    print(f"  Removed: {changes['removed']}")

    # Test is_vn30
    test_symbols = ["VNM", "HPG", "ABC", "XYZ"]
    print(f"\n🔍 Testing symbols:")
    for sym in test_symbols:
        print(f"  {sym}: {'✅ VN30' if fetcher.is_vn30(sym) else '❌ Not VN30'}")
