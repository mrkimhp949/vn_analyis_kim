# -*- coding: utf-8 -*-
"""
VNINDEX Cache Module - Singleton cache để tránh load VNINDEX nhiều lần

Giải quyết vấn đề:
- VNINDEX được load nhiều lần trong 1 cycle (entry scan, exit check, regime detection, ML signals)
- Mỗi lần load tốn ~1-2s API call
- Cache với TTL 5 phút giảm đáng kể số API calls

Usage:
    from src.data.vnindex_cache import get_cached_vnindex, invalidate_vnindex_cache
    
    # Get cached data (auto-refresh if expired)
    vnindex_df = get_cached_vnindex()
    
    # Force refresh (e.g., after market close)
    invalidate_vnindex_cache()
"""

import logging
import time
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# VNINDEX CACHE - Singleton
# =============================================================================
_vnindex_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 300,  # 5 minutes TTL
    "lookback": 250,  # Default lookback days
}


def get_cached_vnindex(
    lookback: int = 250,
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Get VNINDEX data with caching to avoid repeated API calls.

    Cache TTL: 5 minutes (configurable via set_vnindex_cache_ttl)

    Args:
        lookback: Number of days to look back (default: 250 for ~1 year)
        force_refresh: Force reload even if cache is valid

    Returns:
        VNINDEX DataFrame or None if load fails
    """
    from src.data.loader import load_data

    current_time = time.time()
    cache = _vnindex_cache

    # Check if cache is valid (and lookback matches)
    cache_valid = (
        cache["data"] is not None
        and (current_time - cache["timestamp"]) < cache["ttl"]
        and cache["lookback"] >= lookback
        and not force_refresh
    )

    if cache_valid:
        logger.debug(f"📦 Using cached VNINDEX data ({len(cache['data'])} bars)")
        return cache["data"]

    # Cache expired or empty - reload
    try:
        logger.debug(f"🔄 Loading fresh VNINDEX data (lookback={lookback})...")
        vnindex_df = load_data("VNINDEX", lookback=lookback, is_index=True)

        if vnindex_df is not None and not vnindex_df.empty and len(vnindex_df) >= 50:
            # Update cache
            cache["data"] = vnindex_df
            cache["timestamp"] = current_time
            cache["lookback"] = lookback
            logger.info(f"✅ VNINDEX cache updated: {len(vnindex_df)} bars")
            return vnindex_df
        else:
            bars = len(vnindex_df) if vnindex_df is not None else 0
            logger.warning(f"⚠️ VNINDEX data insufficient: {bars} bars")
            # Keep old cache if new data is bad
            if cache["data"] is not None:
                logger.info("📦 Using stale VNINDEX cache (new data insufficient)")
                return cache["data"]
            return None

    except Exception as e:
        logger.warning(f"⚠️ Failed to load VNINDEX: {e}")
        # Return stale cache if available
        if cache["data"] is not None:
            logger.info("📦 Using stale VNINDEX cache (load failed)")
            return cache["data"]
        return None


def invalidate_vnindex_cache():
    """Force invalidate VNINDEX cache (e.g., after market close)"""
    _vnindex_cache["data"] = None
    _vnindex_cache["timestamp"] = 0
    logger.info("🗑️ VNINDEX cache invalidated")


def set_vnindex_cache_ttl(ttl_seconds: int):
    """
    Set cache TTL (time-to-live) in seconds.

    Recommended values:
    - 60-120: During active trading (more responsive)
    - 300-600: During scanning/analysis (reduce API calls)
    - 3600: After market close (data won't change)

    Args:
        ttl_seconds: Cache TTL in seconds
    """
    _vnindex_cache["ttl"] = ttl_seconds
    logger.info(f"⏱️ VNINDEX cache TTL set to {ttl_seconds}s")


def get_vnindex_cache_info() -> dict:
    """Get cache status information for debugging"""
    cache = _vnindex_cache
    current_time = time.time()

    age = current_time - cache["timestamp"] if cache["timestamp"] > 0 else -1
    is_valid = cache["data"] is not None and age < cache["ttl"] and age >= 0

    return {
        "has_data": cache["data"] is not None,
        "bars": len(cache["data"]) if cache["data"] is not None else 0,
        "age_seconds": age,
        "ttl_seconds": cache["ttl"],
        "is_valid": is_valid,
        "lookback": cache["lookback"],
    }
