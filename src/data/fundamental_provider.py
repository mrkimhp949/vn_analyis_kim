# -*- coding: utf-8 -*-
"""
Fundamental Data Provider
Provides fundamental data (P/E, P/B, ROE, Debt Ratio, etc.) for Vietnamese stocks
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class FundamentalDataProvider:
    """
    Provider for fundamental data

    Data sources (in priority order):
    1. Local cache (memory)
    2. Local CSV file (data/fundamental_data.csv)
    3. TCBS API (free, no auth required)
    4. VNDirect API
    5. SSI API
    6. Fallback to None
    """

    def __init__(self, cache_ttl_hours: int = 24, csv_file: str = "data/fundamental_data.csv"):
        """
        Args:
            cache_ttl_hours: Cache time-to-live in hours
            csv_file: Path to CSV file with fundamental data
        """
        self.cache_ttl_hours = cache_ttl_hours
        self.csv_file = csv_file
        self._cache = {}  # {symbol: {data: dict, timestamp: datetime}}
        self._csv_data = {}  # {symbol: data dict}
        self._load_csv_data()

    def _load_csv_data(self):
        """Load fundamental data from CSV file"""
        try:
            import os

            if not os.path.exists(self.csv_file):
                logger.debug(f"CSV file not found: {self.csv_file}")
                return

            df = pd.read_csv(self.csv_file)

            for _, row in df.iterrows():
                symbol = row["symbol"]
                self._csv_data[symbol] = {
                    "pe_ratio": row.get("pe_ratio"),
                    "pb_ratio": row.get("pb_ratio"),
                    "roe": row.get("roe"),
                    "debt_ratio": row.get("debt_ratio"),
                    "market_cap": row.get("market_cap"),
                    "eps": row.get("eps"),
                    "profit_margin": row.get("profit_margin"),
                    "timestamp": datetime.now(),
                }

            logger.info(f"Loaded fundamental data for {len(self._csv_data)} symbols from CSV")
        except Exception as e:
            logger.warning(f"Error loading CSV data: {e}")

    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Get fundamental data for a symbol

        Returns:
            Dict with fundamental metrics or None if unavailable
            {
                'pe_ratio': float,
                'pb_ratio': float,
                'roe': float,  # Return on Equity (%)
                'debt_ratio': float,  # Debt/Equity
                'market_cap': float,  # VND
                'eps': float,  # Earnings per share
                'bvps': float,  # Book value per share
                'revenue_growth': float,  # YoY revenue growth (%)
                'profit_margin': float,  # Net profit margin (%)
                'timestamp': datetime
            }
        """
        # Check memory cache first
        if symbol in self._cache:
            cached = self._cache[symbol]
            age = datetime.now() - cached["timestamp"]
            if age.total_seconds() < self.cache_ttl_hours * 3600:
                logger.debug(
                    f"[{symbol}] Using cached fundamental data (age: {age.total_seconds()/3600:.1f}h)"
                )
                return cached["data"]

        # Try CSV data
        if symbol in self._csv_data:
            logger.debug(f"[{symbol}] Using CSV fundamental data")
            data = self._csv_data[symbol].copy()
            # Cache it
            self._cache[symbol] = {"data": data, "timestamp": datetime.now()}
            return data

        # Try to fetch from API
        data = self._fetch_from_tcbs(symbol)

        if data:
            # Cache the data
            self._cache[symbol] = {"data": data, "timestamp": datetime.now()}
            return data

        logger.warning(f"[{symbol}] No fundamental data available")
        return None

    def _fetch_from_tcbs(self, symbol: str) -> Optional[Dict]:
        """
        Fetch fundamental data from TCBS API

        TCBS provides multiple endpoints:
        1. Financial ratios: /tcanalysis/v1/finance/{symbol}/financialratio
        2. Company overview: /stock/stock-realtime/{symbol}/overview
        """
        try:
            import requests

            # Try financial ratios endpoint first
            url1 = f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{symbol}/financialratio"
            params1 = {"yearly": 0, "isAll": True}  # Quarterly data

            response1 = requests.get(url1, params=params1, timeout=5)

            fundamentals = {
                "pe_ratio": None,
                "pb_ratio": None,
                "roe": None,
                "debt_ratio": None,
                "market_cap": None,
                "eps": None,
                "bvps": None,
                "revenue_growth": None,
                "profit_margin": None,
                "timestamp": datetime.now(),
            }

            # Parse financial ratios
            if response1.status_code == 200:
                data1 = response1.json()
                if data1 and len(data1) > 0:
                    latest = data1[0]
                    fundamentals["roe"] = latest.get("roe")
                    fundamentals["debt_ratio"] = latest.get("debtOnEquity")
                    fundamentals["eps"] = latest.get("eps")
                    fundamentals["bvps"] = latest.get("bookValuePerShare")
                    fundamentals["revenue_growth"] = latest.get("revenueGrowth")
                    fundamentals["profit_margin"] = latest.get("netProfitMargin")

            # Try company overview endpoint for P/E, P/B, Market Cap
            url2 = f"https://apipubaws.tcbs.com.vn/stock/stock-realtime/{symbol}/overview"
            response2 = requests.get(url2, timeout=5)

            if response2.status_code == 200:
                data2 = response2.json()
                if data2:
                    fundamentals["pe_ratio"] = data2.get("pe")
                    fundamentals["pb_ratio"] = data2.get("pb")
                    fundamentals["market_cap"] = data2.get("marketCap")

                    # If ROE not from ratios, try from overview
                    if fundamentals["roe"] is None:
                        fundamentals["roe"] = data2.get("roe")

            # Check if we got any data
            has_data = any(v is not None for k, v in fundamentals.items() if k != "timestamp")

            if not has_data:
                logger.debug(f"[{symbol}] No fundamental data available from TCBS")
                return None

            logger.debug(
                f"[{symbol}] Fetched fundamentals: "
                f"P/E={fundamentals['pe_ratio']}, "
                f"P/B={fundamentals['pb_ratio']}, "
                f"ROE={fundamentals['roe']}"
            )

            return fundamentals

        except Exception as e:
            logger.warning(f"[{symbol}] Error fetching from TCBS: {e}")
            return None

    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()
        logger.info("Fundamental data cache cleared")


# Singleton instance
_fundamental_provider = None


def get_fundamental_provider() -> FundamentalDataProvider:
    """Get singleton fundamental provider"""
    global _fundamental_provider
    if _fundamental_provider is None:
        _fundamental_provider = FundamentalDataProvider()
    return _fundamental_provider
