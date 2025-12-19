# -*- coding: utf-8 -*-
"""
SSI (Saigon Securities Inc.) Data Provider

Real-time and historical data from SSI APIs for Vietnam stock market.

Features:
- Real-time price data via iBoard API
- Foreign flow data
- Market depth / Order book
- Historical OHLCV data
- Index data (VNINDEX, VN30, HNX)

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from threading import RLock

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

SSI_IBOARD_BASE_URL = "https://iboard.ssi.com.vn"
SSI_GATEWAY_URL = "https://gateway-iboard.ssi.com.vn"
SSI_WGATEWAY_URL = "https://wgateway-iboard.ssi.com.vn"

CACHE_TTL_SECONDS = 60  # 1 minute for realtime data
CACHE_TTL_FOREIGN_FLOW = 300  # 5 minutes for foreign flow


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SSIStockData:
    """Real-time stock data from SSI"""

    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    value: float  # Trading value in VND

    # OHLC
    open: float
    high: float
    low: float
    close: float

    # Foreign trading
    foreign_buy_volume: int = 0
    foreign_sell_volume: int = 0
    foreign_buy_value: float = 0
    foreign_sell_value: float = 0
    foreign_room: float = 0  # Remaining room %

    # Order book
    bid_prices: List[float] = None
    bid_volumes: List[int] = None
    ask_prices: List[float] = None
    ask_volumes: List[int] = None

    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.bid_prices is None:
            self.bid_prices = []
        if self.bid_volumes is None:
            self.bid_volumes = []
        if self.ask_prices is None:
            self.ask_prices = []
        if self.ask_volumes is None:
            self.ask_volumes = []

    @property
    def foreign_net_volume(self) -> int:
        return self.foreign_buy_volume - self.foreign_sell_volume

    @property
    def foreign_net_value(self) -> float:
        return self.foreign_buy_value - self.foreign_sell_value


@dataclass
class SSIForeignFlow:
    """Foreign flow data from SSI"""

    date: str
    total_buy_value: float
    total_sell_value: float
    net_value: float
    top_buys: List[Tuple[str, float]]  # [(symbol, value), ...]
    top_sells: List[Tuple[str, float]]
    source: str = "SSI"
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# =============================================================================
# SSI PROVIDER CLASS
# =============================================================================


class SSIProvider:
    """
    SSI Data Provider for Vietnam Stock Market

    Provides real-time and historical data from SSI's public APIs.

    Usage:
        provider = SSIProvider()

        # Get real-time stock data
        data = provider.get_stock_data("VNM")

        # Get foreign flow
        flow = provider.get_foreign_flow_data()

        # Get all stocks
        all_stocks = provider.get_all_stocks()
    """

    def __init__(self, timeout: int = 10):
        """
        Initialize SSI Provider.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
                "Origin": SSI_IBOARD_BASE_URL,
                "Referer": f"{SSI_IBOARD_BASE_URL}/",
            }
        )

        # Cache
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._lock = RLock()

        logger.info("📊 SSI Provider initialized")

    def _is_cache_valid(self, key: str, ttl: int = CACHE_TTL_SECONDS) -> bool:
        """Check if cache is valid"""
        if key not in self._cache_time:
            return False
        age = (datetime.now() - self._cache_time[key]).total_seconds()
        return age < ttl

    def _get_cached(self, key: str, ttl: int = CACHE_TTL_SECONDS) -> Optional[Any]:
        """Get cached data if valid"""
        with self._lock:
            if self._is_cache_valid(key, ttl):
                return self._cache.get(key)
        return None

    def _set_cache(self, key: str, value: Any):
        """Set cache value"""
        with self._lock:
            self._cache[key] = value
            self._cache_time[key] = datetime.now()

    # =========================================================================
    # Stock Data APIs
    # =========================================================================

    def get_all_stocks(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get all stocks data from SSI iBoard.

        Returns:
            DataFrame with all stocks data
        """
        cache_key = "all_stocks"

        if not force_refresh:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        try:
            url = f"{SSI_IBOARD_BASE_URL}/dchart/api/1.1/defaultAllStocks"

            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Check for empty response before parsing JSON
            if not response.text or not response.text.strip():
                logger.warning("SSI returned empty response for all stocks")
                return None

            try:
                data = response.json()
            except ValueError as json_err:
                logger.warning(f"SSI returned invalid JSON for all stocks: {json_err}")
                return None

            if not data:
                logger.warning("Empty response from SSI all stocks API")
                return None

            # Parse data to DataFrame
            records = []
            for item in data:
                record = {
                    "symbol": item.get("code", ""),
                    "price": float(item.get("lastPrice", 0)) / 1000,  # Convert to VND
                    "change": float(item.get("change", 0)) / 1000,
                    "change_pct": float(item.get("changePc", 0)),
                    "volume": int(item.get("lot", 0)),
                    "value": float(item.get("value", 0)),
                    "open": float(item.get("openPrice", 0)) / 1000,
                    "high": float(item.get("highPrice", 0)) / 1000,
                    "low": float(item.get("lowPrice", 0)) / 1000,
                    "close": float(item.get("lastPrice", 0)) / 1000,
                    "ref_price": float(item.get("refPrice", 0)) / 1000,
                    "ceiling": float(item.get("ceilingPrice", 0)) / 1000,
                    "floor": float(item.get("floorPrice", 0)) / 1000,
                    "foreign_buy": int(item.get("fBuyVol", 0)),
                    "foreign_sell": int(item.get("fSellVol", 0)),
                    "foreign_room": float(item.get("fRoom", 0)),
                }
                records.append(record)

            df = pd.DataFrame(records)

            self._set_cache(cache_key, df)
            logger.info(f"✅ Fetched {len(df)} stocks from SSI")

            return df

        except requests.RequestException as req_err:
            logger.warning(f"SSI all stocks request failed: {req_err}")
            return None
        except Exception as e:
            logger.warning(f"SSI all stocks fetch failed: {e}")
            return None

    def get_stock_data(self, symbol: str, force_refresh: bool = False) -> Optional[SSIStockData]:
        """
        Get real-time data for a specific stock.

        Args:
            symbol: Stock symbol (e.g., "VNM", "FPT")
            force_refresh: Bypass cache

        Returns:
            SSIStockData or None
        """
        cache_key = f"stock_{symbol}"

        if not force_refresh:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        try:
            # Get from all stocks data (more efficient)
            all_stocks = self.get_all_stocks()

            if all_stocks is None or all_stocks.empty:
                return None

            row = all_stocks[all_stocks["symbol"] == symbol.upper()]

            if row.empty:
                logger.warning(f"Symbol {symbol} not found in SSI data")
                return None

            row = row.iloc[0]

            stock_data = SSIStockData(
                symbol=symbol.upper(),
                price=row["close"],
                change=row["change"],
                change_pct=row["change_pct"],
                volume=row["volume"],
                value=row["value"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                foreign_buy_volume=row["foreign_buy"],
                foreign_sell_volume=row["foreign_sell"],
                foreign_room=row["foreign_room"],
            )

            self._set_cache(cache_key, stock_data)
            return stock_data

        except Exception as e:
            logger.warning(f"SSI stock data fetch failed for {symbol}: {e}")
            return None

    # =========================================================================
    # Foreign Flow APIs
    # =========================================================================

    def get_foreign_flow_data(
        self,
        lookback_days: int = 20,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Get foreign flow data for the market.

        Args:
            lookback_days: Number of days to fetch
            force_refresh: Bypass cache

        Returns:
            DataFrame with columns: date, buy_value, sell_value, net_value
        """
        cache_key = f"foreign_flow_{lookback_days}"

        if not force_refresh:
            cached = self._get_cached(cache_key, CACHE_TTL_FOREIGN_FLOW)
            if cached is not None:
                return cached

        try:
            # Get all stocks with foreign data
            all_stocks = self.get_all_stocks(force_refresh=force_refresh)

            if all_stocks is None or all_stocks.empty:
                return None

            # Calculate market-wide foreign flow from current data
            today = datetime.now().strftime("%Y-%m-%d")

            # Aggregate foreign trading
            total_buy = all_stocks["foreign_buy"].sum()
            total_sell = all_stocks["foreign_sell"].sum()

            # Estimate values (foreign_buy is volume, need to multiply by price)
            all_stocks["foreign_buy_value"] = all_stocks["foreign_buy"] * all_stocks["close"] * 1000
            all_stocks["foreign_sell_value"] = (
                all_stocks["foreign_sell"] * all_stocks["close"] * 1000
            )

            total_buy_value = all_stocks["foreign_buy_value"].sum()
            total_sell_value = all_stocks["foreign_sell_value"].sum()

            # For historical data, we'd need a different API
            # This returns today's aggregated data as a single row
            df = pd.DataFrame(
                [
                    {
                        "date": today,
                        "buy_value": total_buy_value,
                        "sell_value": total_sell_value,
                        "net_value": total_buy_value - total_sell_value,
                        "buy_volume": total_buy,
                        "sell_volume": total_sell,
                    }
                ]
            )

            self._set_cache(cache_key, df)
            logger.info(
                f"✅ Fetched foreign flow from SSI: Net {(total_buy_value - total_sell_value)/1e9:.1f}B VND"
            )

            return df

        except Exception as e:
            logger.warning(f"SSI foreign flow fetch failed: {e}")
            return None

    def get_top_foreign_trades(
        self,
        top_n: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top foreign buying and selling stocks.

        Returns:
            {"top_buy": [(symbol, value), ...], "top_sell": [(symbol, value), ...]}
        """
        try:
            all_stocks = self.get_all_stocks()

            if all_stocks is None or all_stocks.empty:
                return {"top_buy": [], "top_sell": []}

            # Calculate net foreign value
            all_stocks["foreign_net_value"] = (
                (all_stocks["foreign_buy"] - all_stocks["foreign_sell"])
                * all_stocks["close"]
                * 1000
            )

            # Sort and get top
            top_buy = all_stocks.nlargest(top_n, "foreign_net_value")[
                ["symbol", "foreign_net_value"]
            ].values.tolist()
            top_buy = [(s, v) for s, v in top_buy if v > 0]

            top_sell = all_stocks.nsmallest(top_n, "foreign_net_value")[
                ["symbol", "foreign_net_value"]
            ].values.tolist()
            top_sell = [(s, abs(v)) for s, v in top_sell if v < 0]

            return {"top_buy": top_buy, "top_sell": top_sell}

        except Exception as e:
            logger.warning(f"SSI top foreign trades fetch failed: {e}")
            return {"top_buy": [], "top_sell": []}

    # =========================================================================
    # Index Data APIs
    # =========================================================================

    def get_index_data(self, index: str = "VNINDEX") -> Optional[Dict]:
        """
        Get index data.

        Args:
            index: Index symbol (VNINDEX, VN30, HNX, etc.)

        Returns:
            Dict with index data
        """
        try:
            url = f"{SSI_IBOARD_BASE_URL}/dchart/api/1.1/defaultIndex"

            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Check for empty response before parsing JSON
            if not response.text or not response.text.strip():
                logger.warning(f"SSI returned empty response for index {index}")
                return None

            # Try parsing JSON with proper error handling
            try:
                data = response.json()
            except ValueError as json_err:
                logger.warning(f"SSI returned invalid JSON for index {index}: {json_err}")
                return None

            # Ensure data is a list
            if not isinstance(data, list):
                logger.warning(f"SSI returned unexpected data type for index: {type(data)}")
                return None

            for item in data:
                if item.get("code", "").upper() == index.upper():
                    return {
                        "symbol": index,
                        "value": float(item.get("lastIndex", 0)),
                        "change": float(item.get("change", 0)),
                        "change_pct": float(item.get("changePc", 0)),
                        "volume": int(item.get("lot", 0)),
                        "value_traded": float(item.get("value", 0)),
                        "advance": int(item.get("advance", 0)),
                        "decline": int(item.get("decline", 0)),
                        "nochange": int(item.get("noChange", 0)),
                    }

            logger.warning(f"Index {index} not found in SSI data")
            return None

        except requests.RequestException as req_err:
            logger.warning(f"SSI index data request failed: {req_err}")
            return None
        except Exception as e:
            logger.error(f"SSI index data fetch failed: {e}")
            return None

    def get_market_breadth(self) -> Optional[Dict]:
        """
        Get market breadth data (advance/decline).

        Returns:
            Dict with advance, decline, nochange counts
        """
        index_data = self.get_index_data("VNINDEX")

        if index_data:
            return {
                "advance": index_data.get("advance", 0),
                "decline": index_data.get("decline", 0),
                "nochange": index_data.get("nochange", 0),
                "ad_ratio": (index_data.get("advance", 0) / max(1, index_data.get("decline", 1))),
            }

        return None

    # =========================================================================
    # Historical Data
    # =========================================================================

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # SSI historical data API
            url = f"{SSI_WGATEWAY_URL}/trading/chart/history"

            params = {
                "symbol": symbol.upper(),
                "resolution": "D",
                "from": int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()),
                "to": int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()),
            }

            response = self._session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            # Check for empty response before parsing JSON
            if not response.text or not response.text.strip():
                logger.warning(f"SSI returned empty response for {symbol} historical data")
                return None

            try:
                data = response.json()
            except ValueError as json_err:
                logger.warning(f"SSI returned invalid JSON for {symbol} historical: {json_err}")
                return None

            if data.get("s") != "ok":
                logger.warning(f"SSI historical data not available for {symbol}")
                return None

            df = pd.DataFrame(
                {
                    "time": pd.to_datetime(data.get("t", []), unit="s"),
                    "open": data.get("o", []),
                    "high": data.get("h", []),
                    "low": data.get("l", []),
                    "close": data.get("c", []),
                    "volume": data.get("v", []),
                }
            )

            df.set_index("time", inplace=True)

            logger.info(f"✅ Fetched {len(df)} historical bars for {symbol} from SSI")
            return df

        except requests.RequestException as req_err:
            logger.warning(f"SSI historical data request failed for {symbol}: {req_err}")
            return None
        except Exception as e:
            logger.warning(f"SSI historical data fetch failed for {symbol}: {e}")
            return None


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_ssi_provider: Optional[SSIProvider] = None
_lock = RLock()


def get_ssi_provider() -> SSIProvider:
    """Get singleton SSI provider instance."""
    global _ssi_provider

    with _lock:
        if _ssi_provider is None:
            _ssi_provider = SSIProvider()

    return _ssi_provider


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("🧪 TESTING SSI PROVIDER")
    print("=" * 70 + "\n")

    provider = get_ssi_provider()

    # Test 1: Get all stocks
    print("1️⃣ Testing get_all_stocks()...")
    stocks = provider.get_all_stocks()
    if stocks is not None:
        print(f"   ✅ Got {len(stocks)} stocks")
        print(f"   Sample: {stocks.head(3).to_dict('records')}")

    # Test 2: Get stock data
    print("\n2️⃣ Testing get_stock_data('VNM')...")
    vnm = provider.get_stock_data("VNM")
    if vnm:
        print(f"   ✅ VNM: {vnm.price:,.0f} VND ({vnm.change_pct:+.2f}%)")
        print(f"   Foreign: Buy {vnm.foreign_buy_volume:,}, Sell {vnm.foreign_sell_volume:,}")

    # Test 3: Get foreign flow
    print("\n3️⃣ Testing get_foreign_flow_data()...")
    flow = provider.get_foreign_flow_data()
    if flow is not None:
        print(f"   ✅ Foreign flow: {flow.to_dict('records')}")

    # Test 4: Get top foreign trades
    print("\n4️⃣ Testing get_top_foreign_trades()...")
    top_trades = provider.get_top_foreign_trades(top_n=5)
    print(f"   Top Buy: {top_trades['top_buy'][:3]}")
    print(f"   Top Sell: {top_trades['top_sell'][:3]}")

    # Test 5: Get index data
    print("\n5️⃣ Testing get_index_data()...")
    vnindex = provider.get_index_data("VNINDEX")
    if vnindex:
        print(f"   ✅ VNINDEX: {vnindex['value']:,.2f} ({vnindex['change_pct']:+.2f}%)")
        print(f"   A/D: {vnindex['advance']}/{vnindex['decline']}")

    print("\n" + "=" * 70)
    print("✅ SSI Provider testing complete!")
    print("=" * 70)
