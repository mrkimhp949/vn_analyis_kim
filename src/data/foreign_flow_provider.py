# -*- coding: utf-8 -*-
"""
Real-time Foreign Flow Data Provider for Vietnam Stock Market

Integrates multiple data sources for foreign investor flow:
- TCBS API (primary)
- SSI API (secondary)
- HOSE/HNX official data (fallback)
- vnstock library

Foreign flow is a key indicator in Vietnam market:
- Foreign investors are ~20% of market volume
- Net buying/selling signals institutional sentiment
- Often leads market moves by 1-2 days

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from threading import Thread, Event, RLock
import json
import os

import pandas as pd

logger = logging.getLogger(__name__)

# Suppress vnstock INFO logs (e.g., "Not a stock" warnings for indices)
logging.getLogger("vnstock").setLevel(logging.WARNING)
logging.getLogger("vnstock.common.data").setLevel(logging.WARNING)


@dataclass
class ForeignFlowRecord:
    """Single day foreign flow record"""

    date: str
    symbol: str
    buy_volume: int
    sell_volume: int
    buy_value: float  # VND
    sell_value: float  # VND
    net_volume: int
    net_value: float
    foreign_room: Optional[float] = None  # Remaining foreign room %
    source: str = ""

    @property
    def is_net_buy(self) -> bool:
        return self.net_value > 0


@dataclass
class MarketForeignFlow:
    """Market-wide foreign flow summary"""

    date: str
    total_buy_value: float
    total_sell_value: float
    net_value: float
    top_net_buy: List[Tuple[str, float]]  # [(symbol, value), ...]
    top_net_sell: List[Tuple[str, float]]
    buy_count: int  # Number of stocks with net buy
    sell_count: int  # Number of stocks with net sell
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class ForeignFlowProvider:
    """
    Real-time Foreign Flow Data Provider

    Features:
    - Multiple data source integration
    - Automatic failover
    - Caching with TTL
    - Background refresh
    - Historical data storage

    Usage:
        provider = ForeignFlowProvider()

        # Get market-wide flow
        market_flow = provider.get_market_flow()

        # Get symbol-specific flow
        symbol_flow = provider.get_symbol_flow("VNM")

        # Start real-time updates
        provider.start_realtime_updates()
    """

    CACHE_FILE = "foreign_flow_cache.json"
    CACHE_TTL_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        cache_file: str = CACHE_FILE,
        auto_refresh: bool = False,
        refresh_interval_seconds: int = 300,
    ):
        """
        Initialize Foreign Flow Provider.

        Args:
            cache_file: File to cache flow data
            auto_refresh: Enable background refresh
            refresh_interval_seconds: Refresh interval
        """
        self.cache_file = cache_file
        self.refresh_interval = refresh_interval_seconds

        # Cache
        self._market_cache: Optional[MarketForeignFlow] = None
        self._symbol_cache: Dict[str, List[ForeignFlowRecord]] = {}
        self._cache_time: Optional[datetime] = None
        self._lock = RLock()

        # Background refresh
        self._stop_event = Event()
        self._refresh_thread: Optional[Thread] = None

        # Load cached data
        self._load_cache()

        # Start auto refresh if enabled
        if auto_refresh:
            self.start_realtime_updates()

    def _load_cache(self):
        """Load cached data from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Load market cache
                    if "market" in data:
                        m = data["market"]
                        self._market_cache = MarketForeignFlow(
                            date=m.get("date", ""),
                            total_buy_value=m.get("total_buy_value", 0),
                            total_sell_value=m.get("total_sell_value", 0),
                            net_value=m.get("net_value", 0),
                            top_net_buy=m.get("top_net_buy", []),
                            top_net_sell=m.get("top_net_sell", []),
                            buy_count=m.get("buy_count", 0),
                            sell_count=m.get("sell_count", 0),
                            source=m.get("source", "cache"),
                        )

                    self._cache_time = datetime.fromisoformat(
                        data.get("timestamp", datetime.now().isoformat())
                    )
                    logger.info(f"📂 Loaded foreign flow cache from {self.cache_file}")

            except Exception as e:
                logger.warning(f"Failed to load foreign flow cache: {e}")

    def _save_cache(self):
        """Save cache to file"""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
            }

            if self._market_cache:
                data["market"] = {
                    "date": self._market_cache.date,
                    "total_buy_value": self._market_cache.total_buy_value,
                    "total_sell_value": self._market_cache.total_sell_value,
                    "net_value": self._market_cache.net_value,
                    "top_net_buy": self._market_cache.top_net_buy,
                    "top_net_sell": self._market_cache.top_net_sell,
                    "buy_count": self._market_cache.buy_count,
                    "sell_count": self._market_cache.sell_count,
                    "source": self._market_cache.source,
                }

            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Failed to save foreign flow cache: {e}")

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if self._cache_time is None:
            return False
        age = (datetime.now() - self._cache_time).total_seconds()
        return age < self.CACHE_TTL_SECONDS

    # =========================================================================
    # Data Fetching - Multiple Sources
    # =========================================================================

    def _fetch_from_vnstock(self) -> Optional[pd.DataFrame]:
        """Fetch foreign flow from vnstock library"""
        try:
            from vnstock import Vnstock

            stock = Vnstock()

            # Get market foreign flow
            # vnstock provides foreign trading data
            df = stock.stock(symbol="VNINDEX", source="VCI").quote.history(
                start=datetime.now().strftime("%Y-%m-%d"),
                end=datetime.now().strftime("%Y-%m-%d"),
            )

            logger.info("✅ Fetched foreign flow from vnstock")
            return df

        except ImportError:
            logger.debug("vnstock not available")
        except Exception as e:
            logger.warning(f"vnstock foreign flow fetch failed: {e}")

        return None

    def _fetch_from_tcbs(self) -> Optional[Dict]:
        """Fetch foreign flow from TCBS API"""
        try:
            import requests

            url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday-investor/all"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Fetched foreign flow from TCBS")
                return data

        except Exception as e:
            logger.warning(f"TCBS foreign flow fetch failed: {e}")

        return None

    def _fetch_from_ssi(self) -> Optional[Dict]:
        """Fetch foreign flow from SSI API"""
        try:
            import requests

            # SSI public API endpoint
            url = "https://iboard.ssi.com.vn/dchart/api/1.1/defaultAllStocks"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Fetched foreign flow from SSI")
                return data

        except Exception as e:
            logger.warning(f"SSI foreign flow fetch failed: {e}")

        return None

    def _fetch_from_cafef(self) -> Optional[Dict]:
        """Fetch foreign flow from CafeF"""
        try:
            import requests

            # CafeF foreign trading page
            url = "https://s.cafef.vn/Ajax/PageNew/DataHistory/GDKhoiNgoai.ashx"
            params = {
                "Symbol": "VNINDEX",
                "StartDate": datetime.now().strftime("%d/%m/%Y"),
                "EndDate": datetime.now().strftime("%d/%m/%Y"),
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Fetched foreign flow from CafeF")
                return data

        except Exception as e:
            logger.warning(f"CafeF foreign flow fetch failed: {e}")

        return None

    # =========================================================================
    # Public API
    # =========================================================================

    def get_market_flow(self, force_refresh: bool = False) -> Optional[MarketForeignFlow]:
        """
        Get market-wide foreign flow summary.

        Args:
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            MarketForeignFlow or None if unavailable
        """
        with self._lock:
            # Check cache
            if not force_refresh and self._is_cache_valid() and self._market_cache:
                return self._market_cache

            # Try multiple sources
            flow_data = None
            source = ""

            # Source 1: TCBS
            tcbs_data = self._fetch_from_tcbs()
            if tcbs_data:
                flow_data = self._parse_tcbs_market_flow(tcbs_data)
                source = "TCBS"

            # Source 2: SSI (fallback)
            if not flow_data:
                ssi_data = self._fetch_from_ssi()
                if ssi_data:
                    flow_data = self._parse_ssi_market_flow(ssi_data)
                    source = "SSI"

            # Source 3: CafeF (fallback)
            if not flow_data:
                cafef_data = self._fetch_from_cafef()
                if cafef_data:
                    flow_data = self._parse_cafef_market_flow(cafef_data)
                    source = "CafeF"

            if flow_data:
                flow_data.source = source
                self._market_cache = flow_data
                self._cache_time = datetime.now()
                self._save_cache()
                return flow_data

            # Return cached data if available (even if stale)
            if self._market_cache:
                logger.warning("Using stale foreign flow cache")
                return self._market_cache

            return None

    def get_symbol_flow(
        self,
        symbol: str,
        lookback_days: int = 20,
        force_refresh: bool = False,
    ) -> List[ForeignFlowRecord]:
        """
        Get foreign flow history for a specific symbol.

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to fetch
            force_refresh: Bypass cache

        Returns:
            List of ForeignFlowRecord
        """
        with self._lock:
            # Check cache
            cache_key = symbol.upper()
            if not force_refresh and cache_key in self._symbol_cache:
                return self._symbol_cache[cache_key]

            records = []

            try:
                # Try vnstock first
                from vnstock import Vnstock

                stock = Vnstock().stock(symbol=symbol, source="VCI")

                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)

                df = stock.quote.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                )

                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        # Extract foreign trading data if available
                        record = ForeignFlowRecord(
                            date=str(row.get("time", row.name))[:10],
                            symbol=symbol,
                            buy_volume=int(row.get("foreignBuyVolume", 0)),
                            sell_volume=int(row.get("foreignSellVolume", 0)),
                            buy_value=float(row.get("foreignBuyValue", 0)),
                            sell_value=float(row.get("foreignSellValue", 0)),
                            net_volume=int(
                                row.get("foreignBuyVolume", 0) - row.get("foreignSellVolume", 0)
                            ),
                            net_value=float(
                                row.get("foreignBuyValue", 0) - row.get("foreignSellValue", 0)
                            ),
                            source="vnstock",
                        )
                        records.append(record)

                    self._symbol_cache[cache_key] = records
                    logger.info(f"✅ Fetched {len(records)} foreign flow records for {symbol}")

            except ImportError:
                logger.debug("vnstock not available for symbol flow")
            except Exception as e:
                logger.warning(f"Failed to fetch symbol flow for {symbol}: {e}")

            return records

    def get_top_foreign_activity(
        self,
        top_n: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top foreign buying and selling stocks.

        Returns:
            {
                "top_buy": [(symbol, net_value), ...],
                "top_sell": [(symbol, net_value), ...],
            }
        """
        market_flow = self.get_market_flow()

        if market_flow:
            return {
                "top_buy": market_flow.top_net_buy[:top_n],
                "top_sell": market_flow.top_net_sell[:top_n],
            }

        return {"top_buy": [], "top_sell": []}

    def get_flow_score(self, symbol: Optional[str] = None) -> float:
        """
        Get foreign flow score (-1 to +1).

        Args:
            symbol: Specific symbol or None for market-wide

        Returns:
            Score from -1 (heavy selling) to +1 (heavy buying)
        """
        if symbol:
            records = self.get_symbol_flow(symbol, lookback_days=5)
            if records:
                total_net = sum(r.net_value for r in records)
                total_volume = sum(abs(r.net_value) for r in records)
                if total_volume > 0:
                    return max(-1, min(1, total_net / total_volume))
        else:
            market_flow = self.get_market_flow()
            if market_flow:
                total = market_flow.total_buy_value + market_flow.total_sell_value
                if total > 0:
                    return max(-1, min(1, market_flow.net_value / total * 2))

        return 0.0

    # =========================================================================
    # Real-time Updates
    # =========================================================================

    def start_realtime_updates(self):
        """Start background thread for real-time updates"""
        if self._refresh_thread and self._refresh_thread.is_alive():
            logger.warning("Real-time updates already running")
            return

        self._stop_event.clear()
        self._refresh_thread = Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()
        logger.info(
            f"🔄 Started foreign flow real-time updates (interval: {self.refresh_interval}s)"
        )

    def stop_realtime_updates(self):
        """Stop background refresh thread"""
        self._stop_event.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)
        logger.info("⏹️ Stopped foreign flow real-time updates")

    def _refresh_loop(self):
        """Background refresh loop"""
        while not self._stop_event.is_set():
            try:
                # Only refresh during trading hours
                if self._is_trading_hours():
                    self.get_market_flow(force_refresh=True)
                    logger.debug("🔄 Foreign flow data refreshed")
            except Exception as e:
                logger.warning(f"Foreign flow refresh failed: {e}")

            # Wait for next refresh
            self._stop_event.wait(self.refresh_interval)

    def _is_trading_hours(self) -> bool:
        """Check if within Vietnam trading hours"""
        now = datetime.now()
        hour, minute = now.hour, now.minute
        time_val = hour * 60 + minute

        # Morning: 9:00-11:30
        # Afternoon: 13:00-14:45
        morning = 9 * 60 <= time_val <= 11 * 60 + 30
        afternoon = 13 * 60 <= time_val <= 14 * 60 + 45

        return morning or afternoon

    # =========================================================================
    # Data Parsing
    # =========================================================================

    def _parse_tcbs_market_flow(self, data: Dict) -> Optional[MarketForeignFlow]:
        """Parse TCBS API response"""
        try:
            # TCBS format varies - adapt as needed
            total_buy = 0
            total_sell = 0
            top_buy = []
            top_sell = []

            if isinstance(data, list):
                for item in data:
                    symbol = item.get("ticker", "")
                    buy_val = item.get("foreignBuyValue", 0)
                    sell_val = item.get("foreignSellValue", 0)
                    net_val = buy_val - sell_val

                    total_buy += buy_val
                    total_sell += sell_val

                    if net_val > 0:
                        top_buy.append((symbol, net_val))
                    elif net_val < 0:
                        top_sell.append((symbol, net_val))

            # Sort by absolute value
            top_buy.sort(key=lambda x: x[1], reverse=True)
            top_sell.sort(key=lambda x: x[1])

            return MarketForeignFlow(
                date=datetime.now().strftime("%Y-%m-%d"),
                total_buy_value=total_buy,
                total_sell_value=total_sell,
                net_value=total_buy - total_sell,
                top_net_buy=top_buy[:10],
                top_net_sell=top_sell[:10],
                buy_count=len([x for x in top_buy if x[1] > 0]),
                sell_count=len([x for x in top_sell if x[1] < 0]),
            )

        except Exception as e:
            logger.warning(f"Failed to parse TCBS data: {e}")
            return None

    def _parse_ssi_market_flow(self, data: Dict) -> Optional[MarketForeignFlow]:
        """Parse SSI API response"""
        try:
            # SSI format - adapt as needed
            stocks = data.get("data", [])

            total_buy = 0
            total_sell = 0
            top_buy = []
            top_sell = []

            for stock in stocks:
                symbol = stock.get("stockSymbol", "")
                buy_val = float(stock.get("foreignBuyValue", 0))
                sell_val = float(stock.get("foreignSellValue", 0))
                net_val = buy_val - sell_val

                total_buy += buy_val
                total_sell += sell_val

                if net_val > 0:
                    top_buy.append((symbol, net_val))
                elif net_val < 0:
                    top_sell.append((symbol, net_val))

            top_buy.sort(key=lambda x: x[1], reverse=True)
            top_sell.sort(key=lambda x: x[1])

            return MarketForeignFlow(
                date=datetime.now().strftime("%Y-%m-%d"),
                total_buy_value=total_buy,
                total_sell_value=total_sell,
                net_value=total_buy - total_sell,
                top_net_buy=top_buy[:10],
                top_net_sell=top_sell[:10],
                buy_count=len(top_buy),
                sell_count=len(top_sell),
            )

        except Exception as e:
            logger.warning(f"Failed to parse SSI data: {e}")
            return None

    def _parse_cafef_market_flow(self, data: Dict) -> Optional[MarketForeignFlow]:
        """Parse CafeF API response"""
        try:
            # CafeF format - adapt as needed
            items = data.get("Data", [])

            if items:
                item = items[0]  # Latest day
                return MarketForeignFlow(
                    date=item.get("Ngay", datetime.now().strftime("%Y-%m-%d")),
                    total_buy_value=float(item.get("GiaTriMua", 0)),
                    total_sell_value=float(item.get("GiaTriBan", 0)),
                    net_value=float(item.get("GiaTriMua", 0)) - float(item.get("GiaTriBan", 0)),
                    top_net_buy=[],
                    top_net_sell=[],
                    buy_count=0,
                    sell_count=0,
                )

        except Exception as e:
            logger.warning(f"Failed to parse CafeF data: {e}")
            return None


# Singleton instance
_provider_instance: Optional[ForeignFlowProvider] = None


def get_foreign_flow_provider(auto_refresh: bool = False) -> ForeignFlowProvider:
    """Get singleton instance of foreign flow provider"""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = ForeignFlowProvider(auto_refresh=auto_refresh)
    return _provider_instance
