# -*- coding: utf-8 -*-
"""
HOSE/HNX Real-time Foreign Flow API Integration

This module provides real-time foreign flow data from official sources:
- HOSE (Ho Chi Minh Stock Exchange) API
- HNX (Hanoi Stock Exchange) API  
- UPCoM (Unlisted Public Company Market) API

Features:
- Real-time foreign trading data with WebSocket support
- Historical foreign flow aggregation
- Automatic failover between data sources
- Rate limiting and caching

Usage:
    from src.data.hose_hnx_realtime_api import (
        get_realtime_foreign_flow_api,
        HOSEHNXForeignFlowAPI,
    )
    
    api = get_realtime_foreign_flow_api()
    flow = api.get_current_foreign_flow("VNM")
    
    # Subscribe to real-time updates
    api.subscribe("VNM", callback=my_handler)
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


class Exchange(Enum):
    """Vietnam stock exchanges"""

    HOSE = "HOSE"  # Ho Chi Minh Stock Exchange
    HNX = "HNX"  # Hanoi Stock Exchange
    UPCOM = "UPCOM"  # Unlisted Public Company Market


# API Endpoints (public data)
API_ENDPOINTS = {
    # VNDirect public API
    "vndirect": {
        "base_url": "https://finfo-api.vndirect.com.vn",
        "foreign_flow": "/v4/foreign_flows",
        "intraday": "/v4/stock_prices",
    },
    # SSI public API
    "ssi": {
        "base_url": "https://iboard.ssi.com.vn/dchart/api",
        "foreign_flow": "/foreignTrade",
        "market_data": "/stockTrade",
    },
    # TCBS public API
    "tcbs": {
        "base_url": "https://apipubaws.tcbs.com.vn",
        "foreign_flow": "/stock-insight/v1/stock/foreign-flow",
        "intraday": "/stock-insight/v1/intraday",
    },
    # CafeF public API
    "cafef": {
        "base_url": "https://s.cafef.vn/ajax",
        "foreign_flow": "/foreigntrade.aspx",
    },
    # FireAnt public API
    "fireant": {
        "base_url": "https://api.fireant.vn",
        "foreign_flow": "/symbols/foreign-trading",
    },
}

# Price limits by exchange
PRICE_LIMITS = {
    Exchange.HOSE: 0.07,  # ±7%
    Exchange.HNX: 0.10,  # ±10%
    Exchange.UPCOM: 0.15,  # ±15%
}

# Rate limiting
DEFAULT_RATE_LIMIT = 60  # requests per minute
CACHE_TTL_SECONDS = 30  # 30 seconds for real-time data


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ForeignFlowSnapshot:
    """Real-time foreign flow snapshot for a symbol"""

    symbol: str
    exchange: Exchange
    timestamp: datetime

    # Volume
    buy_volume: int = 0
    sell_volume: int = 0
    net_volume: int = 0

    # Value (VND)
    buy_value: float = 0.0
    sell_value: float = 0.0
    net_value: float = 0.0

    # Room
    remaining_room: int = 0
    room_percent: float = 0.0
    current_foreign_percent: float = 0.0
    fol_limit: float = 0.0  # Foreign Ownership Limit

    # Intraday stats
    avg_buy_price: float = 0.0
    avg_sell_price: float = 0.0
    buy_orders_count: int = 0
    sell_orders_count: int = 0

    # Status
    is_realtime: bool = True
    data_source: str = "unknown"
    latency_ms: int = 0

    def __post_init__(self):
        if self.buy_volume > 0:
            self.avg_buy_price = self.buy_value / self.buy_volume if self.buy_value else 0
        if self.sell_volume > 0:
            self.avg_sell_price = self.sell_value / self.sell_volume if self.sell_value else 0

    @property
    def net_flow_direction(self) -> str:
        """Get flow direction: BUYING, SELLING, or NEUTRAL"""
        if self.net_value > 1_000_000_000:  # > 1 billion
            return "STRONG_BUYING"
        elif self.net_value > 100_000_000:  # > 100 million
            return "BUYING"
        elif self.net_value < -1_000_000_000:
            return "STRONG_SELLING"
        elif self.net_value < -100_000_000:
            return "SELLING"
        return "NEUTRAL"

    @property
    def is_accumulating(self) -> bool:
        """Check if foreign investors are accumulating"""
        return self.buy_volume > self.sell_volume * 1.2

    @property
    def is_distributing(self) -> bool:
        """Check if foreign investors are distributing"""
        return self.sell_volume > self.buy_volume * 1.2

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "timestamp": self.timestamp.isoformat(),
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "net_volume": self.net_volume,
            "buy_value": self.buy_value,
            "sell_value": self.sell_value,
            "net_value": self.net_value,
            "remaining_room": self.remaining_room,
            "room_percent": self.room_percent,
            "current_foreign_percent": self.current_foreign_percent,
            "fol_limit": self.fol_limit,
            "net_flow_direction": self.net_flow_direction,
            "is_accumulating": self.is_accumulating,
            "is_distributing": self.is_distributing,
            "data_source": self.data_source,
            "is_realtime": self.is_realtime,
        }


@dataclass
class MarketForeignFlowSummary:
    """Market-wide foreign flow summary"""

    exchange: Exchange
    timestamp: datetime

    # Aggregate values
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    net_value: float = 0.0

    # Counts
    stocks_with_net_buy: int = 0
    stocks_with_net_sell: int = 0
    stocks_with_trading: int = 0

    # Top movers
    top_net_buy: List[Tuple[str, float]] = field(default_factory=list)
    top_net_sell: List[Tuple[str, float]] = field(default_factory=list)

    # Status
    data_source: str = "unknown"
    is_realtime: bool = True


@dataclass
class HistoricalForeignFlow:
    """Historical foreign flow data point"""

    symbol: str
    date: date
    buy_volume: int = 0
    sell_volume: int = 0
    net_volume: int = 0
    buy_value: float = 0.0
    sell_value: float = 0.0
    net_value: float = 0.0


# =============================================================================
# BASE DATA SOURCE
# =============================================================================


class ForeignFlowDataSource(ABC):
    """Abstract base class for foreign flow data sources"""

    @abstractmethod
    def get_realtime_flow(self, symbol: str) -> Optional[ForeignFlowSnapshot]:
        """Get real-time foreign flow for a symbol"""
        pass

    @abstractmethod
    def get_market_summary(self, exchange: Exchange) -> Optional[MarketForeignFlowSummary]:
        """Get market-wide foreign flow summary"""
        pass

    @abstractmethod
    def get_historical_flow(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[HistoricalForeignFlow]:
        """Get historical foreign flow data"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Data source name"""
        pass


# =============================================================================
# VNDIRECT DATA SOURCE
# =============================================================================


class VNDirectForeignFlowSource(ForeignFlowDataSource):
    """VNDirect API for foreign flow data"""

    def __init__(self):
        self._base_url = API_ENDPOINTS["vndirect"]["base_url"]
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            }
        )
        self._last_request_time = 0
        self._min_request_interval = 1.0  # seconds

    @property
    def name(self) -> str:
        return "vndirect"

    def is_available(self) -> bool:
        try:
            response = self._session.get(
                f"{self._base_url}/v4/stock_prices",
                params={"q": "code:VNM", "size": 1},
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _rate_limit(self):
        """Apply rate limiting"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def get_realtime_flow(self, symbol: str) -> Optional[ForeignFlowSnapshot]:
        """Get real-time foreign flow from VNDirect"""
        self._rate_limit()

        try:
            start_time = time.time()

            # Get foreign trading data
            response = self._session.get(
                f"{self._base_url}/v4/stock_prices",
                params={
                    "q": f"code:{symbol}",
                    "size": 1,
                    "sort": "date",
                },
                timeout=10,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if not data.get("data"):
                return None

            item = data["data"][0]
            latency = int((time.time() - start_time) * 1000)

            # Determine exchange
            exchange = self._get_exchange(symbol)

            return ForeignFlowSnapshot(
                symbol=symbol,
                exchange=exchange,
                timestamp=datetime.now(),
                buy_volume=item.get("foreignBuyVolume", 0) or 0,
                sell_volume=item.get("foreignSellVolume", 0) or 0,
                net_volume=(item.get("foreignBuyVolume", 0) or 0)
                - (item.get("foreignSellVolume", 0) or 0),
                buy_value=item.get("foreignBuyValue", 0) or 0,
                sell_value=item.get("foreignSellValue", 0) or 0,
                net_value=(item.get("foreignBuyValue", 0) or 0)
                - (item.get("foreignSellValue", 0) or 0),
                remaining_room=item.get("foreignRemainRoom", 0) or 0,
                current_foreign_percent=item.get("foreignCurrentPercent", 0) or 0,
                is_realtime=True,
                data_source="vndirect",
                latency_ms=latency,
            )

        except Exception as e:
            logger.debug(f"VNDirect foreign flow error: {e}")
            return None

    def get_market_summary(self, exchange: Exchange) -> Optional[MarketForeignFlowSummary]:
        """Get market-wide foreign flow summary"""
        self._rate_limit()

        try:
            # Get top foreign trading stocks
            response = self._session.get(
                f"{self._base_url}/v4/stock_prices",
                params={
                    "q": f"floor:{exchange.value}",
                    "size": 100,
                    "sort": "foreignNetValue:desc",
                },
                timeout=15,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if not data.get("data"):
                return None

            items = data["data"]

            total_buy = sum(item.get("foreignBuyValue", 0) or 0 for item in items)
            total_sell = sum(item.get("foreignSellValue", 0) or 0 for item in items)

            net_buy_stocks = [
                (
                    item["code"],
                    (item.get("foreignBuyValue", 0) or 0) - (item.get("foreignSellValue", 0) or 0),
                )
                for item in items
                if (item.get("foreignBuyValue", 0) or 0) > (item.get("foreignSellValue", 0) or 0)
            ]

            net_sell_stocks = [
                (
                    item["code"],
                    (item.get("foreignSellValue", 0) or 0) - (item.get("foreignBuyValue", 0) or 0),
                )
                for item in items
                if (item.get("foreignSellValue", 0) or 0) > (item.get("foreignBuyValue", 0) or 0)
            ]

            return MarketForeignFlowSummary(
                exchange=exchange,
                timestamp=datetime.now(),
                total_buy_value=total_buy,
                total_sell_value=total_sell,
                net_value=total_buy - total_sell,
                stocks_with_net_buy=len(net_buy_stocks),
                stocks_with_net_sell=len(net_sell_stocks),
                stocks_with_trading=len(items),
                top_net_buy=sorted(net_buy_stocks, key=lambda x: x[1], reverse=True)[:10],
                top_net_sell=sorted(net_sell_stocks, key=lambda x: x[1], reverse=True)[:10],
                data_source="vndirect",
                is_realtime=True,
            )

        except Exception as e:
            logger.debug(f"VNDirect market summary error: {e}")
            return None

    def get_historical_flow(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[HistoricalForeignFlow]:
        """Get historical foreign flow data"""
        self._rate_limit()

        try:
            response = self._session.get(
                f"{self._base_url}/v4/stock_prices",
                params={
                    "q": f"code:{symbol}~date:gte:{start_date}~date:lte:{end_date}",
                    "size": 500,
                    "sort": "date",
                },
                timeout=15,
            )

            if response.status_code != 200:
                return []

            data = response.json()
            if not data.get("data"):
                return []

            result = []
            for item in data["data"]:
                try:
                    flow_date = datetime.strptime(item["date"], "%Y-%m-%d").date()
                    result.append(
                        HistoricalForeignFlow(
                            symbol=symbol,
                            date=flow_date,
                            buy_volume=item.get("foreignBuyVolume", 0) or 0,
                            sell_volume=item.get("foreignSellVolume", 0) or 0,
                            net_volume=(item.get("foreignBuyVolume", 0) or 0)
                            - (item.get("foreignSellVolume", 0) or 0),
                            buy_value=item.get("foreignBuyValue", 0) or 0,
                            sell_value=item.get("foreignSellValue", 0) or 0,
                            net_value=(item.get("foreignBuyValue", 0) or 0)
                            - (item.get("foreignSellValue", 0) or 0),
                        )
                    )
                except Exception:
                    continue

            return result

        except Exception as e:
            logger.debug(f"VNDirect historical flow error: {e}")
            return []

    def _get_exchange(self, symbol: str) -> Exchange:
        """Determine exchange for a symbol"""
        # Use cached mapping or API lookup
        # For now, default to HOSE for major stocks
        major_hnx = {"SHB", "NVB", "PVS", "ACB", "SHS", "VCS", "CEO", "NVL"}
        if symbol in major_hnx:
            return Exchange.HNX
        return Exchange.HOSE


# =============================================================================
# TCBS DATA SOURCE
# =============================================================================


class TCBSForeignFlowSource(ForeignFlowDataSource):
    """TCBS API for foreign flow data"""

    def __init__(self):
        self._base_url = API_ENDPOINTS["tcbs"]["base_url"]
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            }
        )
        self._last_request_time = 0
        self._min_request_interval = 0.5

    @property
    def name(self) -> str:
        return "tcbs"

    def is_available(self) -> bool:
        try:
            response = self._session.get(
                f"{self._base_url}/stock-insight/v1/stock/overview",
                params={"ticker": "VNM"},
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def get_realtime_flow(self, symbol: str) -> Optional[ForeignFlowSnapshot]:
        """Get real-time foreign flow from TCBS"""
        self._rate_limit()

        try:
            start_time = time.time()

            response = self._session.get(
                f"{self._base_url}/stock-insight/v1/stock/overview",
                params={"ticker": symbol},
                timeout=10,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if not data:
                return None

            latency = int((time.time() - start_time) * 1000)

            exchange_str = data.get("exchange", "HOSE")
            exchange = (
                Exchange[exchange_str] if exchange_str in Exchange.__members__ else Exchange.HOSE
            )

            return ForeignFlowSnapshot(
                symbol=symbol,
                exchange=exchange,
                timestamp=datetime.now(),
                buy_volume=data.get("foreignBuyVolume", 0) or 0,
                sell_volume=data.get("foreignSellVolume", 0) or 0,
                net_volume=(data.get("foreignBuyVolume", 0) or 0)
                - (data.get("foreignSellVolume", 0) or 0),
                buy_value=data.get("foreignBuyValue", 0) or 0,
                sell_value=data.get("foreignSellValue", 0) or 0,
                net_value=(data.get("foreignBuyValue", 0) or 0)
                - (data.get("foreignSellValue", 0) or 0),
                remaining_room=data.get("foreignRoom", 0) or 0,
                room_percent=data.get("foreignRoomPercent", 0) or 0,
                current_foreign_percent=data.get("foreignPercent", 0) or 0,
                fol_limit=data.get("folLimit", 0) or 0,
                is_realtime=True,
                data_source="tcbs",
                latency_ms=latency,
            )

        except Exception as e:
            logger.debug(f"TCBS foreign flow error: {e}")
            return None

    def get_market_summary(self, exchange: Exchange) -> Optional[MarketForeignFlowSummary]:
        """Get market summary - TCBS doesn't provide this directly"""
        return None

    def get_historical_flow(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[HistoricalForeignFlow]:
        """Get historical flow - TCBS provides limited historical data"""
        return []


# =============================================================================
# SSI DATA SOURCE
# =============================================================================


class SSIForeignFlowSource(ForeignFlowDataSource):
    """SSI API for foreign flow data"""

    def __init__(self):
        self._base_url = API_ENDPOINTS["ssi"]["base_url"]
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            }
        )
        self._last_request_time = 0
        self._min_request_interval = 0.5

    @property
    def name(self) -> str:
        return "ssi"

    def is_available(self) -> bool:
        try:
            response = self._session.get(
                f"{self._base_url}/stockTrade",
                params={"code": "VNM"},
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def get_realtime_flow(self, symbol: str) -> Optional[ForeignFlowSnapshot]:
        """Get real-time foreign flow from SSI"""
        self._rate_limit()

        try:
            start_time = time.time()

            response = self._session.get(
                f"{self._base_url}/stockTrade",
                params={"code": symbol},
                timeout=10,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if not data or not data.get("data"):
                return None

            item = data["data"]
            latency = int((time.time() - start_time) * 1000)

            exchange = Exchange.HOSE  # Default

            return ForeignFlowSnapshot(
                symbol=symbol,
                exchange=exchange,
                timestamp=datetime.now(),
                buy_volume=item.get("foreignerBuyVol", 0) or 0,
                sell_volume=item.get("foreignerSellVol", 0) or 0,
                net_volume=(item.get("foreignerBuyVol", 0) or 0)
                - (item.get("foreignerSellVol", 0) or 0),
                buy_value=item.get("foreignerBuyVal", 0) or 0,
                sell_value=item.get("foreignerSellVal", 0) or 0,
                net_value=(item.get("foreignerBuyVal", 0) or 0)
                - (item.get("foreignerSellVal", 0) or 0),
                remaining_room=item.get("remainRoom", 0) or 0,
                current_foreign_percent=item.get("foreignerPercent", 0) or 0,
                is_realtime=True,
                data_source="ssi",
                latency_ms=latency,
            )

        except Exception as e:
            logger.debug(f"SSI foreign flow error: {e}")
            return None

    def get_market_summary(self, exchange: Exchange) -> Optional[MarketForeignFlowSummary]:
        """Get market summary from SSI"""
        self._rate_limit()

        try:
            response = self._session.get(
                f"{self._base_url}/foreignTrade",
                params={"floor": exchange.value},
                timeout=15,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if not data or not data.get("data"):
                return None

            items = data["data"]

            total_buy = sum(item.get("foreignerBuyVal", 0) or 0 for item in items)
            total_sell = sum(item.get("foreignerSellVal", 0) or 0 for item in items)

            net_buy_stocks = [
                (
                    item["code"],
                    (item.get("foreignerBuyVal", 0) or 0) - (item.get("foreignerSellVal", 0) or 0),
                )
                for item in items
                if (item.get("foreignerBuyVal", 0) or 0) > (item.get("foreignerSellVal", 0) or 0)
            ]

            net_sell_stocks = [
                (
                    item["code"],
                    (item.get("foreignerSellVal", 0) or 0) - (item.get("foreignerBuyVal", 0) or 0),
                )
                for item in items
                if (item.get("foreignerSellVal", 0) or 0) > (item.get("foreignerBuyVal", 0) or 0)
            ]

            return MarketForeignFlowSummary(
                exchange=exchange,
                timestamp=datetime.now(),
                total_buy_value=total_buy,
                total_sell_value=total_sell,
                net_value=total_buy - total_sell,
                stocks_with_net_buy=len(net_buy_stocks),
                stocks_with_net_sell=len(net_sell_stocks),
                stocks_with_trading=len(items),
                top_net_buy=sorted(net_buy_stocks, key=lambda x: x[1], reverse=True)[:10],
                top_net_sell=sorted(net_sell_stocks, key=lambda x: x[1], reverse=True)[:10],
                data_source="ssi",
                is_realtime=True,
            )

        except Exception as e:
            logger.debug(f"SSI market summary error: {e}")
            return None

    def get_historical_flow(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[HistoricalForeignFlow]:
        """SSI doesn't provide historical foreign flow"""
        return []


# =============================================================================
# MAIN FOREIGN FLOW API
# =============================================================================


class HOSEHNXForeignFlowAPI:
    """
    Unified Real-time Foreign Flow API for HOSE/HNX/UPCoM

    Features:
    - Multi-source integration with automatic failover
    - Real-time and historical data
    - Caching with configurable TTL
    - Rate limiting
    - WebSocket subscription support (future)

    Usage:
        api = HOSEHNXForeignFlowAPI()

        # Get real-time flow
        flow = api.get_current_foreign_flow("VNM")

        # Get market summary
        summary = api.get_market_summary(Exchange.HOSE)

        # Get historical data
        history = api.get_historical_flow("VNM", days=30)
    """

    def __init__(
        self,
        enable_vndirect: bool = True,
        enable_tcbs: bool = True,
        enable_ssi: bool = True,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
        auto_failover: bool = True,
    ):
        self._sources: List[ForeignFlowDataSource] = []
        self._cache: Dict[str, Tuple[ForeignFlowSnapshot, datetime]] = {}
        self._market_cache: Dict[str, Tuple[MarketForeignFlowSummary, datetime]] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._auto_failover = auto_failover
        self._lock = RLock()

        # Initialize data sources
        if enable_vndirect:
            self._sources.append(VNDirectForeignFlowSource())
        if enable_tcbs:
            self._sources.append(TCBSForeignFlowSource())
        if enable_ssi:
            self._sources.append(SSIForeignFlowSource())

        # Subscribers for real-time updates
        self._subscribers: Dict[str, List[Callable[[ForeignFlowSnapshot], None]]] = {}
        self._streaming_active = False
        self._streaming_thread: Optional[threading.Thread] = None

        logger.info(f"🌐 HOSEHNXForeignFlowAPI initialized with {len(self._sources)} sources")

    def get_current_foreign_flow(
        self, symbol: str, use_cache: bool = True
    ) -> Optional[ForeignFlowSnapshot]:
        """
        Get current foreign flow for a symbol.

        Args:
            symbol: Stock symbol (e.g., "VNM")
            use_cache: Whether to use cached data

        Returns:
            ForeignFlowSnapshot or None if unavailable
        """
        symbol = symbol.upper()

        # Check cache
        if use_cache:
            with self._lock:
                if symbol in self._cache:
                    snapshot, cached_time = self._cache[symbol]
                    if datetime.now() - cached_time < self._cache_ttl:
                        return snapshot

        # Try each data source
        for source in self._sources:
            try:
                snapshot = source.get_realtime_flow(symbol)
                if snapshot:
                    # Cache result
                    with self._lock:
                        self._cache[symbol] = (snapshot, datetime.now())
                    return snapshot
            except Exception as e:
                logger.debug(f"Source {source.name} failed for {symbol}: {e}")
                if not self._auto_failover:
                    break

        logger.warning(f"No foreign flow data available for {symbol}")
        return None

    def get_batch_foreign_flow(
        self, symbols: List[str], use_cache: bool = True
    ) -> Dict[str, Optional[ForeignFlowSnapshot]]:
        """
        Get foreign flow for multiple symbols.

        Args:
            symbols: List of stock symbols
            use_cache: Whether to use cached data

        Returns:
            Dict mapping symbol to ForeignFlowSnapshot
        """
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_current_foreign_flow(symbol, use_cache)
        return result

    def get_market_summary(
        self, exchange: Exchange = Exchange.HOSE, use_cache: bool = True
    ) -> Optional[MarketForeignFlowSummary]:
        """
        Get market-wide foreign flow summary.

        Args:
            exchange: Exchange to get summary for
            use_cache: Whether to use cached data

        Returns:
            MarketForeignFlowSummary or None
        """
        cache_key = exchange.value

        # Check cache
        if use_cache:
            with self._lock:
                if cache_key in self._market_cache:
                    summary, cached_time = self._market_cache[cache_key]
                    if datetime.now() - cached_time < self._cache_ttl:
                        return summary

        # Try each data source
        for source in self._sources:
            try:
                summary = source.get_market_summary(exchange)
                if summary:
                    with self._lock:
                        self._market_cache[cache_key] = (summary, datetime.now())
                    return summary
            except Exception as e:
                logger.debug(f"Source {source.name} failed for market summary: {e}")
                if not self._auto_failover:
                    break

        return None

    def get_historical_flow(
        self,
        symbol: str,
        days: int = 30,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[HistoricalForeignFlow]:
        """
        Get historical foreign flow data.

        Args:
            symbol: Stock symbol
            days: Number of days to look back (if start_date not provided)
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            List of HistoricalForeignFlow records
        """
        symbol = symbol.upper()

        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=days)

        # Try each data source
        for source in self._sources:
            try:
                history = source.get_historical_flow(symbol, start_date, end_date)
                if history:
                    return history
            except Exception as e:
                logger.debug(f"Source {source.name} failed for historical data: {e}")
                if not self._auto_failover:
                    break

        return []

    def get_foreign_flow_score(self, symbol: str, lookback_days: int = 5) -> Tuple[float, str]:
        """
        Calculate foreign flow score from -100 to +100.

        Args:
            symbol: Stock symbol
            lookback_days: Days to analyze

        Returns:
            (score, description) tuple
        """
        history = self.get_historical_flow(symbol, days=lookback_days)

        if not history:
            return 0.0, "No data"

        total_net = sum(h.net_value for h in history)
        avg_net = total_net / len(history)

        # Count consecutive days
        consecutive = 0
        direction = None
        for h in reversed(history):
            if direction is None:
                direction = "buy" if h.net_value > 0 else "sell"
            if (direction == "buy" and h.net_value > 0) or (
                direction == "sell" and h.net_value < 0
            ):
                consecutive += 1
            else:
                break

        # Calculate score
        if avg_net > 1_000_000_000:  # > 1 billion avg
            score = min(100, 50 + consecutive * 10)
            desc = f"Strong buying ({consecutive} days)"
        elif avg_net > 100_000_000:  # > 100 million avg
            score = min(70, 30 + consecutive * 8)
            desc = f"Buying ({consecutive} days)"
        elif avg_net > 0:
            score = min(40, 10 + consecutive * 5)
            desc = "Light buying"
        elif avg_net > -100_000_000:
            score = max(-40, -10 - consecutive * 5)
            desc = "Light selling"
        elif avg_net > -1_000_000_000:
            score = max(-70, -30 - consecutive * 8)
            desc = f"Selling ({consecutive} days)"
        else:
            score = max(-100, -50 - consecutive * 10)
            desc = f"Strong selling ({consecutive} days)"

        return score, desc

    def subscribe(self, symbol: str, callback: Callable[[ForeignFlowSnapshot], None]):
        """
        Subscribe to real-time updates for a symbol.

        Args:
            symbol: Stock symbol
            callback: Function to call with updates
        """
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self._subscribers:
                self._subscribers[symbol] = []
            self._subscribers[symbol].append(callback)

    def unsubscribe(self, symbol: str, callback: Optional[Callable] = None):
        """
        Unsubscribe from real-time updates.

        Args:
            symbol: Stock symbol
            callback: Specific callback to remove (or None for all)
        """
        symbol = symbol.upper()
        with self._lock:
            if symbol in self._subscribers:
                if callback is None:
                    del self._subscribers[symbol]
                elif callback in self._subscribers[symbol]:
                    self._subscribers[symbol].remove(callback)

    def start_streaming(self, interval_seconds: float = 30.0):
        """
        Start streaming updates to subscribers.

        Args:
            interval_seconds: Update interval
        """
        if self._streaming_active:
            return

        self._streaming_active = True

        def streaming_worker():
            while self._streaming_active:
                with self._lock:
                    symbols = list(self._subscribers.keys())

                for symbol in symbols:
                    try:
                        snapshot = self.get_current_foreign_flow(symbol, use_cache=False)
                        if snapshot:
                            with self._lock:
                                callbacks = self._subscribers.get(symbol, []).copy()
                            for callback in callbacks:
                                try:
                                    callback(snapshot)
                                except Exception as e:
                                    logger.error(f"Callback error for {symbol}: {e}")
                    except Exception as e:
                        logger.debug(f"Streaming error for {symbol}: {e}")

                time.sleep(interval_seconds)

        self._streaming_thread = threading.Thread(target=streaming_worker, daemon=True)
        self._streaming_thread.start()
        logger.info("📡 Foreign flow streaming started")

    def stop_streaming(self):
        """Stop streaming updates."""
        self._streaming_active = False
        if self._streaming_thread:
            self._streaming_thread.join(timeout=5.0)
        logger.info("📡 Foreign flow streaming stopped")

    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        available = []
        for source in self._sources:
            try:
                if source.is_available():
                    available.append(source.name)
            except Exception:
                pass
        return available

    def check_for_entry(
        self,
        symbol: str,
        min_score: float = -20.0,
    ) -> Tuple[bool, int, str]:
        """
        Check if foreign flow supports entry.

        Args:
            symbol: Stock symbol
            min_score: Minimum score required (-100 to +100)

        Returns:
            (can_enter, confidence_adjustment, message)
        """
        snapshot = self.get_current_foreign_flow(symbol)

        if not snapshot:
            return True, 0, "Foreign flow data unavailable"

        score, desc = self.get_foreign_flow_score(symbol)

        # Calculate confidence adjustment
        if score >= 50:
            adjustment = 15
        elif score >= 20:
            adjustment = 10
        elif score >= 0:
            adjustment = 5
        elif score >= -20:
            adjustment = 0
        elif score >= -50:
            adjustment = -10
        else:
            adjustment = -20

        # Check entry condition
        if score < min_score:
            return False, adjustment, f"Foreign selling pressure: {desc}"

        if not snapshot.remaining_room or snapshot.remaining_room <= 0:
            return False, adjustment, "No foreign room available"

        return True, adjustment, f"Foreign flow: {desc} (score: {score:.0f})"


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_api_instance: Optional[HOSEHNXForeignFlowAPI] = None
_api_lock = Lock()


def get_realtime_foreign_flow_api() -> HOSEHNXForeignFlowAPI:
    """Get singleton instance of the foreign flow API."""
    global _api_instance
    with _api_lock:
        if _api_instance is None:
            _api_instance = HOSEHNXForeignFlowAPI()
        return _api_instance


def reset_foreign_flow_api():
    """Reset the singleton instance (for testing)."""
    global _api_instance
    with _api_lock:
        if _api_instance:
            _api_instance.stop_streaming()
        _api_instance = None


def get_foreign_flow_quick(symbol: str) -> Optional[ForeignFlowSnapshot]:
    """Quick access to foreign flow data."""
    return get_realtime_foreign_flow_api().get_current_foreign_flow(symbol)


def check_foreign_flow_entry(symbol: str, min_score: float = -20.0) -> Tuple[bool, int, str]:
    """Check if foreign flow supports entry."""
    return get_realtime_foreign_flow_api().check_for_entry(symbol, min_score)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING HOSE/HNX FOREIGN FLOW API")
    print("=" * 60)

    api = get_realtime_foreign_flow_api()

    # Check available sources
    sources = api.get_available_sources()
    print(f"\n✅ Available sources: {sources}")

    # Test symbols
    test_symbols = ["VNM", "VCB", "HPG", "FPT", "MWG"]

    print("\n📊 Real-time Foreign Flow:")
    print("-" * 60)

    for symbol in test_symbols:
        flow = api.get_current_foreign_flow(symbol)
        if flow:
            print(
                f"  {symbol}: Net={flow.net_value/1e6:,.0f}M | "
                f"Direction={flow.net_flow_direction} | "
                f"Room={flow.remaining_room:,}"
            )
        else:
            print(f"  {symbol}: No data")

    # Test market summary
    print("\n📈 Market Summary (HOSE):")
    print("-" * 60)

    summary = api.get_market_summary(Exchange.HOSE)
    if summary:
        print(f"  Net Value: {summary.net_value/1e9:,.1f}B VND")
        print(f"  Buy Stocks: {summary.stocks_with_net_buy}")
        print(f"  Sell Stocks: {summary.stocks_with_net_sell}")
        if summary.top_net_buy:
            print(f"  Top Buyers: {summary.top_net_buy[:5]}")
    else:
        print("  No market summary available")

    print("\n" + "=" * 60)
