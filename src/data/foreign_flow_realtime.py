# -*- coding: utf-8 -*-
"""
Real-time Foreign Flow Integration Module

IMPROVEMENT #3.1: Enhanced foreign flow data integration from multiple sources

Sources:
- vnstock library (primary)
- TCBS API (via tcbs_provider)
- SSI iBoard (web scraping fallback)
- CafeF (web scraping fallback)

Features:
- Real-time foreign flow updates
- Historical foreign flow analysis
- Foreign room monitoring
- Smart money flow detection
- Automatic failover between sources

Author: Trading Bot Team
Version: 2.0.0
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from threading import Thread, Event, RLock
from functools import lru_cache
import json
import os

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ForeignFlowData:
    """Comprehensive foreign flow data for a symbol"""

    symbol: str
    date: date

    # Volume data
    buy_volume: int = 0
    sell_volume: int = 0
    net_volume: int = 0

    # Value data (VND)
    buy_value: float = 0.0
    sell_value: float = 0.0
    net_value: float = 0.0

    # Room data
    current_room: float = 0.0  # Remaining room (shares)
    room_percent: float = 0.0  # Room as % of total shares
    max_foreign_percent: float = 0.49  # Default 49% limit
    current_foreign_percent: float = 0.0

    # Metadata
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    is_realtime: bool = False

    @property
    def is_net_buy(self) -> bool:
        return self.net_value > 0

    @property
    def has_room(self) -> bool:
        return self.room_percent > 0.01  # > 1% room available

    @property
    def flow_strength(self) -> str:
        """Classify flow strength"""
        abs_value = abs(self.net_value)
        if abs_value > 10_000_000_000:  # 10B VND
            return "VERY_STRONG"
        elif abs_value > 5_000_000_000:  # 5B VND
            return "STRONG"
        elif abs_value > 1_000_000_000:  # 1B VND
            return "MODERATE"
        else:
            return "WEAK"


@dataclass
class ForeignFlowSignal:
    """Trading signal derived from foreign flow analysis"""

    symbol: str
    signal: str  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    confidence: float  # 0-100
    confidence_adjustment: int  # -20 to +15

    # Flow details
    today_net_value: float = 0.0
    week_net_value: float = 0.0
    month_net_value: float = 0.0

    # Trend
    trend: str = "NEUTRAL"  # ACCUMULATING, NEUTRAL, DISTRIBUTING
    trend_days: int = 0

    # Room status
    has_room: bool = True
    room_message: str = ""

    # Analysis
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# BASE DATA SOURCE
# =============================================================================


class ForeignFlowSource(ABC):
    """Abstract base class for foreign flow data sources"""

    @abstractmethod
    def get_daily_flow(self, symbol: str, date: Optional[date] = None) -> Optional[ForeignFlowData]:
        """Get foreign flow for a specific date"""
        pass

    @abstractmethod
    def get_historical_flow(self, symbol: str, days: int = 30) -> List[ForeignFlowData]:
        """Get historical foreign flow data"""
        pass

    @abstractmethod
    def get_market_flow(self) -> Dict[str, ForeignFlowData]:
        """Get market-wide foreign flow summary"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available"""
        pass


# =============================================================================
# VNSTOCK DATA SOURCE (PRIMARY)
# =============================================================================


class VnstockForeignFlowSource(ForeignFlowSource):
    """Foreign flow data from vnstock library"""

    def __init__(self):
        self._vnstock_available = False
        self._stock_class = None
        self._init_vnstock()

    def _init_vnstock(self):
        """Initialize vnstock library"""
        try:
            from vnstock import Vnstock

            self._stock_class = Vnstock
            self._vnstock_available = True
            logger.info("vnstock library initialized for foreign flow")
        except ImportError:
            logger.warning("vnstock not available, will use fallback sources")
        except Exception as e:
            logger.warning(f"vnstock init error: {e}")

    def is_available(self) -> bool:
        return self._vnstock_available

    def get_daily_flow(self, symbol: str, date: Optional[date] = None) -> Optional[ForeignFlowData]:
        """Get foreign flow for a symbol"""
        if not self._vnstock_available:
            return None

        try:
            stock = self._stock_class().stock(symbol=symbol, source="TCBS")

            # Get intraday data with foreign flow
            end_date = date or datetime.now().date()
            start_date = end_date - timedelta(days=5)

            df = stock.quote.history(
                start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
            )

            if df is None or df.empty:
                return None

            # Get the most recent row
            latest = df.iloc[-1]

            # Extract foreign flow data - column names may vary
            buy_vol = self._safe_get(
                latest, ["foreignBuyVolume", "foreign_buy_volume", "buyForeignVolume"], 0
            )
            sell_vol = self._safe_get(
                latest, ["foreignSellVolume", "foreign_sell_volume", "sellForeignVolume"], 0
            )
            buy_val = self._safe_get(latest, ["foreignBuyValue", "foreign_buy_value"], 0)
            sell_val = self._safe_get(latest, ["foreignSellValue", "foreign_sell_value"], 0)

            # Get close price for value calculation if needed
            close_price = self._safe_get(latest, ["close", "Close"], 0)
            if buy_val == 0 and buy_vol > 0 and close_price > 0:
                buy_val = buy_vol * close_price
            if sell_val == 0 and sell_vol > 0 and close_price > 0:
                sell_val = sell_vol * close_price

            return ForeignFlowData(
                symbol=symbol,
                date=end_date,
                buy_volume=int(buy_vol),
                sell_volume=int(sell_vol),
                net_volume=int(buy_vol - sell_vol),
                buy_value=float(buy_val),
                sell_value=float(sell_val),
                net_value=float(buy_val - sell_val),
                source="vnstock",
                is_realtime=True,
            )

        except Exception as e:
            logger.debug(f"vnstock foreign flow error for {symbol}: {e}")
            return None

    def get_historical_flow(self, symbol: str, days: int = 30) -> List[ForeignFlowData]:
        """Get historical foreign flow data"""
        if not self._vnstock_available:
            return []

        try:
            stock = self._stock_class().stock(symbol=symbol, source="TCBS")

            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days + 10)  # Buffer for non-trading days

            df = stock.quote.history(
                start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
            )

            if df is None or df.empty:
                return []

            results = []
            for _, row in df.iterrows():
                buy_vol = self._safe_get(row, ["foreignBuyVolume", "foreign_buy_volume"], 0)
                sell_vol = self._safe_get(row, ["foreignSellVolume", "foreign_sell_volume"], 0)
                close_price = self._safe_get(row, ["close", "Close"], 0)

                # Parse date
                row_date = self._parse_date(row)
                if row_date is None:
                    continue

                results.append(
                    ForeignFlowData(
                        symbol=symbol,
                        date=row_date,
                        buy_volume=int(buy_vol),
                        sell_volume=int(sell_vol),
                        net_volume=int(buy_vol - sell_vol),
                        buy_value=float(buy_vol * close_price) if close_price else 0,
                        sell_value=float(sell_vol * close_price) if close_price else 0,
                        net_value=float((buy_vol - sell_vol) * close_price) if close_price else 0,
                        source="vnstock",
                    )
                )

            return results[-days:]  # Return last N days

        except Exception as e:
            logger.debug(f"vnstock historical flow error for {symbol}: {e}")
            return []

    def get_market_flow(self) -> Dict[str, ForeignFlowData]:
        """Get market-wide foreign flow - returns top movers"""
        if not self._vnstock_available:
            return {}

        try:
            # Get VN30 stocks foreign flow
            from src.config.constants import VN30_SYMBOLS

            result = {}
            for symbol in VN30_SYMBOLS[:10]:  # Top 10 for speed
                flow = self.get_daily_flow(symbol)
                if flow:
                    result[symbol] = flow

            return result

        except Exception as e:
            logger.debug(f"vnstock market flow error: {e}")
            return {}

    def _safe_get(self, row: pd.Series, columns: List[str], default: Any) -> Any:
        """Safely get value from row with multiple column name options"""
        for col in columns:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    return val
        return default

    def _parse_date(self, row: pd.Series) -> Optional[date]:
        """Parse date from row"""
        for col in ["time", "date", "Date", "tradingDate"]:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    if isinstance(val, (datetime, date)):
                        return val if isinstance(val, date) else val.date()
                    try:
                        return pd.to_datetime(val).date()
                    except:
                        pass
        return None


# =============================================================================
# TCBS DATA SOURCE (SECONDARY)
# =============================================================================


class TCBSForeignFlowSource(ForeignFlowSource):
    """Foreign flow data from TCBS API"""

    def __init__(self):
        self._provider = None
        self._init_provider()

    def _init_provider(self):
        """Initialize TCBS provider"""
        try:
            from src.data.tcbs_provider import TCBSDataProvider

            self._provider = TCBSDataProvider()
            logger.info("TCBS provider initialized for foreign flow")
        except ImportError:
            logger.debug("TCBS provider not available")
        except Exception as e:
            logger.debug(f"TCBS provider init error: {e}")

    def is_available(self) -> bool:
        return self._provider is not None

    def get_daily_flow(self, symbol: str, date: Optional[date] = None) -> Optional[ForeignFlowData]:
        """Get foreign flow from TCBS"""
        if not self._provider:
            return None

        try:
            # Use TCBS provider to get foreign data
            data = self._provider.get_stock_info(symbol)
            if not data:
                return None

            return ForeignFlowData(
                symbol=symbol,
                date=date or datetime.now().date(),
                buy_volume=data.get("foreignBuyVolume", 0),
                sell_volume=data.get("foreignSellVolume", 0),
                net_volume=data.get("foreignBuyVolume", 0) - data.get("foreignSellVolume", 0),
                buy_value=data.get("foreignBuyValue", 0),
                sell_value=data.get("foreignSellValue", 0),
                net_value=data.get("foreignBuyValue", 0) - data.get("foreignSellValue", 0),
                current_room=data.get("foreignRoom", 0),
                room_percent=data.get("foreignRoomPercent", 0),
                current_foreign_percent=data.get("foreignPercent", 0),
                source="tcbs",
                is_realtime=True,
            )
        except Exception as e:
            logger.debug(f"TCBS foreign flow error for {symbol}: {e}")
            return None

    def get_historical_flow(self, symbol: str, days: int = 30) -> List[ForeignFlowData]:
        """Get historical flow from TCBS"""
        # TCBS doesn't provide historical foreign flow directly
        return []

    def get_market_flow(self) -> Dict[str, ForeignFlowData]:
        """Get market flow from TCBS"""
        return {}


# =============================================================================
# MAIN REALTIME FOREIGN FLOW MANAGER
# =============================================================================


class RealtimeForeignFlowManager:
    """
    Real-time Foreign Flow Manager with Multi-Source Integration

    Features:
    - Automatic failover between sources
    - Caching with TTL
    - Background refresh
    - Historical analysis
    - Smart money detection

    Usage:
        manager = RealtimeForeignFlowManager()

        # Get real-time signal
        signal = manager.get_foreign_flow_signal("VNM")

        # Check if entry is recommended
        can_enter, adjustment, msg = manager.check_for_entry("VNM", df)
    """

    CACHE_TTL_SECONDS = 60  # 1 minute cache for real-time data
    CACHE_FILE = "data_cache/foreign_flow_realtime.json"

    def __init__(
        self,
        enable_vnstock: bool = True,
        enable_tcbs: bool = True,
        auto_refresh: bool = False,
        refresh_interval: int = 60,
    ):
        """
        Initialize manager.

        Args:
            enable_vnstock: Enable vnstock source
            enable_tcbs: Enable TCBS source
            auto_refresh: Enable background refresh
            refresh_interval: Refresh interval in seconds
        """
        # Initialize sources
        self._sources: List[ForeignFlowSource] = []

        if enable_vnstock:
            vnstock_source = VnstockForeignFlowSource()
            if vnstock_source.is_available():
                self._sources.append(vnstock_source)

        if enable_tcbs:
            tcbs_source = TCBSForeignFlowSource()
            if tcbs_source.is_available():
                self._sources.append(tcbs_source)

        logger.info(f"📊 Foreign Flow Manager initialized with {len(self._sources)} sources")

        # Cache
        self._cache: Dict[str, ForeignFlowData] = {}
        self._cache_times: Dict[str, datetime] = {}
        self._lock = RLock()

        # Analysis cache
        self._signal_cache: Dict[str, ForeignFlowSignal] = {}
        self._historical_cache: Dict[str, List[ForeignFlowData]] = {}

        # Background refresh
        self._stop_event = Event()
        self._refresh_thread: Optional[Thread] = None
        self._refresh_interval = refresh_interval

        if auto_refresh:
            self.start_background_refresh()

    def get_foreign_flow(self, symbol: str, use_cache: bool = True) -> Optional[ForeignFlowData]:
        """
        Get real-time foreign flow data for a symbol.

        Args:
            symbol: Stock symbol
            use_cache: Use cached data if valid

        Returns:
            ForeignFlowData or None
        """
        symbol = symbol.upper()

        # Check cache
        if use_cache:
            cached = self._get_from_cache(symbol)
            if cached:
                return cached

        # Try each source
        for source in self._sources:
            try:
                data = source.get_daily_flow(symbol)
                if data and (data.buy_volume > 0 or data.sell_volume > 0):
                    self._update_cache(symbol, data)
                    return data
            except Exception as e:
                logger.debug(f"Source {source.__class__.__name__} failed for {symbol}: {e}")
                continue

        return None

    def get_historical_flow(self, symbol: str, days: int = 30) -> List[ForeignFlowData]:
        """
        Get historical foreign flow data.

        Args:
            symbol: Stock symbol
            days: Number of days

        Returns:
            List of ForeignFlowData
        """
        symbol = symbol.upper()

        # Check cache
        cache_key = f"{symbol}_{days}"
        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]

        # Try each source
        for source in self._sources:
            try:
                data = source.get_historical_flow(symbol, days)
                if data:
                    self._historical_cache[cache_key] = data
                    return data
            except Exception as e:
                logger.debug(f"Historical source failed: {e}")
                continue

        return []

    def get_foreign_flow_signal(
        self, symbol: str, df: Optional[pd.DataFrame] = None
    ) -> ForeignFlowSignal:
        """
        Get comprehensive foreign flow signal for trading decisions.

        This is the main method for entry filter integration.

        Args:
            symbol: Stock symbol
            df: Optional DataFrame with price data

        Returns:
            ForeignFlowSignal with trading recommendation
        """
        symbol = symbol.upper()

        # Get real-time flow
        today_flow = self.get_foreign_flow(symbol)

        # Get historical flow for trend analysis
        historical = self.get_historical_flow(symbol, days=20)

        # Analyze
        return self._analyze_foreign_flow(symbol, today_flow, historical, df)

    def _analyze_foreign_flow(
        self,
        symbol: str,
        today: Optional[ForeignFlowData],
        historical: List[ForeignFlowData],
        df: Optional[pd.DataFrame],
    ) -> ForeignFlowSignal:
        """Analyze foreign flow and generate signal"""

        signal = ForeignFlowSignal(
            symbol=symbol, signal="NEUTRAL", confidence=50.0, confidence_adjustment=0
        )

        # If no data, return neutral
        if not today and not historical:
            signal.reasons.append("Không có dữ liệu khối ngoại")
            return signal

        # Calculate aggregates
        if today:
            signal.today_net_value = today.net_value
            signal.has_room = today.has_room
            if not today.has_room:
                signal.room_message = f"Room còn lại: {today.room_percent:.1%}"
                signal.warnings.append(signal.room_message)

        # Weekly aggregate
        if len(historical) >= 5:
            week_data = historical[-5:]
            signal.week_net_value = sum(d.net_value for d in week_data)

        # Monthly aggregate
        if len(historical) >= 20:
            signal.month_net_value = sum(d.net_value for d in historical[-20:])

        # Detect trend
        signal.trend, signal.trend_days = self._detect_trend(historical)

        # Generate signal based on analysis
        self._calculate_signal(signal, today, historical)

        return signal

    def _detect_trend(self, historical: List[ForeignFlowData]) -> Tuple[str, int]:
        """Detect foreign flow trend"""
        if len(historical) < 3:
            return "NEUTRAL", 0

        # Count consecutive buy/sell days
        buy_streak = 0
        sell_streak = 0

        for data in reversed(historical):
            if data.net_value > 0:
                if sell_streak == 0:
                    buy_streak += 1
                else:
                    break
            elif data.net_value < 0:
                if buy_streak == 0:
                    sell_streak += 1
                else:
                    break
            else:
                break

        if buy_streak >= 3:
            return "ACCUMULATING", buy_streak
        elif sell_streak >= 3:
            return "DISTRIBUTING", sell_streak
        else:
            return "NEUTRAL", 0

    def _calculate_signal(
        self,
        signal: ForeignFlowSignal,
        today: Optional[ForeignFlowData],
        historical: List[ForeignFlowData],
    ):
        """Calculate trading signal from foreign flow data"""

        # Thresholds (VND)
        STRONG_THRESHOLD = 5_000_000_000  # 5B
        MODERATE_THRESHOLD = 1_000_000_000  # 1B
        WEEK_STRONG_THRESHOLD = 15_000_000_000  # 15B

        adjustment = 0

        # Today's flow
        if today:
            if today.net_value > STRONG_THRESHOLD:
                signal.signal = "STRONG_BUY"
                adjustment += 10
                signal.reasons.append(f"Strong foreign net buying: {today.net_value/1e9:.1f}B VND")
            elif today.net_value > MODERATE_THRESHOLD:
                signal.signal = "BUY"
                adjustment += 5
                signal.reasons.append(f"Foreign net buying: {today.net_value/1e9:.1f}B VND")
            elif today.net_value < -STRONG_THRESHOLD:
                signal.signal = "STRONG_SELL"
                adjustment -= 15
                signal.warnings.append(
                    f"Strong foreign net selling: {abs(today.net_value)/1e9:.1f}B VND"
                )
            elif today.net_value < -MODERATE_THRESHOLD:
                signal.signal = "SELL"
                adjustment -= 8
                signal.warnings.append(f"Foreign net selling: {abs(today.net_value)/1e9:.1f}B VND")

        # Weekly trend bonus/penalty
        if signal.week_net_value > WEEK_STRONG_THRESHOLD:
            adjustment += 5
            signal.reasons.append(f"5-day buy trend: {signal.week_net_value/1e9:.1f}B VND")
        elif signal.week_net_value < -WEEK_STRONG_THRESHOLD:
            adjustment -= 8
            signal.warnings.append(f"5-day sell trend: {abs(signal.week_net_value)/1e9:.1f}B VND")

        # Trend bonus
        if signal.trend == "ACCUMULATING" and signal.trend_days >= 5:
            adjustment += 3
            signal.reasons.append(f"Accumulating {signal.trend_days} consecutive days")
        elif signal.trend == "DISTRIBUTING" and signal.trend_days >= 5:
            adjustment -= 5
            signal.warnings.append(f"Distributing {signal.trend_days} consecutive days")

        # Room penalty
        if not signal.has_room:
            adjustment -= 5
            if signal.signal in ["STRONG_BUY", "BUY"]:
                signal.signal = "NEUTRAL"  # Downgrade if no room

        # Clamp adjustment
        signal.confidence_adjustment = max(-20, min(15, adjustment))
        signal.confidence = 50 + signal.confidence_adjustment

    def check_for_entry(
        self, symbol: str, df: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, int, str]:
        """
        Check foreign flow conditions for entry filter.

        Designed to integrate with entry_logic.py filter pipeline.

        Args:
            symbol: Stock symbol
            df: Optional price DataFrame

        Returns:
            Tuple of (should_proceed, confidence_adjustment, message)
        """
        try:
            signal = self.get_foreign_flow_signal(symbol, df)

            # Block entry only on strong sell with no room
            if signal.signal == "STRONG_SELL" and not signal.has_room:
                return (
                    False,
                    signal.confidence_adjustment,
                    f"🚫 Khối ngoại bán mạnh: {'; '.join(signal.warnings)}",
                )

            # Build message
            messages = signal.reasons + signal.warnings
            message = "; ".join(messages) if messages else "Khối ngoại: trung lập"

            return (True, signal.confidence_adjustment, message)

        except Exception as e:
            logger.debug(f"Foreign flow check failed for {symbol}: {e}")
            return (True, 0, "Dữ liệu khối ngoại không khả dụng")

    def _get_from_cache(self, symbol: str) -> Optional[ForeignFlowData]:
        """Get data from cache if valid"""
        with self._lock:
            if symbol in self._cache:
                cache_time = self._cache_times.get(symbol)
                if cache_time:
                    age = (datetime.now() - cache_time).total_seconds()
                    if age < self.CACHE_TTL_SECONDS:
                        return self._cache[symbol]
        return None

    def _update_cache(self, symbol: str, data: ForeignFlowData):
        """Update cache with new data"""
        with self._lock:
            self._cache[symbol] = data
            self._cache_times[symbol] = datetime.now()

    def start_background_refresh(self):
        """Start background refresh thread"""
        if self._refresh_thread and self._refresh_thread.is_alive():
            return

        self._stop_event.clear()
        self._refresh_thread = Thread(target=self._background_refresh_loop, daemon=True)
        self._refresh_thread.start()
        logger.info("Foreign flow background refresh started")

    def stop_background_refresh(self):
        """Stop background refresh"""
        self._stop_event.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)
        logger.info("⏹️ Foreign flow background refresh stopped")

    def _background_refresh_loop(self):
        """Background refresh loop"""
        while not self._stop_event.is_set():
            try:
                # Refresh VN30 stocks
                from src.config.constants import VN30_SYMBOLS

                for symbol in VN30_SYMBOLS[:10]:
                    if self._stop_event.is_set():
                        break
                    self.get_foreign_flow(symbol, use_cache=False)
                    time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.debug(f"Background refresh error: {e}")

            self._stop_event.wait(self._refresh_interval)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_manager_instance: Optional[RealtimeForeignFlowManager] = None
_lock = RLock()


def get_foreign_flow_manager() -> RealtimeForeignFlowManager:
    """Get or create singleton foreign flow manager"""
    global _manager_instance
    with _lock:
        if _manager_instance is None:
            _manager_instance = RealtimeForeignFlowManager()
        return _manager_instance


def reset_foreign_flow_manager():
    """Reset manager (for testing)"""
    global _manager_instance
    with _lock:
        if _manager_instance:
            _manager_instance.stop_background_refresh()
        _manager_instance = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_realtime_foreign_flow(symbol: str) -> Optional[ForeignFlowData]:
    """Convenience function to get real-time foreign flow"""
    return get_foreign_flow_manager().get_foreign_flow(symbol)


def check_foreign_flow_for_entry(
    symbol: str, df: Optional[pd.DataFrame] = None
) -> Tuple[bool, int, str]:
    """
    Convenience function for entry filter integration.

    Usage in entry_logic.py:
        from src.data.foreign_flow_realtime import check_foreign_flow_for_entry
        can_enter, adjustment, msg = check_foreign_flow_for_entry(symbol, df)
    """
    return get_foreign_flow_manager().check_for_entry(symbol, df)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING REALTIME FOREIGN FLOW MANAGER")
    print("=" * 60)

    manager = RealtimeForeignFlowManager()

    # Test VN30 stocks
    test_symbols = ["VNM", "VCB", "HPG", "FPT", "MWG"]

    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")

        # Get flow
        flow = manager.get_foreign_flow(symbol)
        if flow:
            print(f"   Buy: {flow.buy_volume:,} shares ({flow.buy_value/1e9:.2f}B)")
            print(f"   Sell: {flow.sell_volume:,} shares ({flow.sell_value/1e9:.2f}B)")
            print(f"   Net: {flow.net_value/1e9:+.2f}B VND ({flow.flow_strength})")

        # Get signal
        signal = manager.get_foreign_flow_signal(symbol)
        print(f"   Signal: {signal.signal}")
        print(f"   Adjustment: {signal.confidence_adjustment:+d}")
        print(f"   Trend: {signal.trend} ({signal.trend_days} days)")

        # Check for entry
        can_enter, adj, msg = manager.check_for_entry(symbol)
        print(f"   Can Enter: {can_enter}")
        print(f"   Message: {msg}")

    print("\nTest complete!")
