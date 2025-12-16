# -*- coding: utf-8 -*-
"""
Enhanced Dividend Calendar with Auto-Update

IMPROVEMENT #3.4: Complete Dividend Capture Strategy

Features:
- SQLite database for dividend events
- Auto-update from multiple sources (CafeF, VNDirect, SSI)
- Ex-dividend date tracking with T+2 consideration
- Dividend yield analysis
- Quality scoring for dividend stocks
- Alert system for upcoming dividends

Vietnam Dividend Calendar Characteristics:
- Annual dividends (1-2 times/year typical)
- Ex-date typically 2-3 weeks after announcement
- T+2 settlement: Must own before ex-date to receive
- Dividend types: Cash (tiền mặt), Stock (cổ phiếu)

Author: Trading Bot Team
Version: 2.0.0
"""

import logging
import sqlite3
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from threading import Thread, Event, RLock
from enum import Enum
import json
import re

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================


class DividendType(Enum):
    """Dividend type"""

    CASH = "CASH"  # Cổ tức tiền mặt
    STOCK = "STOCK"  # Cổ tức cổ phiếu
    MIXED = "MIXED"  # Kết hợp
    RIGHTS = "RIGHTS"  # Quyền mua


class DividendQuality(Enum):
    """Dividend stock quality rating"""

    EXCELLENT = "EXCELLENT"  # Consistent, growing dividends
    GOOD = "GOOD"  # Regular payer
    AVERAGE = "AVERAGE"  # Irregular but pays
    POOR = "POOR"  # Unreliable
    UNKNOWN = "UNKNOWN"


class DividendSignal(Enum):
    """Trading signal for dividend capture"""

    STRONG_BUY = "STRONG_BUY"  # High yield, good quality
    BUY = "BUY"  # Reasonable yield
    HOLD = "HOLD"  # Wait or already own
    AVOID = "AVOID"  # Low yield or risky


@dataclass
class DividendEvent:
    """Single dividend event"""

    id: Optional[int] = None
    symbol: str = ""

    # Dates
    announcement_date: Optional[date] = None
    ex_date: Optional[date] = None  # Ngày GDKHQ
    record_date: Optional[date] = None  # Ngày đăng ký cuối cùng
    payment_date: Optional[date] = None

    # Dividend details
    dividend_type: DividendType = DividendType.CASH
    cash_amount: float = 0.0  # VND per share
    stock_ratio: float = 0.0  # e.g., 10% = 0.10

    # Calculated fields
    current_price: float = 0.0
    dividend_yield: float = 0.0

    # Metadata
    source: str = ""
    is_confirmed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def days_until_ex(self) -> int:
        """Days until ex-dividend date"""
        if self.ex_date:
            return (self.ex_date - date.today()).days
        return 999

    @property
    def is_upcoming(self) -> bool:
        """Check if ex-date is in the future"""
        return self.ex_date and self.ex_date > date.today()

    @property
    def is_actionable(self) -> bool:
        """Check if we can still buy before ex-date (T+2)"""
        if not self.ex_date:
            return False
        # Need to buy at least 2 trading days before ex-date
        return self.days_until_ex >= 3

    @property
    def total_value_per_share(self) -> float:
        """Total dividend value (cash + estimated stock value)"""
        total = self.cash_amount
        if self.stock_ratio > 0 and self.current_price > 0:
            total += self.current_price * self.stock_ratio
        return total

    def calculate_yield(self, price: Optional[float] = None) -> float:
        """Calculate dividend yield"""
        p = price or self.current_price
        if p > 0:
            return self.total_value_per_share / p
        return 0.0


@dataclass
class DividendHistory:
    """Historical dividend data for a stock"""

    symbol: str
    events: List[DividendEvent] = field(default_factory=list)

    # Computed metrics
    total_cash_5y: float = 0.0
    total_stock_5y: float = 0.0
    avg_yield_5y: float = 0.0
    consecutive_years: int = 0
    payout_ratio: float = 0.0
    dividend_growth_rate: float = 0.0
    quality: DividendQuality = DividendQuality.UNKNOWN

    def compute_metrics(self, eps: float = 0.0):
        """Compute metrics from events"""
        if not self.events:
            return

        # Sort by date
        self.events.sort(key=lambda e: e.ex_date or date.min, reverse=True)

        # Last 5 years of data
        five_years_ago = date.today() - timedelta(days=5 * 365)
        recent_events = [e for e in self.events if e.ex_date and e.ex_date >= five_years_ago]

        # Total cash dividends
        self.total_cash_5y = sum(e.cash_amount for e in recent_events)
        self.total_stock_5y = sum(e.stock_ratio for e in recent_events)

        # Average yield
        yields = [e.dividend_yield for e in recent_events if e.dividend_yield > 0]
        self.avg_yield_5y = sum(yields) / len(yields) if yields else 0

        # Consecutive years
        self.consecutive_years = self._count_consecutive_years()

        # Payout ratio
        if eps > 0:
            annual_dividend = sum(e.cash_amount for e in self._get_last_year_events())
            self.payout_ratio = annual_dividend / eps

        # Dividend growth
        self.dividend_growth_rate = self._calculate_growth_rate()

        # Quality rating
        self.quality = self._rate_quality()

    def _count_consecutive_years(self) -> int:
        """Count consecutive years of dividend payments"""
        if not self.events:
            return 0

        years_with_div = set()
        for e in self.events:
            if e.ex_date:
                years_with_div.add(e.ex_date.year)

        if not years_with_div:
            return 0

        current_year = date.today().year
        consecutive = 0
        for year in range(current_year, current_year - 20, -1):
            if year in years_with_div:
                consecutive += 1
            else:
                break

        return consecutive

    def _get_last_year_events(self) -> List[DividendEvent]:
        """Get events from last 12 months"""
        one_year_ago = date.today() - timedelta(days=365)
        return [e for e in self.events if e.ex_date and e.ex_date >= one_year_ago]

    def _calculate_growth_rate(self) -> float:
        """Calculate dividend growth rate (CAGR)"""
        if len(self.events) < 2:
            return 0.0

        # Get annual dividends for last 5 years
        annual_divs = {}
        for e in self.events:
            if e.ex_date:
                year = e.ex_date.year
                annual_divs[year] = annual_divs.get(year, 0) + e.cash_amount

        if len(annual_divs) < 2:
            return 0.0

        years = sorted(annual_divs.keys())
        first_div = annual_divs[years[0]]
        last_div = annual_divs[years[-1]]
        n_years = years[-1] - years[0]

        if first_div > 0 and n_years > 0:
            return (last_div / first_div) ** (1 / n_years) - 1
        return 0.0

    def _rate_quality(self) -> DividendQuality:
        """Rate dividend quality"""
        score = 0

        # Consecutive years (max 30 points)
        score += min(30, self.consecutive_years * 6)

        # Average yield (max 25 points)
        if self.avg_yield_5y >= 0.06:
            score += 25
        elif self.avg_yield_5y >= 0.04:
            score += 20
        elif self.avg_yield_5y >= 0.02:
            score += 10

        # Growth rate (max 25 points)
        if self.dividend_growth_rate > 0.10:
            score += 25
        elif self.dividend_growth_rate > 0.05:
            score += 20
        elif self.dividend_growth_rate > 0:
            score += 10

        # Payout ratio (max 20 points)
        if 0.30 <= self.payout_ratio <= 0.60:
            score += 20
        elif 0.20 <= self.payout_ratio <= 0.70:
            score += 15
        elif self.payout_ratio > 0:
            score += 5

        if score >= 80:
            return DividendQuality.EXCELLENT
        elif score >= 60:
            return DividendQuality.GOOD
        elif score >= 40:
            return DividendQuality.AVERAGE
        else:
            return DividendQuality.POOR


@dataclass
class DividendCaptureRecommendation:
    """Recommendation for dividend capture trade"""

    symbol: str
    event: DividendEvent
    signal: DividendSignal
    quality: DividendQuality

    confidence: float = 50.0

    # Entry details
    days_until_ex: int = 0
    last_buy_date: Optional[date] = None  # Last date to buy (T+2)
    recommended_entry_price: float = 0.0

    # Expected returns
    expected_dividend_yield: float = 0.0
    expected_price_drop: float = 0.0  # Expected drop on ex-date
    expected_net_return: float = 0.0  # Dividend - price drop - costs
    risk_reward_ratio: float = 0.0

    # Notes
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# DATABASE LAYER
# =============================================================================


class DividendDatabase:
    """SQLite database for dividend events"""

    DB_FILE = "data/dividend_calendar.db"

    def __init__(self, db_file: Optional[str] = None):
        self.db_file = db_file or self.DB_FILE
        os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
        self._lock = RLock()
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Dividend events table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dividend_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    announcement_date DATE,
                    ex_date DATE,
                    record_date DATE,
                    payment_date DATE,
                    dividend_type TEXT DEFAULT 'CASH',
                    cash_amount REAL DEFAULT 0,
                    stock_ratio REAL DEFAULT 0,
                    current_price REAL DEFAULT 0,
                    dividend_yield REAL DEFAULT 0,
                    source TEXT,
                    is_confirmed INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, ex_date, dividend_type)
                )
            """
            )

            # Dividend quality cache
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dividend_quality (
                    symbol TEXT PRIMARY KEY,
                    total_cash_5y REAL DEFAULT 0,
                    total_stock_5y REAL DEFAULT 0,
                    avg_yield_5y REAL DEFAULT 0,
                    consecutive_years INTEGER DEFAULT 0,
                    payout_ratio REAL DEFAULT 0,
                    dividend_growth_rate REAL DEFAULT 0,
                    quality TEXT DEFAULT 'UNKNOWN',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Indexes
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_symbol 
                ON dividend_events(symbol)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_ex_date 
                ON dividend_events(ex_date)
            """
            )

            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_file, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def add_event(self, event: DividendEvent) -> bool:
        """Add or update dividend event"""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO dividend_events
                        (symbol, announcement_date, ex_date, record_date, payment_date,
                         dividend_type, cash_amount, stock_ratio, current_price,
                         dividend_yield, source, is_confirmed, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            event.symbol,
                            event.announcement_date,
                            event.ex_date,
                            event.record_date,
                            event.payment_date,
                            event.dividend_type.value,
                            event.cash_amount,
                            event.stock_ratio,
                            event.current_price,
                            event.dividend_yield,
                            event.source,
                            1 if event.is_confirmed else 0,
                            datetime.now(),
                        ),
                    )
                    conn.commit()
                    return True
            except Exception as e:
                logger.error(f"Failed to add dividend event: {e}")
                return False

    def get_upcoming_events(
        self, days_ahead: int = 30, min_yield: float = 0.0
    ) -> List[DividendEvent]:
        """Get upcoming dividend events"""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    end_date = date.today() + timedelta(days=days_ahead)

                    cursor.execute(
                        """
                        SELECT * FROM dividend_events
                        WHERE ex_date >= ? AND ex_date <= ?
                        AND dividend_yield >= ?
                        ORDER BY ex_date ASC
                    """,
                        (date.today(), end_date, min_yield),
                    )

                    return [self._row_to_event(row) for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to get upcoming events: {e}")
                return []

    def get_symbol_history(self, symbol: str, years: int = 5) -> List[DividendEvent]:
        """Get dividend history for a symbol"""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    start_date = date.today() - timedelta(days=years * 365)

                    cursor.execute(
                        """
                        SELECT * FROM dividend_events
                        WHERE symbol = ? AND ex_date >= ?
                        ORDER BY ex_date DESC
                    """,
                        (symbol, start_date),
                    )

                    return [self._row_to_event(row) for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to get symbol history: {e}")
                return []

    def get_event_by_symbol_and_date(self, symbol: str, ex_date: date) -> Optional[DividendEvent]:
        """Get specific event by symbol and ex-date"""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT * FROM dividend_events
                        WHERE symbol = ? AND ex_date = ?
                    """,
                        (symbol, ex_date),
                    )

                    row = cursor.fetchone()
                    return self._row_to_event(row) if row else None
            except Exception as e:
                logger.error(f"Failed to get event: {e}")
                return None

    def save_quality(self, history: DividendHistory):
        """Save dividend quality metrics"""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO dividend_quality
                        (symbol, total_cash_5y, total_stock_5y, avg_yield_5y,
                         consecutive_years, payout_ratio, dividend_growth_rate,
                         quality, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            history.symbol,
                            history.total_cash_5y,
                            history.total_stock_5y,
                            history.avg_yield_5y,
                            history.consecutive_years,
                            history.payout_ratio,
                            history.dividend_growth_rate,
                            history.quality.value,
                            datetime.now(),
                        ),
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to save quality: {e}")

    def get_quality(self, symbol: str) -> Optional[DividendQuality]:
        """Get cached quality rating"""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT quality FROM dividend_quality
                        WHERE symbol = ?
                    """,
                        (symbol,),
                    )

                    row = cursor.fetchone()
                    if row:
                        return DividendQuality(row["quality"])
                    return None
            except Exception as e:
                return None

    def _row_to_event(self, row: sqlite3.Row) -> DividendEvent:
        """Convert database row to DividendEvent"""
        return DividendEvent(
            id=row["id"],
            symbol=row["symbol"],
            announcement_date=self._parse_date(row["announcement_date"]),
            ex_date=self._parse_date(row["ex_date"]),
            record_date=self._parse_date(row["record_date"]),
            payment_date=self._parse_date(row["payment_date"]),
            dividend_type=DividendType(row["dividend_type"]),
            cash_amount=row["cash_amount"] or 0,
            stock_ratio=row["stock_ratio"] or 0,
            current_price=row["current_price"] or 0,
            dividend_yield=row["dividend_yield"] or 0,
            source=row["source"] or "",
            is_confirmed=bool(row["is_confirmed"]),
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.now()
            ),
        )

    def _parse_date(self, date_str: Any) -> Optional[date]:
        """Parse date from database"""
        if not date_str:
            return None
        if isinstance(date_str, date):
            return date_str
        try:
            return datetime.strptime(str(date_str), "%Y-%m-%d").date()
        except:
            return None


# =============================================================================
# DATA SOURCE INTERFACE
# =============================================================================


class DividendDataSource(ABC):
    """Abstract data source for dividend data"""

    @abstractmethod
    def fetch_upcoming_dividends(self) -> List[DividendEvent]:
        """Fetch upcoming dividend events"""
        pass

    @abstractmethod
    def fetch_symbol_history(self, symbol: str) -> List[DividendEvent]:
        """Fetch dividend history for a symbol"""
        pass


# =============================================================================
# VNSTOCK DATA SOURCE
# =============================================================================


class VnstockDividendSource(DividendDataSource):
    """Dividend data from vnstock library"""

    def __init__(self):
        self._vnstock_available = False
        self._stock_class = None
        self._init_vnstock()

    def _init_vnstock(self):
        try:
            from vnstock import Vnstock

            self._stock_class = Vnstock
            self._vnstock_available = True
            logger.info("✅ vnstock dividend source initialized")
        except ImportError:
            logger.warning("⚠️ vnstock not available")

    def fetch_upcoming_dividends(self) -> List[DividendEvent]:
        """Fetch upcoming dividends from vnstock"""
        if not self._vnstock_available:
            return []

        events = []

        try:
            # Get VN30 symbols to check
            from src.config.constants import VN30_SYMBOLS

            for symbol in VN30_SYMBOLS[:15]:  # Check top 15
                try:
                    symbol_events = self.fetch_symbol_history(symbol)
                    # Filter upcoming only
                    upcoming = [e for e in symbol_events if e.is_upcoming]
                    events.extend(upcoming)
                except Exception as e:
                    logger.debug(f"Failed to fetch {symbol}: {e}")

            return events

        except Exception as e:
            logger.error(f"Failed to fetch upcoming dividends: {e}")
            return []

    def fetch_symbol_history(self, symbol: str) -> List[DividendEvent]:
        """Fetch dividend history from vnstock"""
        if not self._vnstock_available:
            return []

        try:
            stock = self._stock_class().stock(symbol=symbol, source="TCBS")

            # Get company events
            events_df = stock.company.events()

            if events_df is None or events_df.empty:
                return []

            results = []

            for _, row in events_df.iterrows():
                # Filter dividend-related events
                event_type = str(row.get("eventType", "")).lower()
                if "dividend" not in event_type and "cổ tức" not in event_type:
                    continue

                # Parse event
                try:
                    ex_date = self._parse_date(row.get("exDate", row.get("eventDate")))

                    if ex_date:
                        event = DividendEvent(
                            symbol=symbol,
                            ex_date=ex_date,
                            announcement_date=self._parse_date(row.get("announcementDate")),
                            record_date=self._parse_date(row.get("recordDate")),
                            payment_date=self._parse_date(row.get("paymentDate")),
                            cash_amount=float(row.get("cashDividend", 0) or 0),
                            stock_ratio=float(row.get("stockDividend", 0) or 0) / 100,
                            dividend_type=self._detect_type(row),
                            source="vnstock",
                        )
                        results.append(event)
                except Exception as e:
                    logger.debug(f"Failed to parse event: {e}")

            return results

        except Exception as e:
            logger.debug(f"vnstock history error for {symbol}: {e}")
            return []

    def _parse_date(self, date_val: Any) -> Optional[date]:
        """Parse date from various formats"""
        if not date_val:
            return None
        if isinstance(date_val, date):
            return date_val
        if isinstance(date_val, datetime):
            return date_val.date()
        try:
            # Try common formats
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"]:
                try:
                    return datetime.strptime(str(date_val), fmt).date()
                except:
                    pass
            return pd.to_datetime(date_val).date()
        except:
            return None

    def _detect_type(self, row: pd.Series) -> DividendType:
        """Detect dividend type from row data"""
        cash = float(row.get("cashDividend", 0) or 0)
        stock = float(row.get("stockDividend", 0) or 0)

        if cash > 0 and stock > 0:
            return DividendType.MIXED
        elif stock > 0:
            return DividendType.STOCK
        else:
            return DividendType.CASH


# =============================================================================
# MAIN DIVIDEND CALENDAR MANAGER
# =============================================================================


class EnhancedDividendCalendar:
    """
    Enhanced Dividend Calendar with auto-update and recommendations.

    Features:
    - SQLite database storage
    - Multiple data source support
    - Auto-update scheduler
    - Dividend quality analysis
    - Capture recommendations

    Usage:
        calendar = EnhancedDividendCalendar()

        # Update from sources
        calendar.update_from_sources()

        # Get recommendations
        recs = calendar.get_capture_recommendations(min_yield=0.03)

        # Check specific symbol
        history = calendar.get_dividend_history("VNM")
    """

    def __init__(
        self,
        db_file: Optional[str] = None,
        auto_update: bool = False,
        update_interval_hours: int = 24,
    ):
        """
        Initialize dividend calendar.

        Args:
            db_file: Database file path
            auto_update: Enable auto-update
            update_interval_hours: Update interval
        """
        self.db = DividendDatabase(db_file)

        # Data sources
        self._sources: List[DividendDataSource] = []
        self._init_sources()

        # Auto-update
        self._stop_event = Event()
        self._update_thread: Optional[Thread] = None
        self._update_interval = update_interval_hours * 3600

        if auto_update:
            self.start_auto_update()

        logger.info("📅 Enhanced Dividend Calendar initialized")

    def _init_sources(self):
        """Initialize data sources"""
        # vnstock source
        vnstock_source = VnstockDividendSource()
        self._sources.append(vnstock_source)

    def update_from_sources(self) -> int:
        """
        Update dividend database from all sources.

        Returns:
            Number of events added/updated
        """
        count = 0

        for source in self._sources:
            try:
                # Fetch upcoming
                events = source.fetch_upcoming_dividends()
                for event in events:
                    if self.db.add_event(event):
                        count += 1

                logger.info(f"📥 Fetched {len(events)} events from {source.__class__.__name__}")

            except Exception as e:
                logger.error(f"Source {source.__class__.__name__} failed: {e}")

        return count

    def update_symbol(self, symbol: str) -> List[DividendEvent]:
        """
        Update dividend data for specific symbol.

        Returns:
            List of events
        """
        all_events = []

        for source in self._sources:
            try:
                events = source.fetch_symbol_history(symbol)
                for event in events:
                    self.db.add_event(event)
                    all_events.append(event)
            except Exception as e:
                logger.debug(f"Failed to update {symbol}: {e}")

        # Update quality metrics
        if all_events:
            history = DividendHistory(symbol=symbol, events=all_events)
            history.compute_metrics()
            self.db.save_quality(history)

        return all_events

    def get_upcoming_dividends(
        self, days_ahead: int = 30, min_yield: float = 0.0
    ) -> List[DividendEvent]:
        """Get upcoming dividend events"""
        return self.db.get_upcoming_events(days_ahead, min_yield)

    def get_dividend_history(self, symbol: str) -> DividendHistory:
        """Get dividend history with quality metrics"""
        events = self.db.get_symbol_history(symbol)
        history = DividendHistory(symbol=symbol, events=events)
        history.compute_metrics()
        return history

    def get_capture_recommendations(
        self, min_yield: float = 0.03, days_ahead: int = 30, max_recommendations: int = 10
    ) -> List[DividendCaptureRecommendation]:
        """
        Get dividend capture trade recommendations.

        Args:
            min_yield: Minimum dividend yield (e.g., 0.03 = 3%)
            days_ahead: Days to look ahead
            max_recommendations: Maximum recommendations

        Returns:
            List of recommendations sorted by score
        """
        recommendations = []

        # Get upcoming events
        events = self.get_upcoming_dividends(days_ahead, min_yield)

        for event in events:
            if not event.is_actionable:
                continue

            rec = self._analyze_capture_opportunity(event)
            if rec and rec.signal != DividendSignal.AVOID:
                recommendations.append(rec)

        # Sort by expected return
        recommendations.sort(key=lambda r: r.expected_net_return, reverse=True)

        return recommendations[:max_recommendations]

    def _analyze_capture_opportunity(
        self, event: DividendEvent
    ) -> Optional[DividendCaptureRecommendation]:
        """Analyze dividend capture opportunity"""
        try:
            # Get quality
            quality = self.db.get_quality(event.symbol)
            if quality is None:
                # Try to compute
                history = self.get_dividend_history(event.symbol)
                quality = history.quality

            # Calculate metrics
            dividend_yield = event.dividend_yield or event.calculate_yield()

            # Expected price drop on ex-date (typically 80-95% of dividend)
            expected_drop = event.total_value_per_share * 0.90
            expected_drop_pct = expected_drop / event.current_price if event.current_price else 0

            # Transaction costs (round trip ~1.5%)
            transaction_cost = 0.015

            # Net expected return
            net_return = dividend_yield - expected_drop_pct - transaction_cost

            # Risk/Reward (potential upside vs downside)
            # Upside: dividend + potential price recovery
            # Downside: price drop + no dividend if timing wrong
            rr_ratio = (
                dividend_yield / (expected_drop_pct + transaction_cost)
                if expected_drop_pct > 0
                else 0
            )

            # Determine signal
            signal = self._determine_signal(dividend_yield, net_return, quality, rr_ratio)

            # Calculate confidence
            confidence = self._calculate_confidence(quality, dividend_yield, event.is_confirmed)

            # Last buy date (T+2)
            last_buy_date = event.ex_date - timedelta(days=3) if event.ex_date else None

            # Build reasons/warnings
            reasons = []
            warnings = []

            if dividend_yield >= 0.06:
                reasons.append(f"High yield: {dividend_yield*100:.1f}%")
            elif dividend_yield >= 0.04:
                reasons.append(f"Good yield: {dividend_yield*100:.1f}%")

            if quality == DividendQuality.EXCELLENT:
                reasons.append("Excellent dividend history")
            elif quality == DividendQuality.GOOD:
                reasons.append("Good dividend history")

            if not event.is_confirmed:
                warnings.append("Event not confirmed")

            if event.days_until_ex <= 3:
                warnings.append("Limited time to enter (T+2)")

            if net_return < 0:
                warnings.append(f"Negative expected return after costs")

            return DividendCaptureRecommendation(
                symbol=event.symbol,
                event=event,
                signal=signal,
                quality=quality,
                confidence=confidence,
                days_until_ex=event.days_until_ex,
                last_buy_date=last_buy_date,
                recommended_entry_price=event.current_price,
                expected_dividend_yield=dividend_yield,
                expected_price_drop=expected_drop_pct,
                expected_net_return=net_return,
                risk_reward_ratio=rr_ratio,
                reasons=reasons,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Failed to analyze {event.symbol}: {e}")
            return None

    def _determine_signal(
        self, yield_pct: float, net_return: float, quality: DividendQuality, rr_ratio: float
    ) -> DividendSignal:
        """Determine trading signal"""
        score = 0

        # Yield score
        if yield_pct >= 0.06:
            score += 30
        elif yield_pct >= 0.04:
            score += 20
        elif yield_pct >= 0.03:
            score += 10

        # Quality score
        if quality == DividendQuality.EXCELLENT:
            score += 30
        elif quality == DividendQuality.GOOD:
            score += 20
        elif quality == DividendQuality.AVERAGE:
            score += 10

        # Net return score
        if net_return > 0.02:
            score += 20
        elif net_return > 0:
            score += 10
        elif net_return < -0.01:
            score -= 20

        # R:R score
        if rr_ratio > 1.5:
            score += 20
        elif rr_ratio > 1.0:
            score += 10

        if score >= 70:
            return DividendSignal.STRONG_BUY
        elif score >= 50:
            return DividendSignal.BUY
        elif score >= 30:
            return DividendSignal.HOLD
        else:
            return DividendSignal.AVOID

    def _calculate_confidence(
        self, quality: DividendQuality, yield_pct: float, is_confirmed: bool
    ) -> float:
        """Calculate recommendation confidence"""
        confidence = 50.0

        # Quality bonus
        if quality == DividendQuality.EXCELLENT:
            confidence += 20
        elif quality == DividendQuality.GOOD:
            confidence += 10
        elif quality == DividendQuality.POOR:
            confidence -= 15

        # Yield bonus
        if yield_pct >= 0.05:
            confidence += 10
        elif yield_pct >= 0.03:
            confidence += 5

        # Confirmation bonus
        if is_confirmed:
            confidence += 10
        else:
            confidence -= 10

        return max(0, min(100, confidence))

    def check_for_trade(self, symbol: str, current_price: float) -> Tuple[bool, int, str]:
        """
        Check if dividend capture is recommended.

        For integration with entry filter.

        Returns:
            (should_consider, confidence_adjustment, message)
        """
        try:
            # Get upcoming events for symbol
            events = self.db.get_upcoming_events(30, 0)
            symbol_events = [e for e in events if e.symbol == symbol]

            if not symbol_events:
                return (True, 0, "No upcoming dividend")

            event = symbol_events[0]  # Most recent

            # Update price
            event.current_price = current_price
            event.dividend_yield = event.calculate_yield(current_price)

            # Analyze
            rec = self._analyze_capture_opportunity(event)

            if rec is None:
                return (True, 0, "Dividend analysis unavailable")

            # Determine adjustment
            adjustment = 0
            if rec.signal == DividendSignal.STRONG_BUY:
                adjustment = 10
            elif rec.signal == DividendSignal.BUY:
                adjustment = 5
            elif rec.signal == DividendSignal.AVOID:
                adjustment = -5

            # Build message
            msg = f"Dividend {event.days_until_ex}d away: {event.dividend_yield*100:.1f}% yield"
            if rec.warnings:
                msg += f" ({'; '.join(rec.warnings)})"

            return (True, adjustment, msg)

        except Exception as e:
            logger.debug(f"Dividend check failed for {symbol}: {e}")
            return (True, 0, "Dividend check error")

    def start_auto_update(self):
        """Start background auto-update"""
        if self._update_thread and self._update_thread.is_alive():
            return

        self._stop_event.clear()
        self._update_thread = Thread(target=self._auto_update_loop, daemon=True)
        self._update_thread.start()
        logger.info("🔄 Dividend calendar auto-update started")

    def stop_auto_update(self):
        """Stop auto-update"""
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=5)
        logger.info("⏹️ Dividend calendar auto-update stopped")

    def _auto_update_loop(self):
        """Background update loop"""
        while not self._stop_event.is_set():
            try:
                count = self.update_from_sources()
                logger.info(f"📅 Auto-updated {count} dividend events")
            except Exception as e:
                logger.error(f"Auto-update failed: {e}")

            self._stop_event.wait(self._update_interval)


# =============================================================================
# SINGLETON
# =============================================================================


_calendar_instance: Optional[EnhancedDividendCalendar] = None
_lock = RLock()


def get_dividend_calendar() -> EnhancedDividendCalendar:
    """Get or create singleton dividend calendar"""
    global _calendar_instance
    with _lock:
        if _calendar_instance is None:
            _calendar_instance = EnhancedDividendCalendar()
        return _calendar_instance


def reset_dividend_calendar():
    """Reset calendar instance"""
    global _calendar_instance
    with _lock:
        if _calendar_instance:
            _calendar_instance.stop_auto_update()
        _calendar_instance = None


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING ENHANCED DIVIDEND CALENDAR")
    print("=" * 60)

    calendar = EnhancedDividendCalendar()

    # Update from sources
    print("\n📥 Updating from sources...")
    count = calendar.update_from_sources()
    print(f"   Updated {count} events")

    # Get upcoming dividends
    print("\n📅 Upcoming dividends:")
    events = calendar.get_upcoming_dividends(days_ahead=60)
    for event in events[:5]:
        print(
            f"   {event.symbol}: Ex-date {event.ex_date}, "
            f"Cash: {event.cash_amount:,.0f}, "
            f"Yield: {event.dividend_yield*100:.1f}%"
        )

    # Get recommendations
    print("\n📊 Capture recommendations:")
    recs = calendar.get_capture_recommendations(min_yield=0.02)
    for rec in recs[:5]:
        print(f"   {rec.symbol}: {rec.signal.value}")
        print(f"      Days to ex: {rec.days_until_ex}")
        print(f"      Yield: {rec.expected_dividend_yield*100:.1f}%")
        print(f"      Net return: {rec.expected_net_return*100:.2f}%")
        print(f"      Quality: {rec.quality.value}")

    # Check specific symbol
    print("\n📋 Testing VNM history:")
    history = calendar.get_dividend_history("VNM")
    print(f"   Events: {len(history.events)}")
    print(f"   Avg yield 5y: {history.avg_yield_5y*100:.1f}%")
    print(f"   Consecutive years: {history.consecutive_years}")
    print(f"   Quality: {history.quality.value}")

    print("\n✅ Test complete!")
