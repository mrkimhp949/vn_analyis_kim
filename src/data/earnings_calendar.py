# -*- coding: utf-8 -*-
"""
Earnings Calendar & Corporate Events for Vietnam Stock Market

Tracks:
- Quarterly earnings announcements (BCTC)
- Annual reports
- Ex-dividend dates
- Rights issues
- Stock splits
- AGM/EGM dates

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import json
import os

import pandas as pd

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Corporate event types"""

    EARNINGS_Q1 = "EARNINGS_Q1"
    EARNINGS_Q2 = "EARNINGS_Q2"
    EARNINGS_Q3 = "EARNINGS_Q3"
    EARNINGS_Q4 = "EARNINGS_Q4"
    EARNINGS_ANNUAL = "EARNINGS_ANNUAL"
    EX_DIVIDEND_CASH = "EX_DIVIDEND_CASH"
    EX_DIVIDEND_STOCK = "EX_DIVIDEND_STOCK"
    RIGHTS_ISSUE = "RIGHTS_ISSUE"
    STOCK_SPLIT = "STOCK_SPLIT"
    AGM = "AGM"  # Annual General Meeting
    EGM = "EGM"  # Extraordinary General Meeting
    LISTING = "LISTING"
    DELISTING = "DELISTING"


class EventImpact(Enum):
    """Expected impact level"""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


@dataclass
class CorporateEvent:
    """Corporate event data"""

    symbol: str
    event_type: EventType
    event_date: datetime

    # Event details
    title: str = ""
    description: str = ""
    fiscal_period: str = ""  # Q1 2024, FY 2024, etc.

    # For earnings
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None

    # For dividends
    dividend_amount: Optional[float] = None  # VND per share
    dividend_yield: Optional[float] = None
    record_date: Optional[datetime] = None
    payment_date: Optional[datetime] = None

    # For rights/splits
    ratio: Optional[str] = None  # e.g., "10:1" for split
    subscription_price: Optional[float] = None

    # Metadata
    impact: EventImpact = EventImpact.UNKNOWN
    is_confirmed: bool = False
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def days_until(self) -> int:
        """Days until event"""
        return (self.event_date - datetime.now()).days

    @property
    def is_upcoming(self) -> bool:
        """Check if event is in the future"""
        return self.event_date > datetime.now()

    @property
    def is_earnings(self) -> bool:
        """Check if this is an earnings event"""
        return self.event_type in [
            EventType.EARNINGS_Q1,
            EventType.EARNINGS_Q2,
            EventType.EARNINGS_Q3,
            EventType.EARNINGS_Q4,
            EventType.EARNINGS_ANNUAL,
        ]

    @property
    def is_dividend(self) -> bool:
        """Check if this is a dividend event"""
        return self.event_type in [EventType.EX_DIVIDEND_CASH, EventType.EX_DIVIDEND_STOCK]


@dataclass
class EarningsRiskAssessment:
    """Earnings risk assessment for trading"""

    symbol: str
    has_upcoming_earnings: bool
    days_until_earnings: int

    # Risk levels
    risk_level: str  # HIGH, MEDIUM, LOW, NONE
    position_multiplier: float  # 0.0 to 1.0

    # Recommendations
    should_avoid_entry: bool
    should_reduce_position: bool
    should_exit_before: bool

    # Details
    event: Optional[CorporateEvent] = None
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class EarningsCalendarManager:
    """
    Manages earnings calendar and corporate events

    Vietnam Earnings Schedule (typical):
    - Q1: April 20-30 (deadline: April 30)
    - Q2: July 20-30 (deadline: July 30)
    - Q3: October 20-30 (deadline: October 30)
    - Q4/Annual: January 20 - March 31 (deadline: March 31)

    Trading Rules:
    - Avoid entry 5 days before earnings
    - Reduce position 3 days before
    - Consider exit 1 day before for uncertain positions
    """

    # Vietnam earnings deadlines (day of month)
    EARNINGS_DEADLINES = {
        "Q1": {"month": 4, "day": 30},  # April 30
        "Q2": {"month": 7, "day": 30},  # July 30
        "Q3": {"month": 10, "day": 30},  # October 30
        "Q4": {"month": 3, "day": 31},  # March 31 (next year)
    }

    # Risk thresholds
    HIGH_RISK_DAYS = 3  # Days before earnings = HIGH risk
    MEDIUM_RISK_DAYS = 5  # Days before earnings = MEDIUM risk
    LOW_RISK_DAYS = 10  # Days before earnings = LOW risk

    def __init__(self, cache_file: str = "data_cache/earnings_calendar.json"):
        self._events: Dict[str, List[CorporateEvent]] = {}
        self._cache_file = cache_file
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached events from file"""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for symbol, events in data.items():
                        self._events[symbol] = [self._dict_to_event(e) for e in events]
                logger.info(f"Loaded {len(self._events)} symbols from earnings cache")
        except Exception as e:
            logger.warning(f"Failed to load earnings cache: {e}")

    def _save_cache(self) -> None:
        """Save events to cache file"""
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            data = {}
            for symbol, events in self._events.items():
                data[symbol] = [self._event_to_dict(e) for e in events]
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save earnings cache: {e}")

    def _event_to_dict(self, event: CorporateEvent) -> Dict:
        """Convert event to dictionary"""
        return {
            "symbol": event.symbol,
            "event_type": event.event_type.value,
            "event_date": event.event_date.isoformat(),
            "title": event.title,
            "description": event.description,
            "fiscal_period": event.fiscal_period,
            "eps_estimate": event.eps_estimate,
            "eps_actual": event.eps_actual,
            "dividend_amount": event.dividend_amount,
            "dividend_yield": event.dividend_yield,
            "record_date": event.record_date.isoformat() if event.record_date else None,
            "payment_date": event.payment_date.isoformat() if event.payment_date else None,
            "ratio": event.ratio,
            "impact": event.impact.value,
            "is_confirmed": event.is_confirmed,
            "source": event.source,
        }

    def _dict_to_event(self, data: Dict) -> CorporateEvent:
        """Convert dictionary to event"""
        return CorporateEvent(
            symbol=data["symbol"],
            event_type=EventType(data["event_type"]),
            event_date=datetime.fromisoformat(data["event_date"]),
            title=data.get("title", ""),
            description=data.get("description", ""),
            fiscal_period=data.get("fiscal_period", ""),
            eps_estimate=data.get("eps_estimate"),
            eps_actual=data.get("eps_actual"),
            dividend_amount=data.get("dividend_amount"),
            dividend_yield=data.get("dividend_yield"),
            record_date=(
                datetime.fromisoformat(data["record_date"]) if data.get("record_date") else None
            ),
            payment_date=(
                datetime.fromisoformat(data["payment_date"]) if data.get("payment_date") else None
            ),
            ratio=data.get("ratio"),
            impact=EventImpact(data.get("impact", "UNKNOWN")),
            is_confirmed=data.get("is_confirmed", False),
            source=data.get("source", ""),
        )

    def add_event(self, event: CorporateEvent) -> None:
        """Add a corporate event"""
        if event.symbol not in self._events:
            self._events[event.symbol] = []

        # Check for duplicates
        for existing in self._events[event.symbol]:
            if (
                existing.event_type == event.event_type
                and existing.event_date.date() == event.event_date.date()
            ):
                # Update existing
                self._events[event.symbol].remove(existing)
                break

        self._events[event.symbol].append(event)
        self._events[event.symbol].sort(key=lambda e: e.event_date)
        self._save_cache()

        logger.info(
            f"Added event: {event.symbol} - {event.event_type.value} on {event.event_date.date()}"
        )

    def get_events(
        self,
        symbol: str,
        event_types: List[EventType] = None,
        days_ahead: int = 30,
        include_past: bool = False,
    ) -> List[CorporateEvent]:
        """
        Get events for a symbol

        Args:
            symbol: Stock symbol
            event_types: Filter by event types (None = all)
            days_ahead: Days to look ahead
            include_past: Include past events

        Returns:
            List of CorporateEvent
        """
        if symbol not in self._events:
            # Try to fetch from external source
            self._fetch_events(symbol)

        events = self._events.get(symbol, [])

        # Filter by date
        now = datetime.now()
        cutoff = now + timedelta(days=days_ahead)

        filtered = []
        for event in events:
            if not include_past and event.event_date < now:
                continue
            if event.event_date > cutoff:
                continue
            if event_types and event.event_type not in event_types:
                continue
            filtered.append(event)

        return filtered

    def get_upcoming_earnings(self, symbol: str, days_ahead: int = 30) -> List[CorporateEvent]:
        """Get upcoming earnings events"""
        earnings_types = [
            EventType.EARNINGS_Q1,
            EventType.EARNINGS_Q2,
            EventType.EARNINGS_Q3,
            EventType.EARNINGS_Q4,
            EventType.EARNINGS_ANNUAL,
        ]
        return self.get_events(symbol, earnings_types, days_ahead)

    def get_upcoming_dividends(self, symbol: str, days_ahead: int = 30) -> List[CorporateEvent]:
        """Get upcoming dividend events"""
        dividend_types = [EventType.EX_DIVIDEND_CASH, EventType.EX_DIVIDEND_STOCK]
        return self.get_events(symbol, dividend_types, days_ahead)

    def _fetch_events(self, symbol: str) -> None:
        """Fetch events from external sources (TCBS, VNDirect, etc.)"""
        # Try TCBS for real data
        self._fetch_from_tcbs(symbol)

        # Try VNDirect as backup
        self._fetch_from_vndirect(symbol)

        # Estimate earnings dates based on Vietnam schedule (fallback)
        self._estimate_earnings_dates(symbol)

    def _fetch_from_tcbs(self, symbol: str) -> None:
        """Fetch events from TCBS API"""
        try:
            from src.data.tcbs_provider import get_tcbs_provider
            import requests

            provider = get_tcbs_provider()

            # Get dividend info
            dividend_info = provider.get_dividend_info(symbol)
            if dividend_info and dividend_info.get("ex_date"):
                event = CorporateEvent(
                    symbol=symbol,
                    event_type=EventType.EX_DIVIDEND_CASH,
                    event_date=dividend_info["ex_date"],
                    dividend_amount=dividend_info.get("cash_dividend"),
                    dividend_yield=dividend_info.get("dividend_yield"),
                    source="TCBS",
                    is_confirmed=True,
                )
                self.add_event(event)

            # Try to get earnings/financial report dates from TCBS
            self._fetch_tcbs_financial_reports(symbol)

        except Exception as e:
            logger.debug(f"TCBS event fetch failed for {symbol}: {e}")

    def _fetch_tcbs_financial_reports(self, symbol: str) -> None:
        """Fetch financial report dates from TCBS API"""
        try:
            import requests

            # TCBS financial report API
            url = f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{symbol}/financialreport"
            params = {"yearly": 0, "isAll": False}  # quarterly reports

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                reports = data if isinstance(data, list) else []

                for report in reports[:4]:  # Last 4 quarters
                    try:
                        # Extract report date
                        year = report.get("year", 0)
                        quarter = report.get("quarter", 0)

                        if year and quarter:
                            # Estimate report release date based on quarter
                            # Q1: April, Q2: July, Q3: October, Q4: January next year
                            report_months = {1: 4, 2: 7, 3: 10, 4: 1}
                            month = report_months.get(quarter, 4)
                            report_year = year if quarter != 4 else year + 1

                            # Typical release: 20-25th of the month
                            report_date = datetime(report_year, month, 23)

                            # Only add if in future
                            if report_date > datetime.now():
                                event_type = {
                                    1: EventType.EARNINGS_Q1,
                                    2: EventType.EARNINGS_Q2,
                                    3: EventType.EARNINGS_Q3,
                                    4: EventType.EARNINGS_Q4,
                                }.get(quarter, EventType.EARNINGS_Q1)

                                event = CorporateEvent(
                                    symbol=symbol,
                                    event_type=event_type,
                                    event_date=report_date,
                                    fiscal_period=f"Q{quarter} {year}",
                                    title=f"Q{quarter} {year} Financial Report",
                                    impact=EventImpact.HIGH,
                                    is_confirmed=False,  # Estimated from historical pattern
                                    source="TCBS_ESTIMATED",
                                )

                                # Check if already exists
                                if symbol not in self._events:
                                    self._events[symbol] = []

                                exists = any(
                                    e.event_type == event_type
                                    and e.fiscal_period == event.fiscal_period
                                    for e in self._events[symbol]
                                )

                                if not exists:
                                    self._events[symbol].append(event)

                    except Exception as e:
                        logger.debug(f"Error parsing report: {e}")

        except Exception as e:
            logger.debug(f"TCBS financial reports fetch failed: {e}")

    def _fetch_from_vndirect(self, symbol: str) -> None:
        """Fetch events from VNDirect API"""
        try:
            import requests

            # VNDirect events API
            url = "https://finfo-api.vndirect.com.vn/v4/events"
            params = {"q": f"code:{symbol}", "size": 20, "sort": "eventDate"}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                events = data.get("data", [])

                for item in events:
                    try:
                        event_date_str = item.get("eventDate")
                        event_type_str = item.get("eventType", "").upper()

                        if not event_date_str:
                            continue

                        # Parse date
                        try:
                            event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
                        except ValueError:
                            continue

                        # Only future events
                        if event_date <= datetime.now():
                            continue

                        # Map event type
                        event_type = None
                        if "DIVIDEND" in event_type_str or "CASH" in event_type_str:
                            event_type = EventType.EX_DIVIDEND_CASH
                        elif "STOCK" in event_type_str and "DIVIDEND" in event_type_str:
                            event_type = EventType.EX_DIVIDEND_STOCK
                        elif "AGM" in event_type_str or "ANNUAL" in event_type_str:
                            event_type = EventType.AGM
                        elif "EGM" in event_type_str or "EXTRAORDINARY" in event_type_str:
                            event_type = EventType.EGM
                        elif "RIGHTS" in event_type_str:
                            event_type = EventType.RIGHTS_ISSUE

                        if event_type:
                            event = CorporateEvent(
                                symbol=symbol,
                                event_type=event_type,
                                event_date=event_date,
                                title=item.get("eventTitle", ""),
                                description=item.get("eventContent", ""),
                                dividend_amount=item.get("cashDividend"),
                                ratio=item.get("ratio"),
                                is_confirmed=True,
                                source="VNDirect",
                            )
                            self.add_event(event)

                    except Exception as e:
                        logger.debug(f"Error parsing VNDirect event: {e}")

        except Exception as e:
            logger.debug(f"VNDirect events fetch failed: {e}")

    def _estimate_earnings_dates(self, symbol: str) -> None:
        """Estimate earnings dates based on Vietnam schedule"""
        now = datetime.now()
        year = now.year

        # Generate estimated earnings dates
        for quarter, deadline in self.EARNINGS_DEADLINES.items():
            month = deadline["month"]
            day = deadline["day"]

            # Adjust year for Q4 (reports in next year)
            event_year = year if quarter != "Q4" else year + 1
            if quarter == "Q4" and now.month > 3:
                event_year = year + 1

            # Estimate date (typically 5-10 days before deadline)
            try:
                deadline_date = datetime(event_year, month, day)
                estimated_date = deadline_date - timedelta(days=7)

                # Only add if in future
                if estimated_date > now:
                    event_type = {
                        "Q1": EventType.EARNINGS_Q1,
                        "Q2": EventType.EARNINGS_Q2,
                        "Q3": EventType.EARNINGS_Q3,
                        "Q4": EventType.EARNINGS_Q4,
                    }[quarter]

                    event = CorporateEvent(
                        symbol=symbol,
                        event_type=event_type,
                        event_date=estimated_date,
                        fiscal_period=f"{quarter} {event_year}",
                        title=f"Estimated {quarter} {event_year} Earnings",
                        impact=EventImpact.HIGH,
                        is_confirmed=False,
                        source="ESTIMATED",
                    )

                    # Only add if not already exists
                    if symbol not in self._events:
                        self._events[symbol] = []

                    exists = any(
                        e.event_type == event_type and e.fiscal_period == event.fiscal_period
                        for e in self._events[symbol]
                    )

                    if not exists:
                        self._events[symbol].append(event)
            except ValueError:
                pass

    def assess_earnings_risk(
        self, symbol: str, position_type: str = "ENTRY"  # ENTRY, HOLD, EXIT
    ) -> EarningsRiskAssessment:
        """
        Assess earnings risk for trading decisions

        Args:
            symbol: Stock symbol
            position_type: Type of position decision

        Returns:
            EarningsRiskAssessment
        """
        warnings = []
        recommendations = []

        # Get upcoming earnings
        earnings = self.get_upcoming_earnings(symbol, days_ahead=self.LOW_RISK_DAYS)

        if not earnings:
            return EarningsRiskAssessment(
                symbol=symbol,
                has_upcoming_earnings=False,
                days_until_earnings=999,
                risk_level="NONE",
                position_multiplier=1.0,
                should_avoid_entry=False,
                should_reduce_position=False,
                should_exit_before=False,
                recommendations=["No upcoming earnings - normal trading"],
            )

        # Get nearest earnings
        nearest = earnings[0]
        days_until = nearest.days_until

        # Determine risk level
        if days_until <= self.HIGH_RISK_DAYS:
            risk_level = "HIGH"
            position_multiplier = 0.3
            should_avoid_entry = True
            should_reduce_position = True
            should_exit_before = True
            warnings.append(f"⚠️ Earnings in {days_until} days - HIGH RISK")
            recommendations.append("Avoid new entries")
            recommendations.append("Consider exiting uncertain positions")
            recommendations.append("Reduce position size to 30%")

        elif days_until <= self.MEDIUM_RISK_DAYS:
            risk_level = "MEDIUM"
            position_multiplier = 0.5
            should_avoid_entry = True
            should_reduce_position = True
            should_exit_before = False
            warnings.append(f"⚠️ Earnings in {days_until} days - MEDIUM RISK")
            recommendations.append("Avoid new entries")
            recommendations.append("Reduce position size to 50%")

        elif days_until <= self.LOW_RISK_DAYS:
            risk_level = "LOW"
            position_multiplier = 0.7
            should_avoid_entry = False
            should_reduce_position = False
            should_exit_before = False
            warnings.append(f"ℹ️ Earnings in {days_until} days - LOW RISK")
            recommendations.append("Monitor position closely")
            recommendations.append("Consider reducing size to 70%")

        else:
            risk_level = "NONE"
            position_multiplier = 1.0
            should_avoid_entry = False
            should_reduce_position = False
            should_exit_before = False

        # Add event details
        if nearest.is_confirmed:
            recommendations.append(f"Confirmed earnings: {nearest.fiscal_period}")
        else:
            warnings.append("Earnings date is estimated - may change")

        return EarningsRiskAssessment(
            symbol=symbol,
            has_upcoming_earnings=True,
            days_until_earnings=days_until,
            risk_level=risk_level,
            position_multiplier=position_multiplier,
            should_avoid_entry=should_avoid_entry,
            should_reduce_position=should_reduce_position,
            should_exit_before=should_exit_before,
            event=nearest,
            warnings=warnings,
            recommendations=recommendations,
        )

    def get_all_upcoming_events(
        self, symbols: List[str], days_ahead: int = 14
    ) -> List[CorporateEvent]:
        """Get all upcoming events for multiple symbols"""
        all_events = []
        for symbol in symbols:
            events = self.get_events(symbol, days_ahead=days_ahead)
            all_events.extend(events)

        # Sort by date
        all_events.sort(key=lambda e: e.event_date)
        return all_events

    def get_earnings_calendar_df(self, symbols: List[str], days_ahead: int = 30) -> pd.DataFrame:
        """Get earnings calendar as DataFrame"""
        events = []
        for symbol in symbols:
            earnings = self.get_upcoming_earnings(symbol, days_ahead)
            for e in earnings:
                events.append(
                    {
                        "Symbol": e.symbol,
                        "Event": e.event_type.value,
                        "Date": e.event_date.strftime("%Y-%m-%d"),
                        "Days Until": e.days_until,
                        "Period": e.fiscal_period,
                        "Confirmed": "Yes" if e.is_confirmed else "Estimated",
                        "Impact": e.impact.value,
                    }
                )

        if not events:
            return pd.DataFrame()

        df = pd.DataFrame(events)
        df = df.sort_values("Days Until")
        return df


# Singleton instance
_earnings_manager: Optional[EarningsCalendarManager] = None


def get_earnings_manager() -> EarningsCalendarManager:
    """Get singleton earnings calendar manager"""
    global _earnings_manager
    if _earnings_manager is None:
        _earnings_manager = EarningsCalendarManager()
    return _earnings_manager


# Convenience functions
def is_near_earnings(symbol: str, days: int = 5) -> Tuple[bool, Optional[CorporateEvent]]:
    """Quick check if symbol is near earnings"""
    manager = get_earnings_manager()
    assessment = manager.assess_earnings_risk(symbol)

    if assessment.has_upcoming_earnings and assessment.days_until_earnings <= days:
        return True, assessment.event
    return False, None


def get_earnings_risk_multiplier(symbol: str) -> float:
    """Get position size multiplier based on earnings risk"""
    manager = get_earnings_manager()
    assessment = manager.assess_earnings_risk(symbol)
    return assessment.position_multiplier


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("🧪 TESTING EARNINGS CALENDAR")
    print("=" * 70 + "\n")

    manager = get_earnings_manager()

    # Test symbols
    test_symbols = ["VNM", "VCB", "FPT", "HPG"]

    for symbol in test_symbols:
        print(f"\n📊 {symbol}:")

        # Get earnings risk
        risk = manager.assess_earnings_risk(symbol)

        print(f"   Has Upcoming Earnings: {risk.has_upcoming_earnings}")
        if risk.has_upcoming_earnings:
            print(f"   Days Until: {risk.days_until_earnings}")
            print(f"   Risk Level: {risk.risk_level}")
            print(f"   Position Multiplier: {risk.position_multiplier:.0%}")
            print(f"   Avoid Entry: {risk.should_avoid_entry}")

            if risk.warnings:
                print(f"   Warnings:")
                for w in risk.warnings:
                    print(f"      {w}")

    # Print calendar
    print("\n" + "-" * 70)
    print("📅 EARNINGS CALENDAR (Next 30 days)")
    print("-" * 70)

    df = manager.get_earnings_calendar_df(test_symbols, days_ahead=30)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No upcoming earnings")

    print("\n" + "=" * 70)
    print("✅ Earnings calendar test completed!")
    print("=" * 70)
