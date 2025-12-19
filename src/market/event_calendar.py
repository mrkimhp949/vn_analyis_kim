# -*- coding: utf-8 -*-
"""
Vietnam Economic Event Calendar

Tracks important economic events and their impact on trading:
- SBV meetings and interest rate decisions
- GDP, CPI, PMI releases
- Earnings seasons
- Derivative expiration dates
- Holiday schedules

Author: Trading Bot Team
Version: 1.0.0
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


class EventImpact(Enum):
    """Event impact level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EventType(Enum):
    """Event type."""

    MONETARY_POLICY = "monetary_policy"  # SBV meetings
    ECONOMIC_DATA = "economic_data"  # GDP, CPI, PMI
    EARNINGS = "earnings"  # Company earnings
    DERIVATIVES = "derivatives"  # Futures/options expiration
    HOLIDAY = "holiday"  # Market holidays
    INDEX_REBALANCE = "index_rebalance"  # VN30 rebalancing
    FOREIGN_POLICY = "foreign_policy"  # Room limits, regulations
    OTHER = "other"


# Vietnam market holidays 2024-2025
VN_HOLIDAYS = {
    # 2024
    date(2024, 1, 1): "New Year's Day",
    date(2024, 2, 8): "Lunar New Year's Eve",
    date(2024, 2, 9): "Lunar New Year",
    date(2024, 2, 10): "Lunar New Year",
    date(2024, 2, 11): "Lunar New Year",
    date(2024, 2, 12): "Lunar New Year",
    date(2024, 2, 13): "Lunar New Year",
    date(2024, 4, 18): "Hung Kings' Festival",
    date(2024, 4, 30): "Reunification Day",
    date(2024, 5, 1): "Labor Day",
    date(2024, 9, 2): "National Day",
    date(2024, 9, 3): "National Day Holiday",
    # 2025
    date(2025, 1, 1): "New Year's Day",
    date(2025, 1, 28): "Lunar New Year's Eve",
    date(2025, 1, 29): "Lunar New Year",
    date(2025, 1, 30): "Lunar New Year",
    date(2025, 1, 31): "Lunar New Year",
    date(2025, 2, 1): "Lunar New Year",
    date(2025, 2, 2): "Lunar New Year",
    date(2025, 2, 3): "Lunar New Year",
    date(2025, 4, 7): "Hung Kings' Festival",
    date(2025, 4, 30): "Reunification Day",
    date(2025, 5, 1): "Labor Day",
    date(2025, 9, 2): "National Day",
}

# VN30 rebalancing dates (3rd Friday of Jan, Apr, Jul, Oct)
VN30_REBALANCE_MONTHS = [1, 4, 7, 10]

# Derivative expiration (3rd Thursday of each month)
DERIVATIVE_EXPIRATION_DAY = 3  # Thursday (0=Monday)
DERIVATIVE_EXPIRATION_WEEK = 3  # 3rd occurrence

# Earnings seasons (approximate)
EARNINGS_SEASONS = {
    1: "Q4 earnings (Jan-Feb)",
    4: "Q1 earnings (Apr-May)",
    7: "Q2 earnings (Jul-Aug)",
    10: "Q3 earnings (Oct-Nov)",
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class EconomicEvent:
    """Economic event."""

    event_date: date
    name: str
    event_type: EventType
    impact: EventImpact
    description: str = ""

    # Time if known
    time: Optional[str] = None

    # Previous/forecast/actual values (for data releases)
    previous_value: Optional[str] = None
    forecast_value: Optional[str] = None
    actual_value: Optional[str] = None

    # Related symbols
    affected_symbols: List[str] = field(default_factory=list)

    # Trading implications
    trading_note: str = ""
    volatility_expected: bool = False

    source: str = ""

    def to_dict(self) -> Dict:
        return {
            "date": self.event_date.isoformat(),
            "name": self.name,
            "type": self.event_type.value,
            "impact": self.impact.value,
            "description": self.description,
            "time": self.time,
            "previous": self.previous_value,
            "forecast": self.forecast_value,
            "actual": self.actual_value,
            "affected_symbols": self.affected_symbols,
            "trading_note": self.trading_note,
            "volatility_expected": self.volatility_expected,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EconomicEvent":
        return cls(
            event_date=date.fromisoformat(data["date"]),
            name=data["name"],
            event_type=EventType(data.get("type", "other")),
            impact=EventImpact(data.get("impact", "medium")),
            description=data.get("description", ""),
            time=data.get("time"),
            previous_value=data.get("previous"),
            forecast_value=data.get("forecast"),
            actual_value=data.get("actual"),
            affected_symbols=data.get("affected_symbols", []),
            trading_note=data.get("trading_note", ""),
            volatility_expected=data.get("volatility_expected", False),
            source=data.get("source", ""),
        )


@dataclass
class EventCalendarSummary:
    """Summary of events for a period."""

    period_start: date
    period_end: date

    total_events: int = 0
    high_impact_count: int = 0

    # By type
    events_by_type: Dict[str, int] = field(default_factory=dict)

    # Key events
    key_events: List[EconomicEvent] = field(default_factory=list)

    # Trading implication
    risk_level: str = "normal"  # low, normal, elevated, high
    position_adjustment: float = 1.0

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "period": f"{self.period_start} to {self.period_end}",
            "total_events": self.total_events,
            "high_impact_count": self.high_impact_count,
            "events_by_type": self.events_by_type,
            "key_events": [e.to_dict() for e in self.key_events[:5]],
            "risk_level": self.risk_level,
            "position_adjustment": self.position_adjustment,
            "notes": self.notes,
        }


# =============================================================================
# EVENT CALENDAR
# =============================================================================


class VietnamEventCalendar:
    """
    Vietnam Economic Event Calendar.

    Tracks and provides access to economic events affecting
    the Vietnam stock market.

    Usage:
        calendar = VietnamEventCalendar()

        # Check upcoming events
        events = calendar.get_upcoming_events(days=7)

        # Check if trading should be adjusted
        check = calendar.check_trading_conditions()
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        auto_generate: bool = True,
    ):
        self._cache_path = cache_path or Path("data_cache/event_calendar.json")

        self._events: List[EconomicEvent] = []
        self._holidays: Dict[date, str] = VN_HOLIDAYS.copy()

        # Load cached events
        self._load_cache()

        # Generate standard events
        if auto_generate:
            self._generate_standard_events()

        logger.info(f"📅 Event Calendar initialized with {len(self._events)} events")

    def _load_cache(self):
        """Load events from cache."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._events = [EconomicEvent.from_dict(e) for e in data.get("events", [])]
                logger.debug(f"Loaded {len(self._events)} cached events")

            except Exception as e:
                logger.warning(f"Failed to load event cache: {e}")

    def _save_cache(self):
        """Save events to cache."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "updated": datetime.now().isoformat(),
                "events": [e.to_dict() for e in self._events],
            }

            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save event cache: {e}")

    def _generate_standard_events(self):
        """Generate standard recurring events."""
        today = date.today()

        # Generate for current year and next year
        for year in [today.year, today.year + 1]:
            self._generate_holiday_events(year)
            self._generate_derivative_expirations(year)
            self._generate_vn30_rebalancing(year)
            self._generate_economic_releases(year)

        # Remove duplicates
        seen = set()
        unique_events = []
        for event in self._events:
            key = (event.event_date, event.name)
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        self._events = sorted(unique_events, key=lambda e: e.event_date)

        self._save_cache()

    def _generate_holiday_events(self, year: int):
        """Generate holiday events."""
        for holiday_date, name in self._holidays.items():
            if holiday_date.year == year:
                self._events.append(
                    EconomicEvent(
                        event_date=holiday_date,
                        name=name,
                        event_type=EventType.HOLIDAY,
                        impact=EventImpact.HIGH,
                        description=f"Market closed for {name}",
                        trading_note="Market closed - no trading",
                        volatility_expected=False,
                        source="calendar",
                    )
                )

    def _get_nth_weekday(self, year: int, month: int, weekday: int, n: int) -> date:
        """Get the nth occurrence of a weekday in a month."""
        first_day = date(year, month, 1)
        first_weekday = first_day + timedelta(days=(weekday - first_day.weekday() + 7) % 7)
        return first_weekday + timedelta(weeks=n - 1)

    def _generate_derivative_expirations(self, year: int):
        """Generate derivative expiration dates."""
        for month in range(1, 13):
            try:
                exp_date = self._get_nth_weekday(
                    year, month, DERIVATIVE_EXPIRATION_DAY, DERIVATIVE_EXPIRATION_WEEK
                )

                if exp_date >= date.today() - timedelta(days=30):
                    self._events.append(
                        EconomicEvent(
                            event_date=exp_date,
                            name=f"VN30F Expiration ({month}/{year})",
                            event_type=EventType.DERIVATIVES,
                            impact=EventImpact.MEDIUM,
                            description="VN30 Index Futures expiration",
                            time="14:30",
                            trading_note="Increased volatility expected, especially near close",
                            volatility_expected=True,
                            source="calendar",
                        )
                    )
            except ValueError:
                pass

    def _generate_vn30_rebalancing(self, year: int):
        """Generate VN30 rebalancing dates."""
        for month in VN30_REBALANCE_MONTHS:
            try:
                # Effective date is usually 3rd Friday
                rebalance_date = self._get_nth_weekday(year, month, 4, 3)  # Friday

                # Announcement is usually 2 weeks before
                announce_date = rebalance_date - timedelta(days=14)

                if announce_date >= date.today() - timedelta(days=30):
                    self._events.append(
                        EconomicEvent(
                            event_date=announce_date,
                            name=f"VN30 Rebalance Announcement (Q{(month-1)//3+1}/{year})",
                            event_type=EventType.INDEX_REBALANCE,
                            impact=EventImpact.MEDIUM,
                            description="VN30 index constituent changes announced",
                            trading_note="Watch for additions/deletions - can cause significant price moves",
                            volatility_expected=True,
                            source="calendar",
                        )
                    )

                    self._events.append(
                        EconomicEvent(
                            event_date=rebalance_date,
                            name=f"VN30 Rebalance Effective (Q{(month-1)//3+1}/{year})",
                            event_type=EventType.INDEX_REBALANCE,
                            impact=EventImpact.HIGH,
                            description="VN30 index changes take effect",
                            trading_note="High volume expected in added/removed stocks",
                            volatility_expected=True,
                            source="calendar",
                        )
                    )
            except ValueError:
                pass

    def _generate_economic_releases(self, year: int):
        """Generate economic data release events."""
        # CPI - usually around 25th of each month
        for month in range(1, 13):
            try:
                cpi_date = date(year, month, 25)
                if cpi_date.weekday() >= 5:  # Weekend
                    cpi_date = cpi_date - timedelta(days=cpi_date.weekday() - 4)

                if cpi_date >= date.today() - timedelta(days=30):
                    self._events.append(
                        EconomicEvent(
                            event_date=cpi_date,
                            name=f"Vietnam CPI ({month}/{year})",
                            event_type=EventType.ECONOMIC_DATA,
                            impact=EventImpact.MEDIUM,
                            description="Monthly Consumer Price Index release",
                            trading_note="May affect banking and consumer stocks",
                            volatility_expected=False,
                            source="calendar",
                        )
                    )
            except ValueError:
                pass

        # GDP - quarterly
        gdp_months = {3: "Q4", 6: "Q1", 9: "Q2", 12: "Q3"}
        for month, quarter in gdp_months.items():
            try:
                gdp_date = date(year, month, 28)
                if gdp_date.weekday() >= 5:
                    gdp_date = gdp_date - timedelta(days=gdp_date.weekday() - 4)

                if gdp_date >= date.today() - timedelta(days=30):
                    self._events.append(
                        EconomicEvent(
                            event_date=gdp_date,
                            name=f"Vietnam GDP {quarter} ({year})",
                            event_type=EventType.ECONOMIC_DATA,
                            impact=EventImpact.HIGH,
                            description=f"Quarterly GDP growth rate for {quarter}",
                            trading_note="Can significantly move market, especially if differs from forecast",
                            volatility_expected=True,
                            source="calendar",
                        )
                    )
            except ValueError:
                pass

        # PMI - first business day of each month
        for month in range(1, 13):
            try:
                pmi_date = date(year, month, 1)
                while pmi_date.weekday() >= 5:  # Skip weekend
                    pmi_date = pmi_date + timedelta(days=1)

                if pmi_date >= date.today() - timedelta(days=30):
                    self._events.append(
                        EconomicEvent(
                            event_date=pmi_date,
                            name=f"Vietnam PMI ({month}/{year})",
                            event_type=EventType.ECONOMIC_DATA,
                            impact=EventImpact.LOW,
                            description="Manufacturing Purchasing Managers Index",
                            trading_note="Watch for industrial sector stocks",
                            volatility_expected=False,
                            source="calendar",
                        )
                    )
            except ValueError:
                pass

    def add_event(self, event: EconomicEvent):
        """Add a custom event."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.event_date)
        self._save_cache()

    def get_events(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        event_types: Optional[List[EventType]] = None,
        impact_filter: Optional[List[EventImpact]] = None,
    ) -> List[EconomicEvent]:
        """
        Get events for a date range.

        Args:
            start_date: Start date (default: today)
            end_date: End date (default: 7 days from start)
            event_types: Filter by event types
            impact_filter: Filter by impact levels
        """
        if start_date is None:
            start_date = date.today()
        if end_date is None:
            end_date = start_date + timedelta(days=7)

        events = [e for e in self._events if start_date <= e.event_date <= end_date]

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        if impact_filter:
            events = [e for e in events if e.impact in impact_filter]

        return events

    def get_upcoming_events(
        self,
        days: int = 7,
        include_holidays: bool = True,
    ) -> List[EconomicEvent]:
        """Get upcoming events."""
        today = date.today()
        events = self.get_events(today, today + timedelta(days=days))

        if not include_holidays:
            events = [e for e in events if e.event_type != EventType.HOLIDAY]

        return events

    def is_holiday(self, check_date: Optional[date] = None) -> bool:
        """Check if a date is a holiday."""
        if check_date is None:
            check_date = date.today()

        # Check weekends
        if check_date.weekday() >= 5:
            return True

        return check_date in self._holidays

    def get_next_trading_day(
        self,
        from_date: Optional[date] = None,
    ) -> date:
        """Get the next trading day."""
        if from_date is None:
            from_date = date.today()

        next_day = from_date + timedelta(days=1)
        while self.is_holiday(next_day):
            next_day = next_day + timedelta(days=1)

        return next_day

    def get_trading_days_until(
        self,
        target_date: date,
        from_date: Optional[date] = None,
    ) -> int:
        """Count trading days until a target date."""
        if from_date is None:
            from_date = date.today()

        count = 0
        current = from_date
        while current < target_date:
            current = current + timedelta(days=1)
            if not self.is_holiday(current):
                count += 1

        return count

    def check_trading_conditions(
        self,
        lookahead_days: int = 3,
    ) -> Dict:
        """
        Check trading conditions based on upcoming events.

        Returns:
            Dict with risk_level, adjustments, and notes
        """
        today = date.today()
        events = self.get_upcoming_events(days=lookahead_days)

        # Check for immediate concerns
        today_events = [e for e in events if e.event_date == today]
        tomorrow_events = [e for e in events if e.event_date == today + timedelta(days=1)]

        high_impact_soon = sum(
            1
            for e in events
            if e.impact == EventImpact.HIGH and e.event_date <= today + timedelta(days=2)
        )

        # Determine risk level
        risk_level = "normal"
        position_adj = 1.0
        notes = []

        # Check holidays
        if self.is_holiday(today):
            risk_level = "closed"
            position_adj = 0.0
            notes.append("Market closed for holiday")

        # Tomorrow is holiday - caution on positions
        elif self.is_holiday(today + timedelta(days=1)):
            risk_level = "elevated"
            position_adj = 0.85
            holiday_name = self._holidays.get(today + timedelta(days=1), "Holiday")
            notes.append(f"Tomorrow is {holiday_name} - consider reducing positions")

        # High impact events soon
        elif high_impact_soon >= 2:
            risk_level = "high"
            position_adj = 0.75
            notes.append(f"{high_impact_soon} high-impact events in next 2 days")
        elif high_impact_soon == 1:
            risk_level = "elevated"
            position_adj = 0.90
            notes.append("High-impact event approaching")

        # Derivative expiration
        derivative_events = [e for e in events if e.event_type == EventType.DERIVATIVES]
        if derivative_events:
            risk_level = max(
                risk_level, "elevated", key=["normal", "elevated", "high", "closed"].index
            )
            position_adj = min(position_adj, 0.90)
            notes.append("Derivative expiration approaching")

        # VN30 rebalancing
        rebalance_events = [e for e in events if e.event_type == EventType.INDEX_REBALANCE]
        if rebalance_events:
            notes.append("VN30 rebalancing event approaching")

        # Build result
        result = {
            "date": today.isoformat(),
            "is_trading_day": not self.is_holiday(today),
            "risk_level": risk_level,
            "position_adjustment": position_adj,
            "high_impact_events": high_impact_soon,
            "upcoming_events": len(events),
            "today_events": [e.to_dict() for e in today_events],
            "notes": notes,
        }

        return result

    def get_week_summary(self) -> EventCalendarSummary:
        """Get summary for the upcoming week."""
        today = date.today()
        end = today + timedelta(days=7)

        events = self.get_events(today, end)

        summary = EventCalendarSummary(
            period_start=today,
            period_end=end,
            total_events=len(events),
            high_impact_count=sum(1 for e in events if e.impact == EventImpact.HIGH),
        )

        # Count by type
        type_counts: Dict[str, int] = {}
        for event in events:
            type_str = event.event_type.value
            type_counts[type_str] = type_counts.get(type_str, 0) + 1
        summary.events_by_type = type_counts

        # Key events (high impact)
        summary.key_events = [e for e in events if e.impact == EventImpact.HIGH][:5]

        # Determine risk level
        if summary.high_impact_count >= 3:
            summary.risk_level = "high"
            summary.position_adjustment = 0.75
        elif summary.high_impact_count >= 1:
            summary.risk_level = "elevated"
            summary.position_adjustment = 0.90
        else:
            summary.risk_level = "normal"
            summary.position_adjustment = 1.0

        # Notes
        if summary.high_impact_count > 0:
            summary.notes.append(f"{summary.high_impact_count} high-impact events this week")

        holidays = sum(1 for e in events if e.event_type == EventType.HOLIDAY)
        if holidays > 0:
            summary.notes.append(f"{holidays} market holiday(s)")

        return summary


# =============================================================================
# SINGLETON
# =============================================================================

_calendar_instance: Optional[VietnamEventCalendar] = None


def get_event_calendar() -> VietnamEventCalendar:
    """Get singleton calendar instance."""
    global _calendar_instance
    if _calendar_instance is None:
        _calendar_instance = VietnamEventCalendar()
    return _calendar_instance


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("🧪 TESTING VIETNAM EVENT CALENDAR")
    print("=" * 70 + "\n")

    calendar = get_event_calendar()

    # Test upcoming events
    print("1️⃣ Upcoming Events (7 days):")
    events = calendar.get_upcoming_events(days=7)
    for event in events[:5]:
        print(f"   📅 {event.event_date}: {event.name} [{event.impact.value}]")

    # Test trading conditions
    print("\n2️⃣ Trading Conditions Check:")
    conditions = calendar.check_trading_conditions()
    print(f"   Risk Level: {conditions['risk_level']}")
    print(f"   Position Adj: {conditions['position_adjustment']}")
    print(f"   Notes: {conditions['notes']}")

    # Test week summary
    print("\n3️⃣ Week Summary:")
    summary = calendar.get_week_summary()
    print(f"   Total Events: {summary.total_events}")
    print(f"   High Impact: {summary.high_impact_count}")
    print(f"   Risk Level: {summary.risk_level}")

    # Test holiday check
    print("\n4️⃣ Holiday Check:")
    print(f"   Today is holiday: {calendar.is_holiday()}")
    print(f"   Next trading day: {calendar.get_next_trading_day()}")

    print("\n" + "=" * 70)
    print("✅ Vietnam Event Calendar testing complete!")
    print("=" * 70)
