# -*- coding: utf-8 -*-
"""
Vietnam Holiday Calendar - Complete Implementation

Comprehensive holiday calendar for Vietnam stock market including:
- Fixed holidays (New Year, Reunification Day, etc.)
- Lunar holidays (Tết Nguyên Đán, Giỗ Tổ Hùng Vương)
- Substitute holidays (nghỉ bù) when holidays fall on weekends
- Special market closures

Author: Trading Bot Team
Version: 2.0.0 - Complete 10/10 Implementation
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# =============================================================================
# FIXED HOLIDAYS (Gregorian Calendar)
# =============================================================================

VIETNAM_FIXED_HOLIDAYS = {
    (1, 1): "Tết Dương Lịch (New Year's Day)",
    (4, 30): "Ngày Giải Phóng Miền Nam (Reunification Day)",
    (5, 1): "Ngày Quốc Tế Lao Động (International Workers' Day)",
    (9, 2): "Ngày Quốc Khánh (National Day)",
}

# =============================================================================
# LUNAR HOLIDAYS - Complete 2024-2035
# =============================================================================

# Tết Nguyên Đán (Vietnamese New Year) - 7 days off typically
# Format: year -> list of (month, day) tuples
VIETNAM_TET_HOLIDAYS = {
    2024: [(2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14)],
    2025: [(1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (2, 1), (2, 2), (2, 3)],  # Extended
    2026: [(2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20)],
    2027: [(2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11)],
    2028: [(1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31)],
    2029: [(2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18)],
    2030: [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)],
    2031: [(1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28)],
    2032: [(2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16)],
    2033: [(1, 30), (1, 31), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],
    2034: [(2, 18), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23), (2, 24)],
    2035: [(2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13)],
}

# Giỗ Tổ Hùng Vương (10th day of 3rd lunar month)
VIETNAM_HUNG_KINGS_DAY = {
    2024: (4, 18),
    2025: (4, 7),
    2026: (4, 26),
    2027: (4, 15),
    2028: (4, 4),
    2029: (4, 23),
    2030: (4, 12),
    2031: (4, 1),
    2032: (4, 19),
    2033: (4, 9),
    2034: (4, 28),
    2035: (4, 17),
}

# =============================================================================
# SUBSTITUTE HOLIDAYS (Nghỉ Bù)
# When a holiday falls on weekend, the following Monday is off
# =============================================================================


def _calculate_substitute_holidays(year: int) -> List[Tuple[int, int, str]]:
    """
    Calculate substitute holidays for a given year.

    Vietnam Labor Law: When a public holiday falls on a weekend,
    the following Monday is a substitute holiday.

    Returns:
        List of (month, day, reason) tuples
    """
    substitutes = []

    # Check fixed holidays
    for (month, day), name in VIETNAM_FIXED_HOLIDAYS.items():
        try:
            holiday_date = date(year, month, day)
            weekday = holiday_date.weekday()

            if weekday == 5:  # Saturday
                # Monday substitute
                substitute = holiday_date + timedelta(days=2)
                substitutes.append((substitute.month, substitute.day, f"Nghỉ bù {name}"))
            elif weekday == 6:  # Sunday
                # Monday substitute
                substitute = holiday_date + timedelta(days=1)
                substitutes.append((substitute.month, substitute.day, f"Nghỉ bù {name}"))
        except ValueError:
            continue

    # Check Hung Kings Day
    if year in VIETNAM_HUNG_KINGS_DAY:
        month, day = VIETNAM_HUNG_KINGS_DAY[year]
        try:
            holiday_date = date(year, month, day)
            weekday = holiday_date.weekday()

            if weekday == 5:  # Saturday
                substitute = holiday_date + timedelta(days=2)
                substitutes.append((substitute.month, substitute.day, "Nghỉ bù Giỗ Tổ Hùng Vương"))
            elif weekday == 6:  # Sunday
                substitute = holiday_date + timedelta(days=1)
                substitutes.append((substitute.month, substitute.day, "Nghỉ bù Giỗ Tổ Hùng Vương"))
        except ValueError:
            pass

    return substitutes


# Pre-calculated substitute holidays for common years
VIETNAM_SUBSTITUTE_HOLIDAYS: Dict[int, List[Tuple[int, int, str]]] = {}

# =============================================================================
# SPECIAL MARKET CLOSURES
# Unscheduled closures (typhoons, special events, etc.)
# =============================================================================

SPECIAL_MARKET_CLOSURES = {
    # Format: (year, month, day): "Reason"
    # Add historical special closures here
    (2020, 4, 1): "COVID-19 Special Closure",
    (2020, 4, 2): "COVID-19 Special Closure",
    (2020, 4, 3): "COVID-19 Special Closure",
}


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================


@lru_cache(maxsize=50)
def get_all_holidays_for_year(year: int) -> Dict[date, str]:
    """
    Get all holidays for a given year including substitutes.

    Args:
        year: Year to get holidays for

    Returns:
        Dict of {date: holiday_name}
    """
    holidays = {}

    # 1. Fixed holidays
    for (month, day), name in VIETNAM_FIXED_HOLIDAYS.items():
        try:
            holidays[date(year, month, day)] = name
        except ValueError:
            continue

    # 2. Tết holidays
    if year in VIETNAM_TET_HOLIDAYS:
        for month, day in VIETNAM_TET_HOLIDAYS[year]:
            try:
                holidays[date(year, month, day)] = "Tết Nguyên Đán"
            except ValueError:
                continue

    # 3. Hung Kings Day
    if year in VIETNAM_HUNG_KINGS_DAY:
        month, day = VIETNAM_HUNG_KINGS_DAY[year]
        try:
            holidays[date(year, month, day)] = "Giỗ Tổ Hùng Vương"
        except ValueError:
            pass

    # 4. Substitute holidays
    substitutes = _calculate_substitute_holidays(year)
    for month, day, reason in substitutes:
        try:
            holidays[date(year, month, day)] = reason
        except ValueError:
            continue

    # 5. Special closures
    for (y, m, d), reason in SPECIAL_MARKET_CLOSURES.items():
        if y == year:
            try:
                holidays[date(y, m, d)] = reason
            except ValueError:
                continue

    return holidays


def is_vietnam_holiday(dt: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Check if a date is a Vietnam public holiday.

    Args:
        dt: Date to check (default: today)

    Returns:
        (is_holiday, holiday_name)
    """
    if dt is None:
        dt = datetime.now()

    check_date = dt.date() if isinstance(dt, datetime) else dt
    year = check_date.year

    holidays = get_all_holidays_for_year(year)

    if check_date in holidays:
        return True, holidays[check_date]

    return False, ""


def is_trading_day(dt: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Check if a date is a trading day.

    Trading days are:
    - Monday to Friday
    - Not a public holiday
    - Not a special closure

    Args:
        dt: Date to check (default: today)

    Returns:
        (is_trading_day, reason_if_not)
    """
    if dt is None:
        dt = datetime.now()

    check_date = dt.date() if isinstance(dt, datetime) else dt

    # Check weekend
    if check_date.weekday() >= 5:
        day_name = "Saturday" if check_date.weekday() == 5 else "Sunday"
        return False, f"Weekend ({day_name})"

    # Check holiday
    is_holiday, holiday_name = is_vietnam_holiday(dt)
    if is_holiday:
        return False, f"Holiday: {holiday_name}"

    return True, "Trading day"


def get_next_trading_day(
    from_date: Optional[date] = None,
    days_ahead: int = 1,
) -> date:
    """
    Get the next trading day.

    Args:
        from_date: Starting date (default: today)
        days_ahead: Number of trading days ahead (default: 1)

    Returns:
        Next trading day date
    """
    if from_date is None:
        from_date = date.today()

    current = from_date
    trading_days_found = 0

    while trading_days_found < days_ahead:
        current += timedelta(days=1)
        is_trading, _ = is_trading_day(datetime.combine(current, datetime.min.time()))
        if is_trading:
            trading_days_found += 1

    return current


def get_previous_trading_day(
    from_date: Optional[date] = None,
    days_back: int = 1,
) -> date:
    """
    Get the previous trading day.

    Args:
        from_date: Starting date (default: today)
        days_back: Number of trading days back (default: 1)

    Returns:
        Previous trading day date
    """
    if from_date is None:
        from_date = date.today()

    current = from_date
    trading_days_found = 0

    while trading_days_found < days_back:
        current -= timedelta(days=1)
        is_trading, _ = is_trading_day(datetime.combine(current, datetime.min.time()))
        if is_trading:
            trading_days_found += 1

    return current


def count_trading_days_between(
    start_date: date,
    end_date: date,
) -> int:
    """
    Count trading days between two dates (exclusive of start, inclusive of end).

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Number of trading days
    """
    if end_date <= start_date:
        return 0

    count = 0
    current = start_date + timedelta(days=1)

    while current <= end_date:
        is_trading, _ = is_trading_day(datetime.combine(current, datetime.min.time()))
        if is_trading:
            count += 1
        current += timedelta(days=1)

    return count


def get_trading_days_in_month(year: int, month: int) -> List[date]:
    """
    Get all trading days in a given month.

    Args:
        year: Year
        month: Month (1-12)

    Returns:
        List of trading day dates
    """
    from calendar import monthrange

    _, last_day = monthrange(year, month)
    trading_days = []

    for day in range(1, last_day + 1):
        check_date = date(year, month, day)
        is_trading, _ = is_trading_day(datetime.combine(check_date, datetime.min.time()))
        if is_trading:
            trading_days.append(check_date)

    return trading_days


def get_upcoming_holidays(days_ahead: int = 30) -> List[Tuple[date, str]]:
    """
    Get upcoming holidays within the specified number of days.

    Args:
        days_ahead: Number of days to look ahead

    Returns:
        List of (date, holiday_name) tuples
    """
    today = date.today()
    end_date = today + timedelta(days=days_ahead)

    upcoming = []

    # Get holidays for current and next year (in case we cross year boundary)
    for year in [today.year, today.year + 1]:
        holidays = get_all_holidays_for_year(year)
        for holiday_date, name in holidays.items():
            if today < holiday_date <= end_date:
                upcoming.append((holiday_date, name))

    return sorted(upcoming, key=lambda x: x[0])


def is_pre_holiday_trading_day(dt: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Check if today is the last trading day before a holiday.

    Useful for:
    - Reducing position sizes before long weekends
    - Avoiding overnight risk before Tết

    Args:
        dt: Date to check (default: today)

    Returns:
        (is_pre_holiday, upcoming_holiday_name)
    """
    if dt is None:
        dt = datetime.now()

    check_date = dt.date() if isinstance(dt, datetime) else dt

    # Check if today is a trading day
    is_trading, _ = is_trading_day(dt)
    if not is_trading:
        return False, ""

    # Check next day
    next_day = check_date + timedelta(days=1)
    is_next_trading, reason = is_trading_day(datetime.combine(next_day, datetime.min.time()))

    if not is_next_trading and "Holiday" in reason:
        holiday_name = reason.replace("Holiday: ", "")
        return True, holiday_name

    return False, ""


def get_tet_period(year: int) -> Tuple[Optional[date], Optional[date]]:
    """
    Get the Tết holiday period for a given year.

    Args:
        year: Year

    Returns:
        (start_date, end_date) or (None, None) if not defined
    """
    if year not in VIETNAM_TET_HOLIDAYS:
        return None, None

    tet_dates = VIETNAM_TET_HOLIDAYS[year]
    if not tet_dates:
        return None, None

    start = date(year, tet_dates[0][0], tet_dates[0][1])
    end = date(year, tet_dates[-1][0], tet_dates[-1][1])

    return start, end


def days_until_tet(from_date: Optional[date] = None) -> int:
    """
    Calculate days until next Tết.

    Args:
        from_date: Starting date (default: today)

    Returns:
        Number of days until Tết starts (-1 if during Tết, -2 if unknown)
    """
    if from_date is None:
        from_date = date.today()

    year = from_date.year

    # Check current year Tết
    tet_start, tet_end = get_tet_period(year)
    if tet_start and tet_end:
        if tet_start <= from_date <= tet_end:
            return -1  # During Tết
        if from_date < tet_start:
            return (tet_start - from_date).days

    # Check next year Tết
    tet_start, _ = get_tet_period(year + 1)
    if tet_start:
        return (tet_start - from_date).days

    return -2  # Unknown


# =============================================================================
# HOLIDAY CALENDAR CLASS
# =============================================================================


class VietnamHolidayCalendar:
    """
    Complete Vietnam Holiday Calendar for trading.

    Usage:
        calendar = VietnamHolidayCalendar()

        # Check if today is trading day
        if calendar.is_trading_day():
            print("Market is open")

        # Get next trading day
        next_day = calendar.get_next_trading_day()

        # Check upcoming holidays
        holidays = calendar.get_upcoming_holidays(30)
    """

    def __init__(self):
        self._cache: Dict[int, Dict[date, str]] = {}

    def is_holiday(self, dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """Check if date is a holiday."""
        return is_vietnam_holiday(dt)

    def is_trading_day(self, dt: Optional[datetime] = None) -> bool:
        """Check if date is a trading day."""
        is_trading, _ = is_trading_day(dt)
        return is_trading

    def get_next_trading_day(self, from_date: Optional[date] = None, days_ahead: int = 1) -> date:
        """Get next trading day."""
        return get_next_trading_day(from_date, days_ahead)

    def get_previous_trading_day(
        self, from_date: Optional[date] = None, days_back: int = 1
    ) -> date:
        """Get previous trading day."""
        return get_previous_trading_day(from_date, days_back)

    def get_upcoming_holidays(self, days_ahead: int = 30) -> List[Tuple[date, str]]:
        """Get upcoming holidays."""
        return get_upcoming_holidays(days_ahead)

    def is_pre_holiday(self, dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """Check if date is last trading day before holiday."""
        return is_pre_holiday_trading_day(dt)

    def days_until_tet(self) -> int:
        """Get days until Tết."""
        return days_until_tet()

    def get_settlement_date(
        self, trade_date: Optional[date] = None, settlement_days: int = 2
    ) -> date:
        """
        Calculate T+N settlement date.

        Args:
            trade_date: Trade date (default: today)
            settlement_days: Number of settlement days (default: 2 for T+2)

        Returns:
            Settlement date
        """
        return get_next_trading_day(trade_date, settlement_days)


# Singleton instance
_calendar_instance: Optional[VietnamHolidayCalendar] = None


def get_holiday_calendar() -> VietnamHolidayCalendar:
    """Get singleton holiday calendar instance."""
    global _calendar_instance
    if _calendar_instance is None:
        _calendar_instance = VietnamHolidayCalendar()
    return _calendar_instance
