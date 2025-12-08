"""
Vietnam Trading Schedule
Lịch chạy theo giờ giao dịch VN, tránh gọi trong giờ nghỉ T3/T7
"""

from datetime import datetime, time, timedelta
from typing import Optional, Tuple

import pytz

# Timezone VN
VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")

# Giờ giao dịch VN
TRADING_HOURS = {
    "morning": (time(9, 0), time(11, 30)),  # 9:00 - 11:30
    "afternoon": (time(13, 0), time(15, 0)),  # 13:00 - 15:00
}

# Ngày nghỉ (0=Monday, 6=Sunday)
# Thứ 7 = 5, Chủ Nhật = 6
NON_TRADING_DAYS = [5, 6]  # Thứ 7 và Chủ Nhật (FIXED: was incorrectly [2, 5])

# Vietnam public holidays (can be updated annually)
# Format: (month, day) - does not include lunar calendar holidays which vary
VIETNAM_FIXED_HOLIDAYS = [
    (1, 1),  # New Year's Day
    (4, 30),  # Reunification Day
    (5, 1),  # International Workers' Day
    (9, 2),  # National Day
]

# =============================================================================
# LUNAR CALENDAR HOLIDAYS (Auto-detect by year)
# Updated: 2024-2030 with official government announcements
# Source: Vietnamese Government Portal, SSC announcements
# =============================================================================

# Tết Nguyên Đán (Vietnamese New Year) - varies by lunar calendar
VIETNAM_TET_HOLIDAYS = {
    2024: [(2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14)],  # Year of Dragon
    2025: [(1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (2, 1), (2, 2)],  # Year of Snake
    2026: [(2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20)],  # Year of Horse
    2027: [(2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11)],  # Year of Goat
    2028: [(1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31)],  # Year of Monkey
    2029: [(2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18)],  # Year of Rooster
    2030: [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)],  # Year of Dog
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
}


def get_lunar_holidays_for_year(year: int) -> list:
    """
    Get lunar holidays for a specific year.

    Args:
        year: Year to get holidays for

    Returns:
        List of (month, day) tuples for lunar holidays
    """
    holidays = []

    # Tết holidays
    if year in VIETNAM_TET_HOLIDAYS:
        holidays.extend(VIETNAM_TET_HOLIDAYS[year])
    else:
        # Fallback: use approximate dates if year not defined
        # Typically late Jan or early Feb
        import logging

        logging.warning(f"Lunar holidays for {year} not defined, using approximate dates")
        holidays.extend([(1, 28), (1, 29), (1, 30), (1, 31), (2, 1), (2, 2), (2, 3)])

    # Giỗ Tổ Hùng Vương
    if year in VIETNAM_HUNG_KINGS_DAY:
        holidays.append(VIETNAM_HUNG_KINGS_DAY[year])
    else:
        # Typically mid-April
        holidays.append((4, 10))

    return holidays


def get_all_holidays_for_year(year: int) -> list:
    """
    Get all Vietnam holidays (fixed + lunar) for a specific year.

    Args:
        year: Year to get holidays for

    Returns:
        List of (month, day) tuples
    """
    return VIETNAM_FIXED_HOLIDAYS + get_lunar_holidays_for_year(year)


# Combined holidays - auto-detect current year
def _get_current_year_holidays() -> list:
    """Get holidays for current year."""
    current_year = datetime.now().year
    return get_all_holidays_for_year(current_year)


VIETNAM_HOLIDAYS = _get_current_year_holidays()


def is_public_holiday(dt: Optional[datetime] = None) -> bool:
    """
    Check if date is a Vietnam public holiday.

    Supports multi-year lookup - automatically detects year from input date.

    Args:
        dt: Datetime to check (None = today)

    Returns:
        True if it's a public holiday
    """
    if dt is None:
        dt = datetime.now(VN_TZ)
    else:
        if dt.tzinfo is None:
            dt = VN_TZ.localize(dt)
        else:
            dt = dt.astimezone(VN_TZ)

    # Get holidays for the specific year
    year = dt.year
    holidays_for_year = get_all_holidays_for_year(year)

    month_day = (dt.month, dt.day)
    return month_day in holidays_for_year


def is_trading_hour(dt: Optional[datetime] = None) -> bool:
    """
    Kiểm tra xem có phải giờ giao dịch không

    Args:
        dt: Datetime để check (None = hiện tại)

    Returns:
        True nếu đang trong giờ giao dịch
    """
    if dt is None:
        dt = datetime.now(VN_TZ)
    else:
        # Ensure timezone
        if dt.tzinfo is None:
            dt = VN_TZ.localize(dt)
        else:
            dt = dt.astimezone(VN_TZ)

    # Check weekend (Saturday = 5, Sunday = 6)
    weekday = dt.weekday()
    if weekday in NON_TRADING_DAYS:
        return False

    # Check public holidays
    if is_public_holiday(dt):
        return False

    # Check giờ giao dịch
    current_time = dt.time()

    # Morning session
    morning_start, morning_end = TRADING_HOURS["morning"]
    if morning_start <= current_time <= morning_end:
        return True

    # Afternoon session
    afternoon_start, afternoon_end = TRADING_HOURS["afternoon"]
    if afternoon_start <= current_time <= afternoon_end:
        return True

    return False


def is_trading_day(dt: Optional[datetime] = None) -> bool:
    """
    Kiểm tra xem có phải ngày giao dịch không (không phải T7/CN và không phải ngày lễ)

    Args:
        dt: Datetime để check (None = hiện tại)

    Returns:
        True nếu là ngày giao dịch
    """
    if dt is None:
        dt = datetime.now(VN_TZ)
    else:
        if dt.tzinfo is None:
            dt = VN_TZ.localize(dt)
        else:
            dt = dt.astimezone(VN_TZ)

    # Check weekend
    weekday = dt.weekday()
    if weekday in NON_TRADING_DAYS:
        return False

    # Check public holidays
    if is_public_holiday(dt):
        return False

    return True


def get_next_trading_time(dt: Optional[datetime] = None) -> Optional[datetime]:
    """
    Lấy thời gian giao dịch tiếp theo

    Args:
        dt: Datetime hiện tại (None = hiện tại)

    Returns:
        Datetime của giờ giao dịch tiếp theo hoặc None
    """
    if dt is None:
        dt = datetime.now(VN_TZ)
    else:
        if dt.tzinfo is None:
            dt = VN_TZ.localize(dt)
        else:
            dt = dt.astimezone(VN_TZ)

    # Nếu đang trong giờ giao dịch, return ngay
    if is_trading_hour(dt):
        return dt

    current_time = dt.time()
    current_weekday = dt.weekday()

    # Nếu là T7 hoặc CN, chuyển sang ngày tiếp theo
    if current_weekday in NON_TRADING_DAYS:
        days_ahead = 1
        if current_weekday == 5:  # T7 -> T2
            days_ahead = 2
        elif current_weekday == 6:  # CN -> T2
            days_ahead = 1

        next_date = dt.date() + timedelta(days=days_ahead)

        # Skip holidays
        while is_public_holiday(VN_TZ.localize(datetime.combine(next_date, time(9, 0)))):
            next_date += timedelta(days=1)
            # Also skip weekends
            if next_date.weekday() in NON_TRADING_DAYS:
                next_date += timedelta(days=1 if next_date.weekday() == 5 else 2)

        next_dt = VN_TZ.localize(datetime.combine(next_date, TRADING_HOURS["morning"][0]))
        return next_dt

    # Nếu trước 9:00, return 9:00 hôm nay
    if current_time < TRADING_HOURS["morning"][0]:
        return VN_TZ.localize(datetime.combine(dt.date(), TRADING_HOURS["morning"][0]))

    # Nếu giữa 11:30 và 13:00, return 13:00 hôm nay
    if TRADING_HOURS["morning"][1] < current_time < TRADING_HOURS["afternoon"][0]:
        return VN_TZ.localize(datetime.combine(dt.date(), TRADING_HOURS["afternoon"][0]))

    # Nếu sau 15:00, return 9:00 ngày mai
    if current_time > TRADING_HOURS["afternoon"][1]:
        next_date = dt.date() + timedelta(days=1)
        # Skip T7 và CN
        next_weekday = next_date.weekday()
        if next_weekday in NON_TRADING_DAYS:
            if next_weekday == 5:  # T7 -> T2
                next_date += timedelta(days=2)
            elif next_weekday == 6:  # CN -> T2
                next_date += timedelta(days=1)

        # Skip holidays
        while is_public_holiday(VN_TZ.localize(datetime.combine(next_date, time(9, 0)))):
            next_date += timedelta(days=1)
            if next_date.weekday() in NON_TRADING_DAYS:
                next_date += timedelta(days=1 if next_date.weekday() == 6 else 2)

        return VN_TZ.localize(datetime.combine(next_date, TRADING_HOURS["morning"][0]))

    return None


def get_trading_sessions_today(
    dt: Optional[datetime] = None,
) -> list[Tuple[datetime, datetime]]:
    """
    Lấy danh sách các session giao dịch trong ngày

    Returns:
        List of (start, end) tuples
    """
    if dt is None:
        dt = datetime.now(VN_TZ)
    else:
        if dt.tzinfo is None:
            dt = VN_TZ.localize(dt)
        else:
            dt = dt.astimezone(VN_TZ)

    if not is_trading_day(dt):
        return []

    sessions = []
    today = dt.date()

    # Morning session
    morning_start = VN_TZ.localize(datetime.combine(today, TRADING_HOURS["morning"][0]))
    morning_end = VN_TZ.localize(datetime.combine(today, TRADING_HOURS["morning"][1]))
    sessions.append((morning_start, morning_end))

    # Afternoon session
    afternoon_start = VN_TZ.localize(datetime.combine(today, TRADING_HOURS["afternoon"][0]))
    afternoon_end = VN_TZ.localize(datetime.combine(today, TRADING_HOURS["afternoon"][1]))
    sessions.append((afternoon_start, afternoon_end))

    return sessions


def is_near_session_boundary(dt: Optional[datetime] = None, minutes: int = 5) -> Tuple[bool, str]:
    """
    Check if current time is near session boundary (avoid trading)

    Args:
        dt: Datetime to check (None = now)
        minutes: Minutes before/after boundary to avoid

    Returns:
        (is_near_boundary, boundary_type)
        - is_near_boundary: True if near a session boundary
        - boundary_type: "AM_END", "PM_START", "PM_END", or ""
    """
    if dt is None:
        dt = datetime.now(VN_TZ)
    else:
        if dt.tzinfo is None:
            dt = VN_TZ.localize(dt)
        else:
            dt = dt.astimezone(VN_TZ)

    current_time = dt.time()

    def subtract_minutes(t: time, mins: int) -> time:
        """Safely subtract minutes from time"""
        total_mins = t.hour * 60 + t.minute - mins
        if total_mins < 0:
            total_mins = 0
        return time(total_mins // 60, total_mins % 60)

    def add_minutes(t: time, mins: int) -> time:
        """Safely add minutes to time"""
        total_mins = t.hour * 60 + t.minute + mins
        if total_mins >= 24 * 60:
            total_mins = 24 * 60 - 1
        return time(total_mins // 60, total_mins % 60)

    # Morning session end (11:30)
    am_end = TRADING_HOURS["morning"][1]
    am_end_start = subtract_minutes(am_end, minutes)
    if am_end_start <= current_time <= am_end:
        return (True, "AM_END")

    # Afternoon session start (13:00)
    pm_start = TRADING_HOURS["afternoon"][0]
    pm_start_end = add_minutes(pm_start, minutes)
    if pm_start <= current_time <= pm_start_end:
        return (True, "PM_START")

    # Afternoon session end (15:00)
    pm_end = TRADING_HOURS["afternoon"][1]
    pm_end_start = subtract_minutes(pm_end, minutes)
    if pm_end_start <= current_time <= pm_end:
        return (True, "PM_END")

    return (False, "")


def should_run_scheduled_task(dt: Optional[datetime] = None) -> bool:
    """
    Kiểm tra xem có nên chạy scheduled task không

    - Chỉ chạy trong giờ giao dịch
    - Không chạy vào T7/CN và ngày lễ

    Args:
        dt: Datetime để check (None = hiện tại)

    Returns:
        True nếu nên chạy
    """
    return is_trading_hour(dt) and is_trading_day(dt)


# Test
if __name__ == "__main__":
    print("🧪 Testing Vietnam Trading Schedule\n")

    now = datetime.now(VN_TZ)
    print(f"Current time: {now}")
    print(f"Is trading hour: {is_trading_hour(now)}")
    print(f"Is trading day: {is_trading_day(now)}")
    print(f"Should run task: {should_run_scheduled_task(now)}")

    next_trading = get_next_trading_time(now)
    if next_trading:
        print(f"Next trading time: {next_trading}")

    sessions = get_trading_sessions_today(now)
    print(f"Trading sessions today: {len(sessions)}")
    for start, end in sessions:
        print(f"  {start.time()} - {end.time()}")
