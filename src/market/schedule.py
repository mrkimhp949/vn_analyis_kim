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
# Thứ 3 = 2, Thứ 7 = 5
NON_TRADING_DAYS = [2, 5]  # Thứ 3 và Thứ 7


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

    # Check ngày nghỉ (Thứ 3 và Thứ 7)
    weekday = dt.weekday()
    if weekday in NON_TRADING_DAYS:
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
    Kiểm tra xem có phải ngày giao dịch không (không phải T3/T7)

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

    weekday = dt.weekday()
    return weekday not in NON_TRADING_DAYS


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

    # Nếu là T3 hoặc T7, chuyển sang ngày tiếp theo
    if current_weekday in NON_TRADING_DAYS:
        days_ahead = 1
        if current_weekday == 2:  # T3 -> T4
            days_ahead = 1
        elif current_weekday == 5:  # T7 -> T2
            days_ahead = 2

        next_date = dt.date() + timedelta(days=days_ahead)
        next_dt = VN_TZ.localize(
            datetime.combine(next_date, TRADING_HOURS["morning"][0])
        )
        return next_dt

    # Nếu trước 9:00, return 9:00 hôm nay
    if current_time < TRADING_HOURS["morning"][0]:
        return VN_TZ.localize(datetime.combine(dt.date(), TRADING_HOURS["morning"][0]))

    # Nếu giữa 11:30 và 13:00, return 13:00 hôm nay
    if TRADING_HOURS["morning"][1] < current_time < TRADING_HOURS["afternoon"][0]:
        return VN_TZ.localize(
            datetime.combine(dt.date(), TRADING_HOURS["afternoon"][0])
        )

    # Nếu sau 15:00, return 9:00 ngày mai
    if current_time > TRADING_HOURS["afternoon"][1]:
        next_date = dt.date() + timedelta(days=1)
        # Skip T3 và T7
        next_weekday = next_date.weekday()
        if next_weekday in NON_TRADING_DAYS:
            if next_weekday == 2:  # T3 -> T4
                next_date += timedelta(days=1)
            elif next_weekday == 5:  # T7 -> T2
                next_date += timedelta(days=2)

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
    afternoon_start = VN_TZ.localize(
        datetime.combine(today, TRADING_HOURS["afternoon"][0])
    )
    afternoon_end = VN_TZ.localize(
        datetime.combine(today, TRADING_HOURS["afternoon"][1])
    )
    sessions.append((afternoon_start, afternoon_end))

    return sessions


def should_run_scheduled_task(dt: Optional[datetime] = None) -> bool:
    """
    Kiểm tra xem có nên chạy scheduled task không

    - Chỉ chạy trong giờ giao dịch
    - Không chạy vào T3 và T7

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
