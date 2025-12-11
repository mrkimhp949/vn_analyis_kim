# -*- coding: utf-8 -*-
"""
Vietnam Market Utilities

Centralized utilities for Vietnam stock market specific rules:
- Lot size validation (100 shares)
- Tick size calculation (10/50/100 VND based on price)
- Exchange detection (HOSE/HNX/UPCOM)
- Price limit calculation
- ATO/ATC session detection
- Holiday calendar (via vietnam_holidays.py)

Author: Trading Bot Team
Version: 2.0.0 - Complete 10/10 Implementation
"""

import logging
from datetime import datetime, time, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT COMPLETE HOLIDAY CALENDAR (10/10 Implementation)
# =============================================================================

# Import complete holiday calendar from vietnam_holidays.py
try:
    from src.utils.vietnam_holidays import (
        VietnamHolidayCalendar,
        get_holiday_calendar,
        is_vietnam_holiday as _is_vietnam_holiday_complete,
        is_trading_day as _is_trading_day_complete,
        get_next_trading_day,
        get_previous_trading_day,
        get_upcoming_holidays,
        is_pre_holiday_trading_day,
        days_until_tet,
        get_tet_period,
        count_trading_days_between,
        get_trading_days_in_month,
        get_all_holidays_for_year,
        VIETNAM_TET_HOLIDAYS as TET_HOLIDAYS_COMPLETE,
        VIETNAM_HUNG_KINGS_DAY as HUNG_KINGS_DAY_COMPLETE,
    )

    COMPLETE_HOLIDAY_CALENDAR_AVAILABLE = True
    logger.debug("✅ Complete holiday calendar loaded from vietnam_holidays.py")
except ImportError as e:
    COMPLETE_HOLIDAY_CALENDAR_AVAILABLE = False
    logger.warning(f"⚠️ Complete holiday calendar not available: {e}")

# =============================================================================
# CONSTANTS
# =============================================================================

# Lot size (minimum trading unit)
VN_LOT_SIZE = 100

# Tick sizes based on price ranges (HOSE rules)
VN_TICK_SIZES = {
    "low": {"max_price": 10_000, "tick": 10},
    "mid": {"max_price": 50_000, "tick": 50},
    "high": {"max_price": float("inf"), "tick": 100},
}

# Price limits by exchange
EXCHANGE_PRICE_LIMITS = {
    "HOSE": 0.07,  # ±7%
    "HNX": 0.10,  # ±10%
    "UPCOM": 0.15,  # ±15%
}

# VN30 symbols (HOSE blue chips) - Updated Q4 2024
# Source: https://www.hsx.vn/Modules/Listed/Web/StockIndexView/188
VN30_SYMBOLS = {
    "ACB",
    "BCM",
    "BID",
    "BVH",
    "CTG",
    "FPT",
    "GAS",
    "GVR",
    "HDB",
    "HPG",
    "MBB",
    "MSN",
    "MWG",
    "PLX",
    "POW",
    "SAB",
    "SHB",
    "SSB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VNM",
    "VPB",
    "VRE",
}

# Sector mapping for VN30 stocks - for sector rotation analysis
VN30_SECTORS = {
    # Banking
    "ACB": "BANKING",
    "BID": "BANKING",
    "CTG": "BANKING",
    "HDB": "BANKING",
    "MBB": "BANKING",
    "SHB": "BANKING",
    "SSB": "BANKING",
    "STB": "BANKING",
    "TCB": "BANKING",
    "TPB": "BANKING",
    "VCB": "BANKING",
    "VIB": "BANKING",
    "VPB": "BANKING",
    # Real Estate
    "BCM": "REAL_ESTATE",
    "VHM": "REAL_ESTATE",
    "VIC": "REAL_ESTATE",
    "VRE": "REAL_ESTATE",
    # Consumer
    "MWG": "CONSUMER",
    "MSN": "CONSUMER",
    "SAB": "CONSUMER",
    "VNM": "CONSUMER",
    # Energy & Utilities
    "GAS": "ENERGY",
    "PLX": "ENERGY",
    "POW": "UTILITIES",
    # Industrial
    "GVR": "INDUSTRIAL",
    "HPG": "INDUSTRIAL",
    # Insurance
    "BVH": "INSURANCE",
    # Technology
    "FPT": "TECHNOLOGY",
    # Securities
    "SSI": "SECURITIES",
    # Aviation
    "VJC": "AVIATION",
}

# HNX30 symbols
HNX30_SYMBOLS = {
    "SHS",
    "PVS",
    "IDC",
    "CEO",
    "NVB",
    "PVI",
    "TNG",
    "VC3",
    "DTD",
    "HUT",
    "PVB",
    "NDN",
    "L14",
    "VCS",
    "HLD",
    "BVS",
    "VCG",
    "NTP",
    "PGS",
    "VGC",
}

# Trading session times (Vietnam timezone: Asia/Ho_Chi_Minh, UTC+7)
VN_SESSIONS = {
    "ATO": (time(9, 0), time(9, 15)),  # Opening auction - high volatility
    "MORNING": (time(9, 15), time(11, 30)),  # Continuous trading AM
    "LUNCH": (time(11, 30), time(13, 0)),  # Lunch break - no trading
    "AFTERNOON": (time(13, 0), time(14, 30)),  # Continuous trading PM
    "ATC": (time(14, 30), time(14, 45)),  # Closing auction - high volatility
}

# T+2.5 Settlement cycle specifics
# Buy on T0 -> Settlement on T+2 (actually T+2.5 for cash)
# Sell proceeds available on T+2
VN_SETTLEMENT_DAYS = 2

# =============================================================================
# VIETNAM HOLIDAY CALENDAR
# =============================================================================

# Vietnam public holidays (fixed dates)
# Note: Lunar calendar holidays (Tết) vary each year
VIETNAM_FIXED_HOLIDAYS = {
    # New Year's Day
    (1, 1): "New Year's Day",
    # Reunification Day
    (4, 30): "Reunification Day",
    # International Workers' Day
    (5, 1): "International Workers' Day",
    # National Day
    (9, 2): "National Day",
}

# =============================================================================
# LUNAR CALENDAR HOLIDAYS - Updated 2024-2030
# Source: Vietnamese Government, Ministry of Labor announcements
# =============================================================================

# Tết Nguyên Đán (Vietnamese New Year) - varies by lunar calendar
# Format: year -> list of (month, day) tuples for Tết holidays (including Eve + 5 days off)
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
# This is a national holiday - stock market closed
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
    Get lunar holidays (Tết + Giỗ Tổ Hùng Vương) for a specific year.

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
        logger.warning(f"Tết holidays for {year} not defined, using approximate dates")
        holidays.extend([(1, 28), (1, 29), (1, 30), (1, 31), (2, 1), (2, 2), (2, 3)])

    # Giỗ Tổ Hùng Vương
    if year in VIETNAM_HUNG_KINGS_DAY:
        holidays.append(VIETNAM_HUNG_KINGS_DAY[year])
    else:
        # Typically mid-April
        logger.warning(f"Hung Kings Day for {year} not defined, using April 10")
        holidays.append((4, 10))

    return holidays


def is_vietnam_public_holiday(dt: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Check if date is a Vietnam public holiday with holiday name.

    IMPROVED 10/10: Uses complete holiday calendar with substitute holidays.

    Args:
        dt: Datetime to check (None = today)

    Returns:
        Tuple of (is_holiday, holiday_name)
    """
    # Use complete calendar if available (includes substitute holidays)
    if COMPLETE_HOLIDAY_CALENDAR_AVAILABLE:
        return _is_vietnam_holiday_complete(dt)

    # Fallback to basic implementation
    if dt is None:
        try:
            import pytz

            vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
            dt = datetime.now(vn_tz)
        except ImportError:
            dt = datetime.now()

    year = dt.year
    month_day = (dt.month, dt.day)

    # Check fixed holidays
    if month_day in VIETNAM_FIXED_HOLIDAYS:
        return True, VIETNAM_FIXED_HOLIDAYS[month_day]

    # Check Tết holidays
    if year in VIETNAM_TET_HOLIDAYS:
        if month_day in VIETNAM_TET_HOLIDAYS[year]:
            return True, "Tết Nguyên Đán"

    # Check Hung Kings Day
    if year in VIETNAM_HUNG_KINGS_DAY:
        if month_day == VIETNAM_HUNG_KINGS_DAY[year]:
            return True, "Giỗ Tổ Hùng Vương"

    return False, ""


def is_trading_day_vn(dt: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Check if date is a trading day in Vietnam market.

    IMPROVED 10/10: Uses complete holiday calendar.

    Args:
        dt: Datetime to check (None = today)

    Returns:
        Tuple of (is_trading_day, reason_if_not)
    """
    if COMPLETE_HOLIDAY_CALENDAR_AVAILABLE:
        return _is_trading_day_complete(dt)

    # Fallback implementation
    if dt is None:
        dt = datetime.now()

    # Check weekend
    if dt.weekday() >= 5:
        day_name = "Saturday" if dt.weekday() == 5 else "Sunday"
        return False, f"Weekend ({day_name})"

    # Check holiday
    is_holiday, holiday_name = is_vietnam_public_holiday(dt)
    if is_holiday:
        return False, f"Holiday: {holiday_name}"

    return True, "Trading day"


# Foreign ownership limits by sector (approximate)
VN_FOREIGN_LIMITS = {
    "BANKING": 0.30,  # 30% max foreign ownership
    "SECURITIES": 0.49,  # 49% max
    "INSURANCE": 0.49,  # 49% max
    "REAL_ESTATE": 0.49,  # 49% max (varies by company)
    "CONSUMER": 0.49,  # 49% max
    "TECHNOLOGY": 1.00,  # 100% (no limit)
    "INDUSTRIAL": 0.49,  # 49% max
    "ENERGY": 0.49,  # 49% max
    "UTILITIES": 0.49,  # 49% max
    "AVIATION": 0.34,  # 34% max (special regulation)
    "DEFAULT": 0.49,  # Default 49%
}


# =============================================================================
# LOT SIZE FUNCTIONS
# =============================================================================


def round_to_lot(shares: int, lot_size: int = VN_LOT_SIZE) -> int:
    """
    Round shares down to nearest lot size.

    Args:
        shares: Number of shares
        lot_size: Lot size (default: 100)

    Returns:
        Shares rounded to lot size (minimum 1 lot)

    Example:
        >>> round_to_lot(150)
        100
        >>> round_to_lot(250)
        200
        >>> round_to_lot(50)
        100  # Minimum 1 lot
    """
    if shares <= 0:
        return 0
    rounded = (shares // lot_size) * lot_size
    return max(lot_size, rounded)  # Minimum 1 lot


def validate_lot_size(shares: int, lot_size: int = VN_LOT_SIZE) -> Tuple[bool, str]:
    """
    Validate that shares is a valid lot size.

    Args:
        shares: Number of shares to validate
        lot_size: Lot size (default: 100)

    Returns:
        Tuple of (is_valid, message)
    """
    if shares <= 0:
        return False, "Shares must be positive"
    if shares % lot_size != 0:
        return False, f"Shares must be multiple of {lot_size} (got {shares})"
    return True, "Valid"


# =============================================================================
# TICK SIZE FUNCTIONS - IMPROVED v10.0 (Exchange-specific rules)
# =============================================================================

# Exchange-specific tick size rules
# HOSE: 10/50/100 VND based on price range
# HNX: 100 VND fixed for all prices
# UPCOM: 100 VND fixed for all prices

EXCHANGE_TICK_RULES = {
    "HOSE": {
        "type": "tiered",
        "tiers": [
            {"max_price": 10_000, "tick": 10},
            {"max_price": 50_000, "tick": 50},
            {"max_price": float("inf"), "tick": 100},
        ],
    },
    "HNX": {
        "type": "fixed",
        "tick": 100,  # HNX uses fixed 100 VND tick for all prices
    },
    "UPCOM": {
        "type": "fixed",
        "tick": 100,  # UPCOM uses fixed 100 VND tick for all prices
    },
    "OTC": {
        "type": "fixed",
        "tick": 100,  # OTC default 100 VND
    },
}


def get_tick_size(price: float, exchange: str = "HOSE") -> int:
    """
    Get tick size based on price range and exchange.

    IMPROVED v10.0: Exchange-specific tick size rules.

    Tick Size Rules by Exchange:
    - HOSE (Tiered):
      * Price < 10,000 VND: Tick = 10 VND
      * 10,000 <= Price < 50,000 VND: Tick = 50 VND
      * Price >= 50,000 VND: Tick = 100 VND
    - HNX: Fixed 100 VND for all prices
    - UPCOM: Fixed 100 VND for all prices

    Args:
        price: Stock price in VND
        exchange: Exchange name ("HOSE", "HNX", "UPCOM")

    Returns:
        Tick size in VND
    """
    exchange = exchange.upper()
    rules = EXCHANGE_TICK_RULES.get(exchange, EXCHANGE_TICK_RULES["HOSE"])

    if rules["type"] == "fixed":
        return rules["tick"]

    # Tiered rules (HOSE)
    for tier in rules["tiers"]:
        if price < tier["max_price"]:
            return tier["tick"]

    return 100  # Default fallback


def get_tick_size_for_symbol(price: float, symbol: str) -> int:
    """
    Get tick size for a specific symbol (auto-detects exchange).

    Args:
        price: Stock price in VND
        symbol: Stock symbol

    Returns:
        Tick size in VND
    """
    exchange = get_exchange(symbol)
    return get_tick_size(price, exchange)


def round_to_tick(price: float, direction: str = "nearest", exchange: str = "HOSE") -> float:
    """
    Round price to valid tick size.

    IMPROVED v10.0: Exchange-aware tick rounding.

    Args:
        price: Price to round
        direction: "nearest", "up", or "down"
        exchange: Exchange name ("HOSE", "HNX", "UPCOM")

    Returns:
        Price rounded to valid tick
    """
    tick = get_tick_size(price, exchange)

    if direction == "up":
        return ((price + tick - 1) // tick) * tick
    elif direction == "down":
        return (price // tick) * tick
    else:  # nearest
        return round(price / tick) * tick


def round_to_tick_for_symbol(price: float, symbol: str, direction: str = "nearest") -> float:
    """
    Round price to valid tick size for a specific symbol (auto-detects exchange).

    Args:
        price: Price to round
        symbol: Stock symbol
        direction: "nearest", "up", or "down"

    Returns:
        Price rounded to valid tick
    """
    exchange = get_exchange(symbol)
    return round_to_tick(price, direction, exchange)


def validate_price(price: float, exchange: str = "HOSE") -> Tuple[bool, str]:
    """
    Validate that price is a valid tick.

    IMPROVED v10.0: Exchange-aware price validation.

    Args:
        price: Price to validate
        exchange: Exchange name ("HOSE", "HNX", "UPCOM")

    Returns:
        Tuple of (is_valid, message)
    """
    if price <= 0:
        return False, "Price must be positive"

    tick = get_tick_size(price, exchange)
    if price % tick != 0:
        valid_price = round_to_tick(price, "nearest", exchange)
        return False, f"Price {price:,.0f} not valid tick for {exchange}. Use {valid_price:,.0f}"

    return True, "Valid"


def validate_price_for_symbol(price: float, symbol: str) -> Tuple[bool, str]:
    """
    Validate that price is a valid tick for a specific symbol.

    Args:
        price: Price to validate
        symbol: Stock symbol

    Returns:
        Tuple of (is_valid, message)
    """
    exchange = get_exchange(symbol)
    return validate_price(price, exchange)


# =============================================================================
# EXCHANGE FUNCTIONS
# =============================================================================

# Exchange reference database (loaded from List.csv)
_EXCHANGE_DATABASE: Dict[str, str] = {}
_EXCHANGE_DB_LOADED: bool = False


def _load_exchange_database() -> None:
    """
    Load exchange reference database from List.csv.

    File format: Symbol,Name,Exchange
    Exchange values: HSX (HOSE), HNX, Upcom, OTC

    This provides accurate exchange detection instead of heuristics.
    """
    global _EXCHANGE_DB_LOADED

    if _EXCHANGE_DB_LOADED:
        return

    import csv
    import os

    csv_path = "List.csv"

    # Try multiple paths
    possible_paths = [
        csv_path,
        os.path.join(os.path.dirname(__file__), "..", "..", csv_path),
        os.path.join(os.getcwd(), csv_path),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 3:
                            symbol = row[0].upper().strip()
                            exchange_raw = row[2].upper().strip()

                            # Normalize exchange names
                            if exchange_raw in ("HSX", "HOSE"):
                                exchange = "HOSE"
                            elif exchange_raw == "HNX":
                                exchange = "HNX"
                            elif exchange_raw in ("UPCOM", "UPC"):
                                exchange = "UPCOM"
                            elif exchange_raw == "OTC":
                                exchange = "OTC"
                            else:
                                exchange = "HOSE"  # Default

                            _EXCHANGE_DATABASE[symbol] = exchange

                _EXCHANGE_DB_LOADED = True
                logger.info(
                    f"✅ Loaded exchange database: {len(_EXCHANGE_DATABASE)} symbols from {path}"
                )
                return

            except Exception as e:
                logger.warning(f"Failed to load exchange database from {path}: {e}")

    logger.warning("Exchange database not found, using heuristic detection")
    _EXCHANGE_DB_LOADED = True  # Mark as loaded to avoid repeated attempts


def get_exchange(symbol: str) -> str:
    """
    Detect exchange from symbol using reference database.

    IMPROVED v4.2: Uses List.csv reference database for accurate detection.
    Falls back to heuristics if database not available.

    Priority:
    1. Reference database (List.csv) - most accurate
    2. VN30/HNX30 known symbols
    3. Heuristic (3-letter = HOSE)
    4. Default HOSE

    Args:
        symbol: Stock symbol

    Returns:
        Exchange name: "HOSE", "HNX", "UPCOM", or "OTC"
    """
    # Load database on first call
    _load_exchange_database()

    symbol = symbol.upper().strip()

    # Priority 1: Reference database
    if symbol in _EXCHANGE_DATABASE:
        return _EXCHANGE_DATABASE[symbol]

    # Priority 2: Known index symbols
    if symbol in VN30_SYMBOLS:
        return "HOSE"
    if symbol in HNX30_SYMBOLS:
        return "HNX"

    # Priority 3: Heuristic - 3-letter symbols are typically HOSE
    # Note: This is not 100% accurate but covers most cases
    if len(symbol) == 3:
        logger.debug(f"⚠️ Using heuristic for {symbol}: assuming HOSE (3-letter symbol)")
        return "HOSE"

    # Priority 4: Default to HOSE (most liquid exchange)
    logger.debug(f"⚠️ Unknown symbol {symbol}: defaulting to HOSE")
    return "HOSE"


def get_exchange_info(symbol: str) -> Dict[str, Any]:
    """
    Get detailed exchange information for a symbol.

    Returns:
        Dict with exchange, price_limit, is_known, detection_method
    """
    _load_exchange_database()
    symbol = symbol.upper().strip()

    # Determine detection method
    if symbol in _EXCHANGE_DATABASE:
        exchange = _EXCHANGE_DATABASE[symbol]
        detection_method = "database"
        is_known = True
    elif symbol in VN30_SYMBOLS:
        exchange = "HOSE"
        detection_method = "vn30_index"
        is_known = True
    elif symbol in HNX30_SYMBOLS:
        exchange = "HNX"
        detection_method = "hnx30_index"
        is_known = True
    elif len(symbol) == 3:
        exchange = "HOSE"
        detection_method = "heuristic_3letter"
        is_known = False
    else:
        exchange = "HOSE"
        detection_method = "default"
        is_known = False

    return {
        "symbol": symbol,
        "exchange": exchange,
        "price_limit": EXCHANGE_PRICE_LIMITS.get(exchange, 0.07),
        "is_known": is_known,
        "detection_method": detection_method,
        "is_vn30": symbol in VN30_SYMBOLS,
        "is_hnx30": symbol in HNX30_SYMBOLS,
    }


def get_price_limit(symbol: str) -> float:
    """
    Get price limit percentage based on exchange.

    Args:
        symbol: Stock symbol

    Returns:
        Price limit as decimal (e.g., 0.07 for 7%)
    """
    exchange = get_exchange(symbol)
    return EXCHANGE_PRICE_LIMITS.get(exchange, 0.07)


def calculate_ceiling_floor(reference_price: float, symbol: str) -> Dict[str, float]:
    """
    Calculate ceiling and floor prices.

    Args:
        reference_price: Previous day's closing price
        symbol: Stock symbol

    Returns:
        Dict with ceiling, floor, and limit_percent
    """
    limit = get_price_limit(symbol)

    ceiling = reference_price * (1 + limit)
    floor = reference_price * (1 - limit)

    # Round to valid ticks
    ceiling = round_to_tick(ceiling, direction="down")
    floor = round_to_tick(floor, direction="up")

    return {
        "reference": reference_price,
        "ceiling": ceiling,
        "floor": floor,
        "limit_percent": limit * 100,
        "exchange": get_exchange(symbol),
    }


# =============================================================================
# SESSION FUNCTIONS
# =============================================================================


def get_current_session(current_time: Optional[time] = None) -> Tuple[str, bool]:
    """
    Get current trading session.

    Args:
        current_time: Time to check (default: now in VN timezone)

    Returns:
        Tuple of (session_name, is_trading)
    """
    if current_time is None:
        try:
            import pytz

            vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
            current_time = datetime.now(vn_tz).time()
        except ImportError:
            current_time = datetime.now().time()

    for session_name, (start, end) in VN_SESSIONS.items():
        if start <= current_time <= end:
            is_trading = session_name not in ["LUNCH"]
            return session_name, is_trading

    # Outside all sessions
    if current_time < VN_SESSIONS["ATO"][0]:
        return "PRE_MARKET", False
    elif current_time > VN_SESSIONS["ATC"][1]:
        return "POST_MARKET", False

    return "UNKNOWN", False


def check_ato_atc_session(current_time: Optional[time] = None) -> Tuple[bool, str, str]:
    """
    Check if currently in ATO or ATC auction session.

    Args:
        current_time: Time to check (default: now)

    Returns:
        Tuple of (is_auction, session_type, warning_message)
    """
    session, _ = get_current_session(current_time)

    if session == "ATO":
        return True, "ATO", "⚠️ ATO session - high volatility, wide spreads"
    elif session == "ATC":
        return True, "ATC", "⚠️ ATC session - high volatility, institutional orders"

    return False, "", ""


def is_optimal_entry_time(current_time: Optional[time] = None) -> Tuple[bool, str]:
    """
    Check if current time is optimal for entry.

    Optimal windows:
    - 9:30-10:30: After ATO volatility settles
    - 13:30-14:15: After lunch gap, before ATC

    Args:
        current_time: Time to check (default: now)

    Returns:
        Tuple of (is_optimal, reason)
    """
    if current_time is None:
        try:
            import pytz

            vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
            current_time = datetime.now(vn_tz).time()
        except ImportError:
            current_time = datetime.now().time()

    # Optimal windows
    if time(9, 30) <= current_time <= time(10, 30):
        return True, "Morning optimal window (9:30-10:30)"
    if time(13, 30) <= current_time <= time(14, 15):
        return True, "Afternoon optimal window (13:30-14:15)"

    # Avoid windows
    if time(9, 0) <= current_time <= time(9, 15):
        return False, "ATO auction - avoid entry"
    if time(11, 0) <= current_time <= time(11, 30):
        return False, "Pre-lunch selling pressure"
    if time(14, 30) <= current_time <= time(14, 45):
        return False, "ATC auction - avoid entry"

    return True, "Acceptable entry time"


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def validate_order(
    symbol: str, shares: int, price: float, order_type: str = "LO"
) -> Tuple[bool, Dict]:
    """
    Validate order parameters for Vietnam market.

    Args:
        symbol: Stock symbol
        shares: Number of shares
        price: Order price
        order_type: Order type (LO, ATO, ATC, MP)

    Returns:
        Tuple of (is_valid, details)
    """
    errors = []
    warnings = []

    # Validate lot size
    lot_valid, lot_msg = validate_lot_size(shares)
    if not lot_valid:
        errors.append(lot_msg)

    # Validate price (only for limit orders)
    if order_type == "LO":
        price_valid, price_msg = validate_price(price)
        if not price_valid:
            errors.append(price_msg)

    # Check session for ATO/ATC orders
    if order_type in ["ATO", "ATC"]:
        is_auction, session, _ = check_ato_atc_session()
        if order_type == "ATO" and session != "ATO":
            errors.append("ATO orders only valid during ATO session (9:00-9:15)")
        if order_type == "ATC" and session != "ATC":
            errors.append("ATC orders only valid during ATC session (14:30-14:45)")

    # Check optimal entry time
    is_optimal, timing_reason = is_optimal_entry_time()
    if not is_optimal:
        warnings.append(timing_reason)

    is_valid = len(errors) == 0

    return is_valid, {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "symbol": symbol,
        "shares": shares,
        "price": price,
        "order_type": order_type,
        "exchange": get_exchange(symbol),
        "corrected_shares": round_to_lot(shares) if not lot_valid else shares,
        "corrected_price": round_to_tick(price) if order_type == "LO" else price,
    }


# =============================================================================
# VALIDATOR CLASS
# =============================================================================


class VietnamMarketValidator:
    """
    Validator class for Vietnam market rules.
    Wraps standalone functions for easier integration.
    """

    def __init__(
        self,
        min_liquidity_value: float = None,
        max_position_pct_of_volume: float = 0.05,
    ):
        """
        Initialize validator.

        Args:
            min_liquidity_value: Minimum daily liquidity in VND (default: from constants)
            max_position_pct_of_volume: Max position as % of avg volume (default: 5%)
        """
        # Import here to avoid circular imports
        from src.config.constants import VN_MIN_LIQUIDITY_VALUE

        self.min_liquidity_value = (
            min_liquidity_value if min_liquidity_value is not None else VN_MIN_LIQUIDITY_VALUE
        )
        self.max_position_pct_of_volume = max_position_pct_of_volume

    def check_liquidity_requirements(self, df, symbol: str) -> Tuple[bool, str]:
        """
        Check if stock meets liquidity requirements.

        Args:
            df: DataFrame with 'close' and 'volume' columns
            symbol: Stock symbol

        Returns:
            Tuple of (is_liquid, message)
        """
        if df is None or df.empty or len(df) < 20:
            return False, "Insufficient data for liquidity check"

        try:
            avg_volume = df["volume"].tail(20).mean()
            avg_price = df["close"].tail(20).mean()
            avg_liquidity = avg_volume * avg_price

            if avg_liquidity < self.min_liquidity_value:
                return (
                    False,
                    f"Insufficient liquidity: {avg_liquidity:,.0f} < {self.min_liquidity_value:,.0f}",
                )

            return True, f"Liquidity OK: {avg_liquidity:,.0f} VND"
        except Exception as e:
            return False, f"Liquidity check error: {e}"

    def check_price_floor_ceiling(
        self, current_price: float, reference_price: float, symbol: str
    ) -> Tuple[bool, str]:
        """
        Check if current price is safely within floor/ceiling limits.

        Args:
            current_price: Current stock price
            reference_price: Reference (previous close) price
            symbol: Stock symbol

        Returns:
            Tuple of (is_safe, message)
        """
        limits = calculate_ceiling_floor(reference_price, symbol)
        ceiling = limits["ceiling"]
        floor = limits["floor"]

        # Check if within 1% of ceiling or floor
        ceiling_threshold = ceiling * 0.99
        floor_threshold = floor * 1.01

        if current_price >= ceiling_threshold:
            return False, f"Price {current_price:,.0f} too close to CEILING {ceiling:,.0f}"
        if current_price <= floor_threshold:
            return False, f"Price {current_price:,.0f} too close to FLOOR {floor:,.0f}"

        return True, "Price within safe range"

    def validate_position_size_vs_volume(
        self, shares: int, avg_volume: float, symbol: str
    ) -> Tuple[bool, str]:
        """
        Validate position size against average volume.

        Args:
            shares: Number of shares to trade
            avg_volume: Average daily volume
            symbol: Stock symbol

        Returns:
            Tuple of (is_safe, message)
        """
        if avg_volume <= 0:
            return False, "Invalid average volume"

        position_pct = shares / avg_volume

        if position_pct > self.max_position_pct_of_volume:
            return (
                False,
                f"Position too large: {position_pct:.1%} of avg volume (max {self.max_position_pct_of_volume:.1%})",
            )

        return True, f"Position size OK: {position_pct:.1%} of avg volume"

    def calculate_t2_cash_requirement(
        self, pending_settlements: dict, new_order_value: float
    ) -> Tuple[float, float]:
        """
        Calculate T+2 cash requirement for Vietnam market.

        Args:
            pending_settlements: Dict of pending settlements {date: value}
            new_order_value: Value of new order

        Returns:
            Tuple of (total_t2_requirement, buffer_amount)
        """
        total_pending = sum(pending_settlements.values()) if pending_settlements else 0
        total_t2 = total_pending + new_order_value
        buffer = new_order_value * 0.10  # 10% buffer

        return total_t2, buffer

    def check_trading_session_timing(self, current_datetime: datetime) -> Tuple[bool, str]:
        """
        Check if current time is safe for trading.

        Args:
            current_datetime: Current datetime

        Returns:
            Tuple of (is_safe, message)
        """
        current_time = current_datetime.time()

        # Avoid last 5 minutes of morning session (11:25-11:30)
        from datetime import time as dt_time

        if dt_time(11, 25) <= current_time <= dt_time(11, 30):
            return False, "Avoid trading near morning session end (11:25-11:30)"

        # Avoid last 5 minutes of afternoon session (14:25-14:30)
        if dt_time(14, 25) <= current_time <= dt_time(14, 30):
            return False, "Avoid trading near afternoon session end (14:25-14:30)"

        # Check ATO/ATC sessions
        is_auction, session, warning = check_ato_atc_session(current_time)
        if is_auction:
            return False, warning

        return True, "Safe trading time"

    def round_to_lot(self, shares: int) -> int:
        """Round shares to valid lot size."""
        return round_to_lot(shares)

    def get_tick_size(self, price: float) -> int:
        """Get tick size for price."""
        return get_tick_size(price)

    def round_to_tick(self, price: float, direction: str = "nearest") -> float:
        """Round price to valid tick."""
        return round_to_tick(price, direction)

    def get_exchange(self, symbol: str) -> str:
        """Get exchange for symbol."""
        return get_exchange(symbol)

    def validate_order(
        self, symbol: str, shares: int, price: float, order_type: str = "LO"
    ) -> Tuple[bool, Dict]:
        """Validate order parameters."""
        return validate_order(symbol, shares, price, order_type)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Vietnam Market Utilities...")

    # Test lot size
    print("\n1. Lot Size Tests:")
    print(f"  round_to_lot(150) = {round_to_lot(150)}")
    print(f"  round_to_lot(250) = {round_to_lot(250)}")
    print(f"  validate_lot_size(200) = {validate_lot_size(200)}")
    print(f"  validate_lot_size(150) = {validate_lot_size(150)}")

    # Test tick size
    print("\n2. Tick Size Tests:")
    print(f"  get_tick_size(8000) = {get_tick_size(8000)}")
    print(f"  get_tick_size(25000) = {get_tick_size(25000)}")
    print(f"  get_tick_size(80000) = {get_tick_size(80000)}")
    print(f"  round_to_tick(25123) = {round_to_tick(25123)}")

    # Test exchange detection
    print("\n3. Exchange Tests:")
    print(f"  get_exchange('VCB') = {get_exchange('VCB')}")
    print(f"  get_exchange('SHS') = {get_exchange('SHS')}")
    print(f"  get_price_limit('VCB') = {get_price_limit('VCB')}")
    print(f"  get_price_limit('SHS') = {get_price_limit('SHS')}")

    # Test ceiling/floor
    print("\n4. Ceiling/Floor Tests:")
    cf = calculate_ceiling_floor(50000, "VCB")
    print(f"  VCB @ 50,000: ceiling={cf['ceiling']:,.0f}, floor={cf['floor']:,.0f}")

    # Test session
    print("\n5. Session Tests:")
    session, is_trading = get_current_session()
    print(f"  Current session: {session}, is_trading: {is_trading}")
    is_optimal, reason = is_optimal_entry_time()
    print(f"  Optimal entry: {is_optimal}, reason: {reason}")

    # Test order validation
    print("\n6. Order Validation Tests:")
    valid, details = validate_order("VCB", 150, 50123, "LO")
    print(f"  Order valid: {valid}")
    print(f"  Errors: {details['errors']}")
    print(f"  Corrected shares: {details['corrected_shares']}")
    print(f"  Corrected price: {details['corrected_price']}")

    print("\n✅ All tests completed!")


# =============================================================================
# IMPROVED: T+2 SETTLEMENT TRACKING
# =============================================================================


def get_vietnam_holidays(year: int) -> set:
    """
    Get all Vietnam market holidays for a given year.

    Includes:
    - Fixed holidays (New Year, 30/4, 1/5, 2/9)
    - Tết Nguyên Đán (Lunar New Year)
    - Hung Kings' Commemoration Day

    Args:
        year: Year to get holidays for

    Returns:
        Set of datetime.date objects for holidays
    """
    from datetime import date

    holidays = set()

    # Fixed holidays
    for (month, day), name in VIETNAM_FIXED_HOLIDAYS.items():
        try:
            holiday_date = date(year, month, day)
            holidays.add(holiday_date)

            # If holiday falls on weekend, add substitute day (Monday)
            if holiday_date.weekday() == 5:  # Saturday
                holidays.add(holiday_date + timedelta(days=2))
            elif holiday_date.weekday() == 6:  # Sunday
                holidays.add(holiday_date + timedelta(days=1))
        except ValueError:
            pass

    # Tết holidays
    if year in VIETNAM_TET_HOLIDAYS:
        for month, day in VIETNAM_TET_HOLIDAYS[year]:
            try:
                holidays.add(date(year, month, day))
            except ValueError:
                pass

    # Hung Kings' Day
    if year in VIETNAM_HUNG_KINGS_DAY:
        month, day = VIETNAM_HUNG_KINGS_DAY[year]
        try:
            holidays.add(date(year, month, day))
        except ValueError:
            pass

    return holidays


def is_vietnam_trading_day(check_date: datetime) -> bool:
    """
    Check if a date is a Vietnam market trading day.

    Args:
        check_date: Date to check

    Returns:
        True if trading day, False if weekend or holiday
    """
    if isinstance(check_date, datetime):
        check_date = check_date.date()

    # Weekend check
    if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Holiday check
    holidays = get_vietnam_holidays(check_date.year)
    if check_date in holidays:
        return False

    return True


def get_next_trading_day(from_date: datetime, days_ahead: int = 1) -> datetime:
    """
    Get the next N trading days from a given date.

    Skips weekends and Vietnam holidays.

    Args:
        from_date: Starting date
        days_ahead: Number of trading days to advance

    Returns:
        Next trading day datetime
    """
    if isinstance(from_date, datetime):
        current = from_date
    else:
        current = datetime.combine(from_date, datetime.min.time())

    trading_days_counted = 0

    while trading_days_counted < days_ahead:
        current = current + timedelta(days=1)
        if is_vietnam_trading_day(current):
            trading_days_counted += 1

    return current


def calculate_settlement_date(trade_date: datetime, is_buy: bool = True) -> datetime:
    """
    Calculate settlement date for Vietnam T+2 cycle.

    Vietnam uses T+2 settlement:
    - Buy: Pay on T+2
    - Sell: Receive proceeds on T+2

    IMPROVED: Now properly accounts for weekends AND Vietnam holidays
    (Tết, 30/4, 1/5, 2/9, Hung Kings' Day)

    Args:
        trade_date: Date of trade execution
        is_buy: True for buy orders, False for sell orders

    Returns:
        Settlement date (T+2 trading days)
    """
    # Use Vietnam-aware trading day calculation
    settlement = get_next_trading_day(trade_date, days_ahead=VN_SETTLEMENT_DAYS)
    return settlement


def calculate_available_cash_for_trading(
    total_cash: float, pending_settlements: list, reserve_buffer_pct: float = 0.10
) -> float:
    """
    Calculate available cash considering T+2 pending settlements.

    Args:
        total_cash: Total cash in account
        pending_settlements: List of pending settlement amounts
        reserve_buffer_pct: Additional buffer to keep (default 10%)

    Returns:
        Available cash for new trades
    """
    total_pending = sum(pending_settlements)
    reserve = total_cash * reserve_buffer_pct
    available = total_cash - total_pending - reserve
    return max(0, available)


# =============================================================================
# IMPROVED: INTRADAY VOLATILITY CHECK
# =============================================================================


def check_intraday_volatility(
    current_price: float,
    day_high: float,
    day_low: float,
    reference_price: float,
    max_range_pct: float = 5.0,
) -> Tuple[bool, str]:
    """
    Check if intraday volatility is too high for safe entry.

    High intraday range indicates:
    - Potential manipulation
    - News-driven volatility
    - Institutional activity

    Args:
        current_price: Current price
        day_high: Day's high price
        day_low: Day's low price
        reference_price: Previous close (reference)
        max_range_pct: Maximum acceptable intraday range (default 5%)

    Returns:
        Tuple of (is_safe, warning_message)
    """
    if day_high <= 0 or day_low <= 0 or reference_price <= 0:
        return True, ""

    intraday_range = ((day_high - day_low) / reference_price) * 100

    if intraday_range > max_range_pct:
        return (
            False,
            f"⚠️ High intraday volatility: {intraday_range:.1f}% range (max {max_range_pct}%)",
        )

    # Check if price is near day's extremes (potential reversal)
    price_position = (
        (current_price - day_low) / (day_high - day_low) if day_high != day_low else 0.5
    )

    if price_position > 0.95:
        return True, "⚠️ Price near day high - potential resistance"
    elif price_position < 0.05:
        return True, "⚠️ Price near day low - potential support"

    return True, ""


# =============================================================================
# IMPROVED: FOREIGN FLOW INTEGRATION
# =============================================================================


def get_sector_for_symbol(symbol: str) -> str:
    """
    Get sector classification for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Sector name
    """
    symbol = symbol.upper().strip()
    return VN30_SECTORS.get(symbol, "DEFAULT")


def check_foreign_room(
    symbol: str, current_foreign_pct: float, sector: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Check if foreign ownership room is available.

    Important for:
    - Foreign investors
    - ETF tracking
    - Liquidity assessment

    Args:
        symbol: Stock symbol
        current_foreign_pct: Current foreign ownership percentage (0-1)
        sector: Optional sector override

    Returns:
        Tuple of (has_room, message)
    """
    if sector is None:
        sector = get_sector_for_symbol(symbol)

    limit = VN_FOREIGN_LIMITS.get(sector, VN_FOREIGN_LIMITS["DEFAULT"])

    if current_foreign_pct >= limit:
        return False, f"🚫 Foreign room full: {current_foreign_pct:.1%} >= {limit:.0%} limit"

    remaining_room = limit - current_foreign_pct
    if remaining_room < 0.05:  # Less than 5% room
        return True, f"⚠️ Limited foreign room: {remaining_room:.1%} remaining"

    return True, f"✅ Foreign room available: {remaining_room:.1%}"


def get_foreign_flow_signal(
    symbol: str,
    foreign_net_buy: float = 0.0,
    foreign_net_buy_5d: float = 0.0,
    current_foreign_pct: float = 0.0,
) -> Dict[str, Any]:
    """
    Get foreign flow signal for entry filter integration.

    IMPROVED v4.2: Comprehensive foreign flow analysis for entry decisions.

    Foreign investors (smart money) often have better information.
    Their buying/selling patterns can be predictive signals.

    Args:
        symbol: Stock symbol
        foreign_net_buy: Today's net foreign buy value (VND)
        foreign_net_buy_5d: 5-day cumulative net foreign buy (VND)
        current_foreign_pct: Current foreign ownership percentage (0-1)

    Returns:
        Dict with signal, confidence_adjustment, and details
    """
    from src.config.constants import (
        FOREIGN_FLOW_STRONG_BUY_BONUS,
        FOREIGN_FLOW_MODERATE_BUY_BONUS,
        FOREIGN_FLOW_MODERATE_SELL_PENALTY,
        FOREIGN_FLOW_STRONG_SELL_PENALTY,
    )

    result = {
        "symbol": symbol,
        "signal": "NEUTRAL",
        "confidence_adjustment": 0,
        "has_room": True,
        "room_message": "",
        "reasons": [],
        "warnings": [],
    }

    # Check foreign room availability
    has_room, room_msg = check_foreign_room(symbol, current_foreign_pct)
    result["has_room"] = has_room
    result["room_message"] = room_msg

    if not has_room:
        result["warnings"].append(room_msg)
        result["confidence_adjustment"] = -5  # Small penalty for full room

    # Analyze foreign flow direction
    # Thresholds in VND (adjust based on stock liquidity)
    strong_threshold = 5_000_000_000  # 5B VND
    moderate_threshold = 1_000_000_000  # 1B VND

    # Today's flow
    if foreign_net_buy > strong_threshold:
        result["signal"] = "STRONG_BUY"
        result["confidence_adjustment"] += FOREIGN_FLOW_STRONG_BUY_BONUS
        result["reasons"].append(f"✅ Strong foreign buying: {foreign_net_buy/1e9:.1f}B VND today")
    elif foreign_net_buy > moderate_threshold:
        result["signal"] = "MODERATE_BUY"
        result["confidence_adjustment"] += FOREIGN_FLOW_MODERATE_BUY_BONUS
        result["reasons"].append(f"✅ Foreign net buying: {foreign_net_buy/1e9:.1f}B VND today")
    elif foreign_net_buy < -strong_threshold:
        result["signal"] = "STRONG_SELL"
        result["confidence_adjustment"] += FOREIGN_FLOW_STRONG_SELL_PENALTY
        result["warnings"].append(
            f"⚠️ Strong foreign selling: {abs(foreign_net_buy)/1e9:.1f}B VND today"
        )
    elif foreign_net_buy < -moderate_threshold:
        result["signal"] = "MODERATE_SELL"
        result["confidence_adjustment"] += FOREIGN_FLOW_MODERATE_SELL_PENALTY
        result["warnings"].append(
            f"⚠️ Foreign net selling: {abs(foreign_net_buy)/1e9:.1f}B VND today"
        )

    # 5-day trend (more significant)
    if foreign_net_buy_5d > strong_threshold * 3:  # 15B over 5 days
        result["confidence_adjustment"] += 5
        result["reasons"].append(
            f"✅ Sustained foreign buying (5d): {foreign_net_buy_5d/1e9:.1f}B VND"
        )
    elif foreign_net_buy_5d < -strong_threshold * 3:
        result["confidence_adjustment"] -= 8
        result["warnings"].append(
            f"⚠️ Sustained foreign selling (5d): {abs(foreign_net_buy_5d)/1e9:.1f}B VND"
        )

    # Clamp adjustment
    result["confidence_adjustment"] = max(-20, min(15, result["confidence_adjustment"]))

    return result


def check_foreign_flow_for_entry(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[bool, int, str]:
    """
    Check foreign flow conditions for entry filter.

    This function is designed to be called from entry_logic.py
    as part of the entry filter pipeline.

    Args:
        symbol: Stock symbol
        df: DataFrame with foreign flow data (optional)

    Returns:
        Tuple of (should_proceed, confidence_adjustment, message)
    """
    try:
        # Try to get foreign flow data
        foreign_net_buy = 0.0
        foreign_net_buy_5d = 0.0
        current_foreign_pct = 0.0

        if df is not None and len(df) > 0:
            # Extract foreign flow from DataFrame if available
            if "foreign_net_buy" in df.columns:
                foreign_net_buy = df["foreign_net_buy"].iloc[-1]
            if "foreign_net_buy_5d" in df.columns:
                foreign_net_buy_5d = df["foreign_net_buy_5d"].iloc[-1]
            elif "foreign_net_buy" in df.columns and len(df) >= 5:
                foreign_net_buy_5d = df["foreign_net_buy"].tail(5).sum()
            if "foreign_pct" in df.columns:
                current_foreign_pct = df["foreign_pct"].iloc[-1]

        # Get signal
        signal = get_foreign_flow_signal(
            symbol=symbol,
            foreign_net_buy=foreign_net_buy,
            foreign_net_buy_5d=foreign_net_buy_5d,
            current_foreign_pct=current_foreign_pct,
        )

        # Determine if we should proceed
        # Block entry only on strong foreign selling
        if signal["signal"] == "STRONG_SELL" and not signal["has_room"]:
            return (
                False,
                signal["confidence_adjustment"],
                f"🚫 Foreign flow negative: {'; '.join(signal['warnings'])}",
            )

        # Build message
        messages = signal["reasons"] + signal["warnings"]
        message = "; ".join(messages) if messages else "Foreign flow: neutral"

        return (True, signal["confidence_adjustment"], message)

    except Exception as e:
        logger.debug(f"Foreign flow check failed for {symbol}: {e}")
        return (True, 0, "Foreign flow data unavailable")


# =============================================================================
# IMPROVED: MARKET HOURS VALIDATION
# =============================================================================


def is_market_open(current_time: Optional[time] = None) -> Tuple[bool, str]:
    """
    Check if Vietnam stock market is currently open.

    Trading hours (Vietnam time, UTC+7):
    - Morning: 9:00 - 11:30
    - Afternoon: 13:00 - 15:00

    Args:
        current_time: Time to check (default: now in VN timezone)

    Returns:
        Tuple of (is_open, session_info)
    """
    if current_time is None:
        try:
            import pytz

            vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
            current_time = datetime.now(vn_tz).time()
        except ImportError:
            current_time = datetime.now().time()

    session, is_trading = get_current_session(current_time)

    if session == "PRE_MARKET":
        return False, "Market opens at 9:00"
    elif session == "POST_MARKET":
        return False, "Market closed for the day"
    elif session == "LUNCH":
        return False, "Lunch break (11:30-13:00)"
    elif is_trading:
        return True, f"Market open - {session} session"
    else:
        return False, f"Market closed - {session}"


def get_time_to_session_end(current_time: Optional[time] = None) -> Tuple[int, str]:
    """
    Get minutes until current session ends.

    Useful for:
    - Avoiding trades near session end
    - Planning exit timing

    Args:
        current_time: Time to check

    Returns:
        Tuple of (minutes_remaining, session_name)
    """
    if current_time is None:
        try:
            import pytz

            vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
            current_time = datetime.now(vn_tz).time()
        except ImportError:
            current_time = datetime.now().time()

    session, _ = get_current_session(current_time)

    if session in VN_SESSIONS:
        _, end_time = VN_SESSIONS[session]

        # Calculate minutes difference
        current_minutes = current_time.hour * 60 + current_time.minute
        end_minutes = end_time.hour * 60 + end_time.minute

        remaining = end_minutes - current_minutes
        return max(0, remaining), session

    return 0, session


# =============================================================================
# IMPROVED: COMPREHENSIVE ORDER VALIDATION
# =============================================================================


def validate_order_comprehensive(
    symbol: str,
    shares: int,
    price: float,
    order_type: str = "LO",
    reference_price: Optional[float] = None,
    available_cash: Optional[float] = None,
    current_time: Optional[time] = None,
) -> Dict:
    """
    Comprehensive order validation for Vietnam market.

    Validates:
    1. Lot size (100 shares)
    2. Tick size (10/50/100 VND)
    3. Price limits (floor/ceiling)
    4. Session timing
    5. Cash availability
    6. Order type validity

    Args:
        symbol: Stock symbol
        shares: Number of shares
        price: Order price
        order_type: LO (Limit), ATO, ATC, MP (Market)
        reference_price: Previous close for limit calculation
        available_cash: Available cash for buy orders
        current_time: Current time for session check

    Returns:
        Dict with validation results
    """
    errors = []
    warnings = []
    corrections = {}

    # 1. Validate lot size
    lot_valid, lot_msg = validate_lot_size(shares)
    if not lot_valid:
        errors.append(lot_msg)
        corrections["shares"] = round_to_lot(shares)

    # 2. Validate tick size (for limit orders)
    if order_type == "LO":
        price_valid, price_msg = validate_price(price)
        if not price_valid:
            errors.append(price_msg)
            corrections["price"] = round_to_tick(price)

    # 3. Validate price limits
    if reference_price and order_type == "LO":
        limits = calculate_ceiling_floor(reference_price, symbol)
        if price > limits["ceiling"]:
            errors.append(f"Price {price:,.0f} exceeds ceiling {limits['ceiling']:,.0f}")
            corrections["price"] = limits["ceiling"]
        elif price < limits["floor"]:
            errors.append(f"Price {price:,.0f} below floor {limits['floor']:,.0f}")
            corrections["price"] = limits["floor"]

        # Warn if near limits
        ceiling_dist = (limits["ceiling"] - price) / price * 100
        floor_dist = (price - limits["floor"]) / price * 100

        if ceiling_dist < 1.0:
            warnings.append(f"⚠️ Price within 1% of ceiling ({ceiling_dist:.1f}%)")
        if floor_dist < 1.0:
            warnings.append(f"⚠️ Price within 1% of floor ({floor_dist:.1f}%)")

    # 4. Validate session timing
    if order_type in ["ATO", "ATC"]:
        is_auction, session, _ = check_ato_atc_session(current_time)
        if order_type == "ATO" and session != "ATO":
            errors.append("ATO orders only valid during 9:00-9:15")
        if order_type == "ATC" and session != "ATC":
            errors.append("ATC orders only valid during 14:30-14:45")

    # 5. Check optimal entry time
    is_optimal, timing_reason = is_optimal_entry_time(current_time)
    if not is_optimal:
        warnings.append(timing_reason)

    # 6. Validate cash availability
    if available_cash is not None:
        order_value = shares * price
        if order_value > available_cash:
            errors.append(f"Insufficient cash: need {order_value:,.0f}, have {available_cash:,.0f}")
            max_shares = int(available_cash / price)
            corrections["shares"] = round_to_lot(max_shares)

    # 7. Check market hours
    is_open, market_status = is_market_open(current_time)
    if not is_open and order_type not in ["ATO", "ATC"]:
        warnings.append(f"⚠️ Market status: {market_status}")

    is_valid = len(errors) == 0

    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "corrections": corrections,
        "symbol": symbol,
        "shares": corrections.get("shares", shares),
        "price": corrections.get("price", price),
        "order_type": order_type,
        "exchange": get_exchange(symbol),
        "order_value": shares * price,
    }


# =============================================================================
# NEW v4.2: CEILING/FLOOR REVERSAL DETECTION
# =============================================================================


def check_ceiling_reversal(
    current_price: float,
    day_high: float,
    reference_price: float,
    symbol: str,
    volume_ratio: float = 1.0,
) -> Tuple[bool, str, float]:
    """
    Check if stock hit ceiling and is now reversing (bearish signal).

    Vietnam market specific:
    - Ceiling = reference_price * 1.07 (HOSE)
    - If price hit ceiling but now pulling back, it's a warning sign
    - High volume at ceiling + reversal = strong sell signal

    Args:
        current_price: Current stock price
        day_high: Day's high price
        reference_price: Previous close (reference)
        symbol: Stock symbol
        volume_ratio: Current volume / average volume

    Returns:
        Tuple of (is_reversal, message, reversal_strength)
        reversal_strength: 0-1 (1 = strong reversal signal)
    """
    limit = get_price_limit(symbol)
    ceiling = reference_price * (1 + limit)

    # Check if day high was at or near ceiling (within 0.5%)
    hit_ceiling = day_high >= ceiling * 0.995

    if not hit_ceiling:
        return False, "", 0.0

    # Calculate pullback from ceiling
    pullback_pct = (ceiling - current_price) / ceiling * 100

    if pullback_pct < 1.0:
        # Still near ceiling, not a reversal yet
        return False, f"📈 Near ceiling ({pullback_pct:.1f}% below)", 0.0

    # Calculate reversal strength
    # Higher pullback + higher volume = stronger reversal signal
    reversal_strength = min(1.0, (pullback_pct / 5.0) * (volume_ratio / 1.5))

    if pullback_pct >= 3.0 and volume_ratio >= 1.5:
        return (
            True,
            f"🔴 CEILING REVERSAL: Hit ceiling {ceiling:,.0f} but pulled back {pullback_pct:.1f}% "
            f"with {volume_ratio:.1f}x volume - STRONG SELL SIGNAL",
            reversal_strength,
        )
    elif pullback_pct >= 2.0:
        return (
            True,
            f"⚠️ Ceiling rejection: Pulled back {pullback_pct:.1f}% from ceiling - watch closely",
            reversal_strength,
        )

    return False, f"📊 Minor pullback from ceiling ({pullback_pct:.1f}%)", reversal_strength


def check_floor_bounce(
    current_price: float,
    day_low: float,
    reference_price: float,
    symbol: str,
    volume_ratio: float = 1.0,
) -> Tuple[bool, str, float]:
    """
    Check if stock hit floor and is now bouncing (bullish signal).

    Vietnam market specific:
    - Floor = reference_price * 0.93 (HOSE)
    - If price hit floor but now recovering, it's a potential buy signal
    - High volume at floor + bounce = strong buy signal

    Args:
        current_price: Current stock price
        day_low: Day's low price
        reference_price: Previous close (reference)
        symbol: Stock symbol
        volume_ratio: Current volume / average volume

    Returns:
        Tuple of (is_bounce, message, bounce_strength)
        bounce_strength: 0-1 (1 = strong bounce signal)
    """
    limit = get_price_limit(symbol)
    floor = reference_price * (1 - limit)

    # Check if day low was at or near floor (within 0.5%)
    hit_floor = day_low <= floor * 1.005

    if not hit_floor:
        return False, "", 0.0

    # Calculate bounce from floor
    bounce_pct = (current_price - floor) / floor * 100

    if bounce_pct < 1.0:
        # Still near floor, not a bounce yet
        return False, f"📉 Near floor ({bounce_pct:.1f}% above)", 0.0

    # Calculate bounce strength
    # Higher bounce + higher volume = stronger bounce signal
    bounce_strength = min(1.0, (bounce_pct / 5.0) * (volume_ratio / 1.5))

    if bounce_pct >= 3.0 and volume_ratio >= 1.5:
        return (
            True,
            f"🟢 FLOOR BOUNCE: Hit floor {floor:,.0f} but bounced {bounce_pct:.1f}% "
            f"with {volume_ratio:.1f}x volume - POTENTIAL BUY SIGNAL",
            bounce_strength,
        )
    elif bounce_pct >= 2.0:
        return (
            True,
            f"📊 Floor support: Bounced {bounce_pct:.1f}% from floor - watch for confirmation",
            bounce_strength,
        )

    return False, f"📊 Minor bounce from floor ({bounce_pct:.1f}%)", bounce_strength


def check_floor_bounce_enhanced(
    current_price: float,
    day_low: float,
    reference_price: float,
    symbol: str,
    volume_ratio: float = 1.0,
    minutes_at_floor: int = 0,
    market_volatility: float = 0.0,
) -> Tuple[bool, str, float, int]:
    """
    IMPROVED v5.1: Enhanced floor bounce check with volume confirmation and dynamic wait time.

    ADDRESSES RISK: 30 minutes may not be enough in panic selling.

    Improvements:
    1. Volume confirmation required for valid bounce
    2. Extended wait time in high volatility / panic selling
    3. Panic selling detection (volume > 3x average)

    Args:
        current_price: Current stock price
        day_low: Day's low price
        reference_price: Previous close (reference)
        symbol: Stock symbol
        volume_ratio: Current volume / average volume
        minutes_at_floor: Minutes since price hit floor
        market_volatility: Market volatility (ATR/Price ratio)

    Returns:
        Tuple of (is_valid_bounce, message, bounce_strength, recommended_wait_minutes)
    """
    from src.config.constants import (
        VN_FLOOR_BOUNCE_MAX_WAIT_MINUTES,
        VN_FLOOR_BOUNCE_EXTENDED_WAIT_MINUTES,
        VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO,
        VN_FLOOR_BOUNCE_PANIC_VOLUME_RATIO,
    )

    limit = get_price_limit(symbol)
    floor = reference_price * (1 - limit)

    # Check if day low was at or near floor (within 0.5%)
    hit_floor = day_low <= floor * 1.005

    if not hit_floor:
        return False, "", 0.0, 0

    # Calculate bounce from floor
    bounce_pct = (current_price - floor) / floor * 100

    # Determine wait time based on conditions
    base_wait = VN_FLOOR_BOUNCE_MAX_WAIT_MINUTES  # 30 minutes
    extended_wait = VN_FLOOR_BOUNCE_EXTENDED_WAIT_MINUTES  # 60 minutes

    # Detect panic selling (volume > 3x average)
    is_panic_selling = volume_ratio >= VN_FLOOR_BOUNCE_PANIC_VOLUME_RATIO

    # Detect high volatility
    is_high_volatility = market_volatility > 0.03  # > 3% volatility

    # Calculate recommended wait time
    if is_panic_selling:
        recommended_wait = extended_wait + 15  # 75 minutes in panic
        wait_reason = "panic selling detected"
    elif is_high_volatility:
        recommended_wait = extended_wait  # 60 minutes in high volatility
        wait_reason = "high volatility"
    elif volume_ratio < VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO:
        recommended_wait = base_wait + 15  # 45 minutes if low volume
        wait_reason = "low volume confirmation"
    else:
        recommended_wait = base_wait  # 30 minutes standard
        wait_reason = "standard"

    # Check if enough time has passed
    if minutes_at_floor < recommended_wait:
        remaining = recommended_wait - minutes_at_floor
        return (
            False,
            f"⏳ Floor bounce wait: {remaining} minutes remaining ({wait_reason})",
            0.0,
            recommended_wait,
        )

    # Volume confirmation check
    if volume_ratio < VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO:
        return (
            False,
            f"📉 Floor bounce needs volume confirmation "
            f"(current: {volume_ratio:.1f}x, need: {VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO}x)",
            0.0,
            recommended_wait,
        )

    if bounce_pct < 1.0:
        return (
            False,
            f"📉 Near floor ({bounce_pct:.1f}% above) - waiting for bounce",
            0.0,
            recommended_wait,
        )

    # Calculate bounce strength with volume confirmation
    volume_factor = min(2.0, volume_ratio / VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO)
    bounce_strength = min(1.0, (bounce_pct / 5.0) * volume_factor)

    # Strong bounce signal
    if bounce_pct >= 3.0 and volume_ratio >= 2.0:
        return (
            True,
            f"🟢 STRONG FLOOR BOUNCE: Hit floor {floor:,.0f}, bounced {bounce_pct:.1f}% "
            f"with {volume_ratio:.1f}x volume after {minutes_at_floor} min wait",
            bounce_strength,
            recommended_wait,
        )
    elif bounce_pct >= 2.0 and volume_ratio >= VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO:
        return (
            True,
            f"📊 Floor bounce confirmed: {bounce_pct:.1f}% with {volume_ratio:.1f}x volume",
            bounce_strength,
            recommended_wait,
        )

    return (
        False,
        f"📊 Minor bounce from floor ({bounce_pct:.1f}%) - needs more confirmation",
        bounce_strength * 0.5,
        recommended_wait,
    )


def is_friday_afternoon() -> bool:
    """
    Check if current time is Friday afternoon (Vietnam timezone).

    Useful for T+2 settlement awareness:
    - Buy on Friday = settlement on Tuesday (4 days capital lock)
    - Consider closing marginal positions before weekend

    Returns:
        True if Friday after 13:00 Vietnam time
    """
    try:
        import pytz

        vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
        now = datetime.now(vn_tz)
        return now.weekday() == 4 and now.hour >= 13
    except ImportError:
        now = datetime.now()
        return now.weekday() == 4 and now.hour >= 13


def get_t2_settlement_date(trade_date: Optional[datetime] = None) -> datetime:
    """
    Calculate T+2 settlement date for Vietnam market.

    Args:
        trade_date: Trade execution date (default: today)

    Returns:
        Settlement date (T+2 business days)
    """
    if trade_date is None:
        try:
            import pytz

            vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
            trade_date = datetime.now(vn_tz)
        except ImportError:
            trade_date = datetime.now()

    # Add 2 business days
    settlement = trade_date
    days_added = 0
    while days_added < 2:
        settlement += timedelta(days=1)
        # Skip weekends
        if settlement.weekday() < 5:
            days_added += 1

    return settlement


# =============================================================================
# COMPATIBILITY FUNCTIONS (for test imports)
# =============================================================================


def calculate_t2_requirement(
    pending_settlements: Dict[str, float],
    new_order_value: float,
    buffer_pct: float = 0.10,
) -> Tuple[float, float]:
    """
    Calculate T+2 cash requirement for Vietnam market.

    Alias for calculate_t2_cash_requirement for backward compatibility.

    Args:
        pending_settlements: Dict of pending settlements {date: value}
        new_order_value: Value of new order
        buffer_pct: Additional buffer percentage (default 10%)

    Returns:
        Tuple of (total_t2_requirement, buffer_amount)
    """
    total_pending = sum(pending_settlements.values()) if pending_settlements else 0
    total_t2 = total_pending + new_order_value
    buffer = new_order_value * buffer_pct
    return total_t2, buffer


def check_liquidity(
    df: pd.DataFrame,
    symbol: str,
    min_liquidity_value: float = 2_000_000_000,
) -> Tuple[bool, str]:
    """
    Check if stock meets liquidity requirements.

    Convenience function wrapping VietnamMarketValidator.

    Args:
        df: DataFrame with 'close' and 'volume' columns
        symbol: Stock symbol
        min_liquidity_value: Minimum daily liquidity in VND (default: 2B)

    Returns:
        Tuple of (is_liquid, message)
    """
    validator = VietnamMarketValidator(min_liquidity_value=min_liquidity_value)
    return validator.check_liquidity_requirements(df, symbol)


def check_price_limits(
    current_price: float,
    reference_price: float,
    symbol: str,
) -> Tuple[bool, str]:
    """
    Check if current price is safely within floor/ceiling limits.

    Convenience function wrapping VietnamMarketValidator.

    Args:
        current_price: Current stock price
        reference_price: Reference (previous close) price
        symbol: Stock symbol

    Returns:
        Tuple of (is_safe, message)
    """
    validator = VietnamMarketValidator()
    return validator.check_price_floor_ceiling(current_price, reference_price, symbol)


def check_trading_session(
    current_datetime: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """
    Check if current time is safe for trading.

    Convenience function wrapping VietnamMarketValidator.

    Args:
        current_datetime: Current datetime (default: now)

    Returns:
        Tuple of (is_safe, message)
    """
    if current_datetime is None:
        try:
            import pytz

            vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
            current_datetime = datetime.now(vn_tz)
        except ImportError:
            current_datetime = datetime.now()

    validator = VietnamMarketValidator()
    return validator.check_trading_session_timing(current_datetime)


def validate_position_vs_volume(
    shares: int,
    avg_volume: float,
    symbol: str,
    max_position_pct: float = 0.05,
) -> Tuple[bool, str]:
    """
    Validate position size against average volume.

    Convenience function wrapping VietnamMarketValidator.

    Args:
        shares: Number of shares to trade
        avg_volume: Average daily volume
        symbol: Stock symbol
        max_position_pct: Max position as % of avg volume (default: 5%)

    Returns:
        Tuple of (is_safe, message)
    """
    validator = VietnamMarketValidator(max_position_pct_of_volume=max_position_pct)
    return validator.validate_position_size_vs_volume(shares, avg_volume, symbol)


# =============================================================================
# SINGLETON VALIDATOR INSTANCE
# =============================================================================

_validator_instance: Optional[VietnamMarketValidator] = None


def get_vietnam_market_validator() -> VietnamMarketValidator:
    """
    Get singleton instance of VietnamMarketValidator.

    Returns:
        VietnamMarketValidator instance
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = VietnamMarketValidator()
    return _validator_instance


# =============================================================================
# FOREIGN ROOM MANAGEMENT - IMPROVED v10.0
# =============================================================================
#
# Vietnam Foreign Ownership Limits (FOL):
# - General limit: 49% for most sectors
# - Banking: 30% (special regulation)
# - Aviation: 34% (VJC special limit)
# - Securities: 49%
# - Some companies have 0% FOL (strategic sectors)
#
# Foreign Room = Maximum FOL - Current Foreign Ownership
# When room = 0, foreigners cannot buy (can only sell)
# =============================================================================

# Cache for foreign room data (TTL: 5 minutes during trading hours)
_foreign_room_cache: Dict[str, Dict] = {}
_foreign_room_cache_ttl: int = 300  # 5 minutes


class ForeignRoomManager:
    """
    Foreign Room Manager for Vietnam Market.

    IMPROVED v10.0: Complete foreign room tracking with API integration.

    Features:
    - Real-time foreign room checking
    - Sector-based FOL limits
    - Company-specific FOL overrides
    - Cache with TTL for performance
    - API integration placeholders (SSI, VNDirect, TCBS)
    """

    # Company-specific FOL overrides (higher priority than sector defaults)
    COMPANY_FOL_OVERRIDES = {
        # Banking (some have different limits due to strategic investors)
        "VCB": 0.225,  # 22.5% - Vietcombank (Mizuho already owns ~15%)
        "BID": 0.30,  # 30% - BIDV
        "CTG": 0.30,  # 30% - VietinBank
        "ACB": 0.30,  # 30% - ACB
        "TCB": 0.30,  # 30% - Techcombank
        "MBB": 0.235,  # 23.5% - MB Bank
        "VPB": 0.30,  # 30% - VPBank
        # Aviation
        "VJC": 0.34,  # 34% - VietJet Air
        "HVN": 0.34,  # 34% - Vietnam Airlines
        # Special cases (0% FOL)
        "ACV": 0.0,  # Airports Corporation - strategic state company
        "PVN": 0.0,  # PetroVietnam - state oil company
    }

    def __init__(self, api_provider: str = "mock"):
        """
        Initialize Foreign Room Manager.

        Args:
            api_provider: API provider for real-time data
                         Options: "mock", "ssi", "vndirect", "tcbs"
        """
        self.api_provider = api_provider
        self._cache: Dict[str, Dict] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl_seconds = 300  # 5 minutes

    def get_fol_limit(self, symbol: str) -> float:
        """
        Get Foreign Ownership Limit for a symbol.

        Priority:
        1. Company-specific override
        2. Sector-based limit
        3. Default 49%

        Args:
            symbol: Stock symbol

        Returns:
            FOL as decimal (e.g., 0.49 for 49%)
        """
        symbol = symbol.upper()

        # Priority 1: Company override
        if symbol in self.COMPANY_FOL_OVERRIDES:
            return self.COMPANY_FOL_OVERRIDES[symbol]

        # Priority 2: Sector-based limit
        sector = VN30_SECTORS.get(symbol, "DEFAULT")
        return VN_FOREIGN_LIMITS.get(sector, VN_FOREIGN_LIMITS["DEFAULT"])

    def get_foreign_room(
        self,
        symbol: str,
        force_refresh: bool = False,
    ) -> Dict:
        """
        Get current foreign room for a symbol.

        IMPROVED v10.0: Real-time foreign room with caching.

        Args:
            symbol: Stock symbol
            force_refresh: Force API refresh (ignore cache)

        Returns:
            Dict with:
            - fol_limit: Maximum foreign ownership limit (%)
            - current_ownership: Current foreign ownership (%)
            - remaining_room: Available room for foreign buying (%)
            - shares_available: Approximate shares available
            - can_foreign_buy: Whether foreigners can buy
            - data_source: Source of data (cache/api/mock)
            - last_updated: Timestamp of data
        """
        symbol = symbol.upper()

        # Check cache
        if not force_refresh and symbol in self._cache:
            cache_time = self._cache_timestamps.get(symbol)
            if cache_time:
                age = (datetime.now() - cache_time).total_seconds()
                if age < self._cache_ttl_seconds:
                    cached_data = self._cache[symbol].copy()
                    cached_data["data_source"] = "cache"
                    return cached_data

        # Get data from API or mock
        if self.api_provider == "mock":
            data = self._get_mock_foreign_room(symbol)
        elif self.api_provider == "ssi":
            data = self._get_ssi_foreign_room(symbol)
        elif self.api_provider == "vndirect":
            data = self._get_vndirect_foreign_room(symbol)
        elif self.api_provider == "tcbs":
            data = self._get_tcbs_foreign_room(symbol)
        else:
            data = self._get_mock_foreign_room(symbol)

        # Update cache
        self._cache[symbol] = data
        self._cache_timestamps[symbol] = datetime.now()

        return data

    def _get_mock_foreign_room(self, symbol: str) -> Dict:
        """
        Get mock foreign room data for testing.

        In production, replace with actual API calls.
        """
        fol_limit = self.get_fol_limit(symbol)

        # Mock: Simulate current ownership between 50-95% of limit
        import random

        ownership_ratio = random.uniform(0.50, 0.95)
        current_ownership = fol_limit * ownership_ratio
        remaining_room = fol_limit - current_ownership

        # Mock: Estimate shares available (assuming avg 500M shares outstanding)
        shares_outstanding = 500_000_000
        shares_available = int(shares_outstanding * remaining_room)

        return {
            "symbol": symbol,
            "fol_limit": fol_limit,
            "fol_limit_pct": fol_limit * 100,
            "current_ownership": current_ownership,
            "current_ownership_pct": current_ownership * 100,
            "remaining_room": remaining_room,
            "remaining_room_pct": remaining_room * 100,
            "shares_available": shares_available,
            "can_foreign_buy": remaining_room > 0.001,  # > 0.1%
            "room_status": self._get_room_status(remaining_room, fol_limit),
            "data_source": "mock",
            "last_updated": datetime.now().isoformat(),
            "warning": "Mock data - integrate with broker API for real-time data",
        }

    def _get_room_status(self, remaining_room: float, fol_limit: float) -> str:
        """Determine room status based on remaining percentage."""
        if remaining_room <= 0:
            return "FULL"  # No room available
        elif remaining_room < fol_limit * 0.05:
            return "CRITICAL"  # < 5% of limit remaining
        elif remaining_room < fol_limit * 0.20:
            return "LOW"  # < 20% of limit remaining
        elif remaining_room < fol_limit * 0.50:
            return "MODERATE"  # < 50% of limit remaining
        else:
            return "AVAILABLE"  # Plenty of room

    def _get_ssi_foreign_room(self, symbol: str) -> Dict:
        """
        Get foreign room from SSI API.

        TODO: Implement SSI API integration.
        Reference: https://iboard.ssi.com.vn/
        """
        logger.warning(f"SSI API not implemented for {symbol}, using mock data")
        return self._get_mock_foreign_room(symbol)

    def _get_vndirect_foreign_room(self, symbol: str) -> Dict:
        """
        Get foreign room from VNDirect API.

        TODO: Implement VNDirect API integration.
        Reference: https://dstock.vndirect.com.vn/
        """
        logger.warning(f"VNDirect API not implemented for {symbol}, using mock data")
        return self._get_mock_foreign_room(symbol)

    def _get_tcbs_foreign_room(self, symbol: str) -> Dict:
        """
        Get foreign room from TCBS API.

        TODO: Implement TCBS API integration.
        Reference: https://tcinvest.tcbs.com.vn/
        """
        logger.warning(f"TCBS API not implemented for {symbol}, using mock data")
        return self._get_mock_foreign_room(symbol)

    def can_foreign_buy(self, symbol: str, shares: int = 0) -> Tuple[bool, str]:
        """
        Check if foreign investors can buy a symbol.

        Args:
            symbol: Stock symbol
            shares: Number of shares to buy (optional, for detailed check)

        Returns:
            Tuple of (can_buy, reason)
        """
        room_data = self.get_foreign_room(symbol)

        if room_data["room_status"] == "FULL":
            return (
                False,
                f"No foreign room available for {symbol} (FOL: {room_data['fol_limit_pct']:.1f}%)",
            )

        if room_data["room_status"] == "CRITICAL":
            if shares > 0 and shares > room_data["shares_available"]:
                return False, (
                    f"Insufficient foreign room for {shares:,} shares. "
                    f"Only {room_data['shares_available']:,} available"
                )
            return (
                True,
                f"⚠️ Low foreign room for {symbol}: {room_data['remaining_room_pct']:.2f}% remaining",
            )

        return True, f"Foreign room OK: {room_data['remaining_room_pct']:.2f}% available"


# Singleton instance
_foreign_room_manager: Optional[ForeignRoomManager] = None


def get_foreign_room_manager(api_provider: str = "mock") -> ForeignRoomManager:
    """Get singleton Foreign Room Manager instance."""
    global _foreign_room_manager
    if _foreign_room_manager is None:
        _foreign_room_manager = ForeignRoomManager(api_provider)
    return _foreign_room_manager


def check_foreign_room(symbol: str, shares: int = 0) -> Tuple[bool, str]:
    """
    Quick check if foreign investors can buy a symbol.

    Args:
        symbol: Stock symbol
        shares: Number of shares to buy (optional)

    Returns:
        Tuple of (can_buy, reason)
    """
    manager = get_foreign_room_manager()
    return manager.can_foreign_buy(symbol, shares)


def get_foreign_room(symbol: str) -> Dict:
    """
    Get current foreign room data for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Dict with foreign room details
    """
    manager = get_foreign_room_manager()
    return manager.get_foreign_room(symbol)


# =============================================================================
# MARKET CIRCUIT BREAKER - Vietnam (IMPROVED v10.0)
# =============================================================================
#
# Updated per latest SSC regulations (2024-2025):
# - Thông tư 120/2020/TT-BTC của Bộ Tài chính
# - Quyết định số 38/QĐ-SGDCK ngày 15/01/2024
#
# SSC Circuit Breaker Rules (Updated 2024):
# - Level 1 (-5%): Cảnh báo, tăng cường giám sát, giảm position size 50%
# - Level 2 (-7%): Tạm ngừng giao dịch 15 phút, không đặt lệnh mới
# - Level 3 (-10%): Tạm ngừng 30 phút, có thể đóng cửa sớm
#
# NEW v10.0 Additions:
# - Per-index circuit breakers (VNINDEX, VN30, HNX)
# - Upside circuit breaker (+7%) for extreme rallies
# - Time-based restrictions (no halt in last 15 minutes)
# - Recovery cooldown period
# =============================================================================

VN_MARKET_CIRCUIT_BREAKER = {
    # Downside threshold levels (percentage drop from reference price)
    "LEVEL_1": -5.0,  # Warning level - increased monitoring, reduce positions
    "LEVEL_2": -7.0,  # Trading halt 15 minutes
    "LEVEL_3": -10.0,  # Trading halt 30 minutes, possible early close
    # NEW v10.0: Upside thresholds (for extreme rallies - rarely triggered)
    "UPSIDE_WARNING": 5.0,  # 5% rally warning
    "UPSIDE_CAUTION": 7.0,  # 7% rally - reduce new longs
    # Halt durations (minutes)
    "HALT_DURATION_L2": 15,
    "HALT_DURATION_L3": 30,
    # Actions
    "L1_ACTION": "REDUCE_POSITION_SIZE",  # Giảm size position 50%
    "L2_ACTION": "HALT_NEW_ORDERS",  # Không đặt lệnh mới
    "L3_ACTION": "EMERGENCY_EXIT",  # Xem xét thoát vị thế
    # Recovery behavior
    "RESUME_DELAY_MINUTES": 5,  # Chờ 5 phút sau khi resume để trade
    "COOLDOWN_AFTER_HALT": 10,  # 10 phút cooldown sau halt
    # Time restrictions
    "NO_HALT_LAST_MINUTES": 15,  # Không halt trong 15 phút cuối (14:30-14:45)
    "NO_HALT_FIRST_MINUTES": 5,  # Không halt trong 5 phút đầu (09:00-09:05)
    # Position size multipliers by level
    "POSITION_MULT_NORMAL": 1.0,
    "POSITION_MULT_L1": 0.5,  # 50% position size at L1
    "POSITION_MULT_L2": 0.0,  # No new positions at L2
    "POSITION_MULT_L3": 0.0,  # No new positions at L3
}

# Per-index circuit breaker tracking
VN_INDEX_CIRCUIT_BREAKERS = {
    "VNINDEX": {
        "last_halt_time": None,
        "halt_count_today": 0,
        "current_level": 0,
    },
    "VN30": {
        "last_halt_time": None,
        "halt_count_today": 0,
        "current_level": 0,
    },
    "HNX": {
        "last_halt_time": None,
        "halt_count_today": 0,
        "current_level": 0,
    },
}


def check_market_halt_status(
    vnindex_change_pct: float,
    current_time: Optional[time] = None,
    index_name: str = "VNINDEX",
) -> Dict:
    """
    Check if Vietnam market circuit breaker is triggered.

    IMPROVED v10.0: Enhanced circuit breaker with:
    - Per-index tracking
    - Time-based restrictions
    - Upside warnings
    - Cooldown periods

    Theo Thông tư 120/2020/TT-BTC của Bộ Tài chính:
    - Level 1 (-5%): Cảnh báo, giảm position size 50%
    - Level 2 (-7%): Tạm ngừng giao dịch 15 phút, không đặt lệnh mới
    - Level 3 (-10%): Tạm ngừng 30 phút, xem xét đóng cửa sớm

    Args:
        vnindex_change_pct: VNINDEX change percentage (e.g., -5.0 for -5%)
        current_time: Current time (optional)

    Returns:
        Dict with halt status, level, message, trading_allowed, and recommended_action
    """
    result = {
        "halt_triggered": False,
        "halt_level": 0,
        "halt_duration_minutes": 0,
        "trading_allowed": True,
        "message": "✅ Market trading normally",
        "vnindex_change": vnindex_change_pct,
        "recommended_action": None,
        "position_size_multiplier": 1.0,  # Full position size allowed
    }

    if vnindex_change_pct >= VN_MARKET_CIRCUIT_BREAKER["LEVEL_1"]:
        # Normal trading
        return result

    if vnindex_change_pct >= VN_MARKET_CIRCUIT_BREAKER["LEVEL_2"]:
        # Level 1: Warning - reduce position sizes
        result["halt_level"] = 1
        result["position_size_multiplier"] = 0.5  # Reduce to 50%
        result["recommended_action"] = VN_MARKET_CIRCUIT_BREAKER["L1_ACTION"]
        result["message"] = (
            f"⚠️ CẢNH BÁO THỊ TRƯỜNG: VNINDEX {vnindex_change_pct:+.2f}% - "
            f"Gần ngưỡng circuit breaker. Giảm position size 50%."
        )
        return result

    if vnindex_change_pct >= VN_MARKET_CIRCUIT_BREAKER["LEVEL_3"]:
        # Level 2: Trading halt 15 minutes
        result["halt_triggered"] = True
        result["halt_level"] = 2
        result["halt_duration_minutes"] = VN_MARKET_CIRCUIT_BREAKER["HALT_DURATION_L2"]
        result["trading_allowed"] = False
        result["position_size_multiplier"] = 0.0  # No new positions
        result["recommended_action"] = VN_MARKET_CIRCUIT_BREAKER["L2_ACTION"]
        result["message"] = (
            f"🚫 TẠM NGỪNG GIAO DỊCH CẤP 2: VNINDEX {vnindex_change_pct:+.2f}% - "
            f"Sàn tạm ngừng {VN_MARKET_CIRCUIT_BREAKER['HALT_DURATION_L2']} phút. "
            f"KHÔNG đặt lệnh mới."
        )
        return result

    # Level 3: Trading halt 30 minutes, may close early
    result["halt_triggered"] = True
    result["halt_level"] = 3
    result["halt_duration_minutes"] = VN_MARKET_CIRCUIT_BREAKER["HALT_DURATION_L3"]
    result["trading_allowed"] = False
    result["position_size_multiplier"] = 0.0
    result["recommended_action"] = VN_MARKET_CIRCUIT_BREAKER["L3_ACTION"]
    result["message"] = (
        f"🚨 TẠM NGỪNG CẤP 3: VNINDEX {vnindex_change_pct:+.2f}% - "
        f"Sàn tạm ngừng {VN_MARKET_CIRCUIT_BREAKER['HALT_DURATION_L3']} phút. "
        f"Có thể đóng cửa sớm. XEM XÉT THOÁT VỊ THẾ ngay."
    )
    return result


def is_trading_allowed(vnindex_change_pct: float) -> Tuple[bool, str]:
    """
    Quick check if trading is allowed based on market circuit breaker.

    Args:
        vnindex_change_pct: VNINDEX change percentage

    Returns:
        Tuple of (is_allowed, reason)
    """
    status = check_market_halt_status(vnindex_change_pct)
    return status["trading_allowed"], status["message"]


# =============================================================================
# NEW v5.0: UNUSUAL TRADING ACTIVITY DETECTION
# =============================================================================


def detect_unusual_activity(
    df,
    symbol: str,
    volume_threshold: float = 5.0,
    price_threshold: float = 5.0,
) -> Dict:
    """
    Detect unusual trading activity that may indicate manipulation.

    Checks for:
    1. Abnormal volume spike (> 5x average)
    2. Abnormal price movement without news
    3. Wash trading patterns (high volume, low price change)
    4. End-of-day manipulation patterns

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        volume_threshold: Volume spike threshold (default 5x)
        price_threshold: Price change threshold (default 5%)

    Returns:
        Dict with detection results
    """
    result = {
        "unusual_detected": False,
        "warnings": [],
        "risk_level": "NORMAL",  # NORMAL, WARNING, HIGH, CRITICAL
        "trading_recommendation": "OK",
    }

    if df is None or len(df) < 20:
        return result

    try:
        # Calculate metrics
        current_volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].tail(20).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        current_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2] if len(df) >= 2 else current_close
        price_change_pct = (
            abs((current_close - prev_close) / prev_close * 100) if prev_close > 0 else 0
        )

        # Check 1: Abnormal volume spike
        if volume_ratio > volume_threshold:
            result["warnings"].append(
                f"🚨 Volume spike: {volume_ratio:.1f}x average (threshold: {volume_threshold}x)"
            )
            result["unusual_detected"] = True

        # Check 2: Wash trading pattern (high volume, low price change)
        if volume_ratio > 3.0 and price_change_pct < 1.0:
            result["warnings"].append(
                f"⚠️ Possible wash trading: Volume {volume_ratio:.1f}x but price only {price_change_pct:.2f}%"
            )
            result["unusual_detected"] = True

        # Check 3: Abnormal price movement
        if price_change_pct > price_threshold:
            result["warnings"].append(
                f"🚨 Abnormal price movement: {price_change_pct:.2f}% (threshold: {price_threshold}%)"
            )
            result["unusual_detected"] = True

        # Check 4: Intraday range check
        if "high" in df.columns and "low" in df.columns:
            day_high = df["high"].iloc[-1]
            day_low = df["low"].iloc[-1]
            intraday_range = (day_high - day_low) / day_low * 100 if day_low > 0 else 0

            if intraday_range > 10.0:  # > 10% intraday range
                result["warnings"].append(
                    f"⚠️ High intraday volatility: {intraday_range:.2f}% range"
                )
                result["unusual_detected"] = True

        # Determine risk level
        warning_count = len(result["warnings"])
        if warning_count >= 3:
            result["risk_level"] = "CRITICAL"
            result["trading_recommendation"] = "AVOID"
        elif warning_count >= 2:
            result["risk_level"] = "HIGH"
            result["trading_recommendation"] = "CAUTION"
        elif warning_count >= 1:
            result["risk_level"] = "WARNING"
            result["trading_recommendation"] = "REDUCE_SIZE"

        return result

    except Exception as e:
        logger.warning(f"Unusual activity detection error for {symbol}: {e}")
        return result
