# -*- coding: utf-8 -*-
"""
Vietnam Market Utilities

Centralized utilities for Vietnam stock market specific rules:
- Lot size validation (100 shares)
- Tick size calculation (10/50/100 VND based on price)
- Exchange detection (HOSE/HNX/UPCOM)
- Price limit calculation
- ATO/ATC session detection

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from datetime import datetime, time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

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

# VN30 symbols (HOSE blue chips) - Updated 2024
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

# Trading session times
VN_SESSIONS = {
    "ATO": (time(9, 0), time(9, 15)),
    "MORNING": (time(9, 15), time(11, 30)),
    "LUNCH": (time(11, 30), time(13, 0)),
    "AFTERNOON": (time(13, 0), time(14, 30)),
    "ATC": (time(14, 30), time(14, 45)),
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
# TICK SIZE FUNCTIONS
# =============================================================================


def get_tick_size(price: float) -> int:
    """
    Get tick size based on price range (HOSE rules).

    Vietnam tick sizes:
    - Price < 10,000 VND: Tick = 10 VND
    - 10,000 <= Price < 50,000 VND: Tick = 50 VND
    - Price >= 50,000 VND: Tick = 100 VND

    Args:
        price: Stock price in VND

    Returns:
        Tick size in VND
    """
    if price < 10_000:
        return 10
    elif price < 50_000:
        return 50
    else:
        return 100


def round_to_tick(price: float, direction: str = "nearest") -> float:
    """
    Round price to valid tick size.

    Args:
        price: Price to round
        direction: "nearest", "up", or "down"

    Returns:
        Price rounded to valid tick
    """
    tick = get_tick_size(price)

    if direction == "up":
        return ((price + tick - 1) // tick) * tick
    elif direction == "down":
        return (price // tick) * tick
    else:  # nearest
        return round(price / tick) * tick


def validate_price(price: float) -> Tuple[bool, str]:
    """
    Validate that price is a valid tick.

    Args:
        price: Price to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if price <= 0:
        return False, "Price must be positive"

    tick = get_tick_size(price)
    if price % tick != 0:
        valid_price = round_to_tick(price)
        return False, f"Price {price:,.0f} not valid tick. Use {valid_price:,.0f}"

    return True, "Valid"


# =============================================================================
# EXCHANGE FUNCTIONS
# =============================================================================


def get_exchange(symbol: str) -> str:
    """
    Detect exchange from symbol.

    Logic:
    - VN30 symbols → HOSE
    - HNX30 symbols → HNX
    - 3-letter symbols → likely HOSE
    - Default → HOSE (most liquid)

    Note: For production, this should use a reference database.

    Args:
        symbol: Stock symbol

    Returns:
        Exchange name: "HOSE", "HNX", or "UPCOM"
    """
    symbol = symbol.upper().strip()

    if symbol in VN30_SYMBOLS:
        return "HOSE"
    if symbol in HNX30_SYMBOLS:
        return "HNX"

    # Heuristic: 3-letter symbols are typically HOSE
    if len(symbol) == 3:
        return "HOSE"

    # Default to HOSE
    return "HOSE"


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
