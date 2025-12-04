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
from datetime import datetime, time, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd

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
# VALIDATOR CLASS
# =============================================================================


class VietnamMarketValidator:
    """
    Validator class for Vietnam market rules.
    Wraps standalone functions for easier integration.
    """

    def __init__(
        self,
        min_liquidity_value: float = 2_000_000_000,
        max_position_pct_of_volume: float = 0.05,
    ):
        """
        Initialize validator.

        Args:
            min_liquidity_value: Minimum daily liquidity in VND (default: 2B)
            max_position_pct_of_volume: Max position as % of avg volume (default: 5%)
        """
        self.min_liquidity_value = min_liquidity_value
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


def calculate_settlement_date(trade_date: datetime, is_buy: bool = True) -> datetime:
    """
    Calculate settlement date for Vietnam T+2 cycle.

    Vietnam uses T+2 settlement:
    - Buy: Pay on T+2
    - Sell: Receive proceeds on T+2

    Note: Weekends and holidays are skipped.

    Args:
        trade_date: Date of trade execution
        is_buy: True for buy orders, False for sell orders

    Returns:
        Settlement date
    """
    from pandas.tseries.offsets import BDay

    # T+2 business days
    settlement = trade_date + BDay(2)
    return settlement.to_pydatetime() if hasattr(settlement, "to_pydatetime") else settlement


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
