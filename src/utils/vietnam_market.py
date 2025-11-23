"""
Vietnam Market-Specific Utilities
Handles Vietnam stock market rules and constraints
"""

import logging
from datetime import datetime, time
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class VietnamMarketValidator:
    """
    Validator for Vietnam market-specific rules

    Handles:
    - Price floor/ceiling limits (±7%)
    - T+2 settlement
    - Trading session boundaries
    - Liquidity requirements
    """

    def __init__(self, config=None):
        """
        Initialize validator with config

        Args:
            config: TradingConfig instance (optional)
        """
        if config:
            self.daily_price_limit_pct = config.vn_daily_price_limit_pct
            self.check_price_limits = config.vn_check_price_limits
            self.avoid_floor_ceiling_pct = config.vn_avoid_floor_ceiling_pct
            self.settlement_days = config.vn_settlement_days
            self.reserve_t2_cash = config.vn_reserve_t2_cash
            self.t2_cash_buffer_pct = config.vn_t2_cash_buffer_pct
            self.min_daily_value = config.vn_min_daily_value
            self.max_position_pct_of_volume = config.vn_max_position_pct_of_volume
            self.avoid_session_boundaries = config.vn_avoid_session_boundaries
            self.session_boundary_minutes = config.vn_session_boundary_minutes
            self.session_am_end = config.vn_trading_session_am_end
            self.session_pm_start = config.vn_trading_session_pm_start
        else:
            # Default values
            self.daily_price_limit_pct = 7.0
            self.check_price_limits = True
            self.avoid_floor_ceiling_pct = 2.0
            self.settlement_days = 2
            self.reserve_t2_cash = True
            self.t2_cash_buffer_pct = 0.10
            self.min_daily_value = 2_000_000_000
            self.max_position_pct_of_volume = 0.05
            self.avoid_session_boundaries = True
            self.session_boundary_minutes = 5
            self.session_am_end = "11:30"
            self.session_pm_start = "13:00"

    def check_price_floor_ceiling(
        self,
        current_price: float,
        reference_price: float,
        symbol: str = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if price is near floor or ceiling limit

        Vietnam stocks have ±7% daily price limit from reference price
        (usually previous day's closing price)

        Args:
            current_price: Current price
            reference_price: Reference price (yesterday's close)
            symbol: Stock symbol for logging

        Returns:
            (is_safe, warning_message)
            - is_safe: True if price is NOT near floor/ceiling
            - warning_message: Warning if near limit, None otherwise
        """
        if not self.check_price_limits:
            return (True, None)

        if reference_price <= 0:
            logger.warning(f"[{symbol}] Invalid reference price: {reference_price}")
            return (True, None)

        # Calculate floor and ceiling
        floor = reference_price * (1 - self.daily_price_limit_pct / 100)
        ceiling = reference_price * (1 + self.daily_price_limit_pct / 100)

        # Calculate distance from floor/ceiling
        distance_from_floor = ((current_price - floor) / floor) * 100
        distance_from_ceiling = ((ceiling - current_price) / current_price) * 100

        # Check if too close to floor
        if distance_from_floor < self.avoid_floor_ceiling_pct:
            warning = (
                f"Price near FLOOR: {current_price:,.0f} within {distance_from_floor:.1f}% "
                f"of floor {floor:,.0f}. Avoid entry (high risk of further drop)."
            )
            logger.warning(f"[{symbol}] {warning}")
            return (False, warning)

        # Check if too close to ceiling
        if distance_from_ceiling < self.avoid_floor_ceiling_pct:
            warning = (
                f"Price near CEILING: {current_price:,.0f} within {distance_from_ceiling:.1f}% "
                f"of ceiling {ceiling:,.0f}. Avoid entry (limited upside, high resistance)."
            )
            logger.warning(f"[{symbol}] {warning}")
            return (False, warning)

        # Safe - not near limits
        return (True, None)

    def check_trading_session_timing(
        self,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if current time is safe for trading (avoid session boundaries)

        Vietnam market sessions:
        - Morning: 09:00 - 11:30
        - Afternoon: 13:00 - 14:45

        Avoid trading within N minutes of session boundaries to prevent:
        - Order execution issues
        - Price volatility near open/close
        - Liquidity dry-ups

        Args:
            current_time: Current datetime (default: now)

        Returns:
            (is_safe, warning_message)
        """
        if not self.avoid_session_boundaries:
            return (True, None)

        if current_time is None:
            current_time = datetime.now()

        current_time_only = current_time.time()

        # Parse session boundaries
        am_end = time.fromisoformat(self.session_am_end)
        pm_start = time.fromisoformat(self.session_pm_start)

        # Calculate boundary windows
        am_end_warning = (am_end.hour, am_end.minute - self.session_boundary_minutes)
        pm_start_warning = (pm_start.hour, pm_start.minute + self.session_boundary_minutes)

        # Check if near AM session end (e.g., 11:25-11:30 if boundary_minutes=5)
        if time(am_end_warning[0], am_end_warning[1]) <= current_time_only <= am_end:
            warning = (
                f"Near morning session end ({self.session_am_end}). "
                f"Avoid trading within {self.session_boundary_minutes} minutes of session boundary."
            )
            logger.warning(warning)
            return (False, warning)

        # Check if near PM session start (e.g., 13:00-13:05 if boundary_minutes=5)
        if pm_start <= current_time_only <= time(pm_start_warning[0], pm_start_warning[1]):
            warning = (
                f"Near afternoon session start ({self.session_pm_start}). "
                f"Avoid trading within {self.session_boundary_minutes} minutes of session boundary."
            )
            logger.warning(warning)
            return (False, warning)

        return (True, None)

    def calculate_t2_cash_requirement(
        self,
        pending_settlements: Dict[str, float],
        new_trade_value: float = 0
    ) -> Tuple[float, float]:
        """
        Calculate T+2 cash requirement

        Vietnam market uses T+2 settlement:
        - Day T: Trade executed
        - Day T+2: Cash settlement

        Must reserve cash for pending T+2 obligations

        Args:
            pending_settlements: Dict of {date: amount} for pending T+2 settlements
            new_trade_value: Value of new trade to add

        Returns:
            (total_required, buffer_amount)
            - total_required: Total cash needed for T+2
            - buffer_amount: Extra buffer (10%)
        """
        if not self.reserve_t2_cash:
            return (0, 0)

        # Sum all pending settlements
        total_pending = sum(pending_settlements.values())

        # Add new trade
        total_required = total_pending + new_trade_value

        # Calculate buffer
        buffer = total_required * self.t2_cash_buffer_pct

        return (total_required, buffer)

    def validate_position_size_vs_volume(
        self,
        position_shares: int,
        avg_daily_volume: float,
        symbol: str = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that position size is not too large relative to daily volume

        Large positions relative to volume can cause:
        - Slippage on entry/exit
        - Market impact
        - Difficulty exiting in emergency

        Args:
            position_shares: Planned position size in shares
            avg_daily_volume: Average daily volume in shares
            symbol: Stock symbol for logging

        Returns:
            (is_safe, warning_message)
        """
        if avg_daily_volume <= 0:
            logger.warning(f"[{symbol}] Invalid avg daily volume: {avg_daily_volume}")
            return (True, None)

        # Calculate position as % of daily volume
        position_pct = (position_shares / avg_daily_volume) * 100

        # Check against limit
        max_pct = self.max_position_pct_of_volume * 100

        if position_pct > max_pct:
            warning = (
                f"Position too large: {position_shares:,} shares "
                f"= {position_pct:.1f}% of daily volume {avg_daily_volume:,.0f}. "
                f"Max allowed: {max_pct:.1f}%. Risk of slippage/impact."
            )
            logger.warning(f"[{symbol}] {warning}")
            return (False, warning)

        return (True, None)

    def check_liquidity_requirements(
        self,
        df: pd.DataFrame,
        symbol: str = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if stock meets Vietnam market liquidity requirements

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol

        Returns:
            (is_liquid, warning_message)
        """
        if len(df) < 20:
            warning = "Insufficient data to check liquidity"
            logger.warning(f"[{symbol}] {warning}")
            return (False, warning)

        # Calculate average daily value (last 20 days)
        df_recent = df.tail(20).copy()
        df_recent['daily_value'] = df_recent['close'] * df_recent['volume']
        avg_daily_value = df_recent['daily_value'].mean()

        # Check minimum daily value
        if avg_daily_value < self.min_daily_value:
            warning = (
                f"Insufficient liquidity: avg daily value {avg_daily_value/1_000_000_000:.2f}B VND "
                f"< minimum {self.min_daily_value/1_000_000_000:.2f}B VND"
            )
            logger.warning(f"[{symbol}] {warning}")
            return (False, warning)

        return (True, None)


# Global singleton
_validator = None


def get_vietnam_market_validator() -> VietnamMarketValidator:
    """Get singleton instance of Vietnam market validator"""
    global _validator
    if _validator is None:
        from src.config.trading_config import get_config
        config = get_config()
        _validator = VietnamMarketValidator(config.trading)
    return _validator


# Convenience functions
def check_price_limits(current_price: float, reference_price: float, symbol: str = None) -> Tuple[bool, Optional[str]]:
    """Check if price is safe (not near floor/ceiling)"""
    validator = get_vietnam_market_validator()
    return validator.check_price_floor_ceiling(current_price, reference_price, symbol)


def check_trading_session(current_time: Optional[datetime] = None) -> Tuple[bool, Optional[str]]:
    """Check if current time is safe for trading"""
    validator = get_vietnam_market_validator()
    return validator.check_trading_session_timing(current_time)


def calculate_t2_requirement(pending: Dict[str, float], new_trade: float = 0) -> Tuple[float, float]:
    """Calculate T+2 cash requirement"""
    validator = get_vietnam_market_validator()
    return validator.calculate_t2_cash_requirement(pending, new_trade)


def validate_position_vs_volume(shares: int, volume: float, symbol: str = None) -> Tuple[bool, Optional[str]]:
    """Validate position size vs daily volume"""
    validator = get_vietnam_market_validator()
    return validator.validate_position_size_vs_volume(shares, volume, symbol)


def check_liquidity(df: pd.DataFrame, symbol: str = None) -> Tuple[bool, Optional[str]]:
    """Check liquidity requirements"""
    validator = get_vietnam_market_validator()
    return validator.check_liquidity_requirements(df, symbol)
