"""
Indicator Utilities - Centralized indicator calculations
Eliminates duplicate code and ensures consistency
"""

import logging
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class IndicatorUtils:
    """Utility class for indicator calculations"""

    @staticmethod
    def get_atr(df: pd.DataFrame, default_pct: float = 0.02) -> float:
        """
        Get ATR with robust fallback

        Args:
            df: DataFrame with OHLCV data
            default_pct: Default ATR as percentage of price

        Returns:
            ATR value
        """
        try:
            if "atr" in df.columns and not df["atr"].isnull().all():
                from utils.dataframe_utils import safe_get_latest

                atr = safe_get_latest(df, "atr", 0)
                if pd.notna(atr) and atr > 0:
                    return float(atr)

            # Fallback: calculate from price
            from utils.dataframe_utils import safe_get_latest

            price = safe_get_latest(df, "close", 100000)
            return float(price * default_pct)

        except Exception as e:
            logger.warning(f"Error getting ATR: {e}")
            try:
                if len(df) > 0 and "close" in df.columns:
                    price = safe_get_latest(df, "close", 100000)
                    return float(price * default_pct)
                else:
                    return float(100_000 * default_pct)  # Safe fallback
            except Exception:
                return float(100_000 * default_pct)  # Ultimate fallback

    @staticmethod
    def get_rsi(df: pd.DataFrame, default: float = 50.0) -> float:
        """Get RSI with fallback"""
        try:
            if "rsi" in df.columns and not df["rsi"].isnull().all():
                from utils.dataframe_utils import safe_get_latest

                rsi = safe_get_latest(df, "rsi", 50)
                if pd.notna(rsi) and 0 <= rsi <= 100:
                    return float(rsi)
            return default
        except Exception as e:
            logger.warning(f"Error getting RSI: {e}")
            return default

    @staticmethod
    def get_support_resistance(
        df: pd.DataFrame, lookback: int = 20
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate support and resistance levels

        Returns:
            (support, resistance) tuple
        """
        try:
            if len(df) < lookback:
                return None, None

            from utils.dataframe_utils import safe_rolling_operation

            support = safe_rolling_operation(df, "low", lookback, "min")
            resistance = safe_rolling_operation(df, "high", lookback, "max")

            # Validate
            if pd.notna(support) and support > 0:
                support = float(support)
            else:
                support = None

            if pd.notna(resistance) and resistance > 0:
                resistance = float(resistance)
            else:
                resistance = None

            return support, resistance

        except Exception as e:
            logger.warning(f"Error calculating S/R: {e}")
            return None, None


class StopLossCalculator:
    """
    Robust stop loss calculation with comprehensive validation
    """

    @staticmethod
    def calculate_stop_loss(
        entry_price: float,
        atr: float,
        support_level: Optional[float] = None,
        atr_multiplier: float = 2.0,
        min_stop_pct: float = 0.03,  # Minimum 3% stop
        max_stop_pct: float = 0.10,  # Maximum 10% stop
    ) -> Tuple[float, str]:
        """
        Calculate stop loss with robust validation

        Args:
            entry_price: Entry price
            atr: Average True Range
            support_level: Support level (optional)
            atr_multiplier: ATR multiplier for stop
            min_stop_pct: Minimum stop loss percentage
            max_stop_pct: Maximum stop loss percentage

        Returns:
            (stop_loss, reason) tuple
        """
        # Validate inputs
        if entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {entry_price}")

        if atr <= 0:
            raise ValueError(f"Invalid ATR: {atr}")

        # Calculate ATR-based stop
        atr_stop = entry_price - (atr * atr_multiplier)

        # Calculate ATR as percentage of price (for dynamic minimum)
        atr_pct = atr / entry_price if entry_price > 0 else 0

        # Validate support level
        valid_support = None
        if support_level is not None:
            try:
                if not pd.isna(support_level) and 0 < support_level < entry_price:
                    valid_support = float(support_level)
            except (TypeError, ValueError):
                pass

        # Choose stop loss
        if valid_support is not None:
            # Use tighter of support or ATR-based
            stop_loss = max(valid_support, atr_stop)
            reason = "Support-based" if stop_loss == valid_support else "ATR-based"
        else:
            stop_loss = atr_stop
            reason = "ATR-based"

        # Calculate dynamic minimum based on ATR percentage
        # Formula: min(max(2.0x ATR%, 1.5%), min_stop_pct)
        # This ensures:
        # - Mã ít biến động (ATR < 0.75%): dùng 1.5% minimum
        # - Mã biến động bình thường (ATR ~1-1.5%): dùng ~2-3% (2x ATR)
        # - Mã biến động cao (ATR > 1.5%): dùng min_stop_pct (3%) as cap
        dynamic_min_pct = min(max(2.0 * atr_pct, 0.015), min_stop_pct)

        min_stop = entry_price * (1 - dynamic_min_pct)

        # Only warn if we need to adjust significantly (>= 0.5% difference)
        # Use >= 0.0049 to account for floating point precision (0.015 - 0.01 ≈ 0.005)
        original_stop_pct = (entry_price - stop_loss) / entry_price if entry_price > 0 else 0
        if stop_loss > min_stop and (dynamic_min_pct - original_stop_pct) >= 0.0049:
            logger.warning(
                f"Stop loss {stop_loss:.0f} ({original_stop_pct*100:.2f}%) too close to entry {entry_price:.0f}, "
                f"using dynamic minimum {dynamic_min_pct*100:.2f}% (ATR: {atr_pct*100:.2f}%)"
            )
            stop_loss = min_stop
            reason = f"Dynamic minimum {dynamic_min_pct*100:.2f}% stop"

        # Enforce maximum stop distance
        max_stop = entry_price * (1 - max_stop_pct)
        if stop_loss < max_stop:
            logger.warning(
                f"Stop loss {stop_loss:.0f} too far from entry {entry_price:.0f}, "
                f"using maximum {max_stop_pct*100}%"
            )
            stop_loss = max_stop
            reason = f"Maximum {max_stop_pct*100}% stop"

        # Final validation
        if stop_loss <= 0:
            raise ValueError(f"Calculated stop loss <= 0: {stop_loss}")

        if stop_loss >= entry_price:
            raise ValueError(f"Stop loss {stop_loss:.0f} >= entry_price {entry_price:.0f}")

        return float(stop_loss), reason

    @staticmethod
    def calculate_take_profit_targets(
        entry_price: float, atr: float, risk_reward_ratios: list = [1.5, 3.0, 5.0]
    ) -> list:
        """
        Calculate take profit targets

        Args:
            entry_price: Entry price
            atr: Average True Range
            risk_reward_ratios: List of R:R ratios

        Returns:
            List of take profit prices
        """
        if entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {entry_price}")

        if atr <= 0:
            raise ValueError(f"Invalid ATR: {atr}")

        targets = []
        for rr in risk_reward_ratios:
            tp = entry_price + (atr * rr)
            targets.append(float(tp))

        return targets


# Singleton instances for convenience
indicator_utils = IndicatorUtils()
stop_loss_calculator = StopLossCalculator()


if __name__ == "__main__":
    # Test stop loss calculator
    print("Testing StopLossCalculator...")

    # Test 1: Normal case
    sl, reason = stop_loss_calculator.calculate_stop_loss(
        entry_price=100_000, atr=2_000, support_level=96_000
    )
    print(f"Test 1: SL={sl:,.0f}, Reason={reason}")
    assert 90_000 <= sl < 100_000

    # Test 2: No support
    sl, reason = stop_loss_calculator.calculate_stop_loss(
        entry_price=100_000, atr=2_000, support_level=None
    )
    print(f"Test 2: SL={sl:,.0f}, Reason={reason}")
    assert 90_000 <= sl < 100_000

    # Test 3: Invalid support (> entry)
    sl, reason = stop_loss_calculator.calculate_stop_loss(
        entry_price=100_000, atr=2_000, support_level=105_000  # Invalid
    )
    print(f"Test 3: SL={sl:,.0f}, Reason={reason}")
    assert 90_000 <= sl < 100_000

    # Test 4: Edge case - very small ATR
    sl, reason = stop_loss_calculator.calculate_stop_loss(
        entry_price=100_000, atr=500, support_level=None  # Very small
    )
    print(f"Test 4: SL={sl:,.0f}, Reason={reason}")
    assert sl >= 97_000  # Should enforce minimum 3%

    print("\n✅ All tests passed!")
