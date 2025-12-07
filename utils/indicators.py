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
                atr = df["atr"].iloc[-1]
                if pd.notna(atr) and atr > 0:
                    return float(atr)

            # Fallback: calculate from price
            price = df["close"].iloc[-1]
            return float(price * default_pct)

        except Exception as e:
            logger.warning(f"Error getting ATR: {e}")
            try:
                if len(df) > 0 and "close" in df.columns:
                    return float(df["close"].iloc[-1] * default_pct)
                else:
                    return float(100_000 * default_pct)  # Safe fallback
            except Exception:
                return float(100_000 * default_pct)  # Ultimate fallback

    @staticmethod
    def get_rsi(df: pd.DataFrame, default: float = 50.0) -> float:
        """Get RSI with fallback"""
        try:
            if "rsi" in df.columns and not df["rsi"].isnull().all():
                rsi = df["rsi"].iloc[-1]
                if pd.notna(rsi) and 0 <= rsi <= 100:
                    return float(rsi)
            return default
        except Exception:
            logger.warning("Error getting RSI")
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

            support = df["low"].rolling(lookback).min().iloc[-1]
            resistance = df["high"].rolling(lookback).max().iloc[-1]

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

        except Exception:
            logger.warning("Error calculating S/R")
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

        # Enforce minimum stop distance
        min_stop = entry_price * (1 - min_stop_pct)
        if stop_loss > min_stop:
            logger.warning(
                f"Stop loss {stop_loss:.0f} too close to entry {entry_price:.0f}, "
                f"using minimum {min_stop_pct*100}%"
            )
            stop_loss = min_stop
            reason = f"Minimum {min_stop_pct*100}% stop"

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
        entry_price: float,
        atr: float,
        risk_reward_ratios: list = [1.5, 3.0, 5.0],
        min_tp_percentages: list = [0.06, 0.10, 0.15],
    ) -> list:
        """
        Calculate take profit targets using HYBRID approach.

        Combines ATR-based and percentage-based methods:
        - ATR-based: TP = entry + (ATR × R:R ratio) - adaptive to volatility
        - Percentage-based: TP = entry × (1 + min_tp_pct) - ensures minimum R:R

        Final TP = max(ATR-based, Percentage-based)

        This ensures:
        - TP adapts to stock volatility (ATR)
        - Minimum R:R ratio is always maintained (min_tp_percentages)
        - For VN market with ±7% daily limit, min TP1=6%, TP2=10%, TP3=15%

        Args:
            entry_price: Entry price
            atr: Average True Range
            risk_reward_ratios: List of R:R ratios for ATR-based calculation
            min_tp_percentages: Minimum TP percentages [TP1, TP2, TP3]
                Default: [0.06, 0.10, 0.15] = 6%, 10%, 15%

        Returns:
            List of take profit prices
        """
        if entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {entry_price}")

        if atr <= 0:
            raise ValueError(f"Invalid ATR: {atr}")

        targets = []
        for i, rr in enumerate(risk_reward_ratios):
            # ATR-based TP
            tp_atr = entry_price + (atr * rr)

            # Percentage-based TP (minimum)
            min_pct = min_tp_percentages[i] if i < len(min_tp_percentages) else 0.15
            tp_pct = entry_price * (1 + min_pct)

            # Hybrid: take the higher of the two
            tp = max(tp_atr, tp_pct)
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
