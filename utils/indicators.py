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
        market_regime: Optional[str] = None,  # NEW: Market regime for adaptive stops
        beta: Optional[float] = None,  # NEW: Stock beta for volatility adjustment
    ) -> Tuple[float, str]:
        """
        Calculate stop loss with robust validation and adaptive adjustments.

        IMPROVED v5.0: Adaptive stop loss based on market regime and beta.

        Args:
            entry_price: Entry price
            atr: Average True Range
            support_level: Support level (optional)
            atr_multiplier: ATR multiplier for stop
            min_stop_pct: Minimum stop loss percentage
            max_stop_pct: Maximum stop loss percentage
            market_regime: Market regime (BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY)
            beta: Stock beta relative to market (optional)

        Returns:
            (stop_loss, reason) tuple
        """
        # Validate inputs
        if entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {entry_price}")

        if atr <= 0:
            raise ValueError(f"Invalid ATR: {atr}")

        # NEW v5.0: Adaptive ATR multiplier based on market regime
        adaptive_multiplier = atr_multiplier
        regime_reason = ""

        if market_regime:
            if market_regime == "BULL":
                # Bull market: can use wider stops (let winners run)
                adaptive_multiplier = atr_multiplier * 1.1  # 10% wider
                regime_reason = " (Bull: wider stop)"
            elif market_regime == "BEAR":
                # Bear market: use tighter stops (quick exits)
                adaptive_multiplier = atr_multiplier * 0.8  # 20% tighter
                regime_reason = " (Bear: tighter stop)"
            elif market_regime == "HIGH_VOLATILITY":
                # High volatility: wider stops to avoid whipsaws
                adaptive_multiplier = atr_multiplier * 1.25  # 25% wider
                regime_reason = " (High vol: wider stop)"
            elif market_regime == "SIDEWAYS":
                # Sideways: standard stops
                adaptive_multiplier = atr_multiplier
                regime_reason = " (Sideways: standard)"

        # NEW v5.0: Beta-adjusted stops
        beta_reason = ""
        if beta is not None and beta > 0:
            if beta > 1.3:  # High beta - more volatile
                adaptive_multiplier *= 1.15  # 15% wider
                beta_reason = f" (Beta {beta:.1f}: wider)"
            elif beta < 0.7:  # Low beta - less volatile
                adaptive_multiplier *= 0.85  # 15% tighter
                beta_reason = f" (Beta {beta:.1f}: tighter)"

        # Calculate ATR-based stop with adaptive multiplier
        atr_stop = entry_price - (atr * adaptive_multiplier)

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

        # Append regime and beta info to reason
        reason += regime_reason + beta_reason

        # NEW v5.0: Adaptive min/max stops based on regime
        effective_min_stop_pct = min_stop_pct
        effective_max_stop_pct = max_stop_pct

        if market_regime == "BEAR":
            effective_min_stop_pct = min_stop_pct * 0.8  # Tighter min in bear
            effective_max_stop_pct = max_stop_pct * 0.8  # Tighter max in bear
        elif market_regime == "HIGH_VOLATILITY":
            effective_min_stop_pct = min_stop_pct * 1.2  # Wider min in high vol
            effective_max_stop_pct = max_stop_pct * 1.3  # Wider max in high vol

        # Enforce minimum stop distance
        min_stop = entry_price * (1 - effective_min_stop_pct)
        if stop_loss > min_stop:
            logger.warning(
                f"Stop loss {stop_loss:.0f} too close to entry {entry_price:.0f}, "
                f"using minimum {effective_min_stop_pct*100:.1f}%"
            )
            stop_loss = min_stop
            reason = f"Minimum {effective_min_stop_pct*100:.1f}% stop"

        # Enforce maximum stop distance
        max_stop = entry_price * (1 - effective_max_stop_pct)
        if stop_loss < max_stop:
            logger.warning(
                f"Stop loss {stop_loss:.0f} too far from entry {entry_price:.0f}, "
                f"using maximum {effective_max_stop_pct*100:.1f}%"
            )
            stop_loss = max_stop
            reason = f"Maximum {effective_max_stop_pct*100:.1f}% stop"

        # Final validation
        if stop_loss <= 0:
            raise ValueError(f"Calculated stop loss <= 0: {stop_loss}")

        if stop_loss >= entry_price:
            raise ValueError(f"Stop loss {stop_loss:.0f} >= entry_price {entry_price:.0f}")

        return float(stop_loss), reason

    @staticmethod
    def calculate_adaptive_trailing_stop(
        entry_price: float,
        current_price: float,
        highest_price: float,
        atr: float,
        profit_pct: float,
        market_regime: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Calculate adaptive trailing stop based on profit level and market regime.

        Trailing stop logic:
        - Base trailing: ATR-based distance
        - Profit < 5%: No trailing (use initial stop)
        - Profit 5-10%: Trail at 3x ATR
        - Profit 10-20%: Trail at 2x ATR
        - Profit > 20%: Trail at 1.5x ATR (tight)

        Market regime adjustments:
        - BULL: Looser trailing to capture more upside
        - BEAR: Tighter trailing to protect gains
        - HIGH_VOL: Wider trailing to avoid whipsaws

        Args:
            entry_price: Original entry price
            current_price: Current price
            highest_price: Highest price since entry
            atr: Current ATR
            profit_pct: Current profit percentage (0.10 = 10%)
            market_regime: Market regime for adaptive adjustment

        Returns:
            (trailing_stop, reason) tuple
        """
        # Base trailing distance by profit level
        if profit_pct < 0.05:
            # Not enough profit for trailing
            return entry_price * 0.95, "No trailing (profit < 5%)"

        elif profit_pct < 0.10:
            # Small profit: loose trailing
            atr_mult = 3.0
            reason = "Trailing 3x ATR (5-10% profit)"

        elif profit_pct < 0.20:
            # Moderate profit: medium trailing
            atr_mult = 2.0
            reason = "Trailing 2x ATR (10-20% profit)"

        else:
            # Large profit: tight trailing
            atr_mult = 1.5
            reason = "Trailing 1.5x ATR (>20% profit)"

        # Regime adjustment
        if market_regime:
            if market_regime == "BULL":
                atr_mult *= 1.2  # Looser in bull
                reason += " [Bull: wider]"
            elif market_regime == "BEAR":
                atr_mult *= 0.8  # Tighter in bear
                reason += " [Bear: tighter]"
            elif market_regime == "HIGH_VOLATILITY":
                atr_mult *= 1.3  # Wider in high vol
                reason += " [High vol: wider]"

        # Calculate trailing stop from highest price
        trailing_stop = highest_price - (atr * atr_mult)

        # Ensure stop is above entry (protect initial capital)
        breakeven_stop = entry_price * 1.01  # Slightly above entry
        if profit_pct >= 0.10:
            trailing_stop = max(trailing_stop, breakeven_stop)
            if trailing_stop == breakeven_stop:
                reason += " [Min: breakeven]"

        # Ensure stop is below current price
        if trailing_stop >= current_price:
            trailing_stop = current_price * 0.98  # 2% below current
            reason = "Safety stop (trailing too close)"

        return float(trailing_stop), reason

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
