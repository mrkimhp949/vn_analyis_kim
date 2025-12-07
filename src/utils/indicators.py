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


class BetaCalculator:
    """
    Calculate stock beta relative to market index (VNINDEX).

    Beta measures volatility relative to market:
    - Beta > 1: More volatile than market
    - Beta < 1: Less volatile than market
    - Beta = 1: Same volatility as market
    """

    @staticmethod
    def calculate_beta(
        stock_df: pd.DataFrame,
        market_df: pd.DataFrame,
        lookback: int = 60,
    ) -> float:
        """
        Calculate beta of stock relative to market.

        Args:
            stock_df: Stock OHLCV DataFrame
            market_df: Market index (VNINDEX) DataFrame
            lookback: Number of days for calculation

        Returns:
            Beta value (default 1.0 if calculation fails)
        """
        try:
            if stock_df is None or market_df is None:
                return 1.0

            if len(stock_df) < lookback or len(market_df) < lookback:
                return 1.0

            # Calculate returns
            stock_returns = stock_df["close"].pct_change().tail(lookback).dropna()
            market_returns = market_df["close"].pct_change().tail(lookback).dropna()

            # Align data
            min_len = min(len(stock_returns), len(market_returns))
            if min_len < 20:
                return 1.0

            stock_returns = stock_returns.tail(min_len)
            market_returns = market_returns.tail(min_len)

            # Calculate beta = Cov(stock, market) / Var(market)
            covariance = stock_returns.cov(market_returns)
            market_variance = market_returns.var()

            if market_variance <= 0:
                return 1.0

            beta = covariance / market_variance

            # Clamp to reasonable range
            beta = max(0.3, min(2.5, beta))

            return float(beta)

        except Exception as e:
            logger.debug(f"Beta calculation failed: {e}")
            return 1.0

    @staticmethod
    def get_beta_adjusted_stop_loss(
        entry_price: float,
        beta: float,
        base_stop_pct: float = 0.06,
        high_beta_stop_pct: float = 0.08,
        low_beta_stop_pct: float = 0.05,
        high_beta_threshold: float = 1.2,
        low_beta_threshold: float = 0.8,
    ) -> Tuple[float, str]:
        """
        Calculate beta-adjusted stop loss.

        Higher beta stocks get wider stops to avoid premature exit.
        Lower beta stocks get tighter stops for better risk management.

        Args:
            entry_price: Entry price
            beta: Stock beta
            base_stop_pct: Base stop loss percentage
            high_beta_stop_pct: Stop for high beta stocks
            low_beta_stop_pct: Stop for low beta stocks
            high_beta_threshold: Beta threshold for wider stop
            low_beta_threshold: Beta threshold for tighter stop

        Returns:
            (stop_loss_price, reason)
        """
        if beta >= high_beta_threshold:
            stop_pct = high_beta_stop_pct
            reason = f"High beta ({beta:.2f}) - wider stop {stop_pct*100:.0f}%"
        elif beta <= low_beta_threshold:
            stop_pct = low_beta_stop_pct
            reason = f"Low beta ({beta:.2f}) - tighter stop {stop_pct*100:.0f}%"
        else:
            # Interpolate between low and high
            beta_range = high_beta_threshold - low_beta_threshold
            stop_range = high_beta_stop_pct - low_beta_stop_pct
            beta_position = (beta - low_beta_threshold) / beta_range
            stop_pct = low_beta_stop_pct + (stop_range * beta_position)
            reason = f"Normal beta ({beta:.2f}) - stop {stop_pct*100:.1f}%"

        stop_loss = entry_price * (1 - stop_pct)

        return float(stop_loss), reason


# Singleton instances for convenience
indicator_utils = IndicatorUtils()
stop_loss_calculator = StopLossCalculator()
beta_calculator = BetaCalculator()


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
