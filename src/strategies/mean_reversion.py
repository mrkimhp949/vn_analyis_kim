# -*- coding: utf-8 -*-
"""
Mean Reversion Strategy for Vietnam Market

Mean reversion strategy tailored for Vietnam market:
- RSI-based oversold/overbought detection
- Bollinger Band mean reversion signals
- Volume-weighted price deviation analysis
- Vietnam market-specific adjustments

Usage:
    from src.strategies.mean_reversion import (
        MeanReversionStrategy,
        get_mean_reversion_strategy,
    )
    
    strategy = get_mean_reversion_strategy()
    signal = strategy.generate_signal("VNM", df)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class SignalType(Enum):
    """Signal types."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class ReasonType(Enum):
    """Reason types for signals."""

    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    BB_LOWER = "bollinger_lower"
    BB_UPPER = "bollinger_upper"
    PRICE_DEVIATION = "price_deviation"
    VOLUME_SPIKE = "volume_spike"
    MA_CROSSOVER = "ma_crossover"
    VWAP_DEVIATION = "vwap_deviation"


# Vietnam market characteristics
VN_MARKET_CONFIG = {
    "price_limits": {
        "HOSE": 0.07,
        "HNX": 0.10,
        "UPCOM": 0.15,
    },
    "trading_hours": {
        "ato": (9, 0, 9, 15),
        "morning": (9, 15, 11, 30),
        "afternoon": (13, 0, 14, 30),
        "atc": (14, 30, 14, 45),
    },
    "high_volatility_threshold": 0.05,  # 5% daily move
    "low_liquidity_threshold": 50000,  # shares/day
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MeanReversionSignal:
    """Mean reversion trading signal."""

    symbol: str
    signal_type: SignalType
    confidence: float  # 0-1
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None

    # Signal details
    reasons: List[str] = None
    indicators: Dict[str, float] = None

    # Timestamps
    generated_at: datetime = None
    valid_until: datetime = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.indicators is None:
            self.indicators = {}
        if self.generated_at is None:
            self.generated_at = datetime.now()

    @property
    def is_buy(self) -> bool:
        return self.signal_type in (SignalType.STRONG_BUY, SignalType.BUY)

    @property
    def is_sell(self) -> bool:
        return self.signal_type in (SignalType.STRONG_SELL, SignalType.SELL)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "reasons": self.reasons,
            "indicators": self.indicators,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class IndicatorValues:
    """Container for calculated indicator values."""

    # RSI
    rsi: float = 50.0
    rsi_signal: str = "neutral"

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_position: float = 0.5  # 0 = at lower, 1 = at upper

    # Moving Averages
    sma_20: float = 0.0
    sma_50: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0

    # Volume
    volume_ratio: float = 1.0  # vs 20-day average
    vwap: float = 0.0
    vwap_deviation: float = 0.0

    # Price deviation
    price_deviation_20: float = 0.0  # % from 20-day MA
    price_deviation_50: float = 0.0  # % from 50-day MA

    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0

    # ATR
    atr: float = 0.0
    atr_pct: float = 0.0


# =============================================================================
# INDICATOR CALCULATOR
# =============================================================================


class TechnicalIndicatorCalculator:
    """Calculate technical indicators for mean reversion."""

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return vwap

    @staticmethod
    def calculate_stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()

        k = 100 * (df["close"] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()

        return k, d

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    @classmethod
    def calculate_all_indicators(cls, df: pd.DataFrame) -> IndicatorValues:
        """Calculate all indicators from OHLCV data."""
        if len(df) < 50:
            logger.warning("Insufficient data for indicators (need 50+ bars)")
            return IndicatorValues()

        close = df["close"]
        latest = df.iloc[-1]

        indicators = IndicatorValues()

        # RSI
        rsi_series = cls.calculate_rsi(close)
        indicators.rsi = rsi_series.iloc[-1]

        if indicators.rsi < 30:
            indicators.rsi_signal = "oversold"
        elif indicators.rsi > 70:
            indicators.rsi_signal = "overbought"
        else:
            indicators.rsi_signal = "neutral"

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = cls.calculate_bollinger_bands(close)
        indicators.bb_upper = bb_upper.iloc[-1]
        indicators.bb_middle = bb_middle.iloc[-1]
        indicators.bb_lower = bb_lower.iloc[-1]
        indicators.bb_width = (indicators.bb_upper - indicators.bb_lower) / indicators.bb_middle

        bb_range = indicators.bb_upper - indicators.bb_lower
        if bb_range > 0:
            indicators.bb_position = (latest["close"] - indicators.bb_lower) / bb_range

        # Moving Averages
        indicators.sma_20 = cls.calculate_sma(close, 20).iloc[-1]
        indicators.sma_50 = cls.calculate_sma(close, 50).iloc[-1]
        indicators.ema_12 = cls.calculate_ema(close, 12).iloc[-1]
        indicators.ema_26 = cls.calculate_ema(close, 26).iloc[-1]

        # Price deviation from MAs
        indicators.price_deviation_20 = (
            (latest["close"] - indicators.sma_20) / indicators.sma_20 * 100
        )
        indicators.price_deviation_50 = (
            (latest["close"] - indicators.sma_50) / indicators.sma_50 * 100
        )

        # Volume
        avg_volume = df["volume"].rolling(window=20).mean().iloc[-1]
        indicators.volume_ratio = latest["volume"] / avg_volume if avg_volume > 0 else 1.0

        # VWAP
        vwap = cls.calculate_vwap(df)
        indicators.vwap = vwap.iloc[-1]
        indicators.vwap_deviation = (latest["close"] - indicators.vwap) / indicators.vwap * 100

        # Stochastic
        stoch_k, stoch_d = cls.calculate_stochastic(df)
        indicators.stoch_k = stoch_k.iloc[-1]
        indicators.stoch_d = stoch_d.iloc[-1]

        # ATR
        atr = cls.calculate_atr(df)
        indicators.atr = atr.iloc[-1]
        indicators.atr_pct = indicators.atr / latest["close"] * 100

        return indicators


# =============================================================================
# MEAN REVERSION STRATEGY
# =============================================================================


class MeanReversionStrategy:
    """
    Mean Reversion Strategy for Vietnam Market

    Generates buy signals when:
    - RSI < 30 (oversold)
    - Price below lower Bollinger Band
    - Price significantly below moving averages
    - Volume spike indicating potential reversal

    Generates sell signals when:
    - RSI > 70 (overbought)
    - Price above upper Bollinger Band
    - Price significantly above moving averages

    Vietnam market adjustments:
    - Account for price limits (7%, 10%, 15%)
    - Higher volatility tolerance
    - Reduced signals during ATO/ATC
    """

    # Default parameters
    DEFAULT_PARAMS = {
        # RSI parameters
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "rsi_extreme_oversold": 20,
        "rsi_extreme_overbought": 80,
        # Bollinger Band parameters
        "bb_period": 20,
        "bb_std_dev": 2.0,
        # Price deviation thresholds
        "deviation_threshold": 5.0,  # % from MA
        "extreme_deviation_threshold": 8.0,
        # Volume parameters
        "volume_spike_threshold": 2.0,  # 2x average
        # Stop loss / Target
        "stop_loss_atr_multiplier": 2.0,
        "target_atr_multiplier": 3.0,
        # Confidence weights
        "weight_rsi": 0.25,
        "weight_bb": 0.25,
        "weight_deviation": 0.20,
        "weight_volume": 0.15,
        "weight_stochastic": 0.15,
    }

    def __init__(self, params: Optional[Dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._indicator_calculator = TechnicalIndicatorCalculator()
        self._lock = RLock()

        logger.info("📊 Mean Reversion Strategy initialized")

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        exchange: str = "HOSE",
    ) -> Optional[MeanReversionSignal]:
        """
        Generate mean reversion signal for a symbol.

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            exchange: Exchange (HOSE, HNX, UPCOM)

        Returns:
            MeanReversionSignal or None
        """
        if df is None or len(df) < 50:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        with self._lock:
            # Calculate indicators
            indicators = self._indicator_calculator.calculate_all_indicators(df)

            # Analyze for signals
            buy_score, buy_reasons = self._analyze_buy_conditions(indicators)
            sell_score, sell_reasons = self._analyze_sell_conditions(indicators)

            # Determine signal
            signal_type = SignalType.HOLD
            confidence = 0.0
            reasons = []

            if buy_score > sell_score and buy_score > 0.4:
                if buy_score > 0.7:
                    signal_type = SignalType.STRONG_BUY
                else:
                    signal_type = SignalType.BUY
                confidence = buy_score
                reasons = buy_reasons
            elif sell_score > buy_score and sell_score > 0.4:
                if sell_score > 0.7:
                    signal_type = SignalType.STRONG_SELL
                else:
                    signal_type = SignalType.SELL
                confidence = sell_score
                reasons = sell_reasons
            else:
                return None  # No clear signal

            # Apply Vietnam market adjustments
            confidence = self._apply_vn_market_adjustments(confidence, indicators, exchange)

            if confidence < 0.4:
                return None  # Below threshold after adjustments

            # Calculate entry, stop loss, target
            latest_price = df["close"].iloc[-1]
            entry_price = latest_price

            if signal_type in (SignalType.STRONG_BUY, SignalType.BUY):
                stop_loss = entry_price - (indicators.atr * self.params["stop_loss_atr_multiplier"])
                target_price = entry_price + (indicators.atr * self.params["target_atr_multiplier"])
            else:
                stop_loss = entry_price + (indicators.atr * self.params["stop_loss_atr_multiplier"])
                target_price = entry_price - (indicators.atr * self.params["target_atr_multiplier"])

            # Apply price limits
            price_limit = VN_MARKET_CONFIG["price_limits"].get(exchange, 0.07)
            reference = df["close"].iloc[-2]  # Previous close

            ceiling = reference * (1 + price_limit)
            floor = reference * (1 - price_limit)

            target_price = min(max(target_price, floor), ceiling)
            stop_loss = min(max(stop_loss, floor), ceiling)

            return MeanReversionSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                reasons=reasons,
                indicators={
                    "rsi": indicators.rsi,
                    "bb_position": indicators.bb_position,
                    "price_deviation_20": indicators.price_deviation_20,
                    "volume_ratio": indicators.volume_ratio,
                    "stoch_k": indicators.stoch_k,
                    "atr_pct": indicators.atr_pct,
                },
            )

    def _analyze_buy_conditions(
        self,
        indicators: IndicatorValues,
    ) -> Tuple[float, List[str]]:
        """Analyze buy (long) conditions."""
        score = 0.0
        reasons = []

        params = self.params

        # RSI oversold
        if indicators.rsi < params["rsi_extreme_oversold"]:
            score += params["weight_rsi"]
            reasons.append(f"RSI extreme oversold ({indicators.rsi:.1f})")
        elif indicators.rsi < params["rsi_oversold"]:
            score += params["weight_rsi"] * 0.7
            reasons.append(f"RSI oversold ({indicators.rsi:.1f})")

        # Below lower Bollinger Band
        if indicators.bb_position < 0:
            score += params["weight_bb"]
            reasons.append("Price below lower Bollinger Band")
        elif indicators.bb_position < 0.1:
            score += params["weight_bb"] * 0.6
            reasons.append("Price near lower Bollinger Band")

        # Price deviation from MA
        if indicators.price_deviation_20 < -params["extreme_deviation_threshold"]:
            score += params["weight_deviation"]
            reasons.append(f"Extreme deviation from 20-MA ({indicators.price_deviation_20:.1f}%)")
        elif indicators.price_deviation_20 < -params["deviation_threshold"]:
            score += params["weight_deviation"] * 0.6
            reasons.append(f"Below 20-MA ({indicators.price_deviation_20:.1f}%)")

        # Volume spike (potential reversal)
        if indicators.volume_ratio > params["volume_spike_threshold"]:
            score += params["weight_volume"] * 0.5
            reasons.append(f"Volume spike ({indicators.volume_ratio:.1f}x)")

        # Stochastic oversold
        if indicators.stoch_k < 20 and indicators.stoch_d < 20:
            score += params["weight_stochastic"]
            reasons.append(
                f"Stochastic oversold ({indicators.stoch_k:.1f}/{indicators.stoch_d:.1f})"
            )
        elif indicators.stoch_k < 30:
            score += params["weight_stochastic"] * 0.5

        return min(score, 1.0), reasons

    def _analyze_sell_conditions(
        self,
        indicators: IndicatorValues,
    ) -> Tuple[float, List[str]]:
        """Analyze sell (short) conditions."""
        score = 0.0
        reasons = []

        params = self.params

        # RSI overbought
        if indicators.rsi > params["rsi_extreme_overbought"]:
            score += params["weight_rsi"]
            reasons.append(f"RSI extreme overbought ({indicators.rsi:.1f})")
        elif indicators.rsi > params["rsi_overbought"]:
            score += params["weight_rsi"] * 0.7
            reasons.append(f"RSI overbought ({indicators.rsi:.1f})")

        # Above upper Bollinger Band
        if indicators.bb_position > 1:
            score += params["weight_bb"]
            reasons.append("Price above upper Bollinger Band")
        elif indicators.bb_position > 0.9:
            score += params["weight_bb"] * 0.6
            reasons.append("Price near upper Bollinger Band")

        # Price deviation from MA
        if indicators.price_deviation_20 > params["extreme_deviation_threshold"]:
            score += params["weight_deviation"]
            reasons.append(f"Extreme deviation from 20-MA (+{indicators.price_deviation_20:.1f}%)")
        elif indicators.price_deviation_20 > params["deviation_threshold"]:
            score += params["weight_deviation"] * 0.6
            reasons.append(f"Above 20-MA (+{indicators.price_deviation_20:.1f}%)")

        # Volume spike
        if indicators.volume_ratio > params["volume_spike_threshold"]:
            score += params["weight_volume"] * 0.5
            reasons.append(f"Volume spike ({indicators.volume_ratio:.1f}x)")

        # Stochastic overbought
        if indicators.stoch_k > 80 and indicators.stoch_d > 80:
            score += params["weight_stochastic"]
            reasons.append(
                f"Stochastic overbought ({indicators.stoch_k:.1f}/{indicators.stoch_d:.1f})"
            )
        elif indicators.stoch_k > 70:
            score += params["weight_stochastic"] * 0.5

        return min(score, 1.0), reasons

    def _apply_vn_market_adjustments(
        self,
        confidence: float,
        indicators: IndicatorValues,
        exchange: str,
    ) -> float:
        """Apply Vietnam market-specific adjustments."""
        adjusted = confidence

        # Adjust for volatility
        if indicators.atr_pct > VN_MARKET_CONFIG["high_volatility_threshold"] * 100:
            # High volatility - reduce confidence slightly
            adjusted *= 0.9

        # Adjust for exchange characteristics
        if exchange == "UPCOM":
            # UPCoM has lower liquidity
            adjusted *= 0.85
        elif exchange == "HNX":
            adjusted *= 0.95

        # Bollinger Band width adjustment
        # Very wide bands indicate high volatility regime
        if indicators.bb_width > 0.10:  # > 10% width
            adjusted *= 0.9

        return adjusted

    def scan_symbols(
        self,
        symbols: List[str],
        data_provider,  # Expects get_historical_data(symbol) -> DataFrame
        exchange: str = "HOSE",
    ) -> List[MeanReversionSignal]:
        """
        Scan multiple symbols for mean reversion signals.

        Args:
            symbols: List of stock symbols
            data_provider: Data provider with get_historical_data method
            exchange: Exchange type

        Returns:
            List of signals sorted by confidence
        """
        signals = []

        for symbol in symbols:
            try:
                df = data_provider.get_historical_data(symbol, days=100)
                if df is not None and len(df) >= 50:
                    signal = self.generate_signal(symbol, df, exchange)
                    if signal:
                        signals.append(signal)
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                continue

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)

        return signals

    def backtest(
        self,
        symbol: str,
        df: pd.DataFrame,
        initial_capital: float = 1_000_000_000,  # 1B VND
    ) -> Dict[str, Any]:
        """
        Simple backtest of mean reversion strategy.

        Returns performance metrics.
        """
        if len(df) < 100:
            return {"error": "Insufficient data"}

        capital = initial_capital
        position = 0
        entry_price = 0.0
        trades = []
        equity_curve = [capital]

        # Use rolling window
        for i in range(50, len(df)):
            window = df.iloc[: i + 1].copy()
            current_price = window["close"].iloc[-1]

            signal = self.generate_signal(symbol, window)

            if signal and position == 0:
                # Enter position
                if signal.is_buy:
                    position_size = int(capital * 0.1 / current_price)  # 10% of capital
                    position_size = (position_size // 100) * 100  # Round to lot

                    if position_size > 0:
                        position = position_size
                        entry_price = current_price
                        capital -= position_size * current_price

                        trades.append(
                            {
                                "type": "entry",
                                "date": (
                                    window.index[-1]
                                    if isinstance(window.index[-1], datetime)
                                    else datetime.now()
                                ),
                                "price": current_price,
                                "quantity": position_size,
                                "signal_confidence": signal.confidence,
                            }
                        )

            elif position > 0:
                # Check exit conditions
                pnl_pct = (current_price - entry_price) / entry_price

                # Exit on opposite signal, target hit, or stop loss
                should_exit = False
                exit_reason = ""

                if signal and signal.is_sell:
                    should_exit = True
                    exit_reason = "opposite_signal"
                elif pnl_pct > 0.05:  # 5% target
                    should_exit = True
                    exit_reason = "target"
                elif pnl_pct < -0.03:  # 3% stop
                    should_exit = True
                    exit_reason = "stop_loss"

                if should_exit:
                    capital += position * current_price
                    trades.append(
                        {
                            "type": "exit",
                            "date": (
                                window.index[-1]
                                if isinstance(window.index[-1], datetime)
                                else datetime.now()
                            ),
                            "price": current_price,
                            "quantity": position,
                            "pnl_pct": pnl_pct * 100,
                            "reason": exit_reason,
                        }
                    )
                    position = 0
                    entry_price = 0.0

            # Track equity
            equity = capital + position * current_price
            equity_curve.append(equity)

        # Close any open position
        if position > 0:
            final_price = df["close"].iloc[-1]
            capital += position * final_price

        # Calculate metrics
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100

        equity_series = pd.Series(equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = drawdown.min()

        # Trade statistics
        entry_trades = [t for t in trades if t["type"] == "entry"]
        exit_trades = [t for t in trades if t["type"] == "exit"]

        winning_trades = [t for t in exit_trades if t.get("pnl_pct", 0) > 0]
        win_rate = len(winning_trades) / len(exit_trades) * 100 if exit_trades else 0

        return {
            "symbol": symbol,
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "total_trades": len(entry_trades),
            "win_rate_pct": win_rate,
            "avg_trade_pnl_pct": (
                np.mean([t.get("pnl_pct", 0) for t in exit_trades]) if exit_trades else 0
            ),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_strategy_instance: Optional[MeanReversionStrategy] = None
_lock = RLock()


def get_mean_reversion_strategy(params: Optional[Dict] = None) -> MeanReversionStrategy:
    """Get singleton strategy instance."""
    global _strategy_instance
    with _lock:
        if _strategy_instance is None:
            _strategy_instance = MeanReversionStrategy(params)
        return _strategy_instance


def reset_strategy():
    """Reset singleton."""
    global _strategy_instance
    with _lock:
        _strategy_instance = None


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING MEAN REVERSION STRATEGY")
    print("=" * 60)

    strategy = get_mean_reversion_strategy()

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")

    # Simulate mean-reverting price
    price = 50000
    prices = [price]
    for i in range(199):
        mean = 50000
        change = (mean - prices[-1]) * 0.02 + np.random.normal(0, 500)
        prices.append(prices[-1] + change)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": np.random.randint(100000, 1000000, 200),
        }
    )

    print("\n📊 Sample Data:")
    print(f"  Date Range: {dates[0]} to {dates[-1]}")
    print(f"  Price Range: {min(prices):,.0f} - {max(prices):,.0f}")

    # Generate signal
    print("\n📡 Signal Generation:")
    signal = strategy.generate_signal("TEST", df, "HOSE")

    if signal:
        print(f"\n  Symbol: {signal.symbol}")
        print(f"  Type: {signal.signal_type.value}")
        print(f"  Confidence: {signal.confidence:.2%}")
        print(f"  Entry: {signal.entry_price:,.0f}")
        print(f"  Stop Loss: {signal.stop_loss:,.0f}")
        print(f"  Target: {signal.target_price:,.0f}")
        print(f"  Reasons: {signal.reasons}")
    else:
        print("  No signal generated (market in neutral zone)")

    # Backtest
    print("\n📈 Backtest Results:")
    results = strategy.backtest("TEST", df)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
