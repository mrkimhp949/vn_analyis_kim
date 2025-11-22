import logging
from typing import Any, Dict, List

import pandas as pd
import ta

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class RsiStrategy(BaseStrategy):
    """
    Chiến lược giao dịch đơn giản dựa trên chỉ báo RSI.
    - Mua khi RSI < 30 (quá bán).
    - Bán khi RSI > 70 (quá mua).
    """

    def _validate_config(self):
        required_keys = [
            "rsi_period",
            "buy_threshold",
            "sell_threshold",
            "sl_pct",
            "tp_pct",
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key in RsiStrategy: '{key}'")

    def generate_signals(
        self, market_data: Dict[str, pd.DataFrame], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Tạo tín hiệu dựa trên RSI với multi-indicator confirmation.

        Cải tiến:
        1. EMA trend confirmation (EMA20 > EMA50)
        2. Volume confirmation (volume > 1.2x average)
        3. MACD confirmation (optional, adds bonus confidence)
        4. Price action check (không mua khi đang tạo lower lows)
        5. Support level check
        """
        signals = []
        for symbol, df in market_data.items():
            if df.empty or len(df) < max(50, self.config["rsi_period"]):
                continue

            # Tính các indicators
            rsi = ta.momentum.RSIIndicator(
                close=df["close"], window=self.config["rsi_period"]
            ).rsi()
            ema20 = df["close"].ewm(span=20).mean()
            ema50 = df["close"].ewm(span=50).mean()

            # MACD
            macd_indicator = ta.trend.MACD(df["close"])
            macd = macd_indicator.macd()
            macd_signal = macd_indicator.macd_signal()

            # Volume
            volume_ma20 = df["volume"].rolling(20).mean()

            # Latest values
            latest_rsi = rsi.iloc[-1]
            current_price = df["close"].iloc[-1]
            latest_ema20 = ema20.iloc[-1]
            latest_ema50 = ema50.iloc[-1]
            latest_macd = macd.iloc[-1]
            latest_macd_signal = macd_signal.iloc[-1]
            current_volume = df["volume"].iloc[-1]
            avg_volume = volume_ma20.iloc[-1]

            # Support level (20-day low)
            support_20 = df["low"].rolling(20).min().iloc[-1]

            # =================================================================
            # BUY SIGNAL LOGIC WITH CONFIRMATIONS
            # =================================================================
            if latest_rsi < self.config["buy_threshold"]:

                # Initialize confidence and reasons
                base_confidence = 1 - (latest_rsi / self.config["buy_threshold"])
                confidence_adjustments = []
                reasons = [
                    f"RSI({self.config['rsi_period']}) is {latest_rsi:.2f} < {self.config['buy_threshold']}"
                ]

                pass_filters = True

                # FILTER 1: EMA Trend Confirmation
                if latest_ema20 > latest_ema50:
                    confidence_adjustments.append(0.15)
                    reasons.append(f"EMA20({latest_ema20:.0f}) > EMA50({latest_ema50:.0f})")
                else:
                    # Downtrend - reduce confidence significantly
                    confidence_adjustments.append(-0.25)
                    reasons.append(f"⚠️ Downtrend: EMA20 < EMA50")

                # FILTER 2: Volume Confirmation
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                if volume_ratio >= 1.2:
                    confidence_adjustments.append(0.10)
                    reasons.append(f"Volume {volume_ratio:.1f}x avg")
                elif volume_ratio < 0.8:
                    confidence_adjustments.append(-0.15)
                    reasons.append(f"⚠️ Low volume {volume_ratio:.1f}x")

                # FILTER 3: MACD Confirmation (bonus)
                if latest_macd > latest_macd_signal:
                    confidence_adjustments.append(0.08)
                    reasons.append("MACD bullish")

                # FILTER 4: Price Action Check (not making lower lows)
                if len(df) >= 5:
                    recent_low = df["low"].tail(5).min()
                    prev_low = df["low"].tail(10).head(5).min() if len(df) >= 10 else recent_low

                    if recent_low > prev_low:
                        confidence_adjustments.append(0.05)
                        reasons.append("Higher lows")
                    elif recent_low < prev_low * 0.97:  # Making significantly lower lows
                        confidence_adjustments.append(-0.20)
                        reasons.append("⚠️ Lower lows pattern")

                # FILTER 5: Near Support Level
                distance_to_support = ((current_price - support_20) / current_price) * 100
                if distance_to_support <= 3.0:  # Within 3% of support
                    confidence_adjustments.append(0.12)
                    reasons.append(f"Near support ({distance_to_support:.1f}%)")

                # Calculate final confidence
                final_confidence = base_confidence + sum(confidence_adjustments)
                final_confidence = max(0.0, min(1.0, final_confidence))  # Clamp to [0, 1]

                # Only create signal if confidence is still positive and meets minimum threshold
                min_confidence_threshold = 0.40  # At least 40% confidence required

                if final_confidence >= min_confidence_threshold and pass_filters:
                    signal = {
                        "symbol": symbol,
                        "action": "BUY",
                        "confidence": final_confidence,
                        "reason": " | ".join(reasons),
                        "entry_price": current_price,
                        "strategy_name": self.name,
                        # Additional metadata for analysis
                        "rsi": latest_rsi,
                        "volume_ratio": volume_ratio,
                        "ema_trend": "UP" if latest_ema20 > latest_ema50 else "DOWN",
                    }
                    exit_levels = self.determine_exit_levels(signal)
                    signal.update(exit_levels)
                    signals.append(signal)
                else:
                    logger.debug(
                        f"[{self.name}] {symbol}: RSI signal failed confirmations "
                        f"(confidence: {final_confidence:.2%})"
                    )

            # =================================================================
            # SELL SIGNAL LOGIC (optional - for closing positions)
            # =================================================================
            elif latest_rsi > self.config.get("sell_threshold", 70):
                confidence = (latest_rsi - 70) / 30  # Higher RSI = higher confidence to sell
                reasons = [f"RSI overbought: {latest_rsi:.2f}"]

                # Volume confirmation for sell too
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                if volume_ratio >= 1.2:
                    confidence += 0.10
                    reasons.append(f"High volume {volume_ratio:.1f}x")

                if confidence >= 0.4:  # Minimum 40% confidence
                    signal = {
                        "symbol": symbol,
                        "action": "SELL",
                        "confidence": min(confidence, 1.0),
                        "reason": " | ".join(reasons),
                        "entry_price": current_price,
                        "strategy_name": self.name,
                    }
                    signals.append(signal)

        logger.info(f"[{self.name}] Đã tạo ra {len(signals)} tín hiệu.")
        return signals

    def determine_exit_levels(self, signal: Dict[str, Any]) -> Dict[str, float]:
        """Xác định SL/TP dựa trên cấu hình của chiến lược này."""
        entry_price = signal["entry_price"]
        stop_loss = entry_price * (1 - self.config["sl_pct"])
        take_profit = entry_price * (1 + self.config["tp_pct"])
        return {"stop_loss": stop_loss, "take_profit": take_profit}
