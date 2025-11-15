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
        Tạo tín hiệu dựa trên RSI.
        """
        signals = []
        for symbol, df in market_data.items():
            if df.empty or len(df) < self.config["rsi_period"]:
                continue

            # Tính RSI
            rsi = ta.momentum.RSIIndicator(
                close=df["close"], window=self.config["rsi_period"]
            ).rsi()
            latest_rsi = rsi.iloc[-1]
            current_price = df["close"].iloc[-1]

            # Tạo tín hiệu MUA
            if latest_rsi < self.config["buy_threshold"]:
                signal = {
                    "symbol": symbol,
                    "action": "BUY",
                    "confidence": 1
                    - (
                        latest_rsi / self.config["buy_threshold"]
                    ),  # Confidence cao hơn khi RSI càng thấp
                    "reason": f"RSI({self.config['rsi_period']}) is {latest_rsi:.2f} < {self.config['buy_threshold']}",
                    "entry_price": current_price,
                    "strategy_name": self.name,
                }
                exit_levels = self.determine_exit_levels(signal)
                signal.update(exit_levels)
                signals.append(signal)

            # Logic tạo tín hiệu BÁN (để đóng vị thế) có thể được thêm ở đây
            # Ví dụ:
            # if latest_rsi > self.config["sell_threshold"]:
            #     signals.append({
            #         "symbol": symbol,
            #         "action": "SELL",
            #         ...
            #     })

        logger.info(f"[{self.name}] Đã tạo ra {len(signals)} tín hiệu.")
        return signals

    def determine_exit_levels(self, signal: Dict[str, Any]) -> Dict[str, float]:
        """Xác định SL/TP dựa trên cấu hình của chiến lược này."""
        entry_price = signal["entry_price"]
        stop_loss = entry_price * (1 - self.config["sl_pct"])
        take_profit = entry_price * (1 + self.config["tp_pct"])
        return {"stop_loss": stop_loss, "take_profit": take_profit}
