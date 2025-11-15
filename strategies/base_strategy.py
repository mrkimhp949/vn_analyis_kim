from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class BaseStrategy(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các chiến lược giao dịch.

    Định nghĩa một giao diện chung mà cả live trading và backtesting có thể sử dụng.
    Mỗi chiến lược con phải triển khai các phương thức để tạo tín hiệu,
    tính toán kích thước vị thế, và xác định mức dừng lỗ/chốt lời.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        """
        Khởi tạo chiến lược với một cấu hình cụ thể.

        Args:
            strategy_config (Dict[str, Any]): Một dictionary chứa các tham số
                                               cấu hình cho chiến lược (ví dụ: ngưỡng RSI,
                                               độ dài MA, v.v.).
        """
        self.config = strategy_config
        self.name = strategy_config.get("name", "UnnamedStrategy")
        self._validate_config()

    def _validate_config(self):
        """
        (Tùy chọn) Kiểm tra xem cấu hình chiến lược có hợp lệ không.
        Nên được override bởi các lớp con.
        """

    @abstractmethod
    def generate_signals(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        Phương thức cốt lõi để tạo ra các tín hiệu giao dịch.

        Args:
            market_data (Dict[str, pd.DataFrame]): Dữ liệu thị trường cho các mã
                                                   cổ phiếu cần phân tích.
                                                   Key là symbol, value là DataFrame
                                                   chứa OHLCV.

        Returns:
            List[Dict[str, Any]]: Một danh sách các tín hiệu. Mỗi tín hiệu là một
                                  dictionary chứa ít nhất 'symbol', 'action' ('BUY'/'SELL'),
                                  'confidence', và 'reason'.
                                  Ví dụ:
                                  [
                                      {
                                          'symbol': 'FPT',
                                          'action': 'BUY',
                                          'confidence': 0.85,
                                          'reason': 'RSI < 30 and MACD crossover',
                                          'entry_price': 115000
                                      },
                                      ...
                                  ]
        """
        raise NotImplementedError("Subclasses must implement generate_signals.")

    def calculate_position_size(
        self, signal: Dict[str, Any], portfolio_context: Dict[str, Any]
    ) -> int:
        """
        Tính toán kích thước vị thế dựa trên tín hiệu và rủi ro danh mục.

        Args:
            signal (Dict[str, Any]): Tín hiệu được tạo ra từ `generate_signals`.
            portfolio_context (Dict[str, Any]): Thông tin về danh mục hiện tại,
                                                 ví dụ: tổng giá trị, tiền mặt có sẵn,
                                                 mức độ rủi ro cho phép.

        Returns:
            int: Số lượng cổ phiếu cần mua/bán. Trả về 0 nếu không nên giao dịch.
        """
        # Logic mặc định: sử dụng một phần trăm cố định của vốn
        risk_per_trade = portfolio_context.get("risk_per_trade", 0.02)  # 2% risk
        total_equity = portfolio_context.get("total_equity", 0)
        entry_price = signal.get("entry_price")
        stop_loss_price = signal.get("stop_loss")

        if not all(
            [
                total_equity > 0,
                entry_price > 0,
                stop_loss_price > 0,
                entry_price > stop_loss_price,
            ]
        ):
            return 0

        risk_amount_per_share = entry_price - stop_loss_price
        total_risk_capital = total_equity * risk_per_trade

        shares = int(total_risk_capital / risk_amount_per_share)

        # Làm tròn xuống theo lô 100
        return (shares // 100) * 100

    def determine_exit_levels(
        self, signal: Dict[str, Any]
    ) -> Dict[str, Optional[float]]:
        """
        Xác định các mức dừng lỗ (stop-loss) và chốt lời (take-profit).

        Args:
            signal (Dict[str, Any]): Tín hiệu đầu vào.

        Returns:
            Dict[str, Optional[float]]: Một dictionary chứa 'stop_loss' và 'take_profit'.
        """
        # Logic mặc định đơn giản, có thể được override
        entry_price = signal.get("entry_price")
        if not entry_price:
            return {"stop_loss": None, "take_profit": None}

        # Ví dụ: Stop-loss 8% và Take-profit 15%
        stop_loss = entry_price * (1 - self.config.get("default_sl_pct", 0.08))
        take_profit = entry_price * (1 + self.config.get("default_tp_pct", 0.15))

        return {"stop_loss": stop_loss, "take_profit": take_profit}

    def __str__(self) -> str:
        return f"Strategy(name='{self.name}')"
