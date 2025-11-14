
import pandas as pd
from typing import Dict, Any, List, Optional

from strategies.base_strategy import BaseStrategy
from ml_models import MLPredictor
from features import add_ml_features, get_feature_columns
import logging

logger = logging.getLogger(__name__)

class MlStrategy(BaseStrategy):
    """
    Chiến lược giao dịch dựa trên mô hình Machine Learning (Ensemble).
    
    Kế thừa từ BaseStrategy và triển khai logic tạo tín hiệu cụ thể.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        """
        Khởi tạo chiến lược ML.

        Args:
            strategy_config (Dict[str, Any]): Cấu hình cho chiến lược, bao gồm:
                - 'name': Tên chiến lược.
                - 'confidence_threshold': Ngưỡng tin cậy để tạo tín hiệu.
                - 'sl_pct': Phần trăm dừng lỗ.
                - 'tp_pct': Phần trăm chốt lời.
        """
        super().__init__(strategy_config)
        self.predictor = MLPredictor()
        self.predictor.load_models()
        self.feature_columns = get_feature_columns()

    def _validate_config(self):
        """Kiểm tra các cấu hình cần thiết cho chiến lược ML."""
        required_keys = ["confidence_threshold", "sl_pct", "tp_pct"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key in MlStrategy: '{key}'")

    def generate_signals(self, market_data: Dict[str, pd.DataFrame], **kwargs) -> List[Dict[str, Any]]:
        """
        Tạo tín hiệu MUA/BÁN dựa trên dự đoán của mô hình ML.

        Args:
            market_data (Dict[str, pd.DataFrame]): Dữ liệu thị trường cho các mã.
            **kwargs: Có thể chứa 'vnindex_df' cho việc tính toán features.

        Returns:
            List[Dict[str, Any]]: Danh sách các tín hiệu được tạo ra.
        """
        signals = []
        vnindex_df = kwargs.get("vnindex_df")

        for symbol, df in market_data.items():
            if df.empty or len(df) < 50:
                logger.warning(f"[{self.name}] Không đủ dữ liệu cho mã {symbol}, bỏ qua.")
                continue

            try:
                # 1. Thêm features
                df_with_features = add_ml_features(df, index_df=vnindex_df)
                
                # Lấy dòng dữ liệu cuối cùng để dự đoán
                latest_features = df_with_features[self.feature_columns].iloc[-1:]

                if latest_features.isnull().values.any():
                    logger.warning(f"[{self.name}] Dữ liệu features cho {symbol} có giá trị NaN, bỏ qua.")
                    continue

                # 2. Lấy dự đoán từ mô hình
                confidence = self.predictor.predict(latest_features)[0]
                
                current_price = df['close'].iloc[-1]

                # 3. Tạo tín hiệu nếu đủ ngưỡng tin cậy
                if confidence >= self.config["confidence_threshold"]:
                    signal = {
                        "symbol": symbol,
                        "action": "BUY",
                        "confidence": float(confidence),
                        "reason": f"ML model confidence ({confidence:.2f}) > threshold ({self.config['confidence_threshold']})",
                        "entry_price": current_price,
                        "strategy_name": self.name,
                    }

                    # 4. Xác định mức SL/TP
                    exit_levels = self.determine_exit_levels(signal)
                    signal.update(exit_levels)
                    
                    signals.append(signal)

            except Exception as e:
                logger.error(f"[{self.name}] Lỗi khi tạo tín hiệu cho {symbol}: {e}", exc_info=True)

        logger.info(f"[{self.name}] Đã tạo ra {len(signals)} tín hiệu MUA.")
        return signals

    def determine_exit_levels(self, signal: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """
        Override logic mặc định để sử dụng SL/TP từ config của chiến lược này.
        """
        entry_price = signal.get("entry_price")
        if not entry_price:
            return {"stop_loss": None, "take_profit": None}

        stop_loss = entry_price * (1 - self.config["sl_pct"])
        take_profit = entry_price * (1 + self.config["tp_pct"])

        return {"stop_loss": stop_loss, "take_profit": take_profit}
