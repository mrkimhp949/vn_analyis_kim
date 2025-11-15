# -*- coding: utf-8 -*-
"""
Strategy Manager for the trading bot.
Quản lý và cung cấp các đối tượng chiến lược (entry, exit, sizing).
"""
import logging
from typing import Any, Dict

from src.strategies.exit_logic import ImprovedExitStrategy

# Import các lớp chiến lược
from src.strategies.entry_logic import ImprovedEntryLogic
from src.strategies.position_sizing import EnhancedPositionSizer
from src.config.trading_config import get_config


# Risk calculation constants
RISK_MULTIPLIER = 0.02  # 2% risk multiplier
RISK_DIVISOR = 0.15  # 15% risk divisor


class StrategyManager:
    """
    Lớp chịu trách nhiệm khởi tạo và điều chỉnh các chiến lược
    dựa trên trạng thái thị trường và cấu hình.
    """

    def __init__(self):
        self.trading_config = get_config(validate=False)
        self.entry_logic: ImprovedEntryLogic = None
        self.position_sizer: EnhancedPositionSizer = None
        self.exit_strategy: ImprovedExitStrategy = None
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Khởi tạo các chiến lược với cấu hình mặc định."""
        logging.info("🚀 Khởi tạo các chiến lược trading...")

        # 1. Entry Logic
        try:
            self.entry_logic = ImprovedEntryLogic(
                min_confidence=55,
                min_risk_reward=1.8,
                require_trend_alignment=True,
                require_volume_confirmation=False,
            )
            logging.info("✅ EntryLogic initialized (min_conf=55, R:R=1.8)")
        except Exception:
            logging.critical("❌ Không thể khởi tạo ImprovedEntryLogic", exc_info=True)
            # Có thể raise exception ở đây để dừng bot nếu logic vào lệnh là bắt buộc
            raise

        # 2. Position Sizer
        try:
            self.position_sizer = EnhancedPositionSizer(
                total_capital=100_000_000,
                max_risk_per_trade=self.trading_config.trading.max_position_size
                * RISK_MULTIPLIER
                / RISK_DIVISOR,
                max_position_size=self.trading_config.trading.max_position_size,
                min_position_size=self.trading_config.trading.min_position_size,
                max_total_exposure=0.60,
                max_portfolio_risk=self.trading_config.trading.max_portfolio_risk,
                max_sector_exposure=self.trading_config.trading.max_sector_exposure,
                use_kelly=True,
                kelly_fraction=0.5,
            )
            logging.info("✅ EnhancedPositionSizer initialized (with Kelly Criterion)")
        except Exception:
            logging.critical("❌ Không thể khởi tạo PositionSizer", exc_info=True)
            raise

        # 3. Exit Strategy
        try:
            self.exit_strategy = ImprovedExitStrategy()
            logging.info("✅ ImprovedExitStrategy initialized")
        except Exception:
            logging.critical("❌ Không thể khởi tạo ExitStrategy", exc_info=True)
            raise

    def get_strategies(self) -> Dict[str, Any]:
        """Trả về một dict chứa các đối tượng chiến lược đã được khởi tạo."""
        return {
            "entry_logic": self.entry_logic,
            "position_sizer": self.position_sizer,
            "exit_strategy": self.exit_strategy,
        }

    def apply_market_adjustments(self, market_regime: Dict):
        """
        Điều chỉnh các tham số của chiến lược dựa trên trạng thái thị trường.
        Đây là nơi tập trung logic điều chỉnh động.
        """
        if not self.entry_logic or not self.position_sizer:
            logging.warning(
                "⚠️ EntryLogic hoặc PositionSizer chưa được khởi tạo, không thể điều chỉnh."
            )
            return

        regime = (market_regime or {}).get("regime", "UNKNOWN").upper()
        logging.info(f"⚙️ Áp dụng điều chỉnh chiến lược cho thị trường: {regime}")

        # DYNAMIC THRESHOLD based on market regime
        if regime == "BULL":
            self.entry_logic.min_confidence = 50
            self.entry_logic.min_risk_reward = 1.5
            if hasattr(self.position_sizer, "max_total_exposure"):
                self.position_sizer.max_total_exposure = 0.70
        elif regime == "BEAR":
            self.entry_logic.min_confidence = 65
            self.entry_logic.min_risk_reward = 2.0
            if hasattr(self.position_sizer, "max_total_exposure"):
                self.position_sizer.max_total_exposure = 0.30
        else:  # SIDEWAYS / UNKNOWN
            self.entry_logic.min_confidence = 55
            self.entry_logic.min_risk_reward = 1.8
            if hasattr(self.position_sizer, "max_total_exposure"):
                self.position_sizer.max_total_exposure = 0.50

        logging.info(
            f"   -> min_conf={self.entry_logic.min_confidence}%, "
            f"R:R>={self.entry_logic.min_risk_reward}, "
            f"max_exposure={getattr(self.position_sizer, 'max_total_exposure', 0.0)*100:.0f}%"
        )


# Singleton instance
_strategy_manager = None


def get_strategy_manager() -> "StrategyManager":
    """
    Trả về instance singleton của StrategyManager.
    Khởi tạo nếu chưa tồn tại.
    """
    global _strategy_manager
    if _strategy_manager is None:
        logging.info("Initializing StrategyManager singleton...")
        _strategy_manager = StrategyManager()
    return _strategy_manager
