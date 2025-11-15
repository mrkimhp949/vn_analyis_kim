# -*- coding: utf-8 -*-
"""
improved_exit_logic.py - Smart Exit Strategy
Chiến lược thoát lệnh chuyên nghiệp với trailing stop, take profit bậc thang
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Lý do thoát lệnh"""

    STOP_LOSS = "Stop Loss Hit"
    TRAILING_STOP = "Trailing Stop"
    TAKE_PROFIT_1 = "Take Profit 1 (Partial)"
    TAKE_PROFIT_2 = "Take Profit 2 (Main)"
    TAKE_PROFIT_3 = "Take Profit 3 (Moon)"
    ML_SIGNAL_SELL = "ML Signal SELL"
    TIME_DECAY = "Time Decay (Sideway quá lâu)"
    MARKET_CRASH = "Market Crash Protection"
    EMERGENCY_EXIT = "Emergency Exit (Portfolio protection)"
    REVERSAL_PATTERN = "Bearish Reversal Pattern"
    BREAKDOWN = "Support Breakdown"


@dataclass
class ExitDecision:
    """Decision kết quả thoát lệnh"""

    should_exit: bool
    exit_reason: Optional[ExitReason]
    exit_type: str  # 'FULL', 'PARTIAL_50%', 'PARTIAL_30%'
    exit_price: float
    expected_pnl: float
    expected_pnl_percent: float
    message: str
    urgency: int  # 1-5 (5 = exit ngay lập tức)


class ImprovedExitStrategy:
    """
    Chiến lược thoát lệnh nâng cao

    Tính năng:
    1. Trailing Stop - Bảo vệ lợi nhuận
    2. Take Profit bậc thang - Chốt lời từng phần
    3. Time-based exit - Thoát nếu sideway quá lâu
    4. Market protection - Thoát khi thị trường đảo chiều
    5. Pattern recognition - Thoát khi xuất hiện pattern đảo chiều
    6. Portfolio protection - Thoát khi portfolio loss quá nhiều
    """

    def __init__(
        self,
        take_profit_levels: List[float] = [0.10, 0.15, 0.25],  # 10%, 15%, 25%
        stop_loss_atr_multiplier: float = 2.0,
        trailing_stop_activation: float = 0.08,  # Kích hoạt trailing khi lời 8%
        trailing_stop_distance: float = 0.05,  # Trailing 5% from high
        max_holding_days: int = 20,  # Tối đa 20 ngày
        time_decay_threshold: float = 0.02,
    ):  # Nếu <2% lời sau 20 ngày → thoát
        self.tp_levels = take_profit_levels
        self.sl_atr_mult = stop_loss_atr_multiplier
        self.trailing_activation = trailing_stop_activation
        self.trailing_distance = trailing_stop_distance
        self.max_holding_days = max_holding_days
        self.time_decay_threshold = time_decay_threshold

        # Tracking
        self.position_highs = {}  # {symbol: highest_price_since_entry}

    def check_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        stop_loss: float,
        take_profit_targets: List[float],
        entry_date: datetime,
        df: pd.DataFrame,
        ml_signal: Optional[Dict] = None,
        market_regime: Optional[Dict] = None,
        partial_exits: List[float] = None,
    ) -> ExitDecision:
        """
        Kiểm tra xem có nên thoát lệnh không

        Args:
            symbol: Mã cổ phiếu
            entry_price: Giá vào
            current_price: Giá hiện tại
            stop_loss: Stop loss ban đầu
            take_profit_targets: List [TP1, TP2, TP3]
            entry_date: Ngày vào lệnh
            df: DataFrame với OHLCV + indicators
            ml_signal: Signal từ ML (optional)
            market_regime: Market regime info (optional)
            partial_exits: List các lần đã chốt lời 1 phần (optional)

        Returns:
            ExitDecision
        """

        if partial_exits is None:
            partial_exits = []

        # Calculate P&L
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        pnl_amount = current_price - entry_price  # Per share

        days_held = (datetime.now() - entry_date).days

        # Update highest price
        if symbol not in self.position_highs:
            self.position_highs[symbol] = current_price
        else:
            self.position_highs[symbol] = max(
                self.position_highs[symbol], current_price
            )

        highest_price = self.position_highs[symbol]

        # ====================================================================
        # CHECK 1: STOP LOSS (Ưu tiên cao nhất)
        # ====================================================================
        if current_price <= stop_loss:
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.STOP_LOSS,
                exit_type="FULL",
                exit_price=current_price,
                expected_pnl=pnl_amount,
                expected_pnl_percent=pnl_percent,
                message=f"⛔ STOP LOSS: {pnl_percent:+.2f}%",
                urgency=5,
            )

        # ====================================================================
        # CHECK 2: MARKET CRASH PROTECTION
        # ====================================================================
        if market_regime and market_regime.get("regime") == "BEAR":
            # Nếu thị trường chuyển Bear, thoát ngay kể cả khi đang lời
            if pnl_percent > 3:  # Có lời thì chốt
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.MARKET_CRASH,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"🚨 THỊ TRƯỜNG GIẢM ĐIỂM - Chốt lời sớm: {pnl_percent:+.2f}%",
                    urgency=4,
                )
            elif pnl_percent > -2:  # Lỗ ít thì cũng thoát
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.MARKET_CRASH,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"🚨 THỊ TRƯỜNG GIẢM ĐIỂM - Cắt lỗ sớm: {pnl_percent:+.2f}%",
                    urgency=4,
                )

        # ====================================================================
        # CHECK 3: TAKE PROFIT BẬC THANG
        # ====================================================================
        tp_check = self._check_take_profit_levels(
            current_price, take_profit_targets, partial_exits, pnl_percent, pnl_amount
        )

        if tp_check["should_exit"]:
            return tp_check["decision"]

        # ====================================================================
        # CHECK 4: PROFIT PROTECTION (3-8% profit range)
        # ====================================================================
        # NEW: Protect profit before trailing stop activation
        profit_protection_check = self._check_profit_protection(
            entry_price,
            current_price,
            highest_price,
            pnl_percent,
            pnl_amount,
        )

        if profit_protection_check["should_exit"]:
            return profit_protection_check["decision"]

        # ====================================================================
        # CHECK 5: TRAILING STOP (>= 8% profit)
        # ====================================================================
        trailing_check = self._check_trailing_stop(
            entry_price,
            current_price,
            highest_price,
            pnl_percent,
            pnl_amount,
        )

        if trailing_check["should_exit"]:
            return trailing_check["decision"]

        # ====================================================================
        # CHECK 6: ML SIGNAL SELL (with volume confirmation)
        # ====================================================================
        if ml_signal and ml_signal.get("signal") == "SELL":
            confidence = ml_signal.get("confidence", 0)

            # ENHANCEMENT: Add volume confirmation for ML sell signal
            volume_confirmation = self._check_volume_for_exit(df)

            # Chỉ thoát nếu confidence >= 60 và đang lời (hoặc lỗ ít)
            if confidence >= 60 and pnl_percent > -3:
                # If volume confirms (high volume on sell), increase urgency
                urgency = 4 if volume_confirmation else 3
                message = f"📉 ML SIGNAL SELL (Conf: {confidence}%): {pnl_percent:+.2f}%"
                if volume_confirmation:
                    message += " | Volume confirmed ⚠️"

                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.ML_SIGNAL_SELL,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=message,
                    urgency=urgency,
                )

        # ====================================================================
        # CHECK 7: BEARISH REVERSAL PATTERN
        # ====================================================================
        if len(df) >= 3:
            pattern_check = self._check_reversal_pattern(df, pnl_percent, pnl_amount)

            if pattern_check["should_exit"]:
                return pattern_check["decision"]

        # ====================================================================
        # CHECK 8: SUPPORT BREAKDOWN
        # ====================================================================
        breakdown_check = self._check_support_breakdown(
            df, current_price, entry_price, pnl_percent, pnl_amount
        )

        if breakdown_check["should_exit"]:
            return breakdown_check["decision"]

        # ====================================================================
        # CHECK 9: TIME DECAY
        # ====================================================================
        if days_held >= self.max_holding_days:
            # Nếu giữ quá lâu mà lời < threshold → thoát
            if pnl_percent < self.time_decay_threshold * 100:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.TIME_DECAY,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"⏰ SIDEWAY QUÁ LÂU ({days_held} ngày): {pnl_percent:+.2f}%",
                    urgency=2,
                )

        # ====================================================================
        # NO EXIT - HOLD
        # ====================================================================
        return ExitDecision(
            should_exit=False,
            exit_reason=None,
            exit_type="HOLD",
            exit_price=current_price,
            expected_pnl=pnl_amount,
            expected_pnl_percent=pnl_percent,
            message=f"✅ HOLD - P&L: {pnl_percent:+.2f}% | Days: {days_held}",
            urgency=0,
        )

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _check_take_profit_levels(
        self,
        current_price: float,
        tp_targets: List[float],
        partial_exits: List[float],
        pnl_percent: float,
        pnl_amount: float,
    ) -> Dict:
        """
        Check take profit bậc thang

        Strategy:
        - TP1 (10%): Chốt 30% position
        - TP2 (15%): Chốt 50% còn lại
        - TP3 (25%): Chốt 100% còn lại
        """

        # TP3 - Full exit
        if len(tp_targets) >= 3 and current_price >= tp_targets[2]:
            if len(partial_exits) < 3:
                return {
                    "should_exit": True,
                    "decision": ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.TAKE_PROFIT_3,
                        exit_type="FULL",
                        exit_price=current_price,
                        expected_pnl=pnl_amount,
                        expected_pnl_percent=pnl_percent,
                        message=f"🎯 TP3 - FULL EXIT: {pnl_percent:+.2f}%",
                        urgency=3,
                    ),
                }

        # TP2 - Partial 50%
        if len(tp_targets) >= 2 and current_price >= tp_targets[1]:
            if len(partial_exits) < 2:
                return {
                    "should_exit": True,
                    "decision": ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.TAKE_PROFIT_2,
                        exit_type="PARTIAL_50%",
                        exit_price=current_price,
                        expected_pnl=pnl_amount,
                        expected_pnl_percent=pnl_percent,
                        message=f"🎯 TP2 - CHỐT 50% position: {pnl_percent:+.2f}%",
                        urgency=2,
                    ),
                }

        # TP1 - Partial 30%
        if len(tp_targets) >= 1 and current_price >= tp_targets[0]:
            if len(partial_exits) < 1:
                return {
                    "should_exit": True,
                    "decision": ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.TAKE_PROFIT_1,
                        exit_type="PARTIAL_30%",
                        exit_price=current_price,
                        expected_pnl=pnl_amount,
                        expected_pnl_percent=pnl_percent,
                        message=f"🎯 TP1 - CHỐT 30% position: {pnl_percent:+.2f}%",
                        urgency=1,
                    ),
                }

        return {"should_exit": False}

    def _check_profit_protection(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        pnl_percent: float,
        pnl_amount: float,
    ) -> Dict:
        """
        NEW: Protect profit in the 3-8% profit range.

        This prevents giving back all profits before trailing stop activates at 8%.

        Logic:
        - If profit is between 3-8%:
          - Calculate a dynamic stop based on profit level
          - 3-5% profit: Protect 50% of profit
          - 5-8% profit: Protect 60% of profit
        - Uses highest price to track maximum profit achieved

        Returns:
            Dict with should_exit flag and decision
        """
        # Only activate in 3-8% profit range (before trailing stop)
        if pnl_percent < 3.0 or pnl_percent >= self.trailing_activation * 100:
            return {"should_exit": False}

        # Calculate maximum profit achieved
        max_profit_pct = ((highest_price - entry_price) / entry_price) * 100

        # Dynamic protection based on profit level
        if 3.0 <= pnl_percent < 5.0:
            # Protect 50% of maximum profit
            protection_pct = 0.50
            stop_price = entry_price * (1 + (max_profit_pct / 100) * protection_pct)
        elif 5.0 <= pnl_percent < self.trailing_activation * 100:
            # Protect 60% of maximum profit
            protection_pct = 0.60
            stop_price = entry_price * (1 + (max_profit_pct / 100) * protection_pct)
        else:
            return {"should_exit": False}

        # Check if current price dropped below protection level
        if current_price <= stop_price:
            profit_given_back = max_profit_pct - pnl_percent
            return {
                "should_exit": True,
                "decision": ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.TRAILING_STOP,  # Use same reason for consistency
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"💰 PROFIT PROTECTION: Bảo vệ {protection_pct*100:.0f}% lợi nhuận | "
                    f"Max profit: {max_profit_pct:.1f}% → Current: {pnl_percent:.1f}% "
                    f"(Gave back {profit_given_back:.1f}%)",
                    urgency=4,
                ),
            }

        return {"should_exit": False}

    def _check_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        pnl_percent: float,
        pnl_amount: float,
    ) -> Dict:
        """
        Check trailing stop based on a fixed percentage.
        Kích hoạt khi lời >= 8%
        Trailing stop cách đỉnh 5%.
        """

        # Check if trailing should be activated
        profit_from_entry = (current_price - entry_price) / entry_price

        if profit_from_entry < self.trailing_activation:
            # Chưa đủ lời để kích hoạt trailing
            return {"should_exit": False}

        # Trailing activated - check if price dropped too much from high
        trailing_stop_price = highest_price * (1 - self.trailing_distance)

        if current_price <= trailing_stop_price:
            drawdown_from_high = ((highest_price - current_price) / highest_price) * 100
            return {
                "should_exit": True,
                "decision": ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.TRAILING_STOP,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"📉 TRAILING STOP: Giảm {drawdown_from_high:.1f}% từ đỉnh | "
                    f"P&L: {pnl_percent:+.2f}%",
                    urgency=4,
                ),
            }

        return {"should_exit": False}

    def _check_reversal_pattern(
        self, df: pd.DataFrame, pnl_percent: float, pnl_amount: float
    ) -> Dict:
        """
        Check bearish reversal patterns

        - Bearish engulfing
        - Evening star
        - Shooting star
        """

        if len(df) < 3:
            return {"should_exit": False}

        # Use safe access instead of df.iloc[-1]
        if len(df) < 2:
            return {"should_exit": False}

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Chỉ check pattern nếu đang lời
        if pnl_percent <= 0:
            return {"should_exit": False}

        # Bearish Engulfing
        if (
            prev["close"] > prev["open"]  # Prev bullish
            and latest["close"] < latest["open"]  # Current bearish
            and latest["close"] < prev["open"]
            and latest["open"] > prev["close"]
        ):
            return {
                "should_exit": True,
                "decision": ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.REVERSAL_PATTERN,
                    exit_type="FULL",
                    exit_price=latest["close"],
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"🔴 BEARISH ENGULFING: Chốt lời sớm {pnl_percent:+.2f}%",
                    urgency=3,
                ),
            }

        # Shooting Star (at resistance)
        body = abs(latest["close"] - latest["open"])
        upper_shadow = latest["high"] - max(latest["close"], latest["open"])
        lower_shadow = min(latest["close"], latest["open"]) - latest["low"]

        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            return {
                "should_exit": True,
                "decision": ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.REVERSAL_PATTERN,
                    exit_type="PARTIAL_50%",
                    exit_price=latest["close"],
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"⭐ SHOOTING STAR: Chốt 50% {pnl_percent:+.2f}%",
                    urgency=2,
                ),
            }

        return {"should_exit": False}

    def _check_volume_for_exit(self, df: pd.DataFrame) -> bool:
        """
        ENHANCEMENT: Check if volume confirms exit signal

        Returns:
            True if high volume confirms selling pressure
        """
        if len(df) < 20:
            return False

        try:
            current_volume = safe_get_latest(df, "volume", 0)
            avg_volume = safe_rolling_operation(df, "volume", 20, "mean", 0)

            if avg_volume == 0:
                return False

            # Check if volume is significantly higher (selling pressure)
            volume_ratio = current_volume / avg_volume

            # High volume on down day suggests strong selling
            latest_close = safe_get_latest(df, "close", 0)
            prev_close = df["close"].iloc[-2] if len(df) >= 2 else latest_close
            is_down_day = latest_close < prev_close

            return volume_ratio >= 1.5 and is_down_day

        except Exception:
            logger.warning("⚠️ Error checking volume for exit")
            return False

    def _check_support_breakdown(
        self,
        df: pd.DataFrame,
        current_price: float,
        entry_price: float,
        pnl_percent: float,
        pnl_amount: float,
    ) -> Dict:
        """
        Check nếu giá break support quan trọng
        """

        if len(df) < 20:
            return {"should_exit": False}

        # Support = low của 20 ngày trước
        support = df["low"].iloc[-20:-1].min()

        # Nếu giá break support với volume cao
        if current_price < support:
            volume_surge = (
                safe_get_latest(df, "volume", 0)
                > safe_rolling_operation(df, "volume", 20, "mean", 0) * 1.5
            )

            if volume_surge:
                return {
                    "should_exit": True,
                    "decision": ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.BREAKDOWN,
                        exit_type="FULL",
                        exit_price=current_price,
                        expected_pnl=pnl_amount,
                        expected_pnl_percent=pnl_percent,
                        message=f"📉 SUPPORT BREAKDOWN (Volume confirmed): {pnl_percent:+.2f}%",
                        urgency=4,
                    ),
                }

        return {"should_exit": False}

    def clear_position_tracking(self, symbol: str):
        """
        Dọn dẹp tracking khi đóng vị thế để tránh memory leak

        Args:
            symbol: Mã cổ phiếu cần xóa tracking
        """
        if symbol in self.position_highs:
            del self.position_highs[symbol]
            logger.debug(f"✅ Cleared position tracking for {symbol}")
        else:
            logger.debug(f"⚠️ No tracking found for {symbol}")

    def get_tracked_positions(self) -> List[str]:
        """
        Lấy danh sách các vị thế đang được track

        Returns:
            List các symbol đang được track
        """
        return list(self.position_highs.keys())

    def clear_all_tracking(self):
        """Xóa toàn bộ tracking (dùng khi reset hoặc end of day)"""
        count = len(self.position_highs)
        self.position_highs.clear()
        logger.info(f"🧹 Cleared tracking for {count} positions")

    def format_exit_message(self, symbol: str, decision: ExitDecision) -> str:
        """Format exit decision thành message"""

        urgency_emoji = {5: "🚨🚨🚨", 4: "🚨🚨", 3: "⚠️", 2: "💡", 1: "ℹ️", 0: "✅"}

        if not decision.should_exit:
            return f"✅ **{symbol}** - HOLD\n" f"{decision.message}"

        emoji = urgency_emoji.get(decision.urgency, "⚠️")

        msg = f"{emoji} **{symbol}** - EXIT SIGNAL\n\n"
        msg += f"📍 **Exit Type:** {decision.exit_type}\n"
        msg += f"🎯 **Reason:** {decision.exit_reason.value}\n"
        msg += f"💰 **Exit Price:** {decision.exit_price:,.0f} VNĐ\n"
        msg += f"📊 **P&L:** {decision.expected_pnl_percent:+.2f}%\n"
        msg += f"⚡ **Urgency:** {decision.urgency}/5\n\n"
        msg += f"💬 {decision.message}"

        return msg


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from src.data.loader import load_data
    from src.ml.features.technical import add_ml_features
    from src.ml.signals.generator import MLSignalGenerator
    from utils.dataframe_utils import safe_get_latest

    print("\n" + "=" * 70)
    print("🧪 TESTING IMPROVED EXIT STRATEGY")
    print("=" * 70 + "\n")

    # Test với 1 mã
    symbol = "VNM"
    df = load_data(symbol, 200)
    df = add_ml_features(df)

    # Giả lập position
    entry_price = 80_000
    entry_date = datetime.now() - timedelta(days=10)
    current_price = 88_000  # +10% lời
    stop_loss = 76_000  # -5%
    take_profit_targets = [
        84_000,  # TP1: +5%
        86_000,  # TP2: +7.5%
        88_000,  # TP3: +10%
    ]

    # Get ML signal
    ml_gen = MLSignalGenerator()
    ml_signal = ml_gen.analyze(df)

    # Initialize exit strategy
    exit_strategy = ImprovedExitStrategy()

    # Check exit
    decision = exit_strategy.check_exit(
        symbol=symbol,
        entry_price=entry_price,
        current_price=current_price,
        stop_loss=stop_loss,
        take_profit_targets=take_profit_targets,
        entry_date=entry_date,
        df=df,
        ml_signal=ml_signal,
        market_regime={"regime": "BULL", "tradeable": True},
    )

    # Print result
    print(exit_strategy.format_exit_message(symbol, decision))
    print("\n" + "=" * 70)
