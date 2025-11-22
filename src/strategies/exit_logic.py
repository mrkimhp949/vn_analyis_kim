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
        trailing_stop_distance: float = 0.05,  # Trailing 5% from high (fallback)
        trailing_stop_atr_multiplier: float = 2.0,  # Dynamic: use 2×ATR instead of fixed %
        use_dynamic_trailing: bool = True,  # Use ATR-based trailing stop
        max_holding_days: int = 30,  # Extended from 20 to 30 trading days
        time_decay_threshold: float = 0.02,  # Nếu <2% lời sau 30 ngày → thoát
        default_stop_loss_pct: float = -7.0,
        # SIMPLIFIED PROFIT PROTECTION: Single threshold replaces complex 3-5-8% logic
        profit_protection_activation: float = 0.05,  # Activate at 5% profit
        profit_protection_percent: float = 0.50,  # Protect 50% of max profit
    ):
        self.tp_levels = take_profit_levels
        self.sl_atr_mult = stop_loss_atr_multiplier
        self.trailing_activation = trailing_stop_activation
        self.trailing_distance = trailing_stop_distance
        self.trailing_atr_mult = trailing_stop_atr_multiplier
        self.use_dynamic_trailing = use_dynamic_trailing
        self.max_holding_days = max_holding_days
        self.time_decay_threshold = time_decay_threshold
        self.default_stop_loss_pct = default_stop_loss_pct

        # SIMPLIFIED: Profit protection config
        self.profit_protection_activation = profit_protection_activation
        self.profit_protection_percent = profit_protection_percent

        # Tracking
        self.position_highs = {}  # {symbol: highest_price_since_entry}

    def check_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        stop_loss: Optional[float],
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
            stop_loss: Stop loss ban đầu (có thể None, sẽ dùng fallback 7% dưới entry)
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

        # Ensure stop loss is valid even if missing from stored position
        # Pass df for ATR-based calculation if needed
        stop_loss = self._ensure_stop_loss(symbol, entry_price, stop_loss, df)

        # Calculate P&L
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        pnl_amount = current_price - entry_price  # Per share

        # CRITICAL FIX: Use trading days instead of calendar days
        # to account for weekends and holidays
        try:
            import pandas as pd
            from pandas.tseries.offsets import BDay

            # Count business days (trading days) between entry and now
            trading_days_held = len(pd.date_range(entry_date, datetime.now(), freq=BDay()))

            # Use trading days for time decay logic
            days_held = trading_days_held
            logger.debug(
                f"📅 {symbol} held for {trading_days_held} trading days "
                f"(calendar: {(datetime.now() - entry_date).days} days)"
            )
        except Exception as e:
            # Fallback to calendar days if business day calculation fails
            logger.warning(f"Failed to calculate trading days: {e}, using calendar days")
            days_held = (datetime.now() - entry_date).days

        # Update highest price
        if symbol not in self.position_highs:
            self.position_highs[symbol] = current_price
        else:
            self.position_highs[symbol] = max(self.position_highs[symbol], current_price)

        highest_price = self.position_highs[symbol]

        # ====================================================================
        # CHECK 1: STOP LOSS (Ưu tiên cao nhất)
        # ====================================================================
        # Use stop_loss if available, otherwise fallback to 7% below entry
        effective_stop_loss = stop_loss
        if effective_stop_loss is None or effective_stop_loss <= 0:
            effective_stop_loss = entry_price * 0.93  # 7% below entry as fallback

        if current_price <= effective_stop_loss:
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
        # CHECK 5: TRAILING STOP (>= 8% profit) - Dynamic ATR-based
        # ====================================================================
        trailing_check = self._check_trailing_stop(
            entry_price,
            current_price,
            highest_price,
            pnl_percent,
            pnl_amount,
            df,  # Pass df for ATR calculation
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

    def _ensure_stop_loss(
        self, symbol: str, entry_price: float, stop_loss: Optional[float], df: pd.DataFrame = None
    ) -> float:
        """
        Ensure stop loss is a valid float. Fallback to ATR-based calculation if missing.

        CRITICAL FIX: Stop loss should NEVER be None after portfolio manager validation.
        This is a safety fallback only.

        Priority:
        1. Use provided stop_loss if valid
        2. Calculate from ATR if df available
        3. Use default percentage as last resort
        """
        if isinstance(stop_loss, (int, float)) and stop_loss > 0:
            return float(stop_loss)

        # CRITICAL: This should NEVER happen after portfolio manager fix
        fallback = self._calculate_atr_based_stop_loss(entry_price, df)
        logger.error(
            f"[{symbol}] 🚨 CRITICAL: Stop loss missing/invalid ({stop_loss}). "
            f"Using ATR-based fallback {fallback:,.2f} (-{((entry_price - fallback) / entry_price * 100):.1f}%)"
        )
        return fallback

    def _calculate_atr_based_stop_loss(self, entry_price: float, df: pd.DataFrame = None) -> float:
        """
        Calculate stop loss using ATR (Average True Range) for dynamic risk management.

        Logic:
        1. If df available and has ATR: Use entry_price - (ATR * 2.0)
        2. If df available but no ATR: Calculate ATR from high/low/close
        3. If no df: Fallback to default percentage (-7%)

        Returns:
            Stop loss price (always below entry_price)
        """
        if df is not None and not df.empty:
            try:
                # Try to get ATR from dataframe
                if "atr" in df.columns:
                    atr = safe_get_latest(df, "atr", 0)
                    if atr > 0:
                        stop_loss = entry_price - (atr * 2.0)
                        # Ensure stop loss is reasonable (3% to 10% below entry)
                        min_sl = entry_price * 0.90  # Max 10% risk
                        max_sl = entry_price * 0.97  # Min 3% risk
                        stop_loss = max(min(stop_loss, max_sl), min_sl)
                        logger.info(
                            f"✅ Calculated ATR-based stop loss: {stop_loss:,.2f} "
                            f"(ATR: {atr:.2f}, Risk: {((entry_price - stop_loss) / entry_price * 100):.1f}%)"
                        )
                        return float(stop_loss)

                # Calculate ATR manually if not in dataframe
                if len(df) >= 14 and all(col in df.columns for col in ["high", "low", "close"]):
                    from src.utils.indicators import IndicatorUtils

                    atr = IndicatorUtils.get_atr(df)
                    if atr > 0:
                        stop_loss = entry_price - (atr * 2.0)
                        min_sl = entry_price * 0.90
                        max_sl = entry_price * 0.97
                        stop_loss = max(min(stop_loss, max_sl), min_sl)
                        logger.info(
                            f"✅ Calculated manual ATR-based stop loss: {stop_loss:,.2f} "
                            f"(ATR: {atr:.2f}, Risk: {((entry_price - stop_loss) / entry_price * 100):.1f}%)"
                        )
                        return float(stop_loss)
            except Exception as e:
                logger.warning(f"⚠️ Error calculating ATR-based stop loss: {e}, using percentage fallback")

        # Last resort: Use default percentage
        return self._calculate_percentage_stop_loss(entry_price)

    def _calculate_percentage_stop_loss(self, entry_price: float) -> float:
        """
        Calculate fallback stop loss using configured default percentage.
        """
        pct = self.default_stop_loss_pct if self.default_stop_loss_pct != 0 else -7.0
        if pct >= 0:
            pct = -abs(pct) if pct else -7.0
        multiplier = 1 + (pct / 100.0)
        # Ensure multiplier keeps stop loss below entry and positive
        multiplier = min(multiplier, 0.99)
        multiplier = max(multiplier, 0.01)
        fallback = entry_price * multiplier if entry_price > 0 else entry_price
        return float(fallback) if fallback and fallback > 0 else max(entry_price * 0.9, 0.01)

    def _check_profit_protection(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        pnl_percent: float,
        pnl_amount: float,
    ) -> Dict:
        """
        SIMPLIFIED: Protect profit before trailing stop activates.

        Logic (simplified from previous 3-tier system):
        - Activates when profit >= activation threshold (default 5%)
        - Protects a percentage of maximum profit achieved (default 50%)
        - Simpler than previous 3%→50%, 5%→60%, 8%→trailing logic
        - Exits if price drops below protection level

        Configurable via:
        - profit_protection_activation: When to activate (default 5%)
        - profit_protection_percent: How much of max profit to protect (default 50%)

        Returns:
            Dict with should_exit flag and decision
        """
        # Only activate if profit >= activation threshold AND below trailing activation
        activation_threshold = self.profit_protection_activation * 100
        trailing_threshold = self.trailing_activation * 100

        if pnl_percent < activation_threshold or pnl_percent >= trailing_threshold:
            return {"should_exit": False}

        # Calculate maximum profit achieved
        max_profit_pct = ((highest_price - entry_price) / entry_price) * 100

        # SIMPLIFIED: Single protection percentage instead of tiered approach
        stop_price = entry_price * (
            1 + (max_profit_pct / 100) * self.profit_protection_percent
        )

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
                    message=f"💰 PROFIT PROTECTION: Bảo vệ {self.profit_protection_percent*100:.0f}% lợi nhuận | "
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
        df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        ENHANCED: Check trailing stop with dynamic ATR-based distance.

        Kích hoạt khi lời >= 8%
        Trailing stop:
        - Dynamic mode: 2×ATR below high (adapts to volatility)
        - Fallback mode: Fixed 5% below high
        """

        # Check if trailing should be activated
        profit_from_entry = (current_price - entry_price) / entry_price

        if profit_from_entry < self.trailing_activation:
            # Chưa đủ lời để kích hoạt trailing
            return {"should_exit": False}

        # Calculate trailing stop distance
        if self.use_dynamic_trailing and df is not None and len(df) >= 14:
            # Dynamic ATR-based trailing stop
            try:
                atr = safe_get_latest(df, "atr", 0)
                if atr > 0:
                    # Trailing stop = highest_price - (2 × ATR)
                    trailing_stop_price = highest_price - (self.trailing_atr_mult * atr)
                    distance_type = f"{self.trailing_atr_mult}×ATR"
                else:
                    # Fallback if ATR invalid
                    trailing_stop_price = highest_price * (1 - self.trailing_distance)
                    distance_type = f"{self.trailing_distance*100:.0f}% (ATR unavailable)"
            except Exception as e:
                logger.warning(f"⚠️ Error calculating ATR trailing stop: {e}, using fixed %")
                trailing_stop_price = highest_price * (1 - self.trailing_distance)
                distance_type = f"{self.trailing_distance*100:.0f}% (fallback)"
        else:
            # Fixed percentage trailing stop (fallback)
            trailing_stop_price = highest_price * (1 - self.trailing_distance)
            distance_type = f"{self.trailing_distance*100:.0f}% (fixed)"

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
                    message=f"📉 TRAILING STOP ({distance_type}): Giảm {drawdown_from_high:.1f}% từ đỉnh | "
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
