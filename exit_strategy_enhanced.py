"""
Enhanced Exit Strategy với Dynamic Trailing Stops và Breakeven Stops
Cải thiện từ improved_exit_logic.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from improved_exit_logic import ExitReason, ExitDecision, ImprovedExitStrategy

logger = logging.getLogger(__name__)


@dataclass
class PartialExit:
    """Track partial exit"""

    symbol: str
    exit_date: datetime
    exit_price: float
    shares_exited: int
    remaining_shares: int
    exit_reason: ExitReason
    pnl: float
    pnl_percent: float


class EnhancedExitStrategy(ImprovedExitStrategy):
    """
    Enhanced Exit Strategy với:
    1. Dynamic trailing stops (based on volatility)
    2. Breakeven stops (move SL to entry after TP1)
    3. Better partial exit tracking
    4. Volatility-based adjustments
    """

    def __init__(
        self,
        take_profit_levels: List[float] = [0.10, 0.15, 0.25],
        stop_loss_atr_multiplier: float = 2.0,
        trailing_stop_activation: float = 0.08,
        trailing_stop_distance: float = 0.05,
        max_holding_days: int = 20,
        time_decay_threshold: float = 0.02,
        use_dynamic_trailing: bool = True,
        use_breakeven_stop: bool = True,
        breakeven_activation: float = 0.10,
    ):  # Activate breakeven after 10% profit
        super().__init__(
            take_profit_levels=take_profit_levels,
            stop_loss_atr_multiplier=stop_loss_atr_multiplier,
            trailing_stop_activation=trailing_stop_activation,
            max_holding_days=max_holding_days,
            time_decay_threshold=time_decay_threshold,
        )
        self.use_dynamic_trailing = use_dynamic_trailing
        self.use_breakeven_stop = use_breakeven_stop
        self.breakeven_activation = breakeven_activation

        # Enhanced tracking
        self.partial_exits = {}  # {symbol: [PartialExit, ...]}
        self.breakeven_activated = {}  # {symbol: bool}
        self.dynamic_trailing_stops = (
            {}
        )  # {symbol: float} - current trailing stop price

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
        Enhanced exit check với dynamic trailing và breakeven stops
        """
        if partial_exits is None:
            partial_exits = []

        # Calculate P&L
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        pnl_amount = current_price - entry_price

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
        # ENHANCED: BREAKEVEN STOP
        # ====================================================================
        if self.use_breakeven_stop and pnl_percent >= self.breakeven_activation * 100:
            if symbol not in self.breakeven_activated:
                self.breakeven_activated[symbol] = True
                logger.info(
                    f"✅ Breakeven stop activated for {symbol} at {pnl_percent:.1f}% profit"
                )

            # Move stop loss to breakeven (entry price)
            adjusted_stop_loss = max(stop_loss, entry_price)
        else:
            adjusted_stop_loss = stop_loss

        # ====================================================================
        # CHECK 1: STOP LOSS (with breakeven adjustment)
        # ====================================================================
        if current_price <= adjusted_stop_loss:
            reason = ExitReason.STOP_LOSS
            if adjusted_stop_loss >= entry_price:
                reason = ExitReason.TRAILING_STOP  # Breakeven stop hit
                message = f"🛑 BREAKEVEN STOP: {pnl_percent:+.2f}%"
            else:
                message = f"⛔ STOP LOSS: {pnl_percent:+.2f}%"

            return ExitDecision(
                should_exit=True,
                exit_reason=reason,
                exit_type="FULL",
                exit_price=current_price,
                expected_pnl=pnl_amount,
                expected_pnl_percent=pnl_percent,
                message=message,
                urgency=5,
            )

        # ====================================================================
        # ENHANCED: DYNAMIC TRAILING STOP
        # ====================================================================
        if self.use_dynamic_trailing and pnl_percent >= self.trailing_activation * 100:
            # Calculate dynamic trailing distance based on volatility
            trailing_distance = self._calculate_dynamic_trailing_distance(
                df, pnl_percent
            )

            # Update trailing stop
            if symbol not in self.dynamic_trailing_stops:
                self.dynamic_trailing_stops[symbol] = highest_price * (
                    1 - trailing_distance
                )
            else:
                # Only move trailing stop up, never down
                new_trailing = highest_price * (1 - trailing_distance)
                self.dynamic_trailing_stops[symbol] = max(
                    self.dynamic_trailing_stops[symbol], new_trailing
                )

            # Check if price hit trailing stop
            if current_price <= self.dynamic_trailing_stops[symbol]:
                drawdown_from_high = (
                    (highest_price - current_price) / highest_price
                ) * 100
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.TRAILING_STOP,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"📉 DYNAMIC TRAILING STOP: Giảm {drawdown_from_high:.1f}% từ đỉnh | P&L: {pnl_percent:+.2f}%",
                    urgency=4,
                )
        else:
            # Use standard trailing stop if dynamic not activated
            trailing_check = self._check_trailing_stop(
                entry_price, current_price, highest_price, pnl_percent, pnl_amount
            )
            if trailing_check["should_exit"]:
                return trailing_check["decision"]

        # ====================================================================
        # CHECK 2: MARKET CRASH PROTECTION
        # ====================================================================
        if market_regime and market_regime.get("regime") == "BEAR":
            if pnl_percent > 3:
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
            elif pnl_percent > -2:
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
        # CHECK 3: TAKE PROFIT BẬC THANG (with better tracking)
        # ====================================================================
        tp_check = self._check_take_profit_levels_enhanced(
            current_price,
            take_profit_targets,
            partial_exits,
            pnl_percent,
            pnl_amount,
            symbol,
        )
        if tp_check["should_exit"]:
            return tp_check["decision"]

        # ====================================================================
        # CHECK 4: ML SIGNAL SELL
        # ====================================================================
        if ml_signal and ml_signal.get("signal") == "SELL":
            confidence = ml_signal.get("confidence", 0)
            if confidence >= 60 and pnl_percent > -3:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.ML_SIGNAL_SELL,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"📉 ML SIGNAL SELL (Conf: {confidence}%): {pnl_percent:+.2f}%",
                    urgency=3,
                )

        # ====================================================================
        # CHECK 5: BEARISH REVERSAL PATTERN
        # ====================================================================
        if len(df) >= 3:
            pattern_check = self._check_reversal_pattern(df, pnl_percent, pnl_amount)
            if pattern_check["should_exit"]:
                return pattern_check["decision"]

        # ====================================================================
        # CHECK 6: SUPPORT BREAKDOWN
        # ====================================================================
        breakdown_check = self._check_support_breakdown(
            df, current_price, entry_price, pnl_percent, pnl_amount
        )
        if breakdown_check["should_exit"]:
            return breakdown_check["decision"]

        # ====================================================================
        # CHECK 7: TIME DECAY
        # ====================================================================
        if days_held >= self.max_holding_days:
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

    def _calculate_dynamic_trailing_distance(
        self, df: pd.DataFrame, pnl_percent: float
    ) -> float:
        """
        Calculate dynamic trailing stop distance based on volatility

        Higher volatility = wider trailing stop
        Higher profit = tighter trailing stop (protect more)
        """
        if len(df) < 20:
            return self.trailing_distance  # Default

        # Calculate ATR-based volatility
        if "atr" in df.columns:
            atr = df["atr"].iloc[-1]
            price = df["close"].iloc[-1]
            volatility = (atr / price) if price > 0 else 0.02
        else:
            # Fallback: use price volatility
            returns = df["close"].pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0.02

        # Base trailing distance from volatility
        base_distance = max(volatility * 2, 0.03)  # At least 3%
        base_distance = min(base_distance, 0.10)  # At most 10%

        # Adjust based on profit level
        # More profit = tighter stop (protect more)
        if pnl_percent >= 20:
            profit_adjustment = 0.7  # 30% tighter
        elif pnl_percent >= 15:
            profit_adjustment = 0.8  # 20% tighter
        elif pnl_percent >= 10:
            profit_adjustment = 0.9  # 10% tighter
        else:
            profit_adjustment = 1.0  # No adjustment

        dynamic_distance = base_distance * profit_adjustment

        return max(0.03, min(dynamic_distance, 0.08))  # Clamp between 3% and 8%

    def _check_take_profit_levels_enhanced(
        self,
        current_price: float,
        tp_targets: List[float],
        partial_exits: List[float],
        pnl_percent: float,
        pnl_amount: float,
        symbol: str,
    ) -> Dict:
        """
        Enhanced take profit check với better tracking
        """
        # Track partial exits
        if symbol not in self.partial_exits:
            self.partial_exits[symbol] = []

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

    def record_partial_exit(
        self,
        symbol: str,
        exit_price: float,
        shares_exited: int,
        remaining_shares: int,
        entry_price: float,
        exit_reason: ExitReason,
    ):
        """Record partial exit for tracking"""
        pnl = (exit_price - entry_price) * shares_exited
        pnl_percent = ((exit_price - entry_price) / entry_price) * 100

        partial_exit = PartialExit(
            symbol=symbol,
            exit_date=datetime.now(),
            exit_price=exit_price,
            shares_exited=shares_exited,
            remaining_shares=remaining_shares,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_percent=pnl_percent,
        )

        if symbol not in self.partial_exits:
            self.partial_exits[symbol] = []

        self.partial_exits[symbol].append(partial_exit)
        logger.info(
            f"📝 Recorded partial exit for {symbol}: {shares_exited} shares @ {exit_price:,.0f}"
        )

    def get_partial_exits(self, symbol: str) -> List[PartialExit]:
        """Get all partial exits for a symbol"""
        return self.partial_exits.get(symbol, [])

    def clear_position_tracking(self, symbol: str):
        """Clear tracking data when position is fully closed"""
        if symbol in self.position_highs:
            del self.position_highs[symbol]
        if symbol in self.breakeven_activated:
            del self.breakeven_activated[symbol]
        if symbol in self.dynamic_trailing_stops:
            del self.dynamic_trailing_stops[symbol]
        if symbol in self.partial_exits:
            del self.partial_exits[symbol]
