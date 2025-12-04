# -*- coding: utf-8 -*-
"""
exit_logic.py - Smart Exit Strategy v3.0 (Refactored)

Chiến lược thoát lệnh chuyên nghiệp với:
- Trailing stop động (ATR-based)
- Take profit 2 bậc (simplified từ 3 bậc)
- Profit protection
- Per-symbol performance tracking
- Full transaction cost integration

Changes v3.0:
- Fixed inconsistency giữa config và implementation
- Removed magic numbers - tất cả configurable
- Added proper error handling
- Memory leak prevention với history limits
- Safe DataFrame access utilities
- Improved documentation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

# Import constants at module level for better error handling
try:
    from src.config.constants import (
        DEFAULT_TRAILING_STOP_ACTIVATION,
        DEFAULT_TRAILING_STOP_DISTANCE,
        DEFAULT_TIME_DECAY_THRESHOLD,
        MAX_HOLDING_DAYS,
        MIN_TRADES_FOR_POOR_PERFORMER,
        POOR_PERFORMER_CONSECUTIVE_LOSSES,
        POOR_PERFORMER_WIN_RATE_THRESHOLD,
        ROUND_TRIP_COST,
        VOLUME_SURGE_THRESHOLD,
        # Adaptive holding days
        get_adaptive_holding_days,
        HOLDING_DAYS_DEFAULT,
        ADX_STRONG_TREND_THRESHOLD,
        ADX_WEAK_TREND_THRESHOLD,
    )

    ADAPTIVE_HOLDING_AVAILABLE = True
except ImportError:
    # Fallback defaults if constants not available
    DEFAULT_TRAILING_STOP_ACTIVATION = 0.05
    DEFAULT_TRAILING_STOP_DISTANCE = 0.03
    DEFAULT_TIME_DECAY_THRESHOLD = 0.02
    MAX_HOLDING_DAYS = 20
    MIN_TRADES_FOR_POOR_PERFORMER = 5
    POOR_PERFORMER_CONSECUTIVE_LOSSES = 2
    POOR_PERFORMER_WIN_RATE_THRESHOLD = 0.35
    ROUND_TRIP_COST = 0.016
    VOLUME_SURGE_THRESHOLD = 1.5
    HOLDING_DAYS_DEFAULT = 15
    ADX_STRONG_TREND_THRESHOLD = 25
    ADX_WEAK_TREND_THRESHOLD = 20
    ADAPTIVE_HOLDING_AVAILABLE = False

    def get_adaptive_holding_days(regime: str, adx: float = 20) -> int:
        """Fallback adaptive holding days function."""
        if regime == "BULL":
            return 20 if adx > 25 else 15
        elif regime == "SIDEWAYS":
            return 12 if adx > 20 else 10
        elif regime == "BEAR":
            return 8 if adx > 25 else 6
        elif regime == "HIGH_VOLATILITY":
            return 5
        return 15


from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

# Optional imports with graceful fallback
try:
    from src.market.schedule import is_trading_hour, is_trading_day, is_near_session_boundary

    TRADING_SCHEDULE_AVAILABLE = True
except ImportError:
    TRADING_SCHEDULE_AVAILABLE = False
    is_trading_hour = lambda: True
    is_trading_day = lambda: True
    is_near_session_boundary = lambda minutes=5: (False, None)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================


class ExitReason(Enum):
    """Lý do thoát lệnh - đầy đủ các trường hợp"""

    STOP_LOSS = "Stop Loss Hit"
    TRAILING_STOP = "Trailing Stop"
    TAKE_PROFIT_1 = "Take Profit 1 (Partial)"
    TAKE_PROFIT_2 = "Take Profit 2 (Full Exit)"
    TAKE_PROFIT_3 = "Take Profit 3 (Legacy - Full Exit)"  # Backward compatible
    ML_SIGNAL_SELL = "ML Signal SELL"
    TIME_DECAY = "Time Decay (Sideway quá lâu)"
    MARKET_CRASH = "Market Crash Protection"
    EMERGENCY_EXIT = "Emergency Exit (Portfolio protection)"
    REVERSAL_PATTERN = "Bearish Reversal Pattern"
    BREAKDOWN = "Support Breakdown"
    SESSION_END = "Session End Protection"
    PROFIT_PROTECTION = "Profit Protection"


@dataclass
class ExitDecision:
    """Decision kết quả thoát lệnh với đầy đủ thông tin"""

    should_exit: bool
    exit_reason: Optional[ExitReason]
    exit_type: str  # 'FULL', 'PARTIAL_50%', 'HOLD'
    exit_price: float
    expected_pnl: float
    expected_pnl_percent: float
    message: str
    urgency: int  # 1-5 (5 = exit ngay lập tức)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra info for debugging


@dataclass
class ExitConfig:
    """
    Centralized exit configuration - loại bỏ magic numbers.
    Tất cả thresholds có thể config được.
    """

    # Take Profit levels - IMPROVED v4.2 for VN market shorter cycles
    # Vietnam market characteristics:
    # - Shorter cycles (2-4 weeks typical)
    # - Higher volatility (±7% daily limit)
    # - T+2 settlement affects holding decisions
    # - Transaction costs ~1.5% round trip
    # OPTIMIZED: 4%, 8%, 15% - capture profits faster, account for costs
    # Net profit after costs: ~2.5%, ~6.5%, ~13.5%
    take_profit_levels: Tuple[float, float, float] = (0.04, 0.08, 0.15)

    # Stop Loss - IMPROVED v4.1 with transaction cost awareness
    # Must account for ~1.5% round trip cost
    stop_loss_atr_multiplier: float = 2.0
    default_stop_loss_pct: float = 0.055  # IMPROVED: 5.5% below entry (net ~4% after costs)
    min_stop_loss_pct: float = 0.035  # Min 3.5% risk (net ~2% after costs)
    max_stop_loss_pct: float = 0.075  # Max 7.5% risk (approaching daily limit)

    # Beta-adjusted stop loss - IMPROVED v4.0
    use_beta_adjusted_stops: bool = True  # Enable beta-adjusted stops
    high_beta_stop_loss_pct: float = 0.08  # 8% for beta > 1.2
    low_beta_stop_loss_pct: float = 0.05  # 5% for beta < 0.8
    high_beta_threshold: float = 1.2  # Beta threshold for wider stop
    low_beta_threshold: float = 0.8  # Beta threshold for tighter stop

    # Trailing Stop - IMPROVED v4.1 for VN market
    # VN market has ±7% daily limit, so trailing needs to be responsive
    trailing_stop_activation: float = (
        0.025  # TIGHTENED: Activate at 2.5% profit (net ~1% after costs)
    )
    trailing_stop_distance: float = 0.02  # TIGHTENED: Trail 2% below peak
    trailing_stop_atr_multiplier: float = 1.8  # Slightly tighter ATR multiplier
    use_dynamic_trailing: bool = True

    # Time Decay - IMPROVED v4.1 with T+2 awareness
    # VN market T+2 settlement means capital is tied up longer
    max_holding_days: int = MAX_HOLDING_DAYS
    time_decay_threshold: float = DEFAULT_TIME_DECAY_THRESHOLD
    t2_settlement_days: int = 2  # T+2 settlement cycle

    # Profit Protection - IMPROVED v4.1
    # Protect profits early due to VN market volatility
    profit_protection_activation: float = 0.025  # TIGHTENED: Activate at 2.5% profit
    profit_protection_percent: float = 0.65  # IMPROVED: Protect 65% of max profit

    # NEW v4.2: Session-based exit rules (Vietnam market specific)
    exit_before_lunch_if_profitable: bool = True  # Exit profitable positions before lunch
    lunch_exit_min_profit_pct: float = (
        0.025  # Min 2.5% profit to exit before lunch (net ~1% after costs)
    )
    exit_before_close_if_profitable: bool = True  # Exit before ATC if profitable
    close_exit_min_profit_pct: float = (
        0.02  # Min 2% profit to exit before close (net ~0.5% after costs)
    )

    # NEW v4.2: Friday exit rules (T+2 settlement = capital locked over weekend)
    exit_friday_if_marginal: bool = True  # Exit marginal positions on Friday
    friday_exit_min_profit_pct: float = 0.015  # Min 1.5% profit to hold over weekend
    friday_exit_max_loss_pct: float = -0.02  # Max -2% loss to hold over weekend

    # Partial Exit
    partial_exit_percent: float = 0.50  # Exit 50% at TP1

    # ML Signal
    ml_confidence_threshold: float = 60.0  # Min confidence for ML sell
    ml_min_pnl_for_exit: float = -3.0  # Min P&L to consider ML exit

    # Market Crash
    market_crash_profit_threshold: float = 3.0  # Exit if profit > 3% in crash
    market_crash_loss_threshold: float = -2.0  # Exit if loss < 2% in crash

    # Volume
    volume_surge_ratio: float = VOLUME_SURGE_THRESHOLD

    # Transaction Costs
    include_transaction_costs: bool = True
    round_trip_cost: float = ROUND_TRIP_COST

    # Per-Symbol Performance
    use_per_symbol_performance: bool = True
    poor_performer_max_holding_days: int = 15
    poor_performer_tighter_stop_pct: float = 0.03
    poor_performer_win_rate_threshold: float = 0.35
    poor_performer_consecutive_losses: int = 2


# =============================================================================
# UTILITY FUNCTIONS - Safe DataFrame Access
# =============================================================================


def safe_iloc(df: pd.DataFrame, index: int, column: Optional[str] = None) -> Optional[Any]:
    """
    Safely access DataFrame by iloc index.

    Args:
        df: DataFrame to access
        index: Index position (can be negative)
        column: Optional column name

    Returns:
        Value at position or None if invalid
    """
    try:
        if df is None or df.empty:
            return None
        if abs(index) > len(df):
            return None
        row = df.iloc[index]
        if column is not None:
            return row[column] if column in df.columns else None
        return row
    except (IndexError, KeyError, TypeError):
        return None


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with zero check."""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def calculate_trading_days(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate number of trading days between two dates.
    Uses pandas business day frequency.
    """
    try:
        from pandas.tseries.offsets import BDay

        return len(pd.date_range(start_date, end_date, freq=BDay()))
    except Exception:
        # Fallback to calendar days
        return (end_date - start_date).days


# =============================================================================
# PARTIAL EXIT TRACKER
# =============================================================================


class PartialExitTracker:
    """
    Simplified partial exit tracking with LRU cache for memory management.

    State machine:
    - State 0: No partial exits yet
    - State 1: TP1 hit (partial exited)
    - State 2: TP2 hit (100% exited) - position closed

    Memory Management:
    - Uses OrderedDict for LRU eviction of old symbols
    - MAX_TRACKED_SYMBOLS limits total tracked symbols
    - MAX_HISTORY_PER_SYMBOL limits history per symbol
    """

    MAX_HISTORY_PER_SYMBOL = 50  # Reduced from 100 for memory efficiency
    MAX_TRACKED_SYMBOLS = 200  # Max symbols to track (LRU eviction)

    def __init__(self):
        from collections import OrderedDict

        self._states: OrderedDict[str, int] = OrderedDict()
        self._exit_history: Dict[str, List[Dict]] = {}

    def _touch_symbol(self, symbol: str) -> None:
        """Move symbol to end of OrderedDict (most recently used)."""
        if symbol in self._states:
            self._states.move_to_end(symbol)

    def _evict_if_needed(self) -> None:
        """Evict oldest symbols if over limit (LRU eviction)."""
        while len(self._states) > self.MAX_TRACKED_SYMBOLS:
            # Remove oldest (first) item
            oldest_symbol, _ = self._states.popitem(last=False)
            # Also remove its history
            self._exit_history.pop(oldest_symbol, None)
            logger.debug(f"🗑️ LRU evicted symbol: {oldest_symbol}")

    def get_state(self, symbol: str) -> int:
        """Get current exit state for symbol (0, 1, or 2)"""
        if symbol in self._states:
            self._touch_symbol(symbol)  # Mark as recently used
        return self._states.get(symbol, 0)

    def record_partial_exit(
        self, symbol: str, exit_type: str, price: float, shares: int, reason: Optional[str] = None
    ) -> None:
        """Record a partial exit with LRU cache management."""
        current_state = self.get_state(symbol)

        # Update state
        if exit_type == "PARTIAL_50%":
            self._states[symbol] = 1
        elif exit_type == "FULL":
            self._states[symbol] = 2

        # Touch symbol to mark as recently used
        self._touch_symbol(symbol)

        # Evict old symbols if over limit
        self._evict_if_needed()

        # Record history
        if symbol not in self._exit_history:
            self._exit_history[symbol] = []

        self._exit_history[symbol].append(
            {
                "type": exit_type,
                "price": price,
                "shares": shares,
                "timestamp": datetime.now().isoformat(),
                "state_before": current_state,
                "state_after": self._states[symbol],
                "reason": reason,
            }
        )

        # Prevent memory leak - trim old history per symbol
        if len(self._exit_history[symbol]) > self.MAX_HISTORY_PER_SYMBOL:
            self._exit_history[symbol] = self._exit_history[symbol][-self.MAX_HISTORY_PER_SYMBOL :]

        logger.info(
            f"📊 {symbol} partial exit recorded: {exit_type} @ {price:,.0f} "
            f"(state: {current_state} → {self._states[symbol]})"
        )

    def has_partial_exit(self, symbol: str) -> bool:
        """Check if symbol has had any partial exits"""
        return self.get_state(symbol) >= 1

    def is_fully_exited(self, symbol: str) -> bool:
        """Check if symbol is fully exited"""
        return self.get_state(symbol) >= 2

    def clear_position(self, symbol: str) -> None:
        """Clear tracking for a symbol (after full exit)"""
        self._states.pop(symbol, None)
        # Keep history for analysis

    def get_exit_history(self, symbol: str) -> List[Dict]:
        """Get exit history for a symbol"""
        return self._exit_history.get(symbol, [])

    def get_summary(self) -> Dict[str, int]:
        """Get summary of all tracked positions"""
        return {
            "active_positions": len([s for s, state in self._states.items() if state < 2]),
            "partial_exits": len([s for s, state in self._states.items() if state == 1]),
            "full_exits": len([s for s, state in self._states.items() if state == 2]),
            "total_tracked": len(self._states),
        }

    def clear_all(self) -> int:
        """Clear all tracking data. Returns count of cleared items."""
        count = len(self._states)
        self._states.clear()
        self._exit_history.clear()
        return count

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for monitoring.

        Returns:
            Dict with memory stats for debugging/monitoring
        """
        total_history_entries = sum(len(h) for h in self._exit_history.values())
        return {
            "tracked_symbols": len(self._states),
            "max_tracked_symbols": self.MAX_TRACKED_SYMBOLS,
            "utilization_pct": len(self._states) / self.MAX_TRACKED_SYMBOLS * 100,
            "total_history_entries": total_history_entries,
            "avg_history_per_symbol": (
                total_history_entries / len(self._exit_history) if self._exit_history else 0
            ),
            "max_history_per_symbol": self.MAX_HISTORY_PER_SYMBOL,
            "symbols_with_history": len(self._exit_history),
        }

    def cleanup_fully_exited(self, keep_recent: int = 50) -> int:
        """
        Clean up fully exited positions to free memory.

        Args:
            keep_recent: Number of recent fully exited symbols to keep

        Returns:
            Number of symbols cleaned up
        """
        fully_exited = [s for s, state in self._states.items() if state >= 2]

        # Keep only the most recent ones
        to_remove = fully_exited[:-keep_recent] if len(fully_exited) > keep_recent else []

        for symbol in to_remove:
            self._states.pop(symbol, None)
            self._exit_history.pop(symbol, None)

        if to_remove:
            logger.info(f"🧹 Cleaned up {len(to_remove)} fully exited positions")

        return len(to_remove)


# =============================================================================
# MAIN EXIT STRATEGY CLASS
# =============================================================================


class ImprovedExitStrategy:
    """
    Chiến lược thoát lệnh nâng cao v3.0

    Features:
    1. Trailing Stop - ATR-based dynamic protection
    2. Take Profit - 2 levels (simplified)
    3. Profit Protection - Protect gains before trailing activates
    4. Time-based exit - Exit if sideway too long
    5. Market protection - Exit on market regime change
    6. Pattern recognition - Exit on reversal patterns
    7. Portfolio protection - Emergency exit
    8. Per-Symbol Performance - Tighter rules for poor performers
    9. Session boundary awareness

    All thresholds are configurable via ExitConfig.
    """

    def __init__(self, config: Optional[ExitConfig] = None, **kwargs):
        """
        Initialize exit strategy.

        Args:
            config: ExitConfig object with all settings
            **kwargs: Override individual config values
        """
        # Use provided config or create default
        self.config = config or ExitConfig()

        # Allow kwargs to override config values
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Tracking state
        self.position_highs: Dict[str, float] = {}
        self.partial_exit_tracker = PartialExitTracker()

        # Legacy compatibility aliases
        self.tp_levels = list(self.config.take_profit_levels)
        self.sl_atr_mult = self.config.stop_loss_atr_multiplier
        self.trailing_activation = self.config.trailing_stop_activation
        self.trailing_distance = self.config.trailing_stop_distance
        self.trailing_atr_mult = self.config.trailing_stop_atr_multiplier
        self.use_dynamic_trailing = self.config.use_dynamic_trailing
        self.max_holding_days = self.config.max_holding_days
        self.time_decay_threshold = self.config.time_decay_threshold
        self.default_stop_loss_pct = self.config.default_stop_loss_pct
        self.include_transaction_costs = self.config.include_transaction_costs
        self.partial_exit_percent = self.config.partial_exit_percent
        self.use_per_symbol_performance = self.config.use_per_symbol_performance

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
        partial_exits: Optional[List[float]] = None,
        check_trading_hours: bool = False,
    ) -> ExitDecision:
        """
        Kiểm tra xem có nên thoát lệnh không.

        Args:
            symbol: Mã cổ phiếu
            entry_price: Giá vào
            current_price: Giá hiện tại
            stop_loss: Stop loss ban đầu (có thể None)
            take_profit_targets: List [TP1, TP2] hoặc [TP1, TP2, TP3]
            entry_date: Ngày vào lệnh
            df: DataFrame với OHLCV + indicators
            ml_signal: Signal từ ML (optional)
            market_regime: Market regime info (optional)
            partial_exits: List các lần đã chốt lời 1 phần (optional)
            check_trading_hours: Kiểm tra giờ giao dịch

        Returns:
            ExitDecision with full details
        """
        partial_exits = partial_exits or []

        # Calculate P&L
        pnl_percent, pnl_amount = self._calculate_pnl(entry_price, current_price)

        # Calculate holding days (trading days)
        days_held = calculate_trading_days(entry_date, datetime.now())

        # Update highest price tracking
        self._update_position_high(symbol, current_price)
        highest_price = self.position_highs[symbol]

        # Ensure valid stop loss
        effective_stop_loss = self._ensure_stop_loss(symbol, entry_price, stop_loss, df)

        # Get per-symbol performance
        symbol_perf = self._get_symbol_performance(symbol)
        is_poor_performer = symbol_perf.get("is_poor_performer", False)

        # Check trading hours
        is_outside_trading_hours = False
        if check_trading_hours and TRADING_SCHEDULE_AVAILABLE:
            if not is_trading_day() or not is_trading_hour():
                is_outside_trading_hours = True
                logger.debug(f"[{symbol}] Outside trading hours - only checking stop loss")

        # Build context for checks
        ctx = {
            "symbol": symbol,
            "entry_price": entry_price,
            "current_price": current_price,
            "stop_loss": effective_stop_loss,
            "take_profit_targets": take_profit_targets,
            "entry_date": entry_date,
            "df": df,
            "ml_signal": ml_signal,
            "market_regime": market_regime,
            "partial_exits": partial_exits,
            "pnl_percent": pnl_percent,
            "pnl_amount": pnl_amount,
            "days_held": days_held,
            "highest_price": highest_price,
            "is_poor_performer": is_poor_performer,
            "symbol_perf": symbol_perf,
            "is_outside_trading_hours": is_outside_trading_hours,
        }

        # Run exit checks in priority order
        checks = [
            self._check_stop_loss,
            self._check_gap_down,  # NEW: Check gap down protection
            self._check_friday_weekend,  # NEW v4.2: Friday/weekend risk management
            self._check_breakeven_stop,  # Check breakeven stop after 1R profit
            self._check_session_boundary,
            self._check_market_crash,
            self._check_take_profit,
            self._check_profit_protection,
            self._check_trailing_stop,
            self._check_ml_signal,
            self._check_reversal_pattern,
            self._check_support_breakdown,
            self._check_time_decay,
        ]

        for check_fn in checks:
            try:
                result = check_fn(ctx)
                if result and result.should_exit:
                    return result
            except Exception as e:
                logger.warning(f"⚠️ Error in {check_fn.__name__}: {e}")
                continue

        # No exit - HOLD
        return ExitDecision(
            should_exit=False,
            exit_reason=None,
            exit_type="HOLD",
            exit_price=current_price,
            expected_pnl=pnl_amount,
            expected_pnl_percent=pnl_percent,
            message=f"✅ HOLD - P&L: {pnl_percent:+.2f}% | Days: {days_held}",
            urgency=0,
            metadata={"days_held": days_held, "highest_price": highest_price},
        )

    # =========================================================================
    # P&L CALCULATION
    # =========================================================================

    def _calculate_pnl(self, entry_price: float, current_price: float) -> Tuple[float, float]:
        """
        Calculate P&L with optional transaction costs.

        Returns:
            Tuple of (pnl_percent, pnl_amount_per_share)
        """
        if entry_price <= 0:
            return 0.0, 0.0

        gross_pnl_percent = ((current_price - entry_price) / entry_price) * 100

        if self.config.include_transaction_costs:
            transaction_cost_percent = self.config.round_trip_cost * 100
            pnl_percent = gross_pnl_percent - transaction_cost_percent
            pnl_amount = (current_price - entry_price) - (entry_price * self.config.round_trip_cost)

            logger.debug(
                f"📊 P&L: Gross {gross_pnl_percent:+.2f}% - "
                f"Costs {transaction_cost_percent:.2f}% = Net {pnl_percent:+.2f}%"
            )
        else:
            pnl_percent = gross_pnl_percent
            pnl_amount = current_price - entry_price

        return pnl_percent, pnl_amount

    def _update_position_high(self, symbol: str, current_price: float) -> None:
        """Update highest price tracking for a symbol."""
        if symbol not in self.position_highs:
            self.position_highs[symbol] = current_price
        else:
            self.position_highs[symbol] = max(self.position_highs[symbol], current_price)

    # =========================================================================
    # EXIT CHECK #1: STOP LOSS (IMPROVED v4.2 - Vietnam Market Specific)
    # =========================================================================

    def _check_stop_loss(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Check stop loss - highest priority.

        IMPROVED v4.2 for Vietnam market:
        - Floor price protection: Don't exit at floor (-7%) as it may bounce
        - Ceiling price awareness: Exit quickly if hitting ceiling then reversing
        - T+2 settlement consideration: Factor in capital lock-up
        """
        current_price = ctx["current_price"]
        stop_loss = ctx["stop_loss"]
        entry_price = ctx["entry_price"]
        symbol = ctx["symbol"]
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        is_poor_performer = ctx["is_poor_performer"]
        df = ctx.get("df")

        effective_stop = stop_loss

        # Tighter stop for poor performers
        if is_poor_performer and self.config.use_per_symbol_performance:
            tighter_stop = entry_price * (1 - self.config.poor_performer_tighter_stop_pct)
            if tighter_stop > effective_stop:
                logger.debug(
                    f"📉 {symbol}: Tightening stop from {effective_stop:,.0f} "
                    f"to {tighter_stop:,.0f} (poor performer)"
                )
                effective_stop = tighter_stop

        # IMPROVED: Check if price is at floor (Vietnam ±7% limit)
        # Don't trigger stop loss at floor - may bounce, wait for confirmation
        if df is not None and len(df) >= 2:
            try:
                prev_close = safe_iloc(df, -2, "close")
                if prev_close and prev_close > 0:
                    floor_price = prev_close * 0.93  # -7% floor for HOSE
                    # If current price is within 0.5% of floor, wait for next candle
                    if current_price <= floor_price * 1.005:
                        logger.info(
                            f"📊 {symbol}: Price at floor ({current_price:,.0f} ≈ {floor_price:,.0f}). "
                            f"Waiting for confirmation before stop loss."
                        )
                        return None
            except Exception as e:
                logger.debug(f"Floor check failed: {e}")

        if current_price <= effective_stop:
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.STOP_LOSS,
                exit_type="FULL",
                exit_price=current_price,
                expected_pnl=pnl_amount,
                expected_pnl_percent=pnl_percent,
                message=f"⛔ STOP LOSS: {pnl_percent:+.2f}%",
                urgency=5,
                metadata={"stop_loss": effective_stop, "is_poor_performer": is_poor_performer},
            )

        return None

    # =========================================================================
    # EXIT CHECK #1.5: FRIDAY/WEEKEND RISK (NEW v4.2)
    # =========================================================================

    def _check_friday_weekend(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Check Friday exit rules for Vietnam market.

        Vietnam T+2 settlement means:
        - Buy on Friday = settlement on Tuesday (capital locked 4 days)
        - Weekend gap risk is significant
        - Marginal positions should be closed before weekend

        Exit if:
        - It's Friday afternoon (after 13:00)
        - Position is marginally profitable (< 1.5%) or losing (> -2%)
        - Better to free capital for Monday opportunities
        """
        if not self.config.exit_friday_if_marginal:
            return None

        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        current_price = ctx["current_price"]
        symbol = ctx.get("symbol", "")

        try:
            from datetime import datetime
            import pytz

            vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
            now = datetime.now(vn_tz)

            # Check if Friday afternoon (after 13:00)
            if now.weekday() != 4:  # Not Friday
                return None
            if now.hour < 13:  # Before afternoon session
                return None

            min_profit = self.config.friday_exit_min_profit_pct * 100
            max_loss = self.config.friday_exit_max_loss_pct * 100

            # Exit marginal positions
            if max_loss < pnl_percent < min_profit:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.SESSION_END,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=(
                        f"📅 FRIDAY EXIT: Đóng vị thế marginal {pnl_percent:+.2f}% trước cuối tuần "
                        f"(T+2 = vốn bị khóa 4 ngày, weekend gap risk)"
                    ),
                    urgency=2,
                    metadata={
                        "trigger": "friday_weekend_risk",
                        "day": "Friday",
                        "reason": "marginal_position_weekend_risk",
                    },
                )

        except ImportError:
            logger.debug("pytz not available for Friday check")
        except Exception as e:
            logger.debug(f"Friday check failed: {e}")

        return None

    # =========================================================================
    # EXIT CHECK #1.6: GAP DOWN PROTECTION (NEW)
    # =========================================================================

    def _check_gap_down(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Check for significant gap down and exit to protect capital.

        Vietnam market gaps are significant due to:
        - Overnight news (global markets, company announcements)
        - Foreign investor sentiment changes
        - Regulatory changes

        Exit if:
        - Gap down > 3% from previous close
        - Position is in profit (protect gains)
        - Or gap down > 5% regardless of P&L (emergency exit)
        """
        df = ctx.get("df")
        if df is None or len(df) < 2:
            return None

        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        current_price = ctx["current_price"]
        symbol = ctx.get("symbol", "")

        try:
            prev_close = safe_iloc(df, -2, "close")
            today_open = safe_iloc(df, -1, "open")

            if prev_close is None or today_open is None or prev_close <= 0:
                return None

            gap_percent = (today_open - prev_close) / prev_close * 100

            # Emergency exit: Gap down > 5%
            if gap_percent < -5.0:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.BREAKDOWN,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=(
                        f"🚨 GAP DOWN EMERGENCY: {gap_percent:.1f}% gap - "
                        f"exiting to protect capital | P&L: {pnl_percent:+.2f}%"
                    ),
                    urgency=5,
                    metadata={
                        "gap_percent": gap_percent,
                        "prev_close": prev_close,
                        "today_open": today_open,
                        "trigger": "emergency_gap_down",
                    },
                )

            # Protect profits: Gap down > 3% when in profit
            if gap_percent < -3.0 and pnl_percent > 0:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.PROFIT_PROTECTION,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=(
                        f"📉 GAP DOWN PROTECTION: {gap_percent:.1f}% gap - "
                        f"protecting {pnl_percent:+.2f}% profit"
                    ),
                    urgency=4,
                    metadata={
                        "gap_percent": gap_percent,
                        "prev_close": prev_close,
                        "today_open": today_open,
                        "trigger": "profit_protection_gap_down",
                    },
                )

            # Log significant gaps for monitoring
            if gap_percent < -2.0:
                logger.info(
                    f"📊 {symbol}: Gap down {gap_percent:.1f}% detected "
                    f"(P&L: {pnl_percent:+.2f}%) - monitoring"
                )

        except Exception as e:
            logger.debug(f"Gap down check failed: {e}")

        return None

    # =========================================================================
    # EXIT CHECK #2: SESSION BOUNDARY (IMPROVED v4.1)
    # =========================================================================

    def _check_session_boundary(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Check session boundary - protect profits near session end.

        Vietnam market specific considerations:
        - Lunch break (11:30-13:00): Gap risk, news during break
        - ATC session (14:30-14:45): High volatility, institutional orders
        - Pre-lunch selling pressure (11:00-11:30)
        """
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        current_price = ctx["current_price"]
        symbol = ctx.get("symbol", "")

        try:
            # Import Vietnam market utilities
            from src.utils.vietnam_market import get_time_to_session_end, get_current_session

            minutes_remaining, session = get_time_to_session_end()

            # Check 1: Pre-lunch exit (11:00-11:30)
            # Lunch gap risk - exit profitable positions before lunch
            if session == "MORNING" and minutes_remaining <= 30:
                min_profit = self.config.lunch_exit_min_profit_pct * 100
                if self.config.exit_before_lunch_if_profitable and pnl_percent >= min_profit:
                    return ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.SESSION_END,
                        exit_type="FULL",
                        exit_price=current_price,
                        expected_pnl=pnl_amount,
                        expected_pnl_percent=pnl_percent,
                        message=(
                            f"🍽️ PRE-LUNCH EXIT: Chốt lời {pnl_percent:+.2f}% trước nghỉ trưa "
                            f"({minutes_remaining} phút còn lại) - tránh gap risk"
                        ),
                        urgency=3,
                        metadata={
                            "boundary_type": "LUNCH_BREAK",
                            "minutes_remaining": minutes_remaining,
                            "reason": "lunch_gap_protection",
                        },
                    )

            # Check 2: Pre-ATC exit (14:15-14:30)
            # Exit before ATC auction to avoid volatility
            if session == "AFTERNOON" and minutes_remaining <= 15:
                min_profit = self.config.close_exit_min_profit_pct * 100
                if self.config.exit_before_close_if_profitable and pnl_percent >= min_profit:
                    return ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.SESSION_END,
                        exit_type="FULL",
                        exit_price=current_price,
                        expected_pnl=pnl_amount,
                        expected_pnl_percent=pnl_percent,
                        message=(
                            f"⏰ PRE-ATC EXIT: Chốt lời {pnl_percent:+.2f}% trước phiên ATC "
                            f"({minutes_remaining} phút còn lại) - tránh volatility"
                        ),
                        urgency=3,
                        metadata={
                            "boundary_type": "ATC_SESSION",
                            "minutes_remaining": minutes_remaining,
                            "reason": "atc_volatility_protection",
                        },
                    )

            # Check 3: General session end protection (fallback)
            if TRADING_SCHEDULE_AVAILABLE:
                is_near_boundary, boundary_type = is_near_session_boundary(minutes=5)
                if is_near_boundary and pnl_percent >= 3 and boundary_type in ["AM_END", "PM_END"]:
                    return ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.SESSION_END,
                        exit_type="FULL",
                        exit_price=current_price,
                        expected_pnl=pnl_amount,
                        expected_pnl_percent=pnl_percent,
                        message=f"⏰ SESSION END PROTECTION: Chốt lời {pnl_percent:+.2f}% trước {boundary_type}",
                        urgency=4,
                        metadata={"boundary_type": boundary_type},
                    )

        except ImportError:
            logger.debug("Vietnam market utilities not available for session check")
        except Exception as e:
            logger.debug(f"Session boundary check failed: {e}")

        return None

    # =========================================================================
    # EXIT CHECK #3: MARKET CRASH
    # =========================================================================

    def _check_market_crash(self, ctx: Dict) -> Optional[ExitDecision]:
        """Check market crash protection."""
        market_regime = ctx.get("market_regime")
        if not market_regime or market_regime.get("regime") != "BEAR":
            return None

        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        current_price = ctx["current_price"]

        profit_threshold = self.config.market_crash_profit_threshold
        loss_threshold = self.config.market_crash_loss_threshold

        if pnl_percent > profit_threshold:
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
        elif pnl_percent > loss_threshold:
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

        return None

    # =========================================================================
    # EXIT CHECK #4: TAKE PROFIT
    # =========================================================================

    def _check_take_profit(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Check take profit levels.

        Strategy (simplified to 2 levels):
        - TP1: Partial exit (default 50%)
        - TP2: Full exit
        - TP3: Legacy support for 3-level systems
        """
        current_price = ctx["current_price"]
        tp_targets = ctx["take_profit_targets"]
        partial_exits = ctx["partial_exits"]
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]

        if not tp_targets:
            return None

        num_targets = len(tp_targets)
        num_partial_exits = len(partial_exits)

        # TP3 - Full exit (legacy 3-level support)
        if num_targets >= 3 and current_price >= tp_targets[2] and num_partial_exits < 3:
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.TAKE_PROFIT_3,
                exit_type="FULL",
                exit_price=current_price,
                expected_pnl=pnl_amount,
                expected_pnl_percent=pnl_percent,
                message=f"🎯 TP3 - FULL EXIT: {pnl_percent:+.2f}%",
                urgency=3,
            )

        # TP2 - Full exit (or partial if 3 targets)
        if num_targets >= 2 and current_price >= tp_targets[1] and num_partial_exits < 2:
            is_full_exit = num_targets == 2
            exit_type = "FULL" if is_full_exit else "PARTIAL_50%"
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.TAKE_PROFIT_2,
                exit_type=exit_type,
                exit_price=current_price,
                expected_pnl=pnl_amount,
                expected_pnl_percent=pnl_percent,
                message=f"🎯 TP2 - {'FULL EXIT' if is_full_exit else f'CHỐT {int(self.config.partial_exit_percent*100)}%'}: {pnl_percent:+.2f}%",
                urgency=3 if is_full_exit else 2,
            )

        # TP1 - Partial exit
        if num_targets >= 1 and current_price >= tp_targets[0] and num_partial_exits < 1:
            partial_pct = int(self.config.partial_exit_percent * 100)
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.TAKE_PROFIT_1,
                exit_type="PARTIAL_50%",
                exit_price=current_price,
                expected_pnl=pnl_amount,
                expected_pnl_percent=pnl_percent,
                message=f"🎯 TP1 - CHỐT {partial_pct}% position: {pnl_percent:+.2f}%",
                urgency=1,
            )

        return None

    # =========================================================================
    # EXIT CHECK #4.5: BREAKEVEN STOP (NEW)
    # =========================================================================

    def _check_breakeven_stop(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Move stop to breakeven after achieving 1R profit.

        This protects capital by ensuring no loss after reaching 1R.
        Breakeven = entry_price + transaction_costs (to cover round trip)

        Logic:
        - If max profit reached >= 1R (risk amount), move stop to breakeven
        - Breakeven = entry_price * (1 + round_trip_cost)
        - Exit if price falls back to breakeven level
        """
        entry_price = ctx["entry_price"]
        current_price = ctx["current_price"]
        stop_loss = ctx["stop_loss"]
        highest_price = ctx["highest_price"]
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        symbol = ctx["symbol"]

        # Calculate 1R (risk amount as percentage)
        risk_percent = abs((entry_price - stop_loss) / entry_price) * 100

        # Calculate max profit achieved
        max_profit_pct = ((highest_price - entry_price) / entry_price) * 100

        # Only activate if we've achieved at least 1R profit at some point
        if max_profit_pct < risk_percent:
            return None

        # Calculate breakeven price (entry + transaction costs)
        breakeven_price = entry_price * (1 + self.config.round_trip_cost)

        # Check if price has fallen back to breakeven
        if current_price <= breakeven_price and pnl_percent <= 0.5:  # Small buffer
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.PROFIT_PROTECTION,
                exit_type="FULL",
                exit_price=current_price,
                expected_pnl=pnl_amount,
                expected_pnl_percent=pnl_percent,
                message=(
                    f"🔒 BREAKEVEN STOP: Đã đạt {max_profit_pct:.1f}% (1R={risk_percent:.1f}%) "
                    f"nhưng giá quay về breakeven | P&L: {pnl_percent:+.2f}%"
                ),
                urgency=4,
                metadata={
                    "breakeven_price": breakeven_price,
                    "max_profit_pct": max_profit_pct,
                    "risk_percent": risk_percent,
                    "trigger": "breakeven_after_1R",
                },
            )

        # Log that breakeven stop is active
        if max_profit_pct >= risk_percent:
            logger.debug(
                f"🔒 {symbol}: Breakeven stop active @ {breakeven_price:,.0f} "
                f"(1R achieved: {max_profit_pct:.1f}% >= {risk_percent:.1f}%)"
            )

        return None

    # =========================================================================
    # EXIT CHECK #5: PROFIT PROTECTION
    # =========================================================================

    def _check_profit_protection(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Protect profit before trailing stop activates.

        Activates when profit >= activation threshold but < trailing threshold.
        Protects a percentage of maximum profit achieved.
        """
        entry_price = ctx["entry_price"]
        current_price = ctx["current_price"]
        highest_price = ctx["highest_price"]
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]

        activation_threshold = self.config.profit_protection_activation * 100
        trailing_threshold = self.config.trailing_stop_activation * 100

        # Only activate in the gap between profit protection and trailing
        if pnl_percent < activation_threshold or pnl_percent >= trailing_threshold:
            return None

        # Calculate max profit and protection level
        max_profit_pct = ((highest_price - entry_price) / entry_price) * 100
        protection_level = self.config.profit_protection_percent
        stop_price = entry_price * (1 + (max_profit_pct / 100) * protection_level)

        if current_price <= stop_price:
            profit_given_back = max_profit_pct - pnl_percent
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.PROFIT_PROTECTION,
                exit_type="FULL",
                exit_price=current_price,
                expected_pnl=pnl_amount,
                expected_pnl_percent=pnl_percent,
                message=(
                    f"💰 PROFIT PROTECTION: Bảo vệ {protection_level*100:.0f}% lợi nhuận | "
                    f"Max: {max_profit_pct:.1f}% → Current: {pnl_percent:.1f}% "
                    f"(Gave back {profit_given_back:.1f}%)"
                ),
                urgency=4,
                metadata={"max_profit": max_profit_pct, "stop_price": stop_price},
            )

        return None

    # =========================================================================
    # EXIT CHECK #6: TRAILING STOP
    # =========================================================================

    def _check_trailing_stop(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Check trailing stop with dynamic ATR-based distance.

        Activates when profit >= trailing_activation threshold.
        Uses ATR-based distance if available, otherwise fixed percentage.
        """
        entry_price = ctx["entry_price"]
        current_price = ctx["current_price"]
        highest_price = ctx["highest_price"]
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        df = ctx["df"]

        # Check activation
        profit_from_entry = safe_division(current_price - entry_price, entry_price, 0)
        if profit_from_entry < self.config.trailing_stop_activation:
            return None

        # Calculate trailing stop price
        trailing_stop_price, distance_type = self._calculate_trailing_stop_price(highest_price, df)

        if current_price <= trailing_stop_price:
            drawdown_from_high = ((highest_price - current_price) / highest_price) * 100
            return ExitDecision(
                should_exit=True,
                exit_reason=ExitReason.TRAILING_STOP,
                exit_type="FULL",
                exit_price=current_price,
                expected_pnl=pnl_amount,
                expected_pnl_percent=pnl_percent,
                message=(
                    f"📉 TRAILING STOP ({distance_type}): "
                    f"Giảm {drawdown_from_high:.1f}% từ đỉnh | P&L: {pnl_percent:+.2f}%"
                ),
                urgency=4,
                metadata={
                    "trailing_stop": trailing_stop_price,
                    "highest_price": highest_price,
                    "drawdown": drawdown_from_high,
                },
            )

        return None

    def _calculate_trailing_stop_price(
        self, highest_price: float, df: Optional[pd.DataFrame]
    ) -> Tuple[float, str]:
        """Calculate trailing stop price using ATR or fixed percentage."""
        if self.config.use_dynamic_trailing and df is not None and len(df) >= 14:
            try:
                atr = safe_get_latest(df, "atr", 0)
                if atr > 0:
                    stop_price = highest_price - (self.config.trailing_stop_atr_multiplier * atr)
                    return stop_price, f"{self.config.trailing_stop_atr_multiplier}×ATR"
            except Exception as e:
                logger.debug(f"ATR trailing calculation failed: {e}")

        # Fallback to fixed percentage
        stop_price = highest_price * (1 - self.config.trailing_stop_distance)
        return stop_price, f"{self.config.trailing_stop_distance*100:.0f}%"

    # =========================================================================
    # EXIT CHECK #7: ML SIGNAL
    # =========================================================================

    def _check_ml_signal(self, ctx: Dict) -> Optional[ExitDecision]:
        """Check ML sell signal with volume confirmation."""
        ml_signal = ctx.get("ml_signal")
        if not ml_signal or ml_signal.get("signal") != "SELL":
            return None

        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        current_price = ctx["current_price"]
        df = ctx["df"]

        confidence = ml_signal.get("confidence", 0)

        # Check thresholds
        if confidence < self.config.ml_confidence_threshold:
            return None
        if pnl_percent <= self.config.ml_min_pnl_for_exit:
            return None

        # Volume confirmation
        volume_confirmed = self._check_volume_for_exit(df)
        urgency = 4 if volume_confirmed else 3

        message = f"📉 ML SIGNAL SELL (Conf: {confidence}%): {pnl_percent:+.2f}%"
        if volume_confirmed:
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
            metadata={"ml_confidence": confidence, "volume_confirmed": volume_confirmed},
        )

    def _check_volume_for_exit(self, df: pd.DataFrame) -> bool:
        """Check if volume confirms exit signal (high volume on down day)."""
        if df is None or len(df) < 20:
            return False

        try:
            current_volume = safe_get_latest(df, "volume", 0)
            avg_volume = safe_rolling_operation(df, "volume", 20, "mean", 0)

            volume_ratio = safe_division(current_volume, avg_volume, 0)

            # Check if down day
            latest_close = safe_get_latest(df, "close", 0)
            prev_close = safe_iloc(df, -2, "close")
            if prev_close is None:
                prev_close = latest_close

            is_down_day = latest_close < prev_close

            return volume_ratio >= self.config.volume_surge_ratio and is_down_day
        except Exception:
            return False

    # =========================================================================
    # EXIT CHECK #8: REVERSAL PATTERN
    # =========================================================================

    def _check_reversal_pattern(self, ctx: Dict) -> Optional[ExitDecision]:
        """Check bearish reversal patterns (engulfing, shooting star, distribution volume)."""
        df = ctx["df"]
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]

        # Only check if profitable
        if pnl_percent <= 0 or df is None or len(df) < 3:
            return None

        latest = safe_iloc(df, -1)
        prev = safe_iloc(df, -2)

        if latest is None or prev is None:
            return None

        try:
            # NEW: Distribution Volume Check
            # High volume + price down = institutional selling
            distribution = self._check_distribution_volume(df, latest)
            if distribution["is_distribution"]:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.REVERSAL_PATTERN,
                    exit_type="FULL",
                    exit_price=latest["close"],
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=(
                        f"📊 DISTRIBUTION VOLUME: Volume {distribution['volume_ratio']:.1f}x avg "
                        f"+ price down - institutional selling | P&L: {pnl_percent:+.2f}%"
                    ),
                    urgency=4,
                    metadata={
                        "pattern": "distribution_volume",
                        "volume_ratio": distribution["volume_ratio"],
                    },
                )

            # Bearish Engulfing
            if self._is_bearish_engulfing(prev, latest):
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.REVERSAL_PATTERN,
                    exit_type="FULL",
                    exit_price=latest["close"],
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"🔴 BEARISH ENGULFING: Chốt lời sớm {pnl_percent:+.2f}%",
                    urgency=3,
                    metadata={"pattern": "bearish_engulfing"},
                )

            # Shooting Star
            if self._is_shooting_star(latest):
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.REVERSAL_PATTERN,
                    exit_type="PARTIAL_50%",
                    exit_price=latest["close"],
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"⭐ SHOOTING STAR: Chốt 50% {pnl_percent:+.2f}%",
                    urgency=2,
                    metadata={"pattern": "shooting_star"},
                )
        except (KeyError, TypeError):
            pass

        return None

    def _check_distribution_volume(self, df: pd.DataFrame, latest: pd.Series) -> Dict:
        """
        Check for distribution volume (institutional selling).

        Distribution = High volume + Price down
        This often signals smart money exiting positions.

        Args:
            df: DataFrame with OHLCV
            latest: Latest candle

        Returns:
            Dict with is_distribution, volume_ratio
        """
        try:
            if "volume" not in df.columns or len(df) < 20:
                return {"is_distribution": False, "volume_ratio": 1.0}

            avg_volume = df["volume"].tail(20).mean()
            if avg_volume <= 0:
                return {"is_distribution": False, "volume_ratio": 1.0}

            current_volume = latest.get("volume", 0)
            volume_ratio = current_volume / avg_volume

            # Price down check
            price_down = latest.get("close", 0) < latest.get("open", 0)

            # Distribution: Volume > 2x average AND price down
            is_distribution = volume_ratio >= self.config.volume_surge_ratio and price_down

            return {
                "is_distribution": is_distribution,
                "volume_ratio": volume_ratio,
                "price_down": price_down,
            }
        except Exception:
            return {"is_distribution": False, "volume_ratio": 1.0}

    def _is_bearish_engulfing(self, prev: pd.Series, latest: pd.Series) -> bool:
        """Check for bearish engulfing pattern."""
        try:
            prev_bullish = prev["close"] > prev["open"]
            latest_bearish = latest["close"] < latest["open"]
            engulfs = latest["close"] < prev["open"] and latest["open"] > prev["close"]
            return prev_bullish and latest_bearish and engulfs
        except (KeyError, TypeError):
            return False

    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """Check for shooting star pattern."""
        try:
            body = abs(candle["close"] - candle["open"])
            if body == 0:
                return False
            upper_shadow = candle["high"] - max(candle["close"], candle["open"])
            lower_shadow = min(candle["close"], candle["open"]) - candle["low"]
            return upper_shadow > body * 2 and lower_shadow < body * 0.5
        except (KeyError, TypeError):
            return False

    # =========================================================================
    # EXIT CHECK #9: SUPPORT BREAKDOWN
    # =========================================================================

    def _check_support_breakdown(self, ctx: Dict) -> Optional[ExitDecision]:
        """Check if price breaks support with volume confirmation."""
        df = ctx["df"]
        current_price = ctx["current_price"]
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]

        if df is None or len(df) < 20:
            return None

        try:
            # Support = low of last 20 days (excluding today)
            support = df["low"].iloc[-20:-1].min()

            if current_price < support:
                # Check volume surge
                current_volume = safe_get_latest(df, "volume", 0)
                avg_volume = safe_rolling_operation(df, "volume", 20, "mean", 0)
                volume_surge = current_volume > avg_volume * self.config.volume_surge_ratio

                if volume_surge:
                    return ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.BREAKDOWN,
                        exit_type="FULL",
                        exit_price=current_price,
                        expected_pnl=pnl_amount,
                        expected_pnl_percent=pnl_percent,
                        message=f"📉 SUPPORT BREAKDOWN (Volume confirmed): {pnl_percent:+.2f}%",
                        urgency=4,
                        metadata={"support": support, "volume_surge": True},
                    )
        except Exception:
            pass

        return None

    # =========================================================================
    # EXIT CHECK #10: TIME DECAY
    # =========================================================================

    def _check_time_decay(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Check time decay - exit if holding too long with low profit.

        Adaptive holding period based on:
        1. Market regime (BULL/SIDEWAYS/BEAR)
        2. Trend strength (ADX)
        3. Per-symbol performance
        """
        days_held = ctx["days_held"]
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        current_price = ctx["current_price"]
        market_regime = ctx.get("market_regime")
        df = ctx["df"]
        is_poor_performer = ctx["is_poor_performer"]
        symbol = ctx["symbol"]

        # Calculate adaptive max holding days
        adaptive_max_days = self._calculate_adaptive_holding_days(
            market_regime, df, is_poor_performer
        )

        if days_held >= adaptive_max_days:
            threshold_pct = self.config.time_decay_threshold * 100
            if pnl_percent < threshold_pct:
                regime_note = ""
                if market_regime:
                    regime_note = f" ({market_regime.get('regime', 'UNKNOWN')} market)"

                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.TIME_DECAY,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=(
                        f"⏰ SIDEWAY QUÁ LÂU ({days_held} ngày, limit: {adaptive_max_days})"
                        f"{regime_note}: {pnl_percent:+.2f}%"
                    ),
                    urgency=2,
                    metadata={
                        "days_held": days_held,
                        "adaptive_max_days": adaptive_max_days,
                        "is_poor_performer": is_poor_performer,
                    },
                )

        return None

    def _calculate_adaptive_holding_days(
        self,
        market_regime: Optional[Dict],
        df: Optional[pd.DataFrame],
        is_poor_performer: bool,
    ) -> int:
        """
        Calculate adaptive max holding days based on conditions.

        Uses centralized get_adaptive_holding_days() from constants.py
        for consistent behavior across the codebase.

        Vietnam market characteristics:
        - BULL: 15-20 days (can hold longer in strong trends)
        - SIDEWAYS: 10-12 days (shorter cycles)
        - BEAR: 6-8 days (exit fast)
        - HIGH_VOLATILITY: 5 days (very short)
        """
        # Get regime and ADX
        regime = "SIDEWAYS"  # Default
        adx = 20.0  # Default ADX

        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")

        if df is not None and len(df) > 0 and "adx" in df.columns:
            try:
                adx = safe_get_latest(df, "adx", 20.0)
            except Exception:
                pass

        # Use centralized function from constants.py
        adaptive_max_days = get_adaptive_holding_days(regime, adx)

        # Shorter holding for poor performers
        if is_poor_performer and self.config.use_per_symbol_performance:
            adaptive_max_days = min(adaptive_max_days, self.config.poor_performer_max_holding_days)

        logger.debug(
            f"📅 Adaptive holding: {adaptive_max_days} days "
            f"(regime={regime}, ADX={adx:.1f}, poor_performer={is_poor_performer})"
        )

        return adaptive_max_days

    # =========================================================================
    # STOP LOSS HELPERS
    # =========================================================================

    def _ensure_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: Optional[float],
        df: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Ensure stop loss is valid. Fallback to ATR-based or percentage if missing.

        IMPROVED v4.0: Uses beta-adjusted stop loss when enabled.
        - High beta stocks (>1.2): Wider stop loss (-8%) to avoid premature exit
        - Normal beta (0.8-1.2): Standard stop loss (-6%)
        - Low beta stocks (<0.8): Tighter stop loss (-5%) for capital efficiency
        """
        if isinstance(stop_loss, (int, float)) and stop_loss > 0:
            # If beta-adjusted stops enabled, adjust the provided stop loss
            if self.config.use_beta_adjusted_stops and df is not None:
                beta_adjusted = self._calculate_beta_adjusted_stop_loss(symbol, entry_price, df)
                if beta_adjusted is not None:
                    # Use the wider of the two stops (more conservative)
                    return min(float(stop_loss), beta_adjusted)
            return float(stop_loss)

        # Calculate fallback with beta adjustment
        if self.config.use_beta_adjusted_stops and df is not None:
            beta_adjusted = self._calculate_beta_adjusted_stop_loss(symbol, entry_price, df)
            if beta_adjusted is not None:
                logger.info(
                    f"[{symbol}] 📊 Using beta-adjusted stop loss: {beta_adjusted:,.2f} "
                    f"(-{((entry_price - beta_adjusted) / entry_price * 100):.1f}%)"
                )
                return beta_adjusted

        # Fallback to ATR-based
        fallback = self._calculate_atr_based_stop_loss(entry_price, df)
        logger.warning(
            f"[{symbol}] ⚠️ Stop loss missing/invalid ({stop_loss}). "
            f"Using fallback {fallback:,.2f} "
            f"(-{((entry_price - fallback) / entry_price * 100):.1f}%)"
        )
        return fallback

    def _calculate_beta_adjusted_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        df: pd.DataFrame,
    ) -> Optional[float]:
        """
        Calculate stop loss adjusted by stock's beta (volatility vs market).

        IMPROVED v4.0:
        - High beta stocks (>1.2): Wider stop loss (-8%) to avoid premature exit
        - Normal beta (0.8-1.2): Standard stop loss (-6%)
        - Low beta stocks (<0.8): Tighter stop loss (-5%) for capital efficiency

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            df: DataFrame with price data

        Returns:
            Beta-adjusted stop loss price, or None if beta cannot be calculated
        """
        try:
            # Calculate beta
            beta = self._calculate_stock_beta(df, symbol)

            if beta is None:
                return None

            # Determine stop loss percentage based on beta
            if beta > self.config.high_beta_threshold:
                # High beta: Wider stop
                stop_loss_pct = abs(self.config.high_beta_stop_loss_pct) / 100
                reason = f"high beta ({beta:.2f})"
            elif beta < self.config.low_beta_threshold:
                # Low beta: Tighter stop
                stop_loss_pct = abs(self.config.low_beta_stop_loss_pct) / 100
                reason = f"low beta ({beta:.2f})"
            else:
                # Normal beta: Standard stop
                stop_loss_pct = abs(self.config.default_stop_loss_pct) / 100
                reason = f"normal beta ({beta:.2f})"

            stop_loss = entry_price * (1 - stop_loss_pct)

            logger.debug(
                f"[{symbol}] Beta-adjusted stop: {stop_loss:,.0f} "
                f"(-{stop_loss_pct*100:.1f}%) - {reason}"
            )

            return stop_loss

        except Exception as e:
            logger.debug(f"Beta-adjusted stop loss calculation failed for {symbol}: {e}")
            return None

    def _calculate_stock_beta(
        self,
        df: pd.DataFrame,
        symbol: str,
        lookback: int = 60,
    ) -> Optional[float]:
        """
        Calculate stock's beta vs VNINDEX.

        Beta = Covariance(stock, market) / Variance(market)

        Args:
            df: DataFrame with stock price data
            symbol: Stock symbol (for logging)
            lookback: Number of days for calculation

        Returns:
            Beta value, or None if calculation fails
        """
        try:
            # Get VNINDEX data
            try:
                from src.data.vnindex_cache import get_cached_vnindex

                vnindex_df = get_cached_vnindex(lookback=lookback + 10)
            except ImportError:
                logger.debug("VNINDEX cache not available for beta calculation")
                return None

            if vnindex_df is None or len(vnindex_df) < lookback // 2:
                return None

            # Calculate returns
            if len(df) < lookback // 2:
                return None

            stock_returns = df["close"].pct_change().tail(lookback).dropna()
            market_returns = vnindex_df["close"].pct_change().tail(lookback).dropna()

            # Align data
            min_len = min(len(stock_returns), len(market_returns))
            if min_len < 20:
                return None

            stock_returns = stock_returns.tail(min_len)
            market_returns = market_returns.tail(min_len)

            # Calculate beta
            covariance = stock_returns.cov(market_returns)
            market_variance = market_returns.var()

            if market_variance > 0:
                beta = covariance / market_variance
                return float(beta)

            return None

        except Exception as e:
            logger.debug(f"Beta calculation failed for {symbol}: {e}")
            return None

    def _calculate_atr_based_stop_loss(
        self, entry_price: float, df: Optional[pd.DataFrame] = None
    ) -> float:
        """Calculate stop loss using ATR or fallback to percentage."""
        if df is not None and not df.empty:
            try:
                if "atr" in df.columns:
                    atr = safe_get_latest(df, "atr", 0)
                    if atr > 0:
                        stop_loss = entry_price - (atr * self.config.stop_loss_atr_multiplier)
                        # Clamp to reasonable range
                        min_sl = entry_price * (1 - self.config.max_stop_loss_pct)
                        max_sl = entry_price * (1 - self.config.min_stop_loss_pct)
                        return max(min(stop_loss, max_sl), min_sl)

                # Try manual ATR calculation
                if len(df) >= 14 and all(col in df.columns for col in ["high", "low", "close"]):
                    try:
                        from src.utils.indicators import IndicatorUtils

                        atr = IndicatorUtils.get_atr(df)
                        if atr > 0:
                            stop_loss = entry_price - (atr * self.config.stop_loss_atr_multiplier)
                            min_sl = entry_price * (1 - self.config.max_stop_loss_pct)
                            max_sl = entry_price * (1 - self.config.min_stop_loss_pct)
                            return max(min(stop_loss, max_sl), min_sl)
                    except ImportError:
                        pass
            except Exception as e:
                logger.debug(f"ATR stop loss calculation failed: {e}")

        # Fallback to percentage
        return entry_price * (1 - self.config.default_stop_loss_pct)

    # =========================================================================
    # PER-SYMBOL PERFORMANCE
    # =========================================================================

    def _get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Get historical performance for a symbol from circuit breaker."""
        default_result = {
            "is_poor_performer": False,
            "win_rate": 0.5,
            "total_trades": 0,
            "consecutive_losses": 0,
            "reason": "No data",
        }

        if not self.config.use_per_symbol_performance:
            return default_result

        try:
            from src.risk.per_symbol_circuit_breaker import get_per_symbol_circuit_breaker

            cb = get_per_symbol_circuit_breaker()
            stats = cb.get_symbol_stats(symbol)

            if stats is None:
                return default_result

            # Determine if poor performer
            is_poor = False
            reason = ""

            win_rate_threshold = self.config.poor_performer_win_rate_threshold
            consec_loss_threshold = self.config.poor_performer_consecutive_losses
            min_trades_threshold = MIN_TRADES_FOR_POOR_PERFORMER

            if stats.total_trades >= min_trades_threshold and stats.win_rate < win_rate_threshold:
                is_poor = True
                reason = f"Low win rate: {stats.win_rate:.1%} (min {min_trades_threshold} trades)"

            if stats.consecutive_losses >= consec_loss_threshold:
                is_poor = True
                reason = f"Consecutive losses: {stats.consecutive_losses}"

            if getattr(stats, "blocked", False):
                is_poor = True
                reason = getattr(stats, "blocked_reason", "Blocked")

            return {
                "is_poor_performer": is_poor,
                "win_rate": stats.win_rate,
                "total_trades": stats.total_trades,
                "total_wins": getattr(stats, "total_wins", 0),
                "total_losses": getattr(stats, "total_losses", 0),
                "consecutive_losses": stats.consecutive_losses,
                "blocked": getattr(stats, "blocked", False),
                "reason": reason,
            }
        except ImportError:
            logger.debug("Per-symbol circuit breaker not available")
            return default_result
        except Exception as e:
            logger.debug(f"Error getting symbol performance: {e}")
            return default_result

    # =========================================================================
    # POSITION TRACKING MANAGEMENT
    # =========================================================================

    def clear_position_tracking(self, symbol: str) -> None:
        """Clear tracking for a symbol after position closed."""
        removed = self.position_highs.pop(symbol, None)
        self.partial_exit_tracker.clear_position(symbol)
        if removed is not None:
            logger.debug(f"✅ Cleared position tracking for {symbol}")

    def get_tracked_positions(self) -> List[str]:
        """Get list of symbols being tracked."""
        return list(self.position_highs.keys())

    def clear_all_tracking(self) -> int:
        """Clear all tracking data. Returns count of cleared items."""
        count = len(self.position_highs)
        self.position_highs.clear()
        self.partial_exit_tracker.clear_all()
        logger.info(f"🧹 Cleared tracking for {count} positions")
        return count

    # =========================================================================
    # MESSAGE FORMATTING
    # =========================================================================

    def format_exit_message(
        self, symbol: str, decision: ExitDecision, use_html: bool = True
    ) -> str:
        """
        Format exit decision for Telegram notification.

        Args:
            symbol: Stock symbol
            decision: Exit decision object
            use_html: Use HTML formatting (safer) instead of Markdown

        Returns:
            Formatted message string
        """
        urgency_emoji = {5: "🚨🚨🚨", 4: "🚨🚨", 3: "⚠️", 2: "💡", 1: "ℹ️", 0: "✅"}

        if use_html:
            return self._format_exit_message_html(symbol, decision, urgency_emoji)
        return self._format_exit_message_markdown(symbol, decision, urgency_emoji)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        result = str(text)
        result = result.replace("&", "&amp;")
        result = result.replace("<", "&lt;")
        result = result.replace(">", "&gt;")
        return result

    def _format_exit_message_html(
        self, symbol: str, decision: ExitDecision, urgency_emoji: Dict[int, str]
    ) -> str:
        """Format exit message using HTML."""
        safe_message = self._escape_html(decision.message)
        safe_symbol = self._escape_html(symbol)

        if not decision.should_exit:
            return f"✅ <b>{safe_symbol}</b> - HOLD\n{safe_message}"

        emoji = urgency_emoji.get(decision.urgency, "⚠️")
        exit_reason_text = self._escape_html(
            decision.exit_reason.value if decision.exit_reason else "Unknown"
        )
        safe_exit_type = self._escape_html(str(decision.exit_type))

        msg = f"{emoji} <b>{safe_symbol}</b> - EXIT SIGNAL\n\n"
        msg += f"📍 <b>Exit Type:</b> {safe_exit_type}\n"
        msg += f"🎯 <b>Reason:</b> {exit_reason_text}\n"
        msg += f"💰 <b>Exit Price:</b> {decision.exit_price:,.0f} VNĐ\n"
        msg += f"📊 <b>P&amp;L:</b> {decision.expected_pnl_percent:+.2f}%\n"
        msg += f"⚡ <b>Urgency:</b> {decision.urgency}/5\n\n"
        msg += f"💬 {safe_message}"

        return msg

    def _format_exit_message_markdown(
        self, symbol: str, decision: ExitDecision, urgency_emoji: Dict[int, str]
    ) -> str:
        """Format exit message using Markdown v1 (legacy)."""
        safe_message = decision.message.replace("_", "\\_") if decision.message else ""
        safe_symbol = symbol.replace("_", "\\_")

        if not decision.should_exit:
            return f"✅ *{safe_symbol}* - HOLD\n{safe_message}"

        emoji = urgency_emoji.get(decision.urgency, "⚠️")
        exit_reason_text = (
            decision.exit_reason.value.replace("_", "\\_") if decision.exit_reason else "Unknown"
        )
        safe_exit_type = str(decision.exit_type).replace("_", "\\_")

        msg = f"{emoji} *{safe_symbol}* - EXIT SIGNAL\n\n"
        msg += f"📍 *Exit Type:* {safe_exit_type}\n"
        msg += f"🎯 *Reason:* {exit_reason_text}\n"
        msg += f"💰 *Exit Price:* {decision.exit_price:,.0f} VNĐ\n"
        msg += f"📊 *P&L:* {decision.expected_pnl_percent:+.2f}%\n"
        msg += f"⚡ *Urgency:* {decision.urgency}/5\n\n"
        msg += f"💬 {safe_message}"

        return msg


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_exit_strategy(config: Optional[ExitConfig] = None, **kwargs) -> ImprovedExitStrategy:
    """
    Factory function to create exit strategy with optional config.

    Args:
        config: ExitConfig object
        **kwargs: Override individual config values

    Returns:
        Configured ImprovedExitStrategy instance
    """
    return ImprovedExitStrategy(config=config, **kwargs)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TESTING IMPROVED EXIT STRATEGY v3.0")
    print("=" * 70 + "\n")

    # Test with mock data
    import numpy as np

    # Create mock DataFrame
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "open": np.random.uniform(80000, 90000, 100),
            "high": np.random.uniform(85000, 95000, 100),
            "low": np.random.uniform(75000, 85000, 100),
            "close": np.random.uniform(80000, 90000, 100),
            "volume": np.random.uniform(100000, 500000, 100),
            "atr": np.random.uniform(1000, 3000, 100),
            "adx": np.random.uniform(15, 40, 100),
        },
        index=dates,
    )

    # Test scenarios
    test_cases = [
        {
            "name": "Stop Loss Hit",
            "entry_price": 85000,
            "current_price": 78000,
            "stop_loss": 80000,
        },
        {
            "name": "Take Profit 1",
            "entry_price": 80000,
            "current_price": 90000,
            "stop_loss": 76000,
        },
        {
            "name": "Hold Position",
            "entry_price": 80000,
            "current_price": 82000,
            "stop_loss": 76000,
        },
    ]

    # Create strategy
    strategy = create_exit_strategy()

    for tc in test_cases:
        print(f"\n📋 Test: {tc['name']}")
        print("-" * 40)

        decision = strategy.check_exit(
            symbol="TEST",
            entry_price=tc["entry_price"],
            current_price=tc["current_price"],
            stop_loss=tc["stop_loss"],
            take_profit_targets=[tc["entry_price"] * 1.12, tc["entry_price"] * 1.20],
            entry_date=datetime.now() - timedelta(days=5),
            df=df,
        )

        print(f"Should Exit: {decision.should_exit}")
        print(f"Reason: {decision.exit_reason}")
        print(f"Type: {decision.exit_type}")
        print(f"P&L: {decision.expected_pnl_percent:+.2f}%")
        print(f"Message: {decision.message}")

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)
