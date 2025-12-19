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
    MAX_HOLDING_DAYS = 15  # TIGHTENED: was 20
    MIN_TRADES_FOR_POOR_PERFORMER = 5
    POOR_PERFORMER_CONSECUTIVE_LOSSES = 2
    POOR_PERFORMER_WIN_RATE_THRESHOLD = 0.35
    ROUND_TRIP_COST = 0.016
    VOLUME_SURGE_THRESHOLD = 1.5
    HOLDING_DAYS_DEFAULT = 10  # TIGHTENED: was 15
    ADX_STRONG_TREND_THRESHOLD = 25
    ADX_WEAK_TREND_THRESHOLD = 20
    ADAPTIVE_HOLDING_AVAILABLE = False

    def get_adaptive_holding_days(regime: str, adx: float = 20) -> int:
        """Fallback adaptive holding days function - TIGHTENED for VN market."""
        if regime == "BULL":
            return 15 if adx > 25 else 12  # was 20/15
        elif regime == "SIDEWAYS":
            return 10 if adx > 20 else 8  # was 12/10
        elif regime == "BEAR":
            return 6 if adx > 25 else 5  # was 8/6
        elif regime == "HIGH_VOLATILITY":
            return 4  # was 5
        return 10  # was 15


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

# NEW v9.2: Vietnamese News Sentiment Integration for Exit
VN_NEWS_SENTIMENT_AVAILABLE = False
try:
    from src.sentiment.vn_news_sentiment_integration import (
        get_news_sentiment_integration,
        VNNewsSentimentIntegration,
    )

    VN_NEWS_SENTIMENT_AVAILABLE = True
except ImportError:
    pass

# NEW v10.2: Odd-Lot Trading Integration
ODD_LOT_AVAILABLE = False
try:
    from src.utils.odd_lot_handler import get_odd_lot_handler, OddLotHandler

    ODD_LOT_AVAILABLE = True
except ImportError:
    try:
        from src.strategies.special_instruments import get_odd_lot_logic, OddLotTradingLogic

        ODD_LOT_AVAILABLE = True
    except ImportError:
        pass

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
    FLOOR_BOUNCE_TIMEOUT = "Floor Bounce Timeout (No recovery)"  # NEW v6.0
    PANIC_SELLING = "Panic Selling (High volume at floor)"  # NEW v6.0
    GAP_DOWN_EMERGENCY = "Gap Down Emergency Exit"  # NEW v6.0
    NEWS_SENTIMENT_FORCE_EXIT = "Negative News Sentiment (Force Exit)"  # NEW v9.2
    ODD_LOT_CLEANUP = "Odd-Lot Cleanup (Sell remaining <100 shares)"  # NEW v10.2


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

    RELAXED v10.3: Lower TP levels for faster profit taking in VN market.

    Key improvements:
    - TP levels lowered for more frequent exits
    - Stop loss relaxed to 5% for more room
    - Dynamic TP based on market regime
    - Transaction cost aware calculations
    """

    # Take Profit levels - RELAXED v10.3 for faster profit taking
    # Vietnam market characteristics:
    # - Shorter cycles (2-4 weeks typical)
    # - Higher volatility (±7% daily limit)
    # - T+2 settlement affects holding decisions
    # - Transaction costs ~1.5% round trip
    #
    # R:R Calculation (v10.3 RELAXED):
    # - Stop Loss: 5% gross = ~3.5% net loss after costs
    # - TP1: 5% gross = ~3.5% net profit → R:R = 1.0 (quick exit)
    # - TP2: 8% gross = ~6.5% net profit → R:R = 1.85 ✓
    # - TP3: 12% gross = ~10.5% net profit → R:R = 3.0 ✓
    take_profit_levels: Tuple[float, float, float] = (0.05, 0.08, 0.12)  # RELAXED

    # RELAXED v10.3: Dynamic TP levels by market regime
    # Lower targets to lock profits faster
    use_dynamic_tp_levels: bool = True
    tp_levels_bull: Tuple[float, float, float] = (
        0.07,
        0.12,
        0.20,
    )  # RELAXED from (0.10, 0.18, 0.28)
    tp_levels_bear: Tuple[float, float, float] = (
        0.03,
        0.05,
        0.08,
    )  # RELAXED from (0.05, 0.08, 0.12)
    tp_levels_sideways: Tuple[float, float, float] = (
        0.05,
        0.08,
        0.12,
    )  # RELAXED from (0.07, 0.12, 0.18)
    tp_levels_high_volatility: Tuple[float, float, float] = (
        0.04,
        0.07,
        0.10,
    )  # RELAXED from (0.06, 0.10, 0.15)

    # Stop Loss - RELAXED v10.3 for more room to breathe
    # Wider stops to avoid premature stop-outs
    stop_loss_atr_multiplier: float = 2.0  # RELAXED from 1.8
    default_stop_loss_pct: float = 0.05  # RELAXED: 5% below entry (was 4%)
    min_stop_loss_pct: float = 0.03  # RELAXED: Min 3% risk (was 2.5%)
    max_stop_loss_pct: float = 0.07  # RELAXED: Max 7% risk = VN limit (was 6%)

    # Beta-adjusted stop loss - RELAXED v10.3
    use_beta_adjusted_stops: bool = True
    high_beta_stop_loss_pct: float = 0.07  # RELAXED: 7% for beta > 1.2 (was 6%)
    low_beta_stop_loss_pct: float = 0.04  # RELAXED: 4% for beta < 0.8 (was 3%)
    high_beta_threshold: float = 1.2
    low_beta_threshold: float = 0.8

    # RELAXED v10.3: Minimum R:R enforcement
    min_risk_reward_ratio: float = 1.0  # RELAXED: was 1.5 - allow smaller R:R
    target_risk_reward_ratio: float = 1.5  # RELAXED: was 2.0

    # Trailing Stop - RELAXED v10.3 for VN market
    # Wider trailing to avoid whipsaws
    trailing_stop_activation: float = 0.05  # RELAXED: Activate at 5% profit (was 4%)
    trailing_stop_distance: float = 0.03  # RELAXED: Trail 3% below peak (was 2.5%)
    trailing_stop_atr_multiplier: float = 2.0  # RELAXED: ATR multiplier (was 1.8)
    use_dynamic_trailing: bool = True

    # v10.3: Accelerated trailing after TP1 - RELAXED
    trailing_after_tp1_distance: float = 0.02  # RELAXED: 2% after TP1 (was 1.5%)
    trailing_acceleration_factor: float = 0.90  # RELAXED: Tighten by 10% (was 15%)

    # NEW v10.2: Volatility-aware trailing
    # Widen trailing stop in high volatility to avoid premature exits
    use_volatility_adjusted_trailing: bool = True
    high_vol_trailing_multiplier: float = 1.5  # 50% wider in high volatility
    low_vol_trailing_multiplier: float = 0.8  # 20% tighter in low volatility

    # Time Decay - RELAXED v10.3 with more patience
    # Allow longer holding for trends to develop
    max_holding_days: int = MAX_HOLDING_DAYS  # 15 days (reasonable for VN market)
    time_decay_threshold: float = DEFAULT_TIME_DECAY_THRESHOLD
    t2_settlement_days: int = 2

    # v10.2: Time-based exit - BALANCED (not too aggressive)
    # Start later and reduce slower to capture trends
    time_decay_start_day: int = 10  # RELAXED: Start after day 10 (was 7)
    time_decay_tp_reduction: float = 0.08  # RELAXED: Reduce TP by 8% per day (was 12%)

    # v10.2: BEAR market time decay - BALANCED
    # Still exit faster in BEAR but allow mean reversion
    bear_market_max_holding_days: int = 8  # RELAXED: 8 days in BEAR (was 5)
    bear_market_time_decay_threshold: float = 0.015  # RELAXED: 1.5% profit (was 1%)
    bear_market_time_decay_start_day: int = 5  # RELAXED: Start after 5 days (was 3)
    use_aggressive_bear_time_decay: bool = True  # Keep enabled but with relaxed settings

    # NEW v10.2: Trend-aware time decay
    # If price is trending up, extend holding period
    use_trend_aware_time_decay: bool = True
    trend_extension_days: int = 5  # Extend up to 5 days if in strong uptrend
    trend_extension_min_profit: float = 0.03  # Min 3% profit to qualify for extension

    # NEW v10.1: Momentum reversal exit
    # Exit when trend momentum reverses sharply (avoid giving back profits)
    use_momentum_reversal_exit: bool = True
    momentum_reversal_rsi_drop: float = 15.0  # RSI drops 15+ points from peak = reversal
    momentum_reversal_ema_cross: bool = True  # EMA 5 crosses below EMA 20 = bearish
    momentum_reversal_min_profit: float = 0.02  # Only exit if in profit (2%+)
    momentum_reversal_volume_confirm: float = 1.5  # Confirm with volume > 1.5x avg

    # Profit Protection - IMPROVED v10.0
    profit_protection_activation: float = 0.02  # Activate at 2% profit
    profit_protection_percent: float = 0.70  # IMPROVED: Protect 70% of max profit

    # NEW v10.0: Session-based exit rules (Vietnam market specific)
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

    # NEW v10.2: Odd-Lot Trading Settings
    use_odd_lot_optimization: bool = True  # Enable odd-lot exit optimization
    odd_lot_auto_cleanup: bool = True  # Auto-sell odd-lots when profitable
    odd_lot_cleanup_min_profit_pct: float = 0.01  # Min 1% profit to cleanup odd-lot
    odd_lot_max_hold_days: int = 5  # Max days to hold odd-lot before forced cleanup
    odd_lot_cost_threshold_pct: float = 2.0  # Warn if costs > 2% for odd-lot


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
            self._check_news_sentiment,  # NEW v9.2: Check news sentiment force exit
            self._check_momentum_reversal,  # NEW v10.1: Momentum reversal detection
            self._check_odd_lot_cleanup,  # NEW v10.2: Odd-lot remaining shares cleanup
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

    def _calculate_volume_ratio(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """
        Calculate current volume ratio vs average volume.

        Used for panic selling detection in floor bounce logic.

        Args:
            df: DataFrame with 'volume' column
            lookback: Number of periods for average calculation

        Returns:
            Volume ratio (current / average). Returns 1.0 if calculation fails.
        """
        try:
            if df is None or df.empty or "volume" not in df.columns:
                return 1.0

            if len(df) < lookback + 1:
                lookback = max(1, len(df) - 1)

            current_volume = safe_iloc(df, -1, "volume")
            if current_volume is None or current_volume <= 0:
                return 1.0

            # Calculate average volume (excluding current bar)
            avg_volume = df["volume"].iloc[-(lookback + 1) : -1].mean()
            if avg_volume is None or avg_volume <= 0 or pd.isna(avg_volume):
                return 1.0

            return current_volume / avg_volume

        except Exception as e:
            logger.debug(f"Volume ratio calculation failed: {e}")
            return 1.0

    def _calculate_price_momentum(self, df: pd.DataFrame, lookback: int = 5) -> float:
        """
        Calculate short-term price momentum for floor bounce decision.

        IMPROVED v6.1: Price action confirmation for floor bounce logic.

        Used in combination with volume ratio:
        - Volume ratio > 2.0 AND momentum < -0.5% → EXIT (panic + downtrend)
        - Volume ratio > 2.5 → EXIT (panic regardless of momentum)
        - Momentum > 1% → HOLD (bounce confirmed)

        Args:
            df: DataFrame with 'close' column
            lookback: Number of periods for momentum calculation

        Returns:
            Price momentum as decimal (e.g., -0.005 for -0.5%). Returns 0.0 if calculation fails.
        """
        try:
            if df is None or df.empty or "close" not in df.columns:
                return 0.0

            if len(df) < lookback + 1:
                lookback = max(1, len(df) - 1)

            current_close = safe_iloc(df, -1, "close")
            lookback_close = safe_iloc(df, -(lookback + 1), "close")

            if current_close is None or lookback_close is None or lookback_close <= 0:
                return 0.0

            momentum = (current_close - lookback_close) / lookback_close
            return momentum

        except Exception as e:
            logger.debug(f"Price momentum calculation failed: {e}")
            return 0.0

    def _check_floor_bounce_with_momentum(
        self,
        symbol: str,
        current_price: float,
        floor_price: float,
        volume_ratio: float,
        price_momentum: float,
    ) -> tuple:
        """
        Multi-factor floor bounce decision combining volume and price momentum.

        IMPROVED v6.1: Enhanced floor bounce logic for Vietnam market.

        Decision matrix:
        1. Volume ratio >= 2.5 (panic) → EXIT immediately
        2. Volume ratio >= 2.0 AND momentum < -0.5% → EXIT (high vol + downtrend)
        3. Momentum > 1% → HOLD (bounce confirmed, cancel timer)
        4. Otherwise → Continue monitoring with timer

        Args:
            symbol: Stock symbol
            current_price: Current price
            floor_price: Calculated floor price (-7% from prev close)
            volume_ratio: Current volume / average volume
            price_momentum: Short-term price momentum

        Returns:
            Tuple of (should_exit: bool, reason: str, action: str)
            action: "EXIT", "HOLD", "MONITOR"
        """
        # Import thresholds
        try:
            from src.config.constants import (
                VN_FLOOR_BOUNCE_PANIC_VOLUME_RATIO,
                VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO,
                VN_FLOOR_BOUNCE_RECOVERY_PCT,
            )
        except ImportError:
            VN_FLOOR_BOUNCE_PANIC_VOLUME_RATIO = 2.5
            VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO = 1.5
            VN_FLOOR_BOUNCE_RECOVERY_PCT = 0.01

        # Thresholds for momentum-based decision
        MOMENTUM_EXIT_THRESHOLD = -0.005  # -0.5% momentum = downtrend
        MOMENTUM_RECOVERY_THRESHOLD = 0.01  # +1% momentum = bounce confirmed
        VOLUME_MOMENTUM_EXIT_THRESHOLD = 2.0  # Volume ratio for momentum-based exit

        # Case 1: Panic selling (volume >= 2.5x) → EXIT immediately
        if volume_ratio >= VN_FLOOR_BOUNCE_PANIC_VOLUME_RATIO:
            reason = (
                f"PANIC SELLING: Volume {volume_ratio:.1f}x >= {VN_FLOOR_BOUNCE_PANIC_VOLUME_RATIO}x threshold. "
                f"Momentum: {price_momentum*100:+.2f}%"
            )
            logger.warning(f"🚨 {symbol}: {reason}")
            return True, reason, "EXIT"

        # Case 2: High volume + negative momentum → EXIT
        if (
            volume_ratio >= VOLUME_MOMENTUM_EXIT_THRESHOLD
            and price_momentum < MOMENTUM_EXIT_THRESHOLD
        ):
            reason = (
                f"HIGH VOLUME + DOWNTREND: Volume {volume_ratio:.1f}x with momentum {price_momentum*100:+.2f}% "
                f"(< {MOMENTUM_EXIT_THRESHOLD*100:.1f}% threshold)"
            )
            logger.warning(f"📉 {symbol}: {reason}")
            return True, reason, "EXIT"

        # Case 3: Price recovery confirmed → HOLD (cancel timer)
        if price_momentum >= MOMENTUM_RECOVERY_THRESHOLD:
            reason = (
                f"BOUNCE CONFIRMED: Momentum {price_momentum*100:+.2f}% >= {MOMENTUM_RECOVERY_THRESHOLD*100:.1f}% threshold. "
                f"Volume: {volume_ratio:.1f}x"
            )
            logger.info(f"✅ {symbol}: {reason}")
            return False, reason, "HOLD"

        # Case 4: Continue monitoring
        reason = (
            f"MONITORING: Volume {volume_ratio:.1f}x, Momentum {price_momentum*100:+.2f}%. "
            f"Waiting for clearer signal."
        )
        logger.debug(f"📊 {symbol}: {reason}")
        return False, reason, "MONITOR"

    # =========================================================================
    # EXIT CHECK #1: STOP LOSS (IMPROVED v6.0 - Vietnam Market Specific)
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

        # IMPROVED v6.0: Volume-based floor bounce logic
        # =========================================================================
        # Floor bounce protection with VOLUME-BASED exit triggers
        #
        # Problem: Time-only wait (30 min) may not be enough in panic selling
        # Solution: Use volume ratio to determine exit urgency
        #
        # Volume-based rules:
        # - Volume ratio > 3.0 (panic): EXIT IMMEDIATELY, no bounce wait
        # - Volume ratio 1.5-3.0 (elevated): Extended 60-minute wait
        # - Volume ratio < 1.5 (normal): Standard 30-minute wait
        # - Price recovery > 1%: Cancel timer, resume normal monitoring
        # =========================================================================
        if df is not None and len(df) >= 2:
            try:
                prev_close = safe_iloc(df, -2, "close")
                if prev_close and prev_close > 0:
                    floor_price = prev_close * 0.93  # -7% floor for HOSE

                    # If current price is within 0.5% of floor
                    if current_price <= floor_price * 1.005:
                        from datetime import datetime

                        # Calculate volume ratio for panic detection
                        volume_ratio = self._calculate_volume_ratio(df)

                        # Import floor bounce constants
                        try:
                            from src.config.constants import (
                                VN_FLOOR_BOUNCE_MAX_WAIT_MINUTES,
                                VN_FLOOR_BOUNCE_EXTENDED_WAIT_MINUTES,
                                VN_FLOOR_BOUNCE_PANIC_VOLUME_RATIO,
                                VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO,
                                VN_FLOOR_BOUNCE_RECOVERY_PCT,
                            )
                        except ImportError:
                            VN_FLOOR_BOUNCE_MAX_WAIT_MINUTES = 30
                            VN_FLOOR_BOUNCE_EXTENDED_WAIT_MINUTES = 60
                            VN_FLOOR_BOUNCE_PANIC_VOLUME_RATIO = (
                                2.5  # IMPROVED Priority 1: Increased sensitivity
                            )
                            VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO = 1.5
                            VN_FLOOR_BOUNCE_RECOVERY_PCT = 0.01

                        # IMPROVED v6.1: Multi-factor floor bounce decision
                        # Combines volume ratio AND price momentum for better accuracy
                        price_momentum = self._calculate_price_momentum(df, lookback=5)

                        should_exit, exit_reason, action = self._check_floor_bounce_with_momentum(
                            symbol=symbol,
                            current_price=current_price,
                            floor_price=floor_price,
                            volume_ratio=volume_ratio,
                            price_momentum=price_momentum,
                        )

                        if action == "EXIT":
                            # Clear any existing floor wait tracking
                            floor_wait_key = f"floor_wait_{symbol}"
                            if (
                                hasattr(self, "_floor_wait_times")
                                and floor_wait_key in self._floor_wait_times
                            ):
                                del self._floor_wait_times[floor_wait_key]

                            # Return exit decision for panic/high-volume-downtrend
                            return ExitDecision(
                                should_exit=True,
                                exit_reason=ExitReason.PANIC_SELLING,
                                exit_type="FULL",
                                exit_price=current_price,
                                expected_pnl=pnl_amount,
                                expected_pnl_percent=pnl_percent,
                                message=f"🚨 FLOOR EXIT: {exit_reason}",
                                urgency=5,
                                metadata={
                                    "volume_ratio": volume_ratio,
                                    "price_momentum": price_momentum,
                                    "floor_price": floor_price,
                                },
                            )
                        elif action == "HOLD":
                            # Bounce confirmed - clear tracking and continue normal monitoring
                            floor_wait_key = f"floor_wait_{symbol}"
                            if (
                                hasattr(self, "_floor_wait_times")
                                and floor_wait_key in self._floor_wait_times
                            ):
                                del self._floor_wait_times[floor_wait_key]
                            return None  # Continue to other checks
                        else:  # action == "MONITOR"
                            # Determine wait time based on volume
                            if volume_ratio >= VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO:
                                max_floor_wait_minutes = (
                                    VN_FLOOR_BOUNCE_EXTENDED_WAIT_MINUTES  # 60 min
                                )
                                volume_status = f"elevated ({volume_ratio:.1f}x)"
                            else:
                                max_floor_wait_minutes = VN_FLOOR_BOUNCE_MAX_WAIT_MINUTES  # 30 min
                                volume_status = f"normal ({volume_ratio:.1f}x)"

                            floor_wait_key = f"floor_wait_{symbol}"
                            current_time = datetime.now()

                            # Get or initialize floor wait tracking
                            if not hasattr(self, "_floor_wait_times"):
                                self._floor_wait_times = {}

                            if floor_wait_key not in self._floor_wait_times:
                                # First time at floor - start tracking
                                self._floor_wait_times[floor_wait_key] = {
                                    "start_time": current_time,
                                    "volume_ratio": volume_ratio,
                                    "floor_price": floor_price,
                                }
                                logger.info(
                                    f"📊 {symbol}: Price at floor ({current_price:,.0f} ≈ {floor_price:,.0f}). "
                                    f"Volume {volume_status}. "
                                    f"Starting {max_floor_wait_minutes}-minute wait for bounce."
                                )
                                return None
                            else:
                                # Check if we've waited too long
                                floor_data = self._floor_wait_times[floor_wait_key]
                                floor_wait_start = floor_data["start_time"]
                                wait_minutes = (
                                    current_time - floor_wait_start
                                ).total_seconds() / 60

                                # Update volume ratio (may have changed)
                                floor_data["volume_ratio"] = volume_ratio

                                if wait_minutes < max_floor_wait_minutes:
                                    logger.debug(
                                        f"📊 {symbol}: At floor for {wait_minutes:.0f}/{max_floor_wait_minutes} min. "
                                        f"Volume {volume_status}. Waiting for bounce..."
                                    )
                                    return None
                                else:
                                    # TIME LIMIT EXCEEDED - Exit anyway
                                    logger.warning(
                                        f"⏰ {symbol}: Floor wait timeout ({wait_minutes:.0f} min). "
                                        f"Volume {volume_status}. No bounce detected - triggering stop loss."
                                    )
                                    # Clear the tracking
                                    del self._floor_wait_times[floor_wait_key]
                                    # Fall through to stop loss below
                    else:
                        # Price moved away from floor - check if recovery is significant
                        floor_wait_key = f"floor_wait_{symbol}"
                        if (
                            hasattr(self, "_floor_wait_times")
                            and floor_wait_key in self._floor_wait_times
                        ):
                            floor_data = self._floor_wait_times[floor_wait_key]
                            floor_price = floor_data.get("floor_price", prev_close * 0.93)
                            recovery_pct = (current_price - floor_price) / floor_price

                            try:
                                from src.config.constants import VN_FLOOR_BOUNCE_RECOVERY_PCT
                            except ImportError:
                                VN_FLOOR_BOUNCE_RECOVERY_PCT = 0.01

                            if recovery_pct >= VN_FLOOR_BOUNCE_RECOVERY_PCT:
                                del self._floor_wait_times[floor_wait_key]
                                logger.info(
                                    f"✅ {symbol}: Price recovered {recovery_pct:.1%} from floor - "
                                    f"tracking cleared, resuming normal monitoring"
                                )
                            else:
                                logger.debug(
                                    f"📊 {symbol}: Price slightly above floor ({recovery_pct:.1%} recovery). "
                                    f"Keeping floor tracking active."
                                )
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

        IMPROVED v7.0: Enhanced gap protection with VOLUME CONFIRMATION

        Vietnam market gaps are significant due to:
        - Overnight news (global markets, company announcements)
        - Foreign investor sentiment changes
        - Regulatory changes
        - VN market can gap up to ±7% (full daily limit)

        Exit rules (TIGHTENED v7.0):
        - Gap down > 4%: EMERGENCY EXIT regardless of P&L
        - Gap down 2.5-4% AND profitable: Profit protection exit
        - Gap down 2-2.5% + Volume > 2x: Distribution detected - EXIT
        - Gap up > 4%: Consider partial profit taking

        NEW v7.0: Volume confirmation for gap detection
        - High volume gap down = institutional selling = EXIT immediately
        - Low volume gap down = may recover = monitor
        """
        df = ctx.get("df")
        if df is None or len(df) < 2:
            return None

        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        current_price = ctx["current_price"]
        symbol = ctx.get("symbol", "")

        try:
            # Import gap thresholds from constants
            try:
                from src.config.constants import (
                    VN_GAP_DOWN_EMERGENCY_THRESHOLD,
                    VN_GAP_DOWN_EXIT_THRESHOLD,
                    VN_GAP_UP_PROFIT_TAKE_THRESHOLD,
                )
            except ImportError:
                VN_GAP_DOWN_EMERGENCY_THRESHOLD = -0.04  # -4%
                VN_GAP_DOWN_EXIT_THRESHOLD = -0.025  # -2.5%
                VN_GAP_UP_PROFIT_TAKE_THRESHOLD = 0.04  # +4%

            prev_close = safe_iloc(df, -2, "close")
            today_open = safe_iloc(df, -1, "open")

            if prev_close is None or today_open is None or prev_close <= 0:
                return None

            gap_percent = (today_open - prev_close) / prev_close
            gap_percent_display = gap_percent * 100

            # EMERGENCY EXIT: Gap down > 4% (TIGHTENED from 5%)
            # VN market can gap full 7%, so 4% is already severe
            if gap_percent <= VN_GAP_DOWN_EMERGENCY_THRESHOLD:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.EMERGENCY_EXIT,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=(
                        f"🚨 GAP DOWN EMERGENCY: {gap_percent_display:.1f}% gap "
                        f"(threshold: {VN_GAP_DOWN_EMERGENCY_THRESHOLD*100:.1f}%) - "
                        f"exiting to protect capital | P&L: {pnl_percent:+.2f}%"
                    ),
                    urgency=5,
                    metadata={
                        "gap_percent": gap_percent_display,
                        "prev_close": prev_close,
                        "today_open": today_open,
                        "trigger": "emergency_gap_down",
                        "threshold": VN_GAP_DOWN_EMERGENCY_THRESHOLD * 100,
                    },
                )

            # PROFIT PROTECTION: Gap down 2.5-4% when in profit (TIGHTENED from 3%)
            if gap_percent <= VN_GAP_DOWN_EXIT_THRESHOLD and pnl_percent > 0:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.PROFIT_PROTECTION,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=(
                        f"📉 GAP DOWN PROTECTION: {gap_percent_display:.1f}% gap - "
                        f"protecting {pnl_percent:+.2f}% profit"
                    ),
                    urgency=4,
                    metadata={
                        "gap_percent": gap_percent_display,
                        "prev_close": prev_close,
                        "today_open": today_open,
                        "trigger": "profit_protection_gap_down",
                    },
                )

            # NEW v7.0: VOLUME-CONFIRMED GAP DOWN (Distribution Detection)
            # Gap down 2-2.5% + High volume = institutional distribution = EXIT
            volume_ratio = self._calculate_volume_ratio(df, lookback=20)
            gap_with_volume_threshold = -0.02  # -2%
            volume_multiplier_threshold = 2.0  # 2x average volume

            if (
                gap_percent <= gap_with_volume_threshold
                and gap_percent > VN_GAP_DOWN_EXIT_THRESHOLD
                and volume_ratio >= volume_multiplier_threshold
            ):
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.GAP_DOWN_EMERGENCY,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=(
                        f"🚨 DISTRIBUTION DETECTED: {gap_percent_display:.1f}% gap "
                        f"+ {volume_ratio:.1f}x volume - institutional selling"
                    ),
                    urgency=5,
                    metadata={
                        "gap_percent": gap_percent_display,
                        "volume_ratio": volume_ratio,
                        "prev_close": prev_close,
                        "today_open": today_open,
                        "trigger": "volume_confirmed_gap_down",
                    },
                )

            # GAP UP PROFIT TAKING: Gap up > 4% - IMPROVED v7.0
            # VN market: Strong gap up near ceiling often reverses
            # Take partial profit to lock in gains
            VN_GAP_UP_PARTIAL_THRESHOLD = 0.04  # +4% gap → partial exit (50%)
            VN_GAP_UP_FULL_THRESHOLD = 0.06  # +6% gap → full exit (near ceiling)

            if gap_percent >= VN_GAP_UP_FULL_THRESHOLD and pnl_percent > 3.0:
                # Near ceiling (+6-7%) - exit fully to lock in gains
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.TAKE_PROFIT_2,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=(
                        f"📈 GAP UP NEAR CEILING: {gap_percent_display:.1f}% gap - "
                        f"chốt toàn bộ {pnl_percent:+.2f}% (near ±7% limit)"
                    ),
                    urgency=4,
                    metadata={
                        "gap_percent": gap_percent_display,
                        "prev_close": prev_close,
                        "today_open": today_open,
                        "trigger": "gap_up_near_ceiling",
                    },
                )

            if gap_percent >= VN_GAP_UP_PARTIAL_THRESHOLD and pnl_percent > 2.0:
                # Check if partial exit already done
                partial_exits = ctx.get("partial_exits", [])
                if not partial_exits:  # No partial exit yet
                    return ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.TAKE_PROFIT_1,
                        exit_type="PARTIAL_50%",
                        exit_price=current_price,
                        expected_pnl=pnl_amount,
                        expected_pnl_percent=pnl_percent,
                        message=(
                            f"📈 GAP UP PROFIT TAKE: {gap_percent_display:.1f}% gap - "
                            f"chốt 50% với {pnl_percent:+.2f}% profit (gap may reverse)"
                        ),
                        urgency=3,
                        metadata={
                            "gap_percent": gap_percent_display,
                            "prev_close": prev_close,
                            "today_open": today_open,
                            "trigger": "gap_up_profit_taking",
                        },
                    )
                else:
                    logger.info(
                        f"📈 {symbol}: Gap UP {gap_percent_display:.1f}% but partial exit already done. "
                        f"Holding remaining position."
                    )

            # Log significant gaps for monitoring
            if gap_percent <= -0.02:  # -2%
                logger.info(
                    f"📊 {symbol}: Gap down {gap_percent_display:.1f}% detected "
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

    def _get_dynamic_tp_levels(self, market_regime: Optional[Dict]) -> Tuple[float, float, float]:
        """
        Get dynamic take profit levels based on market regime.

        IMPROVED v6.1: Adaptive TP levels for Vietnam market.

        Rationale:
        - BULL: Wider targets (8%, 15%, 25%) to capture momentum
        - BEAR: Tighter targets (4%, 7%, 10%) to lock profits quickly
        - SIDEWAYS: Standard targets (6%, 10%, 15%)
        - HIGH_VOLATILITY: Moderate targets (5%, 8%, 12%) with quick exits

        Args:
            market_regime: Market regime dict with 'regime' key

        Returns:
            Tuple of (TP1, TP2, TP3) percentages as decimals
        """
        if not self.config.use_dynamic_tp_levels or market_regime is None:
            return self.config.take_profit_levels

        regime = market_regime.get("regime", "SIDEWAYS")
        confidence = market_regime.get("confidence", 50)

        # Only apply dynamic levels if regime confidence is high enough
        if confidence < 60:
            logger.debug(f"📊 Regime confidence {confidence:.0f}% < 60%, using default TP levels")
            return self.config.take_profit_levels

        if regime == "BULL":
            tp_levels = self.config.tp_levels_bull
            logger.debug(f"📈 BULL regime: Using wider TP levels {tp_levels}")
        elif regime == "BEAR":
            tp_levels = self.config.tp_levels_bear
            logger.debug(f"📉 BEAR regime: Using tighter TP levels {tp_levels}")
        elif regime == "HIGH_VOLATILITY":
            tp_levels = self.config.tp_levels_high_volatility
            logger.debug(f"⚡ HIGH_VOL regime: Using moderate TP levels {tp_levels}")
        else:  # SIDEWAYS or unknown
            tp_levels = self.config.tp_levels_sideways
            logger.debug(f"↔️ SIDEWAYS regime: Using standard TP levels {tp_levels}")

        return tp_levels

    def _check_take_profit(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Check take profit levels.

        IMPROVED v6.1: Dynamic TP levels based on market regime.

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
        market_regime = ctx.get("market_regime")
        entry_price = ctx["entry_price"]

        if not tp_targets:
            return None

        # IMPROVED v6.1: Get dynamic TP levels based on regime
        dynamic_tp_pcts = self._get_dynamic_tp_levels(market_regime)

        # Calculate dynamic TP prices from entry price
        dynamic_tp_targets = [entry_price * (1 + tp_pct) for tp_pct in dynamic_tp_pcts]

        # Use dynamic targets if they differ significantly from provided targets
        # This allows override from caller while still benefiting from regime adjustment
        if tp_targets and len(tp_targets) >= 2:
            # Check if provided targets are close to default (within 1%)
            default_tp1 = entry_price * (1 + self.config.take_profit_levels[0])
            if abs(tp_targets[0] - default_tp1) / default_tp1 < 0.01:
                # Provided targets are default, use dynamic instead
                tp_targets = dynamic_tp_targets
                logger.debug(f"📊 Using dynamic TP targets: {[f'{t:,.0f}' for t in tp_targets]}")

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

        IMPROVED v10.1: Aggressive time decay in BEAR market
        - In BEAR, time works against you - exit faster
        - Max 5 days holding in BEAR (vs 15 in BULL)
        - Start time pressure after 3 days (vs 7 in BULL)
        """
        days_held = ctx["days_held"]
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        current_price = ctx["current_price"]
        market_regime = ctx.get("market_regime")
        df = ctx["df"]
        is_poor_performer = ctx["is_poor_performer"]
        symbol = ctx["symbol"]

        # Get current regime
        regime = market_regime.get("regime", "SIDEWAYS") if market_regime else "SIDEWAYS"

        # NEW v10.1: Aggressive BEAR market time decay
        if self.config.use_aggressive_bear_time_decay and regime == "BEAR":
            bear_start_day = self.config.bear_market_time_decay_start_day  # 3 days
            bear_max_days = self.config.bear_market_max_holding_days  # 5 days
            bear_threshold = self.config.bear_market_time_decay_threshold * 100  # 1%

            # Early exit check in BEAR - start pressure after 3 days
            if days_held >= bear_start_day:
                # Calculate minimum expected profit for days held
                # Day 3: need 1%, Day 4: need 1.5%, Day 5: need 2%
                min_expected_profit = bear_threshold + (days_held - bear_start_day) * 0.5

                if pnl_percent < min_expected_profit:
                    return ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.TIME_DECAY,
                        exit_type="FULL",
                        exit_price=current_price,
                        expected_pnl=pnl_amount,
                        expected_pnl_percent=pnl_percent,
                        message=(
                            f"⏰🐻 BEAR MARKET TIME PRESSURE ({days_held} days): "
                            f"P&L {pnl_percent:+.2f}% < required {min_expected_profit:.1f}% - "
                            f"Exit early to avoid further losses"
                        ),
                        urgency=3,
                        metadata={
                            "days_held": days_held,
                            "bear_max_days": bear_max_days,
                            "min_expected_profit": min_expected_profit,
                            "regime": "BEAR",
                        },
                    )

            # Force exit at bear max days
            if days_held >= bear_max_days:
                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.TIME_DECAY,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=(
                        f"⏰🐻 BEAR MAX HOLDING ({days_held} >= {bear_max_days} days): "
                        f"Force exit with P&L {pnl_percent:+.2f}% - No edge holding longer in BEAR"
                    ),
                    urgency=4,
                    metadata={
                        "days_held": days_held,
                        "bear_max_days": bear_max_days,
                        "regime": "BEAR",
                    },
                )

        # Calculate adaptive max holding days (normal logic)
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
    # EXIT CHECK #11: NEWS SENTIMENT (NEW v9.2)
    # =========================================================================

    def _check_news_sentiment(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Check Vietnamese news sentiment for force exit.

        NEW v9.2: Forces exit when severe negative news is detected
        or adjusts exit urgency based on sentiment.

        Returns:
            ExitDecision if force exit, None otherwise
        """
        if not VN_NEWS_SENTIMENT_AVAILABLE:
            return None

        symbol = ctx["symbol"]
        current_price = ctx["current_price"]
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]

        try:
            integration = get_news_sentiment_integration()
            exit_check = integration.check_exit_sentiment(
                symbol=symbol,
                current_pnl_pct=pnl_percent / 100,  # Convert to decimal
            )

            if exit_check is None:
                return None

            # Force exit on severe negative news
            if exit_check.get("should_exit", False):
                urgency = exit_check.get("urgency", 0.5)
                reasons = exit_check.get("reasons", [])
                reason_text = reasons[0] if reasons else "Negative news detected"

                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.NEWS_SENTIMENT_FORCE_EXIT,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=f"📰 NEWS FORCE EXIT: {reason_text} ({pnl_percent:+.2f}%)",
                    urgency=int(urgency * 5) + 1,  # Scale to 1-6
                    metadata={
                        "news_urgency": urgency,
                        "news_reasons": reasons[:3],
                    },
                )

        except Exception as e:
            logger.debug(f"News sentiment exit check error: {e}")

        return None

    # =========================================================================
    # EXIT CHECK #12: MOMENTUM REVERSAL (NEW v10.1)
    # =========================================================================

    def _check_momentum_reversal(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Check for momentum reversal to exit before trend changes.

        NEW v10.1: Detects sharp momentum reversals to protect profits.

        Triggers when:
        1. RSI drops 15+ points from recent peak (momentum loss)
        2. EMA 5 crosses below EMA 20 (short-term trend reversal)
        3. Confirmed by volume surge (distribution)

        Only applies when position is in profit to protect gains.

        Returns:
            ExitDecision if momentum reversal detected, None otherwise
        """
        if not self.config.use_momentum_reversal_exit:
            return None

        df = ctx["df"]
        pnl_percent = ctx["pnl_percent"]
        pnl_amount = ctx["pnl_amount"]
        current_price = ctx["current_price"]
        symbol = ctx["symbol"]

        # Only check if in profit (above threshold)
        min_profit = self.config.momentum_reversal_min_profit * 100  # 2%
        if pnl_percent < min_profit:
            return None

        if df is None or len(df) < 20:
            return None

        try:
            reversal_signals = []
            signal_strength = 0

            # Signal 1: RSI Drop from peak
            if "rsi" in df.columns:
                current_rsi = safe_get_latest(df, "rsi", 50)
                # Get RSI peak in last 10 periods
                rsi_peak = df["rsi"].tail(10).max()
                rsi_drop = rsi_peak - current_rsi

                if rsi_drop >= self.config.momentum_reversal_rsi_drop:  # 15 points
                    reversal_signals.append(
                        f"RSI dropped {rsi_drop:.1f} points (peak: {rsi_peak:.0f})"
                    )
                    signal_strength += 2

            # Signal 2: EMA Cross (bearish)
            if self.config.momentum_reversal_ema_cross:
                # Calculate EMA 5 and EMA 20
                if len(df) >= 20:
                    ema5 = df["close"].ewm(span=5).mean()
                    ema20 = df["close"].ewm(span=20).mean()

                    # Check for bearish cross (EMA5 just crossed below EMA20)
                    if len(ema5) >= 2 and len(ema20) >= 2:
                        prev_ema5 = ema5.iloc[-2]
                        prev_ema20 = ema20.iloc[-2]
                        curr_ema5 = ema5.iloc[-1]
                        curr_ema20 = ema20.iloc[-1]

                        # Bearish cross: EMA5 was above EMA20, now below
                        if prev_ema5 >= prev_ema20 and curr_ema5 < curr_ema20:
                            reversal_signals.append("EMA5 crossed below EMA20 (bearish)")
                            signal_strength += 2

                        # Already in bearish alignment
                        elif curr_ema5 < curr_ema20:
                            reversal_signals.append("EMA5 < EMA20 (bearish alignment)")
                            signal_strength += 1

            # Signal 3: Volume Confirmation
            if "volume" in df.columns:
                current_volume = safe_get_latest(df, "volume", 0)
                avg_volume = df["volume"].tail(20).mean()

                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    if volume_ratio >= self.config.momentum_reversal_volume_confirm:  # 1.5x
                        # Check if down day
                        is_down = safe_get_latest(df, "close", 0) < safe_get_latest(df, "open", 0)
                        if is_down:
                            reversal_signals.append(f"Distribution volume ({volume_ratio:.1f}x)")
                            signal_strength += 1

            # Trigger exit if enough signals (need at least 2 signals or high strength)
            if signal_strength >= 3 or (signal_strength >= 2 and len(reversal_signals) >= 2):
                signals_text = " | ".join(reversal_signals[:3])

                return ExitDecision(
                    should_exit=True,
                    exit_reason=ExitReason.REVERSAL_PATTERN,
                    exit_type="FULL",
                    exit_price=current_price,
                    expected_pnl=pnl_amount,
                    expected_pnl_percent=pnl_percent,
                    message=(
                        f"📉 MOMENTUM REVERSAL: {signals_text} | "
                        f"Protecting {pnl_percent:+.2f}% profit"
                    ),
                    urgency=4,
                    metadata={
                        "reversal_signals": reversal_signals,
                        "signal_strength": signal_strength,
                        "pattern": "momentum_reversal",
                    },
                )

            # Log potential reversal for monitoring
            if reversal_signals:
                logger.debug(
                    f"📊 {symbol}: Potential momentum reversal: {reversal_signals} "
                    f"(strength: {signal_strength}, threshold: 3)"
                )

        except Exception as e:
            logger.debug(f"Momentum reversal check error: {e}")

        return None

    # =========================================================================
    # EXIT CHECK: ODD-LOT CLEANUP (NEW v10.2)
    # =========================================================================

    def _check_odd_lot_cleanup(self, ctx: Dict) -> Optional[ExitDecision]:
        """
        Check if position has odd-lot shares that should be cleaned up.

        NEW v10.2: Optimizes handling of remaining shares < 100.

        Vietnam market rules:
        - Standard lot size: 100 shares
        - Odd-lot (1-99 shares): Higher spreads, minimum commission
        - Best to sell when profitable to avoid cost inefficiency

        Logic:
        - If remaining shares < 100 AND profitable > 1% → SELL
        - If holding odd-lot > 5 days → SELL (free up capital)
        - Warn if cost ratio > 2%

        Args:
            ctx: Exit context dict

        Returns:
            ExitDecision or None
        """
        if not self.config.use_odd_lot_optimization:
            return None

        if not ODD_LOT_AVAILABLE:
            return None

        symbol = ctx.get("symbol", "")
        entry_price = ctx.get("entry_price", 0)
        current_price = ctx.get("current_price", 0)
        pnl_percent = ctx.get("pnl_percent", 0)
        pnl_amount = ctx.get("pnl_amount", 0)
        days_held = ctx.get("days_held", 0)

        try:
            # Get remaining shares from context or position manager
            remaining_shares = ctx.get("remaining_shares", 0)

            if remaining_shares <= 0:
                # Try to get from partial exit tracker
                if self.partial_exit_tracker.has_partial_exit(symbol):
                    # After partial exit, remaining is likely odd-lot
                    # This needs position info from portfolio manager
                    remaining_shares = ctx.get("position_quantity", 0)

            if remaining_shares <= 0 or remaining_shares >= 100:
                return None  # Not an odd-lot situation

            # This is an odd-lot situation
            logger.debug(f"📦 {symbol}: Odd-lot detected: {remaining_shares} shares")

            # Get odd-lot handler
            try:
                odd_lot_handler = get_odd_lot_handler()
            except NameError:
                try:
                    odd_lot_handler = get_odd_lot_logic()
                except NameError:
                    return None

            # Analyze odd-lot
            if hasattr(odd_lot_handler, "optimize_odd_lot_exit"):
                result = odd_lot_handler.optimize_odd_lot_exit(
                    remaining_shares=remaining_shares,
                    current_price=current_price,
                    avg_cost=entry_price,
                )
            else:
                # Use basic analysis
                result = odd_lot_handler.analyze_order(
                    quantity=remaining_shares, price=current_price, is_sell=True
                )

            # Check if profitable and should cleanup
            if self.config.odd_lot_auto_cleanup:
                # Case 1: Profitable odd-lot → cleanup immediately
                if pnl_percent >= self.config.odd_lot_cleanup_min_profit_pct * 100:
                    return ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.ODD_LOT_CLEANUP,
                        exit_type="FULL",
                        exit_price=current_price,
                        expected_pnl=pnl_amount * remaining_shares,
                        expected_pnl_percent=pnl_percent,
                        message=(
                            f"📦 ODD-LOT CLEANUP: Selling {remaining_shares} shares @ "
                            f"{current_price:,.0f} | Net P&L: {pnl_percent:+.2f}%"
                        ),
                        urgency=2,
                        metadata={
                            "odd_lot_shares": remaining_shares,
                            "action": "CLEANUP_PROFITABLE",
                            "cost_analysis": result,
                        },
                    )

                # Case 2: Held too long → cleanup to free capital
                if days_held >= self.config.odd_lot_max_hold_days:
                    return ExitDecision(
                        should_exit=True,
                        exit_reason=ExitReason.ODD_LOT_CLEANUP,
                        exit_type="FULL",
                        exit_price=current_price,
                        expected_pnl=pnl_amount * remaining_shares,
                        expected_pnl_percent=pnl_percent,
                        message=(
                            f"📦 ODD-LOT TIME LIMIT: Held {remaining_shares} shares for "
                            f"{days_held} days (max: {self.config.odd_lot_max_hold_days}) | "
                            f"P&L: {pnl_percent:+.2f}%"
                        ),
                        urgency=2,
                        metadata={
                            "odd_lot_shares": remaining_shares,
                            "action": "CLEANUP_TIME_LIMIT",
                            "days_held": days_held,
                        },
                    )

            # Case 3: Check if worth trading (cost vs return)
            if hasattr(odd_lot_handler, "is_worth_trading"):
                is_worth, reason = odd_lot_handler.is_worth_trading(
                    quantity=remaining_shares,
                    price=current_price,
                    expected_return_pct=max(0, pnl_percent),
                )

                if not is_worth and pnl_percent > 0:
                    # Not worth selling now, but warn
                    logger.info(
                        f"📦 {symbol}: Odd-lot of {remaining_shares} shares - "
                        f"not worth trading yet. {reason}"
                    )

            # Check cost efficiency warning
            if hasattr(odd_lot_handler, "calculate_effective_cost"):
                costs = odd_lot_handler.calculate_effective_cost(remaining_shares, current_price)
                cost_pct = costs.get("cost_pct", 0)

                if cost_pct > self.config.odd_lot_cost_threshold_pct:
                    logger.warning(
                        f"📦 {symbol}: Odd-lot costs high: {cost_pct:.1f}% > "
                        f"{self.config.odd_lot_cost_threshold_pct}%"
                    )

        except Exception as e:
            logger.debug(f"Odd-lot cleanup check error: {e}")

        return None

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
# NEW v7.0: PROFIT LOCK MECHANISM
# =============================================================================


@dataclass
class ProfitLockLevel:
    """Configuration for a profit lock level"""

    profit_threshold: float  # Profit % to trigger this level
    lock_percent: float  # % of profit to lock
    guaranteed_profit: float  # Minimum guaranteed profit %
    description: str


@dataclass
class ProfitLockStatus:
    """Current profit lock status for a position"""

    symbol: str
    current_profit_pct: float
    active_level: Optional[int]  # 0, 1, 2, 3 or None
    locked_profit_pct: float
    guaranteed_profit_pct: float
    stop_loss_price: float
    original_entry_price: float
    message: str
    should_update_stop: bool


class ProfitLockMechanism:
    """
    Lock profit at milestones to protect gains.

    NEW v7.0: Milestone-based profit protection for Vietnam market.

    Problem: Trailing stops can get whipsawed in volatile VN market (±7% daily limit).
    Solution: Lock profit at specific milestones with guaranteed minimum returns.

    Default Milestones:
    - Level 1: +5% profit → Lock 50% → Guaranteed +2.5%
    - Level 2: +10% profit → Lock 60% → Guaranteed +6%
    - Level 3: +15% profit → Lock 70% → Guaranteed +10.5%

    Vietnam-specific adjustments:
    - Account for ±7% daily limit
    - Wider locks for volatile stocks (beta > 1.2)
    - Tighter locks for defensive stocks (beta < 0.8)
    - Transaction costs (~1.5%) factored into guaranteed profit

    Usage:
        lock_mechanism = ProfitLockMechanism()
        status = lock_mechanism.check_profit_lock(
            symbol="VNM",
            entry_price=100000,
            current_price=110000,
            current_stop_loss=95000,
        )
        if status.should_update_stop:
            new_stop = status.stop_loss_price
    """

    # Default profit lock levels
    DEFAULT_LEVELS = [
        ProfitLockLevel(
            profit_threshold=5.0,
            lock_percent=50.0,
            guaranteed_profit=2.5,
            description="Level 1: Lock 50% at +5%",
        ),
        ProfitLockLevel(
            profit_threshold=10.0,
            lock_percent=60.0,
            guaranteed_profit=6.0,
            description="Level 2: Lock 60% at +10%",
        ),
        ProfitLockLevel(
            profit_threshold=15.0,
            lock_percent=70.0,
            guaranteed_profit=10.5,
            description="Level 3: Lock 70% at +15%",
        ),
    ]

    # Beta-adjusted levels for volatile stocks
    HIGH_BETA_LEVELS = [
        ProfitLockLevel(
            profit_threshold=6.0,
            lock_percent=45.0,
            guaranteed_profit=2.7,
            description="High Beta L1: Lock 45% at +6%",
        ),
        ProfitLockLevel(
            profit_threshold=12.0,
            lock_percent=55.0,
            guaranteed_profit=6.6,
            description="High Beta L2: Lock 55% at +12%",
        ),
        ProfitLockLevel(
            profit_threshold=18.0,
            lock_percent=65.0,
            guaranteed_profit=11.7,
            description="High Beta L3: Lock 65% at +18%",
        ),
    ]

    # Tighter levels for defensive stocks
    LOW_BETA_LEVELS = [
        ProfitLockLevel(
            profit_threshold=4.0,
            lock_percent=55.0,
            guaranteed_profit=2.2,
            description="Low Beta L1: Lock 55% at +4%",
        ),
        ProfitLockLevel(
            profit_threshold=8.0,
            lock_percent=65.0,
            guaranteed_profit=5.2,
            description="Low Beta L2: Lock 65% at +8%",
        ),
        ProfitLockLevel(
            profit_threshold=12.0,
            lock_percent=75.0,
            guaranteed_profit=9.0,
            description="Low Beta L3: Lock 75% at +12%",
        ),
    ]

    def __init__(
        self,
        levels: Optional[List[ProfitLockLevel]] = None,
        use_beta_adjustment: bool = True,
        high_beta_threshold: float = 1.2,
        low_beta_threshold: float = 0.8,
        transaction_cost: float = 0.015,  # 1.5% round trip
    ):
        """
        Initialize profit lock mechanism.

        Args:
            levels: Custom profit lock levels (uses DEFAULT_LEVELS if None)
            use_beta_adjustment: Adjust levels based on stock beta
            high_beta_threshold: Beta threshold for wider locks
            low_beta_threshold: Beta threshold for tighter locks
            transaction_cost: Transaction cost to factor into guaranteed profit
        """
        self.default_levels = levels or self.DEFAULT_LEVELS
        self.use_beta_adjustment = use_beta_adjustment
        self.high_beta_threshold = high_beta_threshold
        self.low_beta_threshold = low_beta_threshold
        self.transaction_cost = transaction_cost

        # Track active locks per symbol
        self._active_locks: Dict[str, int] = {}  # symbol -> highest level achieved

    def check_profit_lock(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        current_stop_loss: float,
        beta: Optional[float] = None,
    ) -> ProfitLockStatus:
        """
        Check if profit lock should be activated or updated.

        Args:
            symbol: Stock symbol
            entry_price: Original entry price
            current_price: Current market price
            current_stop_loss: Current stop loss price
            beta: Stock beta (optional, for level adjustment)

        Returns:
            ProfitLockStatus with lock details and new stop loss if applicable
        """
        symbol = symbol.upper()

        # Calculate current profit
        if entry_price <= 0:
            return self._create_status(
                symbol, 0, None, 0, 0, current_stop_loss, entry_price, "Invalid entry price", False
            )

        current_profit_pct = ((current_price - entry_price) / entry_price) * 100

        # Get appropriate levels based on beta
        levels = self._get_levels_for_beta(beta)

        # Get current active level
        current_level = self._active_locks.get(symbol, -1)

        # Find highest applicable level
        new_level = -1
        for i, level in enumerate(levels):
            if current_profit_pct >= level.profit_threshold:
                new_level = i

        # No level triggered
        if new_level < 0:
            return self._create_status(
                symbol,
                current_profit_pct,
                None,
                0,
                0,
                current_stop_loss,
                entry_price,
                f"Profit {current_profit_pct:+.2f}% - no lock triggered yet",
                False,
            )

        # Get the active level
        active_level = levels[new_level]

        # Calculate locked profit and guaranteed profit
        locked_profit_pct = current_profit_pct * (active_level.lock_percent / 100)
        guaranteed_profit_pct = active_level.guaranteed_profit - (self.transaction_cost * 100)

        # Calculate new stop loss price
        new_stop_loss = entry_price * (1 + guaranteed_profit_pct / 100)

        # Check if we should update stop loss
        should_update = False
        message = ""

        if new_level > current_level:
            # New level achieved
            self._active_locks[symbol] = new_level
            should_update = new_stop_loss > current_stop_loss
            message = (
                f"🔒 PROFIT LOCK {new_level + 1}: {active_level.description} | "
                f"Profit: {current_profit_pct:+.2f}% | "
                f"Guaranteed: {guaranteed_profit_pct:+.2f}% (after costs)"
            )
            logger.info(f"{symbol}: {message}")
        elif new_level == current_level:
            # Same level, check if stop should be raised
            should_update = new_stop_loss > current_stop_loss
            if should_update:
                message = (
                    f"📈 LOCK UPDATE: Raising stop to lock {locked_profit_pct:.1f}% profit | "
                    f"Guaranteed: {guaranteed_profit_pct:+.2f}%"
                )
            else:
                message = (
                    f"✅ LOCK ACTIVE: Level {new_level + 1} | "
                    f"Profit: {current_profit_pct:+.2f}% | "
                    f"Guaranteed: {guaranteed_profit_pct:+.2f}%"
                )

        return self._create_status(
            symbol=symbol,
            current_profit_pct=current_profit_pct,
            active_level=new_level,
            locked_profit_pct=locked_profit_pct,
            guaranteed_profit_pct=guaranteed_profit_pct,
            stop_loss_price=new_stop_loss if should_update else current_stop_loss,
            original_entry_price=entry_price,
            message=message,
            should_update_stop=should_update,
        )

    def _get_levels_for_beta(self, beta: Optional[float]) -> List[ProfitLockLevel]:
        """Get appropriate profit lock levels based on beta."""
        if not self.use_beta_adjustment or beta is None:
            return self.default_levels

        if beta > self.high_beta_threshold:
            return self.HIGH_BETA_LEVELS
        elif beta < self.low_beta_threshold:
            return self.LOW_BETA_LEVELS
        else:
            return self.default_levels

    def _create_status(
        self,
        symbol: str,
        current_profit_pct: float,
        active_level: Optional[int],
        locked_profit_pct: float,
        guaranteed_profit_pct: float,
        stop_loss_price: float,
        original_entry_price: float,
        message: str,
        should_update_stop: bool,
    ) -> ProfitLockStatus:
        """Create a ProfitLockStatus object."""
        return ProfitLockStatus(
            symbol=symbol,
            current_profit_pct=current_profit_pct,
            active_level=active_level,
            locked_profit_pct=locked_profit_pct,
            guaranteed_profit_pct=guaranteed_profit_pct,
            stop_loss_price=stop_loss_price,
            original_entry_price=original_entry_price,
            message=message,
            should_update_stop=should_update_stop,
        )

    def get_lock_status(self, symbol: str) -> Optional[int]:
        """Get current lock level for a symbol."""
        return self._active_locks.get(symbol.upper())

    def clear_lock(self, symbol: str) -> None:
        """Clear lock tracking for a symbol (after position closed)."""
        symbol = symbol.upper()
        if symbol in self._active_locks:
            del self._active_locks[symbol]
            logger.debug(f"Cleared profit lock for {symbol}")

    def clear_all_locks(self) -> int:
        """Clear all lock tracking. Returns count of cleared locks."""
        count = len(self._active_locks)
        self._active_locks.clear()
        return count

    def get_all_active_locks(self) -> Dict[str, int]:
        """Get all active locks."""
        return self._active_locks.copy()

    def get_level_info(self, level: int, beta: Optional[float] = None) -> Optional[ProfitLockLevel]:
        """Get information about a specific lock level."""
        levels = self._get_levels_for_beta(beta)
        if 0 <= level < len(levels):
            return levels[level]
        return None


# Singleton instance
_profit_lock_instance: Optional[ProfitLockMechanism] = None


def get_profit_lock_mechanism() -> ProfitLockMechanism:
    """Get singleton instance of profit lock mechanism."""
    global _profit_lock_instance
    if _profit_lock_instance is None:
        _profit_lock_instance = ProfitLockMechanism()
    return _profit_lock_instance


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
