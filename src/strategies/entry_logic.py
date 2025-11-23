# -*- coding: utf-8 -*-
"""
improved_entry_logic.py - Enhanced Entry Signal Logic
Cải thiện logic vào lệnh với nhiều điều kiện hơn
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Import utilities
from src.monitoring.performance import get_performance_monitor
from src.utils.indicators import IndicatorUtils, StopLossCalculator
from src.utils.validation import DataValidator
from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Độ mạnh của tín hiệu"""

    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NO_SIGNAL = 0


@dataclass
class EntrySignal:
    """Container cho entry signal"""

    should_enter: bool
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: int  # 0-100
    strength: SignalStrength
    position_size_multiplier: float  # 0.0 - 1.5
    reasons: list
    warnings: list
    entry_price: float
    stop_loss: float
    take_profit_targets: list
    # NEW: Limit order support
    is_limit_order: bool = False  # True if should use limit order instead of market
    limit_price: Optional[float] = None  # Limit price if is_limit_order = True
    entry_type: str = "MARKET"  # 'MARKET', 'LIMIT', 'PULLBACK', 'BREAKOUT'
    telemetry: Optional[Dict] = None  # Chi tiết chấm điểm để debug/monitor


class ImprovedEntryLogic:
    """
    Logic vào lệnh nâng cao với 12 filters:

    1. Market Regime - Thị trường phải tradeable
    2. Trend Alignment - EMA alignment (20/50/200)
    3. Support/Resistance - Vào gần support, tránh resistance
    4. Volume Confirmation - Volume + OBV + trend
    5. Liquidity Check - Tiered thresholds (large/mid/small/micro caps)
    6. Volatility Filter - ATR/Price trong range hợp lý
    7. RSI Check - Tránh overbought
    8. Price Action - Candlestick patterns
    9. Sector Strength - Relative strength vs market
    10. Multi-Timeframe - Weekly/monthly trend confirmation
    11. Market Breadth - Advance/decline ratio
    12. Portfolio Correlation - Đa dạng hóa portfolio

    Features:
    - Dynamic penalty scaling based on market regime (BULL/BEAR/SIDEWAYS)
    - ML fallback to technical analysis when ML signal unavailable
    - Tiered liquidity thresholds for different market cap sizes
    """

    def __init__(
        self,
        min_confidence: int = 60,
        min_risk_reward: float = 2.0,
        support_distance_percent: float = 3.0,
        require_trend_alignment: bool = True,
        require_volume_confirmation: bool = True,
        regime_aware_filtering: bool = True,  # NEW: Relax filters in BULL/SIDEWAYS markets
        portfolio_manager=None,
        performance_monitor=None,
        min_liquidity_value: float = 5_000_000_000,  # 5B VND daily value (for large caps)
        min_avg_volume: int = 150_000,
        use_tiered_liquidity: bool = True,  # Enable tiered liquidity thresholds
    ):
        """
        Args:
            min_confidence: Confidence tối thiểu để vào lệnh
            min_risk_reward: R:R ratio tối thiểu
            support_distance_percent: Khoảng cách tối đa đến support (%)
            require_trend_alignment: Yêu cầu phải theo trend
            require_volume_confirmation: Yêu cầu volume confirm
            regime_aware_filtering: NEW - Relax certain filters in BULL/SIDEWAYS (reduces false negatives)
            portfolio_manager: Portfolio manager for context-aware decisions
        """
        self.min_confidence = min_confidence
        self.base_min_confidence = min_confidence  # Store original for dynamic adjustment
        self.min_risk_reward = min_risk_reward
        self.support_distance_percent = support_distance_percent
        self.require_trend_alignment = require_trend_alignment
        self.require_volume_confirmation = require_volume_confirmation
        self.regime_aware_filtering = regime_aware_filtering
        self.portfolio_manager = portfolio_manager
        self.performance_monitor = performance_monitor or get_performance_monitor()
        self.min_liquidity_value = min_liquidity_value
        self.min_avg_volume = min_avg_volume
        self.use_tiered_liquidity = use_tiered_liquidity
        self._current_symbol = None

        # Tiered liquidity thresholds from config (replaces hardcoded values)
        from src.config.strategy_config import get_strategy_config

        strategy_config = get_strategy_config()
        self.liquidity_tiers = {
            "large": strategy_config.entry.liquidity_tiers.large_cap,
            "mid": strategy_config.entry.liquidity_tiers.mid_cap,
            "small": strategy_config.entry.liquidity_tiers.small_cap,
        }

        # CRITICAL FIX: Track ML vs Technical-only signals
        self._is_technical_only = False  # Flag to track signal source

        # OPTIMIZATION: Correlation matrix cache to prevent redundant calculations
        self._correlation_cache = None
        self._correlation_cache_time = None
        self._correlation_cache_symbols = None
        self._correlation_cache_ttl = 300  # 5 minutes TTL (default)
        self._correlation_cache_portfolio_hash = None  # NEW: Track portfolio composition

    def _validate_initial_signal(
        self, df: pd.DataFrame, ml_signal: Optional[Dict]
    ) -> tuple[bool, str, float, float]:
        """
        Validate initial data and ML signal
        ENHANCED: Allow fallback to technical analysis when ML signal is None

        Returns:
            (is_valid, signal_type, base_confidence, current_price) or
            (False, reason, 0, 0) if invalid
        """
        try:
            DataValidator.validate_dataframe(df, min_rows=50)
        except Exception as e:
            return (False, f"Data validation failed: {str(e)}", 0, 0)

        # Use safe access instead of df.iloc[-1]
        from utils.dataframe_utils import safe_get_latest

        close_price = safe_get_latest(df, "close", 0)

        # ENHANCEMENT: Fallback to technical analysis if ML signal is None
        # CRITICAL: Track whether signal is ML-based or technical-only
        if ml_signal is None:
            logger.warning(
                "⚠️ ML signal is None - using technical analysis fallback. "
                "This will be tracked separately for performance analysis."
            )
            # Use technical indicators to generate a fallback signal
            base_confidence = self._calculate_technical_confidence(df)

            # IMPROVED: Raise threshold to 50% for technical-only signals
            # Technical analysis should meet same minimum standards as ML
            if base_confidence < 50:  # Raised from 40% to 50%
                return (False, f"Technical confidence thấp ({base_confidence}%)", 0, 0)

            # Determine signal type from technical analysis
            signal_type = self._get_technical_signal(df)
            if signal_type != "BUY":
                return (False, f"Technical signal = {signal_type}", 0, 0)

            # Mark this as technical-only signal for tracking
            # This will be added to metadata in analyze_entry()
            self._is_technical_only = True

            return (True, signal_type, base_confidence, close_price)
        else:
            # ML signal available - mark as ML-based
            self._is_technical_only = False

        signal_type = ml_signal.get("signal", "HOLD")
        base_confidence = ml_signal.get("confidence", 0)

        # Skip if not BUY signal
        if signal_type != "BUY":
            return (False, f"Signal = {signal_type}", 0, 0)

        # Skip if confidence low
        if base_confidence < self.min_confidence:
            return (False, f"Confidence thấp ({base_confidence}%)", 0, 0)

        return (True, signal_type, base_confidence, close_price)

    def _add_adjustment(
        self,
        adjustments: List[int],
        breakdown: List[Dict],
        filter_name: str,
        delta: int,
        note: str,
    ):
        """Add adjustment và lưu telemetry cho filter"""
        adjustments.append(delta)
        breakdown.append(
            {
                "filter": filter_name,
                "delta": delta,
                "note": note,
            }
        )

    def _run_all_filters(
        self,
        df: pd.DataFrame,
        signal_type: str,
        current_price: float,
        market_regime: Optional[Dict],
    ) -> tuple[bool, list, list, list, list]:
        """
        Run all entry filters

        Returns:
            (passed, reasons, warnings, adjustments)
        """
        reasons = []
        warnings = []
        adjustments = []
        adjustment_breakdown = []

        # Determine adjustment scaling factor based on market regime
        # BULL: Scale penalties down (0.7x) to allow more signals
        # BEAR/HIGH_VOL: Scale penalties up (1.2x) to be more selective
        # SIDEWAYS: Normal (1.0x)
        adjustment_scale = 1.0
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            regime_confidence = market_regime.get("confidence", 50)

            if regime == "BULL" and regime_confidence >= 70:
                adjustment_scale = 0.7  # Lighter penalties in strong bull market
            elif regime == "BEAR":
                adjustment_scale = 1.2  # Heavier penalties in bear market
            elif regime == "HIGH_VOLATILITY":
                adjustment_scale = 1.3  # Even heavier in high volatility
            # SIDEWAYS or other: keep 1.0

        # FILTER 1: MARKET REGIME
        if market_regime and not market_regime.get("tradeable", True):
            adjustment_breakdown.append(
                {
                    "filter": "market_regime",
                    "delta": None,
                    "note": "Market regime not tradeable",
                }
            )
            return (
                False,
                [],
                [],
                [],
                adjustment_breakdown,
            )

        # FILTER 2: TREND ALIGNMENT
        trend_check = self._check_trend_alignment(df, signal_type)
        if not trend_check["aligned"]:
            adjustment_breakdown.append(
                {
                    "filter": "trend_alignment",
                    "delta": None,
                    "note": trend_check["reason"],
                }
            )
            if self.require_trend_alignment:
                return (False, [], [], [], adjustment_breakdown)
            else:
                warnings.append(f"⚠️ Trend: {trend_check['reason']}")
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "trend_alignment",
                    -10,
                    trend_check["reason"],
                )
        else:
            reasons.append(f"✅ Trend: {trend_check['reason']}")
            if trend_check["strength"] > 50:
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "trend_alignment",
                    +5,
                    "Strong alignment",
                )

        # FILTER 3: SUPPORT/RESISTANCE
        sr_check = self._check_support_resistance(df, current_price)
        if sr_check["too_close_to_resistance"]:
            warning_msg = f"⚠️ Gần resistance: {sr_check['distance_to_resistance']:.1f}%"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "support_resistance",
                -15,
                warning_msg,
            )
        elif sr_check["bouncing_from_support"]:
            # Bouncing from support is a STRONG reversal signal
            reasons.append(
                f"✅ Bouncing from support (+{sr_check['distance_to_support']:.1f}%) - REVERSAL"
            )
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "support_resistance",
                +15,
                "Bouncing from support",
            )
        elif sr_check["near_support"]:
            reasons.append(f"✅ Gần support (+{sr_check['distance_to_support']:.1f}%)")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "support_resistance",
                +10,
                "Near support",
            )

        # FILTER 4: VOLUME CONFIRMATION
        # ENHANCEMENT: Pass market_regime for dynamic threshold adjustment
        # NEW: Relax volume requirement in BULL/SIDEWAYS if regime_aware_filtering enabled
        # IMPROVEMENT: Also relax for small caps with low liquidity
        volume_check = self._check_volume_confirmation(df, market_regime)

        # Check if this is a small cap (from Filter 5 results, or check now)
        is_small_cap = False
        liquidity_tier = "unknown"
        if "volume" in df.columns and len(df) >= 5:
            avg_volume = df["volume"].tail(20).mean()
            avg_value = avg_volume * current_price
            # Small cap threshold: < 5B VND daily value
            if avg_value < 5_000_000_000:
                is_small_cap = True
                liquidity_tier = "small/micro"

        if not volume_check["confirmed"]:
            volume_note = volume_check["reason"]
            adjustment_breakdown.append(
                {
                    "filter": "volume",
                    "delta": None,
                    "note": volume_note,
                }
            )

            # Regime-aware filtering: Relax volume in BULL/SIDEWAYS
            regime_name = market_regime.get("regime", "SIDEWAYS") if market_regime else "SIDEWAYS"
            should_block_volume = self.require_volume_confirmation

            if self.regime_aware_filtering and regime_name in ["BULL", "SIDEWAYS"]:
                # In BULL/SIDEWAYS: Volume becomes optional (warning only)
                should_block_volume = False
                logger.debug(f"📊 Relaxing volume filter in {regime_name} market (regime-aware)")

            # IMPROVEMENT: Also relax for small caps
            if is_small_cap:
                should_block_volume = False
                logger.debug(
                    f"📊 Relaxing volume filter for small cap ({liquidity_tier}) - "
                    "low liquidity expected"
                )

            if should_block_volume:
                return (False, [], [], [], adjustment_breakdown)
            else:
                warning_msg = f"⚠️ Volume: {volume_note}"
                if is_small_cap:
                    warning_msg += f" (small cap - {liquidity_tier})"
                warnings.append(warning_msg)
                # IMPROVEMENT: Smaller penalty for small caps (-5 instead of -10)
                penalty = -5 if is_small_cap else -10
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "volume",
                    penalty,
                    volume_note,
                )
        else:
            reasons.append(f"✅ Volume: {volume_check['reason']}")
            if volume_check["surge"]:
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "volume",
                    +5,
                    "Volume surge",
                )

        # FILTER 5: LIQUIDITY CHECK (ENHANCED with Vietnam market requirements)
        liquidity_check = self._check_liquidity(df, current_price)
        if liquidity_check["critical"]:
            adjustment_breakdown.append(
                {
                    "filter": "liquidity",
                    "delta": None,
                    "note": "Critical liquidity",
                }
            )
            return (
                False,
                [],
                [],
                [],
                adjustment_breakdown,
            )

        # FILTER 5a: VIETNAM MARKET LIQUIDITY CHECK (NEW)
        vn_liquidity_check = self._check_vietnam_market_liquidity(df)
        if not vn_liquidity_check["sufficient"]:
            adjustment_breakdown.append(
                {
                    "filter": "vietnam_liquidity",
                    "delta": None,
                    "note": vn_liquidity_check["reason"],
                }
            )
            return (
                False,
                [],
                [],
                [],
                adjustment_breakdown,
            )
        elif not liquidity_check["sufficient"]:
            tier = liquidity_check.get("tier", "unknown")
            warning_msg = (
                f"⚠️ Thanh khoản thấp ({tier} cap) "
                f"(avg value: {liquidity_check['avg_value'] / 1_000_000_000:.2f}B VND)"
            )
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "liquidity",
                -15,
                warning_msg,
            )
        else:
            tier = liquidity_check.get("tier", "unknown")
            reasons.append(
                f"✅ Thanh khoản tốt ({tier} cap) "
                f"(avg value: {liquidity_check['avg_value'] / 1_000_000_000:.1f}B VND)"
            )
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "liquidity",
                +5,
                "Good liquidity",
            )

        # FILTER 6: VOLATILITY CHECK
        volatility_check = self._check_volatility(df)
        if volatility_check["too_high"]:
            warning_msg = f"⚠️ Volatility cao: {volatility_check['value']:.2f}%"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "volatility",
                -15,
                warning_msg,
            )
        elif volatility_check["optimal"]:
            reasons.append("✅ Volatility vừa phải")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "volatility",
                +5,
                "Optimal volatility",
            )

        # FILTER 7: RSI CHECK
        rsi_check = self._check_rsi(df)
        if rsi_check["overbought"]:
            warning_msg = f"⚠️ RSI overbought: {rsi_check['value']:.1f}"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "rsi",
                -10,
                warning_msg,
            )
        elif rsi_check["oversold"]:
            # Oversold RSI (<30) is a STRONG buy signal
            reasons.append(f"✅ RSI oversold: {rsi_check['value']:.1f} (strong buy)")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "rsi",
                +15,
                "Oversold RSI",
            )
        elif rsi_check["optimal"]:
            reasons.append(f"✅ RSI: {rsi_check['value']:.1f}")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "rsi",
                +5,
                "Optimal RSI",
            )

        # FILTER 8: PRICE ACTION
        price_action = self._check_price_action(df)
        if price_action["bullish_pattern"]:
            pattern_note = f"Pattern: {price_action['pattern']}"
            reasons.append(f"✅ {pattern_note}")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "price_action",
                +10,
                pattern_note,
            )
        elif price_action["bearish_pattern"]:
            warning_msg = f"⚠️ Pattern: {price_action['pattern']}"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "price_action",
                -10,
                warning_msg,
            )

        # FILTER 9: SECTOR STRENGTH
        sector_strength_check = self._check_sector_strength(df, market_regime)
        if sector_strength_check["is_leading"]:
            reason_msg = f"Ngành dẫn dắt ({sector_strength_check['sector_perf']:.1f}%)"
            reasons.append(f"✅ {reason_msg}")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "sector_strength",
                +10,
                reason_msg,
            )
        elif sector_strength_check["is_lagging"]:
            warning_msg = f"⚠️ Ngành yếu ({sector_strength_check['sector_perf']:.1f}%)"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "sector_strength",
                -15,
                warning_msg,
            )

        # FILTER 10: MULTI-TIMEFRAME CONFIRMATION
        mtf_check = self._check_multi_timeframe_trend(df)
        if not mtf_check["weekly_up"]:
            warning_msg = f"⚠️ Weekly trend yếu ({mtf_check['weekly_change']:.1f}%)"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "multi_timeframe",
                -5,
                warning_msg,
            )
        else:
            reasons.append(f"✅ Weekly trend tăng ({mtf_check['weekly_change']:+.1f}%)")
        if not mtf_check["monthly_up"]:
            warning_msg = f"⚠️ Monthly trend yếu ({mtf_check['monthly_change']:.1f}%)"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "multi_timeframe",
                -5,
                warning_msg,
            )
        else:
            reasons.append(f"✅ Monthly trend tăng ({mtf_check['monthly_change']:+.1f}%)")

        # FILTER 11: MARKET BREADTH
        breadth_check = self._check_market_breadth(market_regime)
        if breadth_check["weak"]:
            warning_msg = "⚠️ Market breadth yếu (ít mã tham gia tăng)"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "market_breadth",
                -10,
                warning_msg,
            )
        elif breadth_check["strong"]:
            reasons.append("✅ Market breadth mạnh (nhiều mã tham gia)")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "market_breadth",
                +5,
                "Market breadth strong",
            )

        # FILTER 12: PORTFOLIO CORRELATION
        correlation_check = self._check_portfolio_correlation(
            df, getattr(self, "_current_symbol", None)
        )
        if correlation_check["too_high"]:
            warning_msg = (
                f"⚠️ Correlation cao với portfolio: {correlation_check['max_correlation']:.2f}"
            )
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "portfolio_correlation",
                -20,
                warning_msg,
            )  # Penalty lớn cho high correlation
        elif correlation_check["good_diversification"]:
            reasons.append(f"✅ Đa dạng hóa tốt (corr: {correlation_check['max_correlation']:.2f})")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "portfolio_correlation",
                +5,
                "Good diversification",
            )

        # Apply scaling factor to all adjustments (only to penalties, not bonuses)
        # This prevents confidence from dropping too fast in favorable markets
        if adjustment_scale != 1.0:
            scaled_adjustments = []
            for idx, adj in enumerate(adjustments):
                if adj < 0:  # Only scale penalties (negative adjustments)
                    new_adj = int(adj * adjustment_scale)
                    scaled_adjustments.append(new_adj)
                    if idx < len(adjustment_breakdown):
                        adjustment_breakdown[idx]["delta"] = new_adj
                        adjustment_breakdown[idx]["note"] += " (scaled)"
                else:  # Keep bonuses unchanged
                    scaled_adjustments.append(adj)
            adjustments = scaled_adjustments

        return (True, reasons, warnings, adjustments, adjustment_breakdown)

    def _calculate_prices_and_risk(
        self, df: pd.DataFrame, entry_price: float, sr_check: Dict
    ) -> tuple[bool, str, float, float, list, float]:
        """
        Calculate entry price, stop loss, take profit targets, and risk/reward

        Returns:
            (success, error_msg, stop_loss, reward, take_profit_targets,
             risk_reward)
        """
        atr = IndicatorUtils.get_atr(df)
        support_level = sr_check.get("support_level", None)

        # Calculate stop loss
        try:
            stop_loss, sl_reason = StopLossCalculator.calculate_stop_loss(
                entry_price=entry_price,
                atr=atr,
                support_level=support_level,
                atr_multiplier=2.0,
            )
            logger.debug(f"Stop loss calculated: {stop_loss:.0f} ({sl_reason})")

            # CRITICAL FIX: Validate stop loss is within acceptable range
            # This ensures stop loss is ALWAYS properly set
            if stop_loss is None or stop_loss <= 0:
                return (False, f"Stop loss invalid: {stop_loss}", 0, 0, [], 0)

            # Ensure stop loss is below entry (for long positions)
            if stop_loss >= entry_price:
                return (False, f"Stop loss ({stop_loss:.0f}) must be below entry ({entry_price:.0f})", 0, 0, [], 0)

            # Enforce minimum stop loss distance (3% of entry price)
            min_stop_distance = entry_price * 0.03  # 3% minimum
            if (entry_price - stop_loss) < min_stop_distance:
                stop_loss = entry_price - min_stop_distance
                logger.warning(f"⚠️ Stop loss too tight, adjusted to 3% below entry: {stop_loss:.0f}")

            # Enforce maximum stop loss distance (10% of entry price)
            max_stop_distance = entry_price * 0.10  # 10% maximum
            if (entry_price - stop_loss) > max_stop_distance:
                stop_loss = entry_price - max_stop_distance
                logger.warning(f"⚠️ Stop loss too wide, adjusted to 10% below entry: {stop_loss:.0f}")

        except ValueError as e:
            return (False, f"Stop loss calculation failed: {str(e)}", 0, 0, [], 0)

        # Calculate take profit targets
        try:
            take_profit_targets = StopLossCalculator.calculate_take_profit_targets(
                entry_price=entry_price, atr=atr, risk_reward_ratios=[1.5, 3.0, 5.0]
            )
        except ValueError as e:
            return (False, f"Take profit calculation failed: {str(e)}", 0, 0, [], 0)

        # Risk/Reward check
        risk = entry_price - stop_loss
        if risk <= 0:
            error_msg = (
                f"Risk calculation error: risk={risk:.0f} "
                f"(entry={entry_price:.0f}, sl={stop_loss:.0f})"
            )
            return (False, error_msg, 0, 0, [], 0)

        if len(take_profit_targets) < 2:
            return (
                False,
                "Không đủ take profit targets để tính reward",
                0,
                0,
                take_profit_targets,
                0,
            )

        reward = take_profit_targets[1] - entry_price  # Use TP2
        if reward <= 0:
            return (False, f"Reward không hợp lệ: {reward:.0f}", 0, 0, [], 0)

        risk_reward = reward / risk
        if risk_reward < self.min_risk_reward:
            error_msg = f"R:R ratio thấp: {risk_reward:.2f} < " f"{self.min_risk_reward:.2f}"
            return (False, error_msg, 0, 0, [], 0)

        return (True, "", stop_loss, reward, take_profit_targets, risk_reward)

    def analyze_entry(
        self,
        df: pd.DataFrame,
        ml_signal: Dict,
        market_regime: Optional[Dict] = None,
        symbol: Optional[str] = None,
    ) -> EntrySignal:
        """
        Phân tích đầy đủ để quyết định có nên vào lệnh

        Args:
            df: DataFrame với OHLCV + indicators
            ml_signal: Signal từ ML model
            market_regime: Thông tin market regime (optional)

        Returns:
            EntrySignal object với đầy đủ thông tin
        """
        # ENHANCEMENT: Adjust thresholds dynamically based on market regime
        self._adjust_thresholds_for_market(market_regime)

        self._current_symbol = symbol
        try:
            # Step 1: Validate initial signal
            (
                is_valid,
                signal_or_reason,
                base_confidence,
                current_price,
            ) = self._validate_initial_signal(df, ml_signal)
            if not is_valid:
                return self._no_signal(signal_or_reason)

            signal_type = signal_or_reason

            # Step 2: Run all filters
            (
                passed,
                reasons,
                warnings,
                adjustments,
                adjustment_breakdown,
            ) = self._run_all_filters(df, signal_type, current_price, market_regime)
            if not passed:
                regime_name = market_regime.get("regime", "UNKNOWN") if market_regime else "N/A"
                return self._no_signal(
                    f"Thị trường: {regime_name}",
                    telemetry={
                        "base_confidence": base_confidence,
                        "adjustments": adjustment_breakdown,
                        "reason": "Filters rejected",
                    },
                )

            # Step 3: Calculate adjusted confidence
            confidence_after_filters = base_confidence + sum(adjustments)
            confidence_after_filters = max(0, min(confidence_after_filters, 100))
            adjusted_confidence = confidence_after_filters

            # Step 3b: Apply performance feedback
            adjusted_confidence, perf_msg = self._apply_performance_feedback(adjusted_confidence)
            if perf_msg:
                if perf_msg.startswith("⚠️"):
                    warnings.append(perf_msg)
                else:
                    reasons.append(perf_msg)

            telemetry = {
                "base_confidence": base_confidence,
                "adjustments": adjustment_breakdown,
                "confidence_after_filters": confidence_after_filters,
                "confidence_after_performance": adjusted_confidence,
                "min_confidence_threshold": self.min_confidence,
                "performance_feedback": perf_msg,
                "market_regime": market_regime,
                # CRITICAL FIX: Track signal source for performance analysis
                "signal_source": "technical_only" if self._is_technical_only else "ml",
                "is_technical_only": self._is_technical_only,
            }

            if adjusted_confidence < self.min_confidence:
                return self._no_signal(
                    f"Confidence sau adjustment: {adjusted_confidence}% < {self.min_confidence}%",
                    telemetry=telemetry,
                )

            # Step 4: Calculate prices and risk/reward
            # ENHANCEMENT: Optimize entry price (pullback, breakout, or current)
            close_price = safe_get_latest(df, "close", 0)
            sr_check = self._check_support_resistance(df, current_price)

            # Optimize entry price based on market conditions
            optimized_entry = self._optimize_entry_price(df, close_price, sr_check, market_regime)
            entry_price = DataValidator.validate_price(
                optimized_entry["entry_price"], "entry_price"
            )

            # ENHANCEMENT: Check if should use limit order
            is_limit_order = optimized_entry.get("entry_type") in ["PULLBACK", "BREAKOUT"]
            limit_price = optimized_entry.get("entry_price") if is_limit_order else None

            # Only use limit order if entry price is significantly different from current
            price_diff_pct = (
                abs(entry_price - close_price) / close_price * 100 if close_price > 0 else 0
            )
            if is_limit_order and price_diff_pct < 0.5:
                # Entry price too close to current - use market order instead
                is_limit_order = False
                limit_price = None
                entry_price = close_price
                optimized_entry["entry_type"] = "MARKET"

            # Add entry price optimization info to reasons if applicable
            if optimized_entry.get("entry_type") != "MARKET":
                entry_reason = f"Entry: {optimized_entry['entry_type']}"
                if optimized_entry.get("optimization_reason"):
                    entry_reason += f" ({optimized_entry['optimization_reason']})"
                if is_limit_order:
                    entry_reason += f" [LIMIT @ {limit_price:,.0f}]"
                reasons.append(f"✅ {entry_reason}")

            (
                success,
                error_msg,
                stop_loss,
                reward,
                take_profit_targets,
                risk_reward,
            ) = self._calculate_prices_and_risk(df, entry_price, sr_check)
            if not success:
                return self._no_signal(error_msg)

            reasons.append(f"✅ R:R ratio: {risk_reward:.2f}")
            telemetry["risk_reward"] = risk_reward

            # Step 5: Determine signal strength and position multiplier
            strength = self._calculate_signal_strength(adjusted_confidence, risk_reward, warnings)
            position_multiplier = self._calculate_position_multiplier(
                strength, adjusted_confidence, warnings, market_regime
            )

            # Step 6: Build entry signal
            entry_signal = EntrySignal(
                should_enter=True,
                signal_type="BUY",
                confidence=int(adjusted_confidence),
                strength=strength,
                position_size_multiplier=position_multiplier,
                reasons=reasons,
                warnings=warnings,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_targets=take_profit_targets,
                # NEW: Limit order support
                is_limit_order=is_limit_order,
                limit_price=limit_price,
                entry_type=optimized_entry.get("entry_type", "MARKET"),
                telemetry=telemetry,
            )

            # IMPROVEMENT: Record signal for performance tracking
            try:
                from src.monitoring.signal_performance_tracker import get_signal_tracker

                tracker = get_signal_tracker()
                signal_source = "technical_only" if self._is_technical_only else "ml"
                tracker.record_signal(
                    symbol=symbol or "UNKNOWN",
                    signal_source=signal_source,
                    confidence=adjusted_confidence,
                    entry_price=entry_price,
                )
            except Exception as e:
                logger.warning(f"Failed to record signal for tracking: {e}")

            # CRITICAL FIX: Log signal source for tracking
            if self._is_technical_only:
                logger.info(
                    f"📊 [{symbol}] Technical-only signal generated "
                    f"(confidence: {adjusted_confidence}%). "
                    "Will track performance separately."
                )
            else:
                logger.debug(
                    f"🤖 [{symbol}] ML-based signal generated "
                    f"(confidence: {adjusted_confidence}%)"
                )

            return entry_signal
        finally:
            self._current_symbol = None
            self._is_technical_only = False  # Reset flag

    # ========================================================================
    # HELPER METHODS - FILTERS
    # ========================================================================

    def _check_trend_alignment(self, df: pd.DataFrame, signal_type: str) -> Dict:
        """
        Check xem signal có align với trend không

        Trend = EMA20 vs EMA50 vs EMA200
        """
        if len(df) < 200:
            return {
                "aligned": True,
                "reason": "Chưa đủ data để check trend",
                "strength": 50,
            }

        ema20 = df["close"].ewm(span=20).mean()
        ema50 = df["close"].ewm(span=50).mean()
        ema200 = df["close"].ewm(span=200).mean()

        latest_price = safe_get_latest(df, "close", 0)
        latest_ema20 = ema20.iloc[-1]
        latest_ema50 = ema50.iloc[-1]
        latest_ema200 = ema200.iloc[-1]

        if signal_type == "BUY":
            # Perfect alignment: Price > EMA20 > EMA50 > EMA200
            perfect = latest_price > latest_ema20 > latest_ema50 > latest_ema200
            good = latest_price > latest_ema20 > latest_ema50
            ok = latest_price > latest_ema20

            # ENHANCEMENT: Check for early reversal signals
            # Price crossing above EMA (potential reversal)
            prev_price = df["close"].iloc[-2] if len(df) >= 2 else latest_price
            prev_ema20 = ema20.iloc[-2] if len(ema20) >= 2 else latest_ema20
            prev_ema50 = ema50.iloc[-2] if len(ema50) >= 2 else latest_ema50

            # Price just crossed above EMA20 (reversal signal)
            price_cross_ema20 = prev_price <= prev_ema20 and latest_price > latest_ema20
            # EMA20 just crossed above EMA50 (trend turning)
            ema20_cross_ema50 = prev_ema20 <= prev_ema50 and latest_ema20 > latest_ema50

            if perfect:
                strength = 100
                return {
                    "aligned": True,
                    "reason": "Perfect uptrend",
                    "strength": strength,
                }
            elif good:
                strength = 75
                return {
                    "aligned": True,
                    "reason": "Strong uptrend",
                    "strength": strength,
                }
            elif price_cross_ema20 or ema20_cross_ema50:
                # Early reversal signal - can catch trends early
                strength = 60
                reason = "Early reversal signal"
                if price_cross_ema20:
                    reason += " (Price crossed EMA20)"
                if ema20_cross_ema50:
                    reason += " (EMA20 crossed EMA50)"
                return {
                    "aligned": True,
                    "reason": reason,
                    "strength": strength,
                }
            elif ok:
                strength = 50
                return {
                    "aligned": True,
                    "reason": "Short-term uptrend",
                    "strength": strength,
                }
            else:
                return {
                    "aligned": False,
                    "reason": "Downtrend or sideway",
                    "strength": 0,
                }

        return {"aligned": True, "reason": "Unknown signal type", "strength": 50}

    def _check_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Check vị trí giá so với support/resistance

        Support: Low của 20 ngày
        Resistance: High của 20 ngày

        Enhanced: Check if price is bouncing FROM support (reversal signal)
        """
        if len(df) < 20:
            return {
                "near_support": False,
                "bouncing_from_support": False,
                "too_close_to_resistance": False,
                "support_level": 0,
                "resistance_level": 0,
                "distance_to_support": 0,
                "distance_to_resistance": 0,
            }

        support = safe_rolling_operation(df, "low", 20, "min", 0)
        resistance = safe_rolling_operation(df, "high", 20, "max", 0)

        distance_to_support = ((current_price - support) / support) * 100
        distance_to_resistance = ((resistance - current_price) / current_price) * 100

        # Near support = trong vòng config threshold
        near_support = distance_to_support <= self.support_distance_percent

        # ENHANCEMENT: Check if price is bouncing FROM support
        # This is a stronger signal than just being near support
        # CRITICAL FIX: Improved detection with volume confirmation and sustained upward movement
        bouncing_from_support = False
        if near_support and len(df) >= 5:
            # Check if price touched/near support recently and is now moving up
            recent_low = safe_rolling_operation(df, "low", 5, "min", 0)

            # Calculate 3-bar average for sustained movement check
            prev_3_avg = df["close"].iloc[-4:-1].mean() if len(df) >= 4 else current_price

            # Price was near support in last 5 days
            if abs(recent_low - support) / support < 0.02:  # Within 2% of support
                # Check for sustained upward movement (1% above 3-bar average)
                if current_price > prev_3_avg * 1.01:
                    # Volume confirmation: current volume > 1.2x recent average
                    current_volume = safe_get_latest(df, "volume", 0)
                    avg_volume_5 = safe_rolling_operation(df, "volume", 5, "mean", 1)

                    if avg_volume_5 > 0 and current_volume > avg_volume_5 * 1.2:
                        bouncing_from_support = True
                        logger.debug(
                            f"✅ Support bounce detected: price {current_price:.0f} > "
                            f"3-bar avg {prev_3_avg:.0f} (+{((current_price/prev_3_avg - 1)*100):.1f}%), "
                            f"volume {current_volume/avg_volume_5:.1f}x"
                        )

        # Too close to resistance = trong vòng 2%
        too_close = distance_to_resistance <= 2

        return {
            "near_support": near_support,
            "bouncing_from_support": bouncing_from_support,
            "too_close_to_resistance": too_close,
            "support_level": support,
            "resistance_level": resistance,
            "distance_to_support": distance_to_support,
            "distance_to_resistance": distance_to_resistance,
        }

    def _check_liquidity(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        NEW: Kiểm tra thanh khoản (giá * volume) với tiered thresholds
        """
        if "volume" not in df.columns or len(df) < 5:
            return {
                "sufficient": True,
                "critical": False,
                "current_value": 0.0,
                "avg_value": 0.0,
                "tier": "unknown",
            }

        current_volume = safe_get_latest(df, "volume", 0)
        avg_volume = df["volume"].tail(20).mean()
        avg_value = avg_volume * current_price
        current_value = current_volume * current_price

        # Determine appropriate tier and thresholds
        if self.use_tiered_liquidity:
            # Try from highest to lowest tier
            tier = None
            min_value_threshold = 0
            min_volume_threshold = 0

            if avg_value >= self.liquidity_tiers["large"]["min_value"]:
                tier = "large"
                min_value_threshold = self.liquidity_tiers["large"]["min_value"]
                min_volume_threshold = self.liquidity_tiers["large"]["min_volume"]
            elif avg_value >= self.liquidity_tiers["mid"]["min_value"]:
                tier = "mid"
                min_value_threshold = self.liquidity_tiers["mid"]["min_value"]
                min_volume_threshold = self.liquidity_tiers["mid"]["min_volume"]
            elif avg_value >= self.liquidity_tiers["small"]["min_value"]:
                tier = "small"
                min_value_threshold = self.liquidity_tiers["small"]["min_value"]
                min_volume_threshold = self.liquidity_tiers["small"]["min_volume"]
            else:
                # Below all tiers - use small cap threshold for evaluation
                # NOTE: Micro caps (<1B VND) will fail this threshold check and get -15 penalty,
                # but won't be rejected unless critical (<500M VND). This allows micro caps
                # to still be considered but with lower confidence scores.
                tier = "micro"
                min_value_threshold = self.liquidity_tiers["small"]["min_value"]
                min_volume_threshold = self.liquidity_tiers["small"]["min_volume"]

            sufficient_value = avg_value >= min_value_threshold
            sufficient_volume = avg_volume >= min_volume_threshold
            sufficient = sufficient_value and sufficient_volume
            critical = avg_value < (min_value_threshold * 0.5)
        else:
            # Legacy: Use original fixed thresholds
            tier = "fixed"
            sufficient_value = avg_value >= self.min_liquidity_value
            sufficient_volume = avg_volume >= self.min_avg_volume
            sufficient = sufficient_value and sufficient_volume
            critical = avg_value < (self.min_liquidity_value * 0.5)

        return {
            "sufficient": sufficient,
            "critical": critical,
            "current_value": current_value,
            "avg_value": avg_value,
            "avg_volume": avg_volume,
            "current_volume": current_volume,
            "tier": tier,
        }

    def _check_multi_timeframe_trend(self, df: pd.DataFrame) -> Dict:
        """
        NEW: Confirm xu hướng trên nhiều timeframe (daily/weekly/monthly)
        """
        if len(df) < 5:
            return {
                "weekly_up": True,
                "monthly_up": True,
                "weekly_change": 0.0,
                "monthly_change": 0.0,
            }

        current_close = safe_get_latest(df, "close", 0)
        weekly_close = df["close"].iloc[-5] if len(df) >= 5 else current_close
        monthly_close = df["close"].iloc[-20] if len(df) >= 20 else weekly_close

        weekly_change = ((current_close / weekly_close) - 1) * 100 if weekly_close else 0
        monthly_change = ((current_close / monthly_close) - 1) * 100 if monthly_close else 0

        weekly_up = weekly_change >= 0
        monthly_up = monthly_change >= 0

        return {
            "weekly_up": weekly_up,
            "monthly_up": monthly_up,
            "weekly_change": weekly_change,
            "monthly_change": monthly_change,
        }

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)

        OBV measures buying/selling pressure by adding volume on up days
        and subtracting on down days.

        Returns:
            Series with OBV values
        """
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])

        return pd.Series(obv, index=df.index)

    def _check_volume_confirmation(
        self, df: pd.DataFrame, market_regime: Optional[Dict] = None
    ) -> Dict:
        """
        ENHANCED: Check volume confirmation với multiple indicators và dynamic threshold

        Checks:
        1. Volume ratio (current vs average)
        2. Volume trend (5-day vs 20-day MA)
        3. OBV (On-Balance Volume) - accumulation/distribution

        ENHANCEMENT: Dynamic threshold based on market regime
        - BULL market: Lower threshold (0.4) - more opportunities
        - BEAR/HIGH_VOL: Higher threshold (0.6) - more selective
        - SIDEWAYS: Normal threshold (0.5)

        Returns:
            Dict with detailed volume analysis
        """
        if len(df) < 20:
            return {
                "confirmed": True,
                "reason": "Chưa đủ data",
                "surge": False,
                "obv_bullish": True,
                "volume_trending": True,
                "confidence": 0.5,
            }

        current_volume = safe_get_latest(df, "volume", 0)
        avg_volume_20 = safe_rolling_operation(df, "volume", 20, "mean", 0)

        if avg_volume_20 == 0:
            return {
                "confirmed": True,
                "reason": "Volume data invalid",
                "surge": False,
                "obv_bullish": True,
                "volume_trending": True,
                "confidence": 0.5,
            }

        # ============================================================
        # ENHANCEMENT: Dynamic threshold based on market regime
        # ============================================================
        base_threshold = 0.5  # Default threshold
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            regime_confidence = market_regime.get("confidence", 50)

            if regime == "BULL" and regime_confidence >= 70:
                # Bull market: Lower threshold to catch more opportunities
                base_threshold = 0.4
            elif regime == "BEAR" or regime == "HIGH_VOLATILITY":
                # Bear/high vol: Higher threshold to be more selective
                base_threshold = 0.6
            # SIDEWAYS: Keep default 0.5

        # ============================================================
        # 1. VOLUME RATIO (existing logic)
        # ============================================================
        volume_ratio = current_volume / avg_volume_20

        # ============================================================
        # 2. VOLUME TREND (NEW)
        # ============================================================
        avg_volume_5 = safe_rolling_operation(df, "volume", 5, "mean", 0)
        volume_trending_up = avg_volume_5 > avg_volume_20

        # ============================================================
        # 3. OBV - ACCUMULATION/DISTRIBUTION (NEW)
        # ============================================================
        obv = self._calculate_obv(df)

        # Calculate OBV slope over last 5 days
        if len(obv) >= 5:
            obv_recent = obv.iloc[-5:]
            obv_slope = (obv_recent.iloc[-1] - obv_recent.iloc[0]) / 5

            # Also check OBV moving average
            obv_ma_5 = obv.rolling(5).mean().iloc[-1]
            obv_ma_20 = obv.rolling(20).mean().iloc[-1]

            obv_bullish = (obv_slope > 0) and (obv_ma_5 > obv_ma_20)
        else:
            obv_bullish = True  # Default to True if not enough data

        # ============================================================
        # 4. COMBINE ALL SIGNALS
        # ============================================================

        # Calculate confidence score (0-1)
        confidence_score = 0.0

        # Volume ratio contributes 40%
        if volume_ratio >= 1.5:
            confidence_score += 0.4
        elif volume_ratio >= 1.2:
            confidence_score += 0.3
        elif volume_ratio >= 1.0:
            confidence_score += 0.2

        # Volume trend contributes 30%
        if volume_trending_up:
            confidence_score += 0.3

        # OBV contributes 30%
        if obv_bullish:
            confidence_score += 0.3

        # ENHANCEMENT: Use dynamic threshold instead of fixed 0.5
        confirmed = confidence_score >= base_threshold

        # Generate detailed reason
        reasons = []
        if volume_ratio >= 1.5:
            reasons.append(f"Volume surge {volume_ratio:.1f}x")
        elif volume_ratio >= 1.2:
            reasons.append(f"Volume tăng {volume_ratio:.1f}x")
        else:
            reasons.append(f"Volume {volume_ratio:.1f}x")

        if volume_trending_up:
            reasons.append("Volume trending up")
        else:
            reasons.append("Volume trending down")

        if obv_bullish:
            reasons.append("OBV bullish (accumulation)")
        else:
            reasons.append("OBV bearish (distribution)")

        return {
            "confirmed": confirmed,
            "reason": " | ".join(reasons),
            "surge": volume_ratio >= 1.5,
            "volume_ratio": volume_ratio,
            "volume_trending": volume_trending_up,
            "obv_bullish": obv_bullish,
            "confidence": confidence_score,
        }

    def _check_volatility(self, df: pd.DataFrame) -> Dict:
        """
        Check volatility (ATR/Price)

        < 2%: Too low (no momentum)
        2-3%: Optimal
        > 4%: Too high (risky)
        """
        # Use safe access instead of df.iloc[-1]
        atr = safe_get_latest(df, "atr", 0)
        price = safe_get_latest(df, "close", 0)

        if price == 0:
            return {"too_high": False, "optimal": True, "value": 0}

        volatility = (atr / price) * 100

        if volatility > 4:
            return {"too_high": True, "optimal": False, "value": volatility}
        elif 2 <= volatility <= 3:
            return {"too_high": False, "optimal": True, "value": volatility}
        else:
            return {"too_high": False, "optimal": False, "value": volatility}

    def _check_rsi(self, df: pd.DataFrame) -> Dict:
        """
        Check RSI

        > 70: Overbought (penalty)
        60-70: Neutral
        30-60: Optimal (good for entry)
        < 30: Oversold (strong buy signal)
        """
        if "rsi" not in df.columns:
            return {"overbought": False, "optimal": True, "oversold": False, "value": 50}

        rsi = safe_get_latest(df, "rsi", 0)

        if pd.isna(rsi):
            return {"overbought": False, "optimal": True, "oversold": False, "value": 50}

        if rsi > 70:
            return {"overbought": True, "optimal": False, "oversold": False, "value": rsi}
        elif 30 <= rsi <= 60:
            return {"overbought": False, "optimal": True, "oversold": False, "value": rsi}
        elif rsi < 30:
            return {"overbought": False, "optimal": False, "oversold": True, "value": rsi}
        else:  # 60 < rsi <= 70
            return {"overbought": False, "optimal": False, "oversold": False, "value": rsi}

    def _check_price_action(self, df: pd.DataFrame) -> Dict:
        """
        Check candlestick patterns (simplified)
        """
        if len(df) < 3:
            return {
                "bullish_pattern": False,
                "bearish_pattern": False,
                "pattern": "None",
            }

        # Use safe access instead of df.iloc[-1]
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Bullish engulfing
        if (
            prev["close"] < prev["open"]  # Prev bearish
            and latest["close"] > latest["open"]  # Current bullish
            and latest["close"] > prev["open"]
            and latest["open"] < prev["close"]
        ):
            return {
                "bullish_pattern": True,
                "bearish_pattern": False,
                "pattern": "Bullish Engulfing",
            }

        # Hammer (at support)
        body = abs(latest["close"] - latest["open"])
        lower_shadow = (
            latest["open"] - latest["low"]
            if latest["close"] > latest["open"]
            else latest["close"] - latest["low"]
        )

        if lower_shadow > body * 2:
            return {
                "bullish_pattern": True,
                "bearish_pattern": False,
                "pattern": "Hammer",
            }

        # Bearish patterns
        if (
            prev["close"] > prev["open"]
            and latest["close"] < latest["open"]
            and latest["close"] < prev["open"]
            and latest["open"] > prev["close"]
        ):
            return {
                "bullish_pattern": False,
                "bearish_pattern": True,
                "pattern": "Bearish Engulfing",
            }

        return {"bullish_pattern": False, "bearish_pattern": False, "pattern": "None"}

    def _check_sector_strength(self, df: pd.DataFrame, market_regime: Optional[Dict]) -> Dict:
        """
        Kiểm tra sức mạnh của ngành so với thị trường chung (VNINDEX).
        Sử dụng RS (Relative Strength)
        """
        if "rs" not in df.columns or df["rs"].isnull().all():
            return {"is_leading": False, "is_lagging": False, "sector_per": 0}

        # RS > 1: Cổ phiếu/ngành mạnh hơn thị trường
        # RS dốc lên: Sức mạnh đang tăng
        latest_rs = safe_get_latest(df, "rs", 0)
        rs_trend = safe_rolling_operation(df, "rs", 10, "mean", 0) > safe_rolling_operation(
            df, "rs", 30, "mean", 0
        )

        is_leading = latest_rs > 1.0 and rs_trend
        is_lagging = latest_rs < 0.95

        # Lấy performance từ market_regime nếu có
        sector_perf = 0
        if market_regime and "sector_performance" in market_regime:
            # Giả sử df có cột 'sector'
            sector = safe_get_latest(df, "sector", 0) if "sector" in df.columns else "UNKNOWN"
            sector_perf = market_regime["sector_performance"].get(sector, 0)

        return {
            "is_leading": is_leading,
            "is_lagging": is_lagging,
            "sector_perf": sector_perf,
        }

    def _check_portfolio_correlation(self, df: pd.DataFrame, symbol: Optional[str]) -> Dict:
        """
        OPTIMIZED: Kiểm tra correlation với portfolio hiện tại (with caching)

        Cache strategy:
        - Cache correlation matrix for 5 minutes
        - Invalidate when portfolio symbols change
        - Reduces redundant calculations during parallel scanning

        Returns:
            Dict with correlation analysis
        """
        if not symbol or not self.portfolio_manager:
            return {
                "too_high": False,
                "good_diversification": False,
                "max_correlation": 0.0,
            }

        try:
            from src.risk.metrics import calculate_portfolio_correlation_risk
            import time

            # Lấy danh sách positions hiện tại
            positions = self.portfolio_manager.get_positions()
            if not positions or len(positions) == 0:
                return {
                    "too_high": False,
                    "good_diversification": True,  # Portfolio rỗng = diversification tốt
                    "max_correlation": 0.0,
                }

            existing_symbols = list(positions.keys())
            all_symbols = existing_symbols + [symbol]
            symbols_key = tuple(sorted(existing_symbols))  # Create hashable key

            # OPTIMIZATION: Check cache validity
            # CRITICAL FIX: Invalidate cache on date change to prevent stale data
            import time
            from datetime import datetime
            import hashlib

            current_time = time.time()

            # IMPROVEMENT: Calculate portfolio hash to detect composition changes
            portfolio_hash = hashlib.md5(str(sorted(existing_symbols)).encode()).hexdigest()

            # Check if cache is from the same date
            cache_date_valid = True
            if self._correlation_cache_time is not None:
                cache_date = datetime.fromtimestamp(self._correlation_cache_time).date()
                current_date = datetime.now().date()
                cache_date_valid = cache_date == current_date

                if not cache_date_valid:
                    logger.debug(
                        f"🗓️ Correlation cache invalidated: date changed "
                        f"from {cache_date} to {current_date}"
                    )

            # IMPROVEMENT: Check if portfolio composition changed
            portfolio_changed = (
                self._correlation_cache_portfolio_hash is not None
                and self._correlation_cache_portfolio_hash != portfolio_hash
            )

            if portfolio_changed:
                logger.debug(
                    "📊 Correlation cache invalidated: portfolio composition changed"
                )

            cache_valid = (
                self._correlation_cache is not None
                and self._correlation_cache_time is not None
                and self._correlation_cache_symbols == symbols_key
                and (current_time - self._correlation_cache_time) < self._correlation_cache_ttl
                and cache_date_valid  # Invalidate on date change
                and not portfolio_changed  # NEW: Invalidate on portfolio change
            )

            if cache_valid:
                # Use cached correlation matrix
                correlation_metrics = self._correlation_cache
                logger.debug(
                    f"✅ Using cached correlation matrix (age: {current_time - self._correlation_cache_time:.0f}s)"
                )
            else:
                # Calculate fresh correlation matrix
                correlation_metrics = calculate_portfolio_correlation_risk(
                    all_symbols,
                    lookback=60,
                    max_avg_correlation=0.70,
                )

                # Update cache
                self._correlation_cache = correlation_metrics
                self._correlation_cache_time = current_time
                self._correlation_cache_symbols = symbols_key
                self._correlation_cache_portfolio_hash = portfolio_hash  # NEW: Store portfolio hash
                logger.debug("🔄 Calculated and cached new correlation matrix")

            max_correlation = correlation_metrics.get("max_correlation", 0.0)
            avg_correlation = correlation_metrics.get("avg_correlation", 0.0)

            # Threshold: > 0.7 = quá cao, < 0.3 = diversification tốt
            too_high = max_correlation > 0.70
            good_diversification = max_correlation < 0.30 and avg_correlation < 0.25

            return {
                "too_high": too_high,
                "good_diversification": good_diversification,
                "max_correlation": max_correlation,
                "avg_correlation": avg_correlation,
            }
        except Exception as e:
            logger.warning(f"⚠️ Error checking portfolio correlation: {e}")
            return {
                "too_high": False,
                "good_diversification": False,
                "max_correlation": 0.0,
            }

    def _check_market_breadth(self, market_regime: Optional[Dict]) -> Dict:
        """
        NEW: Kiểm tra breadth của thị trường (số mã tăng/giảm)
        """
        if not market_regime:
            return {"strong": False, "weak": False}

        details = market_regime.get("details", {})
        breadth = market_regime.get("breadth") or details.get("breadth") or {}

        advancers = breadth.get("advancers") or breadth.get("advancing") or 0
        decliners = breadth.get("decliners") or breadth.get("declining") or 0
        unchanged = breadth.get("unchanged", 0)

        total = advancers + decliners
        if total == 0:
            return {"strong": False, "weak": False}

        advance_ratio = advancers / total

        strong = advance_ratio >= 0.6
        weak = advance_ratio <= 0.4

        return {
            "strong": strong,
            "weak": weak,
            "advance_ratio": advance_ratio,
            "advancers": advancers,
            "decliners": decliners,
            "unchanged": unchanged,
        }

    # ========================================================================
    # SCORING & DECISION
    # ========================================================================

    def _calculate_signal_strength(
        self, confidence: int, risk_reward: float, warnings: list
    ) -> SignalStrength:
        """Tính độ mạnh của signal"""

        # Base score
        score = confidence / 20  # 0-5

        # Bonus for high R:R
        if risk_reward >= 3:
            score += 1
        elif risk_reward >= 2.5:
            score += 0.5

        # Penalty for warnings
        score -= len(warnings) * 0.5

        # Classify
        if score >= 5:
            return SignalStrength.VERY_STRONG
        elif score >= 4:
            return SignalStrength.STRONG
        elif score >= 3:
            return SignalStrength.MODERATE
        elif score >= 2:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK

    def _apply_performance_feedback(self, confidence: float) -> Tuple[float, Optional[str]]:
        """
        NEW: Điều chỉnh confidence dựa trên historical performance
        """
        if not self.performance_monitor:
            return confidence, None

        try:
            metrics = self.performance_monitor.get_metrics()
        except Exception:
            return confidence, None

        total_trades = metrics.get("total_trades", 0)
        if total_trades < 20:
            return confidence, None

        win_rate = metrics.get("win_rate", 0)
        adjustment = 0
        if win_rate >= 60:
            adjustment = +5
        elif win_rate <= 45:
            adjustment = -5

        new_confidence = max(0, min(100, confidence + adjustment))

        if adjustment > 0:
            return new_confidence, f"📈 Hiệu suất tốt (Win rate {win_rate:.1f}%)"
        elif adjustment < 0:
            return new_confidence, f"⚠️ Hiệu suất giảm (Win rate {win_rate:.1f}%)"
        return confidence, None

    def _calculate_position_multiplier(
        self,
        strength: SignalStrength,
        confidence: int,
        warnings: list,
        market_regime: Optional[Dict],
    ) -> float:
        """
        Tính multiplier cho position size

        Returns:
            0.3 - 1.5
        """
        # Base multiplier by strength
        base_multipliers = {
            SignalStrength.VERY_STRONG: 1.3,
            SignalStrength.STRONG: 1.1,
            SignalStrength.MODERATE: 1.0,
            SignalStrength.WEAK: 0.7,
            SignalStrength.VERY_WEAK: 0.5,
        }

        multiplier = base_multipliers.get(strength, 1.0)

        # Adjust by market regime
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            if regime == "BULL":
                multiplier *= 1.1
            elif regime == "SIDEWAYS":
                multiplier *= 0.9

        # Penalize warnings
        multiplier -= len(warnings) * 0.1

        # Clamp
        return max(0.3, min(multiplier, 1.5))

    def _adjust_thresholds_for_market(self, market_regime: Optional[Dict]):
        """
        ENHANCEMENT: Dynamically adjust confidence thresholds based on market conditions

        Logic:
        - BULL market: Lower threshold (more opportunities)
        - BEAR/HIGH_VOLATILITY: Higher threshold (more selective)
        - Consider portfolio heat
        """
        if not market_regime:
            self.min_confidence = self.base_min_confidence
            return

        regime = market_regime.get("regime", "SIDEWAYS")
        regime_confidence = market_regime.get("confidence", 50)

        # Base adjustment by regime type
        if regime == "BULL" and regime_confidence >= 70:
            # Strong bull market - can be less strict
            adjustment = -5
        elif regime == "BEAR":
            # Bear market - be more selective
            adjustment = +10
        elif regime == "HIGH_VOLATILITY":
            # High volatility - require higher confidence
            adjustment = +15
        else:
            # SIDEWAYS or unknown
            adjustment = 0

        # Portfolio heat adjustment
        if self.portfolio_manager:
            try:
                positions = self.portfolio_manager.get_positions()
                num_positions = len(positions)

                # If portfolio is getting crowded, be more selective
                if num_positions >= 8:
                    adjustment += 10
                    logger.info(
                        f"🔥 Portfolio heat: {num_positions} positions. Raising confidence threshold by +10"
                    )
                elif num_positions >= 5:
                    adjustment += 5
                    logger.info(
                        f"🔥 Portfolio heat: {num_positions} positions. Raising confidence threshold by +5"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Could not check portfolio heat: {e}")

        # Apply adjustment (with limits)
        # Allow lower bound 40 in favorable regimes to increase opportunities
        self.min_confidence = max(40, min(80, self.base_min_confidence + adjustment))

        if adjustment != 0:
            logger.info(
                f"📊 Dynamic threshold adjustment: {self.base_min_confidence} → {self.min_confidence} "
                f"(regime: {regime}, adj: {adjustment:+d})"
            )

    def _optimize_entry_price(
        self,
        df: pd.DataFrame,
        current_price: float,
        sr_check: Dict,
        market_regime: Optional[Dict],
    ) -> Dict:
        """
        ENHANCEMENT: Optimize entry price based on market conditions

        Strategies:
        1. PULLBACK: If price is in uptrend but pulled back, wait for entry at support/EMA
        2. BREAKOUT: If price is breaking resistance, use breakout entry
        3. MARKET: Use current price (default for most cases)

        Returns:
            Dict with entry_price, entry_type, and optimization_reason
        """
        if len(df) < 20:
            return {
                "entry_price": current_price,
                "entry_type": "MARKET",
                "optimization_reason": None,
            }

        # Get ATR for pullback calculation
        atr = safe_get_latest(df, "atr", 0)
        if atr == 0:
            # Fallback: estimate ATR as 2% of price
            atr = current_price * 0.02

        # Get EMAs for pullback entry
        ema20 = df["close"].ewm(span=20).mean().iloc[-1]
        ema50 = df["close"].ewm(span=50).mean().iloc[-1] if len(df) >= 50 else ema20

        # Check if price is bouncing from support (use market order - good entry)
        if sr_check.get("bouncing_from_support", False):
            # Price already bounced from support - good entry point
            return {
                "entry_price": current_price,
                "entry_type": "MARKET",
                "optimization_reason": "Bouncing from support",
            }

        # Strategy 1: PULLBACK ENTRY
        # If price is in uptrend but has pulled back, suggest entry near support/EMA
        # ENHANCEMENT: Only suggest pullback entry if price is near the pullback level
        if len(df) >= 50:
            # Check if we're in uptrend
            price_above_ema20 = current_price > ema20
            ema20_above_ema50 = ema20 > ema50

            if price_above_ema20 and ema20_above_ema50:
                # In uptrend - check if pulled back
                recent_high = safe_rolling_operation(df, "high", 5, "max", 0)
                pullback_pct = ((recent_high - current_price) / recent_high) * 100

                support_level = sr_check.get("support_level", 0)

                # If price is already near EMA20/support (within 1%), use EMA20 as entry
                if support_level > 0:
                    distance_to_ema20 = abs(current_price - ema20) / current_price * 100
                    distance_to_support = abs(current_price - support_level) / current_price * 100

                    # If price is within 1% of EMA20 or support, use the better entry
                    if distance_to_ema20 < 1.0 or distance_to_support < 1.0:
                        # Price is near pullback level - use the closer one
                        if distance_to_ema20 < distance_to_support:
                            pullback_entry = max(
                                ema20, current_price * 0.995
                            )  # Use EMA20 or slightly below current
                        else:
                            pullback_entry = max(
                                support_level * 1.01, current_price * 0.995
                            )  # Use support+1% or slightly below current

                        # Only use if it's better than current (at least 0.5% improvement)
                        if pullback_entry < current_price * 0.995 and 1 <= pullback_pct <= 5:
                            return {
                                "entry_price": pullback_entry,
                                "entry_type": "PULLBACK",
                                "optimization_reason": f"Near EMA20/support - entry at pullback level (~{pullback_pct:.1f}% from high)",
                            }

        # Strategy 2: BREAKOUT ENTRY
        # If price is breaking resistance with volume, use breakout entry
        resistance_level = sr_check.get("resistance_level", 0)
        if resistance_level > 0:
            distance_to_resistance = sr_check.get("distance_to_resistance", 100)

            # Breaking resistance (within 1% above resistance)
            if -1 <= distance_to_resistance <= 1:
                # Check volume confirmation
                current_volume = safe_get_latest(df, "volume", 0)
                avg_volume_20 = safe_rolling_operation(df, "volume", 20, "mean", 0)

                if avg_volume_20 > 0 and current_volume > avg_volume_20 * 1.2:
                    # Breakout with volume - use current price (breakout confirmed)
                    return {
                        "entry_price": current_price,
                        "entry_type": "BREAKOUT",
                        "optimization_reason": f"Breaking resistance with volume ({current_volume/avg_volume_20:.1f}x avg)",
                    }
                elif current_price > resistance_level * 1.01:
                    # Price broke resistance - use slight pullback entry if possible
                    breakout_entry = max(current_price * 0.995, resistance_level * 1.005)
                    return {
                        "entry_price": breakout_entry,
                        "entry_type": "BREAKOUT",
                        "optimization_reason": "Breakout - entry on slight pullback",
                    }

        # Strategy 3: RSI OVERSOLD ENTRY
        # If RSI is oversold and bouncing, use market order (already good entry)
        if "rsi" in df.columns:
            rsi = safe_get_latest(df, "rsi", 50)
            if not pd.isna(rsi) and rsi < 30:
                # RSI oversold - current price is good entry
                return {
                    "entry_price": current_price,
                    "entry_type": "MARKET",
                    "optimization_reason": f"RSI oversold ({rsi:.1f}) - good entry",
                }

        # Default: Use market order at current price
        return {
            "entry_price": current_price,
            "entry_type": "MARKET",
            "optimization_reason": None,
        }

    def _calculate_technical_confidence(self, df: pd.DataFrame) -> float:
        """
        Calculate confidence from technical indicators when ML signal is unavailable

        Uses multiple technical factors:
        - RSI position
        - MACD signal
        - Moving average alignment
        - Price action strength
        """
        if len(df) < 50:
            return 0.0

        from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

        confidence = 50.0  # Base confidence

        # RSI check
        if "rsi" in df.columns:
            rsi = safe_get_latest(df, "rsi", 50)
            if not pd.isna(rsi):
                if 30 <= rsi <= 60:  # Good range for entry
                    confidence += 10
                elif 60 < rsi <= 70:
                    confidence += 5

        # Moving average alignment
        if len(df) >= 50:
            ema20 = df["close"].ewm(span=20).mean()
            ema50 = df["close"].ewm(span=50).mean()
            current_price = safe_get_latest(df, "close", 0)
            latest_ema20 = ema20.iloc[-1]
            latest_ema50 = ema50.iloc[-1]

            if current_price > latest_ema20:
                confidence += 10
            if latest_ema20 > latest_ema50:
                confidence += 10

        # Volume confirmation
        if len(df) >= 20:
            current_volume = safe_get_latest(df, "volume", 0)
            avg_volume = safe_rolling_operation(df, "volume", 20, "mean", 0)
            if avg_volume > 0 and current_volume > avg_volume * 1.2:
                confidence += 10

        return min(confidence, 100.0)

    def _get_technical_signal(self, df: pd.DataFrame) -> str:
        """
        Determine signal type from technical analysis
        Returns: "BUY", "SELL", or "HOLD"
        """
        if len(df) < 50:
            return "HOLD"

        from utils.dataframe_utils import safe_get_latest

        current_price = safe_get_latest(df, "close", 0)
        prev_price = df["close"].iloc[-2] if len(df) >= 2 else current_price

        # Simple trend following
        if current_price > prev_price:
            return "BUY"
        elif current_price < prev_price:
            return "SELL"
        else:
            return "HOLD"

    def _check_vietnam_market_liquidity(self, df: pd.DataFrame) -> Dict:
        """
        NEW: Check Vietnam market-specific liquidity requirements

        Uses VietnamMarketValidator to check:
        - Minimum daily trading value (2B VND)
        - Trading continuity

        Returns:
            Dict with sufficient flag and reason
        """
        try:
            from src.utils.vietnam_market import check_liquidity

            is_liquid, warning = check_liquidity(df, self._current_symbol)

            if not is_liquid:
                return {
                    "sufficient": False,
                    "reason": warning or "Vietnam market liquidity requirement not met",
                }

            return {"sufficient": True, "reason": "Vietnam market liquidity OK"}

        except Exception as e:
            logger.warning(f"Error checking Vietnam market liquidity: {e}")
            # Don't block on validation errors - return as sufficient with warning
            return {
                "sufficient": True,
                "reason": f"Liquidity check error: {str(e)}",
            }

    def _no_signal(self, reason: str, telemetry: Optional[Dict] = None) -> EntrySignal:
        """Return no signal with detailed reason"""
        # Build detailed warning message
        warnings = [reason]  # Start with the main reason
        warning_msg = reason  # Initialize with base reason

        # Add telemetry details if available
        if telemetry:
            details = []
            if "adjustment_breakdown" in telemetry:
                for adj in telemetry["adjustment_breakdown"]:
                    if adj.get("delta", 0) < 0:  # Only show negative adjustments (rejections)
                        detail = f"{adj.get('filter', 'Unknown')}: {adj.get('note', 'No reason')}"
                        details.append(detail)
                        warnings.append(detail)  # Add each detail as a separate warning

            # Add base confidence if available
            if "base_confidence" in telemetry:
                details.append(f"Base confidence: {telemetry['base_confidence']:.1f}%")

            # Update warning_msg with details if available
            if details:
                warning_msg = f"{reason} ({'; '.join(details)})"

        # Log the detailed reason at DEBUG level (not WARNING)
        if self._current_symbol:
            logger.debug(f"[No Signal] {self._current_symbol}: {warning_msg}")
        else:
            logger.debug(f"[No Signal] {warning_msg}")

        return EntrySignal(
            should_enter=False,
            signal_type="HOLD",
            confidence=0,
            strength=SignalStrength.NO_SIGNAL,
            position_size_multiplier=0.0,
            reasons=[],
            warnings=warnings,
            entry_price=0,
            stop_loss=0,
            take_profit_targets=[],
            is_limit_order=False,
            limit_price=None,
            entry_type="MARKET",
            telemetry=telemetry,
        )

    def format_signal_message(self, signal: EntrySignal, symbol: str) -> str:
        """Format signal thành message đẹp"""

        if not signal.should_enter:
            return f"⏭️ **{symbol}** - Không vào lệnh\n" f"Lý do: {', '.join(signal.warnings)}"

        msg = f"🎯 **{symbol}** - {signal.signal_type}\n"
        msg += f"💪 Strength: {signal.strength.name}\n"
        msg += f"🎲 Confidence: {signal.confidence}%\n"
        msg += f"📊 Position Size: {signal.position_size_multiplier:.1f}x\n\n"

        msg += f"💰 **Entry:** {signal.entry_price:,.0f} VNĐ\n"
        msg += f"🛑 **Stop Loss:** {signal.stop_loss:,.0f} VNĐ "
        msg += f"({((signal.stop_loss - signal.entry_price)/signal.entry_price * 100):+.1f}%)\n\n"

        msg += "🎯 **Take Profit:**\n"
        for i, tp in enumerate(signal.take_profit_targets, 1):
            tp_pct = ((tp - signal.entry_price) / signal.entry_price) * 100
            msg += f"  TP{i}: {tp:,.0f} VNĐ (+{tp_pct:.1f}%)\n"

        if signal.reasons:
            msg += "\n✅ **Reasons:**\n"
            for reason in signal.reasons:
                msg += f"  • {reason}\n"

        if signal.warnings:
            msg += "\n⚠️ **Warnings:**\n"
            for warning in signal.warnings:
                msg += f"  • {warning}\n"

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
    print("🧪 TESTING IMPROVED ENTRY LOGIC")
    print("=" * 70 + "\n")

    # Test với 1 mã
    symbol = "VNM"
    df = load_data(symbol, 200)
    df = add_ml_features(df)

    # Get ML signal
    ml_gen = MLSignalGenerator()
    ml_signal = ml_gen.analyze(df)

    print("📊 ML Signal: {ml_signal['signal']} ({ml_signal['confidence']}%)")

    # Analyze entry
    entry_logic = ImprovedEntryLogic(
        min_confidence=60,
        min_risk_reward=2.0,
        require_trend_alignment=True,
        require_volume_confirmation=False,  # Relax for testing
    )

    signal = entry_logic.analyze_entry(df, ml_signal, symbol=symbol)

    # Print result
    print("\n" + "=" * 70)
    message = entry_logic.format_signal_message(signal, symbol)
    print(message)
    print("=" * 70)
