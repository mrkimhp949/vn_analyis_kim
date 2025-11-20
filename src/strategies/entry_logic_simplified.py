"""
SIMPLIFIED Entry Logic - From 14 filters down to 8 core filters
Uses centralized config and fundamental data API
Better maintainability, less overfit
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Import configurations
from src.config.entry_config import get_entry_config, EntryLogicConfig

# Import utilities
from src.monitoring.performance import get_performance_monitor
from src.utils.indicators import IndicatorUtils, StopLossCalculator
from src.utils.validation import DataValidator
from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

# Import fundamental data
from src.data.fundamental_data import get_fundamental_manager

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength levels"""

    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NO_SIGNAL = 0


@dataclass
class EntrySignal:
    """Container for entry signal"""

    should_enter: bool
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: int  # 0-100
    strength: SignalStrength
    position_size_multiplier: float  # 0.3 - 1.5
    reasons: List[str]
    warnings: List[str]
    entry_price: float
    stop_loss: float
    take_profit_targets: List[float]
    # Entry optimization
    is_limit_order: bool = False
    limit_price: Optional[float] = None
    entry_type: str = "MARKET"  # 'MARKET', 'LIMIT', 'PULLBACK', 'BREAKOUT'
    telemetry: Optional[Dict] = None


class SimplifiedEntryLogic:
    """
    SIMPLIFIED Entry Logic with 8 CORE FILTERS:

    MANDATORY FILTERS (must pass or reject):
    1. Market Regime - Must be tradeable
    2. Liquidity - Tiered thresholds
    3. Risk/Reward - Must meet minimum ratio

    HIGH PRIORITY FILTERS (major confidence impact):
    4. Trend Alignment - EMA alignment
    5. Support/Resistance - Entry optimization
    6. Volume Confirmation - Multi-indicator
    7. RSI - Momentum check

    MEDIUM PRIORITY FILTERS (moderate impact):
    8. Portfolio Correlation - Diversification

    REMOVED FILTERS (merged or low-impact):
    - Volatility → Merged into Risk/Reward calculation
    - Price Action → Low signal, removed
    - Multi-Timeframe → Redundant with Trend
    - Market Breadth → Noisy, removed
    - Sector Strength → Optional, low impact
    - Earnings/Events → Now via Fundamental API
    - Fundamentals → Now via Fundamental API
    """

    def __init__(
        self,
        config: Optional[EntryLogicConfig] = None,
        portfolio_manager=None,
        performance_monitor=None,
    ):
        """
        Args:
            config: Entry logic configuration (uses default if None)
            portfolio_manager: Portfolio manager for context
            performance_monitor: Performance monitor for feedback
        """
        self.config = config or get_entry_config()
        self.portfolio_manager = portfolio_manager
        self.performance_monitor = performance_monitor or get_performance_monitor()
        self.fundamental_manager = get_fundamental_manager()

        self._current_symbol = None

        logger.info(
            f"✅ Simplified Entry Logic initialized "
            f"(min_confidence={self.config.min_confidence}%, "
            f"simplified={self.config.use_simplified_filters})"
        )

    def analyze_entry(
        self,
        df: pd.DataFrame,
        ml_signal: Optional[Dict],
        market_regime: Optional[Dict] = None,
        symbol: Optional[str] = None,
    ) -> EntrySignal:
        """
        Analyze entry signal with simplified filter logic

        Args:
            df: OHLCV dataframe with indicators
            ml_signal: ML model signal
            market_regime: Market regime info
            symbol: Stock symbol

        Returns:
            EntrySignal with full analysis
        """
        self._current_symbol = symbol

        try:
            # STEP 1: Dynamic threshold adjustment
            self._adjust_thresholds_for_market(market_regime)

            # STEP 2: Validate initial signal
            is_valid, signal_or_reason, base_confidence, current_price = (
                self._validate_initial_signal(df, ml_signal)
            )
            if not is_valid:
                return self._no_signal(signal_or_reason)

            signal_type = signal_or_reason

            # STEP 3: Run SIMPLIFIED filters (8 core filters)
            passed, reasons, warnings, adjustments, breakdown = self._run_simplified_filters(
                df, signal_type, current_price, market_regime
            )

            if not passed:
                regime_name = market_regime.get("regime", "UNKNOWN") if market_regime else "N/A"
                return self._no_signal(
                    f"Market: {regime_name}",
                    telemetry={
                        "base_confidence": base_confidence,
                        "adjustments": breakdown,
                        "reason": "Filters rejected",
                    },
                )

            # STEP 4: Calculate adjusted confidence
            confidence_after_filters = base_confidence + sum(adjustments)
            confidence_after_filters = max(0, min(confidence_after_filters, 100))

            # Apply performance feedback
            adjusted_confidence, perf_msg = self._apply_performance_feedback(
                confidence_after_filters
            )
            if perf_msg:
                (warnings if perf_msg.startswith("⚠️") else reasons).append(perf_msg)

            telemetry = {
                "base_confidence": base_confidence,
                "adjustments": breakdown,
                "confidence_after_filters": confidence_after_filters,
                "confidence_after_performance": adjusted_confidence,
                "min_confidence_threshold": self.config.min_confidence,
                "filters_used": "SIMPLIFIED (8 core)",
            }

            if adjusted_confidence < self.config.min_confidence:
                return self._no_signal(
                    f"Confidence {adjusted_confidence}% < {self.config.min_confidence}%",
                    telemetry=telemetry,
                )

            # STEP 5: Calculate prices and risk/reward
            close_price = safe_get_latest(df, "close", 0)
            sr_check = self._check_support_resistance(df, current_price)

            # Entry optimization
            optimized_entry = self._optimize_entry_price(df, close_price, sr_check, market_regime)
            entry_price = DataValidator.validate_price(
                optimized_entry["entry_price"], "entry_price"
            )

            # Limit order logic
            is_limit_order = optimized_entry.get("entry_type") in ["PULLBACK", "BREAKOUT"]
            limit_price = optimized_entry.get("entry_price") if is_limit_order else None

            price_diff_pct = (
                abs(entry_price - close_price) / close_price * 100 if close_price > 0 else 0
            )
            if (
                is_limit_order
                and price_diff_pct < self.config.entry_optimization.limit_order_min_diff
            ):
                is_limit_order = False
                limit_price = None
                entry_price = close_price
                optimized_entry["entry_type"] = "MARKET"

            if optimized_entry.get("entry_type") != "MARKET":
                entry_reason = f"Entry: {optimized_entry['entry_type']}"
                if optimized_entry.get("optimization_reason"):
                    entry_reason += f" ({optimized_entry['optimization_reason']})"
                if is_limit_order:
                    entry_reason += f" [LIMIT @ {limit_price:,.0f}]"
                reasons.append(f"✅ {entry_reason}")

            # Calculate risk/reward
            success, error_msg, stop_loss, reward, tp_targets, risk_reward = (
                self._calculate_prices_and_risk(df, entry_price, sr_check)
            )
            if not success:
                return self._no_signal(error_msg)

            reasons.append(f"✅ R:R ratio: {risk_reward:.2f}")
            telemetry["risk_reward"] = risk_reward

            # STEP 6: Determine signal strength and position multiplier
            strength = self._calculate_signal_strength(adjusted_confidence, risk_reward, warnings)
            position_multiplier = self._calculate_position_multiplier(
                strength, adjusted_confidence, warnings, market_regime
            )

            # STEP 7: Build entry signal
            return EntrySignal(
                should_enter=True,
                signal_type="BUY",
                confidence=int(adjusted_confidence),
                strength=strength,
                position_size_multiplier=position_multiplier,
                reasons=reasons,
                warnings=warnings,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_targets=tp_targets,
                is_limit_order=is_limit_order,
                limit_price=limit_price,
                entry_type=optimized_entry.get("entry_type", "MARKET"),
                telemetry=telemetry,
            )

        finally:
            self._current_symbol = None

    # ========================================================================
    # SIMPLIFIED FILTERS (8 CORE FILTERS)
    # ========================================================================

    def _run_simplified_filters(
        self,
        df: pd.DataFrame,
        signal_type: str,
        current_price: float,
        market_regime: Optional[Dict],
    ) -> Tuple[bool, List[str], List[str], List[int], List[Dict]]:
        """
        Run SIMPLIFIED 8 core filters instead of 14

        Returns:
            (passed, reasons, warnings, adjustments, breakdown)
        """
        reasons = []
        warnings = []
        adjustments = []
        breakdown = []

        # Get adjustment scale from market regime
        adjustment_scale = self._get_adjustment_scale(market_regime)

        # ============================================================
        # MANDATORY FILTER 1: MARKET REGIME
        # ============================================================
        if market_regime and not market_regime.get("tradeable", True):
            breakdown.append({"filter": "market_regime", "delta": None, "note": "Not tradeable"})
            return False, [], [], [], breakdown

        # ============================================================
        # MANDATORY FILTER 2: LIQUIDITY (Tiered)
        # ============================================================
        liquidity_check = self._check_liquidity(df, current_price)
        if liquidity_check["critical"]:
            breakdown.append({"filter": "liquidity", "delta": None, "note": "Critical liquidity"})
            return False, [], [], [], breakdown

        if not liquidity_check["sufficient"]:
            tier = liquidity_check.get("tier", "unknown")
            warning_msg = (
                f"⚠️ Low liquidity ({tier} cap) " f"({liquidity_check['avg_value']/1e9:.2f}B VND)"
            )
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                breakdown,
                "liquidity",
                self.config.filters.liquidity_low_penalty,
                warning_msg,
            )
        else:
            tier = liquidity_check.get("tier", "unknown")
            reasons.append(f"✅ Good liquidity ({tier} cap)")
            self._add_adjustment(
                adjustments,
                breakdown,
                "liquidity",
                self.config.filters.liquidity_good_bonus,
                "Good liquidity",
            )

        # ============================================================
        # HIGH PRIORITY FILTER 3: TREND ALIGNMENT
        # ============================================================
        trend_check = self._check_trend_alignment(df, signal_type)
        if not trend_check["aligned"]:
            if self.config.require_trend_alignment:
                breakdown.append({"filter": "trend", "delta": None, "note": trend_check["reason"]})
                return False, [], [], [], breakdown
            else:
                warnings.append(f"⚠️ Trend: {trend_check['reason']}")
                self._add_adjustment(
                    adjustments,
                    breakdown,
                    "trend",
                    self.config.filters.trend_weak_penalty,
                    trend_check["reason"],
                )
        else:
            reasons.append(f"✅ Trend: {trend_check['reason']}")
            if trend_check["strength"] > 50:
                self._add_adjustment(
                    adjustments,
                    breakdown,
                    "trend",
                    self.config.filters.trend_perfect_bonus,
                    "Strong alignment",
                )

        # ============================================================
        # HIGH PRIORITY FILTER 4: SUPPORT/RESISTANCE
        # ============================================================
        sr_check = self._check_support_resistance(df, current_price)
        if sr_check["too_close_to_resistance"]:
            warning_msg = f"⚠️ Near resistance: {sr_check['distance_to_resistance']:.1f}%"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                breakdown,
                "support_resistance",
                self.config.filters.resistance_close_penalty,
                warning_msg,
            )
        elif sr_check["bouncing_from_support"]:
            reasons.append(
                f"✅ Bouncing from support (+{sr_check['distance_to_support']:.1f}%) - REVERSAL"
            )
            self._add_adjustment(
                adjustments,
                breakdown,
                "support_resistance",
                self.config.filters.support_bounce_bonus,
                "Support bounce",
            )
        elif sr_check["near_support"]:
            reasons.append(f"✅ Near support (+{sr_check['distance_to_support']:.1f}%)")
            self._add_adjustment(
                adjustments,
                breakdown,
                "support_resistance",
                self.config.filters.support_near_bonus,
                "Near support",
            )

        # ============================================================
        # HIGH PRIORITY FILTER 5: VOLUME CONFIRMATION
        # ============================================================
        volume_check = self._check_volume_confirmation(df, market_regime)
        if not volume_check["confirmed"]:
            if self.config.require_volume_confirmation:
                breakdown.append(
                    {"filter": "volume", "delta": None, "note": volume_check["reason"]}
                )
                return False, [], [], [], breakdown
            else:
                warnings.append(f"⚠️ Volume: {volume_check['reason']}")
                self._add_adjustment(
                    adjustments,
                    breakdown,
                    "volume",
                    self.config.filters.volume_low_penalty,
                    volume_check["reason"],
                )
        else:
            reasons.append(f"✅ Volume: {volume_check['reason']}")
            if volume_check["surge"]:
                self._add_adjustment(
                    adjustments,
                    breakdown,
                    "volume",
                    self.config.filters.volume_surge_bonus,
                    "Volume surge",
                )

        # ============================================================
        # HIGH PRIORITY FILTER 6: RSI
        # ============================================================
        rsi_check = self._check_rsi(df)
        if rsi_check["overbought"]:
            warning_msg = f"⚠️ RSI overbought: {rsi_check['value']:.1f}"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                breakdown,
                "rsi",
                self.config.filters.rsi_overbought_penalty,
                warning_msg,
            )
        elif rsi_check["oversold"]:
            reasons.append(f"✅ RSI oversold: {rsi_check['value']:.1f} (strong buy)")
            self._add_adjustment(
                adjustments,
                breakdown,
                "rsi",
                self.config.filters.rsi_oversold_bonus,
                "Oversold RSI",
            )
        elif rsi_check["optimal"]:
            reasons.append(f"✅ RSI: {rsi_check['value']:.1f}")
            self._add_adjustment(
                adjustments,
                breakdown,
                "rsi",
                self.config.filters.rsi_optimal_bonus,
                "Optimal RSI",
            )

        # ============================================================
        # MEDIUM PRIORITY FILTER 7: PORTFOLIO CORRELATION
        # ============================================================
        correlation_check = self._check_portfolio_correlation(df, self._current_symbol)
        if correlation_check["too_high"]:
            warning_msg = f"⚠️ High correlation: {correlation_check['max_correlation']:.2f}"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                breakdown,
                "correlation",
                self.config.filters.correlation_high_penalty,
                warning_msg,
            )
        elif correlation_check["good_diversification"]:
            reasons.append(
                f"✅ Good diversification (corr: {correlation_check['max_correlation']:.2f})"
            )
            self._add_adjustment(
                adjustments,
                breakdown,
                "correlation",
                self.config.filters.correlation_good_bonus,
                "Good diversification",
            )

        # ============================================================
        # OPTIONAL FILTER 8: FUNDAMENTALS (via API)
        # ============================================================
        fundamental_check = self._check_fundamentals_via_api(self._current_symbol, current_price)
        if fundamental_check["poor_fundamentals"]:
            warning_msg = f"⚠️ Fundamentals: {fundamental_check['reason']}"
            warnings.append(warning_msg)
            self._add_adjustment(
                adjustments,
                breakdown,
                "fundamentals",
                self.config.filters.fundamentals_poor_penalty,
                warning_msg,
            )
        elif fundamental_check["good_fundamentals"]:
            reasons.append(f"✅ Fundamentals: {fundamental_check['reason']}")
            self._add_adjustment(
                adjustments,
                breakdown,
                "fundamentals",
                self.config.filters.fundamentals_good_bonus,
                "Good fundamentals",
            )

        # Apply scaling to penalties (lighter in BULL, heavier in BEAR)
        if adjustment_scale != 1.0:
            scaled_adjustments = []
            for idx, adj in enumerate(adjustments):
                if adj < 0:  # Only scale penalties
                    new_adj = int(adj * adjustment_scale)
                    scaled_adjustments.append(new_adj)
                    if idx < len(breakdown):
                        breakdown[idx]["delta"] = new_adj
                        breakdown[idx]["note"] += " (scaled)"
                else:
                    scaled_adjustments.append(adj)
            adjustments = scaled_adjustments

        return True, reasons, warnings, adjustments, breakdown

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _validate_initial_signal(
        self, df: pd.DataFrame, ml_signal: Optional[Dict]
    ) -> Tuple[bool, str, float, float]:
        """Validate initial signal (with ML fallback to technical)"""
        try:
            DataValidator.validate_dataframe(df, min_rows=50)
        except Exception as e:
            return False, f"Data validation failed: {str(e)}", 0, 0

        close_price = safe_get_latest(df, "close", 0)

        # Fallback to technical if ML signal is None
        if ml_signal is None:
            logger.debug("ML signal is None - using advanced technical fallback")
            from src.ml.signals.technical_fallback import analyze_technical

            technical_signal = analyze_technical(df)
            base_confidence = technical_signal.confidence

            if base_confidence < 40:
                return False, f"Technical confidence low ({base_confidence}%)", 0, 0

            if technical_signal.signal != "BUY":
                return False, f"Technical signal = {technical_signal.signal}", 0, 0

            return True, "BUY", base_confidence, close_price

        signal_type = ml_signal.get("signal", "HOLD")
        base_confidence = ml_signal.get("confidence", 0)

        if signal_type != "BUY":
            return False, f"Signal = {signal_type}", 0, 0

        if base_confidence < self.config.min_confidence:
            return False, f"Confidence low ({base_confidence}%)", 0, 0

        return True, signal_type, base_confidence, close_price

    def _add_adjustment(
        self,
        adjustments: List[int],
        breakdown: List[Dict],
        filter_name: str,
        delta: int,
        note: str,
    ):
        """Add adjustment and telemetry"""
        adjustments.append(delta)
        breakdown.append({"filter": filter_name, "delta": delta, "note": note})

    def _get_adjustment_scale(self, market_regime: Optional[Dict]) -> float:
        """Get adjustment scaling factor based on market regime"""
        if not market_regime:
            return 1.0

        regime = market_regime.get("regime", "SIDEWAYS")
        regime_confidence = market_regime.get("confidence", 50)

        if regime == "BULL" and regime_confidence >= self.config.regime.bull_min_regime_confidence:
            return self.config.regime.bull_penalty_scale
        elif regime == "BEAR":
            return self.config.regime.bear_penalty_scale
        elif regime == "HIGH_VOLATILITY":
            return self.config.regime.high_vol_penalty_scale
        else:
            return self.config.regime.sideways_penalty_scale

    # Filter implementations (simplified versions - reuse from original where possible)
    # For brevity, I'll import the filter methods from original entry_logic
    # In production, these would be refactored into shared utilities

    def _check_trend_alignment(self, df: pd.DataFrame, signal_type: str) -> Dict:
        """Check trend alignment using EMAs"""
        # Simplified implementation - see original for full logic
        # This is a placeholder - in production would import from utils
        if len(df) < 200:
            return {"aligned": True, "reason": "Insufficient data", "strength": 50}

        ema20 = df["close"].ewm(span=20).mean()
        ema50 = df["close"].ewm(span=50).mean()
        ema200 = df["close"].ewm(span=200).mean()

        latest_price = safe_get_latest(df, "close", 0)
        latest_ema20 = ema20.iloc[-1]
        latest_ema50 = ema50.iloc[-1]
        latest_ema200 = ema200.iloc[-1]

        if signal_type == "BUY":
            perfect = latest_price > latest_ema20 > latest_ema50 > latest_ema200
            good = latest_price > latest_ema20 > latest_ema50
            ok = latest_price > latest_ema20

            if perfect:
                return {"aligned": True, "reason": "Perfect uptrend", "strength": 100}
            elif good:
                return {"aligned": True, "reason": "Strong uptrend", "strength": 75}
            elif ok:
                return {"aligned": True, "reason": "Short-term uptrend", "strength": 50}
            else:
                return {"aligned": False, "reason": "Downtrend", "strength": 0}

        return {"aligned": True, "reason": "Unknown signal", "strength": 50}

    def _check_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Check support/resistance levels"""
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

        distance_to_support = ((current_price - support) / support) * 100 if support > 0 else 100
        distance_to_resistance = (
            ((resistance - current_price) / current_price) * 100 if current_price > 0 else 100
        )

        near_support = distance_to_support <= self.config.filters.support_distance_percent
        too_close = distance_to_resistance <= self.config.filters.resistance_proximity_percent

        # Check support bounce
        bouncing_from_support = False
        if near_support and len(df) >= 3:
            recent_low = safe_rolling_operation(df, "low", 3, "min", 0)
            if abs(recent_low - support) / support < self.config.filters.support_bounce_distance:
                prev_close = df["close"].iloc[-2] if len(df) >= 2 else current_price
                if current_price > prev_close:
                    bouncing_from_support = True

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
        """Check liquidity with tiered thresholds"""
        if "volume" not in df.columns or len(df) < 5:
            return {
                "sufficient": True,
                "critical": False,
                "avg_value": 0.0,
                "tier": "unknown",
            }

        current_volume = safe_get_latest(df, "volume", 0)
        avg_volume = df["volume"].tail(20).mean()
        avg_value = avg_volume * current_price

        # Determine tier
        if avg_value >= self.config.filters.liquidity_large_cap:
            tier = "large"
            threshold = self.config.filters.liquidity_large_cap
        elif avg_value >= self.config.filters.liquidity_mid_cap:
            tier = "mid"
            threshold = self.config.filters.liquidity_mid_cap
        elif avg_value >= self.config.filters.liquidity_small_cap:
            tier = "small"
            threshold = self.config.filters.liquidity_small_cap
        else:
            tier = "micro"
            threshold = self.config.filters.liquidity_small_cap

        sufficient = avg_value >= threshold
        critical = avg_value < (threshold * 0.5)

        return {
            "sufficient": sufficient,
            "critical": critical,
            "avg_value": avg_value,
            "tier": tier,
        }

    def _check_volume_confirmation(self, df: pd.DataFrame, market_regime: Optional[Dict]) -> Dict:
        """Volume confirmation with dynamic thresholds"""
        if len(df) < 20:
            return {"confirmed": True, "reason": "Insufficient data", "surge": False}

        current_volume = safe_get_latest(df, "volume", 0)
        avg_volume_20 = safe_rolling_operation(df, "volume", 20, "mean", 0)

        if avg_volume_20 == 0:
            return {"confirmed": True, "reason": "Invalid volume", "surge": False}

        # Dynamic threshold
        base_threshold = self.config.volume.sideways_threshold
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            regime_conf = market_regime.get("confidence", 50)
            if regime == "BULL" and regime_conf >= 70:
                base_threshold = self.config.volume.bull_threshold
            elif regime == "BEAR":
                base_threshold = self.config.volume.bear_threshold

        volume_ratio = current_volume / avg_volume_20
        avg_volume_5 = safe_rolling_operation(df, "volume", 5, "mean", 0)
        volume_trending_up = avg_volume_5 > avg_volume_20

        # Calculate confidence score
        confidence_score = 0.0
        if volume_ratio >= self.config.volume.volume_ratio_strong:
            confidence_score += 0.4
        elif volume_ratio >= self.config.volume.volume_ratio_good:
            confidence_score += 0.3
        elif volume_ratio >= self.config.volume.volume_ratio_neutral:
            confidence_score += 0.2

        if volume_trending_up:
            confidence_score += 0.3

        # OBV check (simplified)
        confidence_score += 0.3  # Placeholder for OBV

        confirmed = confidence_score >= base_threshold
        surge = volume_ratio >= self.config.filters.volume_surge_threshold

        reason = f"Vol {volume_ratio:.1f}x"
        if surge:
            reason += " (surge)"

        return {
            "confirmed": confirmed,
            "reason": reason,
            "surge": surge,
            "volume_ratio": volume_ratio,
        }

    def _check_rsi(self, df: pd.DataFrame) -> Dict:
        """Check RSI"""
        if "rsi" not in df.columns:
            return {"overbought": False, "optimal": True, "oversold": False, "value": 50}

        rsi = safe_get_latest(df, "rsi", 50)
        if pd.isna(rsi):
            return {"overbought": False, "optimal": True, "oversold": False, "value": 50}

        return {
            "overbought": rsi > self.config.filters.rsi_overbought,
            "optimal": self.config.filters.rsi_optimal_min
            <= rsi
            <= self.config.filters.rsi_optimal_max,
            "oversold": rsi < self.config.filters.rsi_oversold,
            "value": rsi,
        }

    def _check_portfolio_correlation(self, df: pd.DataFrame, symbol: Optional[str]) -> Dict:
        """Check portfolio correlation"""
        if not symbol or not self.portfolio_manager:
            return {
                "too_high": False,
                "good_diversification": False,
                "max_correlation": 0.0,
            }

        try:
            from src.risk.metrics import calculate_portfolio_correlation_risk

            positions = self.portfolio_manager.get_positions()
            if not positions or len(positions) == 0:
                return {
                    "too_high": False,
                    "good_diversification": True,
                    "max_correlation": 0.0,
                }

            existing_symbols = list(positions.keys())
            all_symbols = existing_symbols + [symbol]

            correlation_metrics = calculate_portfolio_correlation_risk(
                all_symbols,
                lookback=60,
                max_avg_correlation=self.config.filters.correlation_max_threshold,
            )

            max_corr = correlation_metrics.get("max_correlation", 0.0)
            avg_corr = correlation_metrics.get("avg_correlation", 0.0)

            too_high = max_corr > self.config.filters.correlation_max_threshold
            good_div = (
                max_corr < self.config.filters.correlation_diversification_threshold
                and avg_corr < self.config.filters.correlation_avg_threshold
            )

            return {
                "too_high": too_high,
                "good_diversification": good_div,
                "max_correlation": max_corr,
            }
        except Exception as e:
            logger.warning(f"⚠️ Correlation check error: {e}")
            return {"too_high": False, "good_diversification": False, "max_correlation": 0.0}

    def _check_fundamentals_via_api(self, symbol: Optional[str], current_price: float) -> Dict:
        """Check fundamentals via API"""
        if not symbol:
            return {
                "poor_fundamentals": False,
                "good_fundamentals": False,
                "reason": None,
            }

        try:
            # Get fundamental data from API
            fund_data = self.fundamental_manager.get_fundamental_data(symbol)

            if not fund_data or not fund_data.is_valid():
                return {
                    "poor_fundamentals": False,
                    "good_fundamentals": False,
                    "reason": "No fundamental data",
                }

            reasons = []
            poor = False

            # P/E Check
            if fund_data.pe_ratio is not None:
                if fund_data.pe_ratio > self.config.filters.pe_ratio_max:
                    poor = True
                    reasons.append(f"P/E high ({fund_data.pe_ratio:.1f})")
                elif (
                    self.config.filters.pe_ratio_optimal_min
                    <= fund_data.pe_ratio
                    <= self.config.filters.pe_ratio_optimal_max
                ):
                    reasons.append(f"P/E good ({fund_data.pe_ratio:.1f})")

            # Debt Ratio Check
            if fund_data.debt_ratio is not None:
                if fund_data.debt_ratio > self.config.filters.debt_ratio_max:
                    poor = True
                    reasons.append(f"Debt high ({fund_data.debt_ratio*100:.1f}%)")
                elif fund_data.debt_ratio < self.config.filters.debt_ratio_optimal:
                    reasons.append(f"Debt low ({fund_data.debt_ratio*100:.1f}%)")

            # Check earnings proximity
            earnings_info = self.fundamental_manager.get_earnings_date(symbol)
            if earnings_info and earnings_info.get("next_earnings_date"):
                from datetime import datetime, timedelta

                next_earnings = earnings_info["next_earnings_date"]
                days_until = (next_earnings - datetime.now()).days

                if 0 <= days_until <= self.config.filters.earnings_days_before:
                    poor = True
                    reasons.append(f"Earnings in {days_until} days")

            return {
                "poor_fundamentals": poor,
                "good_fundamentals": len(reasons) > 0 and not poor,
                "reason": " | ".join(reasons) if reasons else None,
            }

        except Exception as e:
            logger.warning(f"⚠️ Fundamental check error: {e}")
            return {
                "poor_fundamentals": False,
                "good_fundamentals": False,
                "reason": None,
            }

    def _calculate_prices_and_risk(
        self, df: pd.DataFrame, entry_price: float, sr_check: Dict
    ) -> Tuple[bool, str, float, float, List[float], float]:
        """Calculate prices and risk/reward"""
        atr = IndicatorUtils.get_atr(df)
        support_level = sr_check.get("support_level", None)

        try:
            stop_loss, sl_reason = StopLossCalculator.calculate_stop_loss(
                entry_price=entry_price,
                atr=atr,
                support_level=support_level,
                atr_multiplier=self.config.risk.stop_loss_atr_multiplier,
            )
        except ValueError as e:
            return False, f"Stop loss calculation failed: {str(e)}", 0, 0, [], 0

        try:
            tp_targets = StopLossCalculator.calculate_take_profit_targets(
                entry_price=entry_price,
                atr=atr,
                risk_reward_ratios=self.config.risk.take_profit_ratios,
            )
        except ValueError as e:
            return False, f"Take profit calculation failed: {str(e)}", 0, 0, [], 0

        risk = entry_price - stop_loss
        if risk <= 0:
            return False, f"Invalid risk: {risk:.0f}", 0, 0, [], 0

        if len(tp_targets) < 2:
            return False, "Insufficient TP targets", 0, 0, tp_targets, 0

        reward = tp_targets[1] - entry_price
        if reward <= 0:
            return False, f"Invalid reward: {reward:.0f}", 0, 0, [], 0

        risk_reward = reward / risk
        if risk_reward < self.config.risk.min_risk_reward:
            return False, f"R:R too low: {risk_reward:.2f}", 0, 0, [], 0

        return True, "", stop_loss, reward, tp_targets, risk_reward

    def _optimize_entry_price(
        self,
        df: pd.DataFrame,
        current_price: float,
        sr_check: Dict,
        market_regime: Optional[Dict],
    ) -> Dict:
        """Entry price optimization"""
        # Simplified - see original for full logic
        return {
            "entry_price": current_price,
            "entry_type": "MARKET",
            "optimization_reason": None,
        }

    def _calculate_signal_strength(
        self, confidence: int, risk_reward: float, warnings: List[str]
    ) -> SignalStrength:
        """Calculate signal strength"""
        score = confidence / 20

        if risk_reward >= 3:
            score += 1
        elif risk_reward >= 2.5:
            score += 0.5

        score -= len(warnings) * 0.5

        if score >= self.config.risk.strength_very_strong:
            return SignalStrength.VERY_STRONG
        elif score >= self.config.risk.strength_strong:
            return SignalStrength.STRONG
        elif score >= self.config.risk.strength_moderate:
            return SignalStrength.MODERATE
        elif score >= self.config.risk.strength_weak:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK

    def _calculate_position_multiplier(
        self,
        strength: SignalStrength,
        confidence: int,
        warnings: List[str],
        market_regime: Optional[Dict],
    ) -> float:
        """Calculate position size multiplier"""
        base_multipliers = {
            SignalStrength.VERY_STRONG: 1.3,
            SignalStrength.STRONG: 1.1,
            SignalStrength.MODERATE: 1.0,
            SignalStrength.WEAK: 0.7,
            SignalStrength.VERY_WEAK: 0.5,
        }

        multiplier = base_multipliers.get(strength, 1.0)

        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            if regime == "BULL":
                multiplier *= 1.1
            elif regime == "SIDEWAYS":
                multiplier *= 0.9

        multiplier -= len(warnings) * 0.1

        return max(
            self.config.risk.position_multiplier_min,
            min(multiplier, self.config.risk.position_multiplier_max),
        )

    def _apply_performance_feedback(self, confidence: float) -> Tuple[float, Optional[str]]:
        """Apply performance feedback"""
        if not self.performance_monitor:
            return confidence, None

        try:
            metrics = self.performance_monitor.get_metrics()
        except Exception:
            return confidence, None

        total_trades = metrics.get("total_trades", 0)
        if total_trades < self.config.performance.min_trades_for_feedback:
            return confidence, None

        win_rate = metrics.get("win_rate", 0)
        adjustment = 0

        if win_rate >= self.config.performance.win_rate_good_threshold:
            adjustment = self.config.performance.confidence_adjustment_good
        elif win_rate <= self.config.performance.win_rate_poor_threshold:
            adjustment = self.config.performance.confidence_adjustment_poor

        new_confidence = max(0, min(100, confidence + adjustment))

        if adjustment > 0:
            return new_confidence, f"📈 Good performance (WR {win_rate:.1f}%)"
        elif adjustment < 0:
            return new_confidence, f"⚠️ Poor performance (WR {win_rate:.1f}%)"

        return confidence, None

    def _adjust_thresholds_for_market(self, market_regime: Optional[Dict]):
        """Dynamically adjust thresholds based on market regime"""
        if not market_regime:
            self.config.min_confidence = self.config.base_min_confidence
            return

        regime = market_regime.get("regime", "SIDEWAYS")
        regime_confidence = market_regime.get("confidence", 50)

        adjustment = 0

        if regime == "BULL" and regime_confidence >= 70:
            adjustment = self.config.regime.bull_confidence_adjustment
        elif regime == "BEAR":
            adjustment = self.config.regime.bear_confidence_adjustment
        elif regime == "HIGH_VOLATILITY":
            adjustment = self.config.regime.high_vol_confidence_adjustment

        # Portfolio heat adjustment
        if self.portfolio_manager:
            try:
                positions = self.portfolio_manager.get_positions()
                num_positions = len(positions)

                if num_positions >= self.config.regime.portfolio_heat_threshold_2:
                    adjustment += self.config.regime.portfolio_heat_adjustment_2
                elif num_positions >= self.config.regime.portfolio_heat_threshold_1:
                    adjustment += self.config.regime.portfolio_heat_adjustment_1
            except Exception as e:
                logger.warning(f"⚠️ Portfolio heat check error: {e}")

        # Apply with bounds
        self.config.min_confidence = max(
            self.config.min_confidence_lower_bound,
            min(
                self.config.min_confidence_upper_bound,
                self.config.base_min_confidence + adjustment,
            ),
        )

        if adjustment != 0:
            logger.info(
                f"📊 Dynamic threshold: {self.config.base_min_confidence} → {self.config.min_confidence} "
                f"(regime: {regime}, adj: {adjustment:+d})"
            )

    def _no_signal(self, reason: str, telemetry: Optional[Dict] = None) -> EntrySignal:
        """Return no signal"""
        return EntrySignal(
            should_enter=False,
            signal_type="HOLD",
            confidence=0,
            strength=SignalStrength.NO_SIGNAL,
            position_size_multiplier=0.0,
            reasons=[],
            warnings=[reason],
            entry_price=0,
            stop_loss=0,
            take_profit_targets=[],
            is_limit_order=False,
            limit_price=None,
            entry_type="MARKET",
            telemetry=telemetry,
        )

    def format_signal_message(self, signal: EntrySignal, symbol: str) -> str:
        """Format signal to message"""
        if not signal.should_enter:
            return f"⏭️ **{symbol}** - No entry\nReason: {', '.join(signal.warnings)}"

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


# Testing
if __name__ == "__main__":
    from src.data.loader import load_data
    from src.ml.features.technical import add_ml_features
    from src.ml.signals.generator import MLSignalGenerator

    print("\n" + "=" * 70)
    print("🧪 TESTING SIMPLIFIED ENTRY LOGIC")
    print("=" * 70 + "\n")

    symbol = "VNM"
    df = load_data(symbol, 200)
    index_df = load_data("VNINDEX", 200, is_index=True)
    df = add_ml_features(df, index_df=index_df)

    ml_gen = MLSignalGenerator()
    ml_signal = ml_gen.analyze(df, index_df)

    print(f"📊 ML Signal: {ml_signal['signal']} ({ml_signal['confidence']}%)")

    entry_logic = SimplifiedEntryLogic()
    signal = entry_logic.analyze_entry(df, ml_signal, symbol=symbol)

    print("\n" + "=" * 70)
    message = entry_logic.format_signal_message(signal, symbol)
    print(message)
    print("=" * 70)
