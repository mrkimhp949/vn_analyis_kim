# -*- coding: utf-8 -*-
"""
Entry Logic Module - Refactored Version

This is the refactored entry logic that uses modularized components:
- technical_checks.py: Technical analysis checks
- technical_scorers.py: Technical indicator scoring
- price_optimizer.py: Price and risk calculations
- sentiment_analyzer.py: Sentiment analysis
- entry_filters.py: Filter implementations
- entry_validators.py: Validation logic

REFACTORED v9.0 (2025-01):
- Reduced from 4400+ lines to ~800 lines
- Delegates to specialized modules
- Maintains same interface for backward compatibility

Version: 9.0.0
Author: Trading Bot Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

# Import constants
from src.config.constants import (
    CORRELATION_CACHE_TTL,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    TECH_ONLY_MIN_CONFIDENCE,
    VN_CEILING_DISTANCE_THRESHOLD,
    VN_CRITICAL_LIQUIDITY_VALUE,
    VN_FLOOR_DISTANCE_THRESHOLD,
    VN_FLOOR_PENALTY,
    VN_MIN_LIQUIDITY_VALUE,
    VIETNAM_PRICE_LIMIT_PERCENT,
)
from src.config.exceptions import DataQualityError

# Import utilities
from src.monitoring.performance import get_performance_monitor
from src.utils.indicators import IndicatorUtils, StopLossCalculator
from src.utils.validation import DataValidator
from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

# Import modularized components
from src.strategies.entry_signal import SignalStrength, EntrySignal, create_no_signal
from src.strategies.technical_checks import TechnicalChecker
from src.strategies.technical_scorers import TechnicalScorer
from src.strategies.price_optimizer import PriceOptimizer, RiskRewardCalculator
from src.strategies.sentiment_analyzer import SentimentAnalyzer, VolumeAnalyzer

# Import special instruments handler for Warrant/ETF
SPECIAL_INSTRUMENTS_AVAILABLE = False
try:
    from src.strategies.warrant_etf_strategy import (
        get_special_instruments_handler,
        detect_instrument_type,
        InstrumentType,
    )

    SPECIAL_INSTRUMENTS_AVAILABLE = True
except ImportError:
    pass

# Type checking imports
if TYPE_CHECKING:
    from src.portfolio.manager import PortfolioManager
    from src.monitoring.performance import PerformanceMonitor

# Optional module imports with availability flags
TRADING_SCHEDULE_AVAILABLE = False
ENHANCED_FILTERS_AVAILABLE = False
SESSION_TRADING_AVAILABLE = False
FUNDAMENTAL_AVAILABLE = False

try:
    from src.market.schedule import is_trading_hour, is_trading_day

    TRADING_SCHEDULE_AVAILABLE = True
except ImportError:
    pass

try:
    from src.strategies.enhanced_entry_filters import (
        EnhancedEntryFilters,
        get_enhanced_entry_filters,
    )

    ENHANCED_FILTERS_AVAILABLE = True
except ImportError:
    pass

try:
    from src.market.session_trading import (
        get_session_manager,
        analyze_entry_timing,
        is_optimal_entry_time,
    )

    SESSION_TRADING_AVAILABLE = True
except ImportError:
    pass

try:
    from src.data.fundamental_analyzer import (
        get_fundamental_analyzer,
        is_near_earnings,
    )

    FUNDAMENTAL_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ImprovedEntryLogic:
    """
    Enhanced entry signal logic with 7 core filters optimized for Vietnam market.

    Core Filters (Always applied):
        1. Market Regime - Market must be tradeable
        2. Vietnam Price Limits - Avoid floor/ceiling (±7%)
        3. Trend Alignment - EMA alignment (20/50)
        4. Liquidity Check - Tiered thresholds + Vietnam min 2B VND
        5. Volatility Filter - ATR/Price in acceptable range
        6. RSI Check - Avoid overbought (>70), favor oversold (<30)
        7. Portfolio Correlation - Max 0.7 correlation for diversification

    This refactored version delegates to specialized modules:
    - TechnicalChecker: All _check_* methods
    - TechnicalScorer: All _score_* methods
    - PriceOptimizer: Price and risk calculations
    - SentimentAnalyzer: Sentiment analysis
    """

    # Adaptive thresholds by market regime
    REGIME_THRESHOLDS = {
        "BULL": {
            "min_confidence": 50,
            "min_risk_reward": 1.8,
            "position_multiplier": 1.2,
            "max_warnings": 6,
        },
        "SIDEWAYS": {
            "min_confidence": 60,
            "min_risk_reward": 2.0,
            "position_multiplier": 1.0,
            "max_warnings": 5,
        },
        "BEAR": {
            "min_confidence": 70,
            "min_risk_reward": 2.5,
            "position_multiplier": 0.6,
            "max_warnings": 3,
        },
        "HIGH_VOLATILITY": {
            "min_confidence": 75,
            "min_risk_reward": 3.0,
            "position_multiplier": 0.5,
            "max_warnings": 2,
        },
    }

    # Liquidity tiers for Vietnam market
    DEFAULT_LIQUIDITY_TIERS = {
        "large": {"min_value": 10_000_000_000, "min_volume": 500_000},  # VN30
        "mid": {"min_value": 3_000_000_000, "min_volume": 200_000},
        "small": {"min_value": 1_000_000_000, "min_volume": 50_000},
    }

    def __init__(
        self,
        min_confidence: int = 45,
        min_risk_reward: float = 1.0,
        support_distance_percent: float = 7.0,
        require_trend_alignment: bool = False,
        require_volume_confirmation: bool = False,
        regime_aware_filtering: bool = True,
        portfolio_manager: Optional["PortfolioManager"] = None,
        performance_monitor: Optional["PerformanceMonitor"] = None,
        min_liquidity_value: float = 1_000_000_000,
        min_avg_volume: int = 50_000,
        use_tiered_liquidity: bool = True,
        use_price_action_filter: bool = False,
        use_sector_strength_filter: bool = True,
        use_market_breadth_filter: bool = True,  # ENABLED: Important for regime confirmation
        use_monthly_timeframe: bool = False,
        soft_filter_mode: bool = True,
        max_warnings_allowed: int = 5,
    ) -> None:
        """Initialize entry logic with configurable parameters."""
        # Core settings
        self.min_confidence = min_confidence
        self.base_min_confidence = min_confidence
        self.base_min_risk_reward = min_risk_reward
        self.min_risk_reward = min_risk_reward
        self.support_distance_percent = support_distance_percent
        self.require_trend_alignment = require_trend_alignment
        self.require_volume_confirmation = require_volume_confirmation
        self.regime_aware_filtering = regime_aware_filtering

        # Dependencies
        self.portfolio_manager = portfolio_manager
        self.performance_monitor = performance_monitor or get_performance_monitor()

        # Liquidity settings
        self.min_liquidity_value = min_liquidity_value
        self.min_avg_volume = min_avg_volume
        self.use_tiered_liquidity = use_tiered_liquidity
        self.liquidity_tiers = self.DEFAULT_LIQUIDITY_TIERS.copy()

        # Optional filter flags
        self.use_price_action_filter = use_price_action_filter
        self.use_sector_strength_filter = use_sector_strength_filter
        self.use_market_breadth_filter = use_market_breadth_filter
        self.use_monthly_timeframe = use_monthly_timeframe
        self.use_sentiment_filter = True

        # Soft filter mode
        self.soft_filter_mode = soft_filter_mode
        self.max_warnings_allowed = max_warnings_allowed
        self.base_max_warnings = max_warnings_allowed

        # Internal state
        self._current_symbol: Optional[str] = None
        self._is_technical_only: bool = False
        self._regime_position_multiplier: float = 1.0

        # Filter tracking
        self._filter_pass_count: Dict[str, int] = {}
        self._filter_fail_count: Dict[str, int] = {}

        # Initialize modular components
        self._technical_checker = TechnicalChecker(
            support_distance_percent=support_distance_percent,
            portfolio_manager=portfolio_manager,
        )
        self._technical_scorer = TechnicalScorer()
        self._price_optimizer = PriceOptimizer(min_risk_reward=min_risk_reward)
        self._sentiment_analyzer = SentimentAnalyzer()
        self._volume_analyzer = VolumeAnalyzer()

    def _adjust_thresholds_for_market(self, market_regime: Optional[Dict]) -> None:
        """Dynamically adjust entry thresholds based on market regime."""
        if not self.regime_aware_filtering or not market_regime:
            self._regime_position_multiplier = 1.0
            self.min_risk_reward = self.base_min_risk_reward
            self.max_warnings_allowed = self.base_max_warnings
            return

        regime = market_regime.get("regime", "SIDEWAYS")
        thresholds = self.REGIME_THRESHOLDS.get(regime, self.REGIME_THRESHOLDS["SIDEWAYS"])

        self.min_confidence = thresholds["min_confidence"]
        self.min_risk_reward = thresholds["min_risk_reward"]
        self._regime_position_multiplier = thresholds["position_multiplier"]
        self.max_warnings_allowed = thresholds["max_warnings"]

        logger.debug(
            f"Adjusted thresholds for {regime}: "
            f"min_conf={self.min_confidence}, min_rr={self.min_risk_reward}, "
            f"pos_mult={self._regime_position_multiplier}"
        )

    def _track_filter(self, filter_name: str, passed: bool, symbol: str) -> None:
        """Track filter pass/fail for performance analysis."""
        if passed:
            self._filter_pass_count[filter_name] = self._filter_pass_count.get(filter_name, 0) + 1
        else:
            self._filter_fail_count[filter_name] = self._filter_fail_count.get(filter_name, 0) + 1

    def _add_adjustment(
        self,
        adjustments: List[int],
        breakdown: List[Dict],
        source: str,
        delta: int,
        note: str,
    ) -> None:
        """Add a confidence adjustment with tracking."""
        adjustments.append(delta)
        breakdown.append(
            {
                "source": source,
                "delta": delta,
                "note": note,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def _validate_initial_signal(
        self,
        ml_signal: Optional[str],
        ml_confidence: Optional[float],
        df: pd.DataFrame,
    ) -> Tuple[str, float, bool]:
        """Validate and normalize initial ML signal."""
        self._is_technical_only = False

        if ml_signal and ml_confidence is not None:
            if ml_signal.upper() not in ["BUY", "SELL", "HOLD"]:
                logger.warning(f"Invalid ML signal type: {ml_signal}")
                return "HOLD", 0.0, False
            return ml_signal.upper(), float(ml_confidence), True

        # Fallback to technical analysis
        self._is_technical_only = True
        tech_signal = self._technical_scorer.get_technical_signal(df)
        tech_confidence = self._technical_scorer.calculate_technical_confidence(df)

        return tech_signal, tech_confidence, tech_signal != "HOLD"

    def analyze_entry(
        self,
        df: pd.DataFrame,
        ml_signal: Optional[str] = None,
        ml_confidence: Optional[float] = None,
        symbol: Optional[str] = None,
        market_regime: Optional[Dict] = None,
    ) -> EntrySignal:
        """
        Main entry analysis method.

        Args:
            df: DataFrame with OHLCV and indicators
            ml_signal: Optional ML model signal ("BUY", "SELL", "HOLD")
            ml_confidence: Optional ML model confidence (0-100)
            symbol: Stock symbol for tracking
            market_regime: Market regime analysis result

        Returns:
            EntrySignal with recommendation and details
        """
        self._current_symbol = symbol
        telemetry = {"symbol": symbol, "start_time": datetime.now().isoformat()}

        # Validate input data
        if df is None or df.empty:
            return create_no_signal("DataFrame is None or empty", telemetry)

        if len(df) < 20:
            return create_no_signal(f"Insufficient data: {len(df)} < 20 bars", telemetry)

        # Adjust thresholds for market regime
        self._adjust_thresholds_for_market(market_regime)

        # Validate initial signal
        signal_type, confidence, is_valid = self._validate_initial_signal(
            ml_signal, ml_confidence, df
        )

        if not is_valid or signal_type != "BUY":
            return create_no_signal(f"No buy signal: {signal_type}", telemetry)

        # Get current price
        current_price = safe_get_latest(df, "close", 0)
        if current_price <= 0:
            return create_no_signal("Invalid current price", telemetry)

        # Initialize tracking
        reasons: List[str] = []
        warnings: List[str] = []
        adjustments: List[int] = []
        adjustment_breakdown: List[Dict] = []

        # Run all filters
        block_reason = self._run_all_filters(
            df=df,
            current_price=current_price,
            signal_type=signal_type,
            market_regime=market_regime,
            reasons=reasons,
            warnings=warnings,
            adjustments=adjustments,
            adjustment_breakdown=adjustment_breakdown,
        )

        if block_reason:
            telemetry["block_reason"] = block_reason
            return create_no_signal(block_reason, telemetry)

        # Check warning count
        max_warnings = self._get_dynamic_max_warnings(market_regime)
        if len(warnings) > max_warnings:
            reason = f"Too many warnings: {len(warnings)} > {max_warnings}"
            return create_no_signal(reason, telemetry)

        # Calculate final confidence
        final_confidence = confidence + sum(adjustments)
        final_confidence = max(0, min(100, final_confidence))

        # Technical-only mode needs higher confidence
        if self._is_technical_only and final_confidence < TECH_ONLY_MIN_CONFIDENCE:
            return create_no_signal(
                f"Technical-only confidence too low: {final_confidence:.0f}", telemetry
            )

        # Check minimum confidence
        if final_confidence < self.min_confidence:
            return create_no_signal(
                f"Confidence too low: {final_confidence:.0f} < {self.min_confidence}", telemetry
            )

        # Calculate prices and risk/reward
        sr_check = self._technical_checker.check_support_resistance(df, current_price)
        price_result = self._price_optimizer.calculate_prices_and_risk(df, current_price, sr_check)

        if not price_result.success:
            return create_no_signal(price_result.error_message, telemetry)

        # Calculate signal strength
        strength = self._calculate_signal_strength(
            final_confidence, price_result.risk_reward, warnings
        )

        # Calculate position multiplier
        position_multiplier = self._calculate_position_multiplier(
            strength, int(final_confidence), warnings, market_regime
        )

        # Build entry signal
        return EntrySignal(
            should_enter=True,
            signal_type=signal_type,
            confidence=int(final_confidence),
            entry_price=price_result.entry_price,
            stop_loss=price_result.stop_loss,
            take_profit_targets=price_result.take_profit_targets,
            risk_reward=price_result.risk_reward,
            reasons=reasons,
            warnings=warnings,
            strength=strength,
            position_multiplier=position_multiplier,
            adjustment_breakdown=adjustment_breakdown,
            entry_type=price_result.entry_type,
            telemetry=telemetry,
        )

    def _run_all_filters(
        self,
        df: pd.DataFrame,
        current_price: float,
        signal_type: str,
        market_regime: Optional[Dict],
        reasons: List[str],
        warnings: List[str],
        adjustments: List[int],
        adjustment_breakdown: List[Dict],
    ) -> Optional[str]:
        """Run all filters and return block reason if any."""
        symbol = self._current_symbol

        # 1. Vietnam Price Limit Check (Blocking)
        price_limit = self._technical_checker.check_vietnam_price_limits(df, current_price)
        if price_limit["near_limit"]:
            if price_limit["limit_type"] == "CEILING":
                self._track_filter("price_limit", False, symbol or "")
                return f"Blocked: {price_limit['warning']}"
            else:  # FLOOR - warning only
                warnings.append(price_limit["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "vn_floor",
                    VN_FLOOR_PENALTY,
                    price_limit["warning"],
                )
        else:
            self._track_filter("price_limit", True, symbol or "")

        # 2. Trend Alignment Check
        trend_check = self._technical_checker.check_trend_alignment(df, signal_type)
        if not trend_check["aligned"]:
            if self.require_trend_alignment:
                self._track_filter("trend", False, symbol or "")
                return f"Trend not aligned: {trend_check['reason']}"
            warnings.append(f"⚠️ Trend: {trend_check['reason']}")
            self._add_adjustment(
                adjustments, adjustment_breakdown, "trend", -10, trend_check["reason"]
            )
        else:
            reasons.append(f"✅ {trend_check['reason']}")
            strength_bonus = min(15, (trend_check["strength"] - 50) // 3)
            if strength_bonus > 0:
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "trend",
                    strength_bonus,
                    f"Trend strength: {trend_check['strength']}",
                )
        self._track_filter("trend", trend_check["aligned"], symbol or "")

        # 3. Liquidity Check
        liquidity = self._check_liquidity(df, current_price)
        if liquidity["critical"]:
            self._track_filter("liquidity", False, symbol or "")
            return "Critical liquidity issue"
        if not liquidity["sufficient"]:
            warnings.append("⚠️ Low liquidity")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "liquidity",
                -15,
                f"Low liquidity tier: {liquidity['tier']}",
            )
        else:
            reasons.append(f"✅ Good liquidity ({liquidity['tier']})")
        self._track_filter("liquidity", liquidity["sufficient"], symbol or "")

        # 4. Volatility Check
        volatility = self._technical_checker.check_volatility(df)
        if volatility["too_high"]:
            warnings.append(f"⚠️ High volatility: {volatility['value']:.1f}%")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "volatility",
                -10,
                f"Volatility {volatility['value']:.1f}%",
            )
        elif volatility["optimal"]:
            reasons.append(f"✅ Optimal volatility: {volatility['value']:.1f}%")
            self._add_adjustment(
                adjustments, adjustment_breakdown, "volatility", 5, "Optimal range"
            )

        # 5. RSI Check
        rsi = self._technical_checker.check_rsi(df)
        if rsi["overbought"]:
            warnings.append(f"⚠️ RSI overbought: {rsi['value']:.0f}")
            self._add_adjustment(
                adjustments, adjustment_breakdown, "rsi", -15, f"Overbought: {rsi['value']:.0f}"
            )
        elif rsi["oversold"]:
            reasons.append(f"✅ RSI oversold: {rsi['value']:.0f}")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "rsi",
                10,
                f"Oversold opportunity: {rsi['value']:.0f}",
            )
        elif rsi["optimal"]:
            reasons.append(f"✅ RSI optimal: {rsi['value']:.0f}")

        # 6. Volume Confirmation
        volume = self._volume_analyzer.check_volume_confirmation(df, market_regime)
        if volume["surge"]:
            reasons.append(f"✅ Volume surge: {volume['volume_ratio']:.1f}x")
            self._add_adjustment(adjustments, adjustment_breakdown, "volume", 10, "Volume surge")
        elif not volume["confirmed"]:
            warnings.append("⚠️ Volume not confirmed")
            if self.require_volume_confirmation:
                self._track_filter("volume", False, symbol or "")
                return "Volume confirmation required"
            self._add_adjustment(adjustments, adjustment_breakdown, "volume", -5, volume["reason"])

        # 7. Portfolio Correlation
        correlation = self._technical_checker.check_portfolio_correlation(df, symbol)
        if correlation["too_high"]:
            warnings.append(f"⚠️ High correlation: {correlation['max_correlation']:.2f}")
            self._add_adjustment(
                adjustments,
                adjustment_breakdown,
                "correlation",
                -10,
                f"Correlation: {correlation['max_correlation']:.2f}",
            )
        elif correlation["good_diversification"]:
            reasons.append("✅ Good diversification")
            self._add_adjustment(
                adjustments, adjustment_breakdown, "correlation", 5, "Diversification bonus"
            )

        # 8. Support/Resistance Analysis
        sr = self._technical_checker.check_support_resistance(df, current_price)
        if sr["bouncing_from_support"]:
            reasons.append("✅ Bouncing from support")
            self._add_adjustment(adjustments, adjustment_breakdown, "support", 15, "Support bounce")
        elif sr["near_support"]:
            reasons.append("✅ Near support")
            self._add_adjustment(adjustments, adjustment_breakdown, "support", 8, "Near support")
        if sr["too_close_to_resistance"]:
            warnings.append("⚠️ Near resistance")
            self._add_adjustment(
                adjustments, adjustment_breakdown, "resistance", -10, "Near resistance"
            )

        # 9. Sector Strength (Optional) - Enhanced with Rotation Signals
        if self.use_sector_strength_filter:
            sector = self._technical_checker.check_sector_strength(df, market_regime, symbol)

            # Use rotation_bonus if available (from SectorRotationAnalyzer)
            rotation_bonus = sector.get("rotation_bonus", 0)

            if sector.get("in_overweight_sector"):
                reasons.append(f"✅ Overweight sector: {sector['sector_id']} (rotation)")
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "sector",
                    max(10, rotation_bonus),
                    f"Sector rotation overweight ({sector.get('rotation_phase', 'N/A')})",
                )
                # Note top picks if available
                if sector.get("top_sector_picks"):
                    logger.debug(f"Sector top picks: {sector['top_sector_picks']}")

            elif sector.get("in_underweight_sector"):
                warnings.append(f"⚠️ Underweight sector: {sector['sector_id']} (rotation)")
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "sector",
                    min(-10, rotation_bonus),
                    f"Sector rotation underweight ({sector.get('rotation_phase', 'N/A')})",
                )

            elif sector["is_leading"]:
                reasons.append(f"✅ Leading sector: {sector['sector_id']}")
                self._add_adjustment(
                    adjustments, adjustment_breakdown, "sector", 10, "Sector leader"
                )
            elif sector["is_lagging"]:
                warnings.append(f"⚠️ Lagging sector: {sector['sector_id']}")
                self._add_adjustment(
                    adjustments, adjustment_breakdown, "sector", -10, "Sector lagging"
                )

        # 10. Market Breadth (Optional)
        if self.use_market_breadth_filter:
            breadth = self._technical_checker.check_market_breadth(market_regime)
            if breadth.get("weak"):
                warnings.append("⚠️ Weak market breadth")
                self._add_adjustment(
                    adjustments, adjustment_breakdown, "breadth", -10, "Weak breadth"
                )
            elif breadth.get("strong"):
                reasons.append("✅ Strong market breadth")
                self._add_adjustment(
                    adjustments, adjustment_breakdown, "breadth", 5, "Strong breadth"
                )

        # 11. Sentiment Analysis (Optional)
        if self.use_sentiment_filter and symbol:
            sentiment = self._sentiment_analyzer.analyze_sentiment(symbol, df)
            if sentiment["adjustment"] != 0:
                if sentiment["adjustment"] > 0:
                    reasons.append(f"✅ {sentiment['sentiment']} sentiment")
                else:
                    warnings.append(f"⚠️ {sentiment['sentiment']} sentiment")
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "sentiment",
                    sentiment["adjustment"],
                    sentiment.get("reason", ""),
                )

        # 12. Special Instruments Check (Warrant/ETF) - NEW 10/10
        if SPECIAL_INSTRUMENTS_AVAILABLE and symbol:
            special_result = self._check_special_instruments(
                symbol, df, adjustments, adjustment_breakdown
            )
            if special_result.get("blocked"):
                return special_result.get("reason")
            if special_result.get("warnings"):
                warnings.extend(special_result["warnings"])
            if special_result.get("reasons"):
                reasons.extend(special_result["reasons"])

        return None  # No blocking reason

    def _check_special_instruments(
        self,
        symbol: str,
        df: pd.DataFrame,
        adjustments: List[int],
        adjustment_breakdown: List[Dict],
    ) -> Dict:
        """
        Check special instruments (Warrant/ETF) and apply adjustments.

        NEW 10/10 Implementation:
        - Detect instrument type (Stock/Warrant/ETF)
        - Apply confidence adjustments
        - Add warnings for high-risk instruments

        Returns:
            Dict with blocked, reason, warnings, reasons
        """
        result = {"blocked": False, "reason": None, "warnings": [], "reasons": []}

        try:
            handler = get_special_instruments_handler()
            instrument_type = handler.get_instrument_type(symbol)

            if instrument_type == InstrumentType.WARRANT:
                # Warrants are high risk - add warnings
                result["warnings"].append("⚠️ WARRANT: High leverage instrument")
                result["warnings"].append("⚠️ WARRANT: Check expiry date before trading")

                # Apply confidence penalty for warrants
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "warrant",
                    -15,
                    "Warrant high-risk penalty",
                )

                # Get additional confidence adjustment
                conf_adj, warnings = handler.get_confidence_adjustment(symbol, df)
                if conf_adj != 0:
                    self._add_adjustment(
                        adjustments,
                        adjustment_breakdown,
                        "warrant_analysis",
                        conf_adj,
                        f"Warrant analysis: {warnings[0] if warnings else 'N/A'}",
                    )
                result["warnings"].extend(warnings)

            elif instrument_type == InstrumentType.ETF:
                # ETFs are lower risk - add positive note
                result["reasons"].append("✅ ETF: Diversified exposure, lower volatility")

                # Get confidence adjustment
                conf_adj, warnings = handler.get_confidence_adjustment(symbol, df)
                if conf_adj != 0:
                    self._add_adjustment(
                        adjustments,
                        adjustment_breakdown,
                        "etf_analysis",
                        conf_adj,
                        f"ETF analysis: {warnings[0] if warnings else 'Good'}",
                    )
                result["warnings"].extend(warnings)

        except Exception as e:
            logger.debug(f"Special instruments check error: {e}")

        return result

    def _check_liquidity(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Check liquidity with tiered thresholds."""
        if "volume" not in df.columns or len(df) < 5:
            return {"sufficient": True, "critical": False, "tier": "unknown"}

        current_volume = safe_get_latest(df, "volume", 0)
        avg_volume = df["volume"].tail(20).mean()
        avg_value = avg_volume * current_price

        if self.use_tiered_liquidity:
            if avg_value >= self.liquidity_tiers["large"]["min_value"]:
                tier = "large"
            elif avg_value >= self.liquidity_tiers["mid"]["min_value"]:
                tier = "mid"
            elif avg_value >= self.liquidity_tiers["small"]["min_value"]:
                tier = "small"
            else:
                tier = "micro"

            min_value = self.liquidity_tiers.get(tier, self.liquidity_tiers["small"])["min_value"]
            sufficient = avg_value >= min_value
            critical = avg_value < (min_value * 0.5)
        else:
            tier = "fixed"
            sufficient = avg_value >= self.min_liquidity_value
            critical = avg_value < (self.min_liquidity_value * 0.5)

        return {
            "sufficient": sufficient,
            "critical": critical,
            "avg_value": avg_value,
            "avg_volume": avg_volume,
            "tier": tier,
        }

    def _get_dynamic_max_warnings(self, market_regime: Optional[Dict]) -> int:
        """Get dynamic max warnings based on market regime."""
        if not market_regime:
            return self.base_max_warnings

        regime = market_regime.get("regime", "SIDEWAYS")
        return self.REGIME_THRESHOLDS.get(regime, {}).get("max_warnings", self.base_max_warnings)

    def _calculate_signal_strength(
        self, confidence: int, risk_reward: float, warnings: list
    ) -> SignalStrength:
        """Calculate signal strength from confidence and R:R."""
        score = confidence / 20  # 0-5

        if risk_reward >= 3:
            score += 1
        elif risk_reward >= 2.5:
            score += 0.5

        score -= len(warnings) * 0.5

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

    def _calculate_position_multiplier(
        self,
        strength: SignalStrength,
        confidence: int,
        warnings: list,
        market_regime: Optional[Dict],
    ) -> float:
        """Calculate position size multiplier."""
        base_multipliers = {
            SignalStrength.VERY_STRONG: 1.3,
            SignalStrength.STRONG: 1.1,
            SignalStrength.MODERATE: 1.0,
            SignalStrength.WEAK: 0.7,
            SignalStrength.VERY_WEAK: 0.5,
        }

        multiplier = base_multipliers.get(strength, 1.0)

        # Apply regime multiplier
        multiplier *= self._regime_position_multiplier

        # Reduce for warnings
        if len(warnings) > 3:
            multiplier *= 0.8

        # NEW 10/10: Apply special instrument multiplier (Warrant/ETF)
        if SPECIAL_INSTRUMENTS_AVAILABLE and self._current_symbol:
            try:
                handler = get_special_instruments_handler()
                instrument_mult = handler.get_position_size_multiplier(self._current_symbol)
                multiplier *= instrument_mult

                if instrument_mult != 1.0:
                    logger.debug(
                        f"[{self._current_symbol}] Special instrument multiplier: {instrument_mult:.2f}"
                    )
            except Exception as e:
                logger.debug(f"Special instrument multiplier error: {e}")

        return max(0.3, min(1.5, multiplier))

    def get_filter_stats(self) -> Dict:
        """Get filter performance statistics."""
        stats = {}
        all_filters = set(self._filter_pass_count.keys()) | set(self._filter_fail_count.keys())

        for filter_name in all_filters:
            passed = self._filter_pass_count.get(filter_name, 0)
            failed = self._filter_fail_count.get(filter_name, 0)
            total = passed + failed
            stats[filter_name] = {
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total if total > 0 else 0,
            }

        return stats

    def get_current_thresholds(self) -> Dict:
        """Get current threshold settings."""
        return {
            "min_confidence": self.min_confidence,
            "min_risk_reward": self.min_risk_reward,
            "max_warnings": self.max_warnings_allowed,
            "position_multiplier": self._regime_position_multiplier,
        }

    def reset_thresholds(self) -> None:
        """Reset thresholds to base values."""
        self.min_confidence = self.base_min_confidence
        self.min_risk_reward = self.base_min_risk_reward
        self.max_warnings_allowed = self.base_max_warnings
        self._regime_position_multiplier = 1.0

    def format_signal_message(self, signal: EntrySignal, symbol: str) -> str:
        """Format signal for display."""
        if not signal.should_enter:
            return f"❌ {symbol}: No entry - {signal.reasons[0] if signal.reasons else 'Unknown'}"

        msg_parts = [
            f"✅ {symbol}: {signal.signal_type}",
            f"Confidence: {signal.confidence}%",
            f"R:R: {signal.risk_reward:.2f}",
            f"Strength: {signal.strength.value}",
        ]

        if signal.warnings:
            msg_parts.append(f"Warnings: {len(signal.warnings)}")

        return " | ".join(msg_parts)
