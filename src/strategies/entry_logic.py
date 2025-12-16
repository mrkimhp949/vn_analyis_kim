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

# NEW v9.1: Order Book Integration for Entry Timing
ORDER_BOOK_AVAILABLE = False
try:
    from src.strategies.order_book_integration import (
        get_order_book_integration,
        OrderBookIntegration,
    )

    ORDER_BOOK_AVAILABLE = True
except ImportError:
    pass

# NEW v9.2: Vietnamese News Sentiment Integration
VN_NEWS_SENTIMENT_AVAILABLE = False
try:
    from src.sentiment.vn_news_sentiment_integration import (
        get_news_sentiment_integration,
        VNNewsSentimentIntegration,
        SentimentDirection,
    )

    VN_NEWS_SENTIMENT_AVAILABLE = True
except ImportError:
    pass

# NEW v9.3: Market Breadth Integration
MARKET_BREADTH_AVAILABLE = False
try:
    from src.market.market_breadth import (
        get_breadth_analyzer,
        MarketBreadthAnalyzer,
        BreadthSignal,
    )

    MARKET_BREADTH_AVAILABLE = True
except ImportError:
    pass

# NEW v9.4: Event Calendar Integration
EVENT_CALENDAR_AVAILABLE = False
try:
    from src.market.event_calendar import (
        get_event_calendar,
        VietnamEventCalendar,
        EventImpact,
    )

    EVENT_CALENDAR_AVAILABLE = True
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
        min_confidence: int = 55,  # IMPROVED: Raised from 45 for VN market volatility
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
        use_order_book_timing: bool = True,  # NEW: Enable order book integration
        use_intraday_momentum_filter: bool = True,  # NEW: Block entries when price moved >3% intraday
        use_vn30_correlation_filter: bool = True,  # NEW: Check correlation with VN30
        use_vn_news_sentiment_filter: bool = True,  # NEW v9.2: Vietnamese news sentiment integration
        intraday_momentum_threshold: float = 0.03,  # NEW: 3% intraday move threshold
        vn30_divergence_threshold: float = 0.04,  # NEW: 4% divergence from VN30 threshold
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

        # NEW: Order Book and Intraday Momentum filters
        self.use_order_book_timing = use_order_book_timing
        self.use_intraday_momentum_filter = use_intraday_momentum_filter
        self.use_vn30_correlation_filter = use_vn30_correlation_filter
        self.use_vn_news_sentiment_filter = use_vn_news_sentiment_filter  # NEW v9.2
        self.intraday_momentum_threshold = intraday_momentum_threshold
        self.vn30_divergence_threshold = vn30_divergence_threshold

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

        # NEW v9.3: Market Breadth Analyzer
        self._breadth_analyzer = None
        if MARKET_BREADTH_AVAILABLE and self.use_market_breadth_filter:
            try:
                self._breadth_analyzer = get_breadth_analyzer()
                logger.info("📊 Market Breadth Analyzer integrated into entry logic")
            except Exception as e:
                logger.warning(f"Failed to initialize breadth analyzer: {e}")

        # NEW v9.4: Event Calendar
        self._event_calendar = None
        if EVENT_CALENDAR_AVAILABLE:
            try:
                self._event_calendar = get_event_calendar()
                logger.info("📅 Event Calendar integrated into entry logic")
            except Exception as e:
                logger.warning(f"Failed to initialize event calendar: {e}")

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

        # 13. Intraday Momentum Filter (NEW) - Block entries when price moved >3% intraday
        if self.use_intraday_momentum_filter:
            intraday_result = self._check_intraday_momentum(df, current_price)
            if intraday_result.get("blocked"):
                self._track_filter("intraday_momentum", False, symbol or "")
                return intraday_result.get("reason")
            if intraday_result.get("warning"):
                warnings.append(intraday_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "intraday_momentum",
                    intraday_result.get("adjustment", -10),
                    intraday_result.get("note", "High intraday move"),
                )
            self._track_filter("intraday_momentum", True, symbol or "")

        # 14. VN30 Correlation Filter (NEW) - Check if stock diverges from VN30
        if self.use_vn30_correlation_filter and symbol:
            vn30_result = self._check_vn30_divergence(df, symbol, market_regime)
            if vn30_result.get("blocked"):
                self._track_filter("vn30_correlation", False, symbol or "")
                return vn30_result.get("reason")
            if vn30_result.get("warning"):
                warnings.append(vn30_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "vn30_correlation",
                    vn30_result.get("adjustment", -10),
                    vn30_result.get("note", "VN30 divergence"),
                )
            elif vn30_result.get("positive"):
                reasons.append(vn30_result["positive"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "vn30_correlation",
                    5,
                    "Aligned with VN30",
                )
            self._track_filter("vn30_correlation", True, symbol or "")

        # 15. Order Book Timing (NEW) - Analyze order book for optimal entry
        if self.use_order_book_timing and ORDER_BOOK_AVAILABLE and symbol:
            order_book_result = self._check_order_book_timing(symbol, current_price)
            if order_book_result.get("blocked"):
                self._track_filter("order_book", False, symbol or "")
                return order_book_result.get("reason")
            if order_book_result.get("warning"):
                warnings.append(order_book_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "order_book",
                    order_book_result.get("adjustment", -5),
                    order_book_result.get("note", "Order book pressure"),
                )
            elif order_book_result.get("positive"):
                reasons.append(order_book_result["positive"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "order_book",
                    order_book_result.get("adjustment", 10),
                    order_book_result.get("note", "Strong bid support"),
                )
            self._track_filter("order_book", True, symbol or "")

        # 16. Vietnamese News Sentiment (NEW v9.2) - Check news for blocking/adjustment
        if self.use_vn_news_sentiment_filter and VN_NEWS_SENTIMENT_AVAILABLE and symbol:
            news_result = self._check_vn_news_sentiment(symbol)
            if news_result.get("blocked"):
                self._track_filter("vn_news_sentiment", False, symbol or "")
                return news_result.get("reason")
            if news_result.get("warning"):
                warnings.append(news_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "vn_news_sentiment",
                    news_result.get("adjustment", -10),
                    news_result.get("note", "Negative news sentiment"),
                )
            elif news_result.get("positive"):
                reasons.append(news_result["positive"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "vn_news_sentiment",
                    news_result.get("adjustment", 5),
                    news_result.get("note", "Positive news sentiment"),
                )
            # Apply position size multiplier from news sentiment
            if news_result.get("position_multiplier", 1.0) != 1.0:
                self._regime_position_multiplier *= news_result.get("position_multiplier", 1.0)
            self._track_filter("vn_news_sentiment", True, symbol or "")

        # 17. Advanced Market Breadth (NEW v9.3) - Real-time breadth analyzer
        if MARKET_BREADTH_AVAILABLE and self._breadth_analyzer and symbol:
            breadth_result = self._check_advanced_breadth()
            if breadth_result.get("blocked"):
                self._track_filter("advanced_breadth", False, symbol or "")
                return breadth_result.get("reason")
            if breadth_result.get("warning"):
                warnings.append(breadth_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "advanced_breadth",
                    breadth_result.get("adjustment", -10),
                    breadth_result.get("note", "Weak market breadth"),
                )
            elif breadth_result.get("positive"):
                reasons.append(breadth_result["positive"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "advanced_breadth",
                    breadth_result.get("adjustment", 5),
                    breadth_result.get("note", "Strong market breadth"),
                )
            # Apply position multiplier from breadth
            if breadth_result.get("position_multiplier", 1.0) != 1.0:
                self._regime_position_multiplier *= breadth_result.get("position_multiplier", 1.0)
            self._track_filter("advanced_breadth", True, symbol or "")

        # 18. Event Calendar Check (NEW v9.4) - Check for high-impact events
        if EVENT_CALENDAR_AVAILABLE and self._event_calendar and symbol:
            event_result = self._check_event_calendar()
            if event_result.get("blocked"):
                self._track_filter("event_calendar", False, symbol or "")
                return event_result.get("reason")
            if event_result.get("warning"):
                warnings.append(event_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "event_calendar",
                    event_result.get("adjustment", -5),
                    event_result.get("note", "High-impact event approaching"),
                )
            # Apply position multiplier for events
            if event_result.get("position_multiplier", 1.0) != 1.0:
                self._regime_position_multiplier *= event_result.get("position_multiplier", 1.0)
            self._track_filter("event_calendar", True, symbol or "")

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

    def _check_intraday_momentum(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Check intraday momentum to avoid chasing extended moves.

        NEW IMPROVEMENT: Block entries when price has moved >3% intraday
        to avoid buying at extended levels.

        Args:
            df: DataFrame with OHLCV data
            current_price: Current price

        Returns:
            Dict with blocked, warning, adjustment, note
        """
        result = {"blocked": False, "warning": None, "adjustment": 0, "note": None}

        try:
            # Get today's open price (first bar or use 'open' of latest bar)
            if "open" not in df.columns:
                return result

            # Use latest bar's open as proxy for session open
            today_open = safe_get_latest(df, "open", 0)
            if today_open <= 0:
                return result

            # Calculate intraday change
            intraday_change = (current_price - today_open) / today_open

            # Check if already moved significantly
            if abs(intraday_change) >= self.intraday_momentum_threshold:
                if intraday_change > 0:
                    # Price already up >3% - risky to chase
                    if intraday_change >= self.intraday_momentum_threshold * 1.5:  # >4.5%
                        result["blocked"] = True
                        result["reason"] = (
                            f"Price already up {intraday_change*100:.1f}% today - too extended"
                        )
                    else:
                        result["warning"] = f"⚠️ Intraday up {intraday_change*100:.1f}% - extended"
                        result["adjustment"] = -10
                        result["note"] = f"Chasing momentum ({intraday_change*100:.1f}%)"
                else:
                    # Price down >3% - might be catching falling knife
                    if intraday_change <= -self.intraday_momentum_threshold * 1.5:  # <-4.5%
                        result["blocked"] = True
                        result["reason"] = (
                            f"Price down {intraday_change*100:.1f}% today - potential falling knife"
                        )
                    else:
                        result["warning"] = f"⚠️ Intraday down {intraday_change*100:.1f}% - cautious"
                        result["adjustment"] = -5
                        result["note"] = f"Catching dip ({intraday_change*100:.1f}%)"

            # Also check if near daily high (could be distribution)
            if "high" in df.columns:
                today_high = safe_get_latest(df, "high", 0)
                if today_high > 0:
                    distance_from_high = (today_high - current_price) / today_high
                    if distance_from_high < 0.005:  # Within 0.5% of high
                        result["warning"] = result.get("warning", "") + " | Near daily high"
                        result["adjustment"] = result.get("adjustment", 0) - 5

        except Exception as e:
            logger.debug(f"Intraday momentum check error: {e}")

        return result

    def _check_vn30_divergence(
        self, df: pd.DataFrame, symbol: str, market_regime: Optional[Dict]
    ) -> Dict:
        """
        Check if stock is diverging from VN30 index.

        NEW IMPROVEMENT: Avoid entries when stock diverges significantly
        from VN30 direction.

        Args:
            df: Stock DataFrame
            symbol: Stock symbol
            market_regime: Market regime info

        Returns:
            Dict with blocked, warning, positive, adjustment, note
        """
        result = {
            "blocked": False,
            "warning": None,
            "positive": None,
            "adjustment": 0,
            "note": None,
        }

        try:
            # Try to get VN30 data from regime info
            if market_regime is None:
                return result

            # Get VN30/VNINDEX change from regime components
            components = market_regime.get("components", {})
            vnindex_change = components.get("vnindex_change", 0)

            # Calculate stock's recent change (5-day)
            if len(df) < 5:
                return result

            stock_change = (df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5]

            # Check divergence
            divergence = stock_change - vnindex_change

            if abs(divergence) >= self.vn30_divergence_threshold:
                if divergence > 0 and vnindex_change < -0.01:
                    # Stock up while market down - could be unsustainable
                    result["warning"] = (
                        f"⚠️ Stock up {stock_change*100:.1f}% vs VN30 down {vnindex_change*100:.1f}%"
                    )
                    result["adjustment"] = -10
                    result["note"] = "Diverging from market - risky"
                elif divergence < 0 and vnindex_change > 0.01:
                    # Stock down while market up - could be weak stock
                    if (
                        divergence <= -self.vn30_divergence_threshold * 1.5
                    ):  # Heavy underperformance
                        result["blocked"] = True
                        result["reason"] = (
                            f"Stock lagging VN30 by {abs(divergence)*100:.1f}% - weak stock"
                        )
                    else:
                        result["warning"] = (
                            f"⚠️ Stock down {stock_change*100:.1f}% vs VN30 up {vnindex_change*100:.1f}%"
                        )
                        result["adjustment"] = -15
                        result["note"] = "Underperforming market"
            elif abs(divergence) < 0.01 and vnindex_change > 0.01:
                # Stock moving with market in uptrend - good
                result["positive"] = f"✅ Aligned with VN30 ({vnindex_change*100:.1f}%)"

        except Exception as e:
            logger.debug(f"VN30 divergence check error: {e}")

        return result

    def _check_order_book_timing(self, symbol: str, current_price: float) -> Dict:
        """
        Check order book for optimal entry timing.

        NEW IMPROVEMENT: Analyze order book depth and imbalance
        to optimize entry timing.

        Args:
            symbol: Stock symbol
            current_price: Current price

        Returns:
            Dict with blocked, warning, positive, adjustment, note
        """
        result = {
            "blocked": False,
            "warning": None,
            "positive": None,
            "adjustment": 0,
            "note": None,
        }

        try:
            if not ORDER_BOOK_AVAILABLE:
                return result

            integration = get_order_book_integration()
            analysis = integration.analyze_entry_timing(symbol, current_price)

            if analysis is None:
                return result

            # Check order book signal
            signal = analysis.get("signal")
            imbalance = analysis.get("imbalance", 0)
            spread_pct = analysis.get("spread_pct", 0)

            # Strong sell pressure - block entry
            if signal == "STRONG_SELL_PRESSURE":
                result["blocked"] = True
                result["reason"] = f"Heavy sell pressure in order book (imbalance: {imbalance:.2f})"
                return result

            # Moderate sell pressure - warning
            if signal == "SELL_PRESSURE":
                result["warning"] = f"⚠️ Sell pressure (imbalance: {imbalance:.2f})"
                result["adjustment"] = -8
                result["note"] = "Order book sell pressure"

            # Strong buy pressure - positive
            elif signal == "STRONG_BUY_PRESSURE":
                result["positive"] = f"✅ Strong bid support (imbalance: {imbalance:.2f})"
                result["adjustment"] = 12
                result["note"] = "Strong bid support"

            # Moderate buy pressure - slight positive
            elif signal == "BUY_PRESSURE":
                result["positive"] = f"✅ Bid support (imbalance: {imbalance:.2f})"
                result["adjustment"] = 5
                result["note"] = "Bid support"

            # Wide spread warning
            if spread_pct > 0.01:  # Spread > 1%
                result["warning"] = f"⚠️ Wide spread ({spread_pct*100:.2f}%)"
                result["adjustment"] = result.get("adjustment", 0) - 5
                result["note"] = (result.get("note", "") + " | Wide spread").strip(" | ")

            # Get recommended entry price
            if analysis.get("recommended_limit_price"):
                result["recommended_price"] = analysis["recommended_limit_price"]

        except Exception as e:
            logger.debug(f"Order book timing check error: {e}")

        return result

    def _check_vn_news_sentiment(self, symbol: str) -> Dict:
        """
        Check Vietnamese news sentiment for entry decision.

        NEW v9.2: Analyze recent Vietnamese news for the symbol
        and adjust entry confidence accordingly.

        Features:
        - Block entry on severe negative news
        - Boost confidence on positive news
        - Adjust position size based on sentiment

        Args:
            symbol: Stock symbol

        Returns:
            Dict with blocked, warning, positive, adjustment, note, position_multiplier
        """
        result = {
            "blocked": False,
            "warning": None,
            "positive": None,
            "adjustment": 0,
            "note": None,
            "position_multiplier": 1.0,
        }

        try:
            if not VN_NEWS_SENTIMENT_AVAILABLE:
                return result

            integration = get_news_sentiment_integration()
            entry_check = integration.check_entry_sentiment(symbol, side="buy")

            if entry_check is None:
                return result

            # Check if entry should be blocked
            if not entry_check.get("should_proceed", True):
                result["blocked"] = True
                reasons = entry_check.get("reasons", [])
                result["reason"] = reasons[0] if reasons else "Blocked by negative news"
                return result

            # Get sentiment direction and adjustments
            sentiment_direction = entry_check.get("sentiment_direction", "neutral")
            confidence_adj = entry_check.get("confidence_adjustment", 0)
            position_mult = entry_check.get("position_multiplier", 1.0)

            # Convert confidence adjustment to our scale (0-100)
            adjustment = int(confidence_adj * 100)  # -15% to +10% -> -15 to +10

            result["position_multiplier"] = position_mult

            # Determine warning or positive based on sentiment
            if sentiment_direction in ("bearish", "very_bearish"):
                result["warning"] = f"⚠️ Bearish news sentiment ({sentiment_direction})"
                result["adjustment"] = min(adjustment, -5)  # At least -5
                result["note"] = "Negative news detected"
            elif sentiment_direction in ("bullish", "very_bullish"):
                result["positive"] = f"✅ Bullish news sentiment ({sentiment_direction})"
                result["adjustment"] = max(adjustment, 5)  # At least +5
                result["note"] = "Positive news detected"
            else:
                # Neutral - no adjustment
                pass

            # Add reasons to notes if available
            reasons = entry_check.get("reasons", [])
            if reasons and result["note"]:
                result["note"] += f": {reasons[0][:50]}"

        except Exception as e:
            logger.debug(f"VN news sentiment check error: {e}")

        return result

    def _check_advanced_breadth(self) -> Dict:
        """
        Check advanced market breadth for entry decision.

        NEW v9.3: Uses MarketBreadthAnalyzer for real-time breadth analysis.

        Features:
        - Block entry on breadth thrust down
        - Boost confidence on breadth thrust up
        - Adjust position size based on breadth

        Returns:
            Dict with blocked, warning, positive, adjustment, note, position_multiplier
        """
        result = {
            "blocked": False,
            "warning": None,
            "positive": None,
            "adjustment": 0,
            "note": None,
            "position_multiplier": 1.0,
        }

        try:
            if self._breadth_analyzer is None:
                return result

            breadth_check = self._breadth_analyzer.check_breadth_for_entry()

            if breadth_check is None:
                return result

            # Check if entry is favorable
            if not breadth_check.get("is_favorable", True):
                # Block on strong bearish breadth
                signal = breadth_check.get("signal", "")
                if signal in ("breadth_thrust_down", "strong_bearish"):
                    result["blocked"] = True
                    result["reason"] = f"🚫 Market breadth unfavorable: {signal}"
                    return result

            # Get adjustment values
            conf_adj = breadth_check.get("confidence_adjustment", 0)
            pos_mult = breadth_check.get("position_multiplier", 1.0)
            signal = breadth_check.get("signal", "neutral")
            score = breadth_check.get("score", 0)

            # Convert to our scale
            adjustment = int(conf_adj * 100)  # -15% to +10% -> -15 to +10
            result["position_multiplier"] = pos_mult

            # Determine warning or positive
            if signal in ("strong_bearish", "bearish"):
                result["warning"] = f"⚠️ Weak market breadth ({signal})"
                result["adjustment"] = min(adjustment, -5)
                result["note"] = f"A/D ratio unfavorable (score: {score:.2f})"
            elif signal in ("bullish", "strong_bullish", "breadth_thrust_up"):
                result["positive"] = f"✅ Strong market breadth ({signal})"
                result["adjustment"] = max(adjustment, 5)
                result["note"] = f"A/D ratio favorable (score: {score:.2f})"

                if breadth_check.get("should_boost", False):
                    result["adjustment"] += 5  # Extra boost for thrust
                    result["note"] = "🚀 Breadth thrust detected!"

        except Exception as e:
            logger.debug(f"Advanced breadth check error: {e}")

        return result

    def _check_event_calendar(self) -> Dict:
        """
        Check event calendar for trading conditions.

        NEW v9.4: Uses VietnamEventCalendar for event awareness.

        Features:
        - Block entry on market holidays
        - Reduce position size before high-impact events
        - Warn on derivative expiration days

        Returns:
            Dict with blocked, warning, adjustment, note, position_multiplier
        """
        result = {
            "blocked": False,
            "warning": None,
            "adjustment": 0,
            "note": None,
            "position_multiplier": 1.0,
        }

        try:
            if self._event_calendar is None:
                return result

            conditions = self._event_calendar.check_trading_conditions()

            if conditions is None:
                return result

            # Check if trading day
            if not conditions.get("is_trading_day", True):
                result["blocked"] = True
                result["reason"] = "🚫 Market is closed today (holiday)"
                return result

            # Get risk level
            risk_level = conditions.get("risk_level", "normal")
            pos_adj = conditions.get("position_adjustment", 1.0)
            notes = conditions.get("notes", [])

            result["position_multiplier"] = pos_adj

            if risk_level == "high":
                result["warning"] = "⚠️ High-impact events approaching"
                result["adjustment"] = -10
                result["note"] = "; ".join(notes[:2]) if notes else "Multiple events soon"
            elif risk_level == "elevated":
                result["warning"] = "⚠️ Market event risk elevated"
                result["adjustment"] = -5
                result["note"] = "; ".join(notes[:2]) if notes else "Event approaching"

        except Exception as e:
            logger.debug(f"Event calendar check error: {e}")

        return result

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
