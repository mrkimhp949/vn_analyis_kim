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
    # NEW v9.7: Entry logic constants
    ENTRY_GAP_BLOCK_THRESHOLD,
    ENTRY_GAP_WARN_THRESHOLD,
    AD_STRONG_ACCUMULATION_CMF,
    AD_MODERATE_ACCUMULATION_CMF,
    AD_STRONG_DISTRIBUTION_CMF,
    AD_MODERATE_DISTRIBUTION_CMF,
    AD_OBV_RISING_THRESHOLD,
    AD_OBV_FALLING_THRESHOLD,
    AD_PRICE_DIVERGENCE_THRESHOLD,
    FOREIGN_HEAVY_SELLING_PCT,
    FOREIGN_MODERATE_SELLING_PCT,
    FOREIGN_LIGHT_SELLING_PCT,
    FOREIGN_HEAVY_BUYING_PCT,
    FOREIGN_MODERATE_BUYING_PCT,
    MARGIN_CRITICAL_THRESHOLD,
    MARGIN_LOW_THRESHOLD,
    MARGIN_MODERATE_THRESHOLD,
    MARGIN_HIGH_PENDING_THRESHOLD,
    CONSECUTIVE_LOSS_DEFAULT_LIMIT,
    CONSECUTIVE_LOSS_DEFAULT_COOLDOWN,
    SECTOR_EXPOSURE_BANKING,
    SECTOR_EXPOSURE_REAL_ESTATE,
    SECTOR_EXPOSURE_SECURITIES,
    SECTOR_EXPOSURE_TECHNOLOGY,
    SECTOR_EXPOSURE_CONSUMER,
    SECTOR_EXPOSURE_DEFAULT,
    SECTOR_EXPOSURE_BEAR_MULTIPLIER,
    # NEW v9.8: Entry Logic Improvements
    FILTER_PRIORITY_CRITICAL,
    FILTER_PRIORITY_IMPORTANT,
    FILTER_PRIORITY_OPTIONAL,
    FILTER_WEIGHT_CRITICAL,
    FILTER_WEIGHT_IMPORTANT,
    FILTER_WEIGHT_OPTIONAL,
    ENTRY_QUALITY_EXCELLENT,
    ENTRY_QUALITY_GOOD,
    ENTRY_QUALITY_ACCEPTABLE,
    ENTRY_QUALITY_REJECT,
    INTRADAY_MOMENTUM_ATR_MULTIPLIER,
    INTRADAY_MOMENTUM_MIN_THRESHOLD,
    INTRADAY_MOMENTUM_MAX_THRESHOLD,
    INTRADAY_MOMENTUM_BLOCK_MULTIPLIER,
    GAP_ATR_MULTIPLIER,
    GAP_MIN_BLOCK_THRESHOLD,
    GAP_MAX_BLOCK_THRESHOLD,
    GAP_BREAKOUT_VOLUME_CONFIRM,
    CONSECUTIVE_LOSS_SMALL_THRESHOLD,
    CONSECUTIVE_LOSS_MEDIUM_THRESHOLD,
    CONSECUTIVE_LOSS_LARGE_THRESHOLD,
    CONSECUTIVE_LOSS_SMALL_WEIGHT,
    CONSECUTIVE_LOSS_MEDIUM_WEIGHT,
    CONSECUTIVE_LOSS_LARGE_WEIGHT,
    CONSECUTIVE_LOSS_WEIGHTED_LIMIT,
    FAST_PATH_MIN_CONFIDENCE,
    FAST_PATH_MIN_RR,
    FAST_PATH_SKIP_FILTERS,
    ENTRY_OPTIMAL_MORNING_START,
    ENTRY_OPTIMAL_MORNING_END,
    ENTRY_OPTIMAL_AFTERNOON_START,
    ENTRY_OPTIMAL_AFTERNOON_END,
    ENTRY_AVOID_LUNCH_START,
    ENTRY_AVOID_LUNCH_END,
    ENTRY_TIME_OPTIMAL_BONUS,
    ENTRY_TIME_AVOID_PENALTY,
    # NEW v9.9: Sector Breadth & Earnings Event Constants
    SECTOR_BREADTH_STRONG,
    SECTOR_BREADTH_NEUTRAL,
    SECTOR_BREADTH_WEAK,
    SECTOR_BREADTH_VERY_WEAK,
    SECTOR_BREADTH_BLOCK,
    SECTOR_BREADTH_STRONG_BONUS,
    SECTOR_BREADTH_NEUTRAL_BONUS,
    SECTOR_BREADTH_WEAK_PENALTY,
    SECTOR_BREADTH_VERY_WEAK_PENALTY,
    EARNINGS_BLOCK_DAYS,
    EARNINGS_WARNING_DAYS,
    EARNINGS_CAUTION_DAYS,
    EARNINGS_WARNING_POSITION_MULT,
    EARNINGS_CAUTION_POSITION_MULT,
    EARNINGS_SEASON_POSITION_MULT,
    VN_EARNINGS_MONTHS,
)
from src.config.exceptions import DataQualityError

# Import utilities
from src.monitoring.performance import get_performance_monitor
from src.monitoring.filter_performance import (
    get_filter_performance_tracker,
    FilterPerformanceTracker,
)
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
        use_session_timing_filter: bool = True,  # NEW v9.5: ATO/ATC session timing optimization
        use_pre_holiday_filter: bool = True,  # NEW v9.5: Pre-holiday risk reduction
        use_gap_analysis_filter: bool = True,  # NEW v9.6: Gap opening risk analysis
        use_accumulation_filter: bool = True,  # NEW v9.6: A/D volume analysis
        use_foreign_flow_filter: bool = True,  # NEW v9.6: Foreign investor flow
        use_margin_check: bool = True,  # NEW v9.6: T+2.5 margin availability
        use_consecutive_loss_protection: bool = True,  # NEW v9.6: Block after 3 losses
        intraday_momentum_threshold: float = 0.03,  # NEW: 3% intraday move threshold
        vn30_divergence_threshold: float = 0.04,  # NEW: 4% divergence from VN30 threshold
        pre_holiday_days: int = 3,  # NEW v9.5: Days before major holiday to reduce exposure
        gap_block_threshold: float = 0.05,  # NEW v9.6: Block entry if gap > 5%
        gap_warn_threshold: float = 0.03,  # NEW v9.6: Warn if gap > 3%
        consecutive_loss_limit: int = 3,  # NEW v9.6: Block after N consecutive losses
        consecutive_loss_cooldown_days: int = 5,  # NEW v9.6: Cool-down period in days
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
        self.use_session_timing_filter = use_session_timing_filter  # NEW v9.5
        self.use_pre_holiday_filter = use_pre_holiday_filter  # NEW v9.5
        self.use_gap_analysis_filter = use_gap_analysis_filter  # NEW v9.6
        self.use_accumulation_filter = use_accumulation_filter  # NEW v9.6
        self.use_foreign_flow_filter = use_foreign_flow_filter  # NEW v9.6
        self.use_margin_check = use_margin_check  # NEW v9.6
        self.use_consecutive_loss_protection = use_consecutive_loss_protection  # NEW v9.6
        self.intraday_momentum_threshold = intraday_momentum_threshold
        self.vn30_divergence_threshold = vn30_divergence_threshold
        self.pre_holiday_days = pre_holiday_days  # NEW v9.5
        self.gap_block_threshold = gap_block_threshold  # NEW v9.6
        self.gap_warn_threshold = gap_warn_threshold  # NEW v9.6
        self.consecutive_loss_limit = consecutive_loss_limit  # NEW v9.6
        self.consecutive_loss_cooldown_days = consecutive_loss_cooldown_days  # NEW v9.6

        # NEW v9.6: Track consecutive losses per symbol
        self._consecutive_losses: Dict[str, int] = {}
        self._last_loss_date: Dict[str, datetime] = {}

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

        # NEW v9.7: Integrated filter performance tracker
        self._filter_tracker: FilterPerformanceTracker = get_filter_performance_tracker()

        # NEW v9.8: Entry Quality Score tracking
        self._filter_scores: Dict[str, float] = {}  # Track individual filter scores
        self._last_entry_quality: float = 0.0  # Last calculated entry quality
        self._weighted_losses: Dict[str, float] = {}  # Weighted loss tracking per symbol

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

    def _track_filter(
        self, filter_name: str, passed: bool, symbol: str, is_warning: bool = False
    ) -> None:
        """
        Track filter pass/fail for performance analysis.

        NEW v9.7: Integrated with FilterPerformanceTracker for comprehensive analytics.

        Args:
            filter_name: Name of the filter being tracked
            passed: Whether the filter passed (True) or blocked (False)
            symbol: Stock symbol being checked
            is_warning: If True, filter passed but added a warning
        """
        # Local tracking (backward compatible)
        if passed:
            self._filter_pass_count[filter_name] = self._filter_pass_count.get(filter_name, 0) + 1
        else:
            self._filter_fail_count[filter_name] = self._filter_fail_count.get(filter_name, 0) + 1

        # NEW v9.7: Integrated tracker for comprehensive analytics
        try:
            if not passed:
                self._filter_tracker.record_filter_check(filter_name, "BLOCKED", symbol)
            elif is_warning:
                self._filter_tracker.record_filter_check(filter_name, "WARNING", symbol)
            else:
                self._filter_tracker.record_filter_check(filter_name, "PASSED", symbol)
        except Exception as e:
            logger.debug(f"Filter tracking error: {e}")

    def _safe_run_filter(
        self,
        filter_name: str,
        filter_func: callable,
        symbol: str,
        default_result: Dict = None,
        **kwargs,
    ) -> Dict:
        """
        Safely run a filter with proper error handling.

        NEW v9.7: Wrapper to ensure filters don't crash the entire entry analysis.
        Logs errors properly instead of silent failures.

        Args:
            filter_name: Name of the filter for logging
            filter_func: The filter function to call
            symbol: Stock symbol being checked
            default_result: Default result to return on error
            **kwargs: Arguments to pass to filter_func

        Returns:
            Filter result dict, or default_result on error
        """
        if default_result is None:
            default_result = {
                "blocked": False,
                "warning": None,
                "positive": None,
                "adjustment": 0,
                "note": None,
            }

        try:
            result = filter_func(**kwargs)
            return result if result is not None else default_result

        except Exception as e:
            # Log with appropriate level based on error type
            if isinstance(e, (KeyError, IndexError, ValueError)):
                logger.warning(
                    f"[{symbol}] Filter '{filter_name}' data error: {type(e).__name__}: {e}"
                )
            elif isinstance(e, (ImportError, ModuleNotFoundError)):
                logger.debug(f"[{symbol}] Filter '{filter_name}' dependency not available: {e}")
            else:
                logger.error(
                    f"[{symbol}] Filter '{filter_name}' unexpected error: {type(e).__name__}: {e}",
                    exc_info=True,
                )

            # Track as passed (don't block on error) but log the issue
            self._track_filter(filter_name, True, symbol)
            return default_result

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

        IMPROVED v9.8: Added Entry Quality Score and Fast Path for high-confidence signals.

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

        # NEW v9.8: Check for Fast Path (high confidence signals skip optional filters)
        use_fast_path = self._should_use_fast_path(confidence, df, current_price)
        if use_fast_path:
            telemetry["fast_path"] = True
            logger.debug(f"[{symbol}] Using fast path (confidence: {confidence})")

        # Run all filters (with fast path consideration)
        block_reason = self._run_all_filters(
            df=df,
            current_price=current_price,
            signal_type=signal_type,
            market_regime=market_regime,
            reasons=reasons,
            warnings=warnings,
            adjustments=adjustments,
            adjustment_breakdown=adjustment_breakdown,
            use_fast_path=use_fast_path,  # NEW v9.8
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

        # NEW v9.8: Calculate Entry Quality Score
        entry_quality = self._calculate_entry_quality_score(
            reasons, warnings, adjustments, market_regime
        )
        self._last_entry_quality = entry_quality
        telemetry["entry_quality"] = entry_quality

        # NEW v9.8: Reject if entry quality too low
        if entry_quality < ENTRY_QUALITY_REJECT:
            return create_no_signal(
                f"Entry quality too low: {entry_quality:.2f} < {ENTRY_QUALITY_REJECT}", telemetry
            )

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
        use_fast_path: bool = False,  # NEW v9.8
    ) -> Optional[str]:
        """
        Run all filters and return block reason if any.

        IMPROVED v9.8: Added fast_path support to skip optional filters
        for high-confidence signals.

        Args:
            use_fast_path: If True, skip optional filters (session_timing, gap_analysis, etc.)
        """
        symbol = self._current_symbol

        # NEW v9.8: Reset filter scores for this analysis
        self._filter_scores = {}

        # 1. Vietnam Price Limit Check (Blocking)
        price_limit = self._technical_checker.check_vietnam_price_limits(df, current_price, symbol or "")
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

            # NEW v9.5: Check sector exposure limit (dynamic based on regime)
            sector_exposure_check = self._check_sector_exposure(
                symbol, sector.get("sector_id", "DEFAULT"), market_regime
            )
            if sector_exposure_check.get("blocked"):
                self._track_filter("sector_exposure", False, symbol or "")
                return sector_exposure_check.get("reason")
            if sector_exposure_check.get("warning"):
                warnings.append(sector_exposure_check["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "sector_exposure",
                    sector_exposure_check.get("adjustment", -10),
                    sector_exposure_check.get("note", "High sector exposure"),
                )

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

        # 19. Session Timing Filter (NEW v9.5) - ATO/ATC session optimization
        # NEW v9.8: Skip in fast path (optional filter)
        if self.use_session_timing_filter and SESSION_TRADING_AVAILABLE:
            if use_fast_path and "session_timing" in FAST_PATH_SKIP_FILTERS:
                self._filter_scores["session_timing"] = 0.5  # Neutral score
                logger.debug(f"[{symbol}] Fast path: skipping session_timing filter")
            else:
                session_result = self._check_session_timing()
                if session_result.get("blocked"):
                    self._track_filter("session_timing", False, symbol or "")
                    return session_result.get("reason")
                if session_result.get("warning"):
                    warnings.append(session_result["warning"])
                    self._add_adjustment(
                        adjustments,
                        adjustment_breakdown,
                        "session_timing",
                        session_result.get("adjustment", -10),
                        session_result.get("note", "Non-optimal entry timing"),
                    )
                    self._filter_scores["session_timing"] = 0.3
                elif session_result.get("positive"):
                    reasons.append(session_result["positive"])
                    self._add_adjustment(
                        adjustments,
                        adjustment_breakdown,
                        "session_timing",
                        session_result.get("adjustment", 5),
                        session_result.get("note", "Optimal entry timing"),
                    )
                    self._filter_scores["session_timing"] = 1.0
                else:
                    self._filter_scores["session_timing"] = 0.5
                # Apply position multiplier from session timing
                if session_result.get("position_multiplier", 1.0) != 1.0:
                    self._regime_position_multiplier *= session_result.get(
                        "position_multiplier", 1.0
                    )
                self._track_filter("session_timing", True, symbol or "")

        # 20. Pre-Holiday Risk Reduction (NEW v9.5) - Reduce exposure before major holidays
        # NEW v9.8: Skip in fast path (optional filter)
        if self.use_pre_holiday_filter:
            if use_fast_path and "pre_holiday" in FAST_PATH_SKIP_FILTERS:
                self._filter_scores["pre_holiday"] = 0.5
                logger.debug(f"[{symbol}] Fast path: skipping pre_holiday filter")
            else:
                holiday_result = self._check_pre_holiday_risk()
                if holiday_result.get("blocked"):
                    self._track_filter("pre_holiday", False, symbol or "")
                    return holiday_result.get("reason")
                if holiday_result.get("warning"):
                    warnings.append(holiday_result["warning"])
                    self._add_adjustment(
                        adjustments,
                        adjustment_breakdown,
                        "pre_holiday",
                        holiday_result.get("adjustment", -15),
                        holiday_result.get("note", "Pre-holiday risk reduction"),
                    )
                    self._filter_scores["pre_holiday"] = 0.3
                else:
                    self._filter_scores["pre_holiday"] = 1.0
                # Apply position multiplier for pre-holiday
                if holiday_result.get("position_multiplier", 1.0) != 1.0:
                    self._regime_position_multiplier *= holiday_result.get(
                        "position_multiplier", 1.0
                    )
                self._track_filter("pre_holiday", True, symbol or "")

        # 21. Gap Analysis Filter (NEW v9.6) - Avoid buying into large gap up
        # NEW v9.8: Skip in fast path (optional filter)
        if self.use_gap_analysis_filter:
            if use_fast_path and "gap_analysis" in FAST_PATH_SKIP_FILTERS:
                self._filter_scores["gap_analysis"] = 0.5
                logger.debug(f"[{symbol}] Fast path: skipping gap_analysis filter")
            else:
                gap_result = self._check_gap_risk(df, current_price)
                if gap_result.get("blocked"):
                    self._track_filter("gap_analysis", False, symbol or "")
                    return gap_result.get("reason")
                if gap_result.get("warning"):
                    warnings.append(gap_result["warning"])
                    self._add_adjustment(
                        adjustments,
                        adjustment_breakdown,
                        "gap_analysis",
                        gap_result.get("adjustment", -10),
                        gap_result.get("note", "Gap risk"),
                    )
                    self._filter_scores["gap_analysis"] = 0.3
                elif gap_result.get("positive"):
                    reasons.append(gap_result["positive"])
                    self._add_adjustment(
                        adjustments,
                        adjustment_breakdown,
                        "gap_analysis",
                        gap_result.get("adjustment", 5),
                        gap_result.get("note", "Gap fill opportunity"),
                    )
                    self._filter_scores["gap_analysis"] = 1.0
                else:
                    self._filter_scores["gap_analysis"] = 0.5
                self._track_filter("gap_analysis", True, symbol or "")

        # 22. Accumulation/Distribution Filter (NEW v9.6) - Smart money flow
        if self.use_accumulation_filter:
            ad_result = self._check_accumulation_distribution(df)
            if ad_result.get("blocked"):
                self._track_filter("accumulation", False, symbol or "")
                return ad_result.get("reason")
            if ad_result.get("warning"):
                warnings.append(ad_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "accumulation",
                    ad_result.get("adjustment", -10),
                    ad_result.get("note", "Distribution detected"),
                )
            elif ad_result.get("positive"):
                reasons.append(ad_result["positive"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "accumulation",
                    ad_result.get("adjustment", 10),
                    ad_result.get("note", "Accumulation detected"),
                )
            self._track_filter("accumulation", True, symbol or "")

        # 23. Foreign Flow Filter (NEW v9.6) - Foreign investor sentiment
        if self.use_foreign_flow_filter and symbol:
            foreign_result = self._check_foreign_flow(symbol, df)
            if foreign_result.get("blocked"):
                self._track_filter("foreign_flow", False, symbol or "")
                return foreign_result.get("reason")
            if foreign_result.get("warning"):
                warnings.append(foreign_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "foreign_flow",
                    foreign_result.get("adjustment", -15),
                    foreign_result.get("note", "Foreign selling pressure"),
                )
            elif foreign_result.get("positive"):
                reasons.append(foreign_result["positive"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "foreign_flow",
                    foreign_result.get("adjustment", 10),
                    foreign_result.get("note", "Foreign buying support"),
                )
            # Apply position multiplier from foreign flow
            if foreign_result.get("position_multiplier", 1.0) != 1.0:
                self._regime_position_multiplier *= foreign_result.get("position_multiplier", 1.0)
            self._track_filter("foreign_flow", True, symbol or "")

        # 24. T+2.5 Margin Check (NEW v9.6) - Settlement awareness
        if self.use_margin_check:
            margin_result = self._check_margin_availability()
            if margin_result.get("blocked"):
                self._track_filter("margin_check", False, symbol or "")
                return margin_result.get("reason")
            if margin_result.get("warning"):
                warnings.append(margin_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "margin_check",
                    margin_result.get("adjustment", -10),
                    margin_result.get("note", "Margin constraint"),
                )
            # Apply position multiplier from margin
            if margin_result.get("position_multiplier", 1.0) != 1.0:
                self._regime_position_multiplier *= margin_result.get("position_multiplier", 1.0)
            self._track_filter("margin_check", True, symbol or "")

        # 25. Consecutive Loss Protection (NEW v9.6) - Block after repeated losses
        if self.use_consecutive_loss_protection and symbol:
            loss_result = self._check_consecutive_losses(symbol)
            if loss_result.get("blocked"):
                self._track_filter("consecutive_loss", False, symbol or "")
                return loss_result.get("reason")
            if loss_result.get("warning"):
                warnings.append(loss_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "consecutive_loss",
                    loss_result.get("adjustment", -15),
                    loss_result.get("note", "Recent losses on symbol"),
                )
            self._track_filter("consecutive_loss", True, symbol or "")

        # 26. Sector Breadth Indicator (NEW v9.9) - % stocks bullish in sector
        if self.use_sector_strength_filter and symbol:
            sector_breadth_result = self._check_sector_breadth(symbol)
            if sector_breadth_result.get("blocked"):
                self._track_filter("sector_breadth", False, symbol or "")
                return sector_breadth_result.get("reason")
            if sector_breadth_result.get("warning"):
                warnings.append(sector_breadth_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "sector_breadth",
                    sector_breadth_result.get("adjustment", -10),
                    sector_breadth_result.get("note", "Weak sector breadth"),
                )
            elif sector_breadth_result.get("positive"):
                reasons.append(sector_breadth_result["positive"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "sector_breadth",
                    sector_breadth_result.get("adjustment", 10),
                    sector_breadth_result.get("note", "Strong sector breadth"),
                )
            self._track_filter("sector_breadth", True, symbol or "")

        # 27. Earnings/Event Block (NEW v9.9) - Block entry before earnings
        if FUNDAMENTAL_AVAILABLE and symbol:
            earnings_result = self._check_earnings_event_block(symbol)
            if earnings_result.get("blocked"):
                self._track_filter("earnings_block", False, symbol or "")
                return earnings_result.get("reason")
            if earnings_result.get("warning"):
                warnings.append(earnings_result["warning"])
                self._add_adjustment(
                    adjustments,
                    adjustment_breakdown,
                    "earnings_block",
                    earnings_result.get("adjustment", -15),
                    earnings_result.get("note", "Earnings approaching"),
                )
            # Apply position multiplier for earnings
            if earnings_result.get("position_multiplier", 1.0) != 1.0:
                self._regime_position_multiplier *= earnings_result.get("position_multiplier", 1.0)
            self._track_filter("earnings_block", True, symbol or "")

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

        IMPROVED v9.8: Dynamic ATR-based threshold instead of fixed 3%.
        Uses 1.5x ATR as threshold, capped between 2.5% and 6%.

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

            # NEW v9.8: Dynamic threshold based on ATR
            dynamic_threshold = self._get_dynamic_momentum_threshold(df, current_price)

            # Check if already moved significantly
            if abs(intraday_change) >= dynamic_threshold:
                if intraday_change > 0:
                    # Price already up - risky to chase
                    block_threshold = dynamic_threshold * INTRADAY_MOMENTUM_BLOCK_MULTIPLIER
                    if intraday_change >= block_threshold:
                        result["blocked"] = True
                        result["reason"] = (
                            f"Price already up {intraday_change*100:.1f}% today "
                            f"(threshold: {block_threshold*100:.1f}%) - too extended"
                        )
                    else:
                        result["warning"] = f"⚠️ Intraday up {intraday_change*100:.1f}% - extended"
                        result["adjustment"] = -10
                        result["note"] = f"Chasing momentum ({intraday_change*100:.1f}%)"
                else:
                    # Price down - might be catching falling knife
                    block_threshold = -dynamic_threshold * INTRADAY_MOMENTUM_BLOCK_MULTIPLIER
                    if intraday_change <= block_threshold:
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

    def _get_dynamic_momentum_threshold(self, df: pd.DataFrame, current_price: float) -> float:
        """
        Calculate dynamic intraday momentum threshold based on ATR.

        NEW v9.8: Instead of fixed 3%, use 1.5x ATR capped between 2.5% and 6%.
        This adapts to stock volatility - volatile stocks get wider threshold.

        Args:
            df: DataFrame with OHLCV data
            current_price: Current price

        Returns:
            Dynamic threshold as decimal (e.g., 0.035 for 3.5%)
        """
        try:
            # Try to get ATR from dataframe
            if "atr" in df.columns:
                atr = safe_get_latest(df, "atr", 0)
            elif "atr_14" in df.columns:
                atr = safe_get_latest(df, "atr_14", 0)
            else:
                # Calculate ATR manually if not available
                if all(col in df.columns for col in ["high", "low", "close"]):
                    high = df["high"].tail(14)
                    low = df["low"].tail(14)
                    close = df["close"].tail(14)
                    tr = pd.concat(
                        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
                        axis=1,
                    ).max(axis=1)
                    atr = tr.mean()
                else:
                    return self.intraday_momentum_threshold  # Fallback to fixed

            if atr <= 0 or current_price <= 0:
                return self.intraday_momentum_threshold

            # Calculate ATR as percentage of price
            atr_pct = atr / current_price

            # Dynamic threshold = 1.5x ATR, capped between min and max
            dynamic_threshold = atr_pct * INTRADAY_MOMENTUM_ATR_MULTIPLIER
            dynamic_threshold = max(INTRADAY_MOMENTUM_MIN_THRESHOLD, dynamic_threshold)
            dynamic_threshold = min(INTRADAY_MOMENTUM_MAX_THRESHOLD, dynamic_threshold)

            return dynamic_threshold

        except Exception as e:
            logger.debug(f"Dynamic momentum threshold error: {e}")
            return self.intraday_momentum_threshold  # Fallback

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

    def _should_use_fast_path(
        self, confidence: float, df: pd.DataFrame, current_price: float
    ) -> bool:
        """
        Determine if fast path should be used for high-confidence signals.

        NEW v9.8: Fast path skips optional filters when:
        - Confidence >= 80%
        - Estimated R:R >= 2.5

        This reduces over-filtering for strong signals.

        Args:
            confidence: Initial confidence score
            df: DataFrame with OHLCV data
            current_price: Current price

        Returns:
            True if fast path should be used
        """
        try:
            # Check confidence threshold
            if confidence < FAST_PATH_MIN_CONFIDENCE:
                return False

            # Estimate R:R (quick check without full calculation)
            sr_check = self._technical_checker.check_support_resistance(df, current_price)
            if sr_check.get("support_level"):
                support = sr_check["support_level"]
                risk = current_price - support
                if risk > 0:
                    # Estimate reward as 2x risk (conservative)
                    estimated_rr = 2.0
                    # If bouncing from support, likely better R:R
                    if sr_check.get("bouncing_from_support"):
                        estimated_rr = 2.5
                    if estimated_rr >= FAST_PATH_MIN_RR:
                        return True

            return False

        except Exception as e:
            logger.debug(f"Fast path check error: {e}")
            return False

    def _calculate_entry_quality_score(
        self,
        reasons: List[str],
        warnings: List[str],
        adjustments: List[int],
        market_regime: Optional[Dict],
    ) -> float:
        """
        Calculate Entry Quality Score based on filter results.

        NEW v9.8: Weighted scoring system instead of binary pass/fail.
        - Critical filters: 2x weight
        - Important filters: 1.5x weight
        - Optional filters: 1x weight

        Score ranges:
        - >= 0.85: Excellent entry
        - >= 0.70: Good entry
        - >= 0.55: Acceptable entry
        - >= 0.40: Poor entry (reduce size)
        - < 0.40: Reject entry

        Args:
            reasons: List of positive reasons
            warnings: List of warnings
            adjustments: List of confidence adjustments
            market_regime: Market regime info

        Returns:
            Entry quality score (0.0 - 1.0)
        """
        try:
            # Start with base score from filter_scores
            if not self._filter_scores:
                # Fallback: estimate from reasons/warnings
                positive_count = len(reasons)
                warning_count = len(warnings)
                total_checks = positive_count + warning_count + 5  # Assume 5 neutral
                base_score = (positive_count + 2.5) / total_checks  # Neutral = 0.5
            else:
                # Calculate weighted average from filter scores
                total_weight = 0.0
                weighted_sum = 0.0

                for filter_name, score in self._filter_scores.items():
                    # Determine filter weight
                    if filter_name in FILTER_PRIORITY_CRITICAL:
                        weight = FILTER_WEIGHT_CRITICAL
                    elif filter_name in FILTER_PRIORITY_IMPORTANT:
                        weight = FILTER_WEIGHT_IMPORTANT
                    else:
                        weight = FILTER_WEIGHT_OPTIONAL

                    weighted_sum += score * weight
                    total_weight += weight

                base_score = weighted_sum / total_weight if total_weight > 0 else 0.5

            # Adjust for warnings
            warning_penalty = len(warnings) * 0.03  # 3% penalty per warning
            base_score -= warning_penalty

            # Adjust for positive adjustments
            positive_adjustments = sum(a for a in adjustments if a > 0)
            negative_adjustments = sum(a for a in adjustments if a < 0)
            adjustment_factor = (positive_adjustments + negative_adjustments) / 100
            base_score += adjustment_factor * 0.1  # 10% of adjustment impact

            # Regime adjustment
            if market_regime:
                regime = market_regime.get("regime", "SIDEWAYS")
                if regime == "BULL":
                    base_score += 0.05  # Bonus in bull market
                elif regime == "BEAR":
                    base_score -= 0.05  # Penalty in bear market

            # Clamp to 0-1 range
            return max(0.0, min(1.0, base_score))

        except Exception as e:
            logger.debug(f"Entry quality score error: {e}")
            return 0.5  # Default to neutral

    def get_entry_quality_label(self, score: float) -> str:
        """
        Get human-readable label for entry quality score.

        Args:
            score: Entry quality score (0.0 - 1.0)

        Returns:
            Label string (EXCELLENT, GOOD, ACCEPTABLE, POOR, REJECT)
        """
        if score >= ENTRY_QUALITY_EXCELLENT:
            return "EXCELLENT"
        elif score >= ENTRY_QUALITY_GOOD:
            return "GOOD"
        elif score >= ENTRY_QUALITY_ACCEPTABLE:
            return "ACCEPTABLE"
        elif score >= ENTRY_QUALITY_REJECT:
            return "POOR"
        else:
            return "REJECT"

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

    def _check_session_timing(self) -> Dict:
        """
        Check session timing for optimal entry.

        IMPROVED v9.8: Enhanced time-of-day optimization with constants.

        Optimal Entry Windows (from constants):
        - 9:30-10:30: After opening volatility settles (BEST)
        - 13:30-14:15: After lunch gap settles, before ATC (GOOD)

        Avoid Entry Windows:
        - 9:00-9:15: ATO auction (high volatility) - BLOCK
        - 11:00-11:30: Pre-lunch selling pressure - WARNING
        - 14:30-14:45: ATC auction (high volatility) - BLOCK

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
            # NEW v9.8: Fallback to simple time check if session manager not available
            if not SESSION_TRADING_AVAILABLE:
                return self._check_time_of_day_simple()

            session_manager = get_session_manager()
            session_info = session_manager.get_current_session()

            if session_info is None:
                return self._check_time_of_day_simple()

            session_type = session_info.session_type
            entry_quality = session_info.entry_quality
            time_remaining = session_info.time_remaining

            # Block during ATO auction (high volatility)
            if session_type.value == "ATO":
                result["blocked"] = True
                result["reason"] = (
                    f"🚫 ATO auction in progress ({time_remaining} min remaining) - "
                    "High volatility, avoid entries"
                )
                return result

            # Block during ATC auction (high volatility)
            if session_type.value == "ATC":
                result["blocked"] = True
                result["reason"] = (
                    f"🚫 ATC auction in progress ({time_remaining} min remaining) - "
                    "High volatility, avoid entries"
                )
                return result

            # Block during lunch break
            if session_type.value == "LUNCH":
                result["blocked"] = True
                result["reason"] = "🚫 Lunch break - Market closed"
                return result

            # Block during closed hours
            if session_type.value in ("CLOSED", "PRE_OPEN", "POST_CLOSE"):
                result["blocked"] = True
                result["reason"] = f"🚫 Market {session_type.value} - Not trading hours"
                return result

            # Warning during pre-lunch period (11:00-11:30)
            if entry_quality == "AVOID":
                result["warning"] = "⚠️ Pre-lunch period - selling pressure likely"
                result["adjustment"] = ENTRY_TIME_AVOID_PENALTY
                result["note"] = "Pre-lunch selling pressure"
                result["position_multiplier"] = 0.7
                return result

            # Optimal entry window
            if entry_quality == "OPTIMAL":
                result["positive"] = f"✅ Optimal entry timing ({session_type.value})"
                result["adjustment"] = ENTRY_TIME_OPTIMAL_BONUS
                result["note"] = "Best entry window"
                result["position_multiplier"] = 1.1
                return result

            # Acceptable window
            if entry_quality == "ACCEPTABLE":
                result["adjustment"] = 0
                result["note"] = "Acceptable entry timing"
                result["position_multiplier"] = 1.0

            # Add session-specific recommendations
            for rec in session_info.recommendations:
                logger.debug(f"Session recommendation: {rec}")

        except Exception as e:
            logger.debug(f"Session timing check error: {e}")

        return result

    def _check_time_of_day_simple(self) -> Dict:
        """
        Simple time-of-day check when session manager not available.

        NEW v9.8: Fallback time optimization using constants.

        Returns:
            Dict with warning, positive, adjustment, note, position_multiplier
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
            now = datetime.now()
            current_hour = now.hour
            current_minute = now.minute
            current_time = (current_hour, current_minute)

            # Check optimal morning window (9:30-10:30)
            morning_start = ENTRY_OPTIMAL_MORNING_START
            morning_end = ENTRY_OPTIMAL_MORNING_END
            if morning_start <= current_time <= morning_end:
                result["positive"] = "✅ Optimal morning entry window"
                result["adjustment"] = ENTRY_TIME_OPTIMAL_BONUS
                result["note"] = "Best morning entry time"
                result["position_multiplier"] = 1.1
                return result

            # Check optimal afternoon window (13:30-14:15)
            afternoon_start = ENTRY_OPTIMAL_AFTERNOON_START
            afternoon_end = ENTRY_OPTIMAL_AFTERNOON_END
            if afternoon_start <= current_time <= afternoon_end:
                result["positive"] = "✅ Optimal afternoon entry window"
                result["adjustment"] = ENTRY_TIME_OPTIMAL_BONUS
                result["note"] = "Best afternoon entry time"
                result["position_multiplier"] = 1.1
                return result

            # Check avoid window (11:00-11:30 pre-lunch)
            avoid_start = ENTRY_AVOID_LUNCH_START
            avoid_end = ENTRY_AVOID_LUNCH_END
            if avoid_start <= current_time <= avoid_end:
                result["warning"] = "⚠️ Pre-lunch period - avoid new entries"
                result["adjustment"] = ENTRY_TIME_AVOID_PENALTY
                result["note"] = "Pre-lunch selling pressure"
                result["position_multiplier"] = 0.7
                return result

        except Exception as e:
            logger.debug(f"Simple time check error: {e}")

        return result

    def _check_pre_holiday_risk(self) -> Dict:
        """
        Check pre-holiday risk and reduce exposure accordingly.

        NEW v9.5: Reduce exposure 2-3 days before major Vietnamese holidays.

        Major holidays requiring reduced exposure:
        - Tết Nguyên Đán (Vietnamese New Year): BLOCK entries 5 days before
        - Giỗ Tổ Hùng Vương: WARNING 2 days before
        - 30/4 & 1/5 (Reunification + Labor Day): WARNING 2 days before
        - National Day (2/9): WARNING 2 days before

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
            from datetime import datetime, timedelta

            today = datetime.now().date()

            # Try to import complete holiday calendar
            try:
                from src.utils.vietnam_holidays import (
                    days_until_tet,
                    get_upcoming_holidays,
                    is_pre_holiday_trading_day,
                )

                # Check days until Tết (most important)
                tet_days = days_until_tet()
                if tet_days is not None and 0 < tet_days <= 5:
                    if tet_days <= 2:
                        result["blocked"] = True
                        result["reason"] = (
                            f"🚫 Tết approaching in {tet_days} days - "
                            "No new entries (market will close for 7+ days)"
                        )
                        return result
                    else:
                        result["warning"] = f"⚠️ Tết approaching in {tet_days} days"
                        result["adjustment"] = -20
                        result["note"] = "Pre-Tết risk reduction"
                        result["position_multiplier"] = 0.5  # Reduce position by 50%
                        return result

                # Check upcoming holidays (next 3 days)
                upcoming = get_upcoming_holidays(days_ahead=self.pre_holiday_days)
                if upcoming:
                    next_holiday = upcoming[0]
                    days_to_holiday = (next_holiday["date"] - today).days

                    if days_to_holiday <= 1:
                        result["blocked"] = True
                        result["reason"] = (
                            f"🚫 Holiday tomorrow: {next_holiday['name']} - "
                            "No new entries before market closure"
                        )
                        return result
                    elif days_to_holiday <= self.pre_holiday_days:
                        result["warning"] = f"⚠️ {next_holiday['name']} in {days_to_holiday} days"
                        result["adjustment"] = -10
                        result["note"] = f"Pre-holiday: {next_holiday['name']}"
                        result["position_multiplier"] = 0.7

                # Check if today is pre-holiday trading day
                if is_pre_holiday_trading_day():
                    if result["warning"] is None:
                        result["warning"] = "⚠️ Pre-holiday trading day - reduced exposure"
                        result["adjustment"] = -5
                        result["note"] = "Pre-holiday trading"
                        result["position_multiplier"] = 0.8

            except ImportError:
                # Fallback: Check basic holidays from constants
                try:
                    from src.config.constants import VIETNAM_TET_HOLIDAYS

                    current_year = today.year
                    if current_year in VIETNAM_TET_HOLIDAYS:
                        tet_dates = VIETNAM_TET_HOLIDAYS[current_year]
                        if tet_dates:
                            first_tet = datetime(
                                current_year, tet_dates[0][0], tet_dates[0][1]
                            ).date()
                            days_to_tet = (first_tet - today).days
                            if 0 < days_to_tet <= 5:
                                result["warning"] = f"⚠️ Tết approaching in {days_to_tet} days"
                                result["adjustment"] = -15
                                result["note"] = "Pre-Tết risk reduction"
                                result["position_multiplier"] = 0.6
                except ImportError:
                    pass

        except Exception as e:
            logger.debug(f"Pre-holiday risk check error: {e}")

        return result

    def _check_sector_exposure(
        self, symbol: str, sector_id: str, market_regime: Optional[Dict]
    ) -> Dict:
        """
        Check sector exposure limits with dynamic adjustment for BEAR market.

        NEW v9.5: Reduces max sector exposure in BEAR market.

        In BEAR market:
        - Banking: 25% → 15% (high correlation with market drops)
        - Real Estate: 20% → 10% (very volatile in downturns)
        - Securities: 15% → 5% (extremely sensitive)
        - Technology: 30% → 20% (more resilient)
        - Consumer: 30% → 25% (defensive, holds better)

        Args:
            symbol: Stock symbol
            sector_id: Sector identifier (e.g., "BANKING", "REAL_ESTATE")
            market_regime: Market regime info

        Returns:
            Dict with blocked, warning, adjustment, note
        """
        result = {
            "blocked": False,
            "warning": None,
            "adjustment": 0,
            "note": None,
        }

        try:
            # Get current regime
            regime = "SIDEWAYS"
            if market_regime:
                regime = market_regime.get("regime", "SIDEWAYS")

            # Try to get dynamic sector exposure from circuit breaker
            try:
                from src.risk.circuit_breaker import get_sector_max_exposure

                max_exposure = get_sector_max_exposure(sector_id, regime)
            except ImportError:
                # Fallback to centralized default values
                default_exposures = {
                    "BANKING": SECTOR_EXPOSURE_BANKING,
                    "REAL_ESTATE": SECTOR_EXPOSURE_REAL_ESTATE,
                    "SECURITIES": SECTOR_EXPOSURE_SECURITIES,
                    "TECHNOLOGY": SECTOR_EXPOSURE_TECHNOLOGY,
                    "CONSUMER": SECTOR_EXPOSURE_CONSUMER,
                    "DEFAULT": SECTOR_EXPOSURE_DEFAULT,
                }
                max_exposure = default_exposures.get(sector_id, SECTOR_EXPOSURE_DEFAULT)

                # Reduce in BEAR/HIGH_VOL using centralized multiplier
                if regime in ("BEAR", "HIGH_VOLATILITY"):
                    max_exposure *= SECTOR_EXPOSURE_BEAR_MULTIPLIER

            # Get current sector exposure from portfolio manager
            current_exposure = 0.0
            if self.portfolio_manager is not None:
                try:
                    positions = self.portfolio_manager.get_positions()
                    total_value = sum(p.get("value", 0) for p in positions.values())

                    # Count positions in this sector
                    sector_value = 0.0
                    for sym, pos in positions.items():
                        pos_sector = self._get_symbol_sector(sym)
                        if pos_sector == sector_id:
                            sector_value += pos.get("value", 0)

                    if total_value > 0:
                        current_exposure = sector_value / total_value
                except Exception as e:
                    logger.debug(f"Error getting sector exposure: {e}")

            # Check if at or over limit
            if current_exposure >= max_exposure:
                result["blocked"] = True
                result["reason"] = (
                    f"🚫 Sector exposure limit: {sector_id} at {current_exposure*100:.1f}% "
                    f"(max: {max_exposure*100:.1f}% in {regime} market)"
                )
                return result

            # Warning if approaching limit (>80% of max)
            if current_exposure >= max_exposure * 0.8:
                result["warning"] = (
                    f"⚠️ High {sector_id} exposure: {current_exposure*100:.1f}% "
                    f"(limit: {max_exposure*100:.1f}% in {regime})"
                )
                result["adjustment"] = -10
                result["note"] = f"Near sector limit ({sector_id})"

            # Extra warning in BEAR market for risky sectors
            if regime == "BEAR" and sector_id in ("BANKING", "REAL_ESTATE", "SECURITIES"):
                if result["warning"] is None:
                    result["warning"] = (
                        f"⚠️ {sector_id} is high-risk in BEAR market "
                        f"(reduced limit: {max_exposure*100:.0f}%)"
                    )
                    result["adjustment"] = -5
                    result["note"] = f"Risky sector in BEAR ({sector_id})"

        except Exception as e:
            logger.debug(f"Sector exposure check error: {e}")

        return result

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol. Returns 'DEFAULT' if not found."""
        try:
            from src.utils.vietnam_market import VN30_SECTORS

            return VN30_SECTORS.get(symbol.upper(), "DEFAULT")
        except ImportError:
            return "DEFAULT"

    def _check_gap_risk(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Check overnight gap risk to avoid buying into large gap openings.

        IMPROVED v9.8: Dynamic ATR-based thresholds + breakout gap detection.
        - Uses 2x ATR as gap threshold (capped 4-8%)
        - Allows breakout gaps with 2x volume confirmation
        - Better handles volatile vs stable stocks

        Args:
            df: DataFrame with OHLCV data
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
            if len(df) < 2:
                return result

            if "open" not in df.columns or "close" not in df.columns:
                return result

            # Get today's open and previous close
            today_open = safe_get_latest(df, "open", 0)
            prev_close = df["close"].iloc[-2] if len(df) >= 2 else 0

            if today_open <= 0 or prev_close <= 0:
                return result

            # Calculate gap percentage
            gap_pct = (today_open - prev_close) / prev_close

            # NEW v9.8: Dynamic threshold based on ATR
            dynamic_block_threshold = self._get_dynamic_gap_threshold(df, current_price)
            dynamic_warn_threshold = dynamic_block_threshold * 0.6  # Warn at 60% of block

            # NEW v9.8: Check for breakout gap with volume confirmation
            is_breakout_gap = self._is_breakout_gap(df, gap_pct)

            # Large gap up
            if gap_pct >= dynamic_block_threshold:
                if is_breakout_gap:
                    # Breakout gap with volume - allow with warning
                    result["warning"] = (
                        f"⚠️ Breakout gap +{gap_pct*100:.1f}% with volume - "
                        "valid but use tight stop"
                    )
                    result["adjustment"] = -5
                    result["note"] = f"Breakout gap {gap_pct*100:.1f}%"
                else:
                    result["blocked"] = True
                    result["reason"] = (
                        f"🚫 Large gap up: +{gap_pct*100:.1f}% "
                        f"(threshold: {dynamic_block_threshold*100:.1f}%) - "
                        "Don't chase, wait for pullback"
                    )
                return result

            # Moderate gap up - warning
            if gap_pct >= dynamic_warn_threshold:
                if is_breakout_gap:
                    result["positive"] = f"✅ Breakout gap +{gap_pct*100:.1f}% with volume"
                    result["adjustment"] = 5
                    result["note"] = "Volume-confirmed breakout"
                else:
                    result["warning"] = f"⚠️ Gap up +{gap_pct*100:.1f}% - risk of retracement"
                    result["adjustment"] = -10
                    result["note"] = f"Gap up {gap_pct*100:.1f}%"
                return result

            # Large gap down - potential falling knife
            if gap_pct <= -dynamic_block_threshold:
                result["blocked"] = True
                result["reason"] = (
                    f"🚫 Large gap down: {gap_pct*100:.1f}% - "
                    "Potential falling knife, wait for stabilization"
                )
                return result

            # Moderate gap down - cautious
            if gap_pct <= -dynamic_warn_threshold:
                result["warning"] = f"⚠️ Gap down {gap_pct*100:.1f}% - catching knife risk"
                result["adjustment"] = -8
                result["note"] = f"Gap down {gap_pct*100:.1f}%"
                return result

            # Small gap down with recovery - potential gap fill opportunity
            from src.config.constants import ENTRY_GAP_FILL_MIN, ENTRY_GAP_FILL_MAX

            if ENTRY_GAP_FILL_MIN <= gap_pct < ENTRY_GAP_FILL_MAX:
                gap_fill_potential = (prev_close - current_price) / prev_close
                if gap_fill_potential > 0.01:
                    result["positive"] = f"✅ Gap fill opportunity ({gap_pct*100:.1f}% gap)"
                    result["adjustment"] = 5
                    result["note"] = "Gap fill potential"

        except Exception as e:
            logger.debug(f"Gap risk check error: {e}")

        return result

    def _get_dynamic_gap_threshold(self, df: pd.DataFrame, current_price: float) -> float:
        """
        Calculate dynamic gap threshold based on ATR.

        NEW v9.8: Uses 2x ATR as threshold, capped between 4% and 8%.

        Args:
            df: DataFrame with OHLCV data
            current_price: Current price

        Returns:
            Dynamic threshold as decimal
        """
        try:
            # Try to get ATR
            if "atr" in df.columns:
                atr = safe_get_latest(df, "atr", 0)
            elif "atr_14" in df.columns:
                atr = safe_get_latest(df, "atr_14", 0)
            else:
                return ENTRY_GAP_BLOCK_THRESHOLD  # Fallback to fixed 5%

            if atr <= 0 or current_price <= 0:
                return ENTRY_GAP_BLOCK_THRESHOLD

            # Calculate ATR as percentage
            atr_pct = atr / current_price

            # Dynamic threshold = 2x ATR, capped
            dynamic_threshold = atr_pct * GAP_ATR_MULTIPLIER
            dynamic_threshold = max(GAP_MIN_BLOCK_THRESHOLD, dynamic_threshold)
            dynamic_threshold = min(GAP_MAX_BLOCK_THRESHOLD, dynamic_threshold)

            return dynamic_threshold

        except Exception:
            return ENTRY_GAP_BLOCK_THRESHOLD

    def _is_breakout_gap(self, df: pd.DataFrame, gap_pct: float) -> bool:
        """
        Check if gap is a valid breakout gap with volume confirmation.

        NEW v9.8: Breakout gaps (with 2x volume) are valid entry opportunities.

        Args:
            df: DataFrame with OHLCV data
            gap_pct: Gap percentage

        Returns:
            True if breakout gap with volume confirmation
        """
        try:
            if gap_pct <= 0:  # Only for gap ups
                return False

            if "volume" not in df.columns or len(df) < 20:
                return False

            # Get current and average volume
            current_volume = safe_get_latest(df, "volume", 0)
            avg_volume = df["volume"].tail(20).mean()

            if avg_volume <= 0:
                return False

            # Check volume confirmation (2x average)
            volume_ratio = current_volume / avg_volume
            if volume_ratio >= GAP_BREAKOUT_VOLUME_CONFIRM:
                # Also check if breaking above recent high
                if "high" in df.columns:
                    recent_high = df["high"].tail(20).max()
                    current_price = safe_get_latest(df, "close", 0)
                    if current_price > recent_high:
                        return True

            return False

        except Exception:
            return False

    def _check_accumulation_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Check accumulation/distribution pattern to detect smart money flow.

        NEW v9.6: Analyze volume patterns to identify institutional activity.

        Indicators used:
        - OBV (On Balance Volume) trend
        - CMF (Chaikin Money Flow)
        - Volume pattern analysis

        Args:
            df: DataFrame with OHLCV data

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
            required_cols = ["close", "high", "low", "volume"]
            if not all(col in df.columns for col in required_cols):
                return result

            if len(df) < 20:
                return result

            # Calculate OBV (On Balance Volume)
            obv = self._calculate_obv(df)
            if obv is None:
                return result

            # OBV trend analysis (5-period vs 20-period)
            obv_short = obv.tail(5).mean()
            obv_long = obv.tail(20).mean()

            # Calculate CMF (Chaikin Money Flow)
            cmf = self._calculate_cmf(df, period=20)

            # Price trend
            price_trend = (df["close"].iloc[-1] - df["close"].iloc[-10]) / df["close"].iloc[-10]

            # Analyze accumulation/distribution using centralized constants
            if cmf is not None:
                # Strong accumulation: CMF > 0.10 and OBV rising
                if (
                    cmf > AD_STRONG_ACCUMULATION_CMF
                    and obv_short > obv_long * AD_OBV_RISING_THRESHOLD
                ):
                    result["positive"] = "✅ Strong accumulation detected (CMF/OBV rising)"
                    result["adjustment"] = 15
                    result["note"] = f"Accumulation: CMF={cmf:.2f}"

                # Moderate accumulation
                elif cmf > AD_MODERATE_ACCUMULATION_CMF and obv_short > obv_long:
                    result["positive"] = "✅ Accumulation pattern (positive money flow)"
                    result["adjustment"] = 8
                    result["note"] = f"Accumulation: CMF={cmf:.2f}"

                # Strong distribution: CMF < -0.10 and OBV falling
                elif (
                    cmf < AD_STRONG_DISTRIBUTION_CMF
                    and obv_short < obv_long * AD_OBV_FALLING_THRESHOLD
                ):
                    result["warning"] = "⚠️ Distribution detected - smart money exiting"
                    result["adjustment"] = -15
                    result["note"] = f"Distribution: CMF={cmf:.2f}"

                    # Block if price rising while distribution (divergence)
                    if (
                        price_trend > AD_PRICE_DIVERGENCE_THRESHOLD
                    ):  # Price up >3% while distribution
                        result["blocked"] = True
                        result["reason"] = (
                            "🚫 Bearish divergence: Price rising while smart money distributing "
                            f"(CMF={cmf:.2f})"
                        )
                        return result

                # Moderate distribution
                elif cmf < AD_MODERATE_DISTRIBUTION_CMF:
                    result["warning"] = "⚠️ Negative money flow - cautious"
                    result["adjustment"] = -8
                    result["note"] = f"Selling pressure: CMF={cmf:.2f}"

            # OBV divergence check (price up, OBV down = bearish)
            if price_trend > 0.02 and obv_short < obv_long * 0.90:
                if result["warning"] is None:
                    result["warning"] = "⚠️ OBV bearish divergence"
                    result["adjustment"] = -10
                    result["note"] = "Volume not confirming price rise"

        except Exception as e:
            logger.debug(f"A/D analysis error: {e}")

        return result

    def _calculate_obv(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate On Balance Volume."""
        try:
            close = df["close"]
            volume = df["volume"]

            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]

            for i in range(1, len(df)):
                if close.iloc[i] > close.iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i - 1]

            return obv
        except Exception:
            return None

    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> Optional[float]:
        """Calculate Chaikin Money Flow."""
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            volume = df["volume"]

            # Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low + 1e-10)

            # Money Flow Volume
            mfv = mfm * volume

            # CMF = Sum(MFV) / Sum(Volume) over period
            cmf = mfv.tail(period).sum() / (volume.tail(period).sum() + 1e-10)

            return cmf
        except Exception:
            return None

    def _check_foreign_flow(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Check foreign investor flow for the symbol.

        NEW v9.6: Vietnam market is heavily influenced by foreign flows.
        Net foreign selling can be a strong bearish signal.

        Args:
            symbol: Stock symbol
            df: DataFrame with data (may contain foreign_buy/sell columns)

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
            # Try to get foreign flow data from various sources
            foreign_data = None

            # Option 1: Check if df has foreign flow columns
            if "foreign_buy" in df.columns and "foreign_sell" in df.columns:
                recent_buy = df["foreign_buy"].tail(5).sum()
                recent_sell = df["foreign_sell"].tail(5).sum()
                foreign_data = {
                    "net_flow": recent_buy - recent_sell,
                    "buy_volume": recent_buy,
                    "sell_volume": recent_sell,
                }

            # Option 2: Try to get from market data service
            if foreign_data is None:
                try:
                    from src.data.market_data import get_foreign_flow

                    foreign_data = get_foreign_flow(symbol, days=5)
                except ImportError:
                    pass

            # Option 3: Try from SSI data source
            if foreign_data is None:
                try:
                    from src.data.ssi_data import get_ssi_foreign_flow

                    foreign_data = get_ssi_foreign_flow(symbol)
                except ImportError:
                    pass

            if foreign_data is None:
                return result

            net_flow = foreign_data.get("net_flow", 0)
            buy_volume = foreign_data.get("buy_volume", 0)
            sell_volume = foreign_data.get("sell_volume", 0)
            total_volume = buy_volume + sell_volume

            if total_volume <= 0:
                return result

            # Calculate net flow percentage
            net_pct = net_flow / total_volume if total_volume > 0 else 0

            # Heavy foreign selling - block entry (using centralized constants)
            if net_pct <= FOREIGN_HEAVY_SELLING_PCT:  # Net selling > 30% of foreign volume
                result["blocked"] = True
                result["reason"] = (
                    f"🚫 Heavy foreign selling: net {net_pct*100:.0f}% - "
                    "Wait for foreign selling to subside"
                )
                return result

            # Moderate foreign selling - warning
            if net_pct <= FOREIGN_MODERATE_SELLING_PCT:  # Net selling > 15%
                result["warning"] = f"⚠️ Foreign net selling: {net_pct*100:.0f}%"
                result["adjustment"] = -12
                result["note"] = f"Foreign selling pressure ({net_pct*100:.0f}%)"
                result["position_multiplier"] = 0.7
                return result

            # Light selling
            if net_pct <= FOREIGN_LIGHT_SELLING_PCT:
                result["warning"] = f"⚠️ Light foreign selling: {net_pct*100:.0f}%"
                result["adjustment"] = -5
                result["note"] = "Light foreign selling"
                return result

            # Heavy foreign buying - positive
            if net_pct >= FOREIGN_HEAVY_BUYING_PCT:  # Net buying > 25%
                result["positive"] = f"✅ Strong foreign buying: +{net_pct*100:.0f}%"
                result["adjustment"] = 15
                result["note"] = f"Foreign accumulation (+{net_pct*100:.0f}%)"
                result["position_multiplier"] = 1.1
                return result

            # Moderate buying
            if net_pct >= FOREIGN_MODERATE_BUYING_PCT:
                result["positive"] = f"✅ Foreign net buying: +{net_pct*100:.0f}%"
                result["adjustment"] = 8
                result["note"] = "Foreign buying support"

        except Exception as e:
            logger.debug(f"Foreign flow check error: {e}")

        return result

    def _check_margin_availability(self) -> Dict:
        """
        Check margin availability considering T+2.5 settlement.

        NEW v9.6: Vietnam has T+2.5 settlement, meaning trades settle
        2.5 days after execution. This affects buying power.

        Args:
            None (uses portfolio_manager)

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
            if self.portfolio_manager is None:
                return result

            # Get margin info from portfolio manager
            margin_info = None
            try:
                margin_info = self.portfolio_manager.get_margin_info()
            except AttributeError:
                pass

            if margin_info is None:
                # Try alternative method
                try:
                    buying_power = self.portfolio_manager.get_buying_power()
                    total_value = self.portfolio_manager.get_total_value()
                    pending_settlement = getattr(self.portfolio_manager, "pending_settlement", 0)

                    margin_info = {
                        "buying_power": buying_power,
                        "total_value": total_value,
                        "pending_settlement": pending_settlement,
                        "margin_used_pct": (
                            1 - (buying_power / total_value) if total_value > 0 else 0
                        ),
                    }
                except Exception:
                    return result

            # Extract margin metrics
            margin_used_pct = margin_info.get("margin_used_pct", 0)
            pending_settlement = margin_info.get("pending_settlement", 0)
            buying_power = margin_info.get("buying_power", 0)
            total_value = margin_info.get("total_value", 1)

            # Calculate effective margin considering T+2.5 pending
            effective_bp_pct = (buying_power - pending_settlement) / total_value

            # Block if margin is critically low (using centralized constants)
            if effective_bp_pct <= MARGIN_CRITICAL_THRESHOLD:  # < 5% available
                result["blocked"] = True
                result["reason"] = (
                    f"🚫 Insufficient margin: {effective_bp_pct*100:.1f}% available "
                    f"(T+2.5 pending: {pending_settlement:,.0f} VND)"
                )
                return result

            # Warning if margin is low
            if effective_bp_pct <= MARGIN_LOW_THRESHOLD:  # < 15% available
                result["warning"] = (
                    f"⚠️ Low margin: {effective_bp_pct*100:.1f}% available "
                    f"(T+2.5 pending: {pending_settlement:,.0f})"
                )
                result["adjustment"] = -10
                result["note"] = f"Low margin ({effective_bp_pct*100:.1f}%)"
                result["position_multiplier"] = 0.5  # Reduce position size
                return result

            # Moderate margin usage - reduce position
            if effective_bp_pct <= MARGIN_MODERATE_THRESHOLD:  # < 30% available
                result["warning"] = f"⚠️ Moderate margin: {effective_bp_pct*100:.1f}% available"
                result["adjustment"] = -5
                result["note"] = "Moderate margin usage"
                result["position_multiplier"] = 0.7

            # Check for high pending settlement (T+2.5 risk)
            if pending_settlement > total_value * MARGIN_HIGH_PENDING_THRESHOLD:  # > 30% pending
                if result["warning"] is None:
                    result["warning"] = f"⚠️ High T+2.5 pending: {pending_settlement:,.0f} VND"
                    result["adjustment"] = -5
                    result["note"] = "High pending settlement"
                    result["position_multiplier"] = min(result.get("position_multiplier", 1.0), 0.8)

        except Exception as e:
            logger.debug(f"Margin check error: {e}")

        return result

    def _check_consecutive_losses(self, symbol: str) -> Dict:
        """
        Check for consecutive losses on the symbol.

        IMPROVED v9.8: Weighted loss tracking based on loss magnitude.
        - Small loss (<3%): counts as 0.5
        - Medium loss (3-6%): counts as 1.0
        - Large loss (>6%): counts as 2.0
        - Block when weighted sum >= 3.0

        Args:
            symbol: Stock symbol

        Returns:
            Dict with blocked, warning, adjustment, note
        """
        result = {
            "blocked": False,
            "warning": None,
            "adjustment": 0,
            "note": None,
        }

        try:
            symbol = symbol.upper()

            # Check if in cool-down period
            if symbol in self._last_loss_date:
                last_loss = self._last_loss_date[symbol]
                days_since = (datetime.now() - last_loss).days

                if days_since < self.consecutive_loss_cooldown_days:
                    # NEW v9.8: Use weighted loss instead of simple count
                    weighted_loss = self._weighted_losses.get(symbol, 0)
                    loss_count = self._consecutive_losses.get(symbol, 0)

                    if weighted_loss >= CONSECUTIVE_LOSS_WEIGHTED_LIMIT:
                        result["blocked"] = True
                        result["reason"] = (
                            f"🚫 {symbol}: Weighted loss {weighted_loss:.1f} "
                            f"({loss_count} trades) - "
                            f"Cool-down: {self.consecutive_loss_cooldown_days - days_since} days remaining"
                        )
                        return result

            # Check current weighted loss
            weighted_loss = self._weighted_losses.get(symbol, 0)
            loss_count = self._consecutive_losses.get(symbol, 0)

            if weighted_loss >= CONSECUTIVE_LOSS_WEIGHTED_LIMIT:
                result["blocked"] = True
                result["reason"] = (
                    f"🚫 {symbol}: Weighted loss {weighted_loss:.1f} ({loss_count} trades) - "
                    f"Blocked for {self.consecutive_loss_cooldown_days} days"
                )
                return result

            # Warning if approaching limit (>= 70% of limit)
            if weighted_loss >= CONSECUTIVE_LOSS_WEIGHTED_LIMIT * 0.7:
                result["warning"] = (
                    f"⚠️ {symbol}: Weighted loss {weighted_loss:.1f} - approaching block"
                )
                result["adjustment"] = -15
                result["note"] = (
                    f"Near loss limit ({weighted_loss:.1f}/{CONSECUTIVE_LOSS_WEIGHTED_LIMIT})"
                )
                return result

            # Light warning for any recent losses
            if weighted_loss > 0:
                result["warning"] = (
                    f"⚠️ {symbol}: {loss_count} recent loss(es) (weighted: {weighted_loss:.1f})"
                )
                result["adjustment"] = int(-5 * weighted_loss)
                result["note"] = f"Recent losses (weighted: {weighted_loss:.1f})"

        except Exception as e:
            logger.debug(f"Consecutive loss check error: {e}")

        return result

    def record_trade_result(
        self, symbol: str, is_win: bool, loss_pct: Optional[float] = None
    ) -> None:
        """
        Record trade result for consecutive loss tracking.

        IMPROVED v9.8: Weighted loss tracking based on loss magnitude.

        Args:
            symbol: Stock symbol
            is_win: True if trade was profitable, False otherwise
            loss_pct: Loss percentage as decimal (e.g., 0.05 for 5% loss)
        """
        try:
            symbol = symbol.upper()

            if is_win:
                # Reset consecutive losses on win
                self._consecutive_losses[symbol] = 0
                self._weighted_losses[symbol] = 0
                if symbol in self._last_loss_date:
                    del self._last_loss_date[symbol]
            else:
                # Increment consecutive losses
                self._consecutive_losses[symbol] = self._consecutive_losses.get(symbol, 0) + 1
                self._last_loss_date[symbol] = datetime.now()

                # NEW v9.8: Calculate weighted loss
                loss_weight = self._calculate_loss_weight(loss_pct)
                self._weighted_losses[symbol] = self._weighted_losses.get(symbol, 0) + loss_weight

                logger.info(
                    f"[{symbol}] Recorded loss #{self._consecutive_losses[symbol]} "
                    f"(weight: {loss_weight:.1f}, total: {self._weighted_losses[symbol]:.1f}, "
                    f"limit: {CONSECUTIVE_LOSS_WEIGHTED_LIMIT})"
                )

        except Exception as e:
            logger.debug(f"Record trade result error: {e}")

    def _calculate_loss_weight(self, loss_pct: Optional[float]) -> float:
        """
        Calculate loss weight based on magnitude.

        NEW v9.8: Weighted loss system.
        - Small loss (<3%): 0.5 weight
        - Medium loss (3-6%): 1.0 weight
        - Large loss (>6%): 2.0 weight

        Args:
            loss_pct: Loss percentage as decimal (e.g., 0.05 for 5%)

        Returns:
            Loss weight (0.5, 1.0, or 2.0)
        """
        if loss_pct is None:
            return CONSECUTIVE_LOSS_MEDIUM_WEIGHT  # Default to medium

        abs_loss = abs(loss_pct)

        if abs_loss < CONSECUTIVE_LOSS_SMALL_THRESHOLD:
            return CONSECUTIVE_LOSS_SMALL_WEIGHT  # 0.5
        elif abs_loss < CONSECUTIVE_LOSS_MEDIUM_THRESHOLD:
            return CONSECUTIVE_LOSS_MEDIUM_WEIGHT  # 1.0
        else:
            return CONSECUTIVE_LOSS_LARGE_WEIGHT  # 2.0

    def reset_loss_tracking(self, symbol: Optional[str] = None) -> None:
        """
        Reset consecutive loss tracking.

        IMPROVED v9.8: Also resets weighted losses.

        Args:
            symbol: Specific symbol to reset, or None to reset all
        """
        if symbol:
            symbol = symbol.upper()
            self._consecutive_losses.pop(symbol, None)
            self._last_loss_date.pop(symbol, None)
            self._weighted_losses.pop(symbol, None)  # NEW v9.8
        else:
            self._consecutive_losses.clear()
            self._last_loss_date.clear()
            self._weighted_losses.clear()  # NEW v9.8

    def _check_sector_breadth(self, symbol: str) -> Dict:
        """
        Check sector breadth indicator - % of stocks bullish in sector.

        NEW v9.9: Sector breadth helps confirm sector-wide momentum.
        - Strong breadth (>60% bullish): Good sector support
        - Neutral breadth (40-60%): Mixed signals
        - Weak breadth (<40%): Sector headwind

        Args:
            symbol: Stock symbol to get sector for

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
            # Get sector for symbol
            sector = self._get_symbol_sector(symbol)
            if sector == "DEFAULT":
                return result

            # Try to get sector breadth data
            breadth_data = None

            # Option 1: From sector rotation strategy
            try:
                from src.strategies.sector_rotation_strategy import (
                    get_sector_rotation_strategy,
                    get_sector_symbols,
                )

                strategy = get_sector_rotation_strategy()
                sector_symbols = get_sector_symbols(sector)

                if sector_symbols:
                    bullish_count = 0
                    total_count = 0

                    # Quick breadth check using available data
                    for sym in sector_symbols[:15]:  # Check top 15 stocks
                        try:
                            from src.data.cache import get_cached_data

                            sym_df = get_cached_data(sym)
                            if sym_df is not None and len(sym_df) >= 20:
                                # Check if bullish: price > EMA20 and RSI > 45
                                close = sym_df["close"].iloc[-1]
                                ema20 = sym_df["close"].tail(20).mean()
                                rsi = self._calculate_rsi_quick(sym_df)

                                if close > ema20 and (rsi is None or rsi > 45):
                                    bullish_count += 1
                                total_count += 1
                        except Exception:
                            continue

                    if total_count >= 5:
                        breadth_data = {
                            "bullish_pct": bullish_count / total_count,
                            "bullish_count": bullish_count,
                            "total_count": total_count,
                            "sector": sector,
                        }
            except ImportError:
                pass

            # Option 2: From market breadth analyzer
            if breadth_data is None and self._breadth_analyzer:
                try:
                    sector_breadth = self._breadth_analyzer.get_sector_breadth(sector)
                    if sector_breadth:
                        breadth_data = {
                            "bullish_pct": sector_breadth.get("advance_decline_ratio", 0.5),
                            "sector": sector,
                        }
                except Exception:
                    pass

            if breadth_data is None:
                return result

            bullish_pct = breadth_data.get("bullish_pct", 0.5)

            # Strong sector breadth (>60% bullish)
            if bullish_pct >= 0.60:
                result["positive"] = (
                    f"✅ Strong sector breadth: {bullish_pct*100:.0f}% bullish in {sector}"
                )
                result["adjustment"] = 10
                result["note"] = f"Sector breadth {bullish_pct*100:.0f}%"
                return result

            # Good breadth (50-60%)
            if bullish_pct >= 0.50:
                result["positive"] = (
                    f"✅ Neutral+ sector breadth: {bullish_pct*100:.0f}% in {sector}"
                )
                result["adjustment"] = 3
                result["note"] = f"Neutral sector breadth"
                return result

            # Weak breadth (40-50%) - warning
            if bullish_pct >= 0.40:
                result["warning"] = (
                    f"⚠️ Weak sector breadth: {bullish_pct*100:.0f}% bullish in {sector}"
                )
                result["adjustment"] = -5
                result["note"] = f"Weak sector breadth {bullish_pct*100:.0f}%"
                return result

            # Very weak breadth (<40%) - strong warning
            if bullish_pct >= 0.25:
                result["warning"] = (
                    f"⚠️ Very weak sector: only {bullish_pct*100:.0f}% bullish in {sector}"
                )
                result["adjustment"] = -15
                result["note"] = f"Sector downtrend ({bullish_pct*100:.0f}%)"
                return result

            # Extremely weak breadth (<25%) - block
            result["blocked"] = True
            result["reason"] = (
                f"🚫 Sector collapse: only {bullish_pct*100:.0f}% bullish in {sector} - "
                "Don't fight the sector trend"
            )

        except Exception as e:
            logger.debug(f"Sector breadth check error: {e}")

        return result

    def _calculate_rsi_quick(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Quick RSI calculation for sector breadth analysis.

        Args:
            df: DataFrame with close prices
            period: RSI period

        Returns:
            RSI value or None
        """
        try:
            if len(df) < period + 1:
                return None

            close = df["close"]
            delta = close.diff()

            gain = (delta.where(delta > 0, 0)).tail(period).mean()
            loss = (-delta.where(delta < 0, 0)).tail(period).mean()

            if loss == 0:
                return 100.0

            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception:
            return None

    def _check_earnings_event_block(self, symbol: str) -> Dict:
        """
        Check if symbol has upcoming earnings or high-impact events.

        NEW v9.9: Block or reduce size before earnings to avoid
        binary event risk.

        Event Risk Handling:
        - Earnings < 3 days: BLOCK entry
        - Earnings 3-7 days: WARNING, reduce size 50%
        - Major corporate action (M&A, rights issue): WARNING

        Args:
            symbol: Stock symbol

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
            # Option 1: Check from fundamental analyzer
            if FUNDAMENTAL_AVAILABLE:
                try:
                    earnings_info = is_near_earnings(symbol)
                    if earnings_info:
                        days_to_earnings = earnings_info.get("days_until", 999)

                        # Block if earnings < 3 days
                        if days_to_earnings <= 3:
                            result["blocked"] = True
                            result["reason"] = (
                                f"🚫 {symbol}: Earnings in {days_to_earnings} day(s) - "
                                "Binary event risk, avoid new positions"
                            )
                            return result

                        # Warning if earnings 3-7 days
                        if days_to_earnings <= 7:
                            result["warning"] = f"⚠️ {symbol}: Earnings in {days_to_earnings} days"
                            result["adjustment"] = -15
                            result["note"] = f"Earnings in {days_to_earnings} days"
                            result["position_multiplier"] = 0.5
                            return result

                        # Light warning if earnings 7-14 days
                        if days_to_earnings <= 14:
                            result["warning"] = (
                                f"⚠️ {symbol}: Earnings approaching ({days_to_earnings} days)"
                            )
                            result["adjustment"] = -5
                            result["note"] = "Earnings approaching"
                            result["position_multiplier"] = 0.75
                            return result

                except Exception as e:
                    logger.debug(f"Earnings check error: {e}")

            # Option 2: Check from event calendar
            if self._event_calendar:
                try:
                    events = self._event_calendar.get_symbol_events(symbol, days_ahead=14)
                    if events:
                        for event in events:
                            event_type = event.get("type", "")
                            days_until = event.get("days_until", 999)
                            impact = event.get("impact", "LOW")

                            # High-impact corporate events
                            if event_type in ("EARNINGS", "DIVIDEND_EX", "AGM", "RIGHTS"):
                                if days_until <= 3 and impact == "HIGH":
                                    result["blocked"] = True
                                    result["reason"] = (
                                        f"🚫 {symbol}: {event_type} in {days_until} day(s) - "
                                        "High-impact event, avoid entry"
                                    )
                                    return result

                                if days_until <= 7:
                                    result["warning"] = (
                                        f"⚠️ {symbol}: {event_type} in {days_until} days"
                                    )
                                    result["adjustment"] = -10
                                    result["note"] = f"{event_type} approaching"
                                    result["position_multiplier"] = 0.6
                                    return result

                except Exception as e:
                    logger.debug(f"Event calendar check error: {e}")

            # Option 3: Fallback - check earnings season (Q1: Feb-Mar, Q2: May-Jun, etc.)
            try:
                from datetime import datetime

                now = datetime.now()
                month = now.month

                # Vietnam earnings seasons (approximate)
                # Q4: Jan-Feb | Q1: Apr-May | Q2: Jul-Aug | Q3: Oct-Nov
                earnings_months = [1, 2, 4, 5, 7, 8, 10, 11]

                if month in earnings_months:
                    # High probability of earnings season
                    result["warning"] = "⚠️ Earnings season - verify no pending announcements"
                    result["adjustment"] = -3
                    result["note"] = "Earnings season caution"
                    result["position_multiplier"] = 0.9

            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Earnings event block check error: {e}")

        return result

    def get_filter_priority(self, filter_name: str) -> str:
        """
        Get priority level for a filter.

        NEW v9.8: Returns CRITICAL, IMPORTANT, or OPTIONAL.

        Args:
            filter_name: Name of the filter

        Returns:
            Priority level string
        """
        if filter_name in FILTER_PRIORITY_CRITICAL:
            return "CRITICAL"
        elif filter_name in FILTER_PRIORITY_IMPORTANT:
            return "IMPORTANT"
        else:
            return "OPTIONAL"

    def get_last_entry_quality(self) -> Tuple[float, str]:
        """
        Get the last calculated entry quality score and label.

        NEW v9.8: Returns (score, label) tuple.

        Returns:
            Tuple of (score, label)
        """
        score = self._last_entry_quality
        label = self.get_entry_quality_label(score)
        return score, label

    # =========================================================================
    # NEW v9.7: Filter Performance Analytics
    # =========================================================================

    def get_filter_rejection_stats(self) -> Dict:
        """
        Get comprehensive filter rejection statistics.

        NEW v9.7: Provides insights into which filters are blocking entries
        and their effectiveness to help identify over-filtering issues.

        Returns:
            Dict with:
            - summary: Overall rejection metrics
            - by_filter: Per-filter statistics
            - recommendations: Actionable suggestions
            - top_blockers: Filters that block most entries
        """
        try:
            # Get local tracking stats
            local_stats = {
                "pass_counts": dict(self._filter_pass_count),
                "fail_counts": dict(self._filter_fail_count),
            }

            # Calculate local rejection rates
            rejection_rates = {}
            for filter_name in set(self._filter_pass_count.keys()) | set(
                self._filter_fail_count.keys()
            ):
                passes = self._filter_pass_count.get(filter_name, 0)
                fails = self._filter_fail_count.get(filter_name, 0)
                total = passes + fails
                if total > 0:
                    rejection_rates[filter_name] = {
                        "total_checks": total,
                        "passed": passes,
                        "blocked": fails,
                        "rejection_rate": fails / total,
                    }

            # Get comprehensive stats from tracker
            tracker_stats = self._filter_tracker.get_all_stats()
            tracker_dashboard = self._filter_tracker.get_dashboard_data()

            # Identify top blockers
            top_blockers = sorted(
                rejection_rates.items(), key=lambda x: x[1]["blocked"], reverse=True
            )[:5]

            # Calculate overall metrics
            total_checks = sum(r["total_checks"] for r in rejection_rates.values())
            total_blocks = sum(r["blocked"] for r in rejection_rates.values())

            return {
                "summary": {
                    "total_filter_checks": total_checks,
                    "total_blocks": total_blocks,
                    "overall_rejection_rate": (
                        total_blocks / total_checks if total_checks > 0 else 0
                    ),
                    "active_filters": len(rejection_rates),
                },
                "by_filter": rejection_rates,
                "top_blockers": [{"filter": name, **stats} for name, stats in top_blockers],
                "tracker_summary": tracker_dashboard.get("summary", {}),
                "recommendations": tracker_dashboard.get("recommendations", []),
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting filter rejection stats: {e}")
            return {
                "error": str(e),
                "pass_counts": dict(self._filter_pass_count),
                "fail_counts": dict(self._filter_fail_count),
            }

    def record_trade_result_for_filters(
        self, symbol: str, filters_passed: List[str], filters_blocked: List[str], is_win: bool
    ) -> None:
        """
        Record trade outcome for filter effectiveness analysis.

        NEW v9.7: Call this after each trade closes to update filter
        effectiveness metrics. This helps identify:
        - Filters with high false negative rate (blocking winning trades)
        - Filters with low precision (passing losing trades)

        Args:
            symbol: Stock symbol
            filters_passed: List of filter names that passed for this entry
            filters_blocked: List of filter names that would have blocked (from backtest)
            is_win: Whether the trade was profitable
        """
        try:
            self._filter_tracker.record_trade_outcome(
                symbol=symbol,
                filters_that_passed=filters_passed,
                filters_that_blocked=filters_blocked,
                won=is_win,
            )

            # Also update consecutive loss tracking
            self.record_trade_result(symbol, is_win)

            logger.debug(
                f"[{symbol}] Recorded trade result: {'WIN' if is_win else 'LOSS'} "
                f"(passed: {len(filters_passed)}, blocked: {len(filters_blocked)})"
            )

        except Exception as e:
            logger.error(f"Error recording trade result for filters: {e}")

    def print_filter_dashboard(self) -> None:
        """
        Print a formatted filter performance dashboard to console.

        NEW v9.7: Useful for debugging and monitoring filter effectiveness.
        """
        try:
            self._filter_tracker.print_dashboard()
        except Exception as e:
            logger.error(f"Error printing filter dashboard: {e}")

            # Fallback to local stats
            print("\n📊 LOCAL FILTER STATS (Fallback)")
            print("=" * 50)
            for filter_name in sorted(self._filter_pass_count.keys()):
                passes = self._filter_pass_count.get(filter_name, 0)
                fails = self._filter_fail_count.get(filter_name, 0)
                total = passes + fails
                rate = fails / total if total > 0 else 0
                print(f"  {filter_name}: {passes}P/{fails}F ({rate:.1%} rejection)")
            print("=" * 50)

    def get_filter_recommendations(self) -> List[Dict]:
        """
        Get actionable recommendations for filter optimization.

        NEW v9.7: Returns suggestions for:
        - Filters that are too strict (high false negative rate)
        - Filters that are too loose (low precision)
        - Potentially redundant filters

        Returns:
            List of recommendation dicts with filter, type, priority, reason, suggestion
        """
        try:
            dashboard = self._filter_tracker.get_dashboard_data()
            return dashboard.get("recommendations", [])
        except Exception as e:
            logger.error(f"Error getting filter recommendations: {e}")
            return []

    def export_filter_analytics(self, filepath: str = "entry_filter_analytics.json") -> str:
        """
        Export comprehensive filter analytics to JSON file.

        NEW v9.7: Exports all filter performance data for external analysis.

        Args:
            filepath: Output file path

        Returns:
            Path to exported file
        """
        import json

        try:
            analytics = {
                "rejection_stats": self.get_filter_rejection_stats(),
                "tracker_dashboard": self._filter_tracker.get_dashboard_data(),
                "redundant_filters": self._filter_tracker.get_redundant_filters(),
                "exported_at": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(analytics, f, indent=2, default=str)

            logger.info(f"Filter analytics exported to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error exporting filter analytics: {e}")
            return ""
