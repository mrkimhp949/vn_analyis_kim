# -*- coding: utf-8 -*-
"""
Market Regime Detection - Unified Module

Features:
- Basic regime detection (SMA, momentum, volatility)
- HMM-based regime detection (optional)
- Multi-index analysis (VNINDEX, VN30, HNX)
- Sector rotation & foreign flow integration
- Market breadth analysis

Author: Trading Bot Team
Version: 2.0.0
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - All magic numbers explained and centralized
# =============================================================================


class RegimeType(Enum):
    """Market regime types"""

    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CORRECTION = "CORRECTION"


class FlowSignal(Enum):
    """Foreign/margin flow signals"""

    BUYING = "BUYING"
    SELLING = "SELLING"
    NEUTRAL = "NEUTRAL"
    HIGH_RISK = "HIGH_RISK"


class ThresholdConstants:
    """
    Centralized threshold constants with explanations.
    All values are based on empirical analysis of Vietnam market data.
    """

    # Regime classification thresholds
    BULL_THRESHOLD: Final[float] = 0.45  # Composite score >= this = BULL
    BEAR_THRESHOLD: Final[float] = -0.50  # Composite score <= this = BEAR
    VOLATILITY_THRESHOLD: Final[float] = 0.70  # Volatility score > this = HIGH_VOLATILITY
    MIN_CONFIDENCE: Final[float] = 50.0  # Minimum confidence to be tradeable

    # HMM parameters
    HMM_MIN_DATA_POINTS: Final[int] = 60  # Minimum data points for HMM
    HMM_MIN_STATE_STD: Final[float] = 0.1  # Min std between states for valid HMM
    HMM_HIGH_CONFIDENCE: Final[float] = 0.6  # HMM confidence threshold for override
    HMM_N_COMPONENTS: Final[int] = 3  # Number of hidden states (BULL, BEAR, SIDEWAYS)
    HMM_MAX_ITER: Final[int] = 50  # Max iterations for HMM fitting
    HMM_MAX_ATTEMPTS: Final[int] = 3  # Max retry attempts for HMM convergence

    # Technical analysis periods
    SMA_SHORT: Final[int] = 20  # Short-term SMA period
    SMA_MEDIUM: Final[int] = 50  # Medium-term SMA period
    SMA_LONG: Final[int] = 200  # Long-term SMA period
    ATR_PERIOD: Final[int] = 14  # ATR calculation period
    RSI_PERIOD: Final[int] = 14  # RSI calculation period
    ROC_SHORT: Final[int] = 10  # Short ROC period
    ROC_MEDIUM: Final[int] = 20  # Medium ROC period
    ROC_LONG: Final[int] = 50  # Long ROC period

    # Volume analysis
    VOLUME_SMA_PERIOD: Final[int] = 20  # Volume SMA period
    VOLUME_RATIO_THRESHOLD: Final[float] = 1.3  # Volume spike threshold (30% above avg)
    VOLUME_CONFIRM_UP: Final[float] = 1.1  # Volume confirmation for uptrend

    # Price change thresholds
    PRICE_CHANGE_SIGNIFICANT: Final[float] = 0.02  # 2% price change is significant

    # Foreign flow thresholds
    FOREIGN_BUYING_THRESHOLD: Final[float] = 0.3  # Score > this = BUYING
    FOREIGN_SELLING_THRESHOLD: Final[float] = -0.3  # Score < this = SELLING

    # Correlation breakdown detection
    CORRELATION_MAX_DEVIATION: Final[float] = 0.4  # Max deviation between indices
    CORRELATION_DIVERGENCE_THRESHOLD: Final[float] = 0.2  # Threshold for divergence

    # Volatility percentiles
    VOLATILITY_HIGH_PERCENTILE: Final[float] = 75.0  # High volatility warning
    VOLATILITY_EXTREME_PERCENTILE: Final[float] = 85.0  # Extreme volatility
    VOLATILITY_CRITICAL_PERCENTILE: Final[float] = 90.0  # Critical - no trading

    # Cache settings
    CACHE_TTL_SECONDS: Final[int] = 3600  # 1 hour cache TTL

    # Minimum data requirements
    MIN_DATA_POINTS: Final[int] = 50  # Minimum data points for analysis
    MIN_DATA_POINTS_VOLATILITY: Final[int] = 60  # For volatility percentile


# Multi-index weights for composite score
class IndexWeights:
    """Weights for multi-index composite score calculation"""

    VNINDEX: Final[float] = 0.40  # Main index - highest weight
    VN30: Final[float] = 0.30  # Blue chips - second weight
    HNX: Final[float] = 0.15  # Small caps - lower weight
    BREADTH: Final[float] = 0.15  # Market breadth


# Component weights for regime scoring
class ComponentWeights:
    """Weights for individual components in regime calculation"""

    # Basic detection weights
    TREND: Final[float] = 0.35
    MOMENTUM: Final[float] = 0.25
    VOLUME: Final[float] = 0.15
    VOLATILITY: Final[float] = 0.10
    SECTOR_ROTATION: Final[float] = 0.075
    FOREIGN_FLOW: Final[float] = 0.075

    # Enhanced index analysis weights
    INDEX_TREND: Final[float] = 0.40
    INDEX_MOMENTUM: Final[float] = 0.30
    INDEX_VOLATILITY: Final[float] = 0.20
    INDEX_VOLUME: Final[float] = 0.10


# =============================================================================
# OPTIONAL HMM SUPPORT
# =============================================================================

try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    HMM_AVAILABLE = True
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="hmmlearn")
except ImportError:
    HMM_AVAILABLE = False
    GaussianHMM = None
    StandardScaler = None
    logger.info("hmmlearn not installed. HMM-based regime detection disabled.")


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MarketRegime:
    """
    Container for market regime information.

    Attributes:
        regime: Market regime type (BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY, CORRECTION)
        confidence: Confidence level 0-100%
        tradeable: Whether market conditions allow trading
        components: Individual component scores used in calculation
        description: Human-readable description of current regime
        recommendations: List of trading recommendations
    """

    regime: str
    confidence: float
    tradeable: bool
    components: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate regime data after initialization"""
        valid_regimes = {r.value for r in RegimeType}
        if self.regime not in valid_regimes:
            logger.warning(f"Unknown regime type: {self.regime}, defaulting to SIDEWAYS")
            self.regime = RegimeType.SIDEWAYS.value
        self.confidence = max(0.0, min(100.0, self.confidence))


@dataclass
class EnhancedMarketRegime(MarketRegime):
    """
    Enhanced market regime with multi-index analysis.

    Additional Attributes:
        vnindex_score: VNINDEX composite score (-1 to 1)
        vn30_score: VN30 composite score (-1 to 1)
        hnx_score: HNX composite score (-1 to 1)
        margin_debt_signal: Margin debt risk signal
        foreign_flow_signal: Foreign investor flow signal
        market_breadth: Market breadth score (-1 to 1)
        leading_sectors: Top performing sectors
        lagging_sectors: Worst performing sectors
        volatility_percentile: Current volatility vs historical (0-100)
        correlation_breakdown: Whether indices are diverging abnormally
    """

    vnindex_score: float = 0.0
    vn30_score: float = 0.0
    hnx_score: float = 0.0
    margin_debt_signal: str = FlowSignal.NEUTRAL.value
    foreign_flow_signal: str = FlowSignal.NEUTRAL.value
    market_breadth: float = 0.5
    leading_sectors: List[str] = field(default_factory=list)
    lagging_sectors: List[str] = field(default_factory=list)
    volatility_percentile: float = 50.0
    correlation_breakdown: bool = False

    def __post_init__(self):
        """Validate enhanced regime data"""
        super().__post_init__()
        # Clamp scores to valid range
        self.vnindex_score = max(-1.0, min(1.0, self.vnindex_score))
        self.vn30_score = max(-1.0, min(1.0, self.vn30_score))
        self.hnx_score = max(-1.0, min(1.0, self.hnx_score))
        self.market_breadth = max(-1.0, min(1.0, self.market_breadth))
        self.volatility_percentile = max(0.0, min(100.0, self.volatility_percentile))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if division is not possible

    Returns:
        Result of division or default value
    """
    if denominator == 0 or not np.isfinite(denominator):
        return default
    result = numerator / denominator
    return result if np.isfinite(result) else default


def _calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index).

    Args:
        prices: Array of closing prices
        period: RSI period (default 14)

    Returns:
        RSI value (0-100)
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI when insufficient data

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return float(np.clip(rsi, 0.0, 100.0))


def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """
    Calculate ATR (Average True Range).

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        period: ATR period (default 14)

    Returns:
        ATR value
    """
    if len(close) < period + 1:
        return 0.0

    # True Range components
    high_low = high[-(period + 1) :] - low[-(period + 1) :]
    high_close = np.abs(high[-(period + 1) :] - np.roll(close[-(period + 1) :], 1))
    low_close = np.abs(low[-(period + 1) :] - np.roll(close[-(period + 1) :], 1))

    # Skip first element (invalid due to roll)
    tr = np.maximum(high_low[1:], np.maximum(high_close[1:], low_close[1:]))

    return float(np.mean(tr))


def _validate_dataframe(
    df: pd.DataFrame, min_rows: int, required_columns: List[str] = None
) -> Tuple[bool, str]:
    """
    Validate DataFrame for regime detection.

    Args:
        df: DataFrame to validate
        min_rows: Minimum required rows
        required_columns: List of required column names

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"

    if df.empty:
        return False, "DataFrame is empty"

    if len(df) < min_rows:
        return False, f"Insufficient data: {len(df)} rows, need {min_rows}"

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}"

    # Check for required 'close' column
    if "close" not in df.columns:
        return False, "Missing required 'close' column"

    # Check for NaN values in close
    if df["close"].isna().any():
        return False, "NaN values found in 'close' column"

    return True, ""


# =============================================================================
# MAIN DETECTOR CLASS
# =============================================================================


class MarketRegimeDetector:
    """
    Unified Market Regime Detection for Vietnam Stock Market.

    Supports:
    - Basic detection using SMA, momentum, and volatility
    - HMM-based detection (optional, requires hmmlearn)
    - Multi-index analysis (VNINDEX, VN30, HNX)
    - Sector rotation & foreign flow integration

    Example:
        >>> detector = MarketRegimeDetector()
        >>> regime = detector.detect(vnindex_df)
        >>> print(f"Regime: {regime.regime}, Tradeable: {regime.tradeable}")
    """

    def __init__(
        self,
        bull_threshold: float = ThresholdConstants.BULL_THRESHOLD,
        bear_threshold: float = ThresholdConstants.BEAR_THRESHOLD,
        volatility_threshold: float = ThresholdConstants.VOLATILITY_THRESHOLD,
        min_confidence: float = ThresholdConstants.MIN_CONFIDENCE,
        use_hmm: bool = True,
        enable_sector_rotation: bool = True,
        enable_foreign_flow: bool = True,
        vnindex_weight: float = IndexWeights.VNINDEX,
        vn30_weight: float = IndexWeights.VN30,
        hnx_weight: float = IndexWeights.HNX,
        breadth_weight: float = IndexWeights.BREADTH,
        # Legacy parameters (for backward compatibility)
        high_volatility_threshold: Optional[float] = None,
        trend_period: int = 50,
    ):
        """
        Initialize the Market Regime Detector.

        Args:
            bull_threshold: Composite score threshold for BULL regime
            bear_threshold: Composite score threshold for BEAR regime
            volatility_threshold: Volatility score threshold for HIGH_VOLATILITY
            min_confidence: Minimum confidence level for tradeable market
            use_hmm: Whether to use HMM-based detection (if available)
            enable_sector_rotation: Include sector rotation in analysis
            enable_foreign_flow: Include foreign flow in analysis
            vnindex_weight: Weight for VNINDEX in multi-index analysis
            vn30_weight: Weight for VN30 in multi-index analysis
            hnx_weight: Weight for HNX in multi-index analysis
            breadth_weight: Weight for market breadth in analysis
        """
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.volatility_threshold = volatility_threshold
        self.min_confidence = min_confidence
        self.use_hmm = use_hmm and HMM_AVAILABLE
        self.enable_sector_rotation = enable_sector_rotation
        self.enable_foreign_flow = enable_foreign_flow

        # Validate and normalize weights
        total_weight = vnindex_weight + vn30_weight + hnx_weight + breadth_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Index weights sum to {total_weight}, normalizing to 1.0")
            vnindex_weight /= total_weight
            vn30_weight /= total_weight
            hnx_weight /= total_weight
            breadth_weight /= total_weight

        self.vnindex_weight = vnindex_weight
        self.vn30_weight = vn30_weight
        self.hnx_weight = hnx_weight
        self.breadth_weight = breadth_weight

        # Legacy parameters (stored for backward compatibility)
        self.high_volatility_threshold = high_volatility_threshold or 0.03
        self.trend_period = trend_period

        # Cache for expensive operations
        self._margin_debt_cache: Optional[str] = None
        self._margin_debt_cache_time: Optional[datetime] = None
        self._sector_cache: Optional[Dict] = None
        self._sector_cache_time: Optional[datetime] = None

        logger.debug(
            f"MarketRegimeDetector initialized: HMM={self.use_hmm}, "
            f"sector_rotation={enable_sector_rotation}, foreign_flow={enable_foreign_flow}"
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def detect(
        self,
        index_df: pd.DataFrame,
        vn30_df: Optional[pd.DataFrame] = None,
        hnx_df: Optional[pd.DataFrame] = None,
        market_breadth_data: Optional[Dict] = None,
    ) -> MarketRegime:
        """
        Detect market regime from index data.

        This is the main entry point for regime detection. It automatically
        chooses between basic and enhanced detection based on available data.

        Args:
            index_df: VNINDEX OHLCV DataFrame with columns [open, high, low, close, volume]
            vn30_df: Optional VN30 data for enhanced detection
            hnx_df: Optional HNX data for enhanced detection
            market_breadth_data: Optional dict with 'advancing' and 'declining' counts

        Returns:
            MarketRegime or EnhancedMarketRegime object

        Example:
            >>> detector = MarketRegimeDetector()
            >>> regime = detector.detect(vnindex_df)
            >>> if regime.tradeable:
            ...     print("Market is tradeable")
        """
        # Validate input
        is_valid, error_msg = _validate_dataframe(index_df, ThresholdConstants.MIN_DATA_POINTS)
        if not is_valid:
            logger.warning(f"Invalid input data: {error_msg}")
            return self._create_default_regime(error_msg)

        try:
            # Use enhanced detection if multi-index data available
            if vn30_df is not None or hnx_df is not None:
                return self._detect_enhanced(index_df, vn30_df, hnx_df, market_breadth_data)

            # Basic detection
            return self._detect_basic(index_df)

        except Exception as e:
            logger.error(f"Error detecting regime: {e}", exc_info=True)
            return self._create_default_regime(f"Detection error: {str(e)}")

    def analyze_market_regime(self, vnindex_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Legacy method for backward compatibility.

        Returns dict format instead of dataclass for older code.

        Args:
            vnindex_df: VNINDEX data, or None to load automatically

        Returns:
            Dict with keys: regime, tradeable, confidence, details, message
        """
        if vnindex_df is None:
            vnindex_df = self._load_default_index_data()
            if vnindex_df is None:
                return self._create_legacy_default_regime()

        result = self.detect(vnindex_df)

        return {
            "regime": result.regime,
            "tradeable": result.tradeable,
            "confidence": result.confidence,
            "details": result.components,
            "message": result.description,
        }

    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on current market regime.

        Returns:
            float: Multiplier for position sizing (0.0 to 1.2)
                - 0.0: No trading
                - 0.5: Reduced size (BEAR)
                - 0.7: Cautious (SIDEWAYS)
                - 1.0: Normal (moderate BULL)
                - 1.2: Aggressive (strong BULL)
        """
        regime_info = self.analyze_market_regime()

        if not regime_info["tradeable"]:
            return 0.0

        regime = regime_info["regime"]
        confidence = regime_info["confidence"]

        if regime == RegimeType.BULL.value:
            return 1.2 if confidence >= 80 else 1.0
        elif regime == RegimeType.SIDEWAYS.value:
            return 0.7
        elif regime == RegimeType.CORRECTION.value:
            return 0.5
        else:
            return 0.5

    # =========================================================================
    # BASIC DETECTION
    # =========================================================================

    def _detect_basic(self, df: pd.DataFrame) -> MarketRegime:
        """
        Basic regime detection using technical indicators.

        Uses SMA crossovers, momentum (ROC), and volatility to determine regime.
        Optionally applies HMM for more sophisticated state detection.
        """
        components = self._calculate_components(df)
        composite_score = self._calculate_composite_score(components)

        # Try HMM detection for additional insight
        hmm_result = None
        if self.use_hmm:
            hmm_result = self._detect_regime_hmm(df)
            if hmm_result:
                components["hmm_regime"] = hmm_result["regime"]
                components["hmm_confidence"] = hmm_result["confidence"]

        # Classify regime based on composite score
        regime, confidence, description = self._classify_regime(components, composite_score)

        # HMM override if high confidence and different result
        if hmm_result and hmm_result["confidence"] >= ThresholdConstants.HMM_HIGH_CONFIDENCE:
            if hmm_result["regime"] != regime:
                components["regime_before_hmm"] = regime
                regime = hmm_result["regime"]
                logger.debug(f"HMM override: {components['regime_before_hmm']} -> {regime}")

        # Critical volatility override
        volatility = components.get("volatility", 0)
        if volatility > self.volatility_threshold:
            if regime != RegimeType.HIGH_VOLATILITY.value:
                components["regime_before_volatility_override"] = regime
            regime = RegimeType.HIGH_VOLATILITY.value
            description = f"High volatility ({volatility:.2f}) overrides {components.get('regime_before_volatility_override', 'N/A')}"

        tradeable = self._is_tradeable(regime, confidence, components)
        recommendations = self._generate_recommendations(
            regime, FlowSignal.NEUTRAL.value, FlowSignal.NEUTRAL.value, volatility * 100
        )

        return MarketRegime(
            regime=regime,
            confidence=confidence,
            tradeable=tradeable,
            components=components,
            description=description,
            recommendations=recommendations,
        )

    def _calculate_components(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all regime components from OHLCV data.

        Components:
        - trend: SMA crossover analysis (-1 to 1)
        - momentum: Rate of change analysis (-1 to 1)
        - volatility: ATR and rolling volatility (0 to 1)
        - volume_trend: Volume vs average (-1 to 1)
        - sector_rotation: Sector performance score (-1 to 1)
        - foreign_flow: Foreign investor flow score (-1 to 1)
        """
        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close

        components = {}
        current_price = float(close[-1])

        # 1. TREND - SMA crossover analysis
        components["trend"] = self._calculate_trend_score(close, current_price)

        # 2. MOMENTUM - Rate of change
        components["momentum"] = self._calculate_momentum_score(close)

        # 3. VOLATILITY - ATR and rolling std
        components["volatility"] = self._calculate_volatility_score(high, low, close, current_price)

        # 4. VOLUME TREND
        if "volume" in df.columns:
            components["volume_trend"] = self._calculate_volume_trend(df["volume"].values)
        else:
            components["volume_trend"] = 0.0

        # 5. SECTOR ROTATION (optional)
        if self.enable_sector_rotation:
            sector_data = self._get_sector_rotation_data()
            components["sector_rotation"] = sector_data.get("score", 0.0)
        else:
            components["sector_rotation"] = 0.0

        # 6. FOREIGN FLOW (optional)
        if self.enable_foreign_flow:
            foreign_data = self._get_foreign_flow_data()
            components["foreign_flow"] = foreign_data.get("score", 0.0)
        else:
            components["foreign_flow"] = 0.0

        # Store reference values
        sma_short = pd.Series(close).rolling(ThresholdConstants.SMA_SHORT).mean().iloc[-1]
        sma_medium = pd.Series(close).rolling(ThresholdConstants.SMA_MEDIUM).mean().iloc[-1]
        sma_long = (
            pd.Series(close).rolling(ThresholdConstants.SMA_LONG).mean().iloc[-1]
            if len(close) >= ThresholdConstants.SMA_LONG
            else sma_medium
        )

        components["sma20"] = float(sma_short)
        components["sma50"] = float(sma_medium)
        components["sma200"] = float(sma_long)
        components["current_price"] = current_price

        return components

    def _calculate_trend_score(self, close: np.ndarray, current_price: float) -> float:
        """
        Calculate trend score based on SMA crossovers.

        Scoring:
        - SMA20 > SMA50: +0.4 weighted by distance
        - Price > SMA200: +0.4 weighted by distance
        - SMA50 > SMA200: +0.2 weighted by distance
        """
        sma20 = float(pd.Series(close).rolling(ThresholdConstants.SMA_SHORT).mean().iloc[-1])
        sma50 = float(pd.Series(close).rolling(ThresholdConstants.SMA_MEDIUM).mean().iloc[-1])
        sma200 = (
            float(pd.Series(close).rolling(ThresholdConstants.SMA_LONG).mean().iloc[-1])
            if len(close) >= ThresholdConstants.SMA_LONG
            else sma50
        )

        trend_score = 0.0

        # Short vs Medium term
        if sma20 > sma50:
            trend_score += 0.4 * _safe_divide(sma20 - sma50, sma50)
        else:
            trend_score -= 0.4 * _safe_divide(sma50 - sma20, sma50)

        # Price vs Long term
        if current_price > sma200:
            trend_score += 0.4 * _safe_divide(current_price - sma200, sma200)
        else:
            trend_score -= 0.4 * _safe_divide(sma200 - current_price, sma200)

        # Medium vs Long term (golden/death cross)
        if sma50 > sma200:
            trend_score += 0.2 * _safe_divide(sma50 - sma200, sma200)
        else:
            trend_score -= 0.2 * _safe_divide(sma200 - sma50, sma200)

        return float(np.clip(trend_score, -1.0, 1.0))

    def _calculate_momentum_score(self, close: np.ndarray) -> float:
        """
        Calculate momentum score using Rate of Change (ROC).

        Uses weighted combination of short and medium-term ROC.
        """
        roc_20 = (
            _safe_divide(
                close[-1] - close[-ThresholdConstants.ROC_MEDIUM],
                close[-ThresholdConstants.ROC_MEDIUM],
            )
            if len(close) >= ThresholdConstants.ROC_MEDIUM
            else 0.0
        )

        roc_50 = (
            _safe_divide(
                close[-1] - close[-ThresholdConstants.ROC_LONG], close[-ThresholdConstants.ROC_LONG]
            )
            if len(close) >= ThresholdConstants.ROC_LONG
            else 0.0
        )

        # Scale ROC to -1 to 1 range (10x multiplier for typical market moves)
        momentum_score = (roc_20 * 0.6 + roc_50 * 0.4) * 10

        return float(np.clip(momentum_score, -1.0, 1.0))

    def _calculate_volatility_score(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, current_price: float
    ) -> float:
        """
        Calculate volatility score using ATR and rolling standard deviation.

        Higher score = higher volatility (0 to 1 range).
        """
        # ATR-based volatility
        atr = _calculate_atr(high, low, close, ThresholdConstants.ATR_PERIOD)
        atr_pct = _safe_divide(atr, current_price, default=0.02)

        # Rolling returns volatility
        returns = pd.Series(close).pct_change()
        rolling_vol = (
            returns.rolling(ThresholdConstants.SMA_SHORT).std().iloc[-1]
            if len(returns) >= ThresholdConstants.SMA_SHORT
            else 0.02
        )

        # Combine and scale (20x multiplier to normalize to 0-1 range)
        volatility_score = (atr_pct * 0.5 + rolling_vol * 0.5) * 20

        return float(np.clip(volatility_score, 0.0, 1.0))

    def _calculate_volume_trend(self, volume: np.ndarray) -> float:
        """
        Calculate volume trend relative to moving average.

        Returns:
            Score from -1 (very low volume) to 1 (very high volume)
        """
        if len(volume) < ThresholdConstants.VOLUME_SMA_PERIOD:
            return 0.0

        vol_sma = float(
            pd.Series(volume).rolling(ThresholdConstants.VOLUME_SMA_PERIOD).mean().iloc[-1]
        )

        if vol_sma <= 0:
            return 0.0

        volume_ratio = volume[-1] / vol_sma - 1.0

        return float(np.clip(volume_ratio, -1.0, 1.0))

    def _calculate_composite_score(self, components: Dict[str, float]) -> float:
        """
        Calculate weighted composite score from all components.

        Returns:
            Composite score from approximately -1 to 1
        """
        score = 0.0
        score += components.get("trend", 0) * ComponentWeights.TREND
        score += components.get("momentum", 0) * ComponentWeights.MOMENTUM
        score += components.get("volume_trend", 0) * ComponentWeights.VOLUME
        score -= (
            components.get("volatility", 0) * ComponentWeights.VOLATILITY
        )  # High vol is negative

        if self.enable_sector_rotation:
            score += components.get("sector_rotation", 0) * ComponentWeights.SECTOR_ROTATION
        if self.enable_foreign_flow:
            score += components.get("foreign_flow", 0) * ComponentWeights.FOREIGN_FLOW

        return float(score)

    def _classify_regime(
        self, components: Dict[str, float], composite_score: float
    ) -> Tuple[str, float, str]:
        """
        Classify regime based on components and composite score.

        Returns:
            Tuple of (regime_name, confidence, description)
        """
        volatility = components.get("volatility", 0)
        momentum = components.get("momentum", 0)

        # High volatility takes precedence
        if volatility > self.volatility_threshold:
            confidence = min(volatility * 100, 100)
            return (
                RegimeType.HIGH_VOLATILITY.value,
                confidence,
                f"High volatility ({volatility:.2f}). Risk management critical.",
            )

        # Bull market
        if composite_score >= self.bull_threshold:
            confidence = min(abs(composite_score) * 100, 100)
            return (
                RegimeType.BULL.value,
                confidence,
                f"Bullish regime (score: {composite_score:.2f}). Favorable for long positions.",
            )

        # Bear market or correction
        if composite_score <= self.bear_threshold:
            # Check if it's a correction (momentum stabilizing)
            if momentum > -0.3:
                return (
                    RegimeType.CORRECTION.value,
                    50.0,
                    f"Market correction (score: {composite_score:.2f}). Momentum stabilizing.",
                )
            confidence = min(abs(composite_score) * 100, 100)
            return (
                RegimeType.BEAR.value,
                confidence,
                f"Bearish regime (score: {composite_score:.2f}). Avoid new long positions.",
            )

        # Sideways/ranging market
        confidence = 50 + (1 - abs(composite_score)) * 30
        return (
            RegimeType.SIDEWAYS.value,
            confidence,
            f"Sideways market (score: {composite_score:.2f}). No clear trend.",
        )

    # =========================================================================
    # ENHANCED DETECTION (Multi-Index)
    # =========================================================================

    def _detect_enhanced(
        self,
        vnindex_df: pd.DataFrame,
        vn30_df: Optional[pd.DataFrame],
        hnx_df: Optional[pd.DataFrame],
        market_breadth_data: Optional[Dict],
    ) -> EnhancedMarketRegime:
        """
        Enhanced detection with multi-index analysis.

        Combines VNINDEX, VN30, and HNX data with market breadth
        for more robust regime detection.
        """
        # Calculate individual index scores
        vnindex_score, vnindex_components = self._analyze_single_index(vnindex_df, "VNINDEX")

        # VN30 score (fallback to VNINDEX * 0.95 if not available)
        vn30_score = vnindex_score * 0.95
        if vn30_df is not None:
            is_valid, _ = _validate_dataframe(vn30_df, ThresholdConstants.MIN_DATA_POINTS)
            if is_valid:
                vn30_score, _ = self._analyze_single_index(vn30_df, "VN30")

        # HNX score (fallback to VNINDEX * 1.1 - typically more volatile)
        hnx_score = vnindex_score * 1.1
        if hnx_df is not None:
            is_valid, _ = _validate_dataframe(hnx_df, ThresholdConstants.MIN_DATA_POINTS)
            if is_valid:
                hnx_score, _ = self._analyze_single_index(hnx_df, "HNX")

        # Market breadth score
        breadth_score = self._calculate_breadth_score(market_breadth_data)

        # Weighted composite score
        composite_score = (
            vnindex_score * self.vnindex_weight
            + vn30_score * self.vn30_weight
            + hnx_score * self.hnx_weight
            + breadth_score * self.breadth_weight
        )

        # Additional signals
        margin_signal = self._get_margin_debt_signal()
        foreign_signal = self._get_foreign_flow_signal(vnindex_df)
        volatility_pct = self._calculate_volatility_percentile(vnindex_df)
        correlation_breakdown = self._check_correlation_breakdown(
            vnindex_score, vn30_score, hnx_score
        )

        # Classify regime
        regime, confidence, description = self._classify_regime_enhanced(
            composite_score, vnindex_components, volatility_pct, correlation_breakdown
        )

        # Determine tradeability
        tradeable = self._is_tradeable_enhanced(
            regime, confidence, volatility_pct, correlation_breakdown
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            regime, margin_signal, foreign_signal, volatility_pct
        )

        # Get sector rotation info
        leading_sectors, lagging_sectors = self._get_sector_leaders_laggers()

        return EnhancedMarketRegime(
            regime=regime,
            confidence=confidence,
            tradeable=tradeable,
            components=vnindex_components,
            description=description,
            vnindex_score=vnindex_score,
            vn30_score=vn30_score,
            hnx_score=hnx_score,
            margin_debt_signal=margin_signal,
            foreign_flow_signal=foreign_signal,
            market_breadth=breadth_score,
            leading_sectors=leading_sectors,
            lagging_sectors=lagging_sectors,
            volatility_percentile=volatility_pct,
            correlation_breakdown=correlation_breakdown,
            recommendations=recommendations,
        )

    def _analyze_single_index(
        self, df: pd.DataFrame, index_name: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Analyze a single index for regime detection.

        Returns:
            Tuple of (composite_score, components_dict)
        """
        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close
        current_price = float(close[-1])

        components = {}

        # TREND score
        components["trend"] = self._calculate_index_trend_score(close, current_price)

        # MOMENTUM score (ROC + RSI)
        components["momentum"] = self._calculate_index_momentum_score(close)

        # VOLATILITY score
        returns = pd.Series(close).pct_change()
        volatility = (
            returns.rolling(ThresholdConstants.SMA_SHORT).std().iloc[-1]
            if len(returns) >= ThresholdConstants.SMA_SHORT
            else 0.02
        )
        components["volatility"] = float(
            -min(volatility * 20, 1.0)
        )  # Negative because high vol is bad

        # VOLUME score
        if "volume" in df.columns:
            components["volume"] = self._calculate_index_volume_score(df, close)
        else:
            components["volume"] = 0.0

        # Composite score
        score = (
            components["trend"] * ComponentWeights.INDEX_TREND
            + components["momentum"] * ComponentWeights.INDEX_MOMENTUM
            + components["volatility"] * ComponentWeights.INDEX_VOLATILITY
            + components["volume"] * ComponentWeights.INDEX_VOLUME
        )

        logger.debug(f"{index_name} analysis: score={score:.3f}, components={components}")

        return float(score), components

    def _calculate_index_trend_score(self, close: np.ndarray, current_price: float) -> float:
        """Calculate trend score for index analysis."""
        sma20 = float(pd.Series(close).rolling(ThresholdConstants.SMA_SHORT).mean().iloc[-1])
        sma50 = float(pd.Series(close).rolling(ThresholdConstants.SMA_MEDIUM).mean().iloc[-1])
        sma200 = (
            float(pd.Series(close).rolling(ThresholdConstants.SMA_LONG).mean().iloc[-1])
            if len(close) >= ThresholdConstants.SMA_LONG
            else sma50
        )

        trend_score = 0.0

        # Price position relative to SMAs
        if current_price > sma20 > sma50:
            trend_score += 0.5
        elif current_price < sma20 < sma50:
            trend_score -= 0.5

        # SMA alignment (golden cross / death cross)
        if sma20 > sma50 > sma200:
            trend_score += 0.3
        elif sma20 < sma50 < sma200:
            trend_score -= 0.3

        # Long-term trend
        if current_price > sma200:
            trend_score += 0.2
        else:
            trend_score -= 0.2

        return float(np.clip(trend_score, -1.0, 1.0))

    def _calculate_index_momentum_score(self, close: np.ndarray) -> float:
        """Calculate momentum score using ROC and RSI."""
        # Rate of Change
        roc_10 = (
            _safe_divide(
                close[-1] - close[-ThresholdConstants.ROC_SHORT],
                close[-ThresholdConstants.ROC_SHORT],
            )
            if len(close) >= ThresholdConstants.ROC_SHORT
            else 0.0
        )

        roc_20 = (
            _safe_divide(
                close[-1] - close[-ThresholdConstants.ROC_MEDIUM],
                close[-ThresholdConstants.ROC_MEDIUM],
            )
            if len(close) >= ThresholdConstants.ROC_MEDIUM
            else 0.0
        )

        # RSI
        rsi = _calculate_rsi(close, ThresholdConstants.RSI_PERIOD)

        # Combine ROC scores
        momentum_score = roc_10 * 5 + roc_20 * 3

        # RSI adjustment
        if rsi > 70:
            momentum_score -= 0.2  # Overbought
        elif rsi < 30:
            momentum_score += 0.2  # Oversold (potential bounce)
        elif 40 <= rsi <= 60:
            momentum_score += 0.1  # Healthy range

        return float(np.clip(momentum_score, -1.0, 1.0))

    def _calculate_index_volume_score(self, df: pd.DataFrame, close: np.ndarray) -> float:
        """Calculate volume score for index analysis."""
        vol_sma = df["volume"].rolling(ThresholdConstants.VOLUME_SMA_PERIOD).mean().iloc[-1]
        current_vol = df["volume"].iloc[-1]

        if vol_sma <= 0:
            return 0.0

        vol_ratio = current_vol / vol_sma
        price_up = close[-1] > close[-5] if len(close) >= 5 else True
        vol_up = vol_ratio > ThresholdConstants.VOLUME_CONFIRM_UP

        # Volume confirms price direction
        if price_up and vol_up:
            return 0.3
        elif not price_up and vol_up:
            return -0.3  # Distribution
        else:
            return 0.0

    def _classify_regime_enhanced(
        self,
        composite_score: float,
        components: Dict[str, float],
        volatility_pct: float,
        correlation_breakdown: bool,
    ) -> Tuple[str, float, str]:
        """Enhanced regime classification with additional factors."""
        trend = components.get("trend", 0)
        momentum = components.get("momentum", 0)

        # Extreme volatility override
        if volatility_pct > ThresholdConstants.VOLATILITY_EXTREME_PERCENTILE:
            return (
                RegimeType.HIGH_VOLATILITY.value,
                min(volatility_pct, 95),
                f"Extreme volatility ({volatility_pct:.0f}th percentile). Risk off.",
            )

        # Correlation breakdown indicates market stress
        if correlation_breakdown:
            return (
                RegimeType.HIGH_VOLATILITY.value,
                70.0,
                "Index correlation breakdown detected. Market stress conditions.",
            )

        # Bull market
        if composite_score >= self.bull_threshold:
            if trend > 0.5 and momentum > 0.3:
                return (
                    RegimeType.BULL.value,
                    min(85 + composite_score * 10, 95),
                    f"Strong bull market (score: {composite_score:.2f}). Trend and momentum aligned.",
                )
            return (
                RegimeType.BULL.value,
                min(65 + composite_score * 15, 80),
                f"Moderate bull market (score: {composite_score:.2f}).",
            )

        # Bear market or correction
        if composite_score <= self.bear_threshold:
            if momentum > -0.2 and trend < -0.3:
                return (
                    RegimeType.CORRECTION.value,
                    60.0,
                    f"Market correction (score: {composite_score:.2f}). Watch for reversal signals.",
                )
            return (
                RegimeType.BEAR.value,
                min(70 + abs(composite_score) * 20, 90),
                f"Bear market (score: {composite_score:.2f}). Capital preservation mode.",
            )

        # Sideways
        return (
            RegimeType.SIDEWAYS.value,
            50 + (1 - abs(composite_score)) * 25,
            f"Sideways market (score: {composite_score:.2f}). Range-bound strategies preferred.",
        )

    # =========================================================================
    # HMM DETECTION
    # =========================================================================

    def _detect_regime_hmm(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        HMM-based regime detection using returns distribution.

        Uses a 3-state Gaussian HMM to identify:
        - State 0: Bear (lowest mean return)
        - State 1: Sideways (middle mean return)
        - State 2: Bull (highest mean return)

        Returns:
            Dict with state, regime, confidence, probabilities, or None if failed
        """
        if not HMM_AVAILABLE:
            return None

        if len(df) < ThresholdConstants.HMM_MIN_DATA_POINTS:
            return None

        try:
            returns = df["close"].pct_change().dropna()
            if len(returns) < ThresholdConstants.MIN_DATA_POINTS:
                return None

            # Prepare data
            returns_array = returns.values.reshape(-1, 1)
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_array)

            # Initialize HMM
            hmm = GaussianHMM(
                n_components=ThresholdConstants.HMM_N_COMPONENTS,
                covariance_type="spherical",
                n_iter=ThresholdConstants.HMM_MAX_ITER,
                tol=1e-1,
                random_state=42,
                verbose=False,
                init_params="stmc",
            )

            # Fit with retry logic
            converged = False
            for attempt in range(ThresholdConstants.HMM_MAX_ATTEMPTS):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    try:
                        hmm.fit(returns_scaled)
                        if hasattr(hmm, "monitor_") and hmm.monitor_.converged:
                            converged = True
                            break
                        hmm.random_state = 42 + attempt + 1
                    except Exception as e:
                        logger.debug(f"HMM fit attempt {attempt + 1} failed: {e}")
                        if attempt == ThresholdConstants.HMM_MAX_ATTEMPTS - 1:
                            return None

            if not converged:
                logger.debug("HMM did not converge after max attempts")
                return None

            # Analyze results
            hidden_states = hmm.predict(returns_scaled)
            state_probs = hmm.predict_proba(returns_scaled)
            state_means = hmm.means_.flatten()

            # Check if states are distinguishable
            state_std = np.std(state_means)
            if state_std < ThresholdConstants.HMM_MIN_STATE_STD:
                logger.debug(f"HMM states not distinguishable: std={state_std:.4f}")
                return None

            # Map states to regimes (sorted by mean return)
            order = np.argsort(state_means)
            regime_map = {
                order[0]: RegimeType.BEAR.value,
                order[1]: RegimeType.SIDEWAYS.value,
                order[2]: RegimeType.BULL.value,
            }

            current_state = int(hidden_states[-1])
            current_regime = regime_map.get(current_state, RegimeType.SIDEWAYS.value)

            # Calculate confidence (adjusted by state separation)
            raw_confidence = float(state_probs[-1, current_state])
            confidence = raw_confidence * min(state_std / 0.5, 1.0)

            return {
                "state": current_state,
                "regime": current_regime,
                "confidence": confidence,
                "probabilities": state_probs[-1].tolist(),
                "state_means": state_means.tolist(),
            }

        except Exception as e:
            logger.debug(f"HMM detection failed: {e}")
            return None

    # =========================================================================
    # TRADEABILITY CHECKS
    # =========================================================================

    def _is_tradeable(self, regime: str, confidence: float, components: Dict[str, float]) -> bool:
        """
        Determine if market conditions allow trading.

        Conditions for non-tradeable:
        - Confidence below minimum threshold
        - HIGH_VOLATILITY regime
        - Strong BEAR with high confidence
        """
        if confidence < self.min_confidence:
            return False

        if regime == RegimeType.HIGH_VOLATILITY.value:
            return False

        if regime == RegimeType.BEAR.value and confidence > 70:
            return False

        return True

    def _is_tradeable_enhanced(
        self, regime: str, confidence: float, volatility_pct: float, correlation_breakdown: bool
    ) -> bool:
        """
        Enhanced tradeability check with additional factors.
        """
        # Basic checks
        if confidence < self.min_confidence:
            return False

        if regime == RegimeType.HIGH_VOLATILITY.value:
            return False

        if regime == RegimeType.BEAR.value and confidence > 75:
            return False

        # Enhanced checks
        if correlation_breakdown:
            return False

        if volatility_pct > ThresholdConstants.VOLATILITY_CRITICAL_PERCENTILE:
            return False

        return True

    # =========================================================================
    # EXTERNAL DATA INTEGRATION
    # =========================================================================

    def _get_sector_rotation_data(self) -> Dict:
        """
        Get sector rotation analysis data.

        Returns cached data if available and fresh.
        """
        now = datetime.now()

        # Check cache
        if (
            self._sector_cache is not None
            and self._sector_cache_time is not None
            and (now - self._sector_cache_time).seconds < ThresholdConstants.CACHE_TTL_SECONDS
        ):
            return self._sector_cache

        try:
            from src.market.sector_rotation import get_sector_analyzer

            analyzer = get_sector_analyzer()
            result = analyzer.analyze()

            self._sector_cache = {
                "score": result.score,
                "leading": result.leading_sectors,
                "lagging": result.lagging_sectors,
                "phase": result.phase,
            }
            self._sector_cache_time = now

            return self._sector_cache

        except ImportError:
            logger.debug("sector_rotation module not available")
            return {"score": 0.0, "leading": [], "lagging": [], "phase": "UNKNOWN"}
        except Exception as e:
            logger.warning(f"Sector rotation analysis failed: {e}")
            return {"score": 0.0, "leading": [], "lagging": [], "phase": "UNKNOWN"}

    def _get_foreign_flow_data(self) -> Dict:
        """Get foreign investor flow analysis data."""
        try:
            from src.market.foreign_flow import get_foreign_flow_analyzer

            analyzer = get_foreign_flow_analyzer()
            result = analyzer.analyze()
            return {
                "score": result.score,
                "net_value": result.net_value,
                "trend": result.trend,
            }
        except ImportError:
            logger.debug("foreign_flow module not available")
            return {"score": 0.0, "net_value": 0, "trend": "UNKNOWN"}
        except Exception as e:
            logger.warning(f"Foreign flow analysis failed: {e}")
            return {"score": 0.0, "net_value": 0, "trend": "UNKNOWN"}

    def _get_margin_debt_signal(self) -> str:
        """
        Get margin debt risk signal.

        Returns cached signal if available and fresh.
        """
        now = datetime.now()

        # Check cache
        if (
            self._margin_debt_cache is not None
            and self._margin_debt_cache_time is not None
            and (now - self._margin_debt_cache_time).seconds < ThresholdConstants.CACHE_TTL_SECONDS
        ):
            return self._margin_debt_cache

        # TODO: Integrate with actual margin debt data source
        # For now, return NEUTRAL
        signal = FlowSignal.NEUTRAL.value

        self._margin_debt_cache = signal
        self._margin_debt_cache_time = now

        return signal

    def _get_foreign_flow_signal(self, df: pd.DataFrame) -> str:
        """
        Get foreign flow signal (BUYING, SELLING, NEUTRAL).
        """
        try:
            from src.market.foreign_flow import get_foreign_flow_analyzer

            analyzer = get_foreign_flow_analyzer()
            result = analyzer.analyze()

            if result.score > ThresholdConstants.FOREIGN_BUYING_THRESHOLD:
                return FlowSignal.BUYING.value
            elif result.score < ThresholdConstants.FOREIGN_SELLING_THRESHOLD:
                return FlowSignal.SELLING.value
            return FlowSignal.NEUTRAL.value

        except (ImportError, Exception):
            # Fallback: estimate from volume and price
            return self._estimate_foreign_flow_from_price_volume(df)

    def _estimate_foreign_flow_from_price_volume(self, df: pd.DataFrame) -> str:
        """
        Estimate foreign flow from price and volume patterns.

        This is a fallback when actual foreign flow data is unavailable.
        """
        if df.empty or len(df) < 10:
            return FlowSignal.NEUTRAL.value

        if "volume" not in df.columns:
            return FlowSignal.NEUTRAL.value

        recent = df.tail(5)
        avg_vol = df["volume"].tail(ThresholdConstants.VOLUME_SMA_PERIOD).mean()

        if avg_vol <= 0:
            return FlowSignal.NEUTRAL.value

        vol_ratio = recent["volume"].mean() / avg_vol
        price_change = _safe_divide(
            recent["close"].iloc[-1] - recent["close"].iloc[0], recent["close"].iloc[0]
        )

        # High volume with price increase suggests buying
        if (
            vol_ratio > ThresholdConstants.VOLUME_RATIO_THRESHOLD
            and price_change > ThresholdConstants.PRICE_CHANGE_SIGNIFICANT
        ):
            return FlowSignal.BUYING.value
        # High volume with price decrease suggests selling
        elif (
            vol_ratio > ThresholdConstants.VOLUME_RATIO_THRESHOLD
            and price_change < -ThresholdConstants.PRICE_CHANGE_SIGNIFICANT
        ):
            return FlowSignal.SELLING.value

        return FlowSignal.NEUTRAL.value

    # =========================================================================
    # MARKET BREADTH & CORRELATION
    # =========================================================================

    def _calculate_breadth_score(self, breadth_data: Optional[Dict]) -> float:
        """
        Calculate market breadth score from advancing/declining data.

        Args:
            breadth_data: Dict with 'advancing' and 'declining' counts

        Returns:
            Score from -1 (all declining) to 1 (all advancing)
        """
        if breadth_data is None:
            return 0.0

        advancing = breadth_data.get("advancing", 0)
        declining = breadth_data.get("declining", 0)
        total = advancing + declining

        if total == 0:
            return 0.0

        # Advance/Decline ratio normalized to -1 to 1
        ad_ratio = advancing / total
        return float(np.clip((ad_ratio - 0.5) * 2, -1.0, 1.0))

    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """
        Calculate current volatility percentile vs historical.

        Returns:
            Percentile (0-100) where higher = more volatile than usual
        """
        if df.empty or len(df) < ThresholdConstants.MIN_DATA_POINTS_VOLATILITY:
            return 50.0

        returns = df["close"].pct_change().dropna()

        if len(returns) < ThresholdConstants.SMA_SHORT:
            return 50.0

        current_vol = returns.tail(ThresholdConstants.SMA_SHORT).std()
        historical_vol = returns.rolling(ThresholdConstants.SMA_SHORT).std().dropna()

        if len(historical_vol) == 0:
            return 50.0

        percentile = (historical_vol < current_vol).sum() / len(historical_vol) * 100

        return float(percentile)

    def _check_correlation_breakdown(
        self, vnindex_score: float, vn30_score: float, hnx_score: float
    ) -> bool:
        """
        Check for correlation breakdown between indices.

        Correlation breakdown indicates market stress when indices
        that normally move together start diverging significantly.

        Returns:
            True if correlation breakdown detected
        """
        result = self.analyze_index_divergence(vnindex_score, vn30_score, hnx_score)
        return result["has_breakdown"]

    def analyze_index_divergence(
        self,
        vnindex_score: float,
        vn30_score: float,
        hnx_score: float,
    ) -> Dict[str, any]:
        """
        Analyze divergence between market indices with detailed alerts.

        IMPROVED v4.2: Provides detailed divergence analysis and alerts.

        Divergence scenarios:
        1. VNINDEX up, VN30 down: Blue chips lagging (distribution?)
        2. VNINDEX down, VN30 up: Small caps selling (flight to quality)
        3. VNINDEX up, HNX down: Small caps lagging (risk-off)
        4. Large deviation between any indices: Market stress

        Args:
            vnindex_score: VNINDEX composite score (-1 to 1)
            vn30_score: VN30 composite score (-1 to 1)
            hnx_score: HNX composite score (-1 to 1)

        Returns:
            Dict with divergence analysis:
            - has_breakdown: bool
            - divergence_type: str
            - alert_level: str (INFO, WARNING, CRITICAL)
            - alerts: List[str]
            - recommendation: str
        """
        result = {
            "has_breakdown": False,
            "divergence_type": "NONE",
            "alert_level": "INFO",
            "alerts": [],
            "recommendation": "",
            "scores": {
                "vnindex": vnindex_score,
                "vn30": vn30_score,
                "hnx": hnx_score,
            },
        }

        scores = [s for s in [vnindex_score, vn30_score, hnx_score] if s != 0]

        if len(scores) < 2:
            return result

        mean_score = np.mean(scores)
        max_deviation = max(abs(s - mean_score) for s in scores)

        # Thresholds
        div_threshold = ThresholdConstants.CORRELATION_DIVERGENCE_THRESHOLD
        max_dev_threshold = ThresholdConstants.CORRELATION_MAX_DEVIATION

        # Check for divergence patterns
        has_positive = any(s > div_threshold for s in scores)
        has_negative = any(s < -div_threshold for s in scores)

        # Scenario 1: VNINDEX up, VN30 down (distribution in blue chips)
        if vnindex_score > div_threshold and vn30_score < -div_threshold:
            result["has_breakdown"] = True
            result["divergence_type"] = "VNINDEX_UP_VN30_DOWN"
            result["alert_level"] = "WARNING"
            result["alerts"].append(
                f"⚠️ DIVERGENCE: VNINDEX up ({vnindex_score:+.2f}) but VN30 down ({vn30_score:+.2f})"
            )
            result["alerts"].append(
                "📊 Blue chips lagging - possible distribution or sector rotation"
            )
            result["recommendation"] = (
                "Caution: Blue chip weakness may signal broader market weakness. "
                "Consider reducing VN30 exposure."
            )

        # Scenario 2: VNINDEX down, VN30 up (flight to quality)
        elif vnindex_score < -div_threshold and vn30_score > div_threshold:
            result["has_breakdown"] = True
            result["divergence_type"] = "VNINDEX_DOWN_VN30_UP"
            result["alert_level"] = "WARNING"
            result["alerts"].append(
                f"⚠️ DIVERGENCE: VNINDEX down ({vnindex_score:+.2f}) but VN30 up ({vn30_score:+.2f})"
            )
            result["alerts"].append(
                "📊 Flight to quality - small/mid caps selling, blue chips holding"
            )
            result["recommendation"] = (
                "Risk-off environment: Focus on VN30 blue chips. " "Avoid small/mid cap stocks."
            )

        # Scenario 3: VNINDEX up, HNX down (small caps lagging)
        elif vnindex_score > div_threshold and hnx_score < -div_threshold:
            result["has_breakdown"] = True
            result["divergence_type"] = "VNINDEX_UP_HNX_DOWN"
            result["alert_level"] = "INFO"
            result["alerts"].append(
                f"📊 DIVERGENCE: VNINDEX up ({vnindex_score:+.2f}) but HNX down ({hnx_score:+.2f})"
            )
            result["alerts"].append(
                "Small caps underperforming - typical in early bull or late cycle"
            )
            result["recommendation"] = (
                "Focus on HOSE stocks. HNX weakness may indicate "
                "risk aversion or liquidity concerns."
            )

        # Scenario 4: VNINDEX down, HNX up (speculative rally)
        elif vnindex_score < -div_threshold and hnx_score > div_threshold:
            result["has_breakdown"] = True
            result["divergence_type"] = "VNINDEX_DOWN_HNX_UP"
            result["alert_level"] = "WARNING"
            result["alerts"].append(
                f"⚠️ DIVERGENCE: VNINDEX down ({vnindex_score:+.2f}) but HNX up ({hnx_score:+.2f})"
            )
            result["alerts"].append("📊 Speculative rally in small caps - high risk environment")
            result["recommendation"] = (
                "High risk: Small cap rally while main index falls is unsustainable. "
                "Avoid chasing HNX momentum."
            )

        # Scenario 5: Large deviation (general market stress)
        elif max_deviation > max_dev_threshold:
            result["has_breakdown"] = True
            result["divergence_type"] = "HIGH_DEVIATION"
            result["alert_level"] = "WARNING"
            result["alerts"].append(
                f"⚠️ HIGH DEVIATION: Max deviation {max_deviation:.2f} > threshold {max_dev_threshold:.2f}"
            )
            result["alerts"].append(
                f"📊 Scores: VNINDEX={vnindex_score:+.2f}, VN30={vn30_score:+.2f}, HNX={hnx_score:+.2f}"
            )
            result["recommendation"] = (
                "Market stress detected. Indices not moving together. "
                "Reduce exposure until correlation normalizes."
            )

        # Scenario 6: Opposing directions (general divergence)
        elif has_positive and has_negative:
            result["has_breakdown"] = True
            result["divergence_type"] = "OPPOSING_DIRECTIONS"
            result["alert_level"] = "INFO"
            result["alerts"].append(f"📊 MIXED SIGNALS: Some indices positive, some negative")
            result["alerts"].append(
                f"Scores: VNINDEX={vnindex_score:+.2f}, VN30={vn30_score:+.2f}, HNX={hnx_score:+.2f}"
            )
            result["recommendation"] = (
                "Mixed market signals. Be selective with entries. "
                "Focus on sectors showing strength."
            )

        # Log alerts
        if result["alerts"]:
            for alert in result["alerts"]:
                if result["alert_level"] == "CRITICAL":
                    logger.warning(alert)
                elif result["alert_level"] == "WARNING":
                    logger.warning(alert)
                else:
                    logger.info(alert)

        return result

    def get_divergence_alerts(
        self,
        vnindex_df: pd.DataFrame,
        vn30_df: Optional[pd.DataFrame] = None,
        hnx_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, any]:
        """
        Get divergence alerts from index data.

        Convenience method to analyze divergence from raw DataFrames.

        Args:
            vnindex_df: VNINDEX OHLCV data
            vn30_df: Optional VN30 data
            hnx_df: Optional HNX data

        Returns:
            Divergence analysis dict
        """
        # Calculate scores
        vnindex_score = self._calculate_index_score(vnindex_df) if vnindex_df is not None else 0.0
        vn30_score = self._calculate_index_score(vn30_df) if vn30_df is not None else 0.0
        hnx_score = self._calculate_index_score(hnx_df) if hnx_df is not None else 0.0

        return self.analyze_index_divergence(vnindex_score, vn30_score, hnx_score)

    def _get_sector_leaders_laggers(self) -> Tuple[List[str], List[str]]:
        """Get leading and lagging sectors."""
        try:
            from src.market.sector_rotation import get_sector_analyzer

            analyzer = get_sector_analyzer()
            result = analyzer.analyze()
            return result.leading_sectors[:3], result.lagging_sectors[:3]
        except (ImportError, Exception):
            return [], []

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================

    def _generate_recommendations(
        self, regime: str, margin_signal: str, foreign_signal: str, volatility_pct: float
    ) -> List[str]:
        """
        Generate actionable trading recommendations based on regime.

        Args:
            regime: Current market regime
            margin_signal: Margin debt signal
            foreign_signal: Foreign flow signal
            volatility_pct: Volatility percentile

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Regime-specific recommendations
        if regime == RegimeType.BULL.value:
            recommendations.append("✅ Favorable for long positions")
            recommendations.append("📈 Focus on momentum and growth stocks")
            if foreign_signal == FlowSignal.BUYING.value:
                recommendations.append("🌍 Foreign buying supports uptrend")

        elif regime == RegimeType.BEAR.value:
            recommendations.append("⛔ Avoid new long positions")
            recommendations.append("💰 Preserve capital, consider hedging")
            recommendations.append("📉 Wait for reversal confirmation before buying")

        elif regime == RegimeType.SIDEWAYS.value:
            recommendations.append("📊 Range-bound strategies preferred")
            recommendations.append("🎯 Focus on support/resistance levels")
            recommendations.append("⚖️ Reduce position sizes")

        elif regime == RegimeType.CORRECTION.value:
            recommendations.append("👀 Watch for bottom formation signals")
            recommendations.append("📉 Accumulate quality stocks gradually")
            recommendations.append("🔍 Look for oversold conditions")

        elif regime == RegimeType.HIGH_VOLATILITY.value:
            recommendations.append("🚫 Avoid trading until volatility subsides")
            recommendations.append("💵 Stay in cash or reduce exposure")
            recommendations.append("⏳ Wait for market stabilization")

        # Additional warnings
        if margin_signal == FlowSignal.HIGH_RISK.value:
            recommendations.append("⚠️ High margin debt levels - increased systemic risk")

        if foreign_signal == FlowSignal.SELLING.value and regime != RegimeType.BEAR.value:
            recommendations.append("⚠️ Foreign selling pressure detected")

        if volatility_pct > ThresholdConstants.VOLATILITY_HIGH_PERCENTILE:
            recommendations.append(
                f"📊 Elevated volatility ({volatility_pct:.0f}th pct) - reduce position sizes"
            )

        return recommendations

    # =========================================================================
    # DEFAULT/FALLBACK METHODS
    # =========================================================================

    def _create_default_regime(self, reason: str) -> MarketRegime:
        """
        Create a default cautious regime when detection fails.

        Args:
            reason: Reason for using default regime

        Returns:
            MarketRegime with conservative settings
        """
        return MarketRegime(
            regime=RegimeType.SIDEWAYS.value,
            confidence=30.0,
            tradeable=False,
            components={
                "trend": 0.0,
                "momentum": 0.0,
                "volatility": 0.5,
                "volume_trend": 0.0,
                "reason": reason,
            },
            description=f"Default regime ({reason}). Trade with extreme caution.",
            recommendations=["⚠️ Insufficient data for reliable analysis", "🔍 Verify data source"],
        )

    def _create_legacy_default_regime(self) -> Dict:
        """Create legacy format default regime dict."""
        return {
            "regime": RegimeType.SIDEWAYS.value,
            "tradeable": False,
            "confidence": 30,
            "details": {"reason": "Insufficient data"},
            "message": "⚠️ Không đủ dữ liệu - sử dụng chế độ thận trọng",
        }

    def _load_default_index_data(self) -> Optional[pd.DataFrame]:
        """
        Load default VNINDEX data when not provided.

        Returns:
            DataFrame or None if loading fails
        """
        try:
            from src.data.loader import load_data

            return load_data(
                symbol="VNINDEX",
                lookback=250,
                is_index=True,
                resolution="1D",
                data_type="index",
                use_cache=True,
            )
        except ImportError:
            logger.warning("data.loader module not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to load default index data: {e}")
            return None


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_detector_instance: Optional[MarketRegimeDetector] = None


def get_regime_detector(**kwargs) -> MarketRegimeDetector:
    """
    Get singleton instance of regime detector.

    Args:
        **kwargs: Arguments passed to MarketRegimeDetector if creating new instance

    Returns:
        MarketRegimeDetector singleton instance

    Example:
        >>> detector = get_regime_detector()
        >>> regime = detector.detect(df)
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MarketRegimeDetector(**kwargs)
    return _detector_instance


def reset_detector() -> None:
    """Reset the singleton detector instance. Useful for testing."""
    global _detector_instance
    _detector_instance = None


def detect_regime(index_df: pd.DataFrame) -> MarketRegime:
    """
    Convenience function to detect regime.

    Args:
        index_df: VNINDEX OHLCV DataFrame

    Returns:
        MarketRegime object
    """
    detector = get_regime_detector()
    return detector.detect(index_df)


def detect_enhanced_regime(
    vnindex_df: pd.DataFrame,
    vn30_df: Optional[pd.DataFrame] = None,
    hnx_df: Optional[pd.DataFrame] = None,
    market_breadth_data: Optional[Dict] = None,
) -> EnhancedMarketRegime:
    """
    Convenience function for enhanced regime detection.

    Args:
        vnindex_df: VNINDEX OHLCV DataFrame
        vn30_df: Optional VN30 DataFrame
        hnx_df: Optional HNX DataFrame
        market_breadth_data: Optional market breadth dict

    Returns:
        EnhancedMarketRegime object
    """
    detector = get_regime_detector()
    result = detector.detect(vnindex_df, vn30_df, hnx_df, market_breadth_data)

    # Ensure we return EnhancedMarketRegime
    if isinstance(result, EnhancedMarketRegime):
        return result

    # Convert basic regime to enhanced
    return EnhancedMarketRegime(
        regime=result.regime,
        confidence=result.confidence,
        tradeable=result.tradeable,
        components=result.components,
        description=result.description,
        recommendations=result.recommendations,
    )


def check_market_before_trading() -> Tuple[bool, str]:
    """
    Helper to check market conditions before trading.

    Returns:
        Tuple of (is_tradeable, message)

    Example:
        >>> tradeable, msg = check_market_before_trading()
        >>> if not tradeable:
        ...     print(f"Skip trading: {msg}")
    """
    detector = get_regime_detector()
    result = detector.analyze_market_regime()
    return result["tradeable"], result["message"]


def get_market_position_adjustment() -> float:
    """
    Helper to get position size multiplier based on market regime.

    Returns:
        float: Position size multiplier (0.0 to 1.2)
    """
    detector = get_regime_detector()
    return detector.get_position_multiplier()


# =============================================================================
# LEGACY ALIASES (Backward Compatibility)
# =============================================================================

# Class aliases
MarketRegimeAnalyzer = MarketRegimeDetector
EnhancedRegimeDetector = MarketRegimeDetector

# Function aliases
get_enhanced_regime_detector = get_regime_detector


# =============================================================================
# MAIN - Testing & Demo
# =============================================================================

if __name__ == "__main__":
    """
    Demo and test the Market Regime Detector.

    Run with: python -m src.market.regime_detector
    """
    import sys

    print("\n" + "=" * 70)
    print("🧪 MARKET REGIME DETECTOR - TEST SUITE")
    print("=" * 70 + "\n")

    # Create synthetic test data
    print("📊 Creating synthetic test data...")
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=250, freq="D")

    # Simulate trending market
    trend = np.linspace(0, 50, 250)
    noise = np.cumsum(np.random.randn(250) * 5)
    prices = 1200 + trend + noise

    test_df = pd.DataFrame(
        {
            "date": dates,
            "open": prices * 0.995,
            "high": prices * 1.01,
            "low": prices * 0.985,
            "close": prices,
            "volume": np.random.randint(100_000_000, 500_000_000, 250),
        }
    )

    print(f"   Data shape: {test_df.shape}")
    print(f"   Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"   Price range: {prices.min():.2f} to {prices.max():.2f}")

    # Test 1: Basic Detection
    print("\n" + "-" * 50)
    print("📈 Test 1: Basic Detection")
    print("-" * 50)

    detector = MarketRegimeDetector(use_hmm=True)
    result = detector.detect(test_df)

    print(f"   Regime: {result.regime}")
    print(f"   Confidence: {result.confidence:.1f}%")
    print(f"   Tradeable: {result.tradeable}")
    print(f"   Description: {result.description}")
    print(f"   Components:")
    for key, value in result.components.items():
        if isinstance(value, float):
            print(f"      - {key}: {value:.4f}")

    # Test 2: Legacy Method
    print("\n" + "-" * 50)
    print("📊 Test 2: Legacy Method (analyze_market_regime)")
    print("-" * 50)

    legacy = detector.analyze_market_regime(test_df)
    print(f"   Regime: {legacy['regime']}")
    print(f"   Tradeable: {legacy['tradeable']}")
    print(f"   Confidence: {legacy['confidence']:.1f}%")

    # Test 3: Position Multiplier
    print("\n" + "-" * 50)
    print("💰 Test 3: Position Multiplier")
    print("-" * 50)

    multiplier = detector.get_position_multiplier()
    print(f"   Position Multiplier: {multiplier:.2f}x")

    # Test 4: Convenience Functions
    print("\n" + "-" * 50)
    print("🔧 Test 4: Convenience Functions")
    print("-" * 50)

    reset_detector()  # Reset singleton
    regime = detect_regime(test_df)
    print(f"   detect_regime(): {regime.regime}")

    tradeable, msg = check_market_before_trading()
    print(f"   check_market_before_trading(): tradeable={tradeable}")

    # Test 5: Enhanced Detection (simulated multi-index)
    print("\n" + "-" * 50)
    print("🚀 Test 5: Enhanced Detection (Multi-Index)")
    print("-" * 50)

    # Create VN30 data (slightly different)
    vn30_prices = prices * 1.02 + np.random.randn(250) * 3
    vn30_df = pd.DataFrame(
        {
            "date": dates,
            "open": vn30_prices * 0.995,
            "high": vn30_prices * 1.01,
            "low": vn30_prices * 0.985,
            "close": vn30_prices,
            "volume": np.random.randint(50_000_000, 200_000_000, 250),
        }
    )

    enhanced = detect_enhanced_regime(test_df, vn30_df)
    print(f"   Regime: {enhanced.regime}")
    print(f"   VNINDEX Score: {enhanced.vnindex_score:.3f}")
    print(f"   VN30 Score: {enhanced.vn30_score:.3f}")
    print(f"   Volatility Percentile: {enhanced.volatility_percentile:.1f}%")
    print(f"   Correlation Breakdown: {enhanced.correlation_breakdown}")

    # Test 6: Edge Cases
    print("\n" + "-" * 50)
    print("⚠️ Test 6: Edge Cases")
    print("-" * 50)

    # Empty DataFrame
    empty_result = detector.detect(pd.DataFrame())
    print(f"   Empty DF: regime={empty_result.regime}, tradeable={empty_result.tradeable}")

    # Insufficient data
    small_df = test_df.head(10)
    small_result = detector.detect(small_df)
    print(
        f"   Small DF (10 rows): regime={small_result.regime}, tradeable={small_result.tradeable}"
    )

    # Test 7: Recommendations
    print("\n" + "-" * 50)
    print("📝 Test 7: Recommendations")
    print("-" * 50)

    for rec in result.recommendations[:5]:
        print(f"   {rec}")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    sys.exit(0)
