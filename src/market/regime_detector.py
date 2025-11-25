"""
Market Regime Detection
Automatically detect market regime based on VN-Index data
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Container for market regime information"""

    regime: str  # BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY
    confidence: float  # 0-100
    tradeable: bool
    components: Dict[str, float]  # Individual scores
    description: str


class MarketRegimeDetector:
    """
    Automatic Market Regime Detection

    Uses multiple indicators to classify market into:
    - BULL: Strong uptrend
    - BEAR: Strong downtrend
    - SIDEWAYS: Range-bound
    - HIGH_VOLATILITY: Unstable, risky

    Components:
    1. Trend (SMA crossovers)
    2. Momentum (ROC)
    3. Volatility (ATR, rolling std)
    4. Breadth (advance/decline if available)
    5. NEW: Sector Rotation Detection
    6. NEW: Foreign Investor Flow Analysis
    """

    def __init__(
        self,
        bull_threshold: float = 0.5,  # Reduced from 0.6 for earlier trend detection
        bear_threshold: float = -0.6,  # IMPROVED: Raised from -0.5 to -0.6 to reduce false bear signals
        volatility_threshold: float = 0.7,
        min_confidence: float = 50.0,
        require_momentum_confirmation: bool = True,  # NEW: Require momentum + trend alignment
        enable_sector_rotation: bool = True,  # NEW: Enable sector rotation analysis
        enable_foreign_flow: bool = True,  # NEW: Enable foreign investor flow analysis
    ):
        """
        Args:
            bull_threshold: Score above this = BULL
            bear_threshold: Score below this = BEAR
            volatility_threshold: Volatility above this = HIGH_VOLATILITY
            min_confidence: Minimum confidence to be tradeable
            require_momentum_confirmation: Require momentum + trend alignment for BEAR classification
        """
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.volatility_threshold = volatility_threshold
        self.min_confidence = min_confidence
        self.require_momentum_confirmation = require_momentum_confirmation
        self.enable_sector_rotation = enable_sector_rotation
        self.enable_foreign_flow = enable_foreign_flow

        # Sector rotation tracking
        self._sector_performance_cache = {}
        self._leading_sectors = []
        self._lagging_sectors = []

    def detect(self, index_df: pd.DataFrame) -> MarketRegime:
        """
        Detect market regime from VN-Index data

        Args:
            index_df: DataFrame with OHLCV data for VN-Index

        Returns:
            MarketRegime object
        """
        if index_df is None or index_df.empty or len(index_df) < 200:
            logger.warning(
                f"Insufficient VN-Index data for regime detection: {len(index_df) if index_df is not None else 0} bars"
            )
            return self._default_regime("Insufficient data")

        try:
            # Calculate components
            components = self._calculate_components(index_df)

            # Calculate composite score
            composite_score = self._calculate_composite_score(components)

            # Determine regime
            regime, confidence, description = self._classify_regime(components, composite_score)

            # Determine tradeable
            tradeable = self._is_tradeable(regime, confidence, components)

            logger.info(
                f"📊 Market Regime: {regime} (confidence: {confidence:.1f}%, tradeable: {tradeable})"
            )
            logger.debug(f"   Components: {components}")

            return MarketRegime(
                regime=regime,
                confidence=confidence,
                tradeable=tradeable,
                components=components,
                description=description,
            )

        except Exception as e:
            logger.error(f"Error detecting market regime: {e}", exc_info=True)
            return self._default_regime(f"Error: {str(e)}")

    def _calculate_components(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate individual regime components"""
        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close

        components = {}

        # 1. TREND COMPONENT (SMA crossovers)
        # Short-term: SMA20 vs SMA50
        sma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        sma50 = pd.Series(close).rolling(50).mean().iloc[-1]
        sma200 = pd.Series(close).rolling(200).mean().iloc[-1]

        # Trend score: -1 (bearish) to +1 (bullish)
        trend_score = 0.0

        # SMA20 vs SMA50 (40% weight)
        if sma20 > sma50:
            trend_score += 0.4 * ((sma20 - sma50) / sma50)
        else:
            trend_score -= 0.4 * ((sma50 - sma20) / sma50)

        # Price vs SMA200 (40% weight)
        current_price = close[-1]
        if current_price > sma200:
            trend_score += 0.4 * ((current_price - sma200) / sma200)
        else:
            trend_score -= 0.4 * ((sma200 - current_price) / sma200)

        # SMA50 vs SMA200 (20% weight)
        if sma50 > sma200:
            trend_score += 0.2 * ((sma50 - sma200) / sma200)
        else:
            trend_score -= 0.2 * ((sma200 - sma50) / sma200)

        components["trend"] = np.clip(trend_score, -1, 1)

        # 2. MOMENTUM COMPONENT (Rate of Change)
        roc_20 = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0
        roc_50 = (close[-1] - close[-50]) / close[-50] if len(close) >= 50 else 0

        # Momentum score: -1 (bearish) to +1 (bullish)
        momentum_score = (roc_20 * 0.6 + roc_50 * 0.4) * 10  # Scale up
        components["momentum"] = np.clip(momentum_score, -1, 1)

        # 3. VOLATILITY COMPONENT (normalized)
        # Calculate ATR
        atr_period = 14
        if len(close) >= atr_period:
            tr = np.maximum(
                high[-(atr_period + 1) :] - low[-(atr_period + 1) :],
                np.abs(high[-(atr_period + 1) :] - close[-(atr_period + 2) : -1]),
            )
            tr = np.maximum(tr, np.abs(low[-(atr_period + 1) :] - close[-(atr_period + 2) : -1]))
            atr = np.mean(tr)
            atr_pct = atr / current_price
        else:
            atr_pct = 0.02  # Default 2%

        # Rolling volatility (20 days)
        returns = pd.Series(close).pct_change()
        rolling_vol = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0.02

        # Volatility score: 0 (low) to 1 (high)
        volatility_score = (atr_pct * 0.5 + rolling_vol * 0.5) * 20  # Scale up
        components["volatility"] = np.clip(volatility_score, 0, 1)

        # 4. VOLUME TREND (if available)
        if "volume" in df.columns:
            volume = df["volume"].values
            if len(volume) >= 20:
                vol_sma20 = pd.Series(volume).rolling(20).mean().iloc[-1]
                current_vol = volume[-1]
                volume_trend = (current_vol / vol_sma20 - 1) if vol_sma20 > 0 else 0
                components["volume_trend"] = np.clip(volume_trend, -1, 1)
            else:
                components["volume_trend"] = 0
        else:
            components["volume_trend"] = 0

        # 5. SECTOR ROTATION (NEW)
        if self.enable_sector_rotation:
            sector_rotation = self._analyze_sector_rotation()
            components["sector_rotation"] = sector_rotation.get("score", 0)
            components["leading_sectors"] = sector_rotation.get("leading", [])
            components["lagging_sectors"] = sector_rotation.get("lagging", [])
        else:
            components["sector_rotation"] = 0

        # 6. FOREIGN INVESTOR FLOW (NEW)
        if self.enable_foreign_flow:
            foreign_flow = self._analyze_foreign_flow()
            components["foreign_flow"] = foreign_flow.get("score", 0)
            components["foreign_net_value"] = foreign_flow.get("net_value", 0)
        else:
            components["foreign_flow"] = 0

        return components

    def _analyze_sector_rotation(self) -> Dict:
        """
        Analyze sector rotation to identify market phase.

        Uses dedicated SectorRotationAnalyzer for comprehensive analysis.

        Returns:
            Dict with sector rotation analysis
        """
        try:
            from src.market.sector_rotation import get_sector_analyzer

            analyzer = get_sector_analyzer()
            result = analyzer.analyze()

            # Update internal tracking
            self._leading_sectors = result.leading_sectors
            self._lagging_sectors = result.lagging_sectors

            return {
                "score": result.score,
                "leading": result.leading_sectors,
                "lagging": result.lagging_sectors,
                "phase": result.phase,
                "confidence": result.confidence,
                "recommendation": result.recommendation,
            }

        except ImportError:
            logger.warning("Sector rotation module not available")
            return {"score": 0, "leading": [], "lagging": [], "phase": "UNKNOWN"}
        except Exception as e:
            logger.warning(f"Sector rotation analysis failed: {e}")
            return {"score": 0, "leading": [], "lagging": [], "phase": "UNKNOWN"}

    def _analyze_foreign_flow(self) -> Dict:
        """
        Analyze foreign investor net buy/sell flow.

        Uses dedicated ForeignFlowAnalyzer for comprehensive analysis.

        Returns:
            Dict with foreign flow analysis
        """
        try:
            from src.market.foreign_flow import get_foreign_flow_analyzer

            analyzer = get_foreign_flow_analyzer()
            result = analyzer.analyze()

            return {
                "score": result.score,
                "net_value": result.net_value,
                "trend": result.trend,
                "strength": result.strength,
                "consecutive_days": result.consecutive_days,
                "vs_average": result.vs_average,
            }

        except ImportError:
            logger.warning("Foreign flow module not available")
            return {"score": 0, "net_value": 0, "trend": "UNKNOWN", "strength": "UNKNOWN"}
        except Exception as e:
            logger.warning(f"Foreign flow analysis failed: {e}")
            return {"score": 0, "net_value": 0, "trend": "UNKNOWN", "strength": "UNKNOWN"}

    def _calculate_composite_score(self, components: Dict[str, float]) -> float:
        """
        Calculate composite regime score

        Weights (IMPROVED with sector rotation and foreign flow):
        - Trend: 35% (reduced from 40%)
        - Momentum: 25% (reduced from 30%)
        - Volume: 15% (reduced from 20%)
        - Volatility: 10% (negative impact)
        - Sector Rotation: 7.5% (NEW)
        - Foreign Flow: 7.5% (NEW)
        """
        score = 0.0

        score += components["trend"] * 0.35
        score += components["momentum"] * 0.25
        score += components["volume_trend"] * 0.15
        score -= components["volatility"] * 0.10  # High volatility reduces score

        # NEW: Add sector rotation and foreign flow
        if self.enable_sector_rotation:
            score += components.get("sector_rotation", 0) * 0.075
        if self.enable_foreign_flow:
            score += components.get("foreign_flow", 0) * 0.075

        return score

    def _classify_regime(
        self, components: Dict[str, float], composite_score: float
    ) -> tuple[str, float, str]:
        """
        Classify regime based on components and composite score

        Returns:
            (regime, confidence, description)
        """
        volatility = components["volatility"]
        trend = components["trend"]
        momentum = components["momentum"]

        # HIGH VOLATILITY check first (overrides others)
        if volatility > self.volatility_threshold:
            confidence = min(volatility * 100, 100)
            description = (
                f"High volatility ({volatility:.2f}) detected. "
                f"ATR and rolling std elevated. Risky market."
            )
            return "HIGH_VOLATILITY", confidence, description

        # BULL regime
        if composite_score >= self.bull_threshold:
            confidence = min(abs(composite_score) * 100, 100)
            description = (
                f"Bullish trend (score: {composite_score:.2f}). "
                f"Trend: {trend:.2f}, Momentum: {momentum:.2f}. "
                f"Good for long positions."
            )
            return "BULL", confidence, description

        # BEAR regime
        elif composite_score <= self.bear_threshold:
            # IMPROVEMENT: Check for momentum divergence before confirming BEAR
            # If momentum is stabilizing despite negative trend, may not be true bear market
            if self.require_momentum_confirmation:
                # Momentum divergence: trend negative but momentum > -0.3 (stabilizing)
                if momentum > -0.3:
                    # Potential correction, not full bear market
                    confidence = 50.0  # Lower confidence
                    description = (
                        f"Market correction (score: {composite_score:.2f}). "
                        f"Trend: {trend:.2f}, but momentum stabilizing ({momentum:.2f}). "
                        f"May be temporary dip, not full bear market. Trade with caution."
                    )
                    return (
                        "SIDEWAYS",
                        confidence,
                        description,
                    )  # Classify as SIDEWAYS instead of BEAR

            confidence = min(abs(composite_score) * 100, 100)
            description = (
                f"Bearish trend (score: {composite_score:.2f}). "
                f"Trend: {trend:.2f}, Momentum: {momentum:.2f}. "
                f"Avoid long positions."
            )
            return "BEAR", confidence, description

        # SIDEWAYS (default)
        else:
            confidence = 50 + (1 - abs(composite_score)) * 30  # 50-80%
            description = (
                f"Sideways market (score: {composite_score:.2f}). "
                f"No clear trend. Trade with caution."
            )
            return "SIDEWAYS", confidence, description

    def _is_tradeable(self, regime: str, confidence: float, components: Dict[str, float]) -> bool:
        """
        Determine if market is tradeable

        Criteria:
        - Confidence >= min_confidence
        - NOT in HIGH_VOLATILITY (unless very short-term scalping)
        - NOT in BEAR with high conviction
        """
        # Confidence too low
        if confidence < self.min_confidence:
            return False

        # High volatility = not tradeable
        if regime == "HIGH_VOLATILITY":
            return False

        # Strong bear = not tradeable
        if regime == "BEAR" and confidence > 70:
            return False

        # Everything else is tradeable
        return True

    def _default_regime(self, reason: str) -> MarketRegime:
        """Return default cautious regime"""
        return MarketRegime(
            regime="SIDEWAYS",
            confidence=30.0,
            tradeable=False,
            components={
                "trend": 0.0,
                "momentum": 0.0,
                "volatility": 0.5,
                "volume_trend": 0.0,
            },
            description=f"Default regime (reason: {reason}). Trade with extreme caution.",
        )


# Singleton instance for easy access
_detector_instance: Optional[MarketRegimeDetector] = None


def get_regime_detector() -> MarketRegimeDetector:
    """Get singleton instance of regime detector"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MarketRegimeDetector()
    return _detector_instance


def detect_regime(index_df: pd.DataFrame) -> MarketRegime:
    """
    Convenience function to detect regime

    Args:
        index_df: VN-Index OHLCV data

    Returns:
        MarketRegime object
    """
    detector = get_regime_detector()
    return detector.detect(index_df)
