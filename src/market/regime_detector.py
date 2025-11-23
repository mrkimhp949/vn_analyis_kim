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
    """

    def __init__(
        self,
        bull_threshold: float = 0.5,  # Reduced from 0.6 for earlier trend detection
        bear_threshold: float = -0.6,  # IMPROVED: Raised from -0.5 to -0.6 to reduce false bear signals
        volatility_threshold: float = 0.7,
        min_confidence: float = 50.0,
        require_momentum_confirmation: bool = True,  # NEW: Require momentum + trend alignment
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

        return components

    def _calculate_composite_score(self, components: Dict[str, float]) -> float:
        """
        Calculate composite regime score

        Weights:
        - Trend: 40%
        - Momentum: 30%
        - Volume: 20%
        - Volatility: 10% (negative impact)
        """
        score = 0.0

        score += components["trend"] * 0.40
        score += components["momentum"] * 0.30
        score += components["volume_trend"] * 0.20
        score -= components["volatility"] * 0.10  # High volatility reduces score

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
                    return "SIDEWAYS", confidence, description  # Classify as SIDEWAYS instead of BEAR

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
