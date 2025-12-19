# -*- coding: utf-8 -*-
"""
Vietnam Market Breadth Analyzer

Real-time market breadth analysis for Vietnam stock market:
- Advance/Decline ratio
- New highs/lows
- Sector breadth
- Volume breadth
- Breadth thrust signals
- Data staleness handling with weight adjustment

Author: Trading Bot Team
Version: 1.1.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from threading import RLock
from enum import Enum

import numpy as np
import pandas as pd

from src.utils.data_staleness import (
    DataStalenessMixin,
    DataFreshness,
    StalenessConfig,
    STALENESS_CONFIGS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

BREADTH_CACHE_TTL = 60  # 1 minute
AD_RATIO_BULLISH_THRESHOLD = 1.5
AD_RATIO_BEARISH_THRESHOLD = 0.67
THRUST_RATIO_THRESHOLD = 0.90  # 90% advancing = breadth thrust


class BreadthSignal(Enum):
    """Market breadth signal types."""

    BREADTH_THRUST_UP = "breadth_thrust_up"  # Strong bullish
    STRONG_BULLISH = "strong_bullish"  # >2:1 A/D
    BULLISH = "bullish"  # >1.5:1 A/D
    NEUTRAL = "neutral"  # Around 1:1
    BEARISH = "bearish"  # <0.67:1 A/D
    STRONG_BEARISH = "strong_bearish"  # <0.5:1 A/D
    BREADTH_THRUST_DOWN = "breadth_thrust_down"  # Panic selling


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MarketBreadthData:
    """Market breadth snapshot."""

    timestamp: datetime

    # Advance/Decline
    advances: int = 0
    declines: int = 0
    unchanged: int = 0

    # Volumes
    advance_volume: int = 0
    decline_volume: int = 0

    # New highs/lows (52-week)
    new_highs: int = 0
    new_lows: int = 0

    # Limit hits (Vietnam specific)
    ceiling_hits: int = 0  # Stocks at +7%
    floor_hits: int = 0  # Stocks at -7%

    # By exchange
    hose_advances: int = 0
    hose_declines: int = 0
    hnx_advances: int = 0
    hnx_declines: int = 0

    source: str = ""

    @property
    def ad_ratio(self) -> float:
        """Advance/Decline ratio."""
        if self.declines == 0:
            return float("inf") if self.advances > 0 else 1.0
        return self.advances / self.declines

    @property
    def ad_difference(self) -> int:
        """Advance - Decline."""
        return self.advances - self.declines

    @property
    def total_issues(self) -> int:
        """Total traded issues."""
        return self.advances + self.declines + self.unchanged

    @property
    def advance_percent(self) -> float:
        """Percentage of advancing stocks."""
        total = self.total_issues
        if total == 0:
            return 0.5
        return self.advances / total

    @property
    def volume_ad_ratio(self) -> float:
        """Volume-weighted A/D ratio."""
        if self.decline_volume == 0:
            return float("inf") if self.advance_volume > 0 else 1.0
        return self.advance_volume / self.decline_volume

    @property
    def high_low_ratio(self) -> float:
        """New highs to new lows ratio."""
        if self.new_lows == 0:
            return float("inf") if self.new_highs > 0 else 1.0
        return self.new_highs / self.new_lows

    @property
    def limit_sentiment(self) -> float:
        """Sentiment from limit hits (-1 to 1)."""
        total = self.ceiling_hits + self.floor_hits
        if total == 0:
            return 0.0
        return (self.ceiling_hits - self.floor_hits) / total

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "advances": self.advances,
            "declines": self.declines,
            "unchanged": self.unchanged,
            "ad_ratio": self.ad_ratio,
            "ad_difference": self.ad_difference,
            "advance_percent": self.advance_percent,
            "volume_ad_ratio": self.volume_ad_ratio,
            "new_highs": self.new_highs,
            "new_lows": self.new_lows,
            "ceiling_hits": self.ceiling_hits,
            "floor_hits": self.floor_hits,
            "limit_sentiment": self.limit_sentiment,
            "source": self.source,
        }


@dataclass
class BreadthAnalysis:
    """Market breadth analysis result."""

    timestamp: datetime

    # Current breadth
    current: MarketBreadthData

    # Signal
    signal: BreadthSignal = BreadthSignal.NEUTRAL
    signal_strength: float = 0.0  # 0 to 1

    # Scores
    ad_score: float = 0.0  # -1 to 1
    volume_score: float = 0.0  # -1 to 1
    hl_score: float = 0.0  # -1 to 1 (high/low)
    limit_score: float = 0.0  # -1 to 1 (ceiling/floor)

    # Combined score
    overall_score: float = 0.0  # -1 to 1

    # Trend
    trend: str = "neutral"  # improving, stable, deteriorating
    ad_5day_trend: float = 0.0  # 5-day cumulative A/D

    # Trading implications
    entry_adjustment: float = 0.0  # Confidence adjustment
    position_multiplier: float = 1.0

    # Flags
    is_breadth_thrust: bool = False
    is_oversold_bounce: bool = False
    is_distribution: bool = False

    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "signal": self.signal.value,
            "signal_strength": self.signal_strength,
            "overall_score": self.overall_score,
            "ad_score": self.ad_score,
            "volume_score": self.volume_score,
            "trend": self.trend,
            "entry_adjustment": self.entry_adjustment,
            "position_multiplier": self.position_multiplier,
            "is_breadth_thrust": self.is_breadth_thrust,
            "reasons": self.reasons,
            "current_data": self.current.to_dict() if self.current else None,
        }


# =============================================================================
# MARKET BREADTH ANALYZER
# =============================================================================


class MarketBreadthAnalyzer(DataStalenessMixin):
    """
    Vietnam Market Breadth Analyzer.

    Analyzes market breadth data from multiple sources to generate
    trading signals and adjustments.

    Features data staleness handling:
    - Stale data (>5 min) reduces weight by 15%
    - Very stale data (>15 min) reduces weight by 50%

    Usage:
        analyzer = MarketBreadthAnalyzer()

        # Get current breadth analysis
        analysis = analyzer.get_analysis()

        # Check breadth for entry
        entry_check = analyzer.check_breadth_for_entry()

        # Get staleness-adjusted score
        adjusted = analyzer.get_adjusted_score()
    """

    def __init__(
        self,
        cache_ttl: int = BREADTH_CACHE_TTL,
        history_days: int = 20,
    ):
        self._cache_ttl = cache_ttl
        self._history_days = history_days

        self._lock = RLock()

        # Initialize staleness tracking with market_breadth config
        self._init_staleness("market_breadth")

        # Cache
        self._cache: Optional[BreadthAnalysis] = None
        # _cache_time is now managed by DataStalenessMixin

        # History for trend calculation
        self._history: List[MarketBreadthData] = []
        self._max_history = history_days * 2  # Keep 2x history days

        # Data providers
        self._ssi_provider = None
        self._init_providers()

        logger.info("📊 Market Breadth Analyzer initialized (with staleness handling)")

    def _init_providers(self):
        """Initialize data providers."""
        try:
            from src.data.ssi_provider import get_ssi_provider

            self._ssi_provider = get_ssi_provider()
        except ImportError:
            logger.debug("SSI provider not available for breadth data")

    def _fetch_breadth_data(self) -> Optional[MarketBreadthData]:
        """Fetch current market breadth data."""

        # Try SSI provider
        if self._ssi_provider is not None:
            try:
                breadth = self._ssi_provider.get_market_breadth()

                if breadth:
                    # Get index data for volumes
                    index_data = self._ssi_provider.get_index_data("VNINDEX")

                    # Get all stocks to calculate ceiling/floor hits
                    all_stocks = self._ssi_provider.get_all_stocks()
                    ceiling_hits = 0
                    floor_hits = 0
                    advance_volume = 0
                    decline_volume = 0

                    if all_stocks is not None and not all_stocks.empty:
                        # Count ceiling/floor hits
                        ceiling_hits = len(
                            all_stocks[all_stocks["close"] >= all_stocks["ceiling"] * 0.999]
                        )
                        floor_hits = len(
                            all_stocks[all_stocks["close"] <= all_stocks["floor"] * 1.001]
                        )

                        # Calculate volume by direction
                        advancing = all_stocks[all_stocks["change"] > 0]
                        declining = all_stocks[all_stocks["change"] < 0]
                        advance_volume = int(advancing["volume"].sum())
                        decline_volume = int(declining["volume"].sum())

                    return MarketBreadthData(
                        timestamp=datetime.now(),
                        advances=breadth.get("advance", 0),
                        declines=breadth.get("decline", 0),
                        unchanged=breadth.get("nochange", 0),
                        advance_volume=advance_volume,
                        decline_volume=decline_volume,
                        ceiling_hits=ceiling_hits,
                        floor_hits=floor_hits,
                        source="SSI",
                    )

            except Exception as e:
                logger.warning(f"SSI breadth fetch failed: {e}")

        # Try to estimate from VNINDEX if no direct data
        try:
            from src.data.vnindex_cache import get_cached_vnindex

            df = get_cached_vnindex(lookback=1)
            if df is not None and not df.empty:
                # Very rough estimate
                latest = df.iloc[-1]
                change = latest.get("change", 0)

                # Estimate based on index change
                if change > 0:
                    advances = int(150 + change * 20)
                    declines = int(100 - change * 10)
                else:
                    advances = int(100 + change * 10)
                    declines = int(150 - change * 20)

                return MarketBreadthData(
                    timestamp=datetime.now(),
                    advances=max(50, advances),
                    declines=max(50, declines),
                    unchanged=30,
                    source="estimated",
                )

        except Exception as e:
            logger.debug(f"VNINDEX estimate failed: {e}")

        return None

    def get_analysis(self, force_refresh: bool = False) -> BreadthAnalysis:
        """
        Get current market breadth analysis.

        Args:
            force_refresh: Bypass cache

        Returns:
            BreadthAnalysis with signals and adjustments
        """
        with self._lock:
            # Check cache
            if not force_refresh and self._cache is not None:
                if self._cache_time is not None:
                    age = (datetime.now() - self._cache_time).total_seconds()
                    if age < self._cache_ttl:
                        return self._cache

            # Fetch current data
            current_data = self._fetch_breadth_data()

            if current_data is None:
                # Return neutral analysis if no data
                return BreadthAnalysis(
                    timestamp=datetime.now(),
                    current=MarketBreadthData(timestamp=datetime.now()),
                    reasons=["No breadth data available"],
                )

            # Add to history
            self._history.append(current_data)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

            # Analyze
            analysis = self._analyze_breadth(current_data)

            # Cache - use staleness mixin method
            self._cache = analysis
            self._update_cache_timestamp()

            return analysis

    def _analyze_breadth(self, current: MarketBreadthData) -> BreadthAnalysis:
        """Perform breadth analysis."""
        analysis = BreadthAnalysis(
            timestamp=datetime.now(),
            current=current,
        )

        # Calculate scores
        analysis.ad_score = self._calculate_ad_score(current)
        analysis.volume_score = self._calculate_volume_score(current)
        analysis.hl_score = self._calculate_hl_score(current)
        analysis.limit_score = current.limit_sentiment

        # Weighted overall score
        analysis.overall_score = (
            analysis.ad_score * 0.40
            + analysis.volume_score * 0.25
            + analysis.hl_score * 0.15
            + analysis.limit_score * 0.20
        )

        # Determine signal
        analysis.signal, analysis.signal_strength = self._determine_signal(current, analysis)

        # Calculate trend
        analysis.trend, analysis.ad_5day_trend = self._calculate_trend()

        # Check special conditions
        analysis.is_breadth_thrust = self._check_breadth_thrust(current)
        analysis.is_oversold_bounce = self._check_oversold_bounce(current)
        analysis.is_distribution = self._check_distribution(current)

        # Calculate trading implications
        analysis.entry_adjustment = self._calculate_entry_adjustment(analysis)
        analysis.position_multiplier = self._calculate_position_multiplier(analysis)

        # Add reasons
        analysis.reasons = self._generate_reasons(analysis)

        return analysis

    def _calculate_ad_score(self, data: MarketBreadthData) -> float:
        """Calculate A/D score (-1 to 1)."""
        # Use advance percentage
        pct = data.advance_percent
        # Map 0-1 to -1 to 1, with 0.5 being neutral
        return (pct - 0.5) * 2

    def _calculate_volume_score(self, data: MarketBreadthData) -> float:
        """Calculate volume A/D score (-1 to 1)."""
        ratio = data.volume_ad_ratio
        # Log transform for better scaling
        if ratio >= 1:
            return min(1, np.log10(ratio) / 0.5)
        else:
            return max(-1, -np.log10(1 / ratio) / 0.5)

    def _calculate_hl_score(self, data: MarketBreadthData) -> float:
        """Calculate high/low score (-1 to 1)."""
        ratio = data.high_low_ratio
        if ratio >= 1:
            return min(1, np.log10(ratio + 1) / 0.5)
        else:
            return max(-1, -np.log10(1 / ratio + 1) / 0.5)

    def _determine_signal(
        self,
        data: MarketBreadthData,
        analysis: BreadthAnalysis,
    ) -> Tuple[BreadthSignal, float]:
        """Determine breadth signal and strength."""
        ad_ratio = data.ad_ratio
        advance_pct = data.advance_percent
        score = analysis.overall_score

        # Check for breadth thrust
        if advance_pct >= THRUST_RATIO_THRESHOLD:
            return BreadthSignal.BREADTH_THRUST_UP, 1.0
        elif advance_pct <= (1 - THRUST_RATIO_THRESHOLD):
            return BreadthSignal.BREADTH_THRUST_DOWN, 1.0

        # Strong signals
        if ad_ratio >= 2.0:
            return BreadthSignal.STRONG_BULLISH, min(1.0, ad_ratio / 3)
        elif ad_ratio <= 0.5:
            return BreadthSignal.STRONG_BEARISH, min(1.0, (1 / ad_ratio) / 3)

        # Normal signals
        if ad_ratio >= AD_RATIO_BULLISH_THRESHOLD:
            return BreadthSignal.BULLISH, min(1.0, (ad_ratio - 1) / 1.5)
        elif ad_ratio <= AD_RATIO_BEARISH_THRESHOLD:
            return BreadthSignal.BEARISH, min(1.0, (1 - ad_ratio) / 0.67)

        # Neutral
        return BreadthSignal.NEUTRAL, 0.5

    def _calculate_trend(self) -> Tuple[str, float]:
        """Calculate breadth trend from history."""
        if len(self._history) < 2:
            return "neutral", 0.0

        # Get last 5 days (or available)
        recent = self._history[-5:]

        # Calculate cumulative A/D
        cumulative_ad = sum(d.ad_difference for d in recent)

        # Compare to earlier period
        if len(self._history) >= 10:
            earlier = self._history[-10:-5]
            earlier_ad = sum(d.ad_difference for d in earlier)

            if cumulative_ad > earlier_ad + 50:
                return "improving", cumulative_ad
            elif cumulative_ad < earlier_ad - 50:
                return "deteriorating", cumulative_ad

        return "stable", cumulative_ad

    def _check_breadth_thrust(self, data: MarketBreadthData) -> bool:
        """Check for breadth thrust signal."""
        return data.advance_percent >= THRUST_RATIO_THRESHOLD

    def _check_oversold_bounce(self, data: MarketBreadthData) -> bool:
        """Check for oversold bounce setup."""
        if len(self._history) < 3:
            return False

        # Previous days were very bearish
        prev_bearish = all(h.ad_ratio < AD_RATIO_BEARISH_THRESHOLD for h in self._history[-3:-1])

        # Current is turning bullish
        current_bullish = data.ad_ratio > 1.0

        return prev_bearish and current_bullish

    def _check_distribution(self, data: MarketBreadthData) -> bool:
        """Check for distribution pattern (high volume on down days)."""
        if data.advances > data.declines:
            return False

        # Check if volume is concentrated in declining stocks
        return data.volume_ad_ratio < 0.7

    def _calculate_entry_adjustment(self, analysis: BreadthAnalysis) -> float:
        """Calculate entry confidence adjustment."""
        adjustment = 0.0

        if analysis.signal == BreadthSignal.BREADTH_THRUST_UP:
            adjustment = 0.10
        elif analysis.signal == BreadthSignal.STRONG_BULLISH:
            adjustment = 0.07
        elif analysis.signal == BreadthSignal.BULLISH:
            adjustment = 0.03
        elif analysis.signal == BreadthSignal.BEARISH:
            adjustment = -0.05
        elif analysis.signal == BreadthSignal.STRONG_BEARISH:
            adjustment = -0.10
        elif analysis.signal == BreadthSignal.BREADTH_THRUST_DOWN:
            adjustment = -0.15

        # Trend adjustment
        if analysis.trend == "improving":
            adjustment += 0.02
        elif analysis.trend == "deteriorating":
            adjustment -= 0.02

        return adjustment

    def _calculate_position_multiplier(self, analysis: BreadthAnalysis) -> float:
        """Calculate position size multiplier."""
        if analysis.signal in (BreadthSignal.BREADTH_THRUST_UP, BreadthSignal.STRONG_BULLISH):
            return 1.15
        elif analysis.signal == BreadthSignal.BULLISH:
            return 1.05
        elif analysis.signal == BreadthSignal.BEARISH:
            return 0.85
        elif analysis.signal in (BreadthSignal.STRONG_BEARISH, BreadthSignal.BREADTH_THRUST_DOWN):
            return 0.70

        return 1.0

    def _generate_reasons(self, analysis: BreadthAnalysis) -> List[str]:
        """Generate human-readable reasons."""
        reasons = []
        current = analysis.current

        reasons.append(
            f"A/D Ratio: {current.ad_ratio:.2f} " f"({current.advances}/{current.declines})"
        )

        if current.ceiling_hits > 5:
            reasons.append(f"✅ {current.ceiling_hits} stocks at ceiling (+7%)")
        if current.floor_hits > 5:
            reasons.append(f"⚠️ {current.floor_hits} stocks at floor (-7%)")

        if analysis.is_breadth_thrust:
            reasons.append("🚀 Breadth thrust detected!")
        if analysis.is_oversold_bounce:
            reasons.append("📈 Oversold bounce setup")
        if analysis.is_distribution:
            reasons.append("⚠️ Distribution pattern detected")

        reasons.append(f"Trend: {analysis.trend} (5-day A/D: {analysis.ad_5day_trend:+.0f})")

        return reasons

    def check_breadth_for_entry(self) -> Dict:
        """
        Check if breadth supports entry.

        Returns:
            Dict with is_favorable, adjustment, reasons
        """
        analysis = self.get_analysis()

        is_favorable = analysis.signal not in (
            BreadthSignal.STRONG_BEARISH,
            BreadthSignal.BREADTH_THRUST_DOWN,
        )

        should_boost = analysis.signal in (
            BreadthSignal.BREADTH_THRUST_UP,
            BreadthSignal.STRONG_BULLISH,
        )

        return {
            "is_favorable": is_favorable,
            "should_boost": should_boost,
            "confidence_adjustment": analysis.entry_adjustment,
            "position_multiplier": analysis.position_multiplier,
            "signal": analysis.signal.value,
            "score": analysis.overall_score,
            "reasons": analysis.reasons,
        }

    def get_adjusted_score(self, use_cache: bool = True) -> float:
        """
        Get staleness-adjusted breadth score.

        Automatically reduces score weight when data is stale:
        - Fresh (<5 min): 100% weight
        - Slightly stale (5-15 min): 85% weight
        - Stale (15-30 min): 50% weight
        - Very stale (30-60 min): 20% weight
        - Expired (>60 min): 0% weight

        Args:
            use_cache: Use cached analysis

        Returns:
            Staleness-adjusted overall score
        """
        if use_cache and self._cache is not None:
            raw_score = self._cache.overall_score
        else:
            analysis = self.get_analysis(force_refresh=not use_cache)
            raw_score = analysis.overall_score

        return self._apply_staleness_weight(raw_score)

    def get_data_quality(self) -> Dict[str, Any]:
        """
        Get data quality status including staleness info.

        Returns:
            Dict with freshness, age, weight, and recommendations
        """
        staleness = self.get_staleness_status()

        recommendations = []
        if staleness["is_stale"]:
            recommendations.append("Consider force_refresh=True for fresh data")
        if staleness["is_expired"]:
            recommendations.append("Data too old - signals may be unreliable")

        return {
            **staleness,
            "recommendations": recommendations,
            "cache_available": self._cache is not None,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_analyzer_instance: Optional[MarketBreadthAnalyzer] = None
_lock = RLock()


def get_breadth_analyzer() -> MarketBreadthAnalyzer:
    """Get singleton analyzer instance."""
    global _analyzer_instance
    with _lock:
        if _analyzer_instance is None:
            _analyzer_instance = MarketBreadthAnalyzer()
        return _analyzer_instance


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("🧪 TESTING MARKET BREADTH ANALYZER")
    print("=" * 70 + "\n")

    analyzer = get_breadth_analyzer()

    # Test analysis
    print("1️⃣ Testing get_analysis()...")
    analysis = analyzer.get_analysis()
    print(f"   Signal: {analysis.signal.value}")
    print(f"   Overall Score: {analysis.overall_score:+.3f}")
    print(f"   Entry Adjustment: {analysis.entry_adjustment:+.3f}")
    print(f"   Reasons: {analysis.reasons[:3]}")

    # Test entry check
    print("\n2️⃣ Testing check_breadth_for_entry()...")
    entry_check = analyzer.check_breadth_for_entry()
    print(f"   Favorable: {entry_check['is_favorable']}")
    print(f"   Adjustment: {entry_check['confidence_adjustment']:+.3f}")

    print("\n" + "=" * 70)
    print("✅ Market Breadth Analyzer testing complete!")
    print("=" * 70)
