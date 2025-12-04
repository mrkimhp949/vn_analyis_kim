# -*- coding: utf-8 -*-
"""
Margin Debt Tracking for Vietnam Stock Market

Tracks market-wide margin debt levels to assess:
- Market leverage risk
- Potential margin call cascades
- Force sell pressure

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class MarginRiskLevel(Enum):
    """Margin risk levels"""

    LOW = "LOW"  # < 2% of market cap
    MODERATE = "MODERATE"  # 2-3% of market cap
    HIGH = "HIGH"  # 3-4% of market cap
    CRITICAL = "CRITICAL"  # > 4% of market cap


@dataclass
class MarginDebtData:
    """Margin debt analysis data"""

    date: datetime
    total_margin_debt: float  # Total margin debt in VND
    market_cap: float  # Total market cap in VND
    margin_ratio: float  # Margin debt / Market cap

    # Risk assessment
    risk_level: MarginRiskLevel
    risk_score: float  # 0-100 (higher = more risk)

    # Trend
    change_1d: float = 0.0  # 1-day change %
    change_5d: float = 0.0  # 5-day change %
    change_20d: float = 0.0  # 20-day change %

    # Thresholds
    margin_call_risk: bool = False
    force_sell_risk: bool = False

    # Recommendations
    position_adjustment: float = 1.0  # Multiplier for position sizing
    warnings: List[str] = field(default_factory=list)


class MarginDebtTracker:
    """
    Tracks and analyzes market-wide margin debt

    Vietnam Market Context:
    - Margin debt typically 1.5-3% of market cap
    - High margin = increased volatility risk
    - Margin calls can trigger cascading sells
    - Monitor during market corrections

    Risk Thresholds:
    - LOW: < 2% margin ratio
    - MODERATE: 2-3% margin ratio
    - HIGH: 3-4% margin ratio
    - CRITICAL: > 4% margin ratio
    """

    # Risk thresholds (margin debt / market cap)
    THRESHOLDS = {
        MarginRiskLevel.LOW: 0.02,
        MarginRiskLevel.MODERATE: 0.03,
        MarginRiskLevel.HIGH: 0.04,
        MarginRiskLevel.CRITICAL: 0.05,
    }

    def __init__(self, cache_ttl: int = 3600):
        self._cache: Optional[MarginDebtData] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = cache_ttl
        self._historical: List[Dict] = []

    def analyze(self, force_refresh: bool = False) -> MarginDebtData:
        """
        Analyze current margin debt situation

        Args:
            force_refresh: Bypass cache

        Returns:
            MarginDebtData with analysis
        """
        # Check cache
        if not force_refresh and self._is_cache_valid():
            return self._cache

        # Fetch data
        margin_data = self._fetch_margin_data()

        if margin_data is None:
            return self._default_result()

        # Calculate metrics
        result = self._calculate_metrics(margin_data)

        # Update cache
        self._cache = result
        self._cache_time = datetime.now()

        return result

    def _fetch_margin_data(self) -> Optional[Dict]:
        """Fetch margin debt data from sources"""
        # Try TCBS provider
        try:
            from src.data.tcbs_provider import get_tcbs_provider

            provider = get_tcbs_provider()
            data = provider.get_margin_statistics()

            if data:
                logger.info("✅ Got margin data from TCBS")
                return data
        except Exception as e:
            logger.debug(f"TCBS margin data failed: {e}")

        # Estimate from market data
        return self._estimate_margin_data()

    def _estimate_margin_data(self) -> Optional[Dict]:
        """Estimate margin debt from market indicators"""
        try:
            from src.data.vnindex_cache import get_cached_vnindex

            vnindex = get_cached_vnindex(lookback=60)
            if vnindex is None or len(vnindex) < 20:
                return None

            # Estimate market cap (VNINDEX * multiplier)
            current_index = vnindex["close"].iloc[-1]
            market_cap = current_index * 5_500_000_000_000  # ~5500T per point

            # Estimate margin based on volatility and trend
            volatility = vnindex["close"].pct_change().std() * 100
            trend = (vnindex["close"].iloc[-1] / vnindex["close"].iloc[-20] - 1) * 100

            # Base margin ratio 1.8%, adjust for conditions
            base_ratio = 0.018

            # Higher volatility = higher margin usage
            vol_adjustment = volatility * 0.002

            # Uptrend = higher margin, downtrend = lower (margin calls)
            trend_adjustment = trend * 0.0005

            margin_ratio = base_ratio + vol_adjustment + trend_adjustment
            margin_ratio = max(0.01, min(0.05, margin_ratio))

            margin_debt = market_cap * margin_ratio

            return {
                "current_margin": margin_debt,
                "market_cap": market_cap,
                "margin_ratio": margin_ratio,
                "is_estimated": True,
            }

        except Exception as e:
            logger.warning(f"Margin estimation failed: {e}")
            return None

    def _calculate_metrics(self, data: Dict) -> MarginDebtData:
        """Calculate margin debt metrics"""
        margin_debt = data.get("current_margin", 0)
        market_cap = data.get("market_cap", 1)
        margin_ratio = data.get("margin_ratio", margin_debt / market_cap)

        # Determine risk level
        risk_level = MarginRiskLevel.LOW
        for level, threshold in sorted(self.THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if margin_ratio >= threshold:
                risk_level = level
                break

        # Calculate risk score (0-100)
        risk_score = min(100, (margin_ratio / 0.05) * 100)

        # Assess specific risks
        margin_call_risk = margin_ratio > 0.035
        force_sell_risk = margin_ratio > 0.045

        # Position adjustment
        if risk_level == MarginRiskLevel.CRITICAL:
            position_adjustment = 0.5
        elif risk_level == MarginRiskLevel.HIGH:
            position_adjustment = 0.7
        elif risk_level == MarginRiskLevel.MODERATE:
            position_adjustment = 0.85
        else:
            position_adjustment = 1.0

        # Warnings
        warnings = []
        if risk_level == MarginRiskLevel.CRITICAL:
            warnings.append("🚨 CRITICAL: Margin debt at dangerous levels")
            warnings.append("High risk of margin call cascade")
        elif risk_level == MarginRiskLevel.HIGH:
            warnings.append("⚠️ HIGH: Elevated margin debt")
            warnings.append("Monitor for potential margin calls")
        elif margin_call_risk:
            warnings.append("⚠️ Margin call risk elevated")

        return MarginDebtData(
            date=datetime.now(),
            total_margin_debt=margin_debt,
            market_cap=market_cap,
            margin_ratio=margin_ratio,
            risk_level=risk_level,
            risk_score=risk_score,
            margin_call_risk=margin_call_risk,
            force_sell_risk=force_sell_risk,
            position_adjustment=position_adjustment,
            warnings=warnings,
        )

    def _is_cache_valid(self) -> bool:
        """Check if cache is valid"""
        if self._cache is None or self._cache_time is None:
            return False
        age = (datetime.now() - self._cache_time).total_seconds()
        return age < self._cache_ttl

    def _default_result(self) -> MarginDebtData:
        """Return default result"""
        return MarginDebtData(
            date=datetime.now(),
            total_margin_debt=0,
            market_cap=0,
            margin_ratio=0.02,
            risk_level=MarginRiskLevel.MODERATE,
            risk_score=40,
            position_adjustment=1.0,
            warnings=["⚠️ Margin data unavailable - using defaults"],
        )

    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on margin risk"""
        data = self.analyze()
        return data.position_adjustment


# Singleton
_margin_tracker: Optional[MarginDebtTracker] = None


def get_margin_tracker() -> MarginDebtTracker:
    """Get singleton margin tracker"""
    global _margin_tracker
    if _margin_tracker is None:
        _margin_tracker = MarginDebtTracker()
    return _margin_tracker


def get_margin_risk_multiplier() -> float:
    """Get position multiplier based on margin risk"""
    return get_margin_tracker().get_position_multiplier()
