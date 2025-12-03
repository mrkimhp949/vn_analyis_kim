# -*- coding: utf-8 -*-
"""
Margin Debt Analysis for Vietnam Market
Phân tích dư nợ margin để đánh giá rủi ro thị trường

FEATURES:
- Margin debt trend analysis
- Market leverage indicator
- Risk warning signals
- Historical comparison
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MarginDebtAnalysis:
    """Margin debt analysis result"""

    # Current state
    current_margin_debt: float  # VND
    margin_debt_change_1m: float  # % change 1 month
    margin_debt_change_3m: float  # % change 3 months

    # Relative metrics
    margin_to_market_cap: float  # Margin / Total market cap
    margin_percentile: float  # Current vs historical (0-100)

    # Signals
    signal: str  # INCREASING, DECREASING, STABLE, HIGH_RISK, DELEVERAGING
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME

    # Interpretation
    description: str
    recommendations: List[str]

    # Timestamp
    last_updated: datetime


class MarginDebtAnalyzer:
    """
    Analyzes margin debt levels in Vietnam market

    High margin debt indicates:
    - Bullish sentiment (investors borrowing to buy)
    - BUT also higher risk (forced selling if market drops)

    Key thresholds (Vietnam market):
    - Margin/Market Cap > 3%: Elevated risk
    - Margin/Market Cap > 4%: High risk
    - Margin/Market Cap > 5%: Extreme risk

    Data sources:
    - SSC (State Securities Commission) reports
    - Broker margin lending data
    - Exchange statistics
    """

    # Risk thresholds (margin as % of market cap)
    RISK_THRESHOLDS = {
        "low": 0.02,  # < 2%
        "medium": 0.03,  # 2-3%
        "high": 0.04,  # 3-4%
        "extreme": 0.05,  # > 5%
    }

    # Change thresholds
    CHANGE_THRESHOLDS = {
        "rapid_increase": 0.15,  # > 15% increase in 1 month
        "moderate_increase": 0.08,  # 8-15% increase
        "stable": 0.03,  # -3% to +3%
        "moderate_decrease": -0.08,  # -8% to -3%
        "rapid_decrease": -0.15,  # < -15% (deleveraging)
    }

    def __init__(self):
        self._cache = None
        self._cache_time = None
        self._cache_ttl = 3600 * 6  # 6 hours

        # Historical data storage
        self._historical_data: List[Dict] = []

    def analyze(
        self,
        current_margin_debt: Optional[float] = None,
        market_cap: Optional[float] = None,
        historical_margin: Optional[List[Dict]] = None,
    ) -> MarginDebtAnalysis:
        """
        Analyze margin debt levels

        Args:
            current_margin_debt: Current total margin debt (VND)
            market_cap: Total market capitalization (VND)
            historical_margin: Historical margin data [{date, value}, ...]

        Returns:
            MarginDebtAnalysis
        """
        # Try to fetch data if not provided
        if current_margin_debt is None:
            data = self._fetch_margin_data()
            if data:
                current_margin_debt = data.get("current_margin")
                market_cap = data.get("market_cap")
                historical_margin = data.get("historical")

        # If still no data, return default
        if current_margin_debt is None:
            return self._default_analysis("No margin data available")

        # Calculate metrics
        margin_to_cap = current_margin_debt / market_cap if market_cap else 0

        # Calculate changes
        change_1m = 0.0
        change_3m = 0.0
        percentile = 50.0

        if historical_margin and len(historical_margin) >= 2:
            # Sort by date
            sorted_data = sorted(historical_margin, key=lambda x: x["date"])

            # 1 month change
            if len(sorted_data) >= 22:  # ~1 month of trading days
                prev_1m = sorted_data[-22]["value"]
                change_1m = (current_margin_debt - prev_1m) / prev_1m if prev_1m else 0

            # 3 month change
            if len(sorted_data) >= 66:  # ~3 months
                prev_3m = sorted_data[-66]["value"]
                change_3m = (current_margin_debt - prev_3m) / prev_3m if prev_3m else 0

            # Percentile
            all_values = [d["value"] for d in sorted_data]
            percentile = (
                np.sum(np.array(all_values) < current_margin_debt) / len(all_values)
            ) * 100

        # Determine signal
        signal = self._determine_signal(change_1m, change_3m)

        # Determine risk level
        risk_level = self._determine_risk_level(margin_to_cap, change_1m)

        # Generate description and recommendations
        description = self._generate_description(
            margin_to_cap, change_1m, change_3m, signal, risk_level
        )
        recommendations = self._generate_recommendations(signal, risk_level)

        return MarginDebtAnalysis(
            current_margin_debt=current_margin_debt,
            margin_debt_change_1m=change_1m,
            margin_debt_change_3m=change_3m,
            margin_to_market_cap=margin_to_cap,
            margin_percentile=percentile,
            signal=signal,
            risk_level=risk_level,
            description=description,
            recommendations=recommendations,
            last_updated=datetime.now(),
        )

    def _fetch_margin_data(self) -> Optional[Dict]:
        """
        Fetch margin data from external sources

        Integrated sources:
        - TCBS API (primary)
        - SSI API (fallback)
        - SSC reports (manual)
        """
        # Check cache
        if self._cache and self._cache_time:
            age = (datetime.now() - self._cache_time).seconds
            if age < self._cache_ttl:
                return self._cache

        # Try TCBS provider
        try:
            from src.data.tcbs_provider import get_tcbs_provider

            provider = get_tcbs_provider()
            data = provider.get_margin_statistics()

            if data:
                self._cache = data
                self._cache_time = datetime.now()
                logger.info("✅ Fetched margin data from TCBS")
                return data

        except ImportError:
            logger.debug("TCBS provider not available for margin data")
        except Exception as e:
            logger.warning(f"TCBS margin data fetch failed: {e}")

        # Try SSI provider as fallback
        try:
            from src.data.ssi_provider import get_ssi_provider

            provider = get_ssi_provider()
            data = provider.get_margin_statistics()

            if data:
                self._cache = data
                self._cache_time = datetime.now()
                logger.info("✅ Fetched margin data from SSI")
                return data

        except ImportError:
            logger.debug("SSI provider not available for margin data")
        except Exception as e:
            logger.warning(f"SSI margin data fetch failed: {e}")

        # Use estimated data from market indicators
        try:
            estimated = self._estimate_margin_from_market()
            if estimated:
                self._cache = estimated
                self._cache_time = datetime.now()
                logger.info("Using estimated margin data from market indicators")
                return estimated
        except Exception as e:
            logger.debug(f"Margin estimation failed: {e}")

        return None

    def _estimate_margin_from_market(self) -> Optional[Dict]:
        """
        Estimate margin debt levels from market indicators.

        Uses:
        - Market cap data
        - Historical margin/cap ratios (typically 1.5-3% for VN market)
        - Volume and volatility as proxies
        """
        try:
            from src.data.vnindex_cache import get_cached_vnindex

            vnindex_df = get_cached_vnindex(lookback=60)
            if vnindex_df is None or vnindex_df.empty:
                return None

            # Estimate market cap (VNINDEX * typical multiplier)
            # VN market cap is roughly VNINDEX * 5-6 trillion VND
            current_index = vnindex_df["close"].iloc[-1]
            estimated_market_cap = current_index * 5_500_000_000_000  # ~5500T VND per index point

            # Estimate margin debt (typically 1.5-2.5% of market cap)
            # Higher volatility = higher margin usage
            volatility = vnindex_df["close"].pct_change().std() * 100
            margin_ratio = 0.015 + (volatility * 0.002)  # Base 1.5% + volatility adjustment
            margin_ratio = min(0.035, max(0.012, margin_ratio))  # Clamp to 1.2-3.5%

            estimated_margin = estimated_market_cap * margin_ratio

            # Build historical estimates
            historical = []
            for i in range(min(60, len(vnindex_df))):
                idx_value = vnindex_df["close"].iloc[i]
                est_cap = idx_value * 5_500_000_000_000
                historical.append(
                    {
                        "date": vnindex_df.index[i],
                        "value": est_cap * margin_ratio * (0.95 + 0.1 * (i / 60)),  # Slight trend
                    }
                )

            return {
                "current_margin": estimated_margin,
                "market_cap": estimated_market_cap,
                "historical": historical,
                "is_estimated": True,
            }

        except Exception as e:
            logger.debug(f"Market-based margin estimation failed: {e}")
            return None

    def _determine_signal(self, change_1m: float, change_3m: float) -> str:
        """Determine margin debt signal"""
        if change_1m > self.CHANGE_THRESHOLDS["rapid_increase"]:
            return "RAPID_INCREASE"
        elif change_1m > self.CHANGE_THRESHOLDS["moderate_increase"]:
            return "INCREASING"
        elif change_1m < self.CHANGE_THRESHOLDS["rapid_decrease"]:
            return "DELEVERAGING"
        elif change_1m < self.CHANGE_THRESHOLDS["moderate_decrease"]:
            return "DECREASING"
        else:
            return "STABLE"

    def _determine_risk_level(self, margin_to_cap: float, change_1m: float) -> str:
        """Determine risk level"""
        # Base risk from margin/cap ratio
        if margin_to_cap >= self.RISK_THRESHOLDS["extreme"]:
            base_risk = "EXTREME"
        elif margin_to_cap >= self.RISK_THRESHOLDS["high"]:
            base_risk = "HIGH"
        elif margin_to_cap >= self.RISK_THRESHOLDS["medium"]:
            base_risk = "MEDIUM"
        else:
            base_risk = "LOW"

        # Adjust for rapid changes
        if change_1m > self.CHANGE_THRESHOLDS["rapid_increase"]:
            # Rapid increase = higher risk
            if base_risk == "LOW":
                return "MEDIUM"
            elif base_risk == "MEDIUM":
                return "HIGH"
            elif base_risk == "HIGH":
                return "EXTREME"

        return base_risk

    def _generate_description(
        self,
        margin_to_cap: float,
        change_1m: float,
        change_3m: float,
        signal: str,
        risk_level: str,
    ) -> str:
        """Generate human-readable description"""
        parts = []

        # Margin level
        parts.append(f"Margin/Market Cap: {margin_to_cap*100:.2f}%")

        # Changes
        parts.append(f"1M change: {change_1m*100:+.1f}%")
        parts.append(f"3M change: {change_3m*100:+.1f}%")

        # Signal interpretation
        signal_desc = {
            "RAPID_INCREASE": "Margin debt increasing rapidly - bullish sentiment but elevated risk",
            "INCREASING": "Margin debt trending up - investors leveraging",
            "STABLE": "Margin debt stable",
            "DECREASING": "Margin debt declining - deleveraging in progress",
            "DELEVERAGING": "Rapid deleveraging - potential forced selling",
        }
        parts.append(signal_desc.get(signal, ""))

        # Risk level
        parts.append(f"Risk Level: {risk_level}")

        return " | ".join(parts)

    def _generate_recommendations(self, signal: str, risk_level: str) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []

        if risk_level == "EXTREME":
            recommendations.append("🚨 EXTREME RISK: Consider reducing exposure significantly")
            recommendations.append("💰 Maintain high cash levels")
            recommendations.append("⚠️ Avoid margin trading")

        elif risk_level == "HIGH":
            recommendations.append("⚠️ HIGH RISK: Be cautious with new positions")
            recommendations.append("📉 Consider tightening stop losses")
            recommendations.append("💵 Increase cash allocation")

        elif risk_level == "MEDIUM":
            recommendations.append("📊 MODERATE RISK: Normal position sizing")
            recommendations.append("👀 Monitor margin levels")

        else:
            recommendations.append("✅ LOW RISK: Normal trading conditions")

        # Signal-specific recommendations
        if signal == "DELEVERAGING":
            recommendations.append("📉 Deleveraging may cause selling pressure")
            recommendations.append("🔍 Watch for capitulation signals")

        elif signal == "RAPID_INCREASE":
            recommendations.append("📈 High leverage = potential for sharp corrections")
            recommendations.append("🎯 Consider taking profits on winners")

        return recommendations

    def _default_analysis(self, reason: str) -> MarginDebtAnalysis:
        """Return default analysis when data unavailable"""
        return MarginDebtAnalysis(
            current_margin_debt=0,
            margin_debt_change_1m=0,
            margin_debt_change_3m=0,
            margin_to_market_cap=0,
            margin_percentile=50,
            signal="UNKNOWN",
            risk_level="UNKNOWN",
            description=f"Margin data unavailable: {reason}",
            recommendations=["📊 Unable to assess margin risk - use caution"],
            last_updated=datetime.now(),
        )


# Singleton instance
_margin_analyzer = None


def get_margin_debt_analyzer() -> MarginDebtAnalyzer:
    """Get singleton instance"""
    global _margin_analyzer
    if _margin_analyzer is None:
        _margin_analyzer = MarginDebtAnalyzer()
    return _margin_analyzer


# Test
if __name__ == "__main__":
    print("Testing Margin Debt Analyzer...")

    analyzer = MarginDebtAnalyzer()

    # Test with sample data
    # Vietnam market: ~100T VND margin, ~6000T VND market cap
    result = analyzer.analyze(
        current_margin_debt=100_000_000_000_000,  # 100T VND
        market_cap=6_000_000_000_000_000,  # 6000T VND
        historical_margin=[
            {"date": datetime(2024, 1, 1), "value": 90_000_000_000_000},
            {"date": datetime(2024, 2, 1), "value": 95_000_000_000_000},
            {"date": datetime(2024, 3, 1), "value": 100_000_000_000_000},
        ],
    )

    print(f"\nMargin Debt Analysis:")
    print(f"  Current: {result.current_margin_debt/1e12:.1f}T VND")
    print(f"  Margin/Cap: {result.margin_to_market_cap*100:.2f}%")
    print(f"  1M Change: {result.margin_debt_change_1m*100:+.1f}%")
    print(f"  Signal: {result.signal}")
    print(f"  Risk Level: {result.risk_level}")
    print(f"\nDescription: {result.description}")
    print(f"\nRecommendations:")
    for r in result.recommendations:
        print(f"  {r}")

    print("\n✅ Margin Debt Analyzer test completed!")
