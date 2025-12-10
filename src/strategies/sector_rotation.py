# -*- coding: utf-8 -*-
"""
Sector Rotation Strategy for Vietnam Stock Market

IMPROVED v10.0: Complete sector rotation analysis.

Features:
- Sector performance tracking
- Rotation signal generation
- Sector correlation analysis
- Economic cycle mapping
- VN30 sector allocation

Vietnam Market Sectors (VN30):
- BANKING: ACB, BID, CTG, HDB, MBB, SHB, SSB, STB, TCB, TPB, VCB, VIB, VPB
- REAL_ESTATE: BCM, VHM, VIC, VRE
- CONSUMER: MWG, MSN, SAB, VNM
- ENERGY: GAS, PLX
- UTILITIES: POW
- INDUSTRIAL: GVR, HPG
- INSURANCE: BVH
- TECHNOLOGY: FPT
- SECURITIES: SSI
- AVIATION: VJC

Author: Trading Bot Team
Version: 10.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class EconomicCycle(Enum):
    """Economic cycle phases"""

    EARLY_RECOVERY = "EARLY_RECOVERY"  # Coming out of recession
    MID_CYCLE = "MID_CYCLE"  # Expansion phase
    LATE_CYCLE = "LATE_CYCLE"  # Peak approaching
    RECESSION = "RECESSION"  # Contraction


class SectorStrength(Enum):
    """Sector strength classification"""

    STRONG = "STRONG"  # Top performing
    NEUTRAL = "NEUTRAL"  # Average performance
    WEAK = "WEAK"  # Underperforming


# VN30 Sector Mapping (from vietnam_market.py)
VN30_SECTORS = {
    # Banking - 13 stocks (largest sector)
    "ACB": "BANKING",
    "BID": "BANKING",
    "CTG": "BANKING",
    "HDB": "BANKING",
    "MBB": "BANKING",
    "SHB": "BANKING",
    "SSB": "BANKING",
    "STB": "BANKING",
    "TCB": "BANKING",
    "TPB": "BANKING",
    "VCB": "BANKING",
    "VIB": "BANKING",
    "VPB": "BANKING",
    # Real Estate - 4 stocks
    "BCM": "REAL_ESTATE",
    "VHM": "REAL_ESTATE",
    "VIC": "REAL_ESTATE",
    "VRE": "REAL_ESTATE",
    # Consumer - 4 stocks
    "MWG": "CONSUMER",
    "MSN": "CONSUMER",
    "SAB": "CONSUMER",
    "VNM": "CONSUMER",
    # Energy - 2 stocks
    "GAS": "ENERGY",
    "PLX": "ENERGY",
    # Utilities - 1 stock
    "POW": "UTILITIES",
    # Industrial - 2 stocks
    "GVR": "INDUSTRIAL",
    "HPG": "INDUSTRIAL",
    # Insurance - 1 stock
    "BVH": "INSURANCE",
    # Technology - 1 stock
    "FPT": "TECHNOLOGY",
    # Securities - 1 stock
    "SSI": "SECURITIES",
    # Aviation - 1 stock
    "VJC": "AVIATION",
}

# Sector rotation based on economic cycle
# Reference: Typical Vietnam market sector rotation patterns
SECTOR_CYCLE_PREFERENCE = {
    EconomicCycle.EARLY_RECOVERY: {
        "preferred": ["TECHNOLOGY", "CONSUMER", "SECURITIES", "REAL_ESTATE"],
        "neutral": ["BANKING", "INDUSTRIAL"],
        "avoid": ["UTILITIES", "ENERGY"],
    },
    EconomicCycle.MID_CYCLE: {
        "preferred": ["BANKING", "INDUSTRIAL", "TECHNOLOGY"],
        "neutral": ["CONSUMER", "REAL_ESTATE", "ENERGY"],
        "avoid": ["UTILITIES"],
    },
    EconomicCycle.LATE_CYCLE: {
        "preferred": ["ENERGY", "UTILITIES", "BANKING"],
        "neutral": ["CONSUMER", "INSURANCE"],
        "avoid": ["TECHNOLOGY", "REAL_ESTATE", "SECURITIES"],
    },
    EconomicCycle.RECESSION: {
        "preferred": ["UTILITIES", "CONSUMER", "INSURANCE"],
        "neutral": ["BANKING"],
        "avoid": ["REAL_ESTATE", "INDUSTRIAL", "AVIATION", "SECURITIES"],
    },
}


@dataclass
class SectorPerformance:
    """Sector performance metrics"""

    sector: str
    return_1d: float = 0.0  # 1-day return
    return_5d: float = 0.0  # 5-day return
    return_20d: float = 0.0  # 20-day (1 month) return
    return_60d: float = 0.0  # 60-day (3 months) return
    volatility_20d: float = 0.0  # 20-day volatility
    relative_strength: float = 0.0  # vs VN30 index
    momentum_score: float = 0.0  # Combined momentum
    strength: SectorStrength = SectorStrength.NEUTRAL
    rank: int = 0  # Rank among all sectors


@dataclass
class SectorRotationSignal:
    """Sector rotation recommendation"""

    timestamp: datetime
    current_cycle: EconomicCycle
    overweight_sectors: List[str]
    underweight_sectors: List[str]
    neutral_sectors: List[str]
    top_picks: List[str]  # Top stock picks
    rotation_strength: float  # 0-100 confidence
    reasoning: List[str]


class SectorRotationAnalyzer:
    """
    Sector Rotation Strategy Analyzer for Vietnam Market.

    IMPROVED v10.0: Complete sector rotation analysis.

    Features:
    - Track sector performance
    - Generate rotation signals
    - Identify sector leaders/laggards
    - Map to economic cycle
    - Recommend sector allocation

    Usage:
        analyzer = SectorRotationAnalyzer()

        # Analyze sector performance
        performance = analyzer.analyze_sectors(price_data)

        # Get rotation signal
        signal = analyzer.get_rotation_signal(price_data)

        # Get sector allocation
        allocation = analyzer.get_recommended_allocation()
    """

    # Sector weights in VN30 (approximate)
    SECTOR_WEIGHTS = {
        "BANKING": 0.40,  # ~40% of VN30
        "REAL_ESTATE": 0.15,
        "CONSUMER": 0.12,
        "TECHNOLOGY": 0.08,
        "INDUSTRIAL": 0.08,
        "ENERGY": 0.06,
        "SECURITIES": 0.04,
        "UTILITIES": 0.03,
        "INSURANCE": 0.02,
        "AVIATION": 0.02,
    }

    def __init__(
        self,
        momentum_periods: List[int] = None,
        min_stocks_per_sector: int = 1,
    ):
        """
        Initialize Sector Rotation Analyzer.

        Args:
            momentum_periods: Periods for momentum calculation [5, 20, 60]
            min_stocks_per_sector: Minimum stocks needed for sector analysis
        """
        self.momentum_periods = momentum_periods or [5, 20, 60]
        self.min_stocks_per_sector = min_stocks_per_sector
        self._sector_cache: Dict[str, SectorPerformance] = {}
        self._last_analysis_time: Optional[datetime] = None

    def get_sector_symbols(self, sector: str) -> List[str]:
        """Get all symbols in a sector."""
        return [symbol for symbol, sec in VN30_SECTORS.items() if sec == sector]

    def get_all_sectors(self) -> List[str]:
        """Get list of all sectors."""
        return list(set(VN30_SECTORS.values()))

    def calculate_sector_return(
        self,
        price_data: Dict[str, pd.DataFrame],
        sector: str,
        days: int,
    ) -> float:
        """
        Calculate sector return over N days.

        Args:
            price_data: Dict of {symbol: DataFrame with 'close' column}
            sector: Sector name
            days: Number of days

        Returns:
            Average return of sector stocks
        """
        symbols = self.get_sector_symbols(sector)
        returns = []

        for symbol in symbols:
            if symbol in price_data:
                df = price_data[symbol]
                if len(df) >= days + 1:
                    current_price = df["close"].iloc[-1]
                    past_price = df["close"].iloc[-days - 1]
                    if past_price > 0:
                        ret = (current_price - past_price) / past_price
                        returns.append(ret)

        if returns:
            return np.mean(returns)
        return 0.0

    def calculate_sector_volatility(
        self,
        price_data: Dict[str, pd.DataFrame],
        sector: str,
        days: int = 20,
    ) -> float:
        """Calculate sector volatility over N days."""
        symbols = self.get_sector_symbols(sector)
        volatilities = []

        for symbol in symbols:
            if symbol in price_data:
                df = price_data[symbol]
                if len(df) >= days:
                    returns = df["close"].pct_change().tail(days)
                    vol = returns.std() * np.sqrt(252)  # Annualized
                    if not np.isnan(vol):
                        volatilities.append(vol)

        if volatilities:
            return np.mean(volatilities)
        return 0.0

    def analyze_sectors(
        self,
        price_data: Dict[str, pd.DataFrame],
        vn30_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, SectorPerformance]:
        """
        Analyze all sectors and return performance metrics.

        Args:
            price_data: Dict of {symbol: DataFrame}
            vn30_data: VN30 index data for relative strength

        Returns:
            Dict of {sector: SectorPerformance}
        """
        results = {}
        vn30_return_20d = 0.0

        # Calculate VN30 benchmark return
        if vn30_data is not None and len(vn30_data) >= 21:
            vn30_return_20d = vn30_data["close"].iloc[-1] / vn30_data["close"].iloc[-21] - 1

        for sector in self.get_all_sectors():
            # Calculate returns
            return_1d = self.calculate_sector_return(price_data, sector, 1)
            return_5d = self.calculate_sector_return(price_data, sector, 5)
            return_20d = self.calculate_sector_return(price_data, sector, 20)
            return_60d = self.calculate_sector_return(price_data, sector, 60)

            # Calculate volatility
            volatility = self.calculate_sector_volatility(price_data, sector, 20)

            # Calculate relative strength (vs VN30)
            relative_strength = return_20d - vn30_return_20d

            # Calculate momentum score (weighted average of returns)
            momentum_score = (return_5d * 0.2 + return_20d * 0.4 + return_60d * 0.4) * 100

            results[sector] = SectorPerformance(
                sector=sector,
                return_1d=return_1d * 100,
                return_5d=return_5d * 100,
                return_20d=return_20d * 100,
                return_60d=return_60d * 100,
                volatility_20d=volatility * 100,
                relative_strength=relative_strength * 100,
                momentum_score=momentum_score,
            )

        # Rank sectors by momentum
        sorted_sectors = sorted(
            results.values(),
            key=lambda x: x.momentum_score,
            reverse=True,
        )

        for i, sector_perf in enumerate(sorted_sectors):
            sector_perf.rank = i + 1

            # Classify strength
            if i < len(sorted_sectors) * 0.33:
                sector_perf.strength = SectorStrength.STRONG
            elif i < len(sorted_sectors) * 0.67:
                sector_perf.strength = SectorStrength.NEUTRAL
            else:
                sector_perf.strength = SectorStrength.WEAK

        self._sector_cache = results
        self._last_analysis_time = datetime.now()

        return results

    def detect_economic_cycle(
        self,
        vn30_data: Optional[pd.DataFrame] = None,
        gdp_growth: Optional[float] = None,
        inflation: Optional[float] = None,
    ) -> EconomicCycle:
        """
        Detect current economic cycle phase.

        Simple heuristic based on:
        - VN30 trend (200-day MA)
        - VN30 momentum
        - GDP growth (if available)
        - Inflation (if available)

        Args:
            vn30_data: VN30 index data
            gdp_growth: GDP growth rate (optional)
            inflation: Inflation rate (optional)

        Returns:
            Detected economic cycle
        """
        score = 0  # Higher = more bullish cycle

        if vn30_data is not None and len(vn30_data) >= 200:
            # Trend: Price vs 200-day MA
            current_price = vn30_data["close"].iloc[-1]
            ma200 = vn30_data["close"].rolling(200).mean().iloc[-1]

            if current_price > ma200 * 1.05:
                score += 2  # Well above MA
            elif current_price > ma200:
                score += 1  # Above MA
            elif current_price < ma200 * 0.95:
                score -= 2  # Well below MA
            else:
                score -= 1  # Below MA

            # Momentum: 60-day return
            return_60d = (
                current_price / vn30_data["close"].iloc[-61] - 1 if len(vn30_data) >= 61 else 0
            )

            if return_60d > 0.15:
                score += 2  # Strong momentum
            elif return_60d > 0.05:
                score += 1  # Positive momentum
            elif return_60d < -0.15:
                score -= 2  # Strong negative momentum
            elif return_60d < -0.05:
                score -= 1  # Negative momentum

        # Economic indicators (if available)
        if gdp_growth is not None:
            if gdp_growth > 7.0:
                score += 1  # Strong GDP (Vietnam target ~6.5%)
            elif gdp_growth < 5.0:
                score -= 1  # Weak GDP

        if inflation is not None:
            if inflation > 5.0:
                score -= 1  # High inflation (bad for growth)
            elif inflation < 2.0:
                score += 1  # Low inflation (room for stimulus)

        # Map score to cycle
        if score >= 3:
            return EconomicCycle.MID_CYCLE
        elif score >= 1:
            return EconomicCycle.LATE_CYCLE
        elif score >= -1:
            return EconomicCycle.EARLY_RECOVERY
        else:
            return EconomicCycle.RECESSION

    def get_rotation_signal(
        self,
        price_data: Dict[str, pd.DataFrame],
        vn30_data: Optional[pd.DataFrame] = None,
    ) -> SectorRotationSignal:
        """
        Generate sector rotation signal.

        Args:
            price_data: Stock price data
            vn30_data: VN30 index data

        Returns:
            SectorRotationSignal with recommendations
        """
        # Analyze sectors
        sector_perf = self.analyze_sectors(price_data, vn30_data)

        # Detect cycle
        cycle = self.detect_economic_cycle(vn30_data)

        # Get cycle preferences
        cycle_prefs = SECTOR_CYCLE_PREFERENCE[cycle]

        # Combine sector strength with cycle preference
        overweight = []
        underweight = []
        neutral = []
        reasoning = []

        for sector, perf in sector_perf.items():
            # Sector in preferred list for current cycle
            is_preferred = sector in cycle_prefs["preferred"]
            is_avoid = sector in cycle_prefs["avoid"]

            # Strong sector in preferred cycle phase
            if perf.strength == SectorStrength.STRONG and is_preferred:
                overweight.append(sector)
                reasoning.append(f"✅ {sector}: Strong momentum + favorable cycle")

            # Strong sector but wrong cycle
            elif perf.strength == SectorStrength.STRONG and is_avoid:
                neutral.append(sector)
                reasoning.append(f"⚠️ {sector}: Strong momentum but late-cycle risk")

            # Weak sector in avoid list
            elif perf.strength == SectorStrength.WEAK and is_avoid:
                underweight.append(sector)
                reasoning.append(f"❌ {sector}: Weak momentum + unfavorable cycle")

            # Preferred sector with any momentum
            elif is_preferred and perf.strength != SectorStrength.WEAK:
                overweight.append(sector)
                reasoning.append(f"✅ {sector}: Favorable cycle position")

            # Avoid sector with any weakness
            elif is_avoid:
                underweight.append(sector)
                reasoning.append(f"❌ {sector}: Cycle headwinds")

            else:
                neutral.append(sector)

        # Select top picks from overweight sectors
        top_picks = []
        for sector in overweight[:3]:  # Top 3 overweight sectors
            symbols = self.get_sector_symbols(sector)
            # Pick symbols that are in price_data
            available_symbols = [s for s in symbols if s in price_data]
            if available_symbols:
                top_picks.extend(available_symbols[:2])  # Top 2 from each

        # Calculate rotation strength (confidence)
        # Higher if sector performance aligns with cycle expectations
        aligned_count = len(overweight) + len(underweight)
        rotation_strength = min(100, aligned_count * 15 + 40)

        return SectorRotationSignal(
            timestamp=datetime.now(),
            current_cycle=cycle,
            overweight_sectors=overweight,
            underweight_sectors=underweight,
            neutral_sectors=neutral,
            top_picks=top_picks[:5],  # Max 5 picks
            rotation_strength=rotation_strength,
            reasoning=reasoning,
        )

    def get_recommended_allocation(
        self,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        vn30_data: Optional[pd.DataFrame] = None,
        total_allocation: float = 1.0,
    ) -> Dict[str, float]:
        """
        Get recommended sector allocation.

        Args:
            price_data: Stock price data (optional, uses cache if available)
            vn30_data: VN30 index data
            total_allocation: Total allocation (1.0 = 100%)

        Returns:
            Dict of {sector: allocation_weight}
        """
        # Get rotation signal
        if price_data:
            signal = self.get_rotation_signal(price_data, vn30_data)
        else:
            # Use cache
            signal = SectorRotationSignal(
                timestamp=datetime.now(),
                current_cycle=EconomicCycle.MID_CYCLE,
                overweight_sectors=list(self._sector_cache.keys())[:3],
                underweight_sectors=[],
                neutral_sectors=[],
                top_picks=[],
                rotation_strength=50,
                reasoning=[],
            )

        # Start with base weights
        allocation = self.SECTOR_WEIGHTS.copy()

        # Adjust based on signal
        overweight_boost = 0.5  # 50% boost
        underweight_cut = 0.5  # 50% cut

        for sector in signal.overweight_sectors:
            if sector in allocation:
                allocation[sector] *= 1 + overweight_boost

        for sector in signal.underweight_sectors:
            if sector in allocation:
                allocation[sector] *= 1 - underweight_cut

        # Normalize to total_allocation
        total = sum(allocation.values())
        if total > 0:
            for sector in allocation:
                allocation[sector] = (allocation[sector] / total) * total_allocation

        return allocation

    def get_sector_report(
        self,
        price_data: Dict[str, pd.DataFrame],
        vn30_data: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate formatted sector rotation report.

        Args:
            price_data: Stock price data
            vn30_data: VN30 index data

        Returns:
            Formatted string report
        """
        # Analyze and get signal
        sector_perf = self.analyze_sectors(price_data, vn30_data)
        signal = self.get_rotation_signal(price_data, vn30_data)
        allocation = self.get_recommended_allocation(price_data, vn30_data)

        lines = [
            "=" * 70,
            "📊 VIETNAM SECTOR ROTATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 70,
            "",
            f"🔄 Economic Cycle: {signal.current_cycle.value}",
            f"📈 Rotation Strength: {signal.rotation_strength:.0f}/100",
            "",
            "-" * 70,
            "SECTOR PERFORMANCE (Sorted by Momentum):",
            "-" * 70,
            f"{'Sector':<15} {'1D':>8} {'5D':>8} {'20D':>8} {'60D':>8} {'RS':>8} {'Rank':>6}",
            "-" * 70,
        ]

        # Sort by rank
        sorted_perf = sorted(sector_perf.values(), key=lambda x: x.rank)

        for perf in sorted_perf:
            emoji = (
                "🟢"
                if perf.strength == SectorStrength.STRONG
                else ("🟡" if perf.strength == SectorStrength.NEUTRAL else "🔴")
            )
            lines.append(
                f"{emoji} {perf.sector:<12} "
                f"{perf.return_1d:>7.2f}% "
                f"{perf.return_5d:>7.2f}% "
                f"{perf.return_20d:>7.2f}% "
                f"{perf.return_60d:>7.2f}% "
                f"{perf.relative_strength:>7.2f}% "
                f"#{perf.rank:>4}"
            )

        lines.extend(
            [
                "",
                "-" * 70,
                "ROTATION RECOMMENDATIONS:",
                "-" * 70,
            ]
        )

        if signal.overweight_sectors:
            lines.append(f"✅ OVERWEIGHT: {', '.join(signal.overweight_sectors)}")
        if signal.neutral_sectors:
            lines.append(f"➖ NEUTRAL: {', '.join(signal.neutral_sectors)}")
        if signal.underweight_sectors:
            lines.append(f"❌ UNDERWEIGHT: {', '.join(signal.underweight_sectors)}")

        if signal.top_picks:
            lines.extend(
                [
                    "",
                    f"⭐ TOP PICKS: {', '.join(signal.top_picks)}",
                ]
            )

        lines.extend(
            [
                "",
                "-" * 70,
                "RECOMMENDED ALLOCATION:",
                "-" * 70,
            ]
        )

        sorted_allocation = sorted(
            allocation.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for sector, weight in sorted_allocation:
            bar = "█" * int(weight * 50)
            lines.append(f"  {sector:<15} {weight*100:>5.1f}% {bar}")

        lines.extend(
            [
                "",
                "-" * 70,
                "REASONING:",
                "-" * 70,
            ]
        )

        for reason in signal.reasoning[:10]:  # Max 10 reasons
            lines.append(f"  {reason}")

        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_sector_analyzer: Optional[SectorRotationAnalyzer] = None


def get_sector_analyzer() -> SectorRotationAnalyzer:
    """Get singleton SectorRotationAnalyzer instance."""
    global _sector_analyzer
    if _sector_analyzer is None:
        _sector_analyzer = SectorRotationAnalyzer()
    return _sector_analyzer


def get_sector_for_symbol(symbol: str) -> str:
    """Get sector for a symbol."""
    return VN30_SECTORS.get(symbol.upper(), "UNKNOWN")


def get_symbols_in_sector(sector: str) -> List[str]:
    """Get all symbols in a sector."""
    return [symbol for symbol, sec in VN30_SECTORS.items() if sec == sector.upper()]


def is_sector_favorable(sector: str, cycle: EconomicCycle) -> bool:
    """Check if sector is favorable in current cycle."""
    prefs = SECTOR_CYCLE_PREFERENCE.get(cycle, {})
    return sector in prefs.get("preferred", [])
