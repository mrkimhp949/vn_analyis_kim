# -*- coding: utf-8 -*-
"""
Sector Rotation Strategy for Vietnam Market

Implements sector rotation trading strategy based on:
1. Sector momentum and relative strength
2. Economic cycle positioning
3. Foreign flow into sectors
4. Sector correlation analysis

Vietnam Market Sectors:
- Banking (ngân hàng)
- Real Estate (bất động sản)
- Consumer (tiêu dùng)
- Energy (năng lượng)
- Industrial (công nghiệp)
- Technology (công nghệ)
- Materials (vật liệu)
- Healthcare (y tế)
- Utilities (tiện ích)

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS & ENUMS
# =============================================================================


class SectorPhase(Enum):
    """Sector cycle phase"""

    ACCUMULATION = "ACCUMULATION"  # Smart money buying
    MARKUP = "MARKUP"  # Uptrend, momentum building
    DISTRIBUTION = "DISTRIBUTION"  # Smart money selling
    MARKDOWN = "MARKDOWN"  # Downtrend


class EconomicCycle(Enum):
    """Economic cycle phase"""

    EARLY_EXPANSION = "EARLY_EXPANSION"
    MID_EXPANSION = "MID_EXPANSION"
    LATE_EXPANSION = "LATE_EXPANSION"
    EARLY_CONTRACTION = "EARLY_CONTRACTION"
    LATE_CONTRACTION = "LATE_CONTRACTION"


class SectorSignal(Enum):
    """Sector trading signal"""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    SELL = "SELL"


# Vietnam market sector mappings
VN_SECTORS = {
    "BANKING": {
        "name": "Ngân hàng",
        "symbols": [
            "VCB",
            "BID",
            "CTG",
            "TCB",
            "MBB",
            "ACB",
            "VPB",
            "HDB",
            "TPB",
            "STB",
            "SHB",
            "VIB",
            "SSB",
            "MSB",
            "LPB",
            "ABB",
            "EIB",
            "OCB",
        ],
        "etf": None,  # No sector ETF yet
        "weight_in_vnindex": 0.35,  # ~35% weight
    },
    "REAL_ESTATE": {
        "name": "Bất động sản",
        "symbols": [
            "VIC",
            "VHM",
            "VRE",
            "NVL",
            "KDH",
            "DXG",
            "NLG",
            "PDR",
            "DIG",
            "CEO",
            "HDG",
            "KBC",
            "IJC",
            "SZC",
            "LDG",
        ],
        "etf": None,
        "weight_in_vnindex": 0.15,
    },
    "CONSUMER": {
        "name": "Tiêu dùng",
        "symbols": ["VNM", "MSN", "SAB", "MWG", "PNJ", "FRT", "DGW", "VEA"],
        "etf": None,
        "weight_in_vnindex": 0.10,
    },
    "ENERGY": {
        "name": "Năng lượng",
        "symbols": ["GAS", "PLX", "PVD", "PVS", "BSR", "OIL", "PVT", "PVC"],
        "etf": None,
        "weight_in_vnindex": 0.08,
    },
    "INDUSTRIAL": {
        "name": "Công nghiệp",
        "symbols": ["HPG", "HSG", "NKG", "SMC", "TLH", "POM", "TVN"],
        "etf": None,
        "weight_in_vnindex": 0.08,
    },
    "TECHNOLOGY": {
        "name": "Công nghệ",
        "symbols": ["FPT", "CMG", "VGI", "ELC", "SAM"],
        "etf": None,
        "weight_in_vnindex": 0.05,
    },
    "MATERIALS": {
        "name": "Vật liệu",
        "symbols": ["GVR", "DPM", "DCM", "DGC", "CSV", "AAA", "BMP"],
        "etf": None,
        "weight_in_vnindex": 0.05,
    },
    "UTILITIES": {
        "name": "Tiện ích",
        "symbols": ["POW", "NT2", "PPC", "REE", "GEG", "PC1"],
        "etf": None,
        "weight_in_vnindex": 0.04,
    },
    "AVIATION_TOURISM": {
        "name": "Hàng không & Du lịch",
        "symbols": ["VJC", "HVN", "ACV", "SCS", "VTR"],
        "etf": None,
        "weight_in_vnindex": 0.03,
    },
    "SECURITIES": {
        "name": "Chứng khoán",
        "symbols": ["SSI", "VND", "HCM", "VCI", "SHS", "MBS", "CTS", "TVS", "BSI"],
        "etf": None,
        "weight_in_vnindex": 0.04,
    },
}

# Sector rotation by economic cycle - Vietnam specific
SECTOR_CYCLE_PREFERENCE = {
    EconomicCycle.EARLY_EXPANSION: {
        "overweight": ["BANKING", "REAL_ESTATE", "SECURITIES"],
        "neutral": ["CONSUMER", "INDUSTRIAL"],
        "underweight": ["UTILITIES", "ENERGY"],
    },
    EconomicCycle.MID_EXPANSION: {
        "overweight": ["TECHNOLOGY", "CONSUMER", "INDUSTRIAL"],
        "neutral": ["BANKING", "MATERIALS"],
        "underweight": ["REAL_ESTATE", "UTILITIES"],
    },
    EconomicCycle.LATE_EXPANSION: {
        "overweight": ["ENERGY", "MATERIALS", "INDUSTRIAL"],
        "neutral": ["CONSUMER"],
        "underweight": ["BANKING", "REAL_ESTATE", "SECURITIES"],
    },
    EconomicCycle.EARLY_CONTRACTION: {
        "overweight": ["UTILITIES", "CONSUMER"],
        "neutral": ["ENERGY"],
        "underweight": ["BANKING", "REAL_ESTATE", "INDUSTRIAL", "SECURITIES"],
    },
    EconomicCycle.LATE_CONTRACTION: {
        "overweight": ["BANKING", "SECURITIES"],  # First to recover
        "neutral": ["UTILITIES", "CONSUMER"],
        "underweight": ["ENERGY", "MATERIALS", "INDUSTRIAL"],
    },
}


@dataclass
class SectorMetrics:
    """Sector analysis metrics"""

    sector: str
    momentum_1m: float  # 1-month momentum
    momentum_3m: float  # 3-month momentum
    relative_strength: float  # vs VNINDEX
    foreign_flow_score: float  # -1 to 1
    volatility: float
    phase: SectorPhase
    signal: SectorSignal
    confidence: float
    top_stocks: List[str]


@dataclass
class SectorRotationConfig:
    """Configuration for sector rotation strategy"""

    # Momentum thresholds
    strong_momentum_threshold: float = 0.10  # 10% = strong momentum
    weak_momentum_threshold: float = -0.05  # -5% = weak momentum

    # Relative strength thresholds
    rs_outperform_threshold: float = 1.10  # 10% outperformance
    rs_underperform_threshold: float = 0.90  # 10% underperformance

    # Foreign flow thresholds
    strong_inflow_threshold: float = 0.5  # Strong foreign buying
    strong_outflow_threshold: float = -0.5  # Strong foreign selling

    # Rotation parameters
    min_holding_period_days: int = 20  # Minimum holding period
    max_sectors_to_hold: int = 3  # Max sectors to overweight
    rebalance_threshold: float = 0.05  # Rebalance when drift > 5%

    # Position sizing
    sector_max_weight: float = 0.30  # Max 30% in single sector
    sector_min_weight: float = 0.05  # Min 5% allocation

    # Signal generation
    min_confidence: float = 0.60  # Minimum confidence for signal
    lookback_momentum_short: int = 20  # 1 month
    lookback_momentum_long: int = 60  # 3 months


class SectorRotationStrategy:
    """
    Sector rotation strategy for Vietnam market

    Key features:
    1. Tracks sector momentum and relative strength
    2. Identifies sector phase (accumulation, markup, distribution, markdown)
    3. Generates rotation signals based on economic cycle
    4. Integrates foreign flow data
    5. Provides rebalancing recommendations
    """

    def __init__(self, config: Optional[SectorRotationConfig] = None):
        self.config = config or SectorRotationConfig()
        self.sectors = VN_SECTORS.copy()

        # State tracking
        self.sector_metrics: Dict[str, SectorMetrics] = {}
        self.current_allocations: Dict[str, float] = {}
        self.rotation_history: List[Dict] = []
        self.last_rebalance: Optional[datetime] = None

        # Try to import foreign flow analyzer
        try:
            from src.market.foreign_flow import get_foreign_flow_analyzer

            self.foreign_flow_analyzer = get_foreign_flow_analyzer()
            self._has_foreign_flow = True
        except ImportError:
            self._has_foreign_flow = False
            logger.warning("Foreign flow analyzer not available")

    def calculate_sector_returns(
        self, sector_data: Dict[str, pd.DataFrame], periods: List[int] = [5, 20, 60]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate sector returns for multiple periods

        Args:
            sector_data: Dict of sector -> DataFrame with price data
            periods: List of lookback periods in days

        Returns:
            Dict of sector -> Dict of period -> return
        """
        returns = {}

        for sector, df in sector_data.items():
            if df is None or df.empty or len(df) < max(periods):
                returns[sector] = {p: 0.0 for p in periods}
                continue

            sector_returns = {}
            close = df["close"].values if "close" in df.columns else df.iloc[:, 0].values

            for period in periods:
                if len(close) > period:
                    ret = (close[-1] / close[-period] - 1) if close[-period] > 0 else 0
                    sector_returns[period] = ret
                else:
                    sector_returns[period] = 0.0

            returns[sector] = sector_returns

        return returns

    def calculate_relative_strength(
        self, sector_returns: Dict[str, float], benchmark_return: float
    ) -> Dict[str, float]:
        """
        Calculate relative strength vs benchmark (VNINDEX)

        RS > 1.0 means sector outperforms
        RS < 1.0 means sector underperforms
        """
        rs_scores = {}

        for sector, ret in sector_returns.items():
            denominator = 1 + benchmark_return
            if abs(denominator) < 1e-6:  # Guard against division by zero
                rs = 1.0
            else:
                # Relative strength = (1 + sector_return) / (1 + benchmark_return)
                rs = (1 + ret) / denominator
            rs_scores[sector] = rs

        return rs_scores

    def detect_sector_phase(
        self,
        momentum_short: float,
        momentum_long: float,
        relative_strength: float,
        foreign_flow: float,
    ) -> SectorPhase:
        """
        Detect sector phase in rotation cycle

        Phases:
        - ACCUMULATION: Low momentum but improving RS, foreign buying
        - MARKUP: Strong momentum, high RS, continued inflows
        - DISTRIBUTION: Peak momentum, RS weakening, foreign selling
        - MARKDOWN: Negative momentum, weak RS, outflows
        """
        strong_mom = self.config.strong_momentum_threshold
        weak_mom = self.config.weak_momentum_threshold

        # Calculate phase score for more robust detection
        accumulation_score = 0
        markup_score = 0
        distribution_score = 0
        markdown_score = 0

        # Momentum improving from low base = accumulation signal
        if momentum_long < weak_mom:
            accumulation_score += 2
        if momentum_short > momentum_long:
            accumulation_score += 2
        if foreign_flow > 0.1:  # More meaningful threshold
            accumulation_score += 1

        # Strong uptrend = markup signal
        if momentum_short > strong_mom:
            markup_score += 2
        if momentum_short > 0.02:  # Moderate positive momentum
            markup_score += 1
        if relative_strength > 1.0:
            markup_score += 2
        if foreign_flow > 0.2:
            markup_score += 1

        # Momentum slowing from high = distribution signal
        if momentum_long > strong_mom and momentum_short < momentum_long:
            distribution_score += 2
        if momentum_short > 0 and momentum_short < momentum_long * 0.5:
            distribution_score += 1
        if foreign_flow < -0.2:
            distribution_score += 2
        if relative_strength < 1.0 and momentum_short > 0:
            distribution_score += 1

        # Downtrend = markdown signal
        if momentum_short < weak_mom:
            markdown_score += 2
        if relative_strength < 0.95:
            markdown_score += 2
        if foreign_flow < -0.3:
            markdown_score += 1
        if momentum_short < momentum_long and momentum_long < 0:
            markdown_score += 1

        # Return phase with highest score
        scores = {
            SectorPhase.ACCUMULATION: accumulation_score,
            SectorPhase.MARKUP: markup_score,
            SectorPhase.DISTRIBUTION: distribution_score,
            SectorPhase.MARKDOWN: markdown_score,
        }

        max_score = max(scores.values())
        top_phases = [phase for phase, score in scores.items() if score == max_score]

        # Tie-breaker: prefer based on momentum direction
        if len(top_phases) > 1:
            if momentum_short > 0:
                # Prefer bullish phases
                priority = [
                    SectorPhase.MARKUP,
                    SectorPhase.ACCUMULATION,
                    SectorPhase.DISTRIBUTION,
                    SectorPhase.MARKDOWN,
                ]
            else:
                # Prefer bearish phases
                priority = [
                    SectorPhase.MARKDOWN,
                    SectorPhase.DISTRIBUTION,
                    SectorPhase.ACCUMULATION,
                    SectorPhase.MARKUP,
                ]
            for phase in priority:
                if phase in top_phases:
                    return phase

        return top_phases[0]

    def generate_sector_signal(
        self,
        phase: SectorPhase,
        momentum: float,
        relative_strength: float,
        foreign_flow: float,
        economic_cycle: Optional[EconomicCycle] = None,
        sector: Optional[str] = None,
    ) -> Tuple[SectorSignal, float]:
        """
        Generate trading signal for a sector

        Returns:
            (signal, confidence)
        """
        score = 0.0

        # Momentum contribution (±30 points)
        if momentum > self.config.strong_momentum_threshold:
            score += 30
        elif momentum > 0.02:
            score += 15
        elif momentum < self.config.weak_momentum_threshold:
            score -= 30
        elif momentum < -0.02:
            score -= 15

        # Relative strength contribution (±25 points)
        if relative_strength > self.config.rs_outperform_threshold:
            score += 25
        elif relative_strength > 1.02:
            score += 12
        elif relative_strength < self.config.rs_underperform_threshold:
            score -= 25
        elif relative_strength < 0.98:
            score -= 12

        # Foreign flow contribution (±25 points)
        if foreign_flow > self.config.strong_inflow_threshold:
            score += 25
        elif foreign_flow > 0.2:
            score += 12
        elif foreign_flow < self.config.strong_outflow_threshold:
            score -= 25
        elif foreign_flow < -0.2:
            score -= 12

        # Phase contribution (±20 points)
        if phase == SectorPhase.ACCUMULATION:
            score += 15  # Good entry opportunity
        elif phase == SectorPhase.MARKUP:
            score += 20  # Strong trend
        elif phase == SectorPhase.DISTRIBUTION:
            score -= 15  # Exit signal
        elif phase == SectorPhase.MARKDOWN:
            score -= 20  # Avoid

        # Economic cycle alignment (±15 points)
        if economic_cycle and sector:
            cycle_pref = SECTOR_CYCLE_PREFERENCE.get(economic_cycle, {})
            if sector in cycle_pref.get("overweight", []):
                score += 15
            elif sector in cycle_pref.get("underweight", []):
                score -= 15

        # Convert score to signal
        # Score range: -115 to +115 (adjusted thresholds)
        if score >= 70:
            signal = SectorSignal.STRONG_BUY
        elif score >= 35:
            signal = SectorSignal.BUY
        elif score >= -25:
            signal = SectorSignal.HOLD
        elif score >= -55:
            signal = SectorSignal.REDUCE
        else:
            signal = SectorSignal.SELL

        # Confidence based on score magnitude
        confidence = min(100, abs(score) + 40) / 100

        return signal, confidence

    def analyze_sector(
        self,
        sector: str,
        sector_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        economic_cycle: Optional[EconomicCycle] = None,
    ) -> SectorMetrics:
        """
        Complete analysis for a single sector

        Args:
            sector: Sector name
            sector_df: DataFrame with sector composite or ETF price
            benchmark_df: VNINDEX DataFrame
            economic_cycle: Current economic cycle phase

        Returns:
            SectorMetrics with full analysis
        """
        # Validate sector
        if sector not in self.sectors:
            logger.warning(f"Unknown sector: {sector}")
            return SectorMetrics(
                sector=sector,
                momentum_1m=0.0,
                momentum_3m=0.0,
                relative_strength=1.0,
                foreign_flow_score=0.0,
                volatility=0.20,
                phase=SectorPhase.MARKDOWN,
                signal=SectorSignal.HOLD,
                confidence=0.0,
                top_stocks=[],
            )

        # Calculate momentum
        returns_s = self.calculate_sector_returns(
            {sector: sector_df},
            [self.config.lookback_momentum_short, self.config.lookback_momentum_long],
        )

        momentum_1m = returns_s.get(sector, {}).get(self.config.lookback_momentum_short, 0)
        momentum_3m = returns_s.get(sector, {}).get(self.config.lookback_momentum_long, 0)

        # Calculate benchmark return
        if benchmark_df is not None and not benchmark_df.empty:
            bench_close = (
                benchmark_df["close"].values
                if "close" in benchmark_df.columns
                else benchmark_df.iloc[:, 0].values
            )
            if len(bench_close) > self.config.lookback_momentum_short:
                bench_return = (
                    bench_close[-1] / bench_close[-self.config.lookback_momentum_short] - 1
                )
            else:
                bench_return = 0
        else:
            bench_return = 0

        # Relative strength (safe division)
        denominator = 1 + bench_return
        if abs(denominator) < 1e-6:  # Guard against division by zero
            rs = 1.0
        else:
            rs = (1 + momentum_1m) / denominator

        # Foreign flow (placeholder - would integrate with foreign flow analyzer)
        foreign_flow = self._get_sector_foreign_flow(sector)

        # Calculate volatility (with division safety)
        if sector_df is not None and not sector_df.empty:
            close = (
                sector_df["close"].values
                if "close" in sector_df.columns
                else sector_df.iloc[:, 0].values
            )
            if len(close) > 20:
                # Safe returns calculation - avoid division by zero
                close_prev = close[:-1]
                valid_mask = close_prev > 1e-6  # Filter out zero/near-zero prices
                if np.sum(valid_mask) > 10:  # Need enough valid data points
                    valid_returns = np.diff(close)[valid_mask] / close_prev[valid_mask]
                    volatility = np.std(valid_returns[-20:]) * np.sqrt(252)  # Annualized
                else:
                    volatility = 0.20  # Default 20%
            else:
                volatility = 0.20  # Default 20%
        else:
            volatility = 0.20

        # Detect phase
        phase = self.detect_sector_phase(momentum_1m, momentum_3m, rs, foreign_flow)

        # Generate signal
        signal, confidence = self.generate_sector_signal(
            phase, momentum_1m, rs, foreign_flow, economic_cycle, sector
        )

        # Get top stocks in sector
        top_stocks = self._get_top_sector_stocks(sector, 5)

        return SectorMetrics(
            sector=sector,
            momentum_1m=momentum_1m,
            momentum_3m=momentum_3m,
            relative_strength=rs,
            foreign_flow_score=foreign_flow,
            volatility=volatility,
            phase=phase,
            signal=signal,
            confidence=confidence,
            top_stocks=top_stocks,
        )

    def _get_sector_foreign_flow(self, sector: str) -> float:
        """Get foreign flow score for sector (-1 to 1)"""
        try:
            # Aggregate foreign flow for sector symbols
            symbols = self.sectors.get(sector, {}).get("symbols", [])
            if not symbols:
                return 0.0

            if self._has_foreign_flow:
                total_flow = 0.0
                count = 0

                for symbol in symbols[:10]:  # Top 10 symbols
                    flow = self.foreign_flow_analyzer.get_flow_score(symbol)
                    if flow is not None:
                        total_flow += flow
                        count += 1

                if count > 0:
                    return total_flow / count

            # Fallback: estimate from sector characteristics
            # Banking/Securities typically have higher foreign interest
            sector_foreign_bias = {
                "BANKING": 0.1,
                "SECURITIES": 0.05,
                "TECHNOLOGY": 0.05,
                "CONSUMER": 0.02,
                "REAL_ESTATE": -0.05,
                "INDUSTRIAL": 0.0,
                "ENERGY": 0.0,
                "MATERIALS": 0.0,
                "UTILITIES": -0.02,
                "AVIATION_TOURISM": 0.0,
            }
            return sector_foreign_bias.get(sector, 0.0)

        except Exception as e:
            logger.warning(f"Error getting foreign flow for {sector}: {e}")
            return 0.0

    def _get_top_sector_stocks(self, sector: str, n: int = 5) -> List[str]:
        """Get top N stocks in sector by market cap/liquidity"""
        symbols = self.sectors.get(sector, {}).get("symbols", [])
        return symbols[:n]  # Simple approach - return first N

    def confirm_entry(
        self,
        sector_metrics: SectorMetrics,
        market_regime: Optional[str] = None,
        sector_correlations: Optional[Dict[str, float]] = None,
        sector_df: Optional[pd.DataFrame] = None,
        volume_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, List[str], float]:
        """
        Confirm entry with comprehensive filters for Vietnam market

        Args:
            sector_metrics: Metrics for the sector to enter
            market_regime: Current market regime ('BULL', 'BEAR', 'NEUTRAL')
            sector_correlations: Correlation with existing positions
            sector_df: Price DataFrame for additional technical checks
            volume_data: Volume DataFrame for liquidity confirmation

        Returns:
            (can_enter, rejection_reasons, entry_quality_score)
        """
        reasons = []
        quality_score = 100.0  # Start with perfect score, deduct for issues

        # =========================================================================
        # CORE FILTERS (Hard rejection)
        # =========================================================================

        # 1. Minimum confidence check
        if sector_metrics.confidence < self.config.min_confidence:
            reasons.append(
                f"Low confidence: {sector_metrics.confidence:.0%} < {self.config.min_confidence:.0%}"
            )
            quality_score -= 30

        # 2. Avoid entry during extreme volatility
        if sector_metrics.volatility > 0.45:  # 45% annualized
            reasons.append(f"High volatility: {sector_metrics.volatility:.0%}")
            quality_score -= 25
        elif sector_metrics.volatility > 0.35:
            quality_score -= 10  # Moderate penalty

        # 3. Avoid distribution and markdown phases for new entries
        if sector_metrics.phase == SectorPhase.DISTRIBUTION:
            reasons.append("Sector in DISTRIBUTION phase - avoid new entries")
            quality_score -= 35
        elif sector_metrics.phase == SectorPhase.MARKDOWN:
            reasons.append("Sector in MARKDOWN phase - strong avoid")
            quality_score -= 50

        # =========================================================================
        # MOMENTUM & TREND FILTERS
        # =========================================================================

        # 4. Require positive momentum alignment for BUY signals
        if sector_metrics.signal in [SectorSignal.BUY, SectorSignal.STRONG_BUY]:
            # Short-term momentum should be positive or recovering
            if sector_metrics.momentum_1m < -0.03:
                reasons.append(f"Negative ST momentum: {sector_metrics.momentum_1m:.1%}")
                quality_score -= 20
            elif sector_metrics.momentum_1m < 0:
                quality_score -= 5  # Minor penalty for flat momentum

            # Momentum should be improving (1M > 3M/3 for acceleration)
            momentum_acceleration = sector_metrics.momentum_1m - (sector_metrics.momentum_3m / 3)
            if momentum_acceleration < -0.02:
                reasons.append(f"Decelerating momentum: {momentum_acceleration:.1%}")
                quality_score -= 15

        # 5. Trend confirmation using price data
        if sector_df is not None and not sector_df.empty and len(sector_df) >= 20:
            close = (
                sector_df["close"].values
                if "close" in sector_df.columns
                else sector_df.iloc[:, 0].values
            )

            # Check price above short-term MA (trend confirmation)
            ma_10 = np.mean(close[-10:]) if len(close) >= 10 else close[-1]
            ma_20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
            current_price = close[-1]

            if current_price < ma_10 < ma_20:
                reasons.append("Price below declining MAs - bearish structure")
                quality_score -= 20
            elif current_price < ma_10:
                quality_score -= 10  # Below short MA
            elif current_price > ma_10 > ma_20:
                quality_score += 5  # Bullish structure bonus

            # Check for recent lower lows (downtrend confirmation)
            if len(close) >= 10:
                recent_low = np.min(close[-5:])
                prior_low = np.min(close[-10:-5])
                if recent_low < prior_low * 0.98:  # Making lower lows
                    quality_score -= 10

        # =========================================================================
        # MARKET REGIME FILTERS
        # =========================================================================

        # 6. Market regime filter - more nuanced
        if market_regime == "BEAR":
            if sector_metrics.signal != SectorSignal.STRONG_BUY:
                reasons.append("Only STRONG_BUY allowed in BEAR market")
                quality_score -= 25
            # In bear market, require stronger RS
            if sector_metrics.relative_strength < 1.08:
                reasons.append(f"Weak RS in bear market: {sector_metrics.relative_strength:.2f}")
                quality_score -= 15
            # Additional: require foreign flow support in bear market
            if sector_metrics.foreign_flow_score < 0.2:
                reasons.append("Insufficient foreign support in bear market")
                quality_score -= 10
        elif market_regime == "NEUTRAL":
            # Neutral market: be more selective
            if sector_metrics.signal == SectorSignal.BUY and sector_metrics.confidence < 0.70:
                quality_score -= 10
        elif market_regime == "BULL":
            # Bull market: bonus for strong signals
            if sector_metrics.signal == SectorSignal.STRONG_BUY:
                quality_score += 10

        # =========================================================================
        # RELATIVE STRENGTH FILTERS
        # =========================================================================

        # 7. Relative strength should be supportive
        if sector_metrics.relative_strength < 0.92:
            reasons.append(
                f"Significant underperformance: RS={sector_metrics.relative_strength:.2f}"
            )
            quality_score -= 20
        elif sector_metrics.relative_strength < 0.98:
            quality_score -= 10
        elif sector_metrics.relative_strength > 1.10:
            quality_score += 5  # Strong outperformance bonus

        # =========================================================================
        # VOLUME/LIQUIDITY FILTERS (Vietnam market specific)
        # =========================================================================

        # 8. Volume confirmation (if data available)
        if volume_data is not None and not volume_data.empty:
            vol = volume_data.values.flatten()
            if len(vol) >= 20:
                avg_vol_20 = np.mean(vol[-20:])
                recent_vol = np.mean(vol[-5:])

                # For BUY signals, prefer increasing volume
                if sector_metrics.signal in [SectorSignal.BUY, SectorSignal.STRONG_BUY]:
                    if recent_vol < avg_vol_20 * 0.7:
                        reasons.append("Low volume - weak conviction")
                        quality_score -= 15
                    elif recent_vol > avg_vol_20 * 1.3:
                        quality_score += 5  # Volume confirmation bonus

        # =========================================================================
        # FOREIGN FLOW FILTERS (Critical for VN market)
        # =========================================================================

        # 9. Foreign flow should support entry
        if sector_metrics.foreign_flow_score < -0.3:
            reasons.append(f"Strong foreign outflow: {sector_metrics.foreign_flow_score:.2f}")
            quality_score -= 20
        elif sector_metrics.foreign_flow_score > 0.3:
            quality_score += 10  # Foreign support bonus

        # =========================================================================
        # CORRELATION FILTERS
        # =========================================================================

        # 10. Correlation check (avoid concentrated risk)
        if sector_correlations:
            high_corr_sectors = [
                s
                for s, corr in sector_correlations.items()
                if corr > 0.70 and s != sector_metrics.sector
            ]
            if len(high_corr_sectors) >= 2:
                reasons.append(
                    f"High correlation with multiple sectors: {', '.join(high_corr_sectors)}"
                )
                quality_score -= 20
            elif high_corr_sectors:
                reasons.append(f"High correlation with: {', '.join(high_corr_sectors)}")
                quality_score -= 10

        # =========================================================================
        # FINAL DECISION
        # =========================================================================

        # Normalize quality score
        quality_score = max(0, min(100, quality_score))

        # Entry allowed if no hard rejections AND quality score >= 50
        can_enter = len(reasons) == 0 or (
            quality_score >= 50 and not any("MARKDOWN" in r or "DISTRIBUTION" in r for r in reasons)
        )

        return can_enter, reasons, quality_score

    def get_sector_correlation(self, sector1: str, sector2: str) -> float:
        """
        Get correlation between two sectors

        Returns pre-defined correlation based on Vietnam market characteristics
        """
        # High correlation pairs in Vietnam market
        high_corr_pairs = [
            ("BANKING", "SECURITIES"),
            ("BANKING", "REAL_ESTATE"),
            ("SECURITIES", "REAL_ESTATE"),
            ("INDUSTRIAL", "MATERIALS"),
            ("ENERGY", "MATERIALS"),
        ]

        # Medium correlation pairs
        medium_corr_pairs = [
            ("BANKING", "CONSUMER"),
            ("REAL_ESTATE", "INDUSTRIAL"),
            ("TECHNOLOGY", "SECURITIES"),
        ]

        pair = tuple(sorted([sector1, sector2]))

        for p in high_corr_pairs:
            if tuple(sorted(p)) == pair:
                return 0.75

        for p in medium_corr_pairs:
            if tuple(sorted(p)) == pair:
                return 0.50

        return 0.30  # Default low correlation

    def detect_market_regime(
        self,
        benchmark_df: pd.DataFrame,
        lookback_short: int = 20,
        lookback_long: int = 60,
    ) -> str:
        """
        Detect current market regime based on VNINDEX

        Returns:
            'BULL', 'BEAR', or 'NEUTRAL'
        """
        if benchmark_df is None or benchmark_df.empty:
            return "NEUTRAL"

        close = (
            benchmark_df["close"].values
            if "close" in benchmark_df.columns
            else benchmark_df.iloc[:, 0].values
        )

        if len(close) < lookback_long:
            return "NEUTRAL"

        # Calculate returns
        ret_short = close[-1] / close[-lookback_short] - 1 if close[-lookback_short] > 0 else 0
        ret_long = close[-1] / close[-lookback_long] - 1 if close[-lookback_long] > 0 else 0

        # Calculate trend using moving averages
        ma_short = np.mean(close[-lookback_short:])
        ma_long = np.mean(close[-lookback_long:])

        # Bull: positive returns and price above MAs
        if ret_short > 0.03 and ret_long > 0.05 and close[-1] > ma_short > ma_long:
            return "BULL"

        # Bear: negative returns and price below MAs
        if ret_short < -0.03 and ret_long < -0.05 and close[-1] < ma_short < ma_long:
            return "BEAR"

        return "NEUTRAL"

    def analyze_all_sectors(
        self,
        sector_data: Dict[str, pd.DataFrame],
        benchmark_df: pd.DataFrame,
        economic_cycle: Optional[EconomicCycle] = None,
    ) -> Dict[str, SectorMetrics]:
        """
        Analyze all sectors and update metrics

        Args:
            sector_data: Dict of sector -> DataFrame
            benchmark_df: VNINDEX DataFrame
            economic_cycle: Current economic cycle

        Returns:
            Dict of sector -> SectorMetrics
        """
        for sector in self.sectors.keys():
            df = sector_data.get(sector)
            if df is not None:
                metrics = self.analyze_sector(sector, df, benchmark_df, economic_cycle)
                self.sector_metrics[sector] = metrics

        return self.sector_metrics

    def get_rotation_recommendations(
        self,
        current_allocations: Optional[Dict[str, float]] = None,
        available_capital: float = 100_000_000,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate sector rotation recommendations

        Args:
            current_allocations: Current sector allocations (weights)
            available_capital: Available capital for rotation
            benchmark_df: VNINDEX DataFrame for market regime detection

        Returns:
            Dict with recommendations
        """
        if not self.sector_metrics:
            return {"error": "No sector metrics available. Run analyze_all_sectors first."}

        # Detect market regime
        market_regime = (
            self.detect_market_regime(benchmark_df) if benchmark_df is not None else "NEUTRAL"
        )

        current = current_allocations or self.current_allocations

        # Rank sectors by signal and confidence
        sector_scores = []
        for sector, metrics in self.sector_metrics.items():
            # Score = signal strength + confidence
            signal_score = {
                SectorSignal.STRONG_BUY: 2,
                SectorSignal.BUY: 1,
                SectorSignal.HOLD: 0,
                SectorSignal.REDUCE: -1,
                SectorSignal.SELL: -2,
            }.get(metrics.signal, 0)

            combined_score = signal_score + metrics.confidence
            sector_scores.append((sector, combined_score, metrics))

        # Sort by score descending
        sector_scores.sort(key=lambda x: x[1], reverse=True)

        # Generate target allocations
        target_allocations = {}
        remaining_weight = 1.0

        # Top sectors get overweight
        for i, (sector, score, metrics) in enumerate(sector_scores):
            if i < self.config.max_sectors_to_hold and score > 0.5:
                # Overweight top sectors
                weight = min(self.config.sector_max_weight, 0.25)
            elif score > 0:
                # Neutral weight for ok sectors
                weight = 0.10
            elif score > -0.5:
                # Underweight for weak sectors
                weight = 0.05
            else:
                # Avoid worst sectors
                weight = 0.0

            target_allocations[sector] = weight
            remaining_weight -= weight

        # Normalize if needed
        total_weight = sum(target_allocations.values())
        if total_weight > 0:
            for sector in target_allocations:
                target_allocations[sector] /= total_weight

        # Calculate changes needed
        changes = []
        for sector in target_allocations:
            current_weight = current.get(sector, 0)
            target_weight = target_allocations[sector]
            diff = target_weight - current_weight

            if abs(diff) > self.config.rebalance_threshold:
                action = "INCREASE" if diff > 0 else "DECREASE"
                amount = abs(diff) * available_capital

                changes.append(
                    {
                        "sector": sector,
                        "action": action,
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "weight_change": diff,
                        "amount": amount,
                        "signal": self.sector_metrics[sector].signal.value,
                        "confidence": self.sector_metrics[sector].confidence,
                        "top_stocks": self.sector_metrics[sector].top_stocks,
                    }
                )

        # Sort changes by importance
        changes.sort(key=lambda x: abs(x["weight_change"]), reverse=True)

        # Validate entries with confirmation logic
        validated_changes = []
        for change in changes:
            if change["action"] == "INCREASE":
                sector = change["sector"]
                metrics = self.sector_metrics[sector]

                # Check correlations with other increasing sectors
                other_increasing = [
                    c["sector"] for c in validated_changes if c["action"] == "INCREASE"
                ]
                correlations = {s: self.get_sector_correlation(sector, s) for s in other_increasing}

                can_enter, reasons, quality_score = self.confirm_entry(
                    metrics, market_regime, correlations
                )
                change["entry_confirmed"] = can_enter
                change["rejection_reasons"] = reasons
                change["entry_quality_score"] = quality_score
            else:
                change["entry_confirmed"] = True
                change["rejection_reasons"] = []
                change["entry_quality_score"] = 100.0  # Exit always allowed

            validated_changes.append(change)

        return {
            "timestamp": datetime.now().isoformat(),
            "market_regime": market_regime,
            "target_allocations": target_allocations,
            "current_allocations": current,
            "recommended_changes": validated_changes,
            "sector_rankings": [(s, sc, m.signal.value) for s, sc, m in sector_scores],
            "total_rebalance_amount": sum(
                c["amount"] for c in validated_changes if c["entry_confirmed"]
            ),
        }

    def get_sector_stock_recommendations(self, sector: str, n_stocks: int = 5) -> List[Dict]:
        """
        Get top stock recommendations within a sector

        This would integrate with individual stock analysis
        """
        if sector not in self.sectors:
            return []

        symbols = self.sectors[sector]["symbols"]
        metrics = self.sector_metrics.get(sector)

        recommendations = []
        for symbol in symbols[:n_stocks]:
            recommendations.append(
                {
                    "symbol": symbol,
                    "sector": sector,
                    "sector_signal": metrics.signal.value if metrics else "UNKNOWN",
                    "sector_confidence": metrics.confidence if metrics else 0,
                    "sector_phase": metrics.phase.value if metrics else "UNKNOWN",
                }
            )

        return recommendations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_rotation_strategy: Optional[SectorRotationStrategy] = None


def get_sector_rotation_strategy(
    config: Optional[SectorRotationConfig] = None,
) -> SectorRotationStrategy:
    """Get singleton sector rotation strategy instance"""
    global _rotation_strategy
    if _rotation_strategy is None:
        _rotation_strategy = SectorRotationStrategy(config)
    return _rotation_strategy


def get_sector_for_symbol(symbol: str) -> Optional[str]:
    """Get sector for a given symbol"""
    symbol = symbol.upper()
    for sector, info in VN_SECTORS.items():
        if symbol in info["symbols"]:
            return sector
    return None


def get_sector_symbols(sector: str) -> List[str]:
    """Get all symbols in a sector"""
    return VN_SECTORS.get(sector, {}).get("symbols", [])


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Sector Rotation Strategy...\n")

    # Create strategy
    strategy = SectorRotationStrategy()

    # Test sector detection
    print("Symbol to Sector Mapping:")
    for symbol in ["VCB", "VNM", "HPG", "FPT", "GAS"]:
        sector = get_sector_for_symbol(symbol)
        print(f"  {symbol} -> {sector}")

    # Create sample sector data
    import numpy as np

    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")

    # Simulate sector price data
    sector_data = {}
    for sector in ["BANKING", "TECHNOLOGY", "REAL_ESTATE"]:
        # Random walk with trend
        trend = 0.001 if sector == "TECHNOLOGY" else 0.0005
        returns = np.random.normal(trend, 0.02, 100)
        prices = 100 * np.cumprod(1 + returns)
        sector_data[sector] = pd.DataFrame({"close": prices}, index=dates)

    # Benchmark (VNINDEX)
    bench_returns = np.random.normal(0.0005, 0.015, 100)
    bench_prices = 1200 * np.cumprod(1 + bench_returns)
    benchmark_df = pd.DataFrame({"close": bench_prices}, index=dates)

    # Analyze sectors
    print("\nSector Analysis:")
    metrics = strategy.analyze_all_sectors(sector_data, benchmark_df, EconomicCycle.MID_EXPANSION)

    for sector, m in metrics.items():
        print(f"\n{sector}:")
        print(f"  Momentum 1M: {m.momentum_1m:.1%}")
        print(f"  Momentum 3M: {m.momentum_3m:.1%}")
        print(f"  Relative Strength: {m.relative_strength:.2f}")
        print(f"  Phase: {m.phase.value}")
        print(f"  Signal: {m.signal.value} ({m.confidence:.0%} confidence)")

    # Get rotation recommendations
    print("\nRotation Recommendations:")
    recs = strategy.get_rotation_recommendations(
        current_allocations={"BANKING": 0.40, "TECHNOLOGY": 0.20, "REAL_ESTATE": 0.40},
        available_capital=100_000_000,
    )

    for change in recs["recommended_changes"]:
        print(
            f"  {change['action']} {change['sector']}: "
            f"{change['current_weight']:.0%} -> {change['target_weight']:.0%} "
            f"({change['amount']:,.0f} VND)"
        )

    print("\n✅ Sector Rotation Strategy test completed!")
