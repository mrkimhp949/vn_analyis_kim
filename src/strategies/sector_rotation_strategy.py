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
            if benchmark_return == 0:
                rs = 1.0
            else:
                # Relative strength = (1 + sector_return) / (1 + benchmark_return)
                rs = (1 + ret) / (1 + benchmark_return)
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

        # Accumulation: beaten down but showing signs of life
        if momentum_long < weak_mom and momentum_short > momentum_long and foreign_flow > 0:
            return SectorPhase.ACCUMULATION

        # Markup: strong uptrend
        elif momentum_short > strong_mom and relative_strength > 1.0:
            return SectorPhase.MARKUP

        # Distribution: momentum slowing, smart money exiting
        elif momentum_short < momentum_long and foreign_flow < -0.3:
            return SectorPhase.DISTRIBUTION

        # Markdown: downtrend
        elif momentum_short < weak_mom and relative_strength < 0.95:
            return SectorPhase.MARKDOWN

        # Default based on momentum
        elif momentum_short > 0:
            return SectorPhase.MARKUP
        else:
            return SectorPhase.MARKDOWN

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
        # Score range: -100 to +100
        if score >= 60:
            signal = SectorSignal.STRONG_BUY
        elif score >= 30:
            signal = SectorSignal.BUY
        elif score >= -20:
            signal = SectorSignal.HOLD
        elif score >= -50:
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

        # Relative strength
        rs = (1 + momentum_1m) / (1 + bench_return) if bench_return != -1 else 1.0

        # Foreign flow (placeholder - would integrate with foreign flow analyzer)
        foreign_flow = self._get_sector_foreign_flow(sector)

        # Calculate volatility
        if sector_df is not None and not sector_df.empty:
            close = (
                sector_df["close"].values
                if "close" in sector_df.columns
                else sector_df.iloc[:, 0].values
            )
            if len(close) > 20:
                returns = np.diff(close) / close[:-1]
                volatility = np.std(returns[-20:]) * np.sqrt(252)  # Annualized
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
        if not self._has_foreign_flow:
            return 0.0

        try:
            # Aggregate foreign flow for sector symbols
            symbols = self.sectors.get(sector, {}).get("symbols", [])
            if not symbols:
                return 0.0

            total_flow = 0.0
            count = 0

            for symbol in symbols[:10]:  # Top 10 symbols
                flow = self.foreign_flow_analyzer.get_flow_score(symbol)
                if flow is not None:
                    total_flow += flow
                    count += 1

            return total_flow / count if count > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error getting foreign flow for {sector}: {e}")
            return 0.0

    def _get_top_sector_stocks(self, sector: str, n: int = 5) -> List[str]:
        """Get top N stocks in sector by market cap/liquidity"""
        symbols = self.sectors.get(sector, {}).get("symbols", [])
        return symbols[:n]  # Simple approach - return first N

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
    ) -> Dict[str, Any]:
        """
        Generate sector rotation recommendations

        Args:
            current_allocations: Current sector allocations (weights)
            available_capital: Available capital for rotation

        Returns:
            Dict with recommendations
        """
        if not self.sector_metrics:
            return {"error": "No sector metrics available. Run analyze_all_sectors first."}

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

        return {
            "timestamp": datetime.now().isoformat(),
            "target_allocations": target_allocations,
            "current_allocations": current,
            "recommended_changes": changes,
            "sector_rankings": [(s, sc, m.signal.value) for s, sc, m in sector_scores],
            "total_rebalance_amount": sum(c["amount"] for c in changes),
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
