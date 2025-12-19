# -*- coding: utf-8 -*-
"""
Portfolio Correlation Analysis Module v10.2

Analyze and monitor correlation between portfolio positions to:
- Avoid over-concentration in correlated assets
- Identify diversification opportunities
- Generate correlation heat map for visualization
- Calculate portfolio-wide correlation risk

Author: Trading Bot Team
Version: 10.2.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from threading import RLock

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class CorrelationConfig:
    """Configuration for correlation analysis."""

    # Correlation thresholds
    high_correlation_threshold: float = 0.70  # Warn if correlation > 0.7
    critical_correlation_threshold: float = 0.85  # Block if correlation > 0.85
    negative_correlation_bonus: float = -0.30  # Bonus for negatively correlated assets

    # Analysis settings
    lookback_days: int = 60  # Days to calculate correlation
    min_data_points: int = 30  # Minimum data points required

    # Portfolio limits
    max_correlated_positions: int = 3  # Max positions with correlation > threshold
    max_sector_correlation: float = 0.80  # Max average correlation within sector

    # Cache settings
    cache_ttl_seconds: int = 3600  # 1 hour cache

    # Vietnam market specific
    vn30_correlation_weight: float = 0.15  # Weight of VN30 correlation in risk calc


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""

    symbol_a: str
    symbol_b: str
    correlation: float
    is_high_correlation: bool
    is_critical: bool
    data_points: int
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "correlation": round(self.correlation, 4),
            "is_high_correlation": self.is_high_correlation,
            "is_critical": self.is_critical,
            "data_points": self.data_points,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class PortfolioCorrelationReport:
    """Comprehensive portfolio correlation report."""

    total_positions: int
    avg_pairwise_correlation: float
    max_correlation: float
    max_correlation_pair: Tuple[str, str]
    high_correlation_pairs: List[CorrelationResult]
    diversification_score: float  # 0-100, higher = better diversified
    sector_correlations: Dict[str, float]
    recommendations: List[str]
    correlation_matrix: Optional[pd.DataFrame] = None
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_positions": self.total_positions,
            "avg_pairwise_correlation": round(self.avg_pairwise_correlation, 4),
            "max_correlation": round(self.max_correlation, 4),
            "max_correlation_pair": self.max_correlation_pair,
            "high_correlation_count": len(self.high_correlation_pairs),
            "diversification_score": round(self.diversification_score, 1),
            "sector_correlations": {k: round(v, 4) for k, v in self.sector_correlations.items()},
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


# =============================================================================
# CORRELATION ANALYZER
# =============================================================================


class PortfolioCorrelationAnalyzer:
    """
    Analyze correlation between portfolio positions.

    Features:
    - Pairwise correlation calculation
    - Sector-based correlation analysis
    - Diversification scoring
    - Real-time correlation monitoring
    - Heat map generation
    """

    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self._cache: Dict[str, Tuple[float, datetime]] = {}
        self._lock = RLock()
        self._price_data_cache: Dict[str, pd.Series] = {}

    def calculate_correlation(
        self,
        symbol_a: str,
        symbol_b: str,
        price_data_a: pd.Series,
        price_data_b: pd.Series,
    ) -> CorrelationResult:
        """
        Calculate correlation between two symbols.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            price_data_a: Price series for symbol A
            price_data_b: Price series for symbol B

        Returns:
            CorrelationResult with correlation details
        """
        # Align data
        aligned_a, aligned_b = price_data_a.align(price_data_b, join="inner")

        if len(aligned_a) < self.config.min_data_points:
            logger.warning(
                f"Insufficient data for {symbol_a}-{symbol_b} correlation: "
                f"{len(aligned_a)} < {self.config.min_data_points}"
            )
            return CorrelationResult(
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                correlation=0.0,
                is_high_correlation=False,
                is_critical=False,
                data_points=len(aligned_a),
            )

        # Calculate returns
        returns_a = aligned_a.pct_change().dropna()
        returns_b = aligned_b.pct_change().dropna()

        # Calculate correlation
        correlation = returns_a.corr(returns_b)

        if pd.isna(correlation):
            correlation = 0.0

        return CorrelationResult(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            correlation=float(correlation),
            is_high_correlation=abs(correlation) > self.config.high_correlation_threshold,
            is_critical=abs(correlation) > self.config.critical_correlation_threshold,
            data_points=len(returns_a),
        )

    def analyze_portfolio(
        self,
        positions: Dict[str, Dict],
        price_data: Dict[str, pd.Series],
        sector_map: Optional[Dict[str, str]] = None,
    ) -> PortfolioCorrelationReport:
        """
        Analyze correlation across entire portfolio.

        Args:
            positions: Dict of symbol -> position info
            price_data: Dict of symbol -> price series
            sector_map: Optional dict of symbol -> sector

        Returns:
            PortfolioCorrelationReport with analysis
        """
        symbols = list(positions.keys())
        n = len(symbols)

        if n < 2:
            return self._empty_report(n)

        # Build correlation matrix
        correlation_matrix = pd.DataFrame(index=symbols, columns=symbols, dtype=float)

        high_correlation_pairs: List[CorrelationResult] = []
        all_correlations: List[float] = []
        max_corr = 0.0
        max_pair = (symbols[0], symbols[1]) if n >= 2 else ("", "")

        for i in range(n):
            correlation_matrix.iloc[i, i] = 1.0
            for j in range(i + 1, n):
                sym_a, sym_b = symbols[i], symbols[j]

                if sym_a not in price_data or sym_b not in price_data:
                    correlation_matrix.iloc[i, j] = 0.0
                    correlation_matrix.iloc[j, i] = 0.0
                    continue

                result = self.calculate_correlation(
                    sym_a, sym_b, price_data[sym_a], price_data[sym_b]
                )

                correlation_matrix.iloc[i, j] = result.correlation
                correlation_matrix.iloc[j, i] = result.correlation
                all_correlations.append(result.correlation)

                if abs(result.correlation) > abs(max_corr):
                    max_corr = result.correlation
                    max_pair = (sym_a, sym_b)

                if result.is_high_correlation:
                    high_correlation_pairs.append(result)

        # Calculate average correlation
        avg_correlation = np.mean(all_correlations) if all_correlations else 0.0

        # Calculate sector correlations
        sector_correlations = self._calculate_sector_correlations(
            symbols, correlation_matrix, sector_map or {}
        )

        # Calculate diversification score
        diversification_score = self._calculate_diversification_score(
            avg_correlation, len(high_correlation_pairs), n
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_correlation, high_correlation_pairs, sector_correlations, n
        )

        return PortfolioCorrelationReport(
            total_positions=n,
            avg_pairwise_correlation=float(avg_correlation),
            max_correlation=float(max_corr),
            max_correlation_pair=max_pair,
            high_correlation_pairs=high_correlation_pairs,
            diversification_score=diversification_score,
            sector_correlations=sector_correlations,
            recommendations=recommendations,
            correlation_matrix=correlation_matrix,
        )

    def check_new_position_correlation(
        self,
        new_symbol: str,
        new_price_data: pd.Series,
        existing_positions: Dict[str, Dict],
        existing_price_data: Dict[str, pd.Series],
    ) -> Tuple[bool, str, List[CorrelationResult]]:
        """
        Check if a new position would create excessive correlation.

        Args:
            new_symbol: Symbol to add
            new_price_data: Price data for new symbol
            existing_positions: Current portfolio positions
            existing_price_data: Price data for existing positions

        Returns:
            (is_allowed, message, correlation_results)
        """
        correlations: List[CorrelationResult] = []
        critical_correlations: List[str] = []
        high_correlations: List[str] = []

        for symbol in existing_positions:
            if symbol not in existing_price_data:
                continue

            result = self.calculate_correlation(
                new_symbol, symbol, new_price_data, existing_price_data[symbol]
            )
            correlations.append(result)

            if result.is_critical:
                critical_correlations.append(f"{symbol} ({result.correlation:.2f})")
            elif result.is_high_correlation:
                high_correlations.append(f"{symbol} ({result.correlation:.2f})")

        # Decision logic
        if critical_correlations:
            return (
                False,
                f"❌ BLOCKED: Critical correlation with {', '.join(critical_correlations)}",
                correlations,
            )

        if len(high_correlations) >= self.config.max_correlated_positions:
            return (
                False,
                f"⚠️ BLOCKED: Too many high correlations: {', '.join(high_correlations)}",
                correlations,
            )

        if high_correlations:
            return (
                True,
                f"⚠️ WARNING: High correlation with {', '.join(high_correlations)}. "
                f"Consider reducing position size.",
                correlations,
            )

        return (True, "✅ Correlation check passed", correlations)

    def generate_heatmap_data(
        self,
        correlation_matrix: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Generate data for correlation heat map visualization.

        Args:
            correlation_matrix: Correlation matrix from analyze_portfolio

        Returns:
            Dict with heat map data for visualization
        """
        if correlation_matrix is None or correlation_matrix.empty:
            return {"error": "No correlation data available"}

        symbols = correlation_matrix.columns.tolist()

        # Convert to list format for visualization
        heatmap_data = []
        for i, sym_a in enumerate(symbols):
            for j, sym_b in enumerate(symbols):
                corr = correlation_matrix.iloc[i, j]
                heatmap_data.append(
                    {
                        "x": sym_a,
                        "y": sym_b,
                        "value": round(float(corr), 4),
                        "color": self._get_correlation_color(corr),
                    }
                )

        return {
            "symbols": symbols,
            "data": heatmap_data,
            "min_value": -1.0,
            "max_value": 1.0,
            "threshold_high": self.config.high_correlation_threshold,
            "threshold_critical": self.config.critical_correlation_threshold,
        }

    def _calculate_sector_correlations(
        self,
        symbols: List[str],
        correlation_matrix: pd.DataFrame,
        sector_map: Dict[str, str],
    ) -> Dict[str, float]:
        """Calculate average correlation within each sector."""
        sector_correlations: Dict[str, List[float]] = {}

        for i, sym_a in enumerate(symbols):
            sector_a = sector_map.get(sym_a, "UNKNOWN")

            for j, sym_b in enumerate(symbols):
                if i >= j:  # Skip diagonal and duplicates
                    continue

                sector_b = sector_map.get(sym_b, "UNKNOWN")

                if sector_a == sector_b:
                    if sector_a not in sector_correlations:
                        sector_correlations[sector_a] = []
                    sector_correlations[sector_a].append(correlation_matrix.iloc[i, j])

        return {
            sector: float(np.mean(corrs)) if corrs else 0.0
            for sector, corrs in sector_correlations.items()
        }

    def _calculate_diversification_score(
        self,
        avg_correlation: float,
        high_correlation_count: int,
        position_count: int,
    ) -> float:
        """
        Calculate diversification score (0-100).

        Higher score = better diversified.
        """
        if position_count < 2:
            return 0.0

        # Base score from average correlation (lower = better)
        # avg_corr of 0.3 = 70 points, 0.7 = 30 points
        base_score = max(0, 100 - (avg_correlation * 100))

        # Penalty for high correlation pairs
        max_pairs = position_count * (position_count - 1) / 2
        high_corr_penalty = (high_correlation_count / max_pairs) * 30 if max_pairs > 0 else 0

        # Bonus for having more positions (diversification)
        position_bonus = min(10, position_count * 2)

        score = base_score - high_corr_penalty + position_bonus
        return max(0, min(100, score))

    def _generate_recommendations(
        self,
        avg_correlation: float,
        high_correlation_pairs: List[CorrelationResult],
        sector_correlations: Dict[str, float],
        position_count: int,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if avg_correlation > 0.6:
            recommendations.append(
                "⚠️ High average portfolio correlation. Consider adding uncorrelated assets."
            )

        if len(high_correlation_pairs) > position_count // 2:
            recommendations.append(
                f"🔴 {len(high_correlation_pairs)} highly correlated pairs. "
                "Consider reducing overlapping positions."
            )

        # Check for critical pairs
        critical_pairs = [p for p in high_correlation_pairs if p.is_critical]
        if critical_pairs:
            for pair in critical_pairs[:3]:  # Top 3 critical
                recommendations.append(
                    f"🚨 Critical correlation ({pair.correlation:.2f}): "
                    f"{pair.symbol_a} - {pair.symbol_b}. Consider closing one."
                )

        # Sector concentration
        for sector, corr in sector_correlations.items():
            if corr > self.config.max_sector_correlation:
                recommendations.append(
                    f"📊 High {sector} sector correlation ({corr:.2f}). "
                    "Consider diversifying across sectors."
                )

        if position_count < 5:
            recommendations.append("💡 Consider adding more positions for better diversification.")

        if not recommendations:
            recommendations.append("✅ Portfolio correlation looks healthy!")

        return recommendations

    def _get_correlation_color(self, corr: float) -> str:
        """Get color code for correlation value."""
        if corr > self.config.critical_correlation_threshold:
            return "#FF0000"  # Red - critical
        elif corr > self.config.high_correlation_threshold:
            return "#FFA500"  # Orange - high
        elif corr > 0.3:
            return "#FFFF00"  # Yellow - moderate
        elif corr > -0.3:
            return "#90EE90"  # Light green - low
        else:
            return "#00FF00"  # Green - negative (diversification benefit)

    def _empty_report(self, position_count: int) -> PortfolioCorrelationReport:
        """Return empty report for insufficient positions."""
        return PortfolioCorrelationReport(
            total_positions=position_count,
            avg_pairwise_correlation=0.0,
            max_correlation=0.0,
            max_correlation_pair=("", ""),
            high_correlation_pairs=[],
            diversification_score=0.0 if position_count < 2 else 50.0,
            sector_correlations={},
            recommendations=["Need at least 2 positions for correlation analysis."],
        )


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_correlation_analyzer: Optional[PortfolioCorrelationAnalyzer] = None
_analyzer_lock = RLock()


def get_correlation_analyzer(
    config: Optional[CorrelationConfig] = None,
) -> PortfolioCorrelationAnalyzer:
    """Get singleton instance of PortfolioCorrelationAnalyzer."""
    global _correlation_analyzer

    with _analyzer_lock:
        if _correlation_analyzer is None:
            _correlation_analyzer = PortfolioCorrelationAnalyzer(config)
        return _correlation_analyzer


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def check_portfolio_correlation(
    positions: Dict[str, Dict],
    price_data: Dict[str, pd.Series],
    sector_map: Optional[Dict[str, str]] = None,
) -> PortfolioCorrelationReport:
    """
    Convenience function to analyze portfolio correlation.

    Usage:
        report = check_portfolio_correlation(positions, price_data)
        print(f"Diversification Score: {report.diversification_score}")
        for rec in report.recommendations:
            print(rec)
    """
    analyzer = get_correlation_analyzer()
    return analyzer.analyze_portfolio(positions, price_data, sector_map)


def can_add_position(
    new_symbol: str,
    new_price_data: pd.Series,
    existing_positions: Dict[str, Dict],
    existing_price_data: Dict[str, pd.Series],
) -> Tuple[bool, str]:
    """
    Check if adding a new position is allowed based on correlation.

    Returns:
        (is_allowed, message)
    """
    analyzer = get_correlation_analyzer()
    allowed, message, _ = analyzer.check_new_position_correlation(
        new_symbol, new_price_data, existing_positions, existing_price_data
    )
    return allowed, message
