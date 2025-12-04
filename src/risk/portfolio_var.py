# -*- coding: utf-8 -*-
"""
Portfolio Value at Risk (VaR) & Advanced Risk Metrics

Features:
- Historical VaR calculation
- Parametric VaR
- Monte Carlo VaR
- Expected Shortfall (CVaR)
- Portfolio beta tracking
- Stress testing scenarios
- Drawdown analysis

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods"""

    HISTORICAL = "HISTORICAL"
    PARAMETRIC = "PARAMETRIC"
    MONTE_CARLO = "MONTE_CARLO"


@dataclass
class VaRResult:
    """Value at Risk calculation result"""

    var_amount: float  # VaR in VND
    var_percent: float  # VaR as % of portfolio
    confidence_level: float  # e.g., 0.95 for 95%
    time_horizon: int  # Days
    method: VaRMethod

    # Additional metrics
    expected_shortfall: float = 0.0  # CVaR
    max_loss_scenario: float = 0.0

    # Interpretation
    interpretation: str = ""


@dataclass
class StressTestResult:
    """Stress test scenario result"""

    scenario_name: str
    portfolio_impact: float  # VND change
    portfolio_impact_pct: float  # % change
    worst_position: str  # Symbol with worst impact
    worst_position_impact: float
    positions_at_risk: int  # Positions with >10% loss

    # Details
    position_impacts: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DrawdownAnalysis:
    """Drawdown analysis result"""

    current_drawdown: float  # Current drawdown %
    max_drawdown: float  # Maximum historical drawdown %
    max_drawdown_date: Optional[datetime] = None
    recovery_days: int = 0  # Days to recover from max DD

    # Risk levels
    is_warning: bool = False  # DD > 10%
    is_critical: bool = False  # DD > 20%

    # Recommendations
    position_reduction: float = 0.0  # Suggested reduction


class PortfolioVaRCalculator:
    """
    Portfolio Value at Risk Calculator

    Calculates VaR using multiple methods:
    1. Historical Simulation
    2. Parametric (Variance-Covariance)
    3. Monte Carlo Simulation

    Vietnam Market Considerations:
    - ±7% daily limit affects tail risk
    - T+2 settlement affects liquidity risk
    - Higher correlation during market stress
    """

    # Vietnam market parameters
    VN_DAILY_LIMIT = 0.07  # ±7% for HOSE
    VN_TRADING_DAYS = 250  # Trading days per year

    # Default confidence levels
    DEFAULT_CONFIDENCE = 0.95
    CONSERVATIVE_CONFIDENCE = 0.99

    def __init__(
        self,
        confidence_level: float = 0.95,
        time_horizon: int = 1,  # Days
        lookback_period: int = 252,  # 1 year
        monte_carlo_simulations: int = 10000,
    ):
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.lookback_period = lookback_period
        self.mc_simulations = monte_carlo_simulations

        # Cache
        self._returns_cache: Dict[str, pd.Series] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None

    def calculate_var(
        self,
        portfolio_value: float,
        positions: Dict[str, Dict],  # {symbol: {value, weight, returns}}
        method: VaRMethod = VaRMethod.HISTORICAL,
        returns_data: Optional[Dict[str, pd.Series]] = None,
    ) -> VaRResult:
        """
        Calculate Value at Risk for portfolio

        Args:
            portfolio_value: Total portfolio value in VND
            positions: Dict of positions with values and weights
            method: VaR calculation method
            returns_data: Historical returns for each position

        Returns:
            VaRResult object
        """
        if not positions:
            return VaRResult(
                var_amount=0,
                var_percent=0,
                confidence_level=self.confidence_level,
                time_horizon=self.time_horizon,
                method=method,
                interpretation="No positions in portfolio",
            )

        # Get or calculate returns
        if returns_data:
            self._returns_cache.update(returns_data)

        # Calculate based on method
        if method == VaRMethod.HISTORICAL:
            var_pct = self._calculate_historical_var(positions)
        elif method == VaRMethod.PARAMETRIC:
            var_pct = self._calculate_parametric_var(positions)
        else:  # Monte Carlo
            var_pct = self._calculate_monte_carlo_var(positions)

        # Apply time horizon scaling (square root of time)
        var_pct_scaled = var_pct * np.sqrt(self.time_horizon)

        # Cap at Vietnam daily limit for single day
        if self.time_horizon == 1:
            var_pct_scaled = min(var_pct_scaled, self.VN_DAILY_LIMIT)

        var_amount = portfolio_value * var_pct_scaled

        # Calculate Expected Shortfall (CVaR)
        es_pct = self._calculate_expected_shortfall(positions)
        es_amount = portfolio_value * es_pct * np.sqrt(self.time_horizon)

        # Generate interpretation
        interpretation = self._generate_interpretation(var_pct_scaled, es_pct)

        return VaRResult(
            var_amount=var_amount,
            var_percent=var_pct_scaled * 100,
            confidence_level=self.confidence_level,
            time_horizon=self.time_horizon,
            method=method,
            expected_shortfall=es_amount,
            max_loss_scenario=portfolio_value * self.VN_DAILY_LIMIT * len(positions),
            interpretation=interpretation,
        )

    def _calculate_historical_var(self, positions: Dict[str, Dict]) -> float:
        """Calculate VaR using historical simulation"""
        # Get portfolio returns
        portfolio_returns = self._get_portfolio_returns(positions)

        if portfolio_returns is None or len(portfolio_returns) < 30:
            # Fallback to parametric with assumed volatility
            return self._estimate_var_from_weights(positions)

        # Calculate VaR as percentile of returns
        var_pct = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)

        return abs(var_pct)

    def _calculate_parametric_var(self, positions: Dict[str, Dict]) -> float:
        """Calculate VaR using variance-covariance method"""
        # Get portfolio volatility
        portfolio_returns = self._get_portfolio_returns(positions)

        if portfolio_returns is None or len(portfolio_returns) < 30:
            return self._estimate_var_from_weights(positions)

        # Calculate portfolio standard deviation
        portfolio_std = portfolio_returns.std()

        # Z-score for confidence level
        from scipy import stats

        z_score = stats.norm.ppf(1 - self.confidence_level)

        var_pct = abs(z_score) * portfolio_std

        return var_pct

    def _calculate_monte_carlo_var(self, positions: Dict[str, Dict]) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        portfolio_returns = self._get_portfolio_returns(positions)

        if portfolio_returns is None or len(portfolio_returns) < 30:
            return self._estimate_var_from_weights(positions)

        # Estimate parameters
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()

        # Generate simulations
        np.random.seed(42)
        simulated_returns = np.random.normal(mean_return, std_return, self.mc_simulations)

        # Calculate VaR from simulations
        var_pct = np.percentile(simulated_returns, (1 - self.confidence_level) * 100)

        return abs(var_pct)

    def _calculate_expected_shortfall(self, positions: Dict[str, Dict]) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        portfolio_returns = self._get_portfolio_returns(positions)

        if portfolio_returns is None or len(portfolio_returns) < 30:
            # Estimate ES as 1.25x VaR
            return self._estimate_var_from_weights(positions) * 1.25

        # Get returns below VaR threshold
        var_threshold = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)

        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]

        if len(tail_returns) == 0:
            return abs(var_threshold)

        return abs(tail_returns.mean())

    def _get_portfolio_returns(self, positions: Dict[str, Dict]) -> Optional[pd.Series]:
        """Calculate weighted portfolio returns"""
        if not self._returns_cache:
            return None

        # Get weights
        total_value = sum(p.get("value", 0) for p in positions.values())
        if total_value == 0:
            return None

        weights = {symbol: pos.get("value", 0) / total_value for symbol, pos in positions.items()}

        # Calculate weighted returns
        portfolio_returns = None

        for symbol, weight in weights.items():
            if symbol in self._returns_cache:
                returns = self._returns_cache[symbol]
                weighted_returns = returns * weight

                if portfolio_returns is None:
                    portfolio_returns = weighted_returns
                else:
                    portfolio_returns = portfolio_returns.add(weighted_returns, fill_value=0)

        return portfolio_returns

    def _estimate_var_from_weights(self, positions: Dict[str, Dict]) -> float:
        """Estimate VaR when historical data unavailable"""
        # Use average Vietnam market volatility (~1.5% daily)
        avg_daily_vol = 0.015

        # Adjust for concentration
        total_value = sum(p.get("value", 0) for p in positions.values())
        if total_value == 0:
            return avg_daily_vol

        weights = [p.get("value", 0) / total_value for p in positions.values()]

        # Herfindahl index for concentration
        hhi = sum(w**2 for w in weights)

        # Higher concentration = higher risk
        concentration_factor = 1 + (hhi - 1 / len(positions)) if len(positions) > 0 else 1

        # Z-score for 95% confidence
        z_score = 1.645

        return avg_daily_vol * z_score * concentration_factor

    def _generate_interpretation(self, var_pct: float, es_pct: float) -> str:
        """Generate human-readable interpretation"""
        var_pct_display = var_pct * 100

        if var_pct_display < 2:
            risk_level = "LOW"
            advice = "Portfolio risk is within acceptable range."
        elif var_pct_display < 5:
            risk_level = "MODERATE"
            advice = "Consider reviewing position sizes."
        elif var_pct_display < 10:
            risk_level = "HIGH"
            advice = "Recommend reducing position sizes or adding hedges."
        else:
            risk_level = "VERY HIGH"
            advice = "Immediate risk reduction recommended."

        return (
            f"Risk Level: {risk_level}. "
            f"With {self.confidence_level*100:.0f}% confidence, "
            f"daily loss will not exceed {var_pct_display:.2f}%. "
            f"In worst {(1-self.confidence_level)*100:.0f}% of cases, "
            f"average loss is {es_pct*100:.2f}%. {advice}"
        )

    def run_stress_test(
        self, portfolio_value: float, positions: Dict[str, Dict], scenario: str = "MARKET_CRASH"
    ) -> StressTestResult:
        """
        Run stress test scenario

        Scenarios:
        - MARKET_CRASH: -15% market drop
        - SECTOR_ROTATION: Banking -20%, Tech +5%
        - FOREIGN_SELLOFF: Foreign-heavy stocks -10%
        - LIQUIDITY_CRISIS: Small caps -25%
        - VN_FLOOR_HIT: All stocks hit floor (-7%)
        """
        scenarios = {
            "MARKET_CRASH": {
                "description": "Market crash (-15%)",
                "default_impact": -0.15,
                "sector_impacts": {
                    "BANKING": -0.18,
                    "REAL_ESTATE": -0.20,
                    "SECURITIES": -0.25,
                    "CONSUMER": -0.10,
                    "TECHNOLOGY": -0.12,
                },
            },
            "SECTOR_ROTATION": {
                "description": "Sector rotation",
                "default_impact": -0.05,
                "sector_impacts": {
                    "BANKING": -0.15,
                    "REAL_ESTATE": -0.12,
                    "TECHNOLOGY": 0.05,
                    "CONSUMER": 0.03,
                },
            },
            "FOREIGN_SELLOFF": {
                "description": "Foreign investor selloff",
                "default_impact": -0.08,
                "high_foreign_impact": -0.15,
            },
            "LIQUIDITY_CRISIS": {
                "description": "Liquidity crisis",
                "default_impact": -0.10,
                "small_cap_impact": -0.25,
                "large_cap_impact": -0.08,
            },
            "VN_FLOOR_HIT": {
                "description": "All stocks hit floor",
                "default_impact": -0.07,  # Vietnam floor limit
            },
        }

        scenario_config = scenarios.get(scenario, scenarios["MARKET_CRASH"])

        # Calculate impact for each position
        position_impacts = {}
        total_impact = 0
        worst_position = ""
        worst_impact = 0
        positions_at_risk = 0

        for symbol, pos in positions.items():
            value = pos.get("value", 0)
            sector = pos.get("sector", "DEFAULT")

            # Get impact for this position
            if scenario == "VN_FLOOR_HIT":
                impact_pct = scenario_config["default_impact"]
            elif scenario == "SECTOR_ROTATION":
                impact_pct = scenario_config["sector_impacts"].get(
                    sector, scenario_config["default_impact"]
                )
            else:
                impact_pct = scenario_config.get("sector_impacts", {}).get(
                    sector, scenario_config["default_impact"]
                )

            impact_amount = value * impact_pct
            position_impacts[symbol] = impact_amount
            total_impact += impact_amount

            if impact_amount < worst_impact:
                worst_impact = impact_amount
                worst_position = symbol

            if impact_pct < -0.10:
                positions_at_risk += 1

        # Generate recommendations
        recommendations = []
        impact_pct = total_impact / portfolio_value if portfolio_value > 0 else 0

        if impact_pct < -0.15:
            recommendations.append("🚨 CRITICAL: Portfolio highly vulnerable to this scenario")
            recommendations.append("Consider reducing overall exposure by 30-50%")
        elif impact_pct < -0.10:
            recommendations.append("⚠️ HIGH RISK: Significant exposure to this scenario")
            recommendations.append("Consider reducing concentrated positions")
        elif impact_pct < -0.05:
            recommendations.append("📊 MODERATE RISK: Some exposure to this scenario")
            recommendations.append("Monitor positions closely")
        else:
            recommendations.append("✅ LOW RISK: Portfolio relatively resilient")

        if worst_position:
            recommendations.append(
                f"Highest risk position: {worst_position} "
                f"({worst_impact/1e6:.1f}M VND potential loss)"
            )

        return StressTestResult(
            scenario_name=scenario_config["description"],
            portfolio_impact=total_impact,
            portfolio_impact_pct=impact_pct * 100,
            worst_position=worst_position,
            worst_position_impact=worst_impact,
            positions_at_risk=positions_at_risk,
            position_impacts=position_impacts,
            recommendations=recommendations,
        )

    def analyze_drawdown(
        self, portfolio_history: pd.Series, current_value: float  # Historical portfolio values
    ) -> DrawdownAnalysis:
        """
        Analyze portfolio drawdown

        Args:
            portfolio_history: Series of historical portfolio values
            current_value: Current portfolio value

        Returns:
            DrawdownAnalysis object
        """
        if portfolio_history is None or len(portfolio_history) < 2:
            return DrawdownAnalysis(current_drawdown=0, max_drawdown=0)

        # Calculate running maximum
        running_max = portfolio_history.expanding().max()

        # Calculate drawdown series
        drawdown = (portfolio_history - running_max) / running_max

        # Current drawdown
        peak = portfolio_history.max()
        current_dd = (current_value - peak) / peak if peak > 0 else 0

        # Maximum drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()

        # Recovery analysis
        recovery_days = 0
        if max_dd < 0:
            # Find recovery point after max drawdown
            post_dd = portfolio_history[portfolio_history.index > max_dd_idx]
            recovery_point = post_dd[post_dd >= running_max[max_dd_idx]]
            if len(recovery_point) > 0:
                recovery_days = (recovery_point.index[0] - max_dd_idx).days

        # Risk flags
        is_warning = current_dd < -0.10
        is_critical = current_dd < -0.20

        # Position reduction recommendation
        if is_critical:
            position_reduction = 0.50  # Reduce 50%
        elif is_warning:
            position_reduction = 0.25  # Reduce 25%
        else:
            position_reduction = 0.0

        return DrawdownAnalysis(
            current_drawdown=current_dd * 100,
            max_drawdown=max_dd * 100,
            max_drawdown_date=max_dd_idx if isinstance(max_dd_idx, datetime) else None,
            recovery_days=recovery_days,
            is_warning=is_warning,
            is_critical=is_critical,
            position_reduction=position_reduction,
        )

    def calculate_portfolio_beta(
        self,
        positions: Dict[str, Dict],
        market_returns: pd.Series,
        position_returns: Dict[str, pd.Series],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate portfolio beta relative to market

        Args:
            positions: Portfolio positions
            market_returns: VNINDEX returns
            position_returns: Returns for each position

        Returns:
            (portfolio_beta, position_betas)
        """
        if market_returns is None or len(market_returns) < 30:
            return 1.0, {}

        market_var = market_returns.var()
        if market_var == 0:
            return 1.0, {}

        # Calculate beta for each position
        position_betas = {}
        total_value = sum(p.get("value", 0) for p in positions.values())

        for symbol, pos in positions.items():
            if symbol in position_returns:
                returns = position_returns[symbol]

                # Align returns
                aligned = pd.concat([returns, market_returns], axis=1).dropna()
                if len(aligned) < 30:
                    position_betas[symbol] = 1.0
                    continue

                # Calculate covariance and beta
                cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
                beta = cov / market_var
                position_betas[symbol] = beta
            else:
                position_betas[symbol] = 1.0

        # Calculate weighted portfolio beta
        portfolio_beta = 0
        for symbol, pos in positions.items():
            weight = pos.get("value", 0) / total_value if total_value > 0 else 0
            beta = position_betas.get(symbol, 1.0)
            portfolio_beta += weight * beta

        return portfolio_beta, position_betas


# Singleton instance
_var_calculator: Optional[PortfolioVaRCalculator] = None


def get_var_calculator() -> PortfolioVaRCalculator:
    """Get singleton VaR calculator"""
    global _var_calculator
    if _var_calculator is None:
        _var_calculator = PortfolioVaRCalculator()
    return _var_calculator


# Convenience functions
def calculate_portfolio_var(
    portfolio_value: float, positions: Dict[str, Dict], confidence: float = 0.95
) -> VaRResult:
    """Quick VaR calculation"""
    calculator = get_var_calculator()
    calculator.confidence_level = confidence
    return calculator.calculate_var(portfolio_value, positions)


def run_stress_test(
    portfolio_value: float, positions: Dict[str, Dict], scenario: str = "MARKET_CRASH"
) -> StressTestResult:
    """Quick stress test"""
    calculator = get_var_calculator()
    return calculator.run_stress_test(portfolio_value, positions, scenario)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("🧪 TESTING PORTFOLIO VAR CALCULATOR")
    print("=" * 70 + "\n")

    # Sample portfolio
    portfolio_value = 500_000_000  # 500M VND
    positions = {
        "VCB": {"value": 150_000_000, "sector": "BANKING"},
        "FPT": {"value": 100_000_000, "sector": "TECHNOLOGY"},
        "VNM": {"value": 100_000_000, "sector": "CONSUMER"},
        "HPG": {"value": 80_000_000, "sector": "INDUSTRIAL"},
        "VHM": {"value": 70_000_000, "sector": "REAL_ESTATE"},
    }

    calculator = get_var_calculator()

    # Calculate VaR
    print("📊 VALUE AT RISK ANALYSIS")
    print("-" * 50)

    var_result = calculator.calculate_var(portfolio_value, positions)
    print(f"Portfolio Value: {portfolio_value/1e6:.0f}M VND")
    print(f"VaR (95%, 1-day): {var_result.var_amount/1e6:.2f}M VND ({var_result.var_percent:.2f}%)")
    print(f"Expected Shortfall: {var_result.expected_shortfall/1e6:.2f}M VND")
    print(f"\n{var_result.interpretation}")

    # Stress tests
    print("\n" + "-" * 50)
    print("🔥 STRESS TEST RESULTS")
    print("-" * 50)

    scenarios = ["MARKET_CRASH", "SECTOR_ROTATION", "VN_FLOOR_HIT"]

    for scenario in scenarios:
        result = calculator.run_stress_test(portfolio_value, positions, scenario)
        print(f"\n{result.scenario_name}:")
        print(
            f"  Impact: {result.portfolio_impact/1e6:.2f}M VND ({result.portfolio_impact_pct:.1f}%)"
        )
        print(f"  Worst Position: {result.worst_position}")
        print(f"  Positions at Risk: {result.positions_at_risk}")

    print("\n" + "=" * 70)
    print("✅ VaR calculator test completed!")
    print("=" * 70)
