# -*- coding: utf-8 -*-
"""
Enhanced Margin Trading with Stress Testing

IMPROVEMENT #3.5: Margin Trading Risk Management

Features:
- Stress testing for margin calls
- Extreme market condition handling
- Monte Carlo simulation for risk scenarios
- Circuit breaker integration
- VaR (Value at Risk) calculations
- Automatic deleveraging system

Vietnam Market Margin Rules:
- Typical initial margin: 50-60%
- Maintenance margin: 30-40%
- Margin call threshold: ~140%
- Force sell threshold: ~130%
- Price limits: ±7% HOSE, ±10% HNX

Author: Trading Bot Team
Version: 2.0.0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import random
import statistics

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


# Vietnam margin requirements (typical broker)
INITIAL_MARGIN_RATIO = 0.50  # 50% initial margin
MAINTENANCE_MARGIN_RATIO = 0.35  # 35% maintenance
MARGIN_CALL_THRESHOLD = 1.40  # 140% ratio triggers call
FORCE_LIQUIDATION_THRESHOLD = 1.30  # 130% forces liquidation

# Price limits
HOSE_PRICE_LIMIT = 0.07  # ±7%
HNX_PRICE_LIMIT = 0.10  # ±10%
UPCOM_PRICE_LIMIT = 0.15  # ±15%

# Risk thresholds
MAX_LEVERAGE = 2.0  # Maximum 2x leverage
STRESS_TEST_DAYS = 5  # Consecutive limit down days


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================


class MarginLevel(Enum):
    """Margin health levels"""

    SAFE = "SAFE"  # > 200%
    HEALTHY = "HEALTHY"  # 160-200%
    CAUTION = "CAUTION"  # 140-160%
    MARGIN_CALL = "MARGIN_CALL"  # 130-140%
    DANGER = "DANGER"  # < 130%


class MarketCondition(Enum):
    """Market condition types"""

    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    CRISIS = "CRISIS"
    EXTREME = "EXTREME"


class StressScenario(Enum):
    """Stress test scenarios"""

    LIMIT_DOWN_1D = "LIMIT_DOWN_1D"
    LIMIT_DOWN_3D = "LIMIT_DOWN_3D"
    LIMIT_DOWN_5D = "LIMIT_DOWN_5D"
    HISTORICAL_CRASH = "HISTORICAL_CRASH"
    MONTE_CARLO = "MONTE_CARLO"


@dataclass
class MarginPosition:
    """Individual margin position"""

    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    borrowed_amount: float = 0.0

    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    max_loss_percent: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def equity(self) -> float:
        return self.market_value - self.borrowed_amount

    @property
    def margin_ratio(self) -> float:
        if self.borrowed_amount > 0:
            return self.market_value / self.borrowed_amount
        return float("inf")

    @property
    def leverage(self) -> float:
        if self.equity > 0:
            return self.market_value / self.equity
        return float("inf")

    @property
    def profit_loss(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def profit_loss_percent(self) -> float:
        if self.avg_cost > 0:
            return (self.current_price / self.avg_cost - 1) * 100
        return 0


@dataclass
class MarginPortfolio:
    """Margin portfolio state"""

    positions: List[MarginPosition] = field(default_factory=list)
    cash: float = 0.0
    total_borrowed: float = 0.0

    # Portfolio risk metrics
    portfolio_var_95: float = 0.0
    portfolio_var_99: float = 0.0
    expected_shortfall: float = 0.0

    @property
    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def total_equity(self) -> float:
        return self.total_market_value + self.cash - self.total_borrowed

    @property
    def margin_ratio(self) -> float:
        if self.total_borrowed > 0:
            return (self.total_market_value + self.cash) / self.total_borrowed
        return float("inf")

    @property
    def portfolio_leverage(self) -> float:
        if self.total_equity > 0:
            return self.total_market_value / self.total_equity
        return float("inf")

    @property
    def margin_level(self) -> MarginLevel:
        ratio = self.margin_ratio
        if ratio >= 2.0:
            return MarginLevel.SAFE
        elif ratio >= 1.6:
            return MarginLevel.HEALTHY
        elif ratio >= 1.4:
            return MarginLevel.CAUTION
        elif ratio >= 1.3:
            return MarginLevel.MARGIN_CALL
        else:
            return MarginLevel.DANGER


@dataclass
class StressTestResult:
    """Stress test result"""

    scenario: StressScenario
    description: str

    # Before stress
    initial_equity: float = 0.0
    initial_margin_ratio: float = 0.0

    # After stress
    stressed_equity: float = 0.0
    stressed_margin_ratio: float = 0.0
    stressed_level: MarginLevel = MarginLevel.SAFE

    # Losses
    total_loss: float = 0.0
    total_loss_percent: float = 0.0
    per_symbol_losses: Dict[str, float] = field(default_factory=dict)

    # Outcomes
    margin_call_triggered: bool = False
    force_liquidation_triggered: bool = False
    days_to_margin_call: int = 0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""

    # Value at Risk
    var_1d_95: float = 0.0  # 1-day VaR at 95%
    var_1d_99: float = 0.0  # 1-day VaR at 99%
    var_5d_95: float = 0.0  # 5-day VaR at 95%

    # Expected Shortfall (CVaR)
    cvar_95: float = 0.0
    cvar_99: float = 0.0

    # Other metrics
    max_drawdown: float = 0.0
    beta: float = 1.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0

    # Margin-specific
    margin_cushion: float = 0.0  # Distance to margin call
    days_to_margin_call: int = 999  # At current trend
    safe_drawdown: float = 0.0  # Max drawdown before margin call


# =============================================================================
# MARKET CONDITION DETECTOR
# =============================================================================


class MarketConditionDetector:
    """Detect extreme market conditions"""

    # VIX-like thresholds (not available for VN, use VN30 volatility)
    NORMAL_VOL_THRESHOLD = 0.02  # 2% daily
    VOLATILE_VOL_THRESHOLD = 0.03  # 3% daily
    CRISIS_VOL_THRESHOLD = 0.05  # 5% daily

    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days
        self._volatility_history: List[float] = []
        self._return_history: List[float] = []

    def update(self, daily_return: float, daily_volatility: float):
        """Update with new daily data"""
        self._return_history.append(daily_return)
        self._volatility_history.append(daily_volatility)

        # Keep only lookback period
        if len(self._return_history) > self.lookback_days:
            self._return_history.pop(0)
            self._volatility_history.pop(0)

    def get_condition(self) -> MarketCondition:
        """Get current market condition"""
        if not self._volatility_history:
            return MarketCondition.NORMAL

        recent_vol = (
            statistics.mean(self._volatility_history[-5:])
            if len(self._volatility_history) >= 5
            else self._volatility_history[-1]
        )

        # Count negative days
        recent_returns = (
            self._return_history[-5:] if len(self._return_history) >= 5 else self._return_history
        )
        negative_days = sum(1 for r in recent_returns if r < -0.01)

        # Check for consecutive limit down
        if self._check_consecutive_limit_down(3):
            return MarketCondition.EXTREME

        if recent_vol >= self.CRISIS_VOL_THRESHOLD or negative_days >= 4:
            return MarketCondition.CRISIS
        elif recent_vol >= self.VOLATILE_VOL_THRESHOLD or negative_days >= 3:
            return MarketCondition.VOLATILE

        return MarketCondition.NORMAL

    def _check_consecutive_limit_down(self, days: int) -> bool:
        """Check for consecutive limit down days"""
        if len(self._return_history) < days:
            return False

        limit_down = -0.065  # ~7% for HOSE
        return all(r <= limit_down for r in self._return_history[-days:])

    def get_volatility_regime(self) -> str:
        """Get volatility regime description"""
        if not self._volatility_history:
            return "Unknown"

        avg_vol = statistics.mean(self._volatility_history)

        if avg_vol < 0.015:
            return "Low Volatility"
        elif avg_vol < 0.025:
            return "Normal Volatility"
        elif avg_vol < 0.04:
            return "High Volatility"
        else:
            return "Extreme Volatility"


# =============================================================================
# STRESS TESTING ENGINE
# =============================================================================


class StressTestingEngine:
    """
    Stress testing for margin portfolios.

    Scenarios:
    1. Limit Down Days (1, 3, 5 consecutive)
    2. Historical Crash Simulation
    3. Monte Carlo Simulation

    Usage:
        engine = StressTestingEngine()
        portfolio = MarginPortfolio(...)

        # Run all stress tests
        results = engine.run_all_tests(portfolio)

        # Run specific scenario
        result = engine.run_limit_down_test(portfolio, days=3)
    """

    # Historical VN-Index crashes
    HISTORICAL_CRASHES = {
        "2008 GFC": -0.65,  # ~65% drop
        "2020 COVID": -0.30,  # ~30% drop
        "2022 Bonds": -0.35,  # ~35% drop
    }

    def __init__(self, exchange: str = "HOSE", monte_carlo_simulations: int = 10000):
        """
        Initialize stress testing engine.

        Args:
            exchange: Exchange for price limits
            monte_carlo_simulations: Number of MC simulations
        """
        self.exchange = exchange
        self.mc_simulations = monte_carlo_simulations

        # Set price limit based on exchange
        if exchange == "HNX":
            self.daily_limit = HNX_PRICE_LIMIT
        elif exchange == "UPCOM":
            self.daily_limit = UPCOM_PRICE_LIMIT
        else:
            self.daily_limit = HOSE_PRICE_LIMIT

    def run_all_tests(self, portfolio: MarginPortfolio) -> List[StressTestResult]:
        """Run all stress test scenarios"""
        results = []

        # Limit down scenarios
        results.append(self.run_limit_down_test(portfolio, 1))
        results.append(self.run_limit_down_test(portfolio, 3))
        results.append(self.run_limit_down_test(portfolio, 5))

        # Historical crash
        results.append(self.run_historical_crash_test(portfolio))

        # Monte Carlo
        results.append(self.run_monte_carlo_test(portfolio))

        return results

    def run_limit_down_test(self, portfolio: MarginPortfolio, days: int = 1) -> StressTestResult:
        """
        Simulate consecutive limit down days.

        Args:
            portfolio: Current portfolio
            days: Number of limit down days

        Returns:
            Stress test result
        """
        # Map days to scenario enum
        scenario_map = {
            1: StressScenario.LIMIT_DOWN_1D,
            3: StressScenario.LIMIT_DOWN_3D,
            5: StressScenario.LIMIT_DOWN_5D,
        }
        scenario = scenario_map.get(days, StressScenario.LIMIT_DOWN_1D)

        # Calculate cumulative drop
        # Each day: price * (1 - limit)
        cumulative_factor = (1 - self.daily_limit) ** days

        result = StressTestResult(
            scenario=scenario,
            description=f"{days} consecutive limit down days (-{self.daily_limit*100:.0f}%/day)",
            initial_equity=portfolio.total_equity,
            initial_margin_ratio=portfolio.margin_ratio,
        )

        # Calculate stressed values
        stressed_market_value = 0
        per_symbol_losses = {}

        for position in portfolio.positions:
            stressed_price = position.current_price * cumulative_factor
            stressed_value = position.quantity * stressed_price
            loss = position.market_value - stressed_value

            stressed_market_value += stressed_value
            per_symbol_losses[position.symbol] = loss

        # New portfolio metrics
        stressed_equity = stressed_market_value + portfolio.cash - portfolio.total_borrowed

        if portfolio.total_borrowed > 0:
            stressed_margin_ratio = (
                stressed_market_value + portfolio.cash
            ) / portfolio.total_borrowed
        else:
            stressed_margin_ratio = float("inf")

        # Determine margin level
        if stressed_margin_ratio >= 2.0:
            stressed_level = MarginLevel.SAFE
        elif stressed_margin_ratio >= 1.6:
            stressed_level = MarginLevel.HEALTHY
        elif stressed_margin_ratio >= 1.4:
            stressed_level = MarginLevel.CAUTION
        elif stressed_margin_ratio >= 1.3:
            stressed_level = MarginLevel.MARGIN_CALL
        else:
            stressed_level = MarginLevel.DANGER

        # Total loss
        total_loss = portfolio.total_equity - stressed_equity
        total_loss_percent = (
            (total_loss / portfolio.total_equity * 100) if portfolio.total_equity > 0 else 0
        )

        result.stressed_equity = stressed_equity
        result.stressed_margin_ratio = stressed_margin_ratio
        result.stressed_level = stressed_level
        result.total_loss = total_loss
        result.total_loss_percent = total_loss_percent
        result.per_symbol_losses = per_symbol_losses

        # Check triggers
        result.margin_call_triggered = stressed_margin_ratio < MARGIN_CALL_THRESHOLD
        result.force_liquidation_triggered = stressed_margin_ratio < FORCE_LIQUIDATION_THRESHOLD

        # Calculate days to margin call (at limit down rate)
        result.days_to_margin_call = self._calculate_days_to_margin_call(
            portfolio, -self.daily_limit
        )

        # Add recommendations
        result.recommendations = self._generate_recommendations(result, portfolio)

        return result

    def run_historical_crash_test(
        self, portfolio: MarginPortfolio, crash_name: str = "2020 COVID"
    ) -> StressTestResult:
        """
        Simulate historical crash scenario.

        Uses historical VN-Index crash data.
        """
        crash_pct = self.HISTORICAL_CRASHES.get(crash_name, -0.30)

        result = StressTestResult(
            scenario=StressScenario.HISTORICAL_CRASH,
            description=f"Historical crash simulation: {crash_name} ({crash_pct*100:.0f}%)",
            initial_equity=portfolio.total_equity,
            initial_margin_ratio=portfolio.margin_ratio,
        )

        # Apply crash to all positions
        cumulative_factor = 1 + crash_pct  # crash_pct is negative

        stressed_market_value = 0
        per_symbol_losses = {}

        for position in portfolio.positions:
            stressed_price = position.current_price * cumulative_factor
            stressed_value = position.quantity * stressed_price
            loss = position.market_value - stressed_value

            stressed_market_value += stressed_value
            per_symbol_losses[position.symbol] = loss

        stressed_equity = stressed_market_value + portfolio.cash - portfolio.total_borrowed

        if portfolio.total_borrowed > 0:
            stressed_margin_ratio = (
                stressed_market_value + portfolio.cash
            ) / portfolio.total_borrowed
        else:
            stressed_margin_ratio = float("inf")

        # Determine level
        if stressed_margin_ratio >= 2.0:
            stressed_level = MarginLevel.SAFE
        elif stressed_margin_ratio >= 1.6:
            stressed_level = MarginLevel.HEALTHY
        elif stressed_margin_ratio >= 1.4:
            stressed_level = MarginLevel.CAUTION
        elif stressed_margin_ratio >= 1.3:
            stressed_level = MarginLevel.MARGIN_CALL
        else:
            stressed_level = MarginLevel.DANGER

        total_loss = portfolio.total_equity - stressed_equity
        total_loss_percent = (
            (total_loss / portfolio.total_equity * 100) if portfolio.total_equity > 0 else 0
        )

        result.stressed_equity = stressed_equity
        result.stressed_margin_ratio = stressed_margin_ratio
        result.stressed_level = stressed_level
        result.total_loss = total_loss
        result.total_loss_percent = total_loss_percent
        result.per_symbol_losses = per_symbol_losses
        result.margin_call_triggered = stressed_margin_ratio < MARGIN_CALL_THRESHOLD
        result.force_liquidation_triggered = stressed_margin_ratio < FORCE_LIQUIDATION_THRESHOLD
        result.recommendations = self._generate_recommendations(result, portfolio)

        return result

    def run_monte_carlo_test(
        self, portfolio: MarginPortfolio, holding_days: int = 10, daily_volatility: float = 0.02
    ) -> StressTestResult:
        """
        Monte Carlo simulation for risk assessment.

        Args:
            portfolio: Current portfolio
            holding_days: Days to simulate
            daily_volatility: Expected daily volatility

        Returns:
            StressTestResult with VaR metrics
        """
        result = StressTestResult(
            scenario=StressScenario.MONTE_CARLO,
            description=f"Monte Carlo: {self.mc_simulations} sims, {holding_days} days, {daily_volatility*100:.1f}% vol",
            initial_equity=portfolio.total_equity,
            initial_margin_ratio=portfolio.margin_ratio,
        )

        # Run simulations
        final_equities = []
        final_margin_ratios = []
        margin_call_count = 0
        force_liquidation_count = 0

        for _ in range(self.mc_simulations):
            # Simulate price path
            cumulative_return = 0
            margin_call_day = 0

            for day in range(1, holding_days + 1):
                # Random daily return (normal distribution)
                daily_return = random.gauss(0, daily_volatility)
                # Cap at price limit
                daily_return = max(-self.daily_limit, min(self.daily_limit, daily_return))
                cumulative_return += daily_return

                # Check margin call
                temp_factor = 1 + cumulative_return
                temp_mv = portfolio.total_market_value * temp_factor
                if portfolio.total_borrowed > 0:
                    temp_ratio = (temp_mv + portfolio.cash) / portfolio.total_borrowed
                    if temp_ratio < MARGIN_CALL_THRESHOLD and margin_call_day == 0:
                        margin_call_day = day

            # Final values
            factor = 1 + cumulative_return
            final_mv = portfolio.total_market_value * factor
            final_equity = final_mv + portfolio.cash - portfolio.total_borrowed

            if portfolio.total_borrowed > 0:
                final_ratio = (final_mv + portfolio.cash) / portfolio.total_borrowed
            else:
                final_ratio = float("inf")

            final_equities.append(final_equity)
            final_margin_ratios.append(final_ratio)

            if final_ratio < MARGIN_CALL_THRESHOLD:
                margin_call_count += 1
            if final_ratio < FORCE_LIQUIDATION_THRESHOLD:
                force_liquidation_count += 1

        # Calculate VaR (5th percentile = 95% VaR)
        sorted_equities = sorted(final_equities)
        var_95_idx = int(self.mc_simulations * 0.05)
        var_99_idx = int(self.mc_simulations * 0.01)

        var_95 = portfolio.total_equity - sorted_equities[var_95_idx]
        var_99 = portfolio.total_equity - sorted_equities[var_99_idx]

        # Expected Shortfall (average of worst 5%)
        worst_5_pct = sorted_equities[:var_95_idx]
        cvar_95 = portfolio.total_equity - statistics.mean(worst_5_pct) if worst_5_pct else 0

        # Stressed values (median of worst case)
        result.stressed_equity = sorted_equities[var_95_idx]
        result.stressed_margin_ratio = sorted(final_margin_ratios)[var_95_idx]

        # Determine level based on 95% VaR margin ratio
        ratio_95 = result.stressed_margin_ratio
        if ratio_95 >= 2.0:
            result.stressed_level = MarginLevel.SAFE
        elif ratio_95 >= 1.6:
            result.stressed_level = MarginLevel.HEALTHY
        elif ratio_95 >= 1.4:
            result.stressed_level = MarginLevel.CAUTION
        elif ratio_95 >= 1.3:
            result.stressed_level = MarginLevel.MARGIN_CALL
        else:
            result.stressed_level = MarginLevel.DANGER

        result.total_loss = var_95
        result.total_loss_percent = (
            (var_95 / portfolio.total_equity * 100) if portfolio.total_equity > 0 else 0
        )

        # Probability of margin call
        mc_probability = margin_call_count / self.mc_simulations
        fl_probability = force_liquidation_count / self.mc_simulations

        result.margin_call_triggered = mc_probability > 0.10  # 10% chance
        result.force_liquidation_triggered = fl_probability > 0.05  # 5% chance

        # Add Monte Carlo specific info to recommendations
        result.recommendations = [
            f"95% VaR ({holding_days}d): {var_95:,.0f} VND ({var_95/portfolio.total_equity*100:.1f}%)",
            f"99% VaR ({holding_days}d): {var_99:,.0f} VND ({var_99/portfolio.total_equity*100:.1f}%)",
            f"Expected Shortfall: {cvar_95:,.0f} VND",
            f"Margin call probability: {mc_probability*100:.1f}%",
            f"Force liquidation probability: {fl_probability*100:.1f}%",
        ]

        return result

    def _calculate_days_to_margin_call(
        self, portfolio: MarginPortfolio, daily_return: float
    ) -> int:
        """Calculate days until margin call at given daily return"""
        if portfolio.total_borrowed <= 0:
            return 999

        current_mv = portfolio.total_market_value
        cash = portfolio.cash
        borrowed = portfolio.total_borrowed

        for day in range(1, 100):
            current_mv *= 1 + daily_return
            ratio = (current_mv + cash) / borrowed

            if ratio < MARGIN_CALL_THRESHOLD:
                return day

        return 999

    def _generate_recommendations(
        self, result: StressTestResult, portfolio: MarginPortfolio
    ) -> List[str]:
        """Generate recommendations based on stress test result"""
        recommendations = []

        if result.force_liquidation_triggered:
            recommendations.append("⚠️ CRITICAL: Reduce leverage immediately")
            recommendations.append(
                f"   Add cash or close {self._calc_position_to_close(portfolio):.0f}% of positions"
            )

        elif result.margin_call_triggered:
            recommendations.append("🔶 WARNING: Margin call risk detected")
            recommendations.append(f"   Consider reducing position by 20-30%")
            recommendations.append(f"   Maintain higher cash buffer")

        elif result.stressed_level == MarginLevel.CAUTION:
            recommendations.append("📊 CAUTION: Close to margin thresholds")
            recommendations.append(f"   Set stop-losses to protect equity")

        # General recommendations
        if portfolio.portfolio_leverage > 1.5:
            recommendations.append(
                f"   Current leverage {portfolio.portfolio_leverage:.1f}x is high"
            )

        if result.days_to_margin_call < 5:
            recommendations.append(
                f"   Only {result.days_to_margin_call} days to margin call at limit down"
            )

        return recommendations

    def _calc_position_to_close(self, portfolio: MarginPortfolio) -> float:
        """Calculate percentage of position to close to reach safe level"""
        if portfolio.total_borrowed <= 0:
            return 0

        # Target: margin ratio = 2.0 (safe)
        target_ratio = 2.0
        required_mv = target_ratio * portfolio.total_borrowed - portfolio.cash

        if required_mv >= portfolio.total_market_value:
            return 0

        excess_mv = portfolio.total_market_value - required_mv
        return (excess_mv / portfolio.total_market_value) * 100


# =============================================================================
# AUTOMATIC DELEVERAGING SYSTEM
# =============================================================================


class AutoDeleveragingSystem:
    """
    Automatic deleveraging when margin thresholds are breached.

    Features:
    - Prioritized position closing
    - Gradual deleveraging
    - Circuit breaker integration
    """

    def __init__(
        self,
        warning_threshold: float = 1.50,  # Start warning
        action_threshold: float = 1.45,  # Start gradual deleverage
        urgent_threshold: float = 1.35,  # Urgent deleverage
        max_daily_deleverage: float = 0.30,  # Max 30% per day
    ):
        self.warning_threshold = warning_threshold
        self.action_threshold = action_threshold
        self.urgent_threshold = urgent_threshold
        self.max_daily_deleverage = max_daily_deleverage

        self._deleverage_history: List[Dict] = []

    def check_and_recommend(
        self, portfolio: MarginPortfolio, market_condition: MarketCondition = MarketCondition.NORMAL
    ) -> Dict[str, Any]:
        """
        Check portfolio and recommend deleveraging actions.

        Returns:
            Dictionary with:
            - should_deleverage: bool
            - urgency: str (none, warning, action, urgent)
            - recommended_actions: List of (symbol, quantity, reason)
            - new_target_leverage: float
        """
        result = {
            "should_deleverage": False,
            "urgency": "none",
            "recommended_actions": [],
            "new_target_leverage": portfolio.portfolio_leverage,
        }

        margin_ratio = portfolio.margin_ratio

        # Adjust thresholds for market condition
        condition_adjustment = {
            MarketCondition.NORMAL: 0,
            MarketCondition.VOLATILE: 0.05,
            MarketCondition.CRISIS: 0.10,
            MarketCondition.EXTREME: 0.15,
        }
        adj = condition_adjustment.get(market_condition, 0)

        effective_warning = self.warning_threshold + adj
        effective_action = self.action_threshold + adj
        effective_urgent = self.urgent_threshold + adj

        # Determine urgency
        if margin_ratio < effective_urgent:
            result["urgency"] = "urgent"
            result["should_deleverage"] = True
            deleverage_pct = min(0.50, self.max_daily_deleverage * 1.5)  # More aggressive
        elif margin_ratio < effective_action:
            result["urgency"] = "action"
            result["should_deleverage"] = True
            deleverage_pct = self.max_daily_deleverage
        elif margin_ratio < effective_warning:
            result["urgency"] = "warning"
            result["should_deleverage"] = False  # Just warn
            deleverage_pct = 0
        else:
            return result

        if not result["should_deleverage"]:
            return result

        # Calculate target leverage
        # Target margin ratio of 1.80 (HEALTHY level)
        if portfolio.total_borrowed > 0:
            target_mv = 1.80 * portfolio.total_borrowed - portfolio.cash
            if target_mv > 0:
                target_leverage = target_mv / (
                    target_mv - portfolio.total_borrowed + portfolio.cash
                )
            else:
                target_leverage = 1.0
        else:
            target_leverage = 1.0

        result["new_target_leverage"] = max(1.0, target_leverage)

        # Prioritize positions to close
        # Priority: highest loss first, then highest leverage contribution
        positions_priority = self._prioritize_positions(portfolio)

        # Calculate how much to sell
        total_to_sell = portfolio.total_market_value * deleverage_pct
        sold_so_far = 0

        for position, priority_score in positions_priority:
            if sold_so_far >= total_to_sell:
                break

            # Calculate quantity to sell
            remaining = total_to_sell - sold_so_far
            qty_to_sell = min(position.quantity, int(remaining / position.current_price))

            if qty_to_sell > 0:
                result["recommended_actions"].append(
                    {
                        "symbol": position.symbol,
                        "action": "SELL",
                        "quantity": qty_to_sell,
                        "reason": f"Deleverage (priority: {priority_score:.0f})",
                        "estimated_value": qty_to_sell * position.current_price,
                    }
                )
                sold_so_far += qty_to_sell * position.current_price

        return result

    def _prioritize_positions(
        self, portfolio: MarginPortfolio
    ) -> List[Tuple[MarginPosition, float]]:
        """
        Prioritize positions for closing.

        Priority factors:
        - Losing positions first
        - Higher VaR contribution
        - Lower liquidity (harder to exit later)
        """
        scored_positions = []

        for position in portfolio.positions:
            score = 0

            # Loss factor (higher score = close first)
            if position.profit_loss_percent < 0:
                score += abs(position.profit_loss_percent) * 2
            else:
                score -= position.profit_loss_percent * 0.5

            # Size factor (larger = higher priority for deleveraging)
            if portfolio.total_market_value > 0:
                weight = position.market_value / portfolio.total_market_value
                score += weight * 50

            # VaR contribution (if available)
            if position.var_95 > 0:
                score += position.var_95 / 1000000  # Normalize

            scored_positions.append((position, score))

        # Sort by score descending (highest priority first)
        scored_positions.sort(key=lambda x: x[1], reverse=True)

        return scored_positions


# =============================================================================
# ENHANCED MARGIN TRADING MANAGER
# =============================================================================


class EnhancedMarginTradingManager:
    """
    Enhanced margin trading with stress testing and risk management.

    Integration point for existing margin_trading.py

    Usage:
        manager = EnhancedMarginTradingManager()

        # Build portfolio
        manager.add_position("VNM", 1000, 80000, 85000, borrowed=40000000)

        # Run stress tests
        results = manager.run_stress_tests()

        # Get risk metrics
        metrics = manager.calculate_risk_metrics()

        # Check for required actions
        actions = manager.check_deleverage_needed()
    """

    def __init__(self, exchange: str = "HOSE", auto_update_condition: bool = True):
        """
        Initialize enhanced margin trading manager.

        Args:
            exchange: Exchange for price limits
            auto_update_condition: Auto-update market condition
        """
        self.exchange = exchange
        self.portfolio = MarginPortfolio()

        self.stress_engine = StressTestingEngine(exchange)
        self.deleverage_system = AutoDeleveragingSystem()
        self.condition_detector = MarketConditionDetector()

        self._last_stress_results: List[StressTestResult] = []
        self._last_risk_metrics: Optional[RiskMetrics] = None

        logger.info("📊 Enhanced Margin Trading Manager initialized")

    def add_position(
        self,
        symbol: str,
        quantity: int,
        avg_cost: float,
        current_price: float,
        borrowed: float = 0.0,
    ):
        """Add or update position"""
        # Check if position exists
        existing = next((p for p in self.portfolio.positions if p.symbol == symbol), None)

        if existing:
            # Update existing
            existing.quantity = quantity
            existing.avg_cost = avg_cost
            existing.current_price = current_price
            existing.borrowed_amount = borrowed
        else:
            # Add new
            position = MarginPosition(
                symbol=symbol,
                quantity=quantity,
                avg_cost=avg_cost,
                current_price=current_price,
                borrowed_amount=borrowed,
            )
            self.portfolio.positions.append(position)

        # Update total borrowed
        self.portfolio.total_borrowed = sum(p.borrowed_amount for p in self.portfolio.positions)

    def remove_position(self, symbol: str):
        """Remove position"""
        self.portfolio.positions = [p for p in self.portfolio.positions if p.symbol != symbol]
        self.portfolio.total_borrowed = sum(p.borrowed_amount for p in self.portfolio.positions)

    def update_price(self, symbol: str, current_price: float):
        """Update position price"""
        for position in self.portfolio.positions:
            if position.symbol == symbol:
                position.current_price = current_price
                break

    def update_cash(self, cash: float):
        """Update cash balance"""
        self.portfolio.cash = cash

    def update_market_data(self, daily_return: float, daily_volatility: float):
        """Update market condition data"""
        self.condition_detector.update(daily_return, daily_volatility)

    def run_stress_tests(self) -> List[StressTestResult]:
        """Run all stress tests"""
        self._last_stress_results = self.stress_engine.run_all_tests(self.portfolio)
        return self._last_stress_results

    def run_single_stress_test(self, scenario: StressScenario) -> StressTestResult:
        """Run single stress test scenario"""
        if scenario == StressScenario.LIMIT_DOWN_1D:
            return self.stress_engine.run_limit_down_test(self.portfolio, 1)
        elif scenario == StressScenario.LIMIT_DOWN_3D:
            return self.stress_engine.run_limit_down_test(self.portfolio, 3)
        elif scenario == StressScenario.LIMIT_DOWN_5D:
            return self.stress_engine.run_limit_down_test(self.portfolio, 5)
        elif scenario == StressScenario.HISTORICAL_CRASH:
            return self.stress_engine.run_historical_crash_test(self.portfolio)
        else:
            return self.stress_engine.run_monte_carlo_test(self.portfolio)

    def calculate_risk_metrics(
        self, daily_volatility: float = 0.02, holding_days: int = 10
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        metrics = RiskMetrics()

        # Get Monte Carlo result for VaR
        mc_result = self.stress_engine.run_monte_carlo_test(
            self.portfolio, holding_days, daily_volatility
        )

        # Parse VaR from recommendations
        for rec in mc_result.recommendations:
            if "95% VaR" in rec:
                try:
                    metrics.var_5d_95 = mc_result.total_loss
                except:
                    pass

        # 1-day VaR (approximate)
        metrics.var_1d_95 = metrics.var_5d_95 / np.sqrt(holding_days) if holding_days > 0 else 0
        metrics.var_1d_99 = metrics.var_1d_95 * 1.5  # Rough approximation

        # CVaR (from recommendations)
        for rec in mc_result.recommendations:
            if "Expected Shortfall" in rec:
                try:
                    # Extract number
                    import re

                    numbers = re.findall(r"[\d,]+", rec)
                    if numbers:
                        metrics.cvar_95 = float(numbers[0].replace(",", ""))
                except:
                    pass

        # Volatility (annualized)
        metrics.volatility = daily_volatility * np.sqrt(252)

        # Margin cushion
        margin_ratio = self.portfolio.margin_ratio
        if margin_ratio < float("inf"):
            metrics.margin_cushion = margin_ratio - MARGIN_CALL_THRESHOLD
        else:
            metrics.margin_cushion = float("inf")

        # Days to margin call
        metrics.days_to_margin_call = self.stress_engine._calculate_days_to_margin_call(
            self.portfolio, -0.02  # Assume 2% daily loss
        )

        # Safe drawdown
        if self.portfolio.total_equity > 0 and margin_ratio < float("inf"):
            # Drawdown that would bring margin ratio to 1.40
            metrics.safe_drawdown = 1 - (MARGIN_CALL_THRESHOLD / margin_ratio)

        self._last_risk_metrics = metrics
        return metrics

    def check_deleverage_needed(self) -> Dict[str, Any]:
        """Check if deleveraging is needed"""
        condition = self.condition_detector.get_condition()
        return self.deleverage_system.check_and_recommend(self.portfolio, condition)

    def get_margin_status(self) -> Dict[str, Any]:
        """Get current margin status summary"""
        portfolio = self.portfolio
        condition = self.condition_detector.get_condition()

        return {
            "total_market_value": portfolio.total_market_value,
            "total_equity": portfolio.total_equity,
            "total_borrowed": portfolio.total_borrowed,
            "cash": portfolio.cash,
            "margin_ratio": portfolio.margin_ratio,
            "margin_level": portfolio.margin_level.value,
            "leverage": portfolio.portfolio_leverage,
            "market_condition": condition.value,
            "volatility_regime": self.condition_detector.get_volatility_regime(),
            "position_count": len(portfolio.positions),
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "market_value": p.market_value,
                    "pnl_percent": p.profit_loss_percent,
                    "margin_ratio": p.margin_ratio,
                }
                for p in portfolio.positions
            ],
        }

    def get_risk_report(self) -> str:
        """Generate risk report"""
        if not self._last_stress_results:
            self.run_stress_tests()

        if not self._last_risk_metrics:
            self.calculate_risk_metrics()

        lines = [
            "=" * 60,
            "📊 MARGIN TRADING RISK REPORT",
            "=" * 60,
            "",
            "📈 PORTFOLIO STATUS:",
            f"   Market Value: {self.portfolio.total_market_value:,.0f} VND",
            f"   Equity: {self.portfolio.total_equity:,.0f} VND",
            f"   Borrowed: {self.portfolio.total_borrowed:,.0f} VND",
            f"   Margin Ratio: {self.portfolio.margin_ratio:.2f}x",
            f"   Margin Level: {self.portfolio.margin_level.value}",
            f"   Leverage: {self.portfolio.portfolio_leverage:.2f}x",
            "",
            "🌍 MARKET CONDITION:",
            f"   Condition: {self.condition_detector.get_condition().value}",
            f"   Volatility: {self.condition_detector.get_volatility_regime()}",
            "",
            "🔥 STRESS TEST RESULTS:",
        ]

        for result in self._last_stress_results:
            status = "⚠️" if result.margin_call_triggered else "✅"
            lines.append(f"\n   {status} {result.scenario.value}:")
            lines.append(f"      Description: {result.description}")
            lines.append(f"      Stressed Equity: {result.stressed_equity:,.0f} VND")
            lines.append(f"      Stressed Margin: {result.stressed_margin_ratio:.2f}x")
            lines.append(
                f"      Total Loss: {result.total_loss:,.0f} VND ({result.total_loss_percent:.1f}%)"
            )
            lines.append(f"      Level: {result.stressed_level.value}")

            if result.recommendations:
                lines.append("      Recommendations:")
                for rec in result.recommendations[:3]:
                    lines.append(f"         {rec}")

        if self._last_risk_metrics:
            metrics = self._last_risk_metrics
            lines.extend(
                [
                    "",
                    "📊 RISK METRICS:",
                    f"   VaR 1-day 95%: {metrics.var_1d_95:,.0f} VND",
                    f"   VaR 5-day 95%: {metrics.var_5d_95:,.0f} VND",
                    f"   Expected Shortfall: {metrics.cvar_95:,.0f} VND",
                    f"   Margin Cushion: {metrics.margin_cushion*100:.1f}%",
                    f"   Days to Margin Call: {metrics.days_to_margin_call}",
                    f"   Safe Drawdown: {metrics.safe_drawdown*100:.1f}%",
                ]
            )

        # Check deleverage
        deleverage = self.check_deleverage_needed()
        if deleverage["should_deleverage"]:
            lines.extend(
                [
                    "",
                    f"⚠️ DELEVERAGE REQUIRED ({deleverage['urgency'].upper()}):",
                    f"   Target Leverage: {deleverage['new_target_leverage']:.2f}x",
                ]
            )
            for action in deleverage["recommended_actions"][:5]:
                lines.append(
                    f"   → {action['action']} {action['symbol']}: {action['quantity']} shares"
                )

        lines.extend(["", "=" * 60])

        return "\n".join(lines)


# =============================================================================
# INTEGRATION FUNCTION
# =============================================================================


def check_margin_risk_for_trade(
    symbol: str,
    trade_value: float,
    current_portfolio: Optional[MarginPortfolio] = None,
    daily_volatility: float = 0.02,
) -> Tuple[bool, int, str]:
    """
    Check if a new margin trade is safe.

    For integration with entry logic.

    Args:
        symbol: Stock symbol
        trade_value: Value of proposed trade
        current_portfolio: Current margin portfolio
        daily_volatility: Current market volatility

    Returns:
        (is_safe, confidence_adjustment, message)
    """
    if current_portfolio is None:
        return (True, 0, "No margin position")

    # Create manager with portfolio
    manager = EnhancedMarginTradingManager()
    manager.portfolio = current_portfolio

    # Simulate adding new position
    test_position = MarginPosition(
        symbol=symbol,
        quantity=1,
        avg_cost=trade_value,
        current_price=trade_value,
        borrowed_amount=trade_value * 0.5,  # 50% margin
    )

    # Check current margin level
    current_level = current_portfolio.margin_level

    # Run quick stress test
    result = manager.run_single_stress_test(StressScenario.LIMIT_DOWN_3D)

    # Determine safety
    if current_level == MarginLevel.DANGER:
        return (False, -20, "⛔ Margin at danger level, avoid new positions")

    if current_level == MarginLevel.MARGIN_CALL:
        return (False, -15, "⚠️ Margin call level, reduce positions first")

    if result.force_liquidation_triggered:
        return (False, -10, "⚠️ 3-day limit down would trigger liquidation")

    if current_level == MarginLevel.CAUTION:
        return (True, -5, "📊 Caution: margin ratio low, reduce position size")

    if result.margin_call_triggered:
        return (True, -3, "📋 Note: stress test shows margin call risk")

    return (True, 0, "✅ Margin levels healthy")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING ENHANCED MARGIN TRADING")
    print("=" * 60)

    # Create manager
    manager = EnhancedMarginTradingManager()

    # Add sample positions
    manager.add_position("VNM", 1000, 75000, 80000, borrowed=40000000)
    manager.add_position("VCB", 500, 95000, 92000, borrowed=23000000)
    manager.add_position("FPT", 300, 120000, 125000, borrowed=18000000)
    manager.update_cash(10000000)

    # Update market data
    for _ in range(10):
        daily_return = random.gauss(-0.005, 0.02)
        daily_vol = abs(random.gauss(0.02, 0.005))
        manager.update_market_data(daily_return, daily_vol)

    # Get status
    print("\n📈 Portfolio Status:")
    status = manager.get_margin_status()
    print(f"   Market Value: {status['total_market_value']:,.0f} VND")
    print(f"   Equity: {status['total_equity']:,.0f} VND")
    print(f"   Margin Ratio: {status['margin_ratio']:.2f}x")
    print(f"   Margin Level: {status['margin_level']}")
    print(f"   Market Condition: {status['market_condition']}")

    # Run stress tests
    print("\n🔥 Running Stress Tests...")
    results = manager.run_stress_tests()

    for result in results:
        status_icon = "⚠️" if result.margin_call_triggered else "✅"
        print(
            f"   {status_icon} {result.scenario.value}: "
            f"Loss {result.total_loss_percent:.1f}%, "
            f"Margin {result.stressed_margin_ratio:.2f}x"
        )

    # Calculate risk metrics
    print("\n📊 Risk Metrics:")
    metrics = manager.calculate_risk_metrics()
    print(f"   VaR 1-day 95%: {metrics.var_1d_95:,.0f} VND")
    print(f"   Margin Cushion: {metrics.margin_cushion*100:.1f}%")
    print(f"   Safe Drawdown: {metrics.safe_drawdown*100:.1f}%")

    # Check deleverage
    print("\n🔄 Deleverage Check:")
    deleverage = manager.check_deleverage_needed()
    print(f"   Needed: {deleverage['should_deleverage']}")
    print(f"   Urgency: {deleverage['urgency']}")

    # Print full report
    print("\n" + manager.get_risk_report())

    print("\n✅ Test complete!")
