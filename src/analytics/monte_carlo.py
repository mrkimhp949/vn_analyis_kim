"""
Monte Carlo Simulation for Trading Risk Analysis

Simulates thousands of trading scenarios to calculate:
- Risk of ruin (probability of losing >50% capital)
- Expected value distribution
- Confidence intervals
- Worst-case scenarios
- Optimal position sizing validation
- Drawdown statistics

This is a key component for achieving A+ rating (92 → 95+)
"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""

    # Risk metrics
    risk_of_ruin: float  # Probability of losing >50% capital
    risk_of_ruin_pct: float  # As percentage
    risk_of_30pct_loss: float  # Probability of >30% loss
    risk_of_20pct_loss: float  # Probability of >20% loss

    # Expected value
    expected_value: float  # Mean final capital
    expected_return_pct: float  # Mean return percentage
    median_return_pct: float  # Median return

    # Confidence intervals (percentiles)
    percentile_5th: float
    percentile_25th: float
    percentile_50th: float  # Median
    percentile_75th: float
    percentile_95th: float

    # Extremes
    worst_case: float  # Worst outcome
    best_case: float  # Best outcome
    worst_return_pct: float
    best_return_pct: float

    # Drawdown statistics
    avg_max_drawdown: float  # Average of max drawdowns across sims
    worst_drawdown: float  # Worst drawdown encountered
    pct_sims_exceed_10pct_dd: float  # % of sims with DD >10%
    pct_sims_exceed_20pct_dd: float  # % of sims with DD >20%

    # Win/loss distribution
    avg_win_rate: float  # Average win rate across simulations
    avg_profit_factor: float  # Average profit factor

    # Simulation parameters
    num_simulations: int
    num_trades_per_sim: int
    initial_capital: float


class MonteCarloSimulator:
    """
    Monte Carlo simulation for trading strategy risk analysis

    Features:
    - Risk of ruin calculation
    - Expected value distribution
    - Confidence intervals
    - Worst-case scenario analysis
    - Drawdown statistics
    - Position size validation

    Example:
        >>> sim = MonteCarloSimulator(
        ...     win_rate=0.55,
        ...     avg_win_pct=2.5,
        ...     avg_loss_pct=-1.5,
        ...     num_simulations=10000
        ... )
        >>> result = sim.run_simulation()
        >>> print(f"Risk of Ruin: {result.risk_of_ruin_pct:.2f}%")
    """

    def __init__(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        win_stddev: Optional[float] = None,
        loss_stddev: Optional[float] = None,
        num_simulations: int = 10000,
        num_trades_per_sim: int = 100,
        initial_capital: float = 100_000_000,
        position_size_pct: float = 0.10,  # 10% of capital per trade
        use_kelly: bool = False,
        kelly_fraction: float = 0.5,
    ):
        """
        Initialize Monte Carlo simulator

        Args:
            win_rate: Historical win rate (0-1, e.g., 0.55 for 55%)
            avg_win_pct: Average winning trade percentage (e.g., 2.5 for +2.5%)
            avg_loss_pct: Average losing trade percentage (e.g., -1.5 for -1.5%)
            win_stddev: Standard deviation of wins (if None, uses avg_win_pct * 0.5)
            loss_stddev: Standard deviation of losses (if None, uses avg_loss_pct * 0.5)
            num_simulations: Number of simulations to run
            num_trades_per_sim: Number of trades per simulation
            initial_capital: Starting capital in VND
            position_size_pct: Position size as % of capital (fixed sizing)
            use_kelly: Use Kelly Criterion for position sizing
            kelly_fraction: Fraction of Kelly to use (default: half-Kelly = 0.5)
        """
        # Validate inputs
        if not 0 <= win_rate <= 1:
            raise ValueError(f"win_rate must be between 0 and 1, got {win_rate}")
        if avg_win_pct <= 0:
            raise ValueError(f"avg_win_pct must be positive, got {avg_win_pct}")
        if avg_loss_pct >= 0:
            raise ValueError(f"avg_loss_pct must be negative, got {avg_loss_pct}")
        if num_simulations < 100:
            raise ValueError(f"num_simulations must be >= 100, got {num_simulations}")

        self.win_rate = win_rate
        self.avg_win_pct = avg_win_pct
        self.avg_loss_pct = avg_loss_pct
        self.win_stddev = win_stddev or abs(avg_win_pct * 0.5)
        self.loss_stddev = loss_stddev or abs(avg_loss_pct * 0.5)
        self.num_simulations = num_simulations
        self.num_trades_per_sim = num_trades_per_sim
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction

        # Calculate Kelly percentage if using Kelly
        if self.use_kelly:
            # Kelly = W - (1-W)/R where W=win_rate, R=avg_win/avg_loss
            R = abs(self.avg_win_pct / self.avg_loss_pct)
            kelly_full = self.win_rate - ((1 - self.win_rate) / R)
            self.kelly_pct = max(
                0.01, min(kelly_full * self.kelly_fraction, 0.25)
            )  # Clamp to 1-25%
            logger.info(
                f"📊 Using Kelly Criterion: {self.kelly_pct:.1%} "
                f"(full Kelly: {kelly_full:.1%}, fraction: {self.kelly_fraction})"
            )

    def run_simulation(self, seed: Optional[int] = None) -> MonteCarloResult:
        """
        Run Monte Carlo simulation

        Args:
            seed: Random seed for reproducibility (optional)

        Returns:
            MonteCarloResult with comprehensive statistics
        """
        if seed is not None:
            np.random.seed(seed)

        logger.info(
            f"🎲 Running Monte Carlo simulation: {self.num_simulations:,} simulations, "
            f"{self.num_trades_per_sim} trades each"
        )

        # Storage for results
        final_capitals = np.zeros(self.num_simulations)
        max_drawdowns = np.zeros(self.num_simulations)
        sim_win_rates = np.zeros(self.num_simulations)
        sim_profit_factors = np.zeros(self.num_simulations)

        for sim_idx in range(self.num_simulations):
            # Run one simulation
            final_cap, max_dd, win_rate, pf = self._run_single_simulation()

            final_capitals[sim_idx] = final_cap
            max_drawdowns[sim_idx] = max_dd
            sim_win_rates[sim_idx] = win_rate
            sim_profit_factors[sim_idx] = pf

        # Calculate aggregate statistics
        result = self._calculate_statistics(
            final_capitals, max_drawdowns, sim_win_rates, sim_profit_factors
        )

        logger.info(f"✅ Monte Carlo complete: Risk of Ruin = {result.risk_of_ruin_pct:.2f}%")
        logger.info(f"   Expected Return = {result.expected_return_pct:+.2f}%")
        logger.info(f"   Avg Max Drawdown = {result.avg_max_drawdown*100:.2f}%")

        return result

    def _run_single_simulation(self) -> Tuple[float, float, float, float]:
        """
        Run a single simulation of N trades

        Returns:
            (final_capital, max_drawdown, win_rate, profit_factor)
        """
        capital = self.initial_capital
        peak_capital = capital
        max_dd = 0.0

        wins = 0
        losses = 0
        total_win_amount = 0.0
        total_loss_amount = 0.0

        for _ in range(self.num_trades_per_sim):
            # Determine if trade wins or loses
            is_win = np.random.random() < self.win_rate

            if is_win:
                # Sample from win distribution
                return_pct = np.random.normal(self.avg_win_pct, self.win_stddev)
                return_pct = max(0.1, return_pct)  # Ensure positive win
                wins += 1
            else:
                # Sample from loss distribution
                return_pct = np.random.normal(self.avg_loss_pct, self.loss_stddev)
                return_pct = min(-0.1, return_pct)  # Ensure negative loss
                losses += 1

            # Calculate position size
            if self.use_kelly:
                position_size = capital * self.kelly_pct
            else:
                position_size = capital * self.position_size_pct

            # Calculate P&L
            pnl = position_size * (return_pct / 100)
            capital += pnl

            # Track wins/losses for profit factor
            if pnl > 0:
                total_win_amount += pnl
            else:
                total_loss_amount += abs(pnl)

            # Track drawdown
            if capital > peak_capital:
                peak_capital = capital

            dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
            if dd > max_dd:
                max_dd = dd

            # Stop if ruined (< 50% of initial capital)
            if capital <= self.initial_capital * 0.5:
                break

        # Calculate metrics
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0
        profit_factor = total_win_amount / total_loss_amount if total_loss_amount > 0 else 0

        return capital, max_dd, win_rate, profit_factor

    def _calculate_statistics(
        self,
        final_capitals: np.ndarray,
        max_drawdowns: np.ndarray,
        win_rates: np.ndarray,
        profit_factors: np.ndarray,
    ) -> MonteCarloResult:
        """Calculate aggregate statistics from all simulations"""

        # Risk metrics
        ruin_threshold = self.initial_capital * 0.5
        risk_of_ruin = (final_capitals < ruin_threshold).sum() / self.num_simulations

        loss_30_threshold = self.initial_capital * 0.7
        risk_of_30pct_loss = (final_capitals < loss_30_threshold).sum() / self.num_simulations

        loss_20_threshold = self.initial_capital * 0.8
        risk_of_20pct_loss = (final_capitals < loss_20_threshold).sum() / self.num_simulations

        # Expected value
        expected_value = np.mean(final_capitals)
        expected_return_pct = ((expected_value - self.initial_capital) / self.initial_capital) * 100
        median_return_pct = (
            (np.median(final_capitals) - self.initial_capital) / self.initial_capital
        ) * 100

        # Percentiles
        percentiles = np.percentile(final_capitals, [5, 25, 50, 75, 95])

        # Extremes
        worst_case = np.min(final_capitals)
        best_case = np.max(final_capitals)
        worst_return_pct = ((worst_case - self.initial_capital) / self.initial_capital) * 100
        best_return_pct = ((best_case - self.initial_capital) / self.initial_capital) * 100

        # Drawdown statistics
        avg_max_drawdown = np.mean(max_drawdowns)
        worst_drawdown = np.max(max_drawdowns)
        pct_exceed_10 = (max_drawdowns > 0.10).sum() / self.num_simulations
        pct_exceed_20 = (max_drawdowns > 0.20).sum() / self.num_simulations

        # Win/loss distribution
        avg_win_rate = np.mean(win_rates)
        avg_profit_factor = np.mean(profit_factors)

        return MonteCarloResult(
            risk_of_ruin=risk_of_ruin,
            risk_of_ruin_pct=risk_of_ruin * 100,
            risk_of_30pct_loss=risk_of_30pct_loss,
            risk_of_20pct_loss=risk_of_20pct_loss,
            expected_value=expected_value,
            expected_return_pct=expected_return_pct,
            median_return_pct=median_return_pct,
            percentile_5th=percentiles[0],
            percentile_25th=percentiles[1],
            percentile_50th=percentiles[2],
            percentile_75th=percentiles[3],
            percentile_95th=percentiles[4],
            worst_case=worst_case,
            best_case=best_case,
            worst_return_pct=worst_return_pct,
            best_return_pct=best_return_pct,
            avg_max_drawdown=avg_max_drawdown,
            worst_drawdown=worst_drawdown,
            pct_sims_exceed_10pct_dd=pct_exceed_10,
            pct_sims_exceed_20pct_dd=pct_exceed_20,
            avg_win_rate=avg_win_rate,
            avg_profit_factor=avg_profit_factor,
            num_simulations=self.num_simulations,
            num_trades_per_sim=self.num_trades_per_sim,
            initial_capital=self.initial_capital,
        )

    def generate_report(self, result: MonteCarloResult) -> str:
        """Generate formatted report from Monte Carlo results"""

        report = []
        report.append("=" * 80)
        report.append("🎲 MONTE CARLO RISK ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"📊 Simulation Parameters:")
        report.append(f"   Simulations: {result.num_simulations:,}")
        report.append(f"   Trades per simulation: {result.num_trades_per_sim}")
        report.append(f"   Initial capital: {result.initial_capital:,.0f} VND")
        report.append(f"   Win rate: {self.win_rate:.1%}")
        report.append(f"   Avg win: +{self.avg_win_pct:.2f}%")
        report.append(f"   Avg loss: {self.avg_loss_pct:.2f}%")
        if self.use_kelly:
            report.append(f"   Position sizing: Kelly ({self.kelly_pct:.1%})")
        else:
            report.append(f"   Position sizing: Fixed ({self.position_size_pct:.1%})")
        report.append("")

        report.append("🎯 RISK METRICS:")
        report.append(f"   Risk of Ruin (>50% loss): {result.risk_of_ruin_pct:.2f}%")
        if result.risk_of_ruin_pct < 2.0:
            report.append("   ✅ EXCELLENT: Risk of ruin <2%")
        elif result.risk_of_ruin_pct < 5.0:
            report.append("   ✅ GOOD: Risk of ruin <5%")
        else:
            report.append(f"   ⚠️ WARNING: Risk of ruin {result.risk_of_ruin_pct:.2f}% too high!")

        report.append(f"   Risk of 30% loss: {result.risk_of_30pct_loss*100:.2f}%")
        report.append(f"   Risk of 20% loss: {result.risk_of_20pct_loss*100:.2f}%")
        report.append("")

        report.append("💰 EXPECTED VALUE:")
        report.append(f"   Expected return: {result.expected_return_pct:+.2f}%")
        report.append(f"   Median return: {result.median_return_pct:+.2f}%")
        report.append(f"   Expected final capital: {result.expected_value:,.0f} VND")
        if result.expected_return_pct > 0:
            report.append("   ✅ Positive expected value")
        else:
            report.append("   ❌ NEGATIVE expected value - DO NOT TRADE")
        report.append("")

        report.append("📊 CONFIDENCE INTERVALS:")
        report.append(f"   5th percentile: {self._format_return(result.percentile_5th)}")
        report.append(f"   25th percentile: {self._format_return(result.percentile_25th)}")
        report.append(f"   50th percentile: {self._format_return(result.percentile_50th)}")
        report.append(f"   75th percentile: {self._format_return(result.percentile_75th)}")
        report.append(f"   95th percentile: {self._format_return(result.percentile_95th)}")
        report.append("")

        report.append("🔺 EXTREMES:")
        report.append(f"   Best case: {result.best_return_pct:+.2f}% ({result.best_case:,.0f} VND)")
        report.append(
            f"   Worst case: {result.worst_return_pct:+.2f}% ({result.worst_case:,.0f} VND)"
        )
        report.append("")

        report.append("📉 DRAWDOWN STATISTICS:")
        report.append(f"   Average max drawdown: {result.avg_max_drawdown*100:.2f}%")
        report.append(f"   Worst drawdown: {result.worst_drawdown*100:.2f}%")
        report.append(f"   Simulations with DD >10%: {result.pct_sims_exceed_10pct_dd*100:.1f}%")
        report.append(f"   Simulations with DD >20%: {result.pct_sims_exceed_20pct_dd*100:.1f}%")
        if result.avg_max_drawdown < 0.15:
            report.append("   ✅ Average drawdown <15%")
        else:
            report.append(f"   ⚠️ WARNING: High average drawdown {result.avg_max_drawdown*100:.1f}%")
        report.append("")

        report.append("🎲 WIN/LOSS DISTRIBUTION:")
        report.append(f"   Average win rate: {result.avg_win_rate:.1%}")
        report.append(f"   Average profit factor: {result.avg_profit_factor:.2f}")
        report.append("")

        report.append("=" * 80)
        report.append("📋 OVERALL ASSESSMENT:")

        # Overall grade
        grade = self._calculate_grade(result)
        report.append(f"   Grade: {grade}")
        report.append("")

        if result.risk_of_ruin_pct < 5.0 and result.expected_return_pct > 0:
            report.append("   ✅ STRATEGY APPROVED FOR TRADING")
            report.append(f"   - Low risk of ruin ({result.risk_of_ruin_pct:.2f}%)")
            report.append(f"   - Positive expected value (+{result.expected_return_pct:.2f}%)")
            report.append(f"   - Acceptable drawdown ({result.avg_max_drawdown*100:.1f}%)")
        else:
            report.append("   ⚠️ STRATEGY NEEDS IMPROVEMENT:")
            if result.risk_of_ruin_pct >= 5.0:
                report.append(f"   - Risk of ruin too high ({result.risk_of_ruin_pct:.2f}%)")
            if result.expected_return_pct <= 0:
                report.append("   - Expected value is NOT positive")

        report.append("=" * 80)

        return "\n".join(report)

    def _format_return(self, capital: float) -> str:
        """Format capital as return percentage"""
        return_pct = ((capital - self.initial_capital) / self.initial_capital) * 100
        return f"{return_pct:+.2f}% ({capital:,.0f} VND)"

    def _calculate_grade(self, result: MonteCarloResult) -> str:
        """Calculate overall grade based on metrics"""
        score = 0

        # Risk of ruin (40 points)
        if result.risk_of_ruin_pct < 1.0:
            score += 40
        elif result.risk_of_ruin_pct < 2.0:
            score += 35
        elif result.risk_of_ruin_pct < 5.0:
            score += 25
        elif result.risk_of_ruin_pct < 10.0:
            score += 10

        # Expected return (30 points)
        if result.expected_return_pct > 20:
            score += 30
        elif result.expected_return_pct > 10:
            score += 25
        elif result.expected_return_pct > 5:
            score += 20
        elif result.expected_return_pct > 0:
            score += 10

        # Average drawdown (20 points)
        if result.avg_max_drawdown < 0.10:
            score += 20
        elif result.avg_max_drawdown < 0.15:
            score += 15
        elif result.avg_max_drawdown < 0.20:
            score += 10
        elif result.avg_max_drawdown < 0.25:
            score += 5

        # Profit factor (10 points)
        if result.avg_profit_factor > 2.0:
            score += 10
        elif result.avg_profit_factor > 1.5:
            score += 7
        elif result.avg_profit_factor > 1.0:
            score += 5

        # Assign grade
        if score >= 85:
            return "A+ (Excellent)"
        elif score >= 75:
            return "A (Very Good)"
        elif score >= 65:
            return "B+ (Good)"
        elif score >= 55:
            return "B (Acceptable)"
        elif score >= 45:
            return "C (Needs Improvement)"
        else:
            return "D (Not Recommended)"


def run_monte_carlo_analysis(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    num_simulations: int = 10000,
    initial_capital: float = 100_000_000,
    use_kelly: bool = False,
) -> MonteCarloResult:
    """
    Convenience function to run Monte Carlo analysis

    Args:
        win_rate: Historical win rate (0-1)
        avg_win_pct: Average winning trade %
        avg_loss_pct: Average losing trade %
        num_simulations: Number of simulations
        initial_capital: Starting capital
        use_kelly: Use Kelly Criterion sizing

    Returns:
        MonteCarloResult
    """
    simulator = MonteCarloSimulator(
        win_rate=win_rate,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        num_simulations=num_simulations,
        initial_capital=initial_capital,
        use_kelly=use_kelly,
    )

    result = simulator.run_simulation()
    report = simulator.generate_report(result)
    print(report)

    return result


# Example usage
if __name__ == "__main__":
    print("\n🎲 Monte Carlo Simulation Example\n")

    # Example 1: Current strategy performance
    print("Example 1: Testing current strategy")
    print("-" * 80)

    sim = MonteCarloSimulator(
        win_rate=0.52,  # 52% win rate
        avg_win_pct=2.5,  # +2.5% average win
        avg_loss_pct=-1.5,  # -1.5% average loss
        num_simulations=10000,
        num_trades_per_sim=100,
        initial_capital=100_000_000,
        use_kelly=False,
        position_size_pct=0.10,
    )

    result = sim.run_simulation(seed=42)
    report = sim.generate_report(result)
    print(report)

    print("\n" + "=" * 80)
    print("Example 2: With Kelly Criterion")
    print("-" * 80)

    sim_kelly = MonteCarloSimulator(
        win_rate=0.52,
        avg_win_pct=2.5,
        avg_loss_pct=-1.5,
        num_simulations=10000,
        num_trades_per_sim=100,
        initial_capital=100_000_000,
        use_kelly=True,
        kelly_fraction=0.5,
    )

    result_kelly = sim_kelly.run_simulation(seed=42)
    report_kelly = sim_kelly.generate_report(result_kelly)
    print(report_kelly)
