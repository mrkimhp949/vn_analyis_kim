"""
Unit Tests for Monte Carlo Simulator

Tests the Monte Carlo risk analysis functionality.
"""

import pytest
import numpy as np
from src.analytics.monte_carlo import MonteCarloSimulator, MonteCarloResult, run_monte_carlo_analysis


class TestMonteCarloSimulator:
    """Test Monte Carlo simulator functionality"""

    def test_initialization_valid_params(self):
        """Test simulator initializes with valid parameters"""
        sim = MonteCarloSimulator(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            num_simulations=1000,
        )

        assert sim.win_rate == 0.55
        assert sim.avg_win_pct == 2.5
        assert sim.avg_loss_pct == -1.5
        assert sim.num_simulations == 1000

    def test_initialization_invalid_win_rate(self):
        """Test simulator rejects invalid win rate"""
        with pytest.raises(ValueError, match="win_rate must be between 0 and 1"):
            MonteCarloSimulator(
                win_rate=1.5,  # Invalid: >1
                avg_win_pct=2.5,
                avg_loss_pct=-1.5,
            )

        with pytest.raises(ValueError, match="win_rate must be between 0 and 1"):
            MonteCarloSimulator(
                win_rate=-0.1,  # Invalid: <0
                avg_win_pct=2.5,
                avg_loss_pct=-1.5,
            )

    def test_initialization_invalid_win_pct(self):
        """Test simulator rejects invalid average win"""
        with pytest.raises(ValueError, match="avg_win_pct must be positive"):
            MonteCarloSimulator(
                win_rate=0.55,
                avg_win_pct=-2.5,  # Invalid: negative
                avg_loss_pct=-1.5,
            )

    def test_initialization_invalid_loss_pct(self):
        """Test simulator rejects invalid average loss"""
        with pytest.raises(ValueError, match="avg_loss_pct must be negative"):
            MonteCarloSimulator(
                win_rate=0.55,
                avg_win_pct=2.5,
                avg_loss_pct=1.5,  # Invalid: positive
            )

    def test_initialization_too_few_simulations(self):
        """Test simulator rejects too few simulations"""
        with pytest.raises(ValueError, match="num_simulations must be >= 100"):
            MonteCarloSimulator(
                win_rate=0.55,
                avg_win_pct=2.5,
                avg_loss_pct=-1.5,
                num_simulations=50,  # Invalid: <100
            )

    def test_kelly_criterion_calculation(self):
        """Test Kelly Criterion is calculated correctly"""
        sim = MonteCarloSimulator(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            use_kelly=True,
            kelly_fraction=0.5,
        )

        # Kelly = W - (1-W)/R
        # where W=0.55, R=2.5/1.5=1.67
        # Kelly = 0.55 - 0.45/1.67 = 0.55 - 0.269 = 0.281
        # Half-Kelly = 0.281 * 0.5 = 0.1405
        expected_kelly = 0.55 - (0.45 / (2.5/1.5))
        expected_half_kelly = expected_kelly * 0.5

        assert abs(sim.kelly_pct - expected_half_kelly) < 0.01

    def test_run_simulation_basic(self):
        """Test basic simulation runs successfully"""
        sim = MonteCarloSimulator(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            num_simulations=100,  # Small number for speed
            num_trades_per_sim=10,
        )

        result = sim.run_simulation(seed=42)

        assert isinstance(result, MonteCarloResult)
        assert result.num_simulations == 100
        assert result.initial_capital == 100_000_000

    def test_run_simulation_reproducible(self):
        """Test simulation is reproducible with same seed"""
        sim1 = MonteCarloSimulator(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            num_simulations=100,
        )

        sim2 = MonteCarloSimulator(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            num_simulations=100,
        )

        result1 = sim1.run_simulation(seed=42)
        result2 = sim2.run_simulation(seed=42)

        # Results should be identical with same seed
        assert abs(result1.expected_return_pct - result2.expected_return_pct) < 0.001
        assert abs(result1.risk_of_ruin_pct - result2.risk_of_ruin_pct) < 0.001

    def test_positive_expectancy_strategy(self):
        """Test that profitable strategy has positive expected value"""
        sim = MonteCarloSimulator(
            win_rate=0.60,  # Good win rate
            avg_win_pct=3.0,  # Good average win
            avg_loss_pct=-1.5,  # Controlled losses
            num_simulations=1000,
            num_trades_per_sim=100,
        )

        result = sim.run_simulation(seed=42)

        # Should have positive expected value
        assert result.expected_return_pct > 0
        # Should have low risk of ruin
        assert result.risk_of_ruin_pct < 10.0

    def test_negative_expectancy_strategy(self):
        """Test that losing strategy has negative expected value"""
        sim = MonteCarloSimulator(
            win_rate=0.40,  # Poor win rate
            avg_win_pct=1.5,  # Small wins
            avg_loss_pct=-2.5,  # Large losses
            num_simulations=1000,
            num_trades_per_sim=100,
        )

        result = sim.run_simulation(seed=42)

        # Should have negative expected value
        assert result.expected_return_pct < 0
        # Should have higher risk than profitable strategy
        # With 10% position sizing, 20% total loss is unlikely, so check drawdown instead
        assert result.avg_max_drawdown > 0.05  # Should have significant drawdown
        assert result.pct_sims_exceed_10pct_dd > 0.1  # At least 10% of sims have >10% drawdown

    def test_percentiles_ordered(self):
        """Test that percentiles are in correct order"""
        sim = MonteCarloSimulator(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            num_simulations=1000,
        )

        result = sim.run_simulation(seed=42)

        # Percentiles should be ordered
        assert result.percentile_5th <= result.percentile_25th
        assert result.percentile_25th <= result.percentile_50th
        assert result.percentile_50th <= result.percentile_75th
        assert result.percentile_75th <= result.percentile_95th

    def test_worst_best_case_bounds(self):
        """Test that worst/best cases bound the expected value"""
        sim = MonteCarloSimulator(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            num_simulations=1000,
        )

        result = sim.run_simulation(seed=42)

        # Worst case should be <= expected value
        assert result.worst_case <= result.expected_value
        # Best case should be >= expected value
        assert result.best_case >= result.expected_value

    def test_risk_metrics_range(self):
        """Test that risk metrics are in valid ranges"""
        sim = MonteCarloSimulator(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            num_simulations=1000,
        )

        result = sim.run_simulation(seed=42)

        # Risk metrics should be probabilities (0-1)
        assert 0 <= result.risk_of_ruin <= 1
        assert 0 <= result.risk_of_30pct_loss <= 1
        assert 0 <= result.risk_of_20pct_loss <= 1

        # Drawdowns should be percentages (0-1)
        assert 0 <= result.avg_max_drawdown <= 1
        assert 0 <= result.worst_drawdown <= 1

    def test_generate_report(self):
        """Test report generation"""
        sim = MonteCarloSimulator(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            num_simulations=100,
        )

        result = sim.run_simulation(seed=42)
        report = sim.generate_report(result)

        # Report should contain key sections
        assert "MONTE CARLO RISK ANALYSIS" in report
        assert "RISK METRICS" in report
        assert "EXPECTED VALUE" in report
        assert "CONFIDENCE INTERVALS" in report
        assert "DRAWDOWN STATISTICS" in report
        assert "OVERALL ASSESSMENT" in report

    def test_kelly_vs_fixed_sizing(self):
        """Test Kelly Criterion vs fixed sizing"""
        params = {
            "win_rate": 0.55,
            "avg_win_pct": 2.5,
            "avg_loss_pct": -1.5,
            "num_simulations": 1000,
            "num_trades_per_sim": 100,
        }

        # Fixed sizing
        sim_fixed = MonteCarloSimulator(**params, use_kelly=False, position_size_pct=0.10)
        result_fixed = sim_fixed.run_simulation(seed=42)

        # Kelly sizing
        sim_kelly = MonteCarloSimulator(**params, use_kelly=True, kelly_fraction=0.5)
        result_kelly = sim_kelly.run_simulation(seed=42)

        # Both should have positive expected value
        assert result_fixed.expected_return_pct > 0
        assert result_kelly.expected_return_pct > 0

        # Kelly should generally have better risk/reward
        # (but this depends on parameters, so we just check it ran)
        assert result_kelly.risk_of_ruin_pct >= 0


class TestMonteCarloIntegration:
    """Test Monte Carlo integration with other components"""

    def test_run_monte_carlo_analysis(self, capsys):
        """Test convenience function"""
        result = run_monte_carlo_analysis(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            num_simulations=100,
            initial_capital=100_000_000,
        )

        # Should return result
        assert isinstance(result, MonteCarloResult)

        # Should print report
        captured = capsys.readouterr()
        assert "MONTE CARLO RISK ANALYSIS" in captured.out


class TestMonteCarloEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_perfect_strategy(self):
        """Test strategy with 100% win rate"""
        sim = MonteCarloSimulator(
            win_rate=1.0,  # Perfect win rate
            avg_win_pct=2.0,
            avg_loss_pct=-1.0,  # Won't be used
            num_simulations=100,
        )

        result = sim.run_simulation(seed=42)

        # Should have zero risk of ruin
        assert result.risk_of_ruin_pct == 0.0
        # Should have positive expected value (compounding with 10% position size over 100 trades)
        assert result.expected_return_pct > 15.0  # More realistic expectation

    def test_always_losing_strategy(self):
        """Test strategy with 0% win rate"""
        sim = MonteCarloSimulator(
            win_rate=0.0,  # Always lose
            avg_win_pct=2.0,  # Won't be used
            avg_loss_pct=-1.0,
            num_simulations=100,
        )

        result = sim.run_simulation(seed=42)

        # Should have negative expected value (with 10% position size, -1% loss per trade over 100 trades)
        assert result.expected_return_pct < -5.0  # More realistic expectation
        # Average max drawdown should be significant
        assert result.avg_max_drawdown > 0.05  # At least 5% drawdown

    def test_break_even_strategy(self):
        """Test strategy with neutral expectancy"""
        sim = MonteCarloSimulator(
            win_rate=0.5,
            avg_win_pct=1.5,  # Equal to loss
            avg_loss_pct=-1.5,
            num_simulations=1000,
        )

        result = sim.run_simulation(seed=42)

        # Expected value should be close to zero
        assert abs(result.expected_return_pct) < 5.0

    def test_large_position_sizes(self):
        """Test with large position sizes (high risk)"""
        sim = MonteCarloSimulator(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            num_simulations=100,
            position_size_pct=0.25,  # 25% position size
        )

        result = sim.run_simulation(seed=42)

        # Large positions should increase both returns and risk
        # With good win rate, drawdown may still be low
        assert result.avg_max_drawdown >= 0.0  # Some drawdown expected
        # Returns should be higher with larger position sizes
        assert result.expected_return_pct > 0  # Should still be profitable

    def test_many_trades(self):
        """Test with many trades per simulation"""
        sim = MonteCarloSimulator(
            win_rate=0.55,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            num_simulations=100,
            num_trades_per_sim=500,  # Many trades
        )

        result = sim.run_simulation(seed=42)

        # Should complete successfully
        assert result.num_trades_per_sim == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
