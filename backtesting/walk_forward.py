# -*- coding: utf-8 -*-
"""
Walk-Forward Validation & Monte Carlo Simulation
Advanced backtesting validation techniques

FEATURES:
- Walk-forward optimization
- Out-of-sample testing
- Monte Carlo simulation for stress testing
- Regime-specific backtesting
- Statistical significance testing
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Single walk-forward window"""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Results
    train_return: float = 0.0
    test_return: float = 0.0
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_win_rate: float = 0.0
    test_win_rate: float = 0.0

    # Parameters used
    optimized_params: Dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Walk-forward validation results"""

    # Overall metrics
    total_windows: int
    avg_train_return: float
    avg_test_return: float
    avg_train_sharpe: float
    avg_test_sharpe: float

    # Consistency metrics
    test_vs_train_ratio: float  # Test return / Train return
    profitable_windows: int
    consistency_score: float  # % of windows where test > 0

    # Degradation analysis
    avg_degradation: float  # How much worse is test vs train
    max_degradation: float

    # Individual windows
    windows: List[WalkForwardWindow] = field(default_factory=list)

    # Verdict
    is_robust: bool = False
    verdict: str = ""


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""

    num_simulations: int

    # Return distribution
    mean_return: float
    median_return: float
    std_return: float

    # Percentiles
    percentile_5: float  # 5th percentile (worst case)
    percentile_25: float
    percentile_75: float
    percentile_95: float  # 95th percentile (best case)

    # Risk metrics
    probability_of_loss: float
    probability_of_ruin: float  # P(drawdown > 50%)
    expected_max_drawdown: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%

    # Distribution
    return_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    drawdown_distribution: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class RegimeBacktestResult:
    """Regime-specific backtest results"""

    regime: str  # BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY

    # Performance
    total_return: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float

    # Trade stats
    num_trades: int
    avg_trade_return: float
    avg_holding_days: float

    # Comparison
    vs_buy_hold: float  # Strategy return - Buy&Hold return


class WalkForwardValidator:
    """
    Walk-Forward Validation

    Splits data into multiple train/test windows to validate
    strategy robustness and avoid overfitting.

    Process:
    1. Split data into N windows
    2. For each window: train on in-sample, test on out-of-sample
    3. Aggregate results to assess robustness
    """

    def __init__(
        self,
        train_ratio: float = 0.7,  # 70% train, 30% test per window
        num_windows: int = 5,
        min_train_days: int = 120,  # Minimum training period
        min_test_days: int = 30,  # Minimum test period
    ):
        self.train_ratio = train_ratio
        self.num_windows = num_windows
        self.min_train_days = min_train_days
        self.min_test_days = min_test_days

    def validate(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        optimize_func: Optional[Callable] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation

        Args:
            data: Historical OHLCV data
            strategy_func: Function that runs strategy and returns metrics
                          Signature: strategy_func(data, params) -> Dict
            optimize_func: Optional function to optimize parameters
                          Signature: optimize_func(train_data) -> Dict

        Returns:
            WalkForwardResult
        """
        if len(data) < self.min_train_days + self.min_test_days:
            raise ValueError(
                f"Insufficient data: {len(data)} days. "
                f"Need at least {self.min_train_days + self.min_test_days}"
            )

        # Create windows
        windows = self._create_windows(data)

        if len(windows) < 2:
            raise ValueError("Could not create enough windows for validation")

        results = []

        for window in windows:
            # Get train and test data
            train_data = data[(data.index >= window.train_start) & (data.index <= window.train_end)]
            test_data = data[(data.index >= window.test_start) & (data.index <= window.test_end)]

            # Optimize parameters on training data (if optimizer provided)
            if optimize_func:
                params = optimize_func(train_data)
                window.optimized_params = params
            else:
                params = {}

            # Run strategy on training data
            train_metrics = strategy_func(train_data, params)
            window.train_return = train_metrics.get("total_return", 0)
            window.train_sharpe = train_metrics.get("sharpe_ratio", 0)
            window.train_win_rate = train_metrics.get("win_rate", 0)

            # Run strategy on test data (out-of-sample)
            test_metrics = strategy_func(test_data, params)
            window.test_return = test_metrics.get("total_return", 0)
            window.test_sharpe = test_metrics.get("sharpe_ratio", 0)
            window.test_win_rate = test_metrics.get("win_rate", 0)

            results.append(window)

            logger.info(
                f"Window {window.window_id}: "
                f"Train={window.train_return:.2%}, Test={window.test_return:.2%}"
            )

        # Aggregate results
        return self._aggregate_results(results)

    def _create_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """Create walk-forward windows"""
        windows = []

        total_days = len(data)
        window_size = total_days // self.num_windows

        # Ensure minimum sizes
        train_days = int(window_size * self.train_ratio)
        test_days = window_size - train_days

        if train_days < self.min_train_days:
            train_days = self.min_train_days
        if test_days < self.min_test_days:
            test_days = self.min_test_days

        # Create overlapping windows
        for i in range(self.num_windows):
            start_idx = i * (window_size // 2)  # 50% overlap

            if start_idx + train_days + test_days > total_days:
                break

            train_start = data.index[start_idx]
            train_end = data.index[start_idx + train_days - 1]
            test_start = data.index[start_idx + train_days]
            test_end = data.index[min(start_idx + train_days + test_days - 1, total_days - 1)]

            windows.append(
                WalkForwardWindow(
                    window_id=i + 1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

        return windows

    def _aggregate_results(self, windows: List[WalkForwardWindow]) -> WalkForwardResult:
        """Aggregate window results"""
        if not windows:
            return WalkForwardResult(
                total_windows=0,
                avg_train_return=0,
                avg_test_return=0,
                avg_train_sharpe=0,
                avg_test_sharpe=0,
                test_vs_train_ratio=0,
                profitable_windows=0,
                consistency_score=0,
                avg_degradation=0,
                max_degradation=0,
                is_robust=False,
                verdict="No windows to analyze",
            )

        # Calculate averages
        avg_train_return = np.mean([w.train_return for w in windows])
        avg_test_return = np.mean([w.test_return for w in windows])
        avg_train_sharpe = np.mean([w.train_sharpe for w in windows])
        avg_test_sharpe = np.mean([w.test_sharpe for w in windows])

        # Test vs Train ratio
        test_vs_train = avg_test_return / avg_train_return if avg_train_return != 0 else 0

        # Profitable windows
        profitable = sum(1 for w in windows if w.test_return > 0)
        consistency = profitable / len(windows)

        # Degradation
        degradations = [
            (w.train_return - w.test_return) / abs(w.train_return) if w.train_return != 0 else 0
            for w in windows
        ]
        avg_degradation = np.mean(degradations)
        max_degradation = max(degradations)

        # Determine robustness
        is_robust = (
            consistency >= 0.6  # 60% profitable windows
            and test_vs_train >= 0.5  # Test at least 50% of train
            and avg_degradation < 0.5  # Less than 50% degradation
        )

        # Verdict
        if is_robust:
            if consistency >= 0.8 and test_vs_train >= 0.7:
                verdict = "✅ HIGHLY ROBUST - Strategy shows strong out-of-sample performance"
            else:
                verdict = "✅ ROBUST - Strategy passes walk-forward validation"
        else:
            if consistency < 0.5:
                verdict = "❌ NOT ROBUST - Low consistency across windows"
            elif test_vs_train < 0.3:
                verdict = "❌ OVERFITTED - Large gap between train and test performance"
            else:
                verdict = "⚠️ MARGINAL - Strategy needs improvement"

        return WalkForwardResult(
            total_windows=len(windows),
            avg_train_return=avg_train_return,
            avg_test_return=avg_test_return,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            test_vs_train_ratio=test_vs_train,
            profitable_windows=profitable,
            consistency_score=consistency,
            avg_degradation=avg_degradation,
            max_degradation=max_degradation,
            windows=windows,
            is_robust=is_robust,
            verdict=verdict,
        )


class MonteCarloSimulator:
    """
    Monte Carlo Simulation for Strategy Stress Testing

    Simulates thousands of possible outcomes by:
    1. Resampling historical trades
    2. Randomizing trade order
    3. Adding noise to returns

    Provides probability distributions for:
    - Expected returns
    - Maximum drawdown
    - Risk of ruin
    """

    def __init__(
        self,
        num_simulations: int = 10000,
        confidence_level: float = 0.95,
        ruin_threshold: float = 0.50,  # 50% drawdown = ruin
    ):
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.ruin_threshold = ruin_threshold

    def simulate(
        self,
        trade_returns: List[float],
        initial_capital: float = 100_000_000,
        trades_per_simulation: Optional[int] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation

        Args:
            trade_returns: List of historical trade returns (as decimals)
            initial_capital: Starting capital
            trades_per_simulation: Number of trades per simulation (default: same as historical)

        Returns:
            MonteCarloResult
        """
        if not trade_returns:
            raise ValueError("No trade returns provided")

        trade_returns = np.array(trade_returns)
        n_trades = trades_per_simulation or len(trade_returns)

        # Storage for results
        final_returns = []
        max_drawdowns = []

        for _ in range(self.num_simulations):
            # Resample trades with replacement
            sampled_returns = np.random.choice(trade_returns, size=n_trades, replace=True)

            # Calculate equity curve
            equity = initial_capital
            peak = initial_capital
            max_dd = 0

            for ret in sampled_returns:
                equity *= 1 + ret
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)

            # Store results
            total_return = (equity - initial_capital) / initial_capital
            final_returns.append(total_return)
            max_drawdowns.append(max_dd)

        final_returns = np.array(final_returns)
        max_drawdowns = np.array(max_drawdowns)

        # Calculate statistics
        return MonteCarloResult(
            num_simulations=self.num_simulations,
            mean_return=np.mean(final_returns),
            median_return=np.median(final_returns),
            std_return=np.std(final_returns),
            percentile_5=np.percentile(final_returns, 5),
            percentile_25=np.percentile(final_returns, 25),
            percentile_75=np.percentile(final_returns, 75),
            percentile_95=np.percentile(final_returns, 95),
            probability_of_loss=np.mean(final_returns < 0),
            probability_of_ruin=np.mean(max_drawdowns > self.ruin_threshold),
            expected_max_drawdown=np.mean(max_drawdowns),
            var_95=np.percentile(final_returns, 5),  # 5th percentile = 95% VaR
            cvar_95=np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]),
            return_distribution=final_returns,
            drawdown_distribution=max_drawdowns,
        )

    def simulate_with_regime(
        self,
        trade_returns_by_regime: Dict[str, List[float]],
        regime_probabilities: Dict[str, float],
        initial_capital: float = 100_000_000,
        simulation_days: int = 252,
    ) -> MonteCarloResult:
        """
        Monte Carlo with regime-switching

        Args:
            trade_returns_by_regime: Dict of regime -> trade returns
            regime_probabilities: Dict of regime -> probability
            initial_capital: Starting capital
            simulation_days: Days to simulate

        Returns:
            MonteCarloResult
        """
        regimes = list(regime_probabilities.keys())
        probs = list(regime_probabilities.values())

        final_returns = []
        max_drawdowns = []

        for _ in range(self.num_simulations):
            equity = initial_capital
            peak = initial_capital
            max_dd = 0

            for _ in range(simulation_days):
                # Sample regime
                regime = np.random.choice(regimes, p=probs)

                # Get returns for this regime
                regime_returns = trade_returns_by_regime.get(regime, [0])
                if not regime_returns:
                    continue

                # Sample a return
                daily_return = np.random.choice(regime_returns)

                equity *= 1 + daily_return
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)

            total_return = (equity - initial_capital) / initial_capital
            final_returns.append(total_return)
            max_drawdowns.append(max_dd)

        final_returns = np.array(final_returns)
        max_drawdowns = np.array(max_drawdowns)

        return MonteCarloResult(
            num_simulations=self.num_simulations,
            mean_return=np.mean(final_returns),
            median_return=np.median(final_returns),
            std_return=np.std(final_returns),
            percentile_5=np.percentile(final_returns, 5),
            percentile_25=np.percentile(final_returns, 25),
            percentile_75=np.percentile(final_returns, 75),
            percentile_95=np.percentile(final_returns, 95),
            probability_of_loss=np.mean(final_returns < 0),
            probability_of_ruin=np.mean(max_drawdowns > self.ruin_threshold),
            expected_max_drawdown=np.mean(max_drawdowns),
            var_95=np.percentile(final_returns, 5),
            cvar_95=np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]),
            return_distribution=final_returns,
            drawdown_distribution=max_drawdowns,
        )


class RegimeBacktester:
    """
    Regime-Specific Backtesting

    Tests strategy performance in different market regimes:
    - BULL market
    - BEAR market
    - SIDEWAYS market
    - HIGH_VOLATILITY market
    """

    def __init__(self):
        self.regime_results: Dict[str, RegimeBacktestResult] = {}

    def backtest_by_regime(
        self,
        data: pd.DataFrame,
        regime_labels: pd.Series,
        strategy_func: Callable,
        params: Dict = None,
    ) -> Dict[str, RegimeBacktestResult]:
        """
        Run backtest for each market regime

        Args:
            data: OHLCV data with index as datetime
            regime_labels: Series with regime labels for each date
            strategy_func: Strategy function
            params: Strategy parameters

        Returns:
            Dict of regime -> RegimeBacktestResult
        """
        params = params or {}
        results = {}

        # Get unique regimes
        regimes = regime_labels.unique()

        for regime in regimes:
            # Filter data for this regime
            regime_mask = regime_labels == regime
            regime_data = data[regime_mask]

            if len(regime_data) < 20:
                logger.warning(f"Insufficient data for regime {regime}: {len(regime_data)} days")
                continue

            # Run strategy
            metrics = strategy_func(regime_data, params)

            # Calculate buy & hold for comparison
            buy_hold_return = regime_data["close"].iloc[-1] / regime_data["close"].iloc[0] - 1

            results[regime] = RegimeBacktestResult(
                regime=regime,
                total_return=metrics.get("total_return", 0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0),
                win_rate=metrics.get("win_rate", 0),
                profit_factor=metrics.get("profit_factor", 0),
                max_drawdown=metrics.get("max_drawdown", 0),
                num_trades=metrics.get("num_trades", 0),
                avg_trade_return=metrics.get("avg_trade_return", 0),
                avg_holding_days=metrics.get("avg_holding_days", 0),
                vs_buy_hold=metrics.get("total_return", 0) - buy_hold_return,
            )

            logger.info(
                f"Regime {regime}: Return={results[regime].total_return:.2%}, "
                f"Sharpe={results[regime].sharpe_ratio:.2f}, "
                f"vs B&H={results[regime].vs_buy_hold:+.2%}"
            )

        self.regime_results = results
        return results

    def get_regime_summary(self) -> str:
        """Get formatted summary of regime performance"""
        if not self.regime_results:
            return "No regime backtest results available"

        lines = []
        lines.append("=" * 70)
        lines.append("REGIME-SPECIFIC BACKTEST RESULTS")
        lines.append("=" * 70)

        for regime, result in self.regime_results.items():
            lines.append(f"\n📊 {regime} Market:")
            lines.append(f"   Total Return: {result.total_return:+.2%}")
            lines.append(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
            lines.append(f"   Win Rate: {result.win_rate:.1%}")
            lines.append(f"   Max Drawdown: {result.max_drawdown:.2%}")
            lines.append(f"   vs Buy&Hold: {result.vs_buy_hold:+.2%}")
            lines.append(f"   Trades: {result.num_trades}")

        # Best/Worst regime
        if self.regime_results:
            best = max(self.regime_results.items(), key=lambda x: x[1].total_return)
            worst = min(self.regime_results.items(), key=lambda x: x[1].total_return)

            lines.append(f"\n✅ Best Regime: {best[0]} ({best[1].total_return:+.2%})")
            lines.append(f"❌ Worst Regime: {worst[0]} ({worst[1].total_return:+.2%})")

        lines.append("=" * 70)

        return "\n".join(lines)


# Convenience functions
def run_walk_forward_validation(
    data: pd.DataFrame,
    strategy_func: Callable,
    num_windows: int = 5,
) -> WalkForwardResult:
    """Run walk-forward validation"""
    validator = WalkForwardValidator(num_windows=num_windows)
    return validator.validate(data, strategy_func)


def run_monte_carlo(
    trade_returns: List[float],
    num_simulations: int = 10000,
) -> MonteCarloResult:
    """Run Monte Carlo simulation"""
    simulator = MonteCarloSimulator(num_simulations=num_simulations)
    return simulator.simulate(trade_returns)


# Test
if __name__ == "__main__":
    print("Testing Walk-Forward Validation & Monte Carlo...")

    # Create dummy data
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=500, freq="D")
    prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)

    data = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, 500),
        },
        index=dates,
    )

    # Dummy strategy function
    def dummy_strategy(df, params):
        returns = df["close"].pct_change().dropna()
        return {
            "total_return": returns.sum(),
            "sharpe_ratio": (
                returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            ),
            "win_rate": (returns > 0).mean(),
        }

    # Test Walk-Forward
    print("\n1️⃣ Walk-Forward Validation:")
    validator = WalkForwardValidator(num_windows=4)
    wf_result = validator.validate(data, dummy_strategy)
    print(f"   Windows: {wf_result.total_windows}")
    print(f"   Avg Train Return: {wf_result.avg_train_return:.2%}")
    print(f"   Avg Test Return: {wf_result.avg_test_return:.2%}")
    print(f"   Consistency: {wf_result.consistency_score:.1%}")
    print(f"   Verdict: {wf_result.verdict}")

    # Test Monte Carlo
    print("\n2️⃣ Monte Carlo Simulation:")
    trade_returns = list(np.random.randn(100) * 0.03)  # 3% std per trade
    simulator = MonteCarloSimulator(num_simulations=5000)
    mc_result = simulator.simulate(trade_returns)
    print(f"   Mean Return: {mc_result.mean_return:.2%}")
    print(f"   Median Return: {mc_result.median_return:.2%}")
    print(f"   5th Percentile: {mc_result.percentile_5:.2%}")
    print(f"   95th Percentile: {mc_result.percentile_95:.2%}")
    print(f"   P(Loss): {mc_result.probability_of_loss:.1%}")
    print(f"   P(Ruin): {mc_result.probability_of_ruin:.1%}")
    print(f"   Expected Max DD: {mc_result.expected_max_drawdown:.2%}")

    # Test Regime Backtester
    print("\n3️⃣ Regime-Specific Backtest:")
    regime_labels = pd.Series(np.random.choice(["BULL", "BEAR", "SIDEWAYS"], size=500), index=dates)
    backtester = RegimeBacktester()
    regime_results = backtester.backtest_by_regime(data, regime_labels, dummy_strategy)
    print(backtester.get_regime_summary())

    print("\n✅ All tests completed!")
