"""
Backtesting Framework for Trading Strategies

FEATURES:
- Basic backtesting engine with Vietnam market support
- Walk-forward validation for robustness testing
- Monte Carlo simulation for stress testing
- Regime-specific backtesting
"""

from backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult, Trade
from backtesting.strategy_runner import StrategyRunner, run_simple_backtest
from backtesting.visualizer import BacktestVisualizer

# Walk-forward validation & Monte Carlo
try:
    from backtesting.walk_forward import (
        WalkForwardWindow,
        WalkForwardResult,
        WalkForwardValidator,
        MonteCarloResult,
        MonteCarloSimulator,
        RegimeBacktestResult,
        RegimeBacktester,
        run_walk_forward_validation,
        run_monte_carlo,
    )

    ADVANCED_BACKTESTING_AVAILABLE = True
except ImportError:
    ADVANCED_BACKTESTING_AVAILABLE = False

__all__ = [
    # Core backtesting
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "StrategyRunner",
    "run_simple_backtest",
    "BacktestVisualizer",
    # Walk-forward validation
    "WalkForwardWindow",
    "WalkForwardResult",
    "WalkForwardValidator",
    "run_walk_forward_validation",
    # Monte Carlo
    "MonteCarloResult",
    "MonteCarloSimulator",
    "run_monte_carlo",
    # Regime backtesting
    "RegimeBacktestResult",
    "RegimeBacktester",
    # Feature flags
    "ADVANCED_BACKTESTING_AVAILABLE",
]
