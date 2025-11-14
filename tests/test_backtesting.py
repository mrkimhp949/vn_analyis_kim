"""
Unit tests for Backtesting Engine
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.engine import BacktestEngine, BacktestConfig, Trade
from backtesting.strategy_runner import StrategyRunner


class TestBacktestEngine:
    """Test BacktestEngine class"""

    def setup_method(self):
        """Setup for each test"""
        self.config = BacktestConfig(
            initial_capital=100_000_000,
            commission_rate=0.0015,
            max_positions=3,
            position_size_pct=0.30
        )
        self.engine = BacktestEngine(self.config)

    def test_engine_initialization(self):
        """Test engine initializes with correct values"""
        assert self.engine.capital == 100_000_000
        assert self.engine.initial_capital == 100_000_000
        assert len(self.engine.positions) == 0
        assert len(self.engine.completed_trades) == 0

    def test_open_position_valid(self):
        """Test opening a valid position"""
        trade = self.engine.open_position(
            symbol='VCB',
            date=datetime(2024, 1, 1),
            entry_price=60000,
            stop_loss=57000,
            take_profit=66000,
            reason='Test entry'
        )

        assert trade is not None
        assert trade.symbol == 'VCB'
        assert trade.entry_price == 60000
        assert trade.shares > 0
        assert 'VCB' in self.engine.positions
        assert self.engine.capital < 100_000_000  # Capital reduced

    def test_open_position_max_positions(self):
        """Test cannot exceed max positions"""
        # Open 3 positions (max)
        for i in range(3):
            self.engine.open_position(
                symbol=f'SYM{i}',
                date=datetime(2024, 1, 1),
                entry_price=50000,
                stop_loss=47000,
                take_profit=55000
            )

        # Try to open 4th position
        trade = self.engine.open_position(
            symbol='SYM4',
            date=datetime(2024, 1, 1),
            entry_price=50000,
            stop_loss=47000,
            take_profit=55000
        )

        assert trade is None
        assert len(self.engine.positions) == 3

    def test_open_position_insufficient_capital(self):
        """Test cannot open position with insufficient capital"""
        # Try to open position that requires more capital than available
        trade = self.engine.open_position(
            symbol='VCB',
            date=datetime(2024, 1, 1),
            entry_price=200_000_000,  # Too expensive
            stop_loss=190_000_000,
            take_profit=220_000_000
        )

        assert trade is None

    def test_close_position_profitable(self):
        """Test closing a profitable position"""
        # Open position
        self.engine.open_position(
            symbol='VCB',
            date=datetime(2024, 1, 1),
            entry_price=60000,
            stop_loss=57000,
            take_profit=66000
        )

        initial_capital = self.engine.capital

        # Close with profit
        trade = self.engine.close_position(
            symbol='VCB',
            date=datetime(2024, 1, 15),
            exit_price=65000,
            reason='Take profit'
        )

        assert trade is not None
        assert trade.pnl > 0
        assert trade.exit_price == 65000
        assert 'VCB' not in self.engine.positions
        assert len(self.engine.completed_trades) == 1
        assert self.engine.capital > initial_capital

    def test_close_position_loss(self):
        """Test closing a losing position"""
        # Open position
        self.engine.open_position(
            symbol='VCB',
            date=datetime(2024, 1, 1),
            entry_price=60000,
            stop_loss=57000,
            take_profit=66000
        )

        initial_capital = self.engine.capital

        # Close with loss
        trade = self.engine.close_position(
            symbol='VCB',
            date=datetime(2024, 1, 15),
            exit_price=56000,
            reason='Stop loss'
        )

        assert trade is not None
        assert trade.pnl < 0
        assert self.engine.capital < initial_capital

    def test_trade_pnl_calculation(self):
        """Test PnL calculation includes commission and slippage"""
        trade = Trade(
            symbol='VCB',
            entry_date=datetime(2024, 1, 1),
            entry_price=60000,
            shares=100,
            stop_loss=57000,
            take_profit=66000
        )

        trade.close_trade(
            exit_date=datetime(2024, 1, 15),
            exit_price=65000,
            exit_reason='Take profit',
            commission_rate=0.0015,
            slippage=0.001
        )

        # Verify commission was charged
        assert trade.commission > 0

        # Verify slippage cost was calculated
        assert trade.slippage_cost > 0

        # Verify PnL is net of costs
        gross_pnl = 100 * (65000 - 60000)
        assert trade.pnl == gross_pnl - trade.commission - trade.slippage_cost

    def test_update_equity(self):
        """Test equity curve updates correctly"""
        # Open position
        self.engine.open_position(
            symbol='VCB',
            date=datetime(2024, 1, 1),
            entry_price=60000,
            stop_loss=57000,
            take_profit=66000
        )

        # Update equity with current prices
        self.engine.update_equity(
            date=datetime(2024, 1, 5),
            current_prices={'VCB': 62000}
        )

        assert len(self.engine.equity_curve) == 1
        assert self.engine.equity_curve[0]['equity'] > self.config.initial_capital

    def test_calculate_results(self):
        """Test results calculation"""
        # Simulate some trades
        for i in range(3):
            self.engine.open_position(
                symbol=f'SYM{i}',
                date=datetime(2024, 1, 1),
                entry_price=50000,
                stop_loss=47000,
                take_profit=55000
            )

            self.engine.update_equity(
                date=datetime(2024, 1, 5),
                current_prices={f'SYM{i}': 52000}
            )

            self.engine.close_position(
                symbol=f'SYM{i}',
                date=datetime(2024, 1, 10),
                exit_price=52000,
                reason='Test close'
            )

        results = self.engine.calculate_results()

        assert results.total_trades == 3
        assert results.winning_trades == 3
        assert results.final_capital > results.initial_capital
        assert results.win_rate > 0


class TestStrategyRunner:
    """Test StrategyRunner class"""

    def test_strategy_runner_initialization(self):
        """Test strategy runner initializes correctly"""
        runner = StrategyRunner()

        assert runner.engine is not None
        assert runner.entry_logic is not None
        assert runner.exit_logic is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
