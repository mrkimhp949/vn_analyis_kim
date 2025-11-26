# -*- coding: utf-8 -*-
"""
Tests for Backtesting Engine Improvements V2

Tests:
1. Lot size rounding (Vietnam: 100 shares)
2. Tick size rounding (Vietnam: 10 VND)
3. Transaction costs from constants.py
4. Partial exit support
5. Circuit breaker functionality
6. Position size multiplier
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    Trade,
    round_to_lot_size,
    round_to_tick_size,
)
from src.config.constants import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_SLIPPAGE,
    VIETNAM_LOT_SIZE,
    VIETNAM_TICK_SIZE,
)


class TestLotSizeRounding:
    """Test Vietnam lot size rounding (100 shares)"""

    def test_round_to_lot_size_exact(self):
        """Test exact lot size"""
        assert round_to_lot_size(100) == 100
        assert round_to_lot_size(200) == 200
        assert round_to_lot_size(1000) == 1000

    def test_round_to_lot_size_down(self):
        """Test rounding down to nearest lot"""
        assert round_to_lot_size(150) == 100
        assert round_to_lot_size(199) == 100
        assert round_to_lot_size(250) == 200
        assert round_to_lot_size(999) == 900

    def test_round_to_lot_size_small(self):
        """Test small amounts round to 0"""
        assert round_to_lot_size(50) == 0
        assert round_to_lot_size(99) == 0

    def test_round_to_lot_size_custom(self):
        """Test custom lot size"""
        assert round_to_lot_size(150, lot_size=50) == 150
        assert round_to_lot_size(175, lot_size=50) == 150


class TestTickSizeRounding:
    """Test Vietnam tick size rounding (10 VND)"""

    def test_round_to_tick_size_exact(self):
        """Test exact tick size"""
        assert round_to_tick_size(25000) == 25000
        assert round_to_tick_size(25010) == 25010

    def test_round_to_tick_size_round(self):
        """Test rounding to nearest tick (Python uses banker's rounding)"""
        # Python's round() uses banker's rounding (round half to even)
        # 25005 / 10 = 2500.5 -> rounds to 2500 (even) -> 25000
        assert round_to_tick_size(25006) == 25010  # Round up
        assert round_to_tick_size(25004) == 25000  # Round down
        assert round_to_tick_size(25016) == 25020

    def test_round_to_tick_size_custom(self):
        """Test custom tick size"""
        # 25050 / 100 = 250.5 -> rounds to 250 (even) -> 25000
        assert round_to_tick_size(25051, tick_size=100) == 25100
        assert round_to_tick_size(25049, tick_size=100) == 25000


class TestBacktestConfig:
    """Test BacktestConfig uses constants.py values"""

    def test_default_commission_rate(self):
        """Test default commission rate from constants"""
        config = BacktestConfig()
        assert config.commission_rate == DEFAULT_COMMISSION_RATE

    def test_default_slippage(self):
        """Test default slippage from constants"""
        config = BacktestConfig()
        assert config.slippage == DEFAULT_SLIPPAGE

    def test_lot_size(self):
        """Test lot size from constants"""
        config = BacktestConfig()
        assert config.lot_size == VIETNAM_LOT_SIZE

    def test_tick_size(self):
        """Test tick size from constants"""
        config = BacktestConfig()
        assert config.tick_size == VIETNAM_TICK_SIZE


class TestBacktestEngineOpenPosition:
    """Test open_position with lot size and tick size"""

    def setup_method(self):
        """Setup test engine"""
        self.config = BacktestConfig(
            initial_capital=100_000_000,
            position_size_pct=0.20,
            max_positions=5,
        )
        self.engine = BacktestEngine(self.config)

    def test_open_position_lot_size_rounding(self):
        """Test shares are rounded to lot size"""
        # 20M / 25000 = 800 shares (exact lot)
        trade = self.engine.open_position(
            symbol="TEST",
            date=datetime.now(),
            entry_price=25000,
            stop_loss=23000,
            take_profit=28000,
        )
        assert trade is not None
        assert trade.shares % VIETNAM_LOT_SIZE == 0

    def test_open_position_tick_size_rounding(self):
        """Test prices are rounded to tick size"""
        trade = self.engine.open_position(
            symbol="TEST",
            date=datetime.now(),
            entry_price=25005,  # Should round to 25010
            stop_loss=23005,  # Should round to 23010
            take_profit=28005,  # Should round to 28010
        )
        assert trade is not None
        assert trade.entry_price % VIETNAM_TICK_SIZE == 0
        assert trade.stop_loss % VIETNAM_TICK_SIZE == 0
        assert trade.take_profit % VIETNAM_TICK_SIZE == 0

    def test_open_position_with_multiplier(self):
        """Test position size multiplier"""
        # Default position (multiplier = 1.0)
        trade1 = self.engine.open_position(
            symbol="TEST1",
            date=datetime.now(),
            entry_price=25000,
            stop_loss=23000,
            take_profit=28000,
            position_size_multiplier=1.0,
        )

        # Reset engine
        self.engine = BacktestEngine(self.config)

        # Smaller position (multiplier = 0.5)
        trade2 = self.engine.open_position(
            symbol="TEST2",
            date=datetime.now(),
            entry_price=25000,
            stop_loss=23000,
            take_profit=28000,
            position_size_multiplier=0.5,
        )

        assert trade1 is not None
        assert trade2 is not None
        # trade2 should have roughly half the shares (after lot rounding)
        assert trade2.shares < trade1.shares


class TestPartialExit:
    """Test partial exit functionality"""

    def setup_method(self):
        """Setup test engine with position"""
        self.config = BacktestConfig(initial_capital=100_000_000)
        self.engine = BacktestEngine(self.config)

        # Open a position
        self.trade = self.engine.open_position(
            symbol="TEST",
            date=datetime.now(),
            entry_price=25000,
            stop_loss=23000,
            take_profit=30000,
        )

    def test_partial_close_50_percent(self):
        """Test 50% partial close"""
        initial_shares = self.trade.shares
        initial_capital = self.engine.capital

        pnl = self.engine.partial_close_position(
            symbol="TEST",
            date=datetime.now(),
            exit_price=27000,  # 8% profit
            exit_percent=0.5,
            reason="TP1",
        )

        assert pnl is not None
        assert pnl > 0  # Should be profitable
        assert self.trade.shares < initial_shares
        assert self.engine.capital > initial_capital
        assert len(self.trade.partial_exits) == 1

    def test_partial_close_converts_to_full_when_all_shares(self):
        """Test partial close converts to full close when exiting all shares"""
        # Try to exit 100%
        result = self.engine.partial_close_position(
            symbol="TEST",
            date=datetime.now(),
            exit_price=27000,
            exit_percent=1.0,
            reason="Full exit",
        )

        # Position should be fully closed
        assert "TEST" not in self.engine.positions
        assert len(self.engine.completed_trades) == 1


class TestCircuitBreaker:
    """Test circuit breaker functionality"""

    def setup_method(self):
        """Setup test engine"""
        self.config = BacktestConfig(
            initial_capital=100_000_000,
            max_daily_loss=0.05,  # 5% daily loss limit
            daily_loss_circuit_breaker=True,
            max_consecutive_losses=3,
        )
        self.engine = BacktestEngine(self.config)

    def test_circuit_breaker_blocks_new_positions(self):
        """Test circuit breaker blocks new positions after trigger"""
        # Manually trigger circuit breaker
        self.engine.circuit_breaker_triggered = True

        # Try to open position
        can_open = self.engine.can_open_position("TEST", 10_000_000)
        assert can_open is False

    def test_consecutive_losses_blocks_symbol(self):
        """Test consecutive losses blocks specific symbol"""
        # Simulate 3 consecutive losses for TEST
        self.engine.consecutive_losses["TEST"] = 3

        # Should not be able to open TEST
        can_open = self.engine.can_open_position("TEST", 10_000_000)
        assert can_open is False

        # But can open other symbols
        can_open_other = self.engine.can_open_position("OTHER", 10_000_000)
        assert can_open_other is True

    def test_win_resets_consecutive_losses(self):
        """Test winning trade resets consecutive losses"""
        # Setup: 2 consecutive losses
        self.engine.consecutive_losses["TEST"] = 2

        # Open and close with profit
        trade = self.engine.open_position(
            symbol="TEST",
            date=datetime.now(),
            entry_price=25000,
            stop_loss=23000,
            take_profit=30000,
        )

        self.engine.close_position(
            symbol="TEST",
            date=datetime.now(),
            exit_price=27000,  # Profit
            reason="TP",
        )

        # Consecutive losses should be reset
        assert self.engine.consecutive_losses["TEST"] == 0


class TestTradePartialExitTracking:
    """Test Trade dataclass partial exit tracking"""

    def test_trade_partial_close(self):
        """Test Trade.partial_close method"""
        trade = Trade(
            symbol="TEST",
            entry_date=datetime.now(),
            entry_price=25000,
            shares=1000,
            initial_shares=1000,
            stop_loss=23000,
            take_profit=30000,
        )

        # Partial close 500 shares at profit
        pnl = trade.partial_close(
            exit_date=datetime.now(),
            exit_price=27000,
            exit_shares=500,
            exit_reason="TP1",
            commission_rate=0.0035,
            slippage=0.001,
        )

        assert pnl > 0
        assert trade.shares == 500
        assert trade.realized_pnl > 0
        assert len(trade.partial_exits) == 1

    def test_trade_full_close_includes_partial_pnl(self):
        """Test full close includes realized PnL from partial exits"""
        trade = Trade(
            symbol="TEST",
            entry_date=datetime.now(),
            entry_price=25000,
            shares=1000,
            initial_shares=1000,
            stop_loss=23000,
            take_profit=30000,
        )

        # Partial close first
        trade.partial_close(
            exit_date=datetime.now(),
            exit_price=27000,
            exit_shares=500,
            exit_reason="TP1",
            commission_rate=0.0035,
            slippage=0.001,
        )

        partial_pnl = trade.realized_pnl

        # Full close remaining
        trade.close_trade(
            exit_date=datetime.now(),
            exit_price=28000,
            exit_reason="TP2",
            commission_rate=0.0035,
            slippage=0.001,
        )

        # Total PnL should include partial exit PnL
        assert trade.pnl > partial_pnl


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
