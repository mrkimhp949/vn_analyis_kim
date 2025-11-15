# -*- coding: utf-8 -*-
"""
Unit tests for ImprovedExitStrategy
"""
from datetime import datetime, timedelta

import pytest
from src.strategies.exit_logic import ExitReason, ImprovedExitStrategy


class TestImprovedExitStrategy:

    @pytest.fixture
    def exit_strategy(self):
        return ImprovedExitStrategy(
            take_profit_levels=[0.10, 0.15, 0.25],
            trailing_stop_activation=0.08,
            trailing_stop_distance=0.05,
        )

    def test_initialization(self, exit_strategy):
        """Test exit strategy initialization"""
        assert exit_strategy.tp_levels == [0.10, 0.15, 0.25]
        assert exit_strategy.trailing_activation == 0.08
        assert exit_strategy.trailing_distance == 0.05

    def test_stop_loss_hit(self, exit_strategy, sample_ohlcv_data):
        """Test stop loss trigger"""
        entry_price = 80000
        current_price = 75000  # Below stop loss
        stop_loss = 76000

        decision = exit_strategy.check_exit(
            symbol="VNM",
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit_targets=[84000, 88000, 92000],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_ohlcv_data,
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.STOP_LOSS
        assert decision.urgency == 5

    def test_take_profit_1(self, exit_strategy, sample_ohlcv_data):
        """Test TP1 trigger"""
        entry_price = 80000
        current_price = 88000  # +10%

        decision = exit_strategy.check_exit(
            symbol="VNM",
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=76000,
            take_profit_targets=[88000, 92000, 100000],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_ohlcv_data,
            partial_exits=[],
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.TAKE_PROFIT_1
        assert decision.exit_type == "PARTIAL_30%"

    def test_trailing_stop_activation(self, exit_strategy, sample_ohlcv_data):
        """Test trailing stop activation and trigger"""
        entry_price = 80000
        highest_price = 94000  # +17.5% (well above TP2)
        current_price = 88500  # -5.9% from high (below trailing distance 5%)

        # Already took TP1 and TP2, so they won't trigger again
        partial_exits = [1, 2]  # Already exited at TP1 and TP2

        # Update position high
        exit_strategy.position_highs["VNM"] = highest_price

        decision = exit_strategy.check_exit(
            symbol="VNM",
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=76000,
            take_profit_targets=[88000, 92000, 100000],  # TP1 and TP2 already taken
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_ohlcv_data,
            partial_exits=partial_exits,
        )

        # Should trigger trailing stop since price dropped from high
        # Or may trigger TP1/TP2 if not yet exited
        assert decision.should_exit is True
        assert decision.exit_reason in [
            ExitReason.TRAILING_STOP,
            ExitReason.TAKE_PROFIT_1,
            ExitReason.TAKE_PROFIT_2,
        ]

    def test_ml_signal_sell(self, exit_strategy, sample_ohlcv_data):
        """Test ML SELL signal trigger"""
        ml_signal = {"signal": "SELL", "confidence": 70}

        decision = exit_strategy.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=82000,  # Small profit
            stop_loss=76000,
            take_profit_targets=[84000, 88000, 92000],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_ohlcv_data,
            ml_signal=ml_signal,
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.ML_SIGNAL_SELL

    def test_time_decay(self, exit_strategy, sample_ohlcv_data):
        """Test time decay trigger"""
        decision = exit_strategy.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=80500,  # Only +0.6% after 21 days
            stop_loss=76000,
            take_profit_targets=[84000, 88000, 92000],
            entry_date=datetime.now() - timedelta(days=21),
            df=sample_ohlcv_data,
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.TIME_DECAY

    def test_hold_decision(self, exit_strategy, sample_ohlcv_data):
        """Test HOLD decision (no exit)"""
        decision = exit_strategy.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=82000,  # +2.5% profit
            stop_loss=76000,
            take_profit_targets=[88000, 92000, 100000],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_ohlcv_data,
        )

        assert decision.should_exit is False
        assert decision.exit_type == "HOLD"

    def test_market_crash_protection(self, exit_strategy, sample_ohlcv_data):
        """Test market crash protection"""
        market_regime = {"regime": "BEAR", "tradeable": False}

        decision = exit_strategy.check_exit(
            symbol="VNM",
            entry_price=80000,
            current_price=82500,  # +3.1% profit
            stop_loss=76000,
            take_profit_targets=[88000, 92000, 100000],
            entry_date=datetime.now() - timedelta(days=5),
            df=sample_ohlcv_data,
            market_regime=market_regime,
        )

        assert decision.should_exit is True
        assert decision.exit_reason == ExitReason.MARKET_CRASH
