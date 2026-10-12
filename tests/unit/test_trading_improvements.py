# -*- coding: utf-8 -*-
"""
Unit Tests for Trading Improvements

Tests for:
1. ExecutionCostTracker
2. MLPerformanceValidator
3. OddLotHandler
4. LiveConsistencyChecker

Author: Trading Bot Team
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class TestExecutionCostTracker:
    """Tests for ExecutionCostTracker."""

    @pytest.fixture
    def tracker(self):
        """Create fresh tracker instance."""
        from src.monitoring.execution_tracker import ExecutionCostTracker

        return ExecutionCostTracker(stats_file="test_execution_stats.json")

    def test_record_execution_buy(self, tracker):
        """Test recording a buy execution."""
        record = tracker.record_execution(
            symbol="VNM",
            order_type="LIMIT",
            side="BUY",
            expected_price=80000,
            executed_price=80100,
            shares=500,
            expected_slippage_pct=0.002,
            expected_commission=100000,
            actual_commission=100500,
            execution_time_ms=1500,
            session="CONTINUOUS",
            avg_daily_volume=1000000,
        )

        assert record.symbol == "VNM"
        assert record.side == "BUY"
        assert record.actual_slippage_pct == pytest.approx(0.00125, rel=0.01)
        assert record.slippage_deviation == pytest.approx(-0.00075, rel=0.01)

    def test_record_execution_sell(self, tracker):
        """Test recording a sell execution."""
        record = tracker.record_execution(
            symbol="HPG",
            order_type="MARKET",
            side="SELL",
            expected_price=25000,
            executed_price=24900,
            shares=1000,
            expected_slippage_pct=0.004,
            expected_commission=62500,
            actual_commission=62250,
            execution_time_ms=500,
            session="CONTINUOUS",
            avg_daily_volume=5000000,
        )

        assert record.symbol == "HPG"
        assert record.side == "SELL"
        # Sell slippage = (expected - executed) / expected
        assert record.actual_slippage_pct == pytest.approx(0.004, rel=0.01)

    def test_calibrated_slippage_insufficient_data(self, tracker):
        """Test calibrated slippage with insufficient data."""
        slippage = tracker.get_calibrated_slippage("UNKNOWN", fallback=0.005)
        assert slippage == 0.005

    def test_calibrated_slippage_with_data(self, tracker):
        """Test calibrated slippage with sufficient data."""
        # Record multiple executions
        for i in range(10):
            tracker.record_execution(
                symbol="VNM",
                order_type="LIMIT",
                side="BUY",
                expected_price=80000,
                executed_price=80000 + i * 10,
                shares=500,
                expected_slippage_pct=0.002,
                expected_commission=100000,
                actual_commission=100000,
                execution_time_ms=1000,
                session="CONTINUOUS",
                avg_daily_volume=1000000,
            )

        slippage = tracker.get_calibrated_slippage("VNM")
        assert slippage > 0
        assert slippage < 0.03  # Should be capped

    def test_execution_quality_score(self, tracker):
        """Test execution quality score calculation."""
        # Record some executions
        for i in range(5):
            tracker.record_execution(
                symbol="FPT",
                order_type="LIMIT",
                side="BUY",
                expected_price=120000,
                executed_price=120100,
                shares=200,
                expected_slippage_pct=0.001,
                expected_commission=60000,
                actual_commission=60000,
                execution_time_ms=800,
                session="CONTINUOUS",
                avg_daily_volume=2000000,
            )

        score, desc = tracker.get_execution_quality_score("FPT")
        assert 0 <= score <= 100
        assert isinstance(desc, str)

    def test_daily_report(self, tracker):
        """Test daily report generation."""
        tracker.record_execution(
            symbol="VNM",
            order_type="LIMIT",
            side="BUY",
            expected_price=80000,
            executed_price=80100,
            shares=500,
            expected_slippage_pct=0.002,
            expected_commission=100000,
            actual_commission=100500,
            execution_time_ms=1500,
            session="CONTINUOUS",
            avg_daily_volume=1000000,
        )

        report = tracker.get_daily_report()
        assert report["total_executions"] >= 1
        assert "total_value" in report


class TestMLPerformanceValidator:
    """Tests for MLPerformanceValidator."""

    @pytest.fixture
    def validator(self):
        """Create fresh validator instance."""
        from src.monitoring.ml_performance_validator import MLPerformanceValidator

        return MLPerformanceValidator(
            stats_file="test_ml_stats.json",
            model_version="test_v1",
        )

    def test_record_prediction(self, validator):
        """Test recording a prediction."""
        pred_id = validator.record_prediction(
            symbol="VNM",
            signal="BUY",
            confidence=75,
            entry_price=80000,
        )

        assert pred_id is not None
        assert "VNM" in pred_id

    def test_record_outcome_win(self, validator):
        """Test recording a winning outcome."""
        pred_id = validator.record_prediction(
            symbol="VNM",
            signal="BUY",
            confidence=75,
            entry_price=80000,
        )

        result = validator.record_outcome(pred_id, 84000, "TP1")

        assert result is not None
        assert result.outcome == "WIN"
        assert result.actual_return_pct == pytest.approx(0.05, rel=0.01)

    def test_record_outcome_loss(self, validator):
        """Test recording a losing outcome."""
        pred_id = validator.record_prediction(
            symbol="HPG",
            signal="BUY",
            confidence=60,
            entry_price=25000,
        )

        result = validator.record_outcome(pred_id, 23500, "STOP_LOSS")

        assert result is not None
        assert result.outcome == "LOSS"
        assert result.actual_return_pct == pytest.approx(-0.06, rel=0.01)

    def test_model_health_insufficient_data(self, validator):
        """Test model health with insufficient data."""
        health = validator.get_model_health()

        assert health.accuracy_status == "INSUFFICIENT_DATA"
        assert health.is_healthy == True  # Assume healthy until proven otherwise

    def test_model_health_with_data(self, validator):
        """Test model health with sufficient data."""
        # Record 35 predictions with outcomes (60% win rate)
        for i in range(35):
            pred_id = validator.record_prediction(
                symbol=f"SYM{i}",
                signal="BUY",
                confidence=70,
                entry_price=100000,
            )

            # 60% wins
            if i % 5 < 3:  # 3 out of 5 = 60%
                validator.record_outcome(pred_id, 105000, "TP1")  # +5%
            else:
                validator.record_outcome(pred_id, 96000, "STOP_LOSS")  # -4%

        health = validator.get_model_health()

        assert health.accuracy_status in ["GOOD", "WARNING"]
        assert health.is_healthy == True

    def test_should_retrain(self, validator):
        """Test retrain recommendation."""
        should_retrain, reason = validator.should_retrain()

        assert isinstance(should_retrain, bool)
        assert isinstance(reason, str)

    def test_confidence_adjustment(self, validator):
        """Test confidence adjustment."""
        adjusted = validator.get_confidence_adjustment(70)

        assert 0 <= adjusted <= 100


class TestOddLotHandler:
    """Tests for OddLotHandler."""

    @pytest.fixture
    def handler(self):
        """Create fresh handler instance."""
        from src.utils.odd_lot_handler import OddLotHandler

        return OddLotHandler()

    def test_detect_odd_lots(self, handler):
        """Test odd-lot detection."""
        positions = {
            "VNM": {
                "shares": 550,
                "avg_price": 80000,
                "metadata": {"last_price": 82000},
            },
            "HPG": {
                "shares": 1000,
                "avg_price": 25000,
                "metadata": {"last_price": 26000},
            },
        }

        odd_lots = handler.detect_odd_lots(positions)

        assert len(odd_lots) == 1
        assert odd_lots[0].symbol == "VNM"
        assert odd_lots[0].odd_lot_shares == 50
        assert odd_lots[0].full_lot_shares == 500

    def test_detect_only_odd_lot(self, handler):
        """Test detection of position with only odd-lot."""
        positions = {
            "FPT": {
                "shares": 75,
                "avg_price": 120000,
                "metadata": {"last_price": 118000},
            },
        }

        odd_lots = handler.detect_odd_lots(positions)

        assert len(odd_lots) == 1
        assert odd_lots[0].odd_lot_shares == 75
        assert odd_lots[0].full_lot_shares == 0

    def test_exit_recommendation_profitable(self, handler):
        """Test exit recommendation for profitable odd-lot."""
        from src.utils.odd_lot_handler import OddLotPosition

        odd_lot = OddLotPosition(
            symbol="VNM",
            odd_lot_shares=50,
            full_lot_shares=500,
            total_shares=550,
            avg_price=80000,
            current_price=84000,
            unrealized_pnl_pct=0.05,
        )

        rec = handler.get_exit_recommendation(odd_lot, "SIDEWAYS", days_held=2)

        assert rec.action == "SELL_NOW"
        assert rec.priority >= 3

    def test_exit_recommendation_small_value(self, handler):
        """Test exit recommendation for small value odd-lot."""
        from src.utils.odd_lot_handler import OddLotPosition

        odd_lot = OddLotPosition(
            symbol="ABC",
            odd_lot_shares=10,
            full_lot_shares=0,
            total_shares=10,
            avg_price=5000,
            current_price=5100,
            unrealized_pnl_pct=0.02,
        )

        rec = handler.get_exit_recommendation(odd_lot, "SIDEWAYS", days_held=1)

        assert rec.action == "IGNORE"
        assert rec.priority == 1

    def test_should_avoid_creating_odd_lot(self, handler):
        """Test odd-lot avoidance check."""
        would_create, recommended = handler.should_avoid_creating_odd_lot(
            current_shares=1000,
            shares_to_sell=350,
        )

        assert would_create == True
        assert recommended % 100 == 0  # Should be rounded to lot size

    def test_should_not_create_odd_lot(self, handler):
        """Test when no odd-lot would be created."""
        would_create, recommended = handler.should_avoid_creating_odd_lot(
            current_shares=1000,
            shares_to_sell=400,
        )

        assert would_create == False
        assert recommended == 400

    def test_round_to_lot_size_down(self, handler):
        """Test rounding down to lot size."""
        assert handler.round_to_lot_size(550, round_up=False) == 500
        assert handler.round_to_lot_size(199, round_up=False) == 100

    def test_round_to_lot_size_up(self, handler):
        """Test rounding up to lot size."""
        assert handler.round_to_lot_size(550, round_up=True) == 600
        assert handler.round_to_lot_size(101, round_up=True) == 200

    def test_cleanup_cost_calculation(self, handler):
        """Test cleanup cost calculation."""
        from src.utils.odd_lot_handler import OddLotPosition

        odd_lots = [
            OddLotPosition(
                symbol="VNM",
                odd_lot_shares=50,
                full_lot_shares=500,
                total_shares=550,
                avg_price=80000,
                current_price=82000,
                unrealized_pnl_pct=0.025,
            ),
        ]

        cost = handler.calculate_cleanup_cost(odd_lots, "SIDEWAYS")

        assert cost["num_odd_lots"] == 1
        assert cost["total_odd_lot_value"] == 50 * 82000
        assert cost["total_estimated_cost"] > 0


class TestLiveConsistencyChecker:
    """Tests for LiveConsistencyChecker."""

    @pytest.fixture
    def checker(self):
        """Create fresh checker instance."""
        from src.backtesting.live_consistency_checker import LiveConsistencyChecker

        return LiveConsistencyChecker()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="D")

        df = pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(200) * 0.5),
                "close": 100 + np.cumsum(np.random.randn(200) * 0.5),
                "high": 0,
                "low": 0,
                "volume": np.random.randint(100000, 1000000, 200),
            },
            index=dates,
        )

        df["high"] = df[["open", "close"]].max(axis=1) + np.random.rand(200) * 0.5
        df["low"] = df[["open", "close"]].min(axis=1) - np.random.rand(200) * 0.5

        return df

    def test_indicator_consistency_same_function(self, checker, sample_df):
        """Test indicator consistency with same function."""

        def calc_sma(data, period=20):
            return data["close"].rolling(period).mean()

        result = checker.check_indicator_consistency(
            sample_df,
            "SMA_20",
            calc_sma,
            calc_sma,
            {"period": 20},
        )

        assert result == True

    def test_indicator_consistency_different_function(self, checker, sample_df):
        """Test indicator consistency with different functions."""

        def calc_sma_v1(data, period=20):
            return data["close"].rolling(period).mean()

        def calc_sma_v2(data, period=20):
            # Slightly different calculation
            return data["close"].rolling(period).mean() + 0.1

        result = checker.check_indicator_consistency(
            sample_df,
            "SMA_20",
            calc_sma_v1,
            calc_sma_v2,
            {"period": 20},
        )

        assert result == False

    def test_transaction_cost_consistency_same(self, checker):
        """Test transaction cost consistency with same values."""
        result = checker.check_transaction_cost_consistency(0.01, 0.01)
        assert result == True

    def test_transaction_cost_consistency_different(self, checker):
        """Test transaction cost consistency with different values."""
        result = checker.check_transaction_cost_consistency(0.008, 0.015)
        # Large difference (0.7%) is CRITICAL, so it should fail
        # This is expected behavior - we want to catch significant cost differences
        assert result == False

    def test_look_ahead_bias_no_bias(self, checker, sample_df):
        """Test look-ahead bias detection with clean function."""

        def calc_sma(data, period=20):
            return data["close"].rolling(period).mean()

        result = checker.check_look_ahead_bias(sample_df, calc_sma, "SMA_20")
        assert result == True

    def test_generate_report(self, checker, sample_df):
        """Test report generation."""

        def calc_sma(data, period=20):
            return data["close"].rolling(period).mean()

        checker.check_indicator_consistency(sample_df, "SMA_20", calc_sma, calc_sma, {"period": 20})
        checker.check_transaction_cost_consistency(0.01, 0.01)

        report = checker.generate_report()

        assert report.total_checks >= 2
        assert report.passed_checks >= 2
        assert isinstance(report.timestamp, str)

    def test_reset(self, checker, sample_df):
        """Test checker reset."""

        def calc_sma(data, period=20):
            return data["close"].rolling(period).mean()

        checker.check_indicator_consistency(sample_df, "SMA_20", calc_sma, calc_sma, {"period": 20})

        checker.reset()
        report = checker.generate_report()

        assert report.total_checks == 0
        assert len(report.issues) == 0


# Cleanup test files after tests
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield

    import os

    test_files = [
        "test_execution_stats.json",
        "test_ml_stats.json",
    ]

    for f in test_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass
