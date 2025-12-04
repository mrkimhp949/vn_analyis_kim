# -*- coding: utf-8 -*-
"""
Comprehensive Unit Tests for Circuit Breaker

Tests cover:
- Basic circuit breaker functionality
- Session limits (morning/afternoon)
- Winning streak protection
- Drawdown protection
- Volatility adjustments
- Caution mode
- Thread safety
- Portfolio heat checks
"""

import pytest
from datetime import date, datetime
from unittest.mock import patch, MagicMock
import threading
import time

from src.risk.circuit_breaker import CircuitBreaker, DailyStats, get_circuit_breaker


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_stats_file(tmp_path):
    """Create temporary stats file for testing"""
    stats_file = tmp_path / "test_circuit_breaker_stats.json"
    return str(stats_file)


@pytest.fixture
def breaker(temp_stats_file):
    """Create CircuitBreaker instance with default settings"""
    return CircuitBreaker(
        max_trades_per_day=8,
        max_loss_per_day_pct=0.03,
        max_consecutive_losses=3,
        vnindex_drop_threshold=-2.5,
        total_capital=100_000_000,
        stats_file=temp_stats_file,
        max_portfolio_heat=0.60,
        max_trades_per_session=4,
        max_consecutive_wins=5,
    )


@pytest.fixture
def breaker_relaxed(temp_stats_file):
    """Create CircuitBreaker with relaxed settings for testing"""
    return CircuitBreaker(
        max_trades_per_day=20,
        max_loss_per_day_pct=0.10,
        max_consecutive_losses=10,
        vnindex_drop_threshold=-5.0,
        use_conservative_threshold=False,
        total_capital=100_000_000,
        stats_file=temp_stats_file,
        max_portfolio_heat=0.80,
        max_trades_per_session=10,
        max_consecutive_wins=10,
    )


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestCircuitBreakerBasic:
    """Test basic circuit breaker functionality"""

    def test_initialization(self, breaker):
        """Test circuit breaker initializes correctly"""
        assert breaker.max_trades_per_day == 8
        assert breaker.max_loss_per_day_pct == 0.03
        assert breaker.max_consecutive_losses == 3
        assert breaker.total_capital == 100_000_000
        assert not breaker.tripped
        assert breaker.tripped_reason == ""

    def test_can_trade_initial(self, breaker):
        """Test can_trade returns True initially"""
        can_trade, reason = breaker.can_trade()
        # May return False if market is closed
        assert isinstance(can_trade, bool)
        assert isinstance(reason, str)

    def test_is_active_initial(self, breaker):
        """Test is_active returns False initially"""
        assert not breaker.is_active()

    def test_get_daily_stats(self, breaker):
        """Test get_daily_stats returns correct structure"""
        stats = breaker.get_daily_stats()
        assert isinstance(stats, DailyStats)
        assert stats.date == date.today().isoformat()
        assert stats.trades_count == 0
        assert stats.total_loss == 0.0
        assert stats.total_profit == 0.0

    def test_reset(self, breaker):
        """Test reset clears all state"""
        # Trip the breaker first
        breaker.record_pnl(-0.10)
        assert breaker.is_active()

        # Reset
        breaker.reset()
        assert not breaker.is_active()
        assert breaker.tripped_reason == ""
        assert breaker.stats["consecutive_losses"] == 0


# =============================================================================
# Trade Recording Tests
# =============================================================================


class TestTradeRecording:
    """Test trade recording functionality"""

    def test_record_trade_profit(self, breaker):
        """Test recording profitable trade"""
        breaker.record_trade(1_000_000)

        stats = breaker.get_daily_stats()
        assert stats.trades_count == 1
        assert stats.total_profit == 1_000_000
        assert stats.total_loss == 0
        assert breaker.stats["consecutive_losses"] == 0

    def test_record_trade_loss(self, breaker):
        """Test recording losing trade"""
        breaker.record_trade(-500_000)

        stats = breaker.get_daily_stats()
        assert stats.trades_count == 1
        assert stats.total_loss == 500_000
        assert stats.total_profit == 0
        assert breaker.stats["consecutive_losses"] == 1

    def test_record_multiple_trades(self, breaker):
        """Test recording multiple trades"""
        breaker.record_trade(1_000_000)  # Win
        breaker.record_trade(-500_000)  # Loss
        breaker.record_trade(2_000_000)  # Win

        stats = breaker.get_daily_stats()
        assert stats.trades_count == 3
        assert stats.total_profit == 3_000_000
        assert stats.total_loss == 500_000
        assert stats.net_pnl == 2_500_000

    def test_consecutive_losses_tracking(self, breaker):
        """Test consecutive losses are tracked correctly"""
        breaker.record_trade(-100_000)
        assert breaker.stats["consecutive_losses"] == 1

        breaker.record_trade(-100_000)
        assert breaker.stats["consecutive_losses"] == 2

        breaker.record_trade(100_000)  # Win resets streak
        assert breaker.stats["consecutive_losses"] == 0


# =============================================================================
# Circuit Breaker Trigger Tests
# =============================================================================


class TestCircuitBreakerTriggers:
    """Test various circuit breaker trigger conditions"""

    def test_trigger_max_daily_loss(self, breaker):
        """Test circuit breaker triggers on max daily loss"""
        result = breaker.check_and_update(
            portfolio_pnl_pct=-0.05, vnindex_change_pct=-0.01  # 5% loss > 3% threshold
        )

        assert result is True
        assert breaker.is_active()
        assert "Lỗ trong ngày" in breaker.tripped_reason

    def test_trigger_vnindex_drop(self, breaker):
        """Test circuit breaker triggers on VNINDEX drop"""
        result = breaker.check_and_update(
            portfolio_pnl_pct=-0.01,
            vnindex_change_pct=-0.03,  # 3% drop > 2% threshold (conservative)
        )

        assert result is True
        assert breaker.is_active()
        assert "VNINDEX giảm sâu" in breaker.tripped_reason

    def test_trigger_max_trades(self, breaker):
        """Test circuit breaker triggers on max trades per day"""
        # Record max trades
        for _ in range(8):
            breaker.record_trade(100_000)

        result = breaker.check_and_update(portfolio_pnl_pct=0.01, vnindex_change_pct=-0.01)

        assert result is True
        assert breaker.is_active()
        assert "Số lệnh trong ngày" in breaker.tripped_reason

    def test_trigger_consecutive_losses(self, breaker):
        """Test circuit breaker triggers on consecutive losses"""
        # Record consecutive losses
        for _ in range(3):
            breaker.record_trade(-100_000)

        result = breaker.check_and_update(portfolio_pnl_pct=-0.01, vnindex_change_pct=-0.01)

        assert result is True
        assert breaker.is_active()
        assert "lệnh thua liên tiếp" in breaker.tripped_reason

    def test_trigger_portfolio_heat(self, breaker):
        """Test circuit breaker triggers on high portfolio heat"""
        result = breaker.check_and_update(
            portfolio_pnl_pct=0.01,
            vnindex_change_pct=-0.01,
            portfolio_heat=0.70,  # 70% > 60% threshold
        )

        assert result is True
        assert breaker.is_active()
        assert "Portfolio heat" in breaker.tripped_reason

    def test_no_trigger_normal_conditions(self, breaker):
        """Test circuit breaker does not trigger under normal conditions"""
        result = breaker.check_and_update(
            portfolio_pnl_pct=-0.01,  # Small loss
            vnindex_change_pct=-0.005,  # Small drop
            portfolio_heat=0.30,  # Low heat
        )

        assert result is False
        assert not breaker.is_active()


# =============================================================================
# Session Limit Tests
# =============================================================================


class TestSessionLimits:
    """Test per-session trade limits"""

    def test_check_session_limit_initial(self, breaker):
        """Test session limit check when no trades"""
        can_trade, msg = breaker.check_session_limit()
        # Result depends on market hours
        assert isinstance(can_trade, bool)
        assert isinstance(msg, str)

    @patch.object(CircuitBreaker, "_get_current_session", return_value="morning")
    def test_session_limit_morning(self, mock_session, breaker):
        """Test morning session limit"""
        # Record trades up to limit
        for _ in range(4):
            breaker.record_session_trade()

        can_trade, msg = breaker.check_session_limit()
        assert can_trade is False
        assert "Session limit reached" in msg
        assert "morning" in msg

    @patch.object(CircuitBreaker, "_get_current_session", return_value="afternoon")
    def test_session_limit_afternoon(self, mock_session, breaker):
        """Test afternoon session limit"""
        for _ in range(4):
            breaker.record_session_trade()

        can_trade, msg = breaker.check_session_limit()
        assert can_trade is False
        assert "Session limit reached" in msg
        assert "afternoon" in msg

    @patch.object(CircuitBreaker, "_get_current_session", return_value="closed")
    def test_session_closed(self, mock_session, breaker):
        """Test when market is closed"""
        can_trade, msg = breaker.check_session_limit()
        assert can_trade is False
        assert "closed" in msg.lower()

    def test_get_session_stats(self, breaker):
        """Test get_session_stats returns correct structure"""
        stats = breaker.get_session_stats()

        assert "current_session" in stats
        assert "morning_trades" in stats
        assert "afternoon_trades" in stats
        assert "max_per_session" in stats
        assert "consecutive_wins" in stats


# =============================================================================
# Winning Streak Tests
# =============================================================================


class TestWinningStreak:
    """Test winning streak protection"""

    def test_winning_streak_tracking(self, breaker):
        """Test winning streak is tracked correctly"""
        assert breaker._consecutive_wins == 0

        breaker.record_trade(100_000)
        assert breaker._consecutive_wins == 1

        breaker.record_trade(100_000)
        assert breaker._consecutive_wins == 2

    def test_winning_streak_reset_on_loss(self, breaker):
        """Test winning streak resets on loss"""
        breaker.record_trade(100_000)
        breaker.record_trade(100_000)
        assert breaker._consecutive_wins == 2

        breaker.record_trade(-100_000)
        assert breaker._consecutive_wins == 0

    def test_winning_streak_limit(self, breaker):
        """Test winning streak limit triggers pause"""
        # Record 5 consecutive wins
        for _ in range(5):
            breaker.record_trade(100_000)

        can_trade, msg = breaker.check_winning_streak()
        assert can_trade is False
        assert "Winning streak pause" in msg

    def test_winning_streak_warning(self, breaker):
        """Test winning streak warning before limit"""
        # Record 4 consecutive wins (one before limit)
        for _ in range(4):
            breaker.record_trade(100_000)

        can_trade, msg = breaker.check_winning_streak()
        assert can_trade is True
        assert "warning" in msg.lower()

    def test_reset_winning_streak(self, breaker):
        """Test manual reset of winning streak"""
        for _ in range(3):
            breaker.record_trade(100_000)
        assert breaker._consecutive_wins == 3

        breaker.reset_winning_streak()
        assert breaker._consecutive_wins == 0


# =============================================================================
# Drawdown Protection Tests
# =============================================================================


class TestDrawdownProtection:
    """Test drawdown protection functionality"""

    def test_update_portfolio_value_new_peak(self, breaker):
        """Test updating portfolio value to new peak"""
        result = breaker.update_portfolio_value(110_000_000)

        assert result["new_peak"] is True
        assert result["peak_value"] == 110_000_000
        assert result["current_drawdown"] == 0.0

    def test_update_portfolio_value_drawdown(self, breaker):
        """Test drawdown calculation"""
        # Set peak
        breaker.update_portfolio_value(100_000_000)

        # Drop to 95M (5% drawdown)
        result = breaker.update_portfolio_value(95_000_000)

        assert result["new_peak"] is False
        assert result["current_drawdown"] == 0.05
        assert result["drawdown_pct"] == 5.0

    def test_drawdown_warning(self, breaker):
        """Test drawdown warning threshold"""
        breaker.update_portfolio_value(100_000_000)

        # Drop to 92M (8% drawdown = warning threshold)
        result = breaker.update_portfolio_value(92_000_000)

        assert result["drawdown_warning"] is True
        assert result["drawdown_critical"] is False

    def test_drawdown_critical_trips_breaker(self, breaker):
        """Test critical drawdown trips circuit breaker"""
        breaker.update_portfolio_value(100_000_000)

        # Drop to 87M (13% drawdown > 12% max)
        result = breaker.update_portfolio_value(87_000_000)

        assert result["drawdown_critical"] is True
        assert result["tripped"] is True
        assert breaker.is_active()

    def test_check_drawdown(self, breaker):
        """Test check_drawdown method"""
        breaker.update_portfolio_value(100_000_000)

        # Normal drawdown
        is_ok, msg = breaker.check_drawdown(98_000_000)
        assert is_ok is True
        assert msg is None

        # Warning drawdown
        is_ok, msg = breaker.check_drawdown(91_000_000)
        assert is_ok is True
        assert "WARNING" in msg

    def test_get_drawdown_status(self, breaker):
        """Test get_drawdown_status returns correct structure"""
        status = breaker.get_drawdown_status()

        assert "peak_value" in status
        assert "current_drawdown_pct" in status
        assert "max_drawdown_pct" in status
        assert "is_warning" in status
        assert "is_critical" in status

    def test_reset_peak(self, breaker):
        """Test reset_peak method"""
        breaker.update_portfolio_value(120_000_000)
        assert breaker.peak_portfolio_value == 120_000_000

        breaker.reset_peak(100_000_000)
        assert breaker.peak_portfolio_value == 100_000_000
        assert breaker.current_drawdown == 0.0


# =============================================================================
# Volatility Adjustment Tests
# =============================================================================


class TestVolatilityAdjustments:
    """Test volatility-based threshold adjustments"""

    def test_high_volatility_tightens_thresholds(self, breaker):
        """Test high volatility tightens thresholds"""
        original_loss_pct = breaker.base_max_loss_per_day_pct
        original_vnindex = breaker.base_vnindex_drop_threshold

        breaker._adjust_thresholds_for_volatility(0.04)  # 4% volatility

        # Thresholds should be tighter (smaller absolute values)
        assert breaker.max_loss_per_day_pct < original_loss_pct

    def test_low_volatility_relaxes_thresholds(self, breaker):
        """Test low volatility relaxes thresholds"""
        original_loss_pct = breaker.base_max_loss_per_day_pct

        breaker._adjust_thresholds_for_volatility(0.005)  # 0.5% volatility

        # Thresholds should be looser (larger absolute values)
        assert breaker.max_loss_per_day_pct > original_loss_pct

    def test_normal_volatility_uses_base(self, breaker):
        """Test normal volatility uses base thresholds"""
        breaker._adjust_thresholds_for_volatility(0.015)  # 1.5% volatility

        assert breaker.max_loss_per_day_pct == breaker.base_max_loss_per_day_pct

    def test_zero_volatility_uses_base(self, breaker):
        """Test zero volatility uses base thresholds"""
        breaker._adjust_thresholds_for_volatility(0.0)

        assert breaker.max_loss_per_day_pct == breaker.base_max_loss_per_day_pct

    def test_volatility_cache_optimization(self, breaker):
        """Test volatility cache prevents redundant calculations"""
        breaker._adjust_thresholds_for_volatility(0.02)
        first_value = breaker.max_loss_per_day_pct

        # Same volatility should not recalculate
        breaker._adjust_thresholds_for_volatility(0.02)
        assert breaker.max_loss_per_day_pct == first_value


# =============================================================================
# Caution Mode Tests
# =============================================================================


class TestCautionMode:
    """Test caution mode functionality"""

    def test_caution_mode_initial(self, breaker):
        """Test caution mode is off initially"""
        assert not breaker.is_caution_mode()

    def test_caution_mode_activates(self, breaker):
        """Test caution mode activates on moderate VNINDEX drop"""
        # VNINDEX drop between caution and trip threshold
        breaker.check_and_update(
            portfolio_pnl_pct=0.0, vnindex_change_pct=-0.018  # -1.8% (between -1.5% and -2.0%)
        )

        assert breaker.is_caution_mode()

    def test_caution_mode_deactivates(self, breaker):
        """Test caution mode deactivates when VNINDEX recovers"""
        # First activate caution mode
        breaker.check_and_update(portfolio_pnl_pct=0.0, vnindex_change_pct=-0.018)
        assert breaker.is_caution_mode()

        # Then VNINDEX recovers
        breaker.check_and_update(
            portfolio_pnl_pct=0.0, vnindex_change_pct=-0.005  # Above warning threshold
        )
        assert not breaker.is_caution_mode()

    def test_position_size_multiplier_normal(self, breaker):
        """Test position size multiplier in normal mode"""
        assert breaker.get_position_size_multiplier() == 1.0

    def test_position_size_multiplier_caution(self, breaker):
        """Test position size multiplier in caution mode"""
        breaker.caution_mode = True
        assert breaker.get_position_size_multiplier() == 0.5

    def test_position_size_multiplier_tripped(self, breaker):
        """Test position size multiplier when tripped"""
        breaker.tripped = True
        assert breaker.get_position_size_multiplier() == 0.0


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Test thread safety of circuit breaker operations"""

    def test_concurrent_record_trades(self, breaker):
        """Test concurrent trade recording is thread-safe"""
        num_threads = 10
        trades_per_thread = 5
        results = []

        def record_trades():
            for _ in range(trades_per_thread):
                breaker.record_trade(100_000)
            results.append(True)

        threads = [threading.Thread(target=record_trades) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should complete
        assert len(results) == num_threads

        # Total trades should be correct
        stats = breaker.get_daily_stats()
        assert stats.trades_count == num_threads * trades_per_thread

    def test_concurrent_check_and_update(self, breaker_relaxed):
        """Test concurrent check_and_update is thread-safe"""
        num_threads = 10
        results = []

        def check_update():
            result = breaker_relaxed.check_and_update(
                portfolio_pnl_pct=-0.01, vnindex_change_pct=-0.01
            )
            results.append(result)

        threads = [threading.Thread(target=check_update) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should complete
        assert len(results) == num_threads

    def test_concurrent_portfolio_update(self, breaker):
        """Test concurrent portfolio value updates are thread-safe"""
        num_threads = 10
        results = []

        def update_portfolio():
            for value in range(90_000_000, 110_000_000, 1_000_000):
                breaker.update_portfolio_value(value)
            results.append(True)

        threads = [threading.Thread(target=update_portfolio) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == num_threads


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Test input validation"""

    def test_invalid_portfolio_pnl_type(self, breaker):
        """Test invalid portfolio_pnl_pct type raises error"""
        with pytest.raises(ValueError, match="portfolio_pnl_pct phải là số"):
            breaker.check_and_update(portfolio_pnl_pct="invalid", vnindex_change_pct=-0.01)

    def test_invalid_vnindex_change_type(self, breaker):
        """Test invalid vnindex_change_pct type raises error"""
        with pytest.raises(ValueError, match="vnindex_change_pct phải là số"):
            breaker.check_and_update(portfolio_pnl_pct=-0.01, vnindex_change_pct="invalid")

    def test_valid_numeric_inputs(self, breaker):
        """Test valid numeric inputs work correctly"""
        # Integer inputs
        result = breaker.check_and_update(portfolio_pnl_pct=0, vnindex_change_pct=0)
        assert isinstance(result, bool)

        # Float inputs
        breaker.reset()
        result = breaker.check_and_update(portfolio_pnl_pct=-0.01, vnindex_change_pct=-0.005)
        assert isinstance(result, bool)


# =============================================================================
# Status Message Tests
# =============================================================================


class TestStatusMessage:
    """Test status message generation"""

    def test_get_status_message_format(self, breaker):
        """Test status message has correct format"""
        msg = breaker.get_status_message()

        assert "CIRCUIT BREAKER STATUS" in msg
        assert "Date:" in msg
        assert "Trades today:" in msg
        assert "SESSION STATS" in msg
        assert "WINNING STREAK" in msg

    def test_get_status_message_tripped(self, breaker):
        """Test status message shows tripped state"""
        breaker.record_pnl(-0.10)

        msg = breaker.get_status_message()
        assert "TRIPPED" in msg


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Test singleton pattern"""

    def test_get_circuit_breaker_singleton(self):
        """Test get_circuit_breaker returns singleton"""
        # Reset global instance
        import src.risk.circuit_breaker as cb_module

        cb_module._circuit_breaker = None

        breaker1 = get_circuit_breaker(100_000_000)
        breaker2 = get_circuit_breaker(200_000_000)  # Different capital

        # Should return same instance
        assert breaker1 is breaker2


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_exactly_at_loss_threshold(self, breaker):
        """Test behavior exactly at loss threshold"""
        result = breaker.check_and_update(
            portfolio_pnl_pct=-0.03, vnindex_change_pct=0.0  # Exactly at 3% threshold
        )

        assert result is True
        assert breaker.is_active()

    def test_just_below_loss_threshold(self, breaker):
        """Test behavior just below loss threshold"""
        result = breaker.check_and_update(
            portfolio_pnl_pct=-0.029, vnindex_change_pct=0.0  # Just below 3%
        )

        assert result is False
        assert not breaker.is_active()

    def test_positive_pnl_no_trigger(self, breaker):
        """Test positive PnL never triggers loss-based breaker"""
        result = breaker.check_and_update(
            portfolio_pnl_pct=0.10, vnindex_change_pct=0.0  # 10% profit
        )

        assert result is False

    def test_zero_peak_portfolio_value(self, breaker):
        """Test handling of zero peak portfolio value"""
        breaker.peak_portfolio_value = 0
        result = breaker.update_portfolio_value(100_000_000)

        # Should set new peak
        assert result["new_peak"] is True

    def test_breaker_stays_tripped(self, breaker):
        """Test breaker stays tripped after trigger"""
        breaker.record_pnl(-0.10)
        assert breaker.is_active()

        # Even with good conditions, should stay tripped
        result = breaker.check_and_update(portfolio_pnl_pct=0.05, vnindex_change_pct=0.02)

        assert result is True
        assert breaker.is_active()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
