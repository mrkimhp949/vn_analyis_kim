# -*- coding: utf-8 -*-
"""
Tests for Vietnam Market Improvements

Tests:
1. Trading schedule (weekend fix, holidays)
2. Session boundary detection
3. Per-symbol circuit breaker integration
4. Price floor/ceiling validation
5. Position size vs volume validation
6. HIGH_VOLATILITY regime handling
"""

import pytest
from datetime import datetime, time, timedelta
from unittest.mock import MagicMock, patch

import pytz


class TestTradingSchedule:
    """Test Vietnam trading schedule fixes"""

    def test_non_trading_days_are_weekend(self):
        """Test that non-trading days are Saturday (5) and Sunday (6)"""
        from src.market.schedule import NON_TRADING_DAYS

        assert 5 in NON_TRADING_DAYS  # Saturday
        assert 6 in NON_TRADING_DAYS  # Sunday
        assert 2 not in NON_TRADING_DAYS  # Tuesday should NOT be non-trading

    def test_is_trading_day_weekday(self):
        """Test weekdays are trading days"""
        from src.market.schedule import is_trading_day, VN_TZ

        # Monday
        monday = VN_TZ.localize(datetime(2024, 1, 8, 10, 0))  # A Monday
        assert is_trading_day(monday) is True

        # Friday
        friday = VN_TZ.localize(datetime(2024, 1, 12, 10, 0))  # A Friday
        assert is_trading_day(friday) is True

    def test_is_trading_day_weekend(self):
        """Test weekends are not trading days"""
        from src.market.schedule import is_trading_day, VN_TZ

        # Saturday
        saturday = VN_TZ.localize(datetime(2024, 1, 13, 10, 0))
        assert is_trading_day(saturday) is False

        # Sunday
        sunday = VN_TZ.localize(datetime(2024, 1, 14, 10, 0))
        assert is_trading_day(sunday) is False

    def test_is_public_holiday(self):
        """Test public holiday detection"""
        from src.market.schedule import is_public_holiday, VN_TZ

        # New Year's Day
        new_year = VN_TZ.localize(datetime(2024, 1, 1, 10, 0))
        assert is_public_holiday(new_year) is True

        # National Day
        national_day = VN_TZ.localize(datetime(2024, 9, 2, 10, 0))
        assert is_public_holiday(national_day) is True

        # Regular day
        regular_day = VN_TZ.localize(datetime(2024, 3, 15, 10, 0))
        assert is_public_holiday(regular_day) is False


class TestSessionBoundary:
    """Test session boundary detection"""

    def test_near_am_end(self):
        """Test detection near morning session end (11:30)"""
        from src.market.schedule import is_near_session_boundary, VN_TZ

        # 11:27 - within 5 minutes of 11:30
        near_am_end = VN_TZ.localize(datetime(2024, 1, 8, 11, 27))
        is_near, boundary_type = is_near_session_boundary(near_am_end, minutes=5)
        assert is_near is True
        assert boundary_type == "AM_END"

    def test_near_pm_start(self):
        """Test detection near afternoon session start (13:00)"""
        from src.market.schedule import is_near_session_boundary, VN_TZ

        # 13:03 - within 5 minutes of 13:00
        near_pm_start = VN_TZ.localize(datetime(2024, 1, 8, 13, 3))
        is_near, boundary_type = is_near_session_boundary(near_pm_start, minutes=5)
        assert is_near is True
        assert boundary_type == "PM_START"

    def test_not_near_boundary(self):
        """Test normal trading time (not near boundary)"""
        from src.market.schedule import is_near_session_boundary, VN_TZ

        # 10:00 - middle of morning session
        mid_session = VN_TZ.localize(datetime(2024, 1, 8, 10, 0))
        is_near, boundary_type = is_near_session_boundary(mid_session, minutes=5)
        assert is_near is False
        assert boundary_type == ""


class TestPerSymbolCircuitBreaker:
    """Test per-symbol circuit breaker"""

    def test_new_symbol_can_trade(self):
        """Test new symbol can trade"""
        from src.risk.per_symbol_circuit_breaker import PerSymbolCircuitBreaker

        cb = PerSymbolCircuitBreaker(max_consecutive_losses=3)
        can_trade, reason = cb.can_trade("NEW_SYMBOL")
        assert can_trade is True

    def test_consecutive_losses_blocks(self):
        """Test consecutive losses blocks symbol"""
        from src.risk.per_symbol_circuit_breaker import PerSymbolCircuitBreaker

        cb = PerSymbolCircuitBreaker(
            max_consecutive_losses=3,
            min_trades_for_winrate_check=10,  # High threshold to avoid win rate block
        )

        # Record 3 consecutive losses
        for _ in range(3):
            cb.record_trade("LOSER", is_win=False, pnl_percent=-2.0)

        can_trade, reason = cb.can_trade("LOSER")
        assert can_trade is False
        # Could be blocked by consecutive losses or win rate
        assert "blocked" in reason.lower()

    def test_win_resets_consecutive_losses(self):
        """Test winning trade resets consecutive losses"""
        from src.risk.per_symbol_circuit_breaker import PerSymbolCircuitBreaker
        import tempfile
        import os

        # Use temp file to avoid loading existing stats
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            cb = PerSymbolCircuitBreaker(
                max_consecutive_losses=3,
                min_trades_for_winrate_check=10,  # High threshold to avoid win rate block
                stats_file=temp_file,
            )

            # Use unique symbol to avoid conflicts
            symbol = "TEST_RESET_123"

            # Record 2 losses
            cb.record_trade(symbol, is_win=False, pnl_percent=-2.0)
            cb.record_trade(symbol, is_win=False, pnl_percent=-2.0)

            # Record 1 win
            cb.record_trade(symbol, is_win=True, pnl_percent=5.0)

            # Check consecutive losses reset
            stats = cb.get_symbol_stats(symbol)
            assert stats.consecutive_losses == 0

            # Should be able to trade (not blocked by consecutive losses)
            # With 3 trades and 1 win, win rate is 33% which is above 30% threshold
            # And min_trades_for_winrate_check=10 so win rate check won't trigger
            can_trade, reason = cb.can_trade(symbol)
            assert can_trade is True, f"Expected can_trade=True but got False: {reason}"
        finally:
            # Cleanup temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestVietnamMarketValidator:
    """Test Vietnam market validator"""

    def test_price_near_ceiling(self):
        """Test detection of price near ceiling"""
        from src.utils.vietnam_market import VietnamMarketValidator

        validator = VietnamMarketValidator()

        # Price at 6.5% above reference (ceiling is 7%)
        reference = 100000
        current = 106500  # 6.5% above

        is_safe, warning = validator.check_price_floor_ceiling(current, reference, "TEST")
        assert is_safe is False
        assert "CEILING" in warning

    def test_price_near_floor(self):
        """Test detection of price near floor"""
        from src.utils.vietnam_market import VietnamMarketValidator

        validator = VietnamMarketValidator()

        # Price at 6.5% below reference (floor is -7%)
        reference = 100000
        current = 93500  # 6.5% below

        is_safe, warning = validator.check_price_floor_ceiling(current, reference, "TEST")
        assert is_safe is False
        assert "FLOOR" in warning

    def test_price_safe(self):
        """Test safe price (not near limits)"""
        from src.utils.vietnam_market import VietnamMarketValidator

        validator = VietnamMarketValidator()

        # Price at 3% above reference
        reference = 100000
        current = 103000

        is_safe, warning = validator.check_price_floor_ceiling(current, reference, "TEST")
        assert is_safe is True
        assert warning is None

    def test_position_vs_volume_too_large(self):
        """Test position too large relative to volume"""
        from src.utils.vietnam_market import VietnamMarketValidator

        validator = VietnamMarketValidator()
        validator.max_position_pct_of_volume = 0.05  # 5%

        # Position is 10% of daily volume
        is_safe, warning = validator.validate_position_size_vs_volume(
            position_shares=10000, avg_daily_volume=100000, symbol="TEST"
        )
        assert is_safe is False
        assert "too large" in warning.lower()

    def test_position_vs_volume_safe(self):
        """Test safe position size relative to volume"""
        from src.utils.vietnam_market import VietnamMarketValidator

        validator = VietnamMarketValidator()
        validator.max_position_pct_of_volume = 0.05  # 5%

        # Position is 3% of daily volume
        is_safe, warning = validator.validate_position_size_vs_volume(
            position_shares=3000, avg_daily_volume=100000, symbol="TEST"
        )
        assert is_safe is True
        assert warning is None


class TestHighVolatilityRegime:
    """Test HIGH_VOLATILITY regime handling"""

    def test_strategy_manager_high_volatility(self):
        """Test strategy manager applies conservative settings for HIGH_VOLATILITY"""
        from src.strategies.manager import StrategyManager

        manager = StrategyManager()

        # Apply HIGH_VOLATILITY regime
        market_regime = {
            "regime": "HIGH_VOLATILITY",
            "confidence": 80,
            "tradeable": False,
        }
        manager.apply_market_adjustments(market_regime)

        # Check conservative settings
        assert manager.entry_logic.min_confidence == 70
        assert manager.entry_logic.min_risk_reward == 2.5
        if hasattr(manager.position_sizer, "max_total_exposure"):
            assert manager.position_sizer.max_total_exposure == 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
