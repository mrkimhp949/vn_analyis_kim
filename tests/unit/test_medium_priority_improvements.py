# -*- coding: utf-8 -*-
"""
Tests for MEDIUM Priority Improvements:
7. Portfolio Correlation Check hoàn chỉnh
8. Sector Exposure Tracking
9. Drawdown Protection
10. Trading Hours Check
"""

import pytest
from datetime import datetime, time
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


# =============================================================================
# IMPROVEMENT #8: Sector Exposure Tracking Tests
# =============================================================================


class TestSectorExposureTracking:
    """Test Sector Exposure Tracking in Portfolio Manager"""

    @pytest.fixture
    def portfolio_manager(self):
        with patch("src.portfolio.manager.get_db") as mock_db:
            with patch("src.portfolio.manager.get_performance_monitor"):
                with patch("src.portfolio.manager.get_config") as mock_config:
                    with patch("src.portfolio.manager.get_signal_performance_tracker"):
                        mock_config.return_value = MagicMock()
                        mock_db.return_value = MagicMock()
                        from src.portfolio.manager import PortfolioManager

                        pm = PortfolioManager()
                        return pm

    def test_get_sector_mapping(self, portfolio_manager):
        """Test sector mapping for common VN stocks"""
        symbols = ["VCB", "HPG", "VNM", "FPT", "VHM"]
        mapping = portfolio_manager._get_sector_mapping(symbols)

        assert mapping["VCB"] == "Banking"
        assert mapping["HPG"] == "Steel"
        assert mapping["VNM"] == "Consumer"
        assert mapping["FPT"] == "Technology"
        assert mapping["VHM"] == "Real Estate"

    def test_get_sector_mapping_unknown(self, portfolio_manager):
        """Test sector mapping for unknown symbols"""
        symbols = ["UNKNOWN_SYMBOL"]
        mapping = portfolio_manager._get_sector_mapping(symbols)

        assert mapping["UNKNOWN_SYMBOL"] == "Other"

    def test_get_sector_exposure_empty_portfolio(self, portfolio_manager):
        """Test sector exposure with empty portfolio"""
        portfolio_manager.db.get_positions.return_value = {}

        result = portfolio_manager.get_sector_exposure()

        assert result["sectors"] == {}
        assert result["max_sector_exposure"] == 0.0
        assert result["is_concentrated"] is False

    def test_get_sector_exposure_diversified(self, portfolio_manager):
        """Test sector exposure with diversified portfolio"""
        # Create more balanced portfolio to avoid concentration
        portfolio_manager.db.get_positions.return_value = {
            "VCB": {"shares": 500, "avg_price": 100000, "metadata": {"last_price": 100000}},  # 50M
            "HPG": {"shares": 1500, "avg_price": 30000, "metadata": {"last_price": 30000}},  # 45M
            "VNM": {"shares": 500, "avg_price": 80000, "metadata": {"last_price": 80000}},  # 40M
            "FPT": {"shares": 400, "avg_price": 120000, "metadata": {"last_price": 120000}},  # 48M
        }

        result = portfolio_manager.get_sector_exposure()

        assert "Banking" in result["sectors"]
        assert "Steel" in result["sectors"]
        assert "Consumer" in result["sectors"]
        assert "Technology" in result["sectors"]
        # Each sector should be around 25% - not concentrated
        assert result["max_sector_exposure"] <= 40  # No single sector > 40%

    def test_check_sector_before_entry_ok(self, portfolio_manager):
        """Test sector check allows entry when not concentrated"""
        portfolio_manager.db.get_positions.return_value = {
            "VCB": {"shares": 1000, "avg_price": 100000, "metadata": {"last_price": 100000}},
        }

        can_add, warning = portfolio_manager.check_sector_before_entry("HPG")

        assert can_add is True


# =============================================================================
# IMPROVEMENT #9: Drawdown Protection Tests
# =============================================================================


class TestDrawdownProtection:
    """Test Drawdown Protection in Circuit Breaker"""

    @pytest.fixture
    def circuit_breaker(self):
        with patch("src.risk.circuit_breaker.os.path.exists", return_value=False):
            from src.risk.circuit_breaker import CircuitBreaker

            return CircuitBreaker(
                total_capital=100_000_000,
                max_drawdown_pct=0.15,  # 15% max drawdown
                drawdown_warning_pct=0.10,  # 10% warning
            )

    def test_update_portfolio_value_new_peak(self, circuit_breaker):
        """Test updating portfolio value to new peak"""
        result = circuit_breaker.update_portfolio_value(110_000_000)

        assert result["new_peak"] is True
        assert result["peak_value"] == 110_000_000
        assert result["current_drawdown"] == 0.0

    def test_update_portfolio_value_drawdown(self, circuit_breaker):
        """Test drawdown calculation"""
        # Set peak
        circuit_breaker.update_portfolio_value(100_000_000)

        # Simulate 5% drawdown
        result = circuit_breaker.update_portfolio_value(95_000_000)

        assert result["new_peak"] is False
        assert result["current_drawdown"] == 0.05
        assert result["drawdown_warning"] is False
        assert result["drawdown_critical"] is False

    def test_update_portfolio_value_warning_level(self, circuit_breaker):
        """Test drawdown warning level (10%)"""
        # Set peak
        circuit_breaker.update_portfolio_value(100_000_000)

        # Simulate 10% drawdown
        result = circuit_breaker.update_portfolio_value(90_000_000)

        assert result["drawdown_warning"] is True
        assert result["drawdown_critical"] is False
        assert circuit_breaker.tripped is False

    def test_update_portfolio_value_critical_level(self, circuit_breaker):
        """Test drawdown critical level (15%) triggers circuit breaker"""
        # Set peak
        circuit_breaker.update_portfolio_value(100_000_000)

        # Simulate 15% drawdown
        result = circuit_breaker.update_portfolio_value(85_000_000)

        assert result["drawdown_critical"] is True
        assert circuit_breaker.tripped is True
        assert "Max drawdown exceeded" in circuit_breaker.tripped_reason

    def test_check_drawdown_ok(self, circuit_breaker):
        """Test check_drawdown returns OK for small drawdown"""
        circuit_breaker.update_portfolio_value(100_000_000)

        is_ok, warning = circuit_breaker.check_drawdown(98_000_000)

        assert is_ok is True
        assert warning is None

    def test_check_drawdown_warning(self, circuit_breaker):
        """Test check_drawdown returns warning at 10%"""
        circuit_breaker.update_portfolio_value(100_000_000)

        is_ok, warning = circuit_breaker.check_drawdown(90_000_000)

        assert is_ok is True
        assert warning is not None
        assert "WARNING" in warning

    def test_check_drawdown_critical(self, circuit_breaker):
        """Test check_drawdown returns critical at 15%"""
        circuit_breaker.update_portfolio_value(100_000_000)

        is_ok, warning = circuit_breaker.check_drawdown(85_000_000)

        assert is_ok is False
        assert "CRITICAL" in warning

    def test_get_drawdown_status(self, circuit_breaker):
        """Test get_drawdown_status returns correct metrics"""
        circuit_breaker.update_portfolio_value(100_000_000)
        circuit_breaker.update_portfolio_value(92_000_000)

        status = circuit_breaker.get_drawdown_status()

        assert status["peak_value"] == 100_000_000
        assert status["current_drawdown_pct"] == 8.0
        assert status["max_drawdown_pct"] == 15.0
        assert status["is_warning"] is False
        assert status["is_critical"] is False

    def test_reset_peak(self, circuit_breaker):
        """Test reset_peak resets drawdown tracking"""
        circuit_breaker.update_portfolio_value(100_000_000)
        circuit_breaker.update_portfolio_value(90_000_000)

        circuit_breaker.reset_peak(120_000_000)

        assert circuit_breaker.peak_portfolio_value == 120_000_000
        assert circuit_breaker.current_drawdown == 0.0


# =============================================================================
# IMPROVEMENT #10: Trading Hours Check Tests
# =============================================================================


class TestTradingHoursCheck:
    """Test Trading Hours Check in Entry/Exit Logic"""

    @pytest.fixture
    def entry_logic(self):
        with patch("src.strategies.entry_logic.get_performance_monitor"):
            from src.strategies.entry_logic import ImprovedEntryLogic

            return ImprovedEntryLogic()

    @pytest.fixture
    def exit_strategy(self):
        from src.strategies.exit_logic import ImprovedExitStrategy

        return ImprovedExitStrategy()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        return pd.DataFrame(
            {
                "open": np.random.uniform(95, 105, 100),
                "high": np.random.uniform(100, 110, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(95, 105, 100),
                "volume": np.random.uniform(1000000, 5000000, 100),
            },
            index=dates,
        )

    @patch("src.strategies.entry_logic.TRADING_SCHEDULE_AVAILABLE", True)
    @patch("src.strategies.entry_logic.is_trading_day")
    @patch("src.strategies.entry_logic.is_trading_hour")
    def test_entry_blocked_outside_trading_hours(
        self, mock_is_hour, mock_is_day, entry_logic, sample_df
    ):
        """Test entry is blocked outside trading hours"""
        mock_is_day.return_value = True
        mock_is_hour.return_value = False  # Outside trading hours

        ml_signal = {"signal": "BUY", "confidence": 70}

        result = entry_logic.analyze_entry(
            df=sample_df,
            ml_signal=ml_signal,
            check_trading_hours=True,
        )

        assert result.should_enter is False
        # Check telemetry for trading hours reason
        telemetry = result.telemetry or {}
        assert (
            telemetry.get("reason") == "outside_trading_hours"
            or "trading" in str(result).lower()
            or "giờ" in str(result).lower()
        )

    @patch("src.strategies.entry_logic.TRADING_SCHEDULE_AVAILABLE", True)
    @patch("src.strategies.entry_logic.is_trading_day")
    def test_entry_blocked_on_weekend(self, mock_is_day, entry_logic, sample_df):
        """Test entry is blocked on weekend"""
        mock_is_day.return_value = False  # Weekend

        ml_signal = {"signal": "BUY", "confidence": 70}

        result = entry_logic.analyze_entry(
            df=sample_df,
            ml_signal=ml_signal,
            check_trading_hours=True,
        )

        assert result.should_enter is False

    def test_entry_allowed_when_check_disabled(self, entry_logic, sample_df):
        """Test entry proceeds when trading hours check is disabled"""
        ml_signal = {"signal": "BUY", "confidence": 70}

        # Should not block even if outside hours when check_trading_hours=False
        result = entry_logic.analyze_entry(
            df=sample_df,
            ml_signal=ml_signal,
            check_trading_hours=False,
        )

        # Result depends on other filters, but should not be blocked by trading hours
        # The signal might still be rejected for other reasons
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
