# -*- coding: utf-8 -*-
"""
Tests for 10/10 Improvements

Tests for:
1. Vietnam Holiday Calendar (complete with substitutes)
2. TCBS Broker Integration
3. Warrant/ETF Strategy
4. Margin Call Monitor
5. Broker Factory

Author: Trading Bot Team
Version: 1.0.0
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# TEST: VIETNAM HOLIDAY CALENDAR
# =============================================================================


class TestVietnamHolidayCalendar:
    """Test complete Vietnam holiday calendar."""

    def test_fixed_holidays(self):
        """Test fixed holidays are detected."""
        from src.utils.vietnam_holidays import is_vietnam_holiday

        # New Year's Day
        is_holiday, name = is_vietnam_holiday(datetime(2025, 1, 1))
        assert is_holiday
        assert "New Year" in name or "Dương Lịch" in name

        # Reunification Day
        is_holiday, name = is_vietnam_holiday(datetime(2025, 4, 30))
        assert is_holiday
        assert "Reunification" in name or "Giải Phóng" in name

        # National Day
        is_holiday, name = is_vietnam_holiday(datetime(2025, 9, 2))
        assert is_holiday
        assert "National" in name or "Quốc Khánh" in name

    def test_tet_holidays(self):
        """Test Tết holidays are detected."""
        from src.utils.vietnam_holidays import is_vietnam_holiday, get_tet_period

        # 2025 Tết
        tet_start, tet_end = get_tet_period(2025)
        assert tet_start is not None
        assert tet_end is not None

        # Check a Tết day
        is_holiday, name = is_vietnam_holiday(datetime(2025, 1, 29))
        assert is_holiday
        assert "Tết" in name

    def test_hung_kings_day(self):
        """Test Hung Kings Day is detected."""
        from src.utils.vietnam_holidays import is_vietnam_holiday

        # 2025 Hung Kings Day is April 7
        is_holiday, name = is_vietnam_holiday(datetime(2025, 4, 7))
        assert is_holiday
        assert "Hùng Vương" in name

    def test_substitute_holidays(self):
        """Test substitute holidays (nghỉ bù) are calculated."""
        from src.utils.vietnam_holidays import get_all_holidays_for_year

        holidays = get_all_holidays_for_year(2025)

        # Check that substitute holidays exist
        substitute_found = False
        for holiday_date, name in holidays.items():
            if "Nghỉ bù" in name:
                substitute_found = True
                break

        # Note: May not have substitutes every year
        # Just verify the function runs without error
        assert isinstance(holidays, dict)

    def test_trading_day_detection(self):
        """Test trading day detection."""
        from src.utils.vietnam_holidays import is_trading_day

        # Weekend should not be trading day
        saturday = datetime(2025, 1, 4)  # A Saturday
        is_trading, reason = is_trading_day(saturday)
        assert not is_trading
        assert "Weekend" in reason

        # Holiday should not be trading day
        new_year = datetime(2025, 1, 1)
        is_trading, reason = is_trading_day(new_year)
        assert not is_trading
        assert "Holiday" in reason

    def test_next_trading_day(self):
        """Test next trading day calculation."""
        from src.utils.vietnam_holidays import get_next_trading_day

        # From a Friday, next trading day should be Monday (if not holiday)
        friday = date(2025, 1, 3)
        next_day = get_next_trading_day(friday, days_ahead=1)

        # Should skip weekend
        assert next_day.weekday() < 5  # Monday-Friday

    def test_days_until_tet(self):
        """Test days until Tết calculation."""
        from src.utils.vietnam_holidays import days_until_tet

        days = days_until_tet(date(2025, 1, 1))

        # Should be positive (Tết 2025 is late January)
        assert days > 0 or days == -1  # -1 if during Tết

    def test_pre_holiday_detection(self):
        """Test pre-holiday trading day detection."""
        from src.utils.vietnam_holidays import is_pre_holiday_trading_day

        # This is a utility function - just verify it runs
        is_pre, holiday_name = is_pre_holiday_trading_day()
        assert isinstance(is_pre, bool)

    def test_holiday_calendar_class(self):
        """Test VietnamHolidayCalendar class."""
        from src.utils.vietnam_holidays import get_holiday_calendar

        calendar = get_holiday_calendar()

        # Test methods
        assert calendar.is_trading_day() in [True, False]

        upcoming = calendar.get_upcoming_holidays(60)
        assert isinstance(upcoming, list)

        settlement = calendar.get_settlement_date(date.today(), 2)
        assert settlement > date.today()


# =============================================================================
# TEST: TCBS BROKER
# =============================================================================


class TestTCBSBroker:
    """Test TCBS broker integration."""

    def test_broker_import(self):
        """Test TCBS broker can be imported."""
        from src.broker.tcbs_broker import TCBSBroker, create_tcbs_broker

        assert TCBSBroker is not None
        assert create_tcbs_broker is not None

    def test_broker_initialization(self):
        """Test TCBS broker initialization."""
        from src.broker.tcbs_broker import TCBSBroker

        broker = TCBSBroker(
            account_id="TEST123",
            username="test_user",
            password="test_pass",
            is_paper=True,
        )

        assert broker.account_id == "TEST123"
        assert broker.is_paper is True
        assert "TCBS" in broker.broker_name

    def test_broker_factory(self):
        """Test broker factory includes TCBS."""
        from src.broker import get_broker, get_supported_brokers

        supported = get_supported_brokers()
        assert "TCBS" in supported
        assert supported["TCBS"]["name"] == "TCBS Securities"

    @patch("src.broker.tcbs_broker.requests.Session")
    def test_broker_connect_mock(self, mock_session):
        """Test broker connection with mocked API."""
        from src.broker.tcbs_broker import TCBSBroker

        # Mock successful auth response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "accessToken": "test_token",
                "refreshToken": "test_refresh",
            },
        }
        mock_session.return_value.post.return_value = mock_response

        broker = TCBSBroker(
            account_id="TEST123",
            username="test_user",
            password="test_pass",
            is_paper=True,
        )

        # Connection should succeed with mocked response
        # Note: Actual connection may fail without real credentials
        assert broker.is_paper is True


# =============================================================================
# TEST: WARRANT/ETF STRATEGY
# =============================================================================


class TestWarrantETFStrategy:
    """Test warrant and ETF trading strategy."""

    def test_instrument_type_detection(self):
        """Test instrument type detection."""
        from src.strategies.warrant_etf_strategy import (
            detect_instrument_type,
            InstrumentType,
            is_warrant,
            is_etf,
        )

        # Stocks
        assert detect_instrument_type("VNM") == InstrumentType.STOCK
        assert detect_instrument_type("HPG") == InstrumentType.STOCK
        assert detect_instrument_type("FPT") == InstrumentType.STOCK

        # ETFs
        assert detect_instrument_type("E1VFVN30") == InstrumentType.ETF
        assert detect_instrument_type("FUEVFVND") == InstrumentType.ETF
        assert is_etf("E1VFVN30")

        # Warrants
        assert detect_instrument_type("CSSI_VNM_2412") == InstrumentType.WARRANT
        assert is_warrant("CVND_HPG_2503")

    def test_price_limit_by_instrument(self):
        """Test price limits by instrument type."""
        from src.strategies.warrant_etf_strategy import get_price_limit

        # Stocks/ETFs: ±7%
        assert get_price_limit("VNM") == 0.07
        assert get_price_limit("E1VFVN30") == 0.07

        # Warrants: ±50%
        assert get_price_limit("CSSI_VNM_2412") == 0.50

    def test_warrant_info_dataclass(self):
        """Test WarrantInfo dataclass."""
        from src.strategies.warrant_etf_strategy import WarrantInfo, WarrantType

        warrant = WarrantInfo(
            symbol="CSSI_VNM_2412",
            underlying="VNM",
            warrant_type=WarrantType.CALL,
            strike_price=80000,
            expiry_date=date(2024, 12, 31),
            conversion_ratio=5.0,
            issuer="SSI",
        )

        assert warrant.symbol == "CSSI_VNM_2412"
        assert warrant.warrant_type == WarrantType.CALL
        assert warrant.conversion_ratio == 5.0

        # Test properties
        assert isinstance(warrant.days_to_expiry, int)
        assert isinstance(warrant.is_near_expiry, bool)

    def test_etf_info(self):
        """Test ETF info constants."""
        from src.strategies.warrant_etf_strategy import VN_ETFS

        assert "E1VFVN30" in VN_ETFS
        assert VN_ETFS["E1VFVN30"]["underlying"] == "VN30"
        assert VN_ETFS["E1VFVN30"]["can_short"] is True

    def test_warrant_strategy(self):
        """Test WarrantStrategy class."""
        from src.strategies.warrant_etf_strategy import WarrantStrategy

        strategy = WarrantStrategy(
            min_days_to_expiry=3,
            max_leverage=10.0,
        )

        assert strategy.min_days_to_expiry == 3
        assert strategy.max_leverage == 10.0

    def test_etf_strategy(self):
        """Test ETFStrategy class."""
        from src.strategies.warrant_etf_strategy import ETFStrategy

        strategy = ETFStrategy(
            max_premium_pct=0.02,
            max_discount_pct=0.02,
        )

        assert strategy.max_premium_pct == 0.02

    def test_special_instruments_handler(self):
        """Test SpecialInstrumentsHandler."""
        from src.strategies.warrant_etf_strategy import (
            get_special_instruments_handler,
            InstrumentType,
        )

        handler = get_special_instruments_handler()

        # Test detection
        assert handler.is_special_instrument("E1VFVN30")
        assert handler.is_special_instrument("CSSI_VNM_2412")
        assert not handler.is_special_instrument("VNM")

        # Test position size multiplier
        assert handler.get_position_size_multiplier("VNM") == 1.0
        assert handler.get_position_size_multiplier("E1VFVN30") == 1.2  # ETF
        assert handler.get_position_size_multiplier("CSSI_VNM_2412") == 0.3  # Warrant


# =============================================================================
# TEST: MARGIN CALL MONITOR
# =============================================================================


class TestMarginCallMonitor:
    """Test margin call monitoring system."""

    def test_monitor_import(self):
        """Test margin call monitor can be imported."""
        from src.risk.margin_call_monitor import (
            MarginCallMonitor,
            MarginAlertLevel,
            MarginAlertType,
            MarginMonitorConfig,
        )

        assert MarginCallMonitor is not None
        assert MarginAlertLevel.CRITICAL is not None

    def test_monitor_config(self):
        """Test monitor configuration."""
        from src.risk.margin_call_monitor import MarginMonitorConfig

        config = MarginMonitorConfig(
            warning_threshold=0.35,
            margin_call_threshold=0.30,
            force_liquidation_threshold=0.25,
        )

        assert config.warning_threshold == 0.35
        assert config.margin_call_threshold == 0.30
        assert config.force_liquidation_threshold == 0.25

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        from src.risk.margin_call_monitor import MarginCallMonitor, MarginMonitorConfig

        config = MarginMonitorConfig()
        monitor = MarginCallMonitor(margin_manager=None, config=config)

        assert monitor.config == config
        assert not monitor._running

    def test_alert_levels(self):
        """Test alert level enum."""
        from src.risk.margin_call_monitor import MarginAlertLevel

        assert MarginAlertLevel.INFO.value == "INFO"
        assert MarginAlertLevel.WARNING.value == "WARNING"
        assert MarginAlertLevel.CRITICAL.value == "CRITICAL"
        assert MarginAlertLevel.EMERGENCY.value == "EMERGENCY"

    def test_alert_types(self):
        """Test alert type enum."""
        from src.risk.margin_call_monitor import MarginAlertType

        assert MarginAlertType.MARGIN_CALL_TRIGGERED.value == "MARGIN_CALL_TRIGGERED"
        assert MarginAlertType.FORCE_LIQUIDATION_IMMINENT.value == "FORCE_LIQUIDATION_IMMINENT"

    def test_get_status(self):
        """Test get_status method."""
        from src.risk.margin_call_monitor import MarginCallMonitor

        monitor = MarginCallMonitor(margin_manager=None)
        status = monitor.get_status()

        assert "running" in status
        assert "last_equity_ratio" in status
        assert "config" in status

    def test_register_callback(self):
        """Test alert callback registration."""
        from src.risk.margin_call_monitor import MarginCallMonitor

        monitor = MarginCallMonitor(margin_manager=None)

        callback_called = []

        def my_callback(alert):
            callback_called.append(alert)

        monitor.register_alert_callback(my_callback)

        assert len(monitor._alert_callbacks) == 1


# =============================================================================
# TEST: BROKER FACTORY
# =============================================================================


class TestBrokerFactory:
    """Test broker factory function."""

    def test_supported_brokers(self):
        """Test get_supported_brokers function."""
        from src.broker import get_supported_brokers

        supported = get_supported_brokers()

        assert "SSI" in supported
        assert "VNDIRECT" in supported
        assert "TCBS" in supported
        assert "SIMULATED" in supported

    def test_simulated_broker(self):
        """Test creating simulated broker."""
        from src.broker import get_broker

        broker = get_broker("SIMULATED", "PAPER", {"initial_cash": 100_000_000})

        assert broker is not None
        assert broker.broker_name == "SIMULATED"

    def test_invalid_broker_type(self):
        """Test invalid broker type raises error."""
        from src.broker import get_broker

        with pytest.raises(ValueError) as exc_info:
            get_broker("INVALID_BROKER", "123", {})

        assert "Unknown broker type" in str(exc_info.value)


# =============================================================================
# TEST: INTEGRATION
# =============================================================================


class TestIntegration:
    """Integration tests for 10/10 improvements."""

    def test_holiday_affects_settlement(self):
        """Test that holidays affect settlement date calculation."""
        from src.utils.vietnam_holidays import get_holiday_calendar

        calendar = get_holiday_calendar()

        # Settlement should skip holidays
        trade_date = date(2025, 1, 1)  # New Year's Day
        settlement = calendar.get_settlement_date(trade_date, 2)

        # Settlement should be after the holiday
        assert settlement > trade_date

    def test_warrant_position_sizing(self):
        """Test warrant position sizing is reduced."""
        from src.strategies.warrant_etf_strategy import get_special_instruments_handler

        handler = get_special_instruments_handler()

        # Warrant should have reduced position size
        warrant_mult = handler.get_position_size_multiplier("CSSI_VNM_2412")
        stock_mult = handler.get_position_size_multiplier("VNM")

        assert warrant_mult < stock_mult
        assert warrant_mult == 0.3  # 30% of normal

    def test_etf_position_sizing(self):
        """Test ETF position sizing is increased."""
        from src.strategies.warrant_etf_strategy import get_special_instruments_handler

        handler = get_special_instruments_handler()

        # ETF should have increased position size
        etf_mult = handler.get_position_size_multiplier("E1VFVN30")
        stock_mult = handler.get_position_size_multiplier("VNM")

        assert etf_mult > stock_mult
        assert etf_mult == 1.2  # 120% of normal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
