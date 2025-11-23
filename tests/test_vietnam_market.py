"""
Unit tests for src/utils/vietnam_market.py
Tests VietnamMarketValidator for Vietnam market-specific rules
"""

from datetime import datetime, time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from src.utils.vietnam_market import (
    VietnamMarketValidator,
    calculate_t2_requirement,
    check_liquidity,
    check_price_limits,
    check_trading_session,
    validate_position_vs_volume,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def validator():
    """Basic validator with default settings"""
    return VietnamMarketValidator()


@pytest.fixture
def custom_validator():
    """Validator with custom settings"""
    return VietnamMarketValidator(config=None)  # Uses defaults


@pytest.fixture
def mock_config():
    """Mock config object with Vietnam market settings"""
    config = MagicMock()
    config.vn_daily_price_limit_pct = 7.0
    config.vn_check_price_limits = True
    config.vn_avoid_floor_ceiling_pct = 2.0
    config.vn_settlement_days = 2
    config.vn_reserve_t2_cash = True
    config.vn_t2_cash_buffer_pct = 0.10
    config.vn_min_daily_value = 2_000_000_000
    config.vn_max_position_pct_of_volume = 0.05
    config.vn_avoid_session_boundaries = True
    config.vn_session_boundary_minutes = 5
    config.vn_trading_session_am_end = "11:30"
    config.vn_trading_session_pm_start = "13:00"
    return config


@pytest.fixture
def liquid_stock_data():
    """DataFrame with sufficient liquidity"""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    # 250k shares * 10k VND = 2.5B VND daily value (above 2B threshold)
    return pd.DataFrame(
        {
            "open": np.random.uniform(9500, 10500, 30),
            "high": np.random.uniform(10000, 11000, 30),
            "low": np.random.uniform(9000, 10000, 30),
            "close": np.random.uniform(9500, 10500, 30),
            "volume": np.random.uniform(250_000, 300_000, 30),
        },
        index=dates,
    )


@pytest.fixture
def illiquid_stock_data():
    """DataFrame with insufficient liquidity"""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    # 50k shares * 10k VND = 500M VND daily value (below 2B threshold)
    return pd.DataFrame(
        {
            "open": np.random.uniform(9500, 10500, 30),
            "high": np.random.uniform(10000, 11000, 30),
            "low": np.random.uniform(9000, 10000, 30),
            "close": np.random.uniform(9500, 10500, 30),
            "volume": np.random.uniform(40_000, 60_000, 30),
        },
        index=dates,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


def test_initialization_default():
    """Test initialization with default settings"""
    validator = VietnamMarketValidator()
    assert validator.daily_price_limit_pct == 7.0
    assert validator.check_price_limits is True
    assert validator.avoid_floor_ceiling_pct == 2.0
    assert validator.settlement_days == 2
    assert validator.reserve_t2_cash is True
    assert validator.t2_cash_buffer_pct == 0.10
    assert validator.min_daily_value == 2_000_000_000
    assert validator.max_position_pct_of_volume == 0.05
    assert validator.avoid_session_boundaries is True
    assert validator.session_boundary_minutes == 5
    assert validator.session_am_end == "11:30"
    assert validator.session_pm_start == "13:00"


def test_initialization_with_config(mock_config):
    """Test initialization with config object"""
    validator = VietnamMarketValidator(config=mock_config)
    assert validator.daily_price_limit_pct == 7.0
    assert validator.check_price_limits is True
    assert validator.avoid_floor_ceiling_pct == 2.0
    assert validator.settlement_days == 2


# =============================================================================
# PRICE FLOOR/CEILING TESTS
# =============================================================================


def test_price_floor_ceiling_safe_price(validator):
    """Test price check when price is safe (middle range)"""
    # Reference: 100,000 VND
    # Floor: 93,000 (-7%)
    # Ceiling: 107,000 (+7%)
    # Current: 100,000 (safe)
    is_safe, warning = validator.check_price_floor_ceiling(
        current_price=100_000, reference_price=100_000, symbol="VNM"
    )
    assert is_safe is True
    assert warning is None


def test_price_near_floor(validator):
    """Test price check when price is near floor"""
    # Reference: 100,000
    # Floor: 93,000
    # Current: 93,500 (within 2% of floor)
    is_safe, warning = validator.check_price_floor_ceiling(
        current_price=93_500, reference_price=100_000, symbol="VNM"
    )
    assert is_safe is False
    assert "FLOOR" in warning
    assert "Avoid entry" in warning


def test_price_near_ceiling(validator):
    """Test price check when price is near ceiling"""
    # Reference: 100,000
    # Ceiling: 107,000
    # Current: 106,500 (within 2% of ceiling)
    is_safe, warning = validator.check_price_floor_ceiling(
        current_price=106_500, reference_price=100_000, symbol="VNM"
    )
    assert is_safe is False
    assert "CEILING" in warning
    assert "Avoid entry" in warning


def test_price_at_floor(validator):
    """Test price check when price is exactly at floor"""
    # Reference: 100,000
    # Floor: 93,000
    # Current: 93,000 (at floor)
    is_safe, warning = validator.check_price_floor_ceiling(
        current_price=93_000, reference_price=100_000, symbol="VNM"
    )
    assert is_safe is False
    assert "FLOOR" in warning


def test_price_at_ceiling(validator):
    """Test price check when price is exactly at ceiling"""
    # Reference: 100,000
    # Ceiling: 107,000
    # Current: 107,000 (at ceiling)
    is_safe, warning = validator.check_price_floor_ceiling(
        current_price=107_000, reference_price=100_000, symbol="VNM"
    )
    assert is_safe is False
    assert "CEILING" in warning


def test_price_check_disabled():
    """Test price check when checking is disabled"""
    validator = VietnamMarketValidator()
    validator.check_price_limits = False
    # Even at floor, should return safe
    is_safe, warning = validator.check_price_floor_ceiling(
        current_price=93_000, reference_price=100_000, symbol="VNM"
    )
    assert is_safe is True
    assert warning is None


def test_price_check_invalid_reference_price(validator):
    """Test price check with invalid reference price (zero/negative)"""
    # Zero reference
    is_safe, warning = validator.check_price_floor_ceiling(
        current_price=100_000, reference_price=0, symbol="VNM"
    )
    assert is_safe is True  # Returns safe when invalid
    assert warning is None

    # Negative reference
    is_safe, warning = validator.check_price_floor_ceiling(
        current_price=100_000, reference_price=-50_000, symbol="VNM"
    )
    assert is_safe is True
    assert warning is None


def test_price_check_custom_avoid_percentage():
    """Test price check with custom avoid percentage"""
    validator = VietnamMarketValidator()
    validator.avoid_floor_ceiling_pct = 5.0  # 5% buffer instead of 2%

    # Reference: 100,000
    # Floor: 93,000
    # Current: 95,000 (2.15% from floor - safe with 2%, unsafe with 5%)
    is_safe, warning = validator.check_price_floor_ceiling(
        current_price=95_000, reference_price=100_000, symbol="VNM"
    )
    assert is_safe is False  # With 5% buffer, this is too close


# =============================================================================
# TRADING SESSION TIMING TESTS
# =============================================================================


def test_trading_session_safe_morning(validator):
    """Test trading session check during safe morning time"""
    # 10:30 AM - safe (not near 11:30 end)
    safe_time = datetime(2024, 1, 15, 10, 30, 0)
    is_safe, warning = validator.check_trading_session_timing(safe_time)
    assert is_safe is True
    assert warning is None


def test_trading_session_safe_afternoon(validator):
    """Test trading session check during safe afternoon time"""
    # 14:00 - safe (not near boundaries)
    safe_time = datetime(2024, 1, 15, 14, 0, 0)
    is_safe, warning = validator.check_trading_session_timing(safe_time)
    assert is_safe is True
    assert warning is None


def test_trading_session_near_am_end(validator):
    """Test trading session check near morning session end"""
    # 11:27 - within 5 minutes of 11:30 end
    boundary_time = datetime(2024, 1, 15, 11, 27, 0)
    is_safe, warning = validator.check_trading_session_timing(boundary_time)
    assert is_safe is False
    assert "morning session end" in warning


def test_trading_session_near_pm_start(validator):
    """Test trading session check near afternoon session start"""
    # 13:03 - within 5 minutes of 13:00 start
    boundary_time = datetime(2024, 1, 15, 13, 3, 0)
    is_safe, warning = validator.check_trading_session_timing(boundary_time)
    assert is_safe is False
    assert "afternoon session start" in warning


def test_trading_session_check_disabled():
    """Test trading session check when disabled"""
    validator = VietnamMarketValidator()
    validator.avoid_session_boundaries = False
    # Even at boundary, should return safe
    boundary_time = datetime(2024, 1, 15, 11, 29, 0)
    is_safe, warning = validator.check_trading_session_timing(boundary_time)
    assert is_safe is True
    assert warning is None


def test_trading_session_default_current_time(validator):
    """Test trading session check with default current time (now)"""
    # Should use datetime.now() if no time provided
    is_safe, warning = validator.check_trading_session_timing()
    # Result depends on actual current time, just verify it returns a tuple
    assert isinstance(is_safe, bool)
    assert warning is None or isinstance(warning, str)


def test_trading_session_custom_boundary_minutes():
    """Test trading session with custom boundary minutes"""
    validator = VietnamMarketValidator()
    validator.session_boundary_minutes = 10  # 10 minutes instead of 5

    # 11:22 - would be safe with 5 min, unsafe with 10 min
    boundary_time = datetime(2024, 1, 15, 11, 22, 0)
    is_safe, warning = validator.check_trading_session_timing(boundary_time)
    assert is_safe is False  # Within 10 minutes of 11:30


# =============================================================================
# T+2 CASH REQUIREMENT TESTS
# =============================================================================


def test_t2_cash_no_pending_settlements(validator):
    """Test T+2 calculation with no pending settlements"""
    total, buffer = validator.calculate_t2_cash_requirement(
        pending_settlements={}, new_trade_value=10_000_000
    )
    assert total == 10_000_000
    assert buffer == 1_000_000  # 10% of 10M


def test_t2_cash_with_pending_settlements(validator):
    """Test T+2 calculation with pending settlements"""
    pending = {"2024-01-17": 5_000_000, "2024-01-18": 3_000_000}
    total, buffer = validator.calculate_t2_cash_requirement(
        pending_settlements=pending, new_trade_value=2_000_000
    )
    assert total == 10_000_000  # 5M + 3M + 2M
    assert buffer == 1_000_000  # 10%


def test_t2_cash_only_pending_no_new_trade(validator):
    """Test T+2 calculation with only pending settlements"""
    pending = {"2024-01-17": 7_500_000}
    total, buffer = validator.calculate_t2_cash_requirement(
        pending_settlements=pending, new_trade_value=0
    )
    assert total == 7_500_000
    assert buffer == 750_000


def test_t2_cash_disabled():
    """Test T+2 calculation when T+2 reservation is disabled"""
    validator = VietnamMarketValidator()
    validator.reserve_t2_cash = False
    total, buffer = validator.calculate_t2_cash_requirement(
        pending_settlements={"2024-01-17": 5_000_000}, new_trade_value=2_000_000
    )
    assert total == 0
    assert buffer == 0


def test_t2_cash_custom_buffer_percentage():
    """Test T+2 calculation with custom buffer percentage"""
    validator = VietnamMarketValidator()
    validator.t2_cash_buffer_pct = 0.20  # 20% instead of 10%
    total, buffer = validator.calculate_t2_cash_requirement(
        pending_settlements={}, new_trade_value=10_000_000
    )
    assert total == 10_000_000
    assert buffer == 2_000_000  # 20% of 10M


# =============================================================================
# POSITION SIZE VS VOLUME TESTS
# =============================================================================


def test_position_size_vs_volume_safe(validator):
    """Test position size validation when size is safe"""
    # 10,000 shares vs 500,000 avg volume = 2% (safe, under 5% limit)
    is_safe, warning = validator.validate_position_size_vs_volume(
        position_shares=10_000, avg_daily_volume=500_000, symbol="VNM"
    )
    assert is_safe is True
    assert warning is None


def test_position_size_vs_volume_too_large(validator):
    """Test position size validation when size is too large"""
    # 30,000 shares vs 500,000 avg volume = 6% (unsafe, exceeds 5% limit)
    is_safe, warning = validator.validate_position_size_vs_volume(
        position_shares=30_000, avg_daily_volume=500_000, symbol="VNM"
    )
    assert is_safe is False
    assert "Position too large" in warning
    assert "slippage" in warning.lower()


def test_position_size_vs_volume_at_limit(validator):
    """Test position size validation when exactly at limit"""
    # 25,000 shares vs 500,000 avg volume = 5% (at limit)
    is_safe, warning = validator.validate_position_size_vs_volume(
        position_shares=25_000, avg_daily_volume=500_000, symbol="VNM"
    )
    assert is_safe is True  # Exactly at limit should be safe


def test_position_size_vs_volume_invalid_volume(validator):
    """Test position size validation with invalid volume (zero/negative)"""
    # Zero volume
    is_safe, warning = validator.validate_position_size_vs_volume(
        position_shares=10_000, avg_daily_volume=0, symbol="VNM"
    )
    assert is_safe is True  # Returns safe when invalid
    assert warning is None

    # Negative volume
    is_safe, warning = validator.validate_position_size_vs_volume(
        position_shares=10_000, avg_daily_volume=-100_000, symbol="VNM"
    )
    assert is_safe is True
    assert warning is None


def test_position_size_vs_volume_custom_limit():
    """Test position size validation with custom limit"""
    validator = VietnamMarketValidator()
    validator.max_position_pct_of_volume = 0.10  # 10% instead of 5%

    # 50,000 shares vs 500,000 volume = 10% (safe with 10%, unsafe with 5%)
    is_safe, warning = validator.validate_position_size_vs_volume(
        position_shares=50_000, avg_daily_volume=500_000, symbol="VNM"
    )
    assert is_safe is True  # Safe with 10% limit


# =============================================================================
# LIQUIDITY REQUIREMENTS TESTS
# =============================================================================


def test_liquidity_sufficient(validator, liquid_stock_data):
    """Test liquidity check with sufficient liquidity"""
    is_liquid, warning = validator.check_liquidity_requirements(liquid_stock_data, symbol="VNM")
    assert is_liquid is True
    assert warning is None


def test_liquidity_insufficient(validator, illiquid_stock_data):
    """Test liquidity check with insufficient liquidity"""
    is_liquid, warning = validator.check_liquidity_requirements(illiquid_stock_data, symbol="VNM")
    assert is_liquid is False
    assert "Insufficient liquidity" in warning
    assert "2.00B VND" in warning


def test_liquidity_insufficient_data(validator):
    """Test liquidity check with insufficient data points"""
    # Only 10 days of data (need 20)
    df = pd.DataFrame(
        {
            "close": np.random.uniform(10000, 11000, 10),
            "volume": np.random.uniform(250_000, 300_000, 10),
        }
    )
    is_liquid, warning = validator.check_liquidity_requirements(df, symbol="VNM")
    assert is_liquid is False
    assert "Insufficient data" in warning


def test_liquidity_empty_dataframe(validator):
    """Test liquidity check with empty DataFrame"""
    df = pd.DataFrame()
    is_liquid, warning = validator.check_liquidity_requirements(df, symbol="VNM")
    assert is_liquid is False
    assert "Insufficient data" in warning


def test_liquidity_custom_min_value():
    """Test liquidity check with custom minimum value"""
    validator = VietnamMarketValidator()
    validator.min_daily_value = 5_000_000_000  # 5B VND instead of 2B

    # Stock with 2.5B VND daily value (safe with 2B, unsafe with 5B)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    df = pd.DataFrame(
        {
            "close": np.ones(30) * 10_000,
            "volume": np.ones(30) * 250_000,  # 250k * 10k = 2.5B
        },
        index=dates,
    )

    is_liquid, warning = validator.check_liquidity_requirements(df, symbol="VNM")
    assert is_liquid is False  # Not enough for 5B requirement


def test_liquidity_exactly_at_minimum(validator):
    """Test liquidity check when exactly at minimum threshold"""
    # Exactly 2B VND daily value
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    df = pd.DataFrame(
        {
            "close": np.ones(30) * 10_000,
            "volume": np.ones(30) * 200_000,  # 200k * 10k = 2B
        },
        index=dates,
    )

    is_liquid, warning = validator.check_liquidity_requirements(df, symbol="VNM")
    # Exactly at minimum should pass
    assert is_liquid is True


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


@patch("src.utils.vietnam_market.get_vietnam_market_validator")
def test_check_price_limits_convenience(mock_get_validator):
    """Test check_price_limits convenience function"""
    mock_validator = MagicMock()
    mock_validator.check_price_floor_ceiling.return_value = (True, None)
    mock_get_validator.return_value = mock_validator

    is_safe, warning = check_price_limits(100_000, 95_000, "VNM")
    assert is_safe is True
    mock_validator.check_price_floor_ceiling.assert_called_once_with(100_000, 95_000, "VNM")


@patch("src.utils.vietnam_market.get_vietnam_market_validator")
def test_check_trading_session_convenience(mock_get_validator):
    """Test check_trading_session convenience function"""
    mock_validator = MagicMock()
    mock_validator.check_trading_session_timing.return_value = (True, None)
    mock_get_validator.return_value = mock_validator

    test_time = datetime(2024, 1, 15, 10, 30, 0)
    is_safe, warning = check_trading_session(test_time)
    assert is_safe is True
    mock_validator.check_trading_session_timing.assert_called_once_with(test_time)


@patch("src.utils.vietnam_market.get_vietnam_market_validator")
def test_calculate_t2_requirement_convenience(mock_get_validator):
    """Test calculate_t2_requirement convenience function"""
    mock_validator = MagicMock()
    mock_validator.calculate_t2_cash_requirement.return_value = (10_000_000, 1_000_000)
    mock_get_validator.return_value = mock_validator

    pending = {"2024-01-17": 5_000_000}
    total, buffer = calculate_t2_requirement(pending, 5_000_000)
    assert total == 10_000_000
    assert buffer == 1_000_000
    mock_validator.calculate_t2_cash_requirement.assert_called_once_with(pending, 5_000_000)


@patch("src.utils.vietnam_market.get_vietnam_market_validator")
def test_validate_position_vs_volume_convenience(mock_get_validator):
    """Test validate_position_vs_volume convenience function"""
    mock_validator = MagicMock()
    mock_validator.validate_position_size_vs_volume.return_value = (True, None)
    mock_get_validator.return_value = mock_validator

    is_safe, warning = validate_position_vs_volume(10_000, 500_000, "VNM")
    assert is_safe is True
    mock_validator.validate_position_size_vs_volume.assert_called_once_with(10_000, 500_000, "VNM")


@patch("src.utils.vietnam_market.get_vietnam_market_validator")
def test_check_liquidity_convenience(mock_get_validator):
    """Test check_liquidity convenience function"""
    mock_validator = MagicMock()
    mock_validator.check_liquidity_requirements.return_value = (True, None)
    mock_get_validator.return_value = mock_validator

    df = pd.DataFrame({"close": [100], "volume": [1000]})
    is_liquid, warning = check_liquidity(df, "VNM")
    assert is_liquid is True
    mock_validator.check_liquidity_requirements.assert_called_once()


# =============================================================================
# EDGE CASES AND INTEGRATION TESTS
# =============================================================================


def test_all_checks_pass_integration(validator, liquid_stock_data):
    """Integration test: all checks pass"""
    # Price check
    is_safe, _ = validator.check_price_floor_ceiling(100_000, 100_000, "VNM")
    assert is_safe is True

    # Session check (safe time)
    safe_time = datetime(2024, 1, 15, 10, 30, 0)
    is_safe, _ = validator.check_trading_session_timing(safe_time)
    assert is_safe is True

    # Position vs volume
    is_safe, _ = validator.validate_position_size_vs_volume(10_000, 500_000, "VNM")
    assert is_safe is True

    # Liquidity
    is_liquid, _ = validator.check_liquidity_requirements(liquid_stock_data, "VNM")
    assert is_liquid is True


def test_all_checks_fail_integration(validator, illiquid_stock_data):
    """Integration test: all checks fail"""
    # Price near floor
    is_safe, warning = validator.check_price_floor_ceiling(93_500, 100_000, "VNM")
    assert is_safe is False
    assert warning is not None

    # Near session boundary
    boundary_time = datetime(2024, 1, 15, 11, 29, 0)
    is_safe, warning = validator.check_trading_session_timing(boundary_time)
    assert is_safe is False
    assert warning is not None

    # Position too large
    is_safe, warning = validator.validate_position_size_vs_volume(50_000, 500_000, "VNM")
    assert is_safe is False
    assert warning is not None

    # Insufficient liquidity
    is_liquid, warning = validator.check_liquidity_requirements(illiquid_stock_data, "VNM")
    assert is_liquid is False
    assert warning is not None


def test_validator_with_none_symbol(validator):
    """Test validators handle None symbol gracefully"""
    # Price check
    is_safe, _ = validator.check_price_floor_ceiling(100_000, 100_000, None)
    assert isinstance(is_safe, bool)

    # Position vs volume
    is_safe, _ = validator.validate_position_size_vs_volume(10_000, 500_000, None)
    assert isinstance(is_safe, bool)


def test_multiple_t2_settlements_complex(validator):
    """Test T+2 calculation with multiple pending settlements"""
    pending = {
        "2024-01-15": 3_000_000,
        "2024-01-16": 2_500_000,
        "2024-01-17": 4_500_000,
        "2024-01-18": 1_000_000,
    }
    total, buffer = validator.calculate_t2_cash_requirement(pending, 5_000_000)
    assert total == 16_000_000  # 3M + 2.5M + 4.5M + 1M + 5M
    assert buffer == 1_600_000  # 10%


def test_price_check_various_reference_prices(validator):
    """Test price check with various reference price levels"""
    # Low price stock
    is_safe, _ = validator.check_price_floor_ceiling(5_000, 5_000, "PENNY")
    assert is_safe is True

    # High price stock
    is_safe, _ = validator.check_price_floor_ceiling(500_000, 500_000, "EXPENSIVE")
    assert is_safe is True

    # Mid price stock
    is_safe, _ = validator.check_price_floor_ceiling(50_000, 50_000, "MID")
    assert is_safe is True
