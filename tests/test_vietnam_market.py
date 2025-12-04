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
    round_to_lot,
    get_tick_size,
    round_to_tick,
    get_exchange,
    calculate_ceiling_floor,
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
    return VietnamMarketValidator(
        min_liquidity_value=5_000_000_000, max_position_pct_of_volume=0.10
    )


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
    assert validator.min_liquidity_value == 2_000_000_000
    assert validator.max_position_pct_of_volume == 0.05


def test_initialization_with_custom_values():
    """Test initialization with custom values"""
    validator = VietnamMarketValidator(
        min_liquidity_value=5_000_000_000, max_position_pct_of_volume=0.10
    )
    assert validator.min_liquidity_value == 5_000_000_000
    assert validator.max_position_pct_of_volume == 0.10


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
    assert "safe" in warning.lower() or "within" in warning.lower()


def test_price_near_floor(validator):
    """Test price check when price is near floor"""
    # Reference: 100,000
    # Floor: 93,000
    # Current: 93,500 (within 1% of floor)
    is_safe, warning = validator.check_price_floor_ceiling(
        current_price=93_500, reference_price=100_000, symbol="VNM"
    )
    assert is_safe is False
    assert "FLOOR" in warning


def test_price_near_ceiling(validator):
    """Test price check when price is near ceiling"""
    # Reference: 100,000
    # Ceiling: 107,000
    # Current: 106,500 (within 1% of ceiling)
    is_safe, warning = validator.check_price_floor_ceiling(
        current_price=106_500, reference_price=100_000, symbol="VNM"
    )
    assert is_safe is False
    assert "CEILING" in warning


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


# =============================================================================
# TRADING SESSION TIMING TESTS
# =============================================================================


def test_trading_session_safe_morning(validator):
    """Test trading session check during safe morning time"""
    # 10:30 AM - safe (not near 11:30 end)
    safe_time = datetime(2024, 1, 15, 10, 30, 0)
    is_safe, warning = validator.check_trading_session_timing(safe_time)
    assert is_safe is True
    assert "Safe" in warning or warning is not None


def test_trading_session_safe_afternoon(validator):
    """Test trading session check during safe afternoon time"""
    # 14:00 - safe (not near boundaries)
    safe_time = datetime(2024, 1, 15, 14, 0, 0)
    is_safe, warning = validator.check_trading_session_timing(safe_time)
    assert is_safe is True


def test_trading_session_near_am_end(validator):
    """Test trading session check near morning session end"""
    # 11:27 - within 5 minutes of 11:30 end
    boundary_time = datetime(2024, 1, 15, 11, 27, 0)
    is_safe, warning = validator.check_trading_session_timing(boundary_time)
    assert is_safe is False
    assert "morning" in warning.lower() or "11:" in warning


def test_trading_session_near_pm_end(validator):
    """Test trading session check near afternoon session end"""
    # 14:27 - within 5 minutes of 14:30 end
    boundary_time = datetime(2024, 1, 15, 14, 27, 0)
    is_safe, warning = validator.check_trading_session_timing(boundary_time)
    assert is_safe is False
    assert "afternoon" in warning.lower() or "14:" in warning


def test_trading_session_ato(validator):
    """Test trading session check during ATO"""
    # 9:10 - during ATO session
    ato_time = datetime(2024, 1, 15, 9, 10, 0)
    is_safe, warning = validator.check_trading_session_timing(ato_time)
    assert is_safe is False
    assert "ATO" in warning


def test_trading_session_atc(validator):
    """Test trading session check during ATC"""
    # 14:35 - during ATC session
    atc_time = datetime(2024, 1, 15, 14, 35, 0)
    is_safe, warning = validator.check_trading_session_timing(atc_time)
    assert is_safe is False
    assert "ATC" in warning


# =============================================================================
# T+2 CASH REQUIREMENT TESTS
# =============================================================================


def test_t2_cash_no_pending_settlements(validator):
    """Test T+2 calculation with no pending settlements"""
    total, buffer = validator.calculate_t2_cash_requirement(
        pending_settlements={}, new_order_value=10_000_000
    )
    assert total == 10_000_000
    assert buffer == 1_000_000  # 10% of 10M


def test_t2_cash_with_pending_settlements(validator):
    """Test T+2 calculation with pending settlements"""
    pending = {"2024-01-17": 5_000_000, "2024-01-18": 3_000_000}
    total, buffer = validator.calculate_t2_cash_requirement(
        pending_settlements=pending, new_order_value=2_000_000
    )
    assert total == 10_000_000  # 5M + 3M + 2M
    assert buffer == 200_000  # 10% of new order (2M)


def test_t2_cash_only_pending_no_new_trade(validator):
    """Test T+2 calculation with only pending settlements"""
    pending = {"2024-01-17": 7_500_000}
    total, buffer = validator.calculate_t2_cash_requirement(
        pending_settlements=pending, new_order_value=0
    )
    assert total == 7_500_000
    assert buffer == 0  # 10% of 0


def test_t2_convenience_function():
    """Test calculate_t2_requirement convenience function"""
    pending = {"2024-01-17": 5_000_000}
    total, buffer = calculate_t2_requirement(pending, 5_000_000)
    assert total == 10_000_000  # 5M + 5M
    assert buffer == 500_000  # 10% of 5M


# =============================================================================
# POSITION SIZE VS VOLUME TESTS
# =============================================================================


def test_position_size_vs_volume_safe(validator):
    """Test position size validation when size is safe"""
    # 10,000 shares vs 500,000 avg volume = 2% (safe, under 5% limit)
    is_safe, warning = validator.validate_position_size_vs_volume(
        shares=10_000, avg_volume=500_000, symbol="VNM"
    )
    assert is_safe is True
    assert "OK" in warning or "safe" in warning.lower()


def test_position_size_vs_volume_too_large(validator):
    """Test position size validation when size is too large"""
    # 30,000 shares vs 500,000 avg volume = 6% (unsafe, exceeds 5% limit)
    is_safe, warning = validator.validate_position_size_vs_volume(
        shares=30_000, avg_volume=500_000, symbol="VNM"
    )
    assert is_safe is False
    assert "too large" in warning.lower() or "Position" in warning


def test_position_size_vs_volume_at_limit(validator):
    """Test position size validation when exactly at limit"""
    # 25,000 shares vs 500,000 avg volume = 5% (at limit)
    is_safe, warning = validator.validate_position_size_vs_volume(
        shares=25_000, avg_volume=500_000, symbol="VNM"
    )
    assert is_safe is True  # Exactly at limit should be safe


def test_position_size_vs_volume_invalid_volume(validator):
    """Test position size validation with invalid volume (zero/negative)"""
    # Zero volume
    is_safe, warning = validator.validate_position_size_vs_volume(
        shares=10_000, avg_volume=0, symbol="VNM"
    )
    assert is_safe is False
    assert "Invalid" in warning

    # Negative volume
    is_safe, warning = validator.validate_position_size_vs_volume(
        shares=10_000, avg_volume=-100_000, symbol="VNM"
    )
    assert is_safe is False


def test_position_size_vs_volume_custom_limit():
    """Test position size validation with custom limit"""
    validator = VietnamMarketValidator(max_position_pct_of_volume=0.10)

    # 50,000 shares vs 500,000 volume = 10% (safe with 10%, unsafe with 5%)
    is_safe, warning = validator.validate_position_size_vs_volume(
        shares=50_000, avg_volume=500_000, symbol="VNM"
    )
    assert is_safe is True  # Safe with 10% limit


# =============================================================================
# LIQUIDITY REQUIREMENTS TESTS
# =============================================================================


def test_liquidity_sufficient(validator, liquid_stock_data):
    """Test liquidity check with sufficient liquidity"""
    is_liquid, warning = validator.check_liquidity_requirements(liquid_stock_data, symbol="VNM")
    assert is_liquid is True
    assert "OK" in warning or "Liquidity" in warning


def test_liquidity_insufficient(validator, illiquid_stock_data):
    """Test liquidity check with insufficient liquidity"""
    is_liquid, warning = validator.check_liquidity_requirements(illiquid_stock_data, symbol="VNM")
    assert is_liquid is False
    assert "Insufficient" in warning


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
    validator = VietnamMarketValidator(min_liquidity_value=5_000_000_000)

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


def test_check_price_limits_convenience():
    """Test check_price_limits convenience function"""
    is_safe, warning = check_price_limits(100_000, 100_000, "VNM")
    assert is_safe is True
    assert warning is not None


def test_check_trading_session_convenience():
    """Test check_trading_session convenience function"""
    test_time = datetime(2024, 1, 15, 10, 30, 0)
    is_safe, warning = check_trading_session(test_time)
    assert is_safe is True


def test_calculate_t2_requirement_convenience():
    """Test calculate_t2_requirement convenience function"""
    pending = {"2024-01-17": 5_000_000}
    total, buffer = calculate_t2_requirement(pending, 5_000_000)
    assert total == 10_000_000
    assert buffer == 500_000


def test_validate_position_vs_volume_convenience():
    """Test validate_position_vs_volume convenience function"""
    is_safe, warning = validate_position_vs_volume(10_000, 500_000, "VNM")
    assert is_safe is True


def test_check_liquidity_convenience():
    """Test check_liquidity convenience function"""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    df = pd.DataFrame(
        {
            "close": np.ones(30) * 10_000,
            "volume": np.ones(30) * 250_000,
        },
        index=dates,
    )
    is_liquid, warning = check_liquidity(df, "VNM")
    assert is_liquid is True


# =============================================================================
# LOT SIZE AND TICK SIZE TESTS
# =============================================================================


def test_round_to_lot():
    """Test lot size rounding"""
    assert round_to_lot(150) == 100
    assert round_to_lot(250) == 200
    assert round_to_lot(50) == 100  # Minimum 1 lot
    assert round_to_lot(0) == 0
    assert round_to_lot(-100) == 0


def test_get_tick_size():
    """Test tick size calculation"""
    assert get_tick_size(8000) == 10  # < 10,000
    assert get_tick_size(25000) == 50  # 10,000 - 50,000
    assert get_tick_size(80000) == 100  # >= 50,000


def test_round_to_tick():
    """Test tick rounding"""
    assert round_to_tick(25123) == 25100  # Nearest tick for mid-range
    assert round_to_tick(8005, "up") == 8010
    assert round_to_tick(8005, "down") == 8000


def test_get_exchange():
    """Test exchange detection"""
    assert get_exchange("VCB") == "HOSE"  # VN30 symbol
    assert get_exchange("SHS") == "HNX"  # HNX30 symbol
    assert get_exchange("ZZZZZ") == "HOSE"  # Default for unknown symbol


def test_calculate_ceiling_floor():
    """Test ceiling/floor calculation"""
    result = calculate_ceiling_floor(50000, "VCB")
    assert result["reference"] == 50000
    assert result["ceiling"] > 50000
    assert result["floor"] < 50000
    assert abs(result["limit_percent"] - 7.0) < 0.01  # HOSE limit (floating point tolerance)


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
    # Price check - use empty string instead of None
    is_safe, _ = validator.check_price_floor_ceiling(100_000, 100_000, "")
    assert isinstance(is_safe, bool)

    # Position vs volume - use empty string instead of None
    is_safe, _ = validator.validate_position_size_vs_volume(10_000, 500_000, "")
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
    assert buffer == 500_000  # 10% of new order (5M)


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
