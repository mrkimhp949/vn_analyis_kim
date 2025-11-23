"""
Unit tests for src/data/tcbs_provider.py - TCBS Provider for Fundamental Data
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.data.tcbs_provider import (
    TCBSProvider,
    get_tcbs_fundamental_data,
    VNSTOCK_AVAILABLE,
)
from src.data.fundamental_data import FundamentalData


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_ratio_data():
    """Create sample financial ratio data from TCBS/VCI"""
    data = {
        "pe": [15.5],
        "pb": [2.3],
        "roe": [18.5],
        "roa": [12.3],
        "de": [0.45],
        "eps": [5000],
        "marketCap": [50000000000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_ratio_data_alternative_keys():
    """Sample data with alternative column names"""
    data = {
        "priceToEarning": [20.0],
        "priceToBook": [3.5],
        "returnOnEquity": [22.0],
        "returnOnAssets": [15.0],
        "debtToEquity": [0.60],
        "earningsPerShare": [6000],
        "von_hoa": [75000000000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_ratio_data_with_nulls():
    """Sample data with null/NaN values"""
    data = {
        "pe": [np.nan],
        "pb": [None],
        "roe": ["-"],
        "roa": ["N/A"],
        "de": ["null"],
        "eps": [""],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_vnstock():
    """Mock Vnstock object"""
    mock = MagicMock()
    return mock


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_tcbs_provider_init_default():
    """Test TCBSProvider initialization with defaults"""
    provider = TCBSProvider()

    assert provider.timeout == 10
    assert provider.source == "VCI"
    assert provider.vnstock is not None


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_tcbs_provider_init_custom():
    """Test TCBSProvider initialization with custom parameters"""
    provider = TCBSProvider(timeout=20, source="TCBS")

    assert provider.timeout == 20
    assert provider.source == "TCBS"
    assert provider.vnstock is not None


@patch("src.data.tcbs_provider.VNSTOCK_AVAILABLE", False)
def test_tcbs_provider_init_vnstock_not_available():
    """Test initialization when vnstock is not available"""
    with pytest.raises(ImportError, match="vnstock package not installed"):
        TCBSProvider()


# ============================================================================
# _IS_VALID_NUMBER TESTS
# ============================================================================


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_is_valid_number_valid_float():
    """Test _is_valid_number with valid float"""
    provider = TCBSProvider()

    assert provider._is_valid_number(15.5) is True
    assert provider._is_valid_number(0.0) is True
    assert provider._is_valid_number(-10.5) is True


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_is_valid_number_valid_int():
    """Test _is_valid_number with valid integers"""
    provider = TCBSProvider()

    assert provider._is_valid_number(100) is True
    assert provider._is_valid_number(0) is True
    assert provider._is_valid_number(-50) is True


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_is_valid_number_valid_string():
    """Test _is_valid_number with valid numeric strings"""
    provider = TCBSProvider()

    assert provider._is_valid_number("15.5") is True
    assert provider._is_valid_number("100") is True


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_is_valid_number_invalid_values():
    """Test _is_valid_number with invalid values"""
    provider = TCBSProvider()

    # None and special strings
    assert provider._is_valid_number(None) is False
    assert provider._is_valid_number(np.nan) is False
    assert provider._is_valid_number("nan") is False
    assert provider._is_valid_number("NaN") is False
    assert provider._is_valid_number("none") is False
    assert provider._is_valid_number("None") is False
    assert provider._is_valid_number("") is False
    assert provider._is_valid_number("null") is False
    assert provider._is_valid_number("n/a") is False
    assert provider._is_valid_number("N/A") is False
    assert provider._is_valid_number("-") is False


# ============================================================================
# _EXTRACT_VALUE TESTS
# ============================================================================


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_extract_value_exact_match(sample_ratio_data):
    """Test _extract_value with exact column name match"""
    provider = TCBSProvider()
    row = sample_ratio_data.iloc[0]

    pe = provider._extract_value(row, ["pe", "PE", "priceToEarning"])
    assert pe == 15.5

    pb = provider._extract_value(row, ["pb", "PB", "priceToBook"])
    assert pb == 2.3


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_extract_value_alternative_keys(sample_ratio_data_alternative_keys):
    """Test _extract_value with alternative column names"""
    provider = TCBSProvider()
    row = sample_ratio_data_alternative_keys.iloc[0]

    pe = provider._extract_value(row, ["pe", "priceToEarning", "PE"])
    assert pe == 20.0

    pb = provider._extract_value(row, ["pb", "priceToBook", "PB"])
    assert pb == 3.5

    roe = provider._extract_value(row, ["roe", "returnOnEquity", "ROE"])
    assert roe == 22.0


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_extract_value_case_insensitive(sample_ratio_data):
    """Test _extract_value with case-insensitive matching"""
    provider = TCBSProvider()
    row = sample_ratio_data.iloc[0]

    # Try uppercase keys when data has lowercase
    pe = provider._extract_value(row, ["PE", "pe"])
    assert pe == 15.5

    # Try mixed case
    market_cap = provider._extract_value(row, ["MARKETCAP", "marketcap", "marketCap"])
    assert market_cap == 50000000000


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_extract_value_not_found(sample_ratio_data):
    """Test _extract_value when key is not found"""
    provider = TCBSProvider()
    row = sample_ratio_data.iloc[0]

    result = provider._extract_value(row, ["nonexistent", "missing"])
    assert result is None


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_extract_value_with_nulls(sample_ratio_data_with_nulls):
    """Test _extract_value with null/invalid values"""
    provider = TCBSProvider()
    row = sample_ratio_data_with_nulls.iloc[0]

    pe = provider._extract_value(row, ["pe"])
    assert pe is None

    pb = provider._extract_value(row, ["pb"])
    assert pb is None

    roe = provider._extract_value(row, ["roe"])
    assert roe is None


# ============================================================================
# GET_FUNDAMENTAL_DATA TESTS
# ============================================================================


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_get_fundamental_data_success(mock_vnstock_class, sample_ratio_data):
    """Test successful retrieval of fundamental data"""
    # Setup mock
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.return_value = sample_ratio_data

    # Test
    provider = TCBSProvider(source="VCI")
    result = provider.get_fundamental_data("VNM")

    # Assertions
    assert result is not None
    assert isinstance(result, FundamentalData)
    assert result.symbol == "VNM"
    assert result.pe_ratio == 15.5
    assert result.pb_ratio == 2.3
    assert result.roe == 18.5
    assert result.roa == 12.3
    assert result.debt_to_equity == 0.45
    assert result.eps == 5000
    assert result.market_cap == 50000000000
    assert result.source == "TCBS/VCI"
    assert isinstance(result.timestamp, datetime)

    # Verify calls
    mock_vnstock.stock.assert_called_once_with(symbol="VNM", source="VCI")
    mock_finance.ratio.assert_called_once_with(period="year", lang="vi")


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_get_fundamental_data_empty_dataframe(mock_vnstock_class):
    """Test when ratio data is empty"""
    # Setup mock
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.return_value = pd.DataFrame()

    # Test
    provider = TCBSProvider()
    result = provider.get_fundamental_data("INVALID")

    # Assertions
    assert result is None


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_get_fundamental_data_missing_pe_and_pb(mock_vnstock_class, sample_ratio_data_with_nulls):
    """Test when both P/E and P/B are missing"""
    # Setup mock
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.return_value = sample_ratio_data_with_nulls

    # Test
    provider = TCBSProvider()
    result = provider.get_fundamental_data("TEST")

    # Assertions - should return None if no P/E or P/B
    assert result is None


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_get_fundamental_data_only_pe(mock_vnstock_class):
    """Test when only P/E is available (P/B is None)"""
    # Create data with only P/E
    data = pd.DataFrame({"pe": [15.5]})

    # Setup mock
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.return_value = data

    # Test
    provider = TCBSProvider()
    result = provider.get_fundamental_data("VNM")

    # Assertions - should succeed with only P/E
    assert result is not None
    assert result.pe_ratio == 15.5
    assert result.pb_ratio is None


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_get_fundamental_data_only_pb(mock_vnstock_class):
    """Test when only P/B is available (P/E is None)"""
    # Create data with only P/B
    data = pd.DataFrame({"pb": [2.5]})

    # Setup mock
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.return_value = data

    # Test
    provider = TCBSProvider()
    result = provider.get_fundamental_data("VCB")

    # Assertions - should succeed with only P/B
    assert result is not None
    assert result.pe_ratio is None
    assert result.pb_ratio == 2.5


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_get_fundamental_data_vci_fails_tcbs_succeeds(mock_vnstock_class, sample_ratio_data):
    """Test VCI failure triggers TCBS fallback"""
    # Setup mock
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    # First call (VCI) fails, second call (TCBS) succeeds
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("VCI connection error")
        else:
            return sample_ratio_data

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.side_effect = side_effect

    # Test with VCI source
    provider = TCBSProvider(source="VCI")
    result = provider.get_fundamental_data("VNM")

    # Assertions - should get data from TCBS fallback
    assert result is not None
    assert result.pe_ratio == 15.5
    # Source should be from TCBS fallback
    assert "TCBS" in result.source


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_get_fundamental_data_both_fail(mock_vnstock_class):
    """Test when both VCI and TCBS fail"""
    # Setup mock to always fail
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.side_effect = Exception("Connection error")

    # Test
    provider = TCBSProvider(source="VCI")
    result = provider.get_fundamental_data("INVALID")

    # Assertions - should return None
    assert result is None


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_get_fundamental_data_tcbs_source_no_fallback(mock_vnstock_class):
    """Test that TCBS source doesn't trigger VCI fallback"""
    # Setup mock to fail
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.side_effect = Exception("TCBS error")

    # Test with TCBS source (not VCI)
    provider = TCBSProvider(source="TCBS")
    result = provider.get_fundamental_data("VNM")

    # Should return None without fallback (fallback only for VCI)
    assert result is None


# ============================================================================
# GET_EARNINGS_DATE TESTS
# ============================================================================


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_get_earnings_date_not_implemented():
    """Test that get_earnings_date returns None (not implemented)"""
    provider = TCBSProvider()
    result = provider.get_earnings_date("VNM")

    assert result is None


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_get_tcbs_fundamental_data_function(mock_vnstock_class, sample_ratio_data):
    """Test convenience function get_tcbs_fundamental_data"""
    # Setup mock
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.return_value = sample_ratio_data

    # Test
    result = get_tcbs_fundamental_data("VNM", source="VCI")

    # Assertions
    assert result is not None
    assert result.symbol == "VNM"
    assert result.pe_ratio == 15.5


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_get_tcbs_fundamental_data_default_source(mock_vnstock_class, sample_ratio_data):
    """Test convenience function with default VCI source"""
    # Setup mock
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.return_value = sample_ratio_data

    # Test without specifying source
    result = get_tcbs_fundamental_data("VNM")

    # Should default to VCI
    assert result is not None
    mock_vnstock.stock.assert_called_with(symbol="VNM", source="VCI")


# ============================================================================
# EDGE CASES AND INTEGRATION TESTS
# ============================================================================


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_multiple_rows_uses_latest(mock_vnstock_class):
    """Test that when multiple rows exist, latest (last) is used"""
    # Create data with multiple rows
    data = pd.DataFrame(
        {
            "pe": [10.0, 12.0, 15.5],  # Latest is 15.5
            "pb": [1.5, 2.0, 2.3],  # Latest is 2.3
        }
    )

    # Setup mock
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.return_value = data

    # Test
    provider = TCBSProvider()
    result = provider.get_fundamental_data("VNM")

    # Should use latest (last row)
    assert result is not None
    assert result.pe_ratio == 15.5
    assert result.pb_ratio == 2.3


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
@patch("src.data.tcbs_provider.Vnstock")
def test_mixed_valid_invalid_values(mock_vnstock_class):
    """Test data with mix of valid and invalid values"""
    # Create data with some valid and some invalid values
    data = pd.DataFrame(
        {
            "pe": [15.5],  # Valid
            "pb": [np.nan],  # Invalid
            "roe": ["N/A"],  # Invalid
            "roa": [12.0],  # Valid
        }
    )

    # Setup mock
    mock_vnstock = MagicMock()
    mock_stock = MagicMock()
    mock_finance = MagicMock()

    mock_vnstock_class.return_value = mock_vnstock
    mock_vnstock.stock.return_value = mock_stock
    mock_stock.finance = mock_finance
    mock_finance.ratio.return_value = data

    # Test
    provider = TCBSProvider()
    result = provider.get_fundamental_data("VNM")

    # Should extract valid values and set None for invalid
    assert result is not None
    assert result.pe_ratio == 15.5
    assert result.pb_ratio is None
    assert result.roe is None
    assert result.roa == 12.0


@pytest.mark.skipif(not VNSTOCK_AVAILABLE, reason="vnstock not installed")
def test_fundamental_data_is_valid():
    """Test that created FundamentalData passes is_valid() check"""
    provider = TCBSProvider()

    # Create valid data
    data = FundamentalData(
        symbol="VNM",
        pe_ratio=15.5,
        pb_ratio=2.3,
        source="TCBS/VCI",
        timestamp=datetime.now(),
    )

    assert data.is_valid() is True


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
