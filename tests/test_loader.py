"""
Unit tests for src/data/loader.py - TCBS Data Loader
"""

import os
import pytest
import pickle
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import Timeout, ConnectionError

from src.data.loader import (
    load_data,
    _download_from_tcbs,
    DATA_CACHE_DIR,
    TCBS_API_BASE,
)
from src.config.exceptions import DataLoadError


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory for testing"""
    cache_dir = tmp_path / "test_data_cache"
    cache_dir.mkdir()

    # Patch DATA_CACHE_DIR to use temp directory
    with patch("src.data.loader.DATA_CACHE_DIR", str(cache_dir)):
        yield str(cache_dir)

    # Cleanup
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


@pytest.fixture
def sample_tcbs_response():
    """Sample TCBS API response"""
    return {
        "data": [
            {
                "tradingDate": "2024-01-01T00:00:00",
                "open": 100.0,
                "high": 105.0,
                "low": 98.0,
                "close": 103.0,
                "volume": 1000000,
            },
            {
                "tradingDate": "2024-01-02T00:00:00",
                "open": 103.0,
                "high": 108.0,
                "low": 102.0,
                "close": 106.0,
                "volume": 1200000,
            },
            {
                "tradingDate": "2024-01-03T00:00:00",
                "open": 106.0,
                "high": 110.0,
                "low": 105.0,
                "close": 108.0,
                "volume": 1100000,
            },
        ]
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame as expected after processing"""
    data = {
        "time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "open": [100.0, 103.0, 106.0],
        "high": [105.0, 108.0, 110.0],
        "low": [98.0, 102.0, 105.0],
        "close": [103.0, 106.0, 108.0],
        "volume": [1000000, 1200000, 1100000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def large_sample_data():
    """Large sample data (60 rows) for testing required_bars"""
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    data = {
        "time": dates,
        "open": [100.0 + i for i in range(60)],
        "high": [105.0 + i for i in range(60)],
        "low": [98.0 + i for i in range(60)],
        "close": [103.0 + i for i in range(60)],
        "volume": [1000000 + i * 10000 for i in range(60)],
    }
    return pd.DataFrame(data)


# ============================================================================
# LOAD_DATA BASIC TESTS
# ============================================================================


@patch("src.data.loader._download_from_tcbs")
def test_load_data_success(mock_download, sample_dataframe, temp_cache_dir):
    """Test successful data loading"""
    mock_download.return_value = sample_dataframe

    result = load_data(
        symbol="VNM",
        start_date="2024-01-01",
        end_date="2024-01-03",
        use_cache=False,
        required_bars=3,
    )

    assert not result.empty
    assert len(result) == 3
    assert list(result.columns) == ["time", "open", "high", "low", "close", "volume"]
    mock_download.assert_called_once()


@patch("src.data.loader._download_from_tcbs")
def test_load_data_with_lookback(mock_download, sample_dataframe, temp_cache_dir):
    """Test load_data with lookback parameter"""
    mock_download.return_value = sample_dataframe

    with patch("src.data.loader.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 1, 10)
        mock_datetime.strftime = datetime.strftime
        mock_datetime.strptime = datetime.strptime

        result = load_data(symbol="VNM", lookback=30, use_cache=False, required_bars=3)

        assert not result.empty
        # Verify that start_date was calculated correctly
        call_args = mock_download.call_args[0]
        start_dt = call_args[1]
        assert (datetime(2024, 1, 10) - start_dt).days == 30


@patch("src.data.loader._download_from_tcbs")
def test_load_data_is_index_param(mock_download, sample_dataframe, temp_cache_dir):
    """Test load_data with is_index=True"""
    mock_download.return_value = sample_dataframe

    result = load_data(
        symbol="VNINDEX", start_date="2024-01-01", end_date="2024-01-03", is_index=True
    )

    # Verify data_type was set to "index"
    call_args = mock_download.call_args[0]
    data_type = call_args[4]
    assert data_type == "index"


def test_load_data_missing_dates():
    """Test load_data without start_date/end_date or lookback"""
    with pytest.raises(ValueError, match="Either provide start_date/end_date or lookback"):
        load_data(symbol="VNM")


@patch("src.data.loader._download_from_tcbs")
def test_load_data_insufficient_bars(mock_download, temp_cache_dir):
    """Test load_data with insufficient bars"""
    small_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "open": [100.0, 103.0],
            "high": [105.0, 108.0],
            "low": [98.0, 102.0],
            "close": [103.0, 106.0],
            "volume": [1000000, 1200000],
        }
    )
    mock_download.return_value = small_df

    result = load_data(
        symbol="VNM",
        start_date="2024-01-01",
        end_date="2024-01-03",
        use_cache=False,
        required_bars=50,  # Require 50 bars but only get 2
    )

    assert result.empty


@patch("src.data.loader._download_from_tcbs")
def test_load_data_empty_response(mock_download, temp_cache_dir):
    """Test load_data with empty dataframe response"""
    mock_download.return_value = pd.DataFrame()

    result = load_data(
        symbol="INVALID", start_date="2024-01-01", end_date="2024-01-03", use_cache=False
    )

    assert result.empty


@patch("src.data.loader._download_from_tcbs")
def test_load_data_download_exception(mock_download, temp_cache_dir):
    """Test load_data handles download exceptions"""
    mock_download.side_effect = Exception("API Error")

    result = load_data(
        symbol="VNM", start_date="2024-01-01", end_date="2024-01-03", use_cache=False
    )

    assert result.empty


# ============================================================================
# CACHING TESTS
# ============================================================================


@patch("src.data.loader._download_from_tcbs")
def test_load_data_caching_enabled(mock_download, large_sample_data, temp_cache_dir):
    """Test that caching works correctly"""
    mock_download.return_value = large_sample_data

    # First call should download and cache
    result1 = load_data(
        symbol="VNM", start_date="2024-01-01", end_date="2024-03-01", use_cache=True
    )

    # Second call should use cache
    result2 = load_data(
        symbol="VNM", start_date="2024-01-01", end_date="2024-03-01", use_cache=True
    )

    # Download should only be called once
    assert mock_download.call_count == 1
    pd.testing.assert_frame_equal(result1, result2)


@patch("src.data.loader._download_from_tcbs")
def test_load_data_caching_disabled(mock_download, sample_dataframe, temp_cache_dir):
    """Test that caching can be disabled"""
    mock_download.return_value = sample_dataframe

    # Both calls should download
    load_data(symbol="VNM", start_date="2024-01-01", end_date="2024-01-03", use_cache=False)
    load_data(symbol="VNM", start_date="2024-01-01", end_date="2024-01-03", use_cache=False)

    # Download should be called twice
    assert mock_download.call_count == 2


@patch("src.data.loader._download_from_tcbs")
def test_load_data_index_no_cache(mock_download, sample_dataframe, temp_cache_dir):
    """Test that indexes are not cached"""
    mock_download.return_value = sample_dataframe

    # Even with use_cache=True, index data should not use cache
    load_data(
        symbol="VNINDEX",
        start_date="2024-01-01",
        end_date="2024-01-03",
        data_type="index",
        use_cache=True,
    )
    load_data(
        symbol="VNINDEX",
        start_date="2024-01-01",
        end_date="2024-01-03",
        data_type="index",
        use_cache=True,
    )

    # Should download twice because indexes don't use cache
    assert mock_download.call_count == 2


@patch("src.data.loader._download_from_tcbs")
def test_load_data_cache_corruption(mock_download, large_sample_data, temp_cache_dir):
    """Test handling of corrupted cache file"""
    mock_download.return_value = large_sample_data

    # Create corrupted cache file
    import hashlib

    cache_key = "VNM_2024-01-01_2024-03-01_1D_stock"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = os.path.join(temp_cache_dir, f"{cache_hash}.pkl")

    os.makedirs(temp_cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        f.write("corrupted data")

    # Should handle corruption and refetch
    result = load_data(symbol="VNM", start_date="2024-01-01", end_date="2024-03-01", use_cache=True)

    assert not result.empty
    mock_download.assert_called_once()


# ============================================================================
# _DOWNLOAD_FROM_TCBS TESTS
# ============================================================================


@patch("requests.get")
@patch("src.data.loader.tcbs_limiter.wait")
def test_download_from_tcbs_success(mock_wait, mock_get, sample_tcbs_response):
    """Test successful download from TCBS API"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_tcbs_response
    mock_get.return_value = mock_response

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    result = _download_from_tcbs("VNM", start, end, "1D", "stock")

    assert not result.empty
    assert len(result) == 3
    assert list(result.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert mock_wait.called


@patch("requests.get")
@patch("src.data.loader.tcbs_limiter.wait")
def test_download_from_tcbs_resolution_conversion(mock_wait, mock_get, sample_tcbs_response):
    """Test resolution format conversion (1D -> D)"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_tcbs_response
    mock_get.return_value = mock_response

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    _download_from_tcbs("VNM", start, end, "1D", "stock")

    # Verify resolution was converted to "D"
    call_args = mock_get.call_args
    params = call_args[1]["params"]
    assert params["resolution"] == "D"


@patch("requests.get")
@patch("src.data.loader.tcbs_limiter.wait")
def test_download_from_tcbs_empty_data(mock_wait, mock_get):
    """Test handling of empty data response"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": []}
    mock_get.return_value = mock_response

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    result = _download_from_tcbs("INVALID", start, end, "1D", "stock")

    assert result.empty
    assert hasattr(result, "source_error")


@patch("requests.get")
@patch("src.data.loader.tcbs_limiter.wait")
def test_download_from_tcbs_invalid_response_format(mock_wait, mock_get):
    """Test handling of invalid API response format"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"invalid": "format"}
    mock_get.return_value = mock_response

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    with pytest.raises(DataLoadError, match="invalid format"):
        _download_from_tcbs("VNM", start, end, "1D", "stock")


@patch("requests.get")
@patch("src.data.loader.tcbs_limiter.wait")
def test_download_from_tcbs_http_error(mock_wait, mock_get):
    """Test handling of HTTP errors"""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    with pytest.raises(DataLoadError, match="status 404"):
        _download_from_tcbs("INVALID", start, end, "1D", "stock")


@patch("requests.get")
@patch("src.data.loader.tcbs_limiter.wait")
@patch("time.sleep")
def test_download_from_tcbs_retry_on_429(mock_sleep, mock_wait, mock_get, sample_tcbs_response):
    """Test retry logic on 429 status code"""
    # First call returns 429, second succeeds
    mock_response_429 = Mock()
    mock_response_429.status_code = 429

    mock_response_success = Mock()
    mock_response_success.status_code = 200
    mock_response_success.json.return_value = sample_tcbs_response

    mock_get.side_effect = [mock_response_429, mock_response_success]

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    result = _download_from_tcbs("VNM", start, end, "1D", "stock")

    assert not result.empty
    assert mock_get.call_count == 2
    assert mock_sleep.called


@patch("requests.get")
@patch("src.data.loader.tcbs_limiter.wait")
@patch("time.sleep")
def test_download_from_tcbs_retry_on_timeout(mock_sleep, mock_wait, mock_get, sample_tcbs_response):
    """Test retry logic on timeout"""
    # First call times out, second succeeds
    mock_response_success = Mock()
    mock_response_success.status_code = 200
    mock_response_success.json.return_value = sample_tcbs_response

    mock_get.side_effect = [Timeout("Request timeout"), mock_response_success]

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    result = _download_from_tcbs("VNM", start, end, "1D", "stock")

    assert not result.empty
    assert mock_get.call_count == 2
    assert mock_sleep.called


@patch("requests.get")
@patch("src.data.loader.tcbs_limiter.wait")
@patch("time.sleep")
def test_download_from_tcbs_max_retries_exceeded(mock_sleep, mock_wait, mock_get):
    """Test max retries exceeded"""
    mock_get.side_effect = Timeout("Request timeout")

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    with pytest.raises(DataLoadError, match="after 5 retries"):
        _download_from_tcbs("VNM", start, end, "1D", "stock")

    # Should retry 5 times (max_retries = 5 in loader.py)
    assert mock_get.call_count == 5


@patch("requests.get")
@patch("src.data.loader.tcbs_limiter.wait")
def test_download_from_tcbs_data_cleaning(mock_wait, mock_get):
    """Test data cleaning (duplicates, null values, invalid prices)"""
    dirty_response = {
        "data": [
            {
                "tradingDate": "2024-01-01T00:00:00",
                "open": 100.0,
                "high": 105.0,
                "low": 98.0,
                "close": 103.0,
                "volume": 1000000,
            },
            {
                "tradingDate": "2024-01-01T00:00:00",  # Duplicate
                "open": 100.0,
                "high": 105.0,
                "low": 98.0,
                "close": 104.0,  # Different close
                "volume": 1000000,
            },
            {
                "tradingDate": "2024-01-02T00:00:00",
                "open": None,  # Null open
                "high": 108.0,
                "low": 102.0,
                "close": 106.0,
                "volume": 1200000,
            },
            {
                "tradingDate": "2024-01-03T00:00:00",
                "open": 106.0,
                "high": 110.0,
                "low": 105.0,
                "close": 0,  # Invalid price (zero)
                "volume": 1100000,
            },
            {
                "tradingDate": "2024-01-04T00:00:00",
                "open": 108.0,
                "high": 112.0,
                "low": 107.0,
                "close": 110.0,
                "volume": 1150000,
            },
        ]
    }

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = dirty_response
    mock_get.return_value = mock_response

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 4)

    result = _download_from_tcbs("VNM", start, end, "1D", "stock")

    # Should keep only valid rows (01-01 duplicate kept last, 01-02 null kept, 01-03 zero removed, 01-04 valid)
    # Actually: 01-01 (1 row from duplicate), 01-02 (kept), 01-04 (kept) = 3 rows
    # Zero close is removed, duplicates keep last
    assert len(result) == 3
    assert all(result["close"] > 0)


# ============================================================================
# EDGE CASES AND INTEGRATION TESTS
# ============================================================================


@patch("src.data.loader._download_from_tcbs")
def test_load_data_sorting(mock_download, temp_cache_dir):
    """Test that data is sorted by time"""
    # Create unsorted data with 60 rows
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    # Reverse the dates to make them unsorted
    unsorted_df = pd.DataFrame(
        {
            "time": dates[::-1],  # Reversed order
            "open": [100.0 + i for i in range(60)],
            "high": [105.0 + i for i in range(60)],
            "low": [98.0 + i for i in range(60)],
            "close": [103.0 + i for i in range(60)],
            "volume": [1000000 + i * 10000 for i in range(60)],
        }
    )
    mock_download.return_value = unsorted_df

    result = load_data(
        symbol="VNM", start_date="2024-01-01", end_date="2024-03-01", use_cache=False
    )

    # Should be sorted by time
    assert not result.empty
    assert result["time"].is_monotonic_increasing


@patch("src.data.loader._download_from_tcbs")
def test_load_data_with_large_dataset(mock_download, large_sample_data, temp_cache_dir):
    """Test load_data with large dataset"""
    mock_download.return_value = large_sample_data

    result = load_data(
        symbol="VNM",
        start_date="2024-01-01",
        end_date="2024-03-01",
        use_cache=False,
        required_bars=50,
    )

    assert not result.empty
    assert len(result) >= 50


def test_tcbs_api_base_url():
    """Test that TCBS API base URL is correctly set"""
    assert TCBS_API_BASE == "https://apipubaws.tcbs.com.vn"


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
