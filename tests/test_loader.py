"""
Unit tests for src/data/loader.py - VNStock Data Loader
Updated to match vnstock-based implementation
"""

import os
import pytest
import pickle
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data.loader import (
    load_data,
    _download_from_vnstock,
    DATA_CACHE_DIR,
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
def sample_vnstock_df():
    """Sample DataFrame as returned by vnstock"""
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


@patch("src.data.loader._download_from_vnstock")
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


@patch("src.data.loader._download_from_vnstock")
def test_load_data_with_lookback(mock_download, sample_dataframe, temp_cache_dir):
    """Test load_data with lookback parameter"""
    mock_download.return_value = sample_dataframe

    result = load_data(symbol="VNM", lookback=30, use_cache=False, required_bars=3)

    assert not result.empty
    mock_download.assert_called_once()
    # Verify that start_date was calculated correctly based on lookback
    call_args = mock_download.call_args[0]
    start_dt = call_args[1]
    end_dt = call_args[2]
    # The difference should be approximately 30 days
    assert (end_dt - start_dt).days == 30


@patch("src.data.loader._download_from_vnstock")
def test_load_data_is_index_param(mock_download, sample_dataframe, temp_cache_dir):
    """Test load_data with is_index=True"""
    mock_download.return_value = sample_dataframe

    result = load_data(
        symbol="VNINDEX",
        start_date="2024-01-01",
        end_date="2024-01-03",
        is_index=True,
        required_bars=3,
    )

    # Verify data_type was set to "index"
    call_args = mock_download.call_args[0]
    data_type = call_args[4]
    assert data_type == "index"


def test_load_data_missing_dates():
    """Test load_data without start_date/end_date or lookback"""
    with pytest.raises(ValueError, match="Either provide start_date/end_date or lookback"):
        load_data(symbol="VNM")


@patch("src.data.loader._download_from_vnstock")
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


@patch("src.data.loader._download_from_vnstock")
def test_load_data_empty_response(mock_download, temp_cache_dir):
    """Test load_data with empty dataframe response"""
    mock_download.return_value = pd.DataFrame()

    result = load_data(
        symbol="INVALID", start_date="2024-01-01", end_date="2024-01-03", use_cache=False
    )

    assert result.empty


@patch("src.data.loader._download_from_vnstock")
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


@patch("src.data.loader._download_from_vnstock")
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


@patch("src.data.loader._download_from_vnstock")
def test_load_data_caching_disabled(mock_download, sample_dataframe, temp_cache_dir):
    """Test that caching can be disabled"""
    mock_download.return_value = sample_dataframe

    # Both calls should download
    load_data(
        symbol="VNM",
        start_date="2024-01-01",
        end_date="2024-01-03",
        use_cache=False,
        required_bars=3,
    )
    load_data(
        symbol="VNM",
        start_date="2024-01-01",
        end_date="2024-01-03",
        use_cache=False,
        required_bars=3,
    )

    # Download should be called twice
    assert mock_download.call_count == 2


@patch("src.data.loader._download_from_vnstock")
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
        required_bars=3,
    )
    load_data(
        symbol="VNINDEX",
        start_date="2024-01-01",
        end_date="2024-01-03",
        data_type="index",
        use_cache=True,
        required_bars=3,
    )

    # Should download twice because indexes don't use cache
    assert mock_download.call_count == 2


@patch("src.data.loader._download_from_vnstock")
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
# _DOWNLOAD_FROM_VNSTOCK TESTS
# ============================================================================


@patch("src.data.loader.Vnstock")
@patch("src.data.loader.tcbs_limiter.wait")
def test_download_from_vnstock_success(mock_wait, mock_vnstock_class, sample_vnstock_df):
    """Test successful download from vnstock"""
    # Setup mock
    mock_stock = MagicMock()
    mock_stock.quote.history.return_value = sample_vnstock_df
    mock_vnstock_instance = MagicMock()
    mock_vnstock_instance.stock.return_value = mock_stock
    mock_vnstock_class.return_value = mock_vnstock_instance

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    result = _download_from_vnstock("VNM", start, end, "1D", "stock")

    assert not result.empty
    assert len(result) == 3
    assert "time" in result.columns
    assert "close" in result.columns
    assert mock_wait.called


@patch("src.data.loader.Vnstock")
@patch("src.data.loader.tcbs_limiter.wait")
def test_download_from_vnstock_interval_conversion(
    mock_wait, mock_vnstock_class, sample_vnstock_df
):
    """Test interval format conversion"""
    mock_stock = MagicMock()
    mock_stock.quote.history.return_value = sample_vnstock_df
    mock_vnstock_instance = MagicMock()
    mock_vnstock_instance.stock.return_value = mock_stock
    mock_vnstock_class.return_value = mock_vnstock_instance

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    _download_from_vnstock("VNM", start, end, "D", "stock")

    # Verify interval was converted to "1D"
    mock_stock.quote.history.assert_called_once()
    call_kwargs = mock_stock.quote.history.call_args[1]
    assert call_kwargs["interval"] == "1D"


@patch("src.data.loader.Vnstock")
@patch("src.data.loader.tcbs_limiter.wait")
def test_download_from_vnstock_empty_data(mock_wait, mock_vnstock_class):
    """Test handling of empty data response"""
    mock_stock = MagicMock()
    mock_stock.quote.history.return_value = None
    mock_vnstock_instance = MagicMock()
    mock_vnstock_instance.stock.return_value = mock_stock
    mock_vnstock_class.return_value = mock_vnstock_instance

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    result = _download_from_vnstock("INVALID", start, end, "1D", "stock")

    assert result.empty


@patch("src.data.loader.Vnstock")
@patch("src.data.loader.tcbs_limiter.wait")
@patch("time.sleep")
def test_download_from_vnstock_retry_on_error(
    mock_sleep, mock_wait, mock_vnstock_class, sample_vnstock_df
):
    """Test retry logic on network errors"""
    # First call raises error, second succeeds
    mock_stock = MagicMock()
    mock_stock.quote.history.side_effect = [Exception("timeout error"), sample_vnstock_df]
    mock_vnstock_instance = MagicMock()
    mock_vnstock_instance.stock.return_value = mock_stock
    mock_vnstock_class.return_value = mock_vnstock_instance

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    result = _download_from_vnstock("VNM", start, end, "1D", "stock")

    assert not result.empty
    assert mock_stock.quote.history.call_count == 2
    assert mock_sleep.called


@patch("src.data.loader.Vnstock")
@patch("src.data.loader.tcbs_limiter.wait")
@patch("time.sleep")
def test_download_from_vnstock_max_retries_exceeded(mock_sleep, mock_wait, mock_vnstock_class):
    """Test max retries exceeded"""
    mock_stock = MagicMock()
    mock_stock.quote.history.side_effect = Exception("timeout error")
    mock_vnstock_instance = MagicMock()
    mock_vnstock_instance.stock.return_value = mock_stock
    mock_vnstock_class.return_value = mock_vnstock_instance

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    with pytest.raises(DataLoadError):
        _download_from_vnstock("VNM", start, end, "1D", "stock")

    # Should retry 3 times (max_retries = 3 in loader.py)
    assert mock_stock.quote.history.call_count == 3


@patch("src.data.loader.Vnstock")
@patch("src.data.loader.tcbs_limiter.wait")
def test_download_from_vnstock_data_cleaning(mock_wait, mock_vnstock_class):
    """Test data cleaning (duplicates, null values, invalid prices)"""
    dirty_df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",  # Duplicate
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "open": [100.0, 100.0, None, 106.0, 108.0],  # Null open
            "high": [105.0, 105.0, 108.0, 110.0, 112.0],
            "low": [98.0, 98.0, 102.0, 105.0, 107.0],
            "close": [103.0, 104.0, 106.0, 0, 110.0],  # Zero close (invalid)
            "volume": [1000000, 1000000, 1200000, 1100000, 1150000],
        }
    )

    mock_stock = MagicMock()
    mock_stock.quote.history.return_value = dirty_df
    mock_vnstock_instance = MagicMock()
    mock_vnstock_instance.stock.return_value = mock_stock
    mock_vnstock_class.return_value = mock_vnstock_instance

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 4)

    result = _download_from_vnstock("VNM", start, end, "1D", "stock")

    # Should clean: remove duplicates, remove zero close prices
    # Original: 5 rows -> after dedup: 4 unique dates -> after removing zero close: 3 rows
    assert len(result) == 3
    assert all(result["close"] > 0)
    assert result["time"].is_unique


@patch("src.data.loader.Vnstock")
@patch("src.data.loader.tcbs_limiter.wait")
def test_download_from_vnstock_column_renaming(mock_wait, mock_vnstock_class):
    """Test column renaming (date -> time)"""
    df_with_date = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "open": [100.0, 103.0, 106.0],
            "high": [105.0, 108.0, 110.0],
            "low": [98.0, 102.0, 105.0],
            "close": [103.0, 106.0, 108.0],
            "volume": [1000000, 1200000, 1100000],
        }
    )

    mock_stock = MagicMock()
    mock_stock.quote.history.return_value = df_with_date
    mock_vnstock_instance = MagicMock()
    mock_vnstock_instance.stock.return_value = mock_stock
    mock_vnstock_class.return_value = mock_vnstock_instance

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    result = _download_from_vnstock("VNM", start, end, "1D", "stock")

    assert "time" in result.columns
    assert "date" not in result.columns


@patch("src.data.loader.VNSTOCK_AVAILABLE", False)
def test_download_from_vnstock_not_installed():
    """Test handling when vnstock is not installed"""
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    with pytest.raises(DataLoadError, match="vnstock library not installed"):
        _download_from_vnstock("VNM", start, end, "1D", "stock")


# ============================================================================
# EDGE CASES AND INTEGRATION TESTS
# ============================================================================


@patch("src.data.loader._download_from_vnstock")
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


@patch("src.data.loader._download_from_vnstock")
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


@patch("src.data.loader._download_from_vnstock")
def test_load_data_different_resolutions(mock_download, sample_dataframe, temp_cache_dir):
    """Test load_data with different resolutions"""
    mock_download.return_value = sample_dataframe

    for resolution in ["D", "1D", "W", "1W", "M", "1M"]:
        result = load_data(
            symbol="VNM",
            start_date="2024-01-01",
            end_date="2024-01-03",
            resolution=resolution,
            use_cache=False,
            required_bars=3,
        )
        assert not result.empty


@patch("src.data.loader._download_from_vnstock")
def test_load_data_index_type(mock_download, sample_dataframe, temp_cache_dir):
    """Test load_data with index data type"""
    mock_download.return_value = sample_dataframe

    result = load_data(
        symbol="VNINDEX",
        start_date="2024-01-01",
        end_date="2024-01-03",
        data_type="index",
        use_cache=False,
        required_bars=3,
    )

    assert not result.empty
    # Verify data_type was passed correctly
    call_args = mock_download.call_args[0]
    assert call_args[4] == "index"


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
