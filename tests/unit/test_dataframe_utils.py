"""
Unit tests for utils/dataframe_utils.py
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from utils.dataframe_utils import (
    safe_get_latest,
    safe_get_range,
    safe_rolling_operation,
    validate_dataframe_basic,
    safe_get_close_price,
    safe_get_volume,
    safe_get_support_resistance,
)


class TestSafeGetLatest:
    """Test safe_get_latest function"""

    def test_normal_case(self):
        """Test normal case with valid data"""
        df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        result = safe_get_latest(df, "close")
        assert result == 104

        result = safe_get_latest(df, "volume")
        assert result == 1400

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()

        result = safe_get_latest(df, "close", default=0.0)
        assert result == 0.0

    def test_missing_column(self):
        """Test with missing column"""
        df = pd.DataFrame({"close": [100, 101, 102]})

        result = safe_get_latest(df, "volume", default=-1)
        assert result == -1

    def test_nan_value(self):
        """Test with NaN value"""
        df = pd.DataFrame({"close": [100, 101, np.nan]})

        result = safe_get_latest(df, "close", default=0.0)
        assert result == 0.0

    def test_none_default(self):
        """Test with None as default"""
        df = pd.DataFrame()

        result = safe_get_latest(df, "close")
        assert result is None


class TestSafeGetRange:
    """Test safe_get_range function"""

    def test_normal_range(self):
        """Test normal range access"""
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104, 105, 106]})

        result = safe_get_range(df, "close", start=-3)
        expected = pd.Series([104, 105, 106], index=[4, 5, 6], name="close")
        pd.testing.assert_series_equal(result, expected)

    def test_range_with_end(self):
        """Test range with end parameter"""
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        result = safe_get_range(df, "close", start=-4, end=-1)
        expected = pd.Series([101, 102, 103], index=[1, 2, 3], name="close")
        pd.testing.assert_series_equal(result, expected)

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()

        result = safe_get_range(df, "close", default=[])
        assert result == []

    def test_out_of_bounds(self):
        """Test with out of bounds range"""
        df = pd.DataFrame({"close": [100, 101]})

        result = safe_get_range(df, "close", start=-10, default=None)
        assert result is None


class TestSafeRollingOperation:
    """Test safe_rolling_operation function"""

    def test_rolling_mean(self):
        """Test rolling mean operation"""
        df = pd.DataFrame({"close": [100, 102, 104, 106, 108]})

        result = safe_rolling_operation(df, "close", window=3, operation="mean")
        expected = (104 + 106 + 108) / 3
        assert result == expected

    def test_rolling_min(self):
        """Test rolling min operation"""
        df = pd.DataFrame({"low": [95, 97, 93, 96, 94]})

        result = safe_rolling_operation(df, "low", window=3, operation="min")
        assert result == 93  # min of [93, 96, 94]

    def test_rolling_max(self):
        """Test rolling max operation"""
        df = pd.DataFrame({"high": [105, 107, 103, 106, 104]})

        result = safe_rolling_operation(df, "high", window=3, operation="max")
        assert result == 106  # max of [103, 106, 104]

    def test_insufficient_data(self):
        """Test with insufficient data for window"""
        df = pd.DataFrame({"close": [100, 101]})

        result = safe_rolling_operation(df, "close", window=5, default=0.0)
        assert result == 0.0

    def test_unsupported_operation(self):
        """Test with unsupported operation"""
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        result = safe_rolling_operation(df, "close", window=3, operation="invalid", default=-1)
        assert result == -1

    def test_nan_result(self):
        """Test when rolling operation returns NaN"""
        df = pd.DataFrame({"close": [100, np.nan, np.nan, np.nan, 104]})

        result = safe_rolling_operation(df, "close", window=3, operation="mean", default=0.0)
        assert result == 0.0  # Should return default when result is NaN


class TestValidateDataframeBasic:
    """Test validate_dataframe_basic function"""

    def test_valid_dataframe(self):
        """Test with valid DataFrame"""
        df = pd.DataFrame({"close": [100, 101, 102], "volume": [1000, 1100, 1200]})

        result = validate_dataframe_basic(df, min_rows=2, required_columns=["close", "volume"])
        assert result is True

    def test_none_dataframe(self):
        """Test with None DataFrame"""
        result = validate_dataframe_basic(None)
        assert result is False

    def test_insufficient_rows(self):
        """Test with insufficient rows"""
        df = pd.DataFrame({"close": [100]})

        result = validate_dataframe_basic(df, min_rows=5)
        assert result is False

    def test_missing_columns(self):
        """Test with missing required columns"""
        df = pd.DataFrame({"close": [100, 101, 102]})

        result = validate_dataframe_basic(df, required_columns=["close", "volume"])
        assert result is False

    def test_empty_required_columns(self):
        """Test with empty required columns list"""
        df = pd.DataFrame({"close": [100, 101, 102]})

        result = validate_dataframe_basic(df, required_columns=[])
        assert result is True


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_safe_get_close_price(self):
        """Test safe_get_close_price function"""
        df = pd.DataFrame({"close": [100, 101, 102]})

        result = safe_get_close_price(df)
        assert result == 102

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = safe_get_close_price(empty_df, default=50.0)
        assert result == 50.0

    def test_safe_get_volume(self):
        """Test safe_get_volume function"""
        df = pd.DataFrame({"volume": [1000, 1100, 1200]})

        result = safe_get_volume(df)
        assert result == 1200

        # Test with missing column
        df_no_volume = pd.DataFrame({"close": [100, 101, 102]})
        result = safe_get_volume(df_no_volume, default=0.0)
        assert result == 0.0

    def test_safe_get_support_resistance(self):
        """Test safe_get_support_resistance function"""
        df = pd.DataFrame(
            {
                "high": [105, 107, 109, 108, 110, 112, 111, 113, 115, 114],
                "low": [95, 97, 99, 98, 100, 102, 101, 103, 105, 104],
            }
        )

        support, resistance = safe_get_support_resistance(df, lookback=5)

        # Should get min of last 5 lows and max of last 5 highs
        expected_support = min([102, 101, 103, 105, 104])  # 101
        expected_resistance = max([112, 111, 113, 115, 114])  # 115

        assert support == expected_support
        assert resistance == expected_resistance

    def test_safe_get_support_resistance_insufficient_data(self):
        """Test support/resistance with insufficient data"""
        df = pd.DataFrame({"high": [105, 107], "low": [95, 97]})

        support, resistance = safe_get_support_resistance(df, lookback=10)

        assert support is None
        assert resistance is None


class TestErrorHandling:
    """Test error handling in various scenarios"""

    @patch("utils.dataframe_utils.logger")
    def test_logging_on_errors(self, mock_logger):
        """Test that errors are properly logged"""
        df = pd.DataFrame()

        safe_get_latest(df, "close", default=0.0)

        mock_logger.debug.assert_called_with("DataFrame is empty, returning default for close")

    def test_exception_in_safe_get_latest(self):
        """Test exception handling in safe_get_latest"""

        # Create a mock DataFrame that raises exception
        class BadDataFrame:
            def __len__(self):
                return 5

            @property
            def columns(self):
                return ["close"]

            def __getitem__(self, key):
                raise RuntimeError("Simulated error")

        bad_df = BadDataFrame()

        result = safe_get_latest(bad_df, "close", default=-1)
        assert result == -1

    def test_exception_in_rolling_operation(self):
        """Test exception handling in rolling operations"""
        # Create DataFrame that will cause rolling operation to fail
        df = pd.DataFrame({"close": [100, 101, 102]})

        # Mock rolling to raise exception
        with patch.object(df["close"], "rolling", side_effect=Exception("Rolling failed")):
            result = safe_rolling_operation(df, "close", window=2, operation="mean", default=-1)
            assert result == -1


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_complete_ohlcv_workflow(self):
        """Test complete workflow with OHLCV data"""
        df = pd.DataFrame(
            {
                "open": [99, 101, 103, 105, 107],
                "high": [101, 103, 105, 107, 109],
                "low": [98, 100, 102, 104, 106],
                "close": [100, 102, 104, 106, 108],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        # Validate DataFrame
        assert validate_dataframe_basic(
            df, min_rows=5, required_columns=["open", "high", "low", "close", "volume"]
        )

        # Get latest values
        close_price = safe_get_close_price(df)
        volume = safe_get_volume(df)

        assert close_price == 108
        assert volume == 1400

        # Get support/resistance
        support, resistance = safe_get_support_resistance(df, lookback=3)

        assert support == 102.0  # min of last 3 lows
        assert resistance == 109  # max of [105, 107, 109]

    def test_edge_cases_workflow(self):
        """Test workflow with various edge cases"""
        # Empty DataFrame
        empty_df = pd.DataFrame()

        assert not validate_dataframe_basic(empty_df, min_rows=1)
        assert safe_get_close_price(empty_df, default=0.0) == 0.0

        # DataFrame with NaN values
        nan_df = pd.DataFrame({"close": [100, np.nan, 102], "volume": [1000, 1100, np.nan]})

        assert validate_dataframe_basic(nan_df, min_rows=2)
        assert safe_get_volume(nan_df, default=0.0) == 0.0  # Latest is NaN

        # DataFrame with missing columns
        partial_df = pd.DataFrame({"close": [100, 101, 102]})

        assert not validate_dataframe_basic(partial_df, required_columns=["close", "volume"])
        assert safe_get_volume(partial_df, default=500.0) == 500.0
