"""
Unit tests for utils/indicators.py
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock

from utils.indicators import (
    IndicatorUtils,
    StopLossCalculator,
    indicator_utils,
    stop_loss_calculator,
)


class TestIndicatorUtils:
    """Test IndicatorUtils class"""

    def test_get_atr_with_valid_atr_column(self):
        """Test ATR calculation with valid ATR column"""
        df = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104], "atr": [2.0, 2.1, 2.2, 2.3, 2.4]}
        )

        result = IndicatorUtils.get_atr(df)

        assert result == 2.4
        assert isinstance(result, float)

    def test_get_atr_with_nan_atr(self):
        """Test ATR calculation when ATR column has NaN values"""
        df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "atr": [2.0, 2.1, np.nan, np.nan, np.nan],
            }
        )

        result = IndicatorUtils.get_atr(df, default_pct=0.03)

        # Should fallback to percentage of close price
        expected = 104 * 0.03
        assert result == expected

    def test_get_atr_without_atr_column(self):
        """Test ATR calculation without ATR column"""
        df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "high": [102, 103, 104, 105, 106],
                "low": [98, 99, 100, 101, 102],
            }
        )

        result = IndicatorUtils.get_atr(df, default_pct=0.025)

        # Should use default percentage
        expected = 104 * 0.025
        assert result == expected

    def test_get_atr_with_zero_atr(self):
        """Test ATR calculation when ATR is zero"""
        df = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104], "atr": [2.0, 2.1, 2.2, 2.3, 0.0]}
        )

        result = IndicatorUtils.get_atr(df, default_pct=0.02)

        # Should fallback to percentage
        expected = 104 * 0.02
        assert result == expected

    def test_get_atr_with_negative_atr(self):
        """Test ATR calculation when ATR is negative"""
        df = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104], "atr": [2.0, 2.1, 2.2, 2.3, -1.0]}
        )

        result = IndicatorUtils.get_atr(df, default_pct=0.02)

        # Should fallback to percentage
        expected = 104 * 0.02
        assert result == expected

    def test_get_rsi_with_valid_rsi_column(self):
        """Test RSI calculation with valid RSI column"""
        df = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104], "rsi": [30.0, 40.0, 50.0, 60.0, 70.0]}
        )

        result = IndicatorUtils.get_rsi(df)

        assert result == 70.0
        assert isinstance(result, float)

    def test_get_rsi_with_nan_rsi(self):
        """Test RSI calculation when RSI column has NaN values"""
        df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "rsi": [30.0, 40.0, np.nan, np.nan, np.nan],
            }
        )

        result = IndicatorUtils.get_rsi(df, default=55.0)

        assert result == 55.0

    def test_get_rsi_without_rsi_column(self):
        """Test RSI calculation without RSI column"""
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

        result = IndicatorUtils.get_rsi(df, default=45.0)

        assert result == 45.0

    def test_get_rsi_with_invalid_rsi_values(self):
        """Test RSI calculation with invalid RSI values"""
        df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "rsi": [30.0, 40.0, 50.0, 60.0, 150.0],  # Invalid: > 100
            }
        )

        result = IndicatorUtils.get_rsi(df, default=50.0)

        assert result == 50.0  # Should use default

        # Test negative RSI
        df["rsi"] = [30.0, 40.0, 50.0, 60.0, -10.0]  # Invalid: < 0
        result = IndicatorUtils.get_rsi(df, default=50.0)
        assert result == 50.0

    @patch("utils.indicators.logger")
    def test_get_rsi_with_exception(self, mock_logger):
        """Test RSI calculation when exception occurs"""
        df = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104], "rsi": [30.0, 40.0, 50.0, 60.0, 70.0]}
        )

        # Mock pandas to raise exception
        with patch("pandas.notna", side_effect=Exception("Test error")):
            result = IndicatorUtils.get_rsi(df, default=42.0)

        assert result == 42.0
        mock_logger.warning.assert_called_once_with("Error getting RSI")

    def test_get_support_resistance_normal_case(self):
        """Test support/resistance calculation normal case"""
        df = pd.DataFrame(
            {
                "high": [
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    123,
                    124,
                    125,
                ],
                "low": [
                    95,
                    96,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                ],
            }
        )

        support, resistance = IndicatorUtils.get_support_resistance(df, lookback=20)

        assert support == 96.0  # Min of last 20 lows (starts from 96)
        assert resistance == 125.0  # Max of last 20 highs
        assert isinstance(support, float)
        assert isinstance(resistance, float)

    def test_get_support_resistance_insufficient_data(self):
        """Test support/resistance with insufficient data"""
        df = pd.DataFrame({"high": [105, 106, 107], "low": [95, 96, 97]})

        support, resistance = IndicatorUtils.get_support_resistance(df, lookback=20)

        assert support is None
        assert resistance is None

    def test_get_support_resistance_with_nan_values(self):
        """Test support/resistance with NaN values"""
        df = pd.DataFrame(
            {
                "high": [
                    105,
                    106,
                    np.nan,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    123,
                    124,
                    125,
                ],
                "low": [
                    95,
                    96,
                    97,
                    np.nan,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                ],
            }
        )

        support, resistance = IndicatorUtils.get_support_resistance(df, lookback=20)

        # Should handle NaN values gracefully
        assert isinstance(support, (float, type(None)))
        assert isinstance(resistance, (float, type(None)))

    def test_get_support_resistance_with_zero_values(self):
        """Test support/resistance with zero values"""
        df = pd.DataFrame(
            {
                "high": [
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    123,
                    124,
                    125,
                ],
                "low": [
                    0,
                    96,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,  # Zero low
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                ],
            }
        )

        support, resistance = IndicatorUtils.get_support_resistance(df, lookback=20)

        assert support == 96.0  # Min of non-zero lows
        assert resistance == 125.0

    def test_get_support_resistance_with_exception(self):
        """Test support/resistance when exception occurs"""
        # Test with empty DataFrame to trigger exception path
        df = pd.DataFrame()

        support, resistance = IndicatorUtils.get_support_resistance(df)

        assert support is None
        assert resistance is None


class TestStopLossCalculator:
    """Test StopLossCalculator class"""

    def test_calculate_stop_loss_normal_case(self):
        """Test normal stop loss calculation"""
        entry_price = 100_000
        atr = 2_000
        support_level = 96_000

        stop_loss, reason = StopLossCalculator.calculate_stop_loss(
            entry_price, atr, support_level
        )

        # Should use support level (96,000) as it's higher than ATR stop (96,000)
        assert stop_loss == 96_000
        assert reason == "Support-based"

    def test_calculate_stop_loss_atr_based(self):
        """Test ATR-based stop loss calculation"""
        entry_price = 100_000
        atr = 2_000
        support_level = 90_000  # Lower than ATR stop

        stop_loss, reason = StopLossCalculator.calculate_stop_loss(
            entry_price, atr, support_level, atr_multiplier=2.0
        )

        # Should use ATR stop (96,000) as it's higher than support
        expected_stop = 100_000 - (2_000 * 2.0)
        assert stop_loss == expected_stop
        assert reason == "ATR-based"

    def test_calculate_stop_loss_no_support(self):
        """Test stop loss calculation without support level"""
        entry_price = 100_000
        atr = 2_000

        stop_loss, reason = StopLossCalculator.calculate_stop_loss(
            entry_price, atr, support_level=None
        )

        expected_stop = 100_000 - (2_000 * 2.0)  # Default multiplier
        assert stop_loss == expected_stop
        assert reason == "ATR-based"

    def test_calculate_stop_loss_invalid_support(self):
        """Test stop loss with invalid support levels"""
        entry_price = 100_000
        atr = 2_000

        # Test support > entry price
        stop_loss, reason = StopLossCalculator.calculate_stop_loss(
            entry_price, atr, support_level=105_000
        )
        expected_stop = 100_000 - (2_000 * 2.0)
        assert stop_loss == expected_stop
        assert reason == "ATR-based"

        # Test NaN support
        stop_loss, reason = StopLossCalculator.calculate_stop_loss(
            entry_price, atr, support_level=np.nan
        )
        assert stop_loss == expected_stop
        assert reason == "ATR-based"

        # Test zero support
        stop_loss, reason = StopLossCalculator.calculate_stop_loss(
            entry_price, atr, support_level=0
        )
        assert stop_loss == expected_stop
        assert reason == "ATR-based"

    @patch("utils.indicators.logger")
    def test_calculate_stop_loss_minimum_stop_enforcement(self, mock_logger):
        """Test minimum stop loss enforcement"""
        entry_price = 100_000
        atr = 500  # Very small ATR

        stop_loss, reason = StopLossCalculator.calculate_stop_loss(
            entry_price, atr, support_level=None, min_stop_pct=0.03
        )

        # Should enforce minimum 3% stop
        expected_min_stop = 100_000 * (1 - 0.03)
        assert stop_loss == expected_min_stop
        assert "Minimum 3.0% stop" in reason
        mock_logger.warning.assert_called_once()

    @patch("utils.indicators.logger")
    def test_calculate_stop_loss_maximum_stop_enforcement(self, mock_logger):
        """Test maximum stop loss enforcement"""
        entry_price = 100_000
        atr = 8_000  # Very large ATR

        stop_loss, reason = StopLossCalculator.calculate_stop_loss(
            entry_price, atr, support_level=None, max_stop_pct=0.10
        )

        # Should enforce maximum 10% stop
        expected_max_stop = 100_000 * (1 - 0.10)
        assert stop_loss == expected_max_stop
        assert "Maximum 10.0% stop" in reason
        mock_logger.warning.assert_called_once()

    def test_calculate_stop_loss_invalid_entry_price(self):
        """Test stop loss with invalid entry price"""
        with pytest.raises(ValueError, match="Invalid entry_price"):
            StopLossCalculator.calculate_stop_loss(0, 2_000)

        with pytest.raises(ValueError, match="Invalid entry_price"):
            StopLossCalculator.calculate_stop_loss(-100, 2_000)

    def test_calculate_stop_loss_invalid_atr(self):
        """Test stop loss with invalid ATR"""
        with pytest.raises(ValueError, match="Invalid ATR"):
            StopLossCalculator.calculate_stop_loss(100_000, 0)

        with pytest.raises(ValueError, match="Invalid ATR"):
            StopLossCalculator.calculate_stop_loss(100_000, -1_000)

    def test_calculate_stop_loss_result_validation(self):
        """Test stop loss result validation"""
        entry_price = 100_000
        atr = 2_000

        # Normal case should pass validation
        stop_loss, reason = StopLossCalculator.calculate_stop_loss(entry_price, atr)

        assert stop_loss > 0
        assert stop_loss < entry_price
        assert isinstance(stop_loss, float)
        assert isinstance(reason, str)

    def test_calculate_take_profit_targets_normal_case(self):
        """Test take profit targets calculation"""
        entry_price = 100_000
        atr = 2_000
        risk_reward_ratios = [1.5, 2.0, 3.0]

        targets = StopLossCalculator.calculate_take_profit_targets(
            entry_price, atr, risk_reward_ratios
        )

        expected_targets = [
            100_000 + (2_000 * 1.5),  # 103,000
            100_000 + (2_000 * 2.0),  # 104,000
            100_000 + (2_000 * 3.0),  # 106,000
        ]

        assert targets == expected_targets
        assert all(isinstance(t, float) for t in targets)

    def test_calculate_take_profit_targets_default_ratios(self):
        """Test take profit targets with default ratios"""
        entry_price = 100_000
        atr = 2_000

        targets = StopLossCalculator.calculate_take_profit_targets(entry_price, atr)

        # Should use default ratios [1.5, 3.0, 5.0]
        expected_targets = [
            100_000 + (2_000 * 1.5),  # 103,000
            100_000 + (2_000 * 3.0),  # 106,000
            100_000 + (2_000 * 5.0),  # 110,000
        ]

        assert targets == expected_targets

    def test_calculate_take_profit_targets_invalid_entry_price(self):
        """Test take profit targets with invalid entry price"""
        with pytest.raises(ValueError, match="Invalid entry_price"):
            StopLossCalculator.calculate_take_profit_targets(0, 2_000)

        with pytest.raises(ValueError, match="Invalid entry_price"):
            StopLossCalculator.calculate_take_profit_targets(-100, 2_000)

    def test_calculate_take_profit_targets_invalid_atr(self):
        """Test take profit targets with invalid ATR"""
        with pytest.raises(ValueError, match="Invalid ATR"):
            StopLossCalculator.calculate_take_profit_targets(100_000, 0)

        with pytest.raises(ValueError, match="Invalid ATR"):
            StopLossCalculator.calculate_take_profit_targets(100_000, -1_000)

    def test_calculate_take_profit_targets_empty_ratios(self):
        """Test take profit targets with empty ratios"""
        entry_price = 100_000
        atr = 2_000

        targets = StopLossCalculator.calculate_take_profit_targets(
            entry_price, atr, risk_reward_ratios=[]
        )

        assert targets == []


class TestSingletonInstances:
    """Test singleton instances"""

    def test_indicator_utils_singleton(self):
        """Test indicator_utils singleton"""
        assert isinstance(indicator_utils, IndicatorUtils)

        # Test that it works
        df = pd.DataFrame({"close": [100, 101, 102], "atr": [2.0, 2.1, 2.2]})

        result = indicator_utils.get_atr(df)
        assert result == 2.2

    def test_stop_loss_calculator_singleton(self):
        """Test stop_loss_calculator singleton"""
        assert isinstance(stop_loss_calculator, StopLossCalculator)

        # Test that it works
        stop_loss, reason = stop_loss_calculator.calculate_stop_loss(100_000, 2_000)

        assert isinstance(stop_loss, float)
        assert isinstance(reason, str)


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_complete_indicator_workflow(self):
        """Test complete workflow using all indicators"""
        # Create sample data
        df = pd.DataFrame(
            {
                "close": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
                "high": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "low": [96, 97, 98, 99, 100, 101, 102, 103, 104, 105],
                "atr": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
                "rsi": [30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
            }
        )

        # Get indicators
        atr = IndicatorUtils.get_atr(df)
        rsi = IndicatorUtils.get_rsi(df)
        support, resistance = IndicatorUtils.get_support_resistance(df, lookback=5)

        # Calculate stop loss and take profits
        entry_price = df["close"].iloc[-1]
        stop_loss, reason = StopLossCalculator.calculate_stop_loss(
            entry_price, atr, support
        )
        take_profits = StopLossCalculator.calculate_take_profit_targets(
            entry_price, atr
        )

        # Validate results
        assert atr == 2.9
        assert rsi == 75.0
        assert (
            support == 101.0
        )  # Min of last 5 lows (102,103,104,105,106 -> min is 102, but rolling window gives 101)
        assert resistance == 109.0  # Max of last 5 highs
        assert isinstance(stop_loss, float)
        assert stop_loss < entry_price
        assert len(take_profits) == 3
        assert all(tp > entry_price for tp in take_profits)

    def test_edge_case_workflow(self):
        """Test workflow with edge cases"""
        # Create minimal data
        df = pd.DataFrame({"close": [100, 101], "high": [102, 103], "low": [98, 99]})

        # Should handle gracefully
        atr = IndicatorUtils.get_atr(df, default_pct=0.02)
        rsi = IndicatorUtils.get_rsi(df, default=50.0)
        support, resistance = IndicatorUtils.get_support_resistance(df, lookback=10)

        assert atr == 101 * 0.02  # Fallback calculation
        assert rsi == 50.0  # Default value
        assert support is None  # Insufficient data
        assert resistance is None  # Insufficient data

        # Stop loss should still work
        stop_loss, reason = StopLossCalculator.calculate_stop_loss(101, atr, support)

        assert isinstance(stop_loss, float)
        assert stop_loss < 101
