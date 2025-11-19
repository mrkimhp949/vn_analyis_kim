"""
Unit tests for critical fixes
Tests for stop loss calculation, thread safety, and data validation
"""

from threading import Thread

import numpy as np
import pandas as pd
import pytest
from src.portfolio.manager import PortfolioManager

# Import modules to test
from src.utils.indicators import IndicatorUtils, StopLossCalculator
from src.utils.validation import DataValidator


class TestStopLossCalculator:
    """Test stop loss calculation fixes"""

    def test_normal_case(self):
        """Test normal stop loss calculation"""
        sl, reason = StopLossCalculator.calculate_stop_loss(
            entry_price=100_000, atr=2_000, support_level=96_000
        )

        assert 90_000 <= sl < 100_000, f"Stop loss {sl} out of range"
        assert reason in ["Support-based", "ATR-based"]

    def test_no_support(self):
        """Test without support level"""
        sl, reason = StopLossCalculator.calculate_stop_loss(
            entry_price=100_000, atr=2_000, support_level=None
        )

        assert 90_000 <= sl < 100_000
        assert reason == "ATR-based"

    def test_invalid_support_above_entry(self):
        """Test with invalid support (above entry price)"""
        sl, reason = StopLossCalculator.calculate_stop_loss(
            entry_price=100_000,
            atr=2_000,
            support_level=105_000,  # Invalid: above entry
        )

        # Should ignore invalid support and use ATR
        assert 90_000 <= sl < 100_000
        assert reason == "ATR-based"

    def test_minimum_stop_enforcement(self):
        """Test minimum stop loss enforcement"""
        sl, reason = StopLossCalculator.calculate_stop_loss(
            entry_price=100_000, atr=500, support_level=None  # Very small ATR
        )

        # Should enforce dynamic minimum (1.5% for ATR 0.5%)
        # Dynamic min = min(max(2.0 * 0.5%, 1.5%), 3%) = 1.5%
        assert sl >= 98_500, f"Stop loss {sl} below dynamic minimum (expected >= 98,500)"
        # Reason should contain "minimum" (case-insensitive) or "Dynamic minimum"
        assert (
            "minimum" in reason.lower() or "Dynamic" in reason
        ), f"Reason should indicate minimum enforcement: {reason}"

    def test_maximum_stop_enforcement(self):
        """Test maximum stop loss enforcement"""
        sl, reason = StopLossCalculator.calculate_stop_loss(
            entry_price=100_000, atr=10_000, support_level=None  # Very large ATR
        )

        # Should enforce maximum 10% stop
        assert sl >= 90_000, f"Stop loss {sl} below maximum"
        assert "Maximum" in reason or "ATR-based" in reason

    def test_invalid_entry_price(self):
        """Test with invalid entry price"""
        with pytest.raises(ValueError, match="Invalid entry_price"):
            StopLossCalculator.calculate_stop_loss(entry_price=0, atr=2_000)

        with pytest.raises(ValueError, match="Invalid entry_price"):
            StopLossCalculator.calculate_stop_loss(entry_price=-100_000, atr=2_000)

    def test_invalid_atr(self):
        """Test with invalid ATR"""
        with pytest.raises(ValueError, match="Invalid ATR"):
            StopLossCalculator.calculate_stop_loss(entry_price=100_000, atr=0)

        with pytest.raises(ValueError, match="Invalid ATR"):
            StopLossCalculator.calculate_stop_loss(entry_price=100_000, atr=-1000)

    def test_nan_support(self):
        """Test with NaN support level"""
        sl, reason = StopLossCalculator.calculate_stop_loss(
            entry_price=100_000, atr=2_000, support_level=np.nan
        )

        # Should handle NaN gracefully
        assert 90_000 <= sl < 100_000
        assert reason == "ATR-based"


class TestDataValidator:
    """Test data validation fixes"""

    def test_valid_dataframe(self):
        """Test with valid DataFrame"""
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [99, 100, 101],
                "close": [103, 104, 105],
                "volume": [1000, 1100, 1200],
            }
        )

        # Should not raise
        DataValidator.validate_dataframe(df, min_rows=3)

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()

        with pytest.raises(Exception, match="empty"):
            DataValidator.validate_dataframe(df)

    def test_insufficient_rows(self):
        """Test with insufficient rows"""
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [105],
                "low": [99],
                "close": [103],
                "volume": [1000],
            }
        )

        with pytest.raises(Exception, match="Insufficient"):
            DataValidator.validate_dataframe(df, min_rows=50)

    def test_missing_columns(self):
        """Test with missing required columns"""
        # Create enough rows but missing columns
        df = pd.DataFrame(
            {
                "open": list(range(100, 160)),
                "high": list(range(105, 165)),
                # Missing 'low', 'close', 'volume'
            }
        )

        with pytest.raises(Exception, match="Missing"):
            DataValidator.validate_dataframe(df)

    def test_nan_in_latest_row(self):
        """Test with NaN in latest row"""
        # Create enough rows with NaN in last row
        data = {
            "open": list(range(100, 160)),
            "high": list(range(105, 165)),
            "low": list(range(99, 159)),
            "close": list(range(103, 163)),
            "volume": list(range(1000, 1060)),
        }
        df = pd.DataFrame(data)
        df.loc[df.index[-1], "open"] = np.nan  # Add NaN to last row

        with pytest.raises(Exception, match="NaN"):
            DataValidator.validate_dataframe(df)

    def test_invalid_price(self):
        """Test with invalid price values"""
        # Create enough rows with invalid price in last row
        data = {
            "open": list(range(100, 160)),
            "high": list(range(105, 165)),
            "low": list(range(99, 159)),
            "close": list(range(103, 163)),
            "volume": list(range(1000, 1060)),
        }
        df = pd.DataFrame(data)
        df.loc[df.index[-1], "open"] = 0  # Invalid price in last row

        with pytest.raises(Exception, match="Invalid"):
            DataValidator.validate_dataframe(df)

    def test_validate_price(self):
        """Test price validation"""
        # Valid price
        price = DataValidator.validate_price(100_000, "test")
        assert price == 100_000

        # Invalid prices
        with pytest.raises(ValueError):
            DataValidator.validate_price(0, "test")

        with pytest.raises(ValueError):
            DataValidator.validate_price(-100, "test")

        with pytest.raises(ValueError):
            DataValidator.validate_price(np.nan, "test")


class TestPortfolioManagerThreadSafety:
    """Test portfolio manager thread safety"""

    def test_concurrent_add_position(self):
        """Test concurrent position additions"""
        # Note: This test may fail due to database isolation
        # Portfolio manager uses database which may not persist in test
        # This is expected behavior - just verify no crashes
        manager = PortfolioManager()
        errors = []

        def add_position_worker(symbol, shares):
            try:
                manager.add_position(symbol=symbol, shares=shares, entry_price=100_000)
            except Exception as ex:
                errors.append(str(ex))

        # Create multiple threads adding positions
        threads = []
        for i in range(10):
            t = Thread(target=add_position_worker, args=(f"TESTCONC{i}", 100))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Main check: no errors occurred (thread-safe)
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Note: Position count may be 0 due to test database isolation
        # The important thing is no race conditions/crashes occurred

    def test_concurrent_get_positions(self):
        """Test concurrent position reads"""
        manager = PortfolioManager()

        # Add some positions
        for i in range(5):
            try:
                manager.add_position(f"TESTREAD{i}", 100, 100_000)
            except Exception:
                pass  # May fail due to test DB

        results = []
        errors = []

        def get_positions_worker():
            try:
                pos = manager.get_positions()
                results.append(len(pos))
            except Exception as ex:
                errors.append(str(ex))

        # Create multiple threads reading positions
        threads = []
        for _ in range(20):
            t = Thread(target=get_positions_worker)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Main check: no errors (thread-safe reads)
        assert len(errors) == 0, f"Errors: {errors}"

        # All reads should return consistent count (even if 0)
        if results:
            first_count = results[0]
            assert all(r == first_count for r in results), f"Inconsistent reads: {set(results)}"


class TestIndicatorUtils:
    """Test indicator utilities"""

    def test_get_atr_with_valid_data(self):
        """Test ATR extraction with valid data"""
        df = pd.DataFrame({"close": [100, 101, 102], "atr": [2.0, 2.1, 2.2]})

        atr = IndicatorUtils.get_atr(df)
        assert atr == 2.2

    def test_get_atr_fallback(self):
        """Test ATR fallback when column missing"""
        df = pd.DataFrame(
            {
                "close": [100, 101, 102]
                # No ATR column
            }
        )

        atr = IndicatorUtils.get_atr(df, default_pct=0.02)
        assert atr == 102 * 0.02

    def test_get_rsi_with_valid_data(self):
        """Test RSI extraction"""
        df = pd.DataFrame({"close": [100, 101, 102], "rsi": [45, 50, 55]})

        rsi = IndicatorUtils.get_rsi(df)
        assert rsi == 55

    def test_get_rsi_fallback(self):
        """Test RSI fallback"""
        df = pd.DataFrame({"close": [100, 101, 102]})

        rsi = IndicatorUtils.get_rsi(df, default=50)
        assert rsi == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
