# -*- coding: utf-8 -*-
"""
Tests for Portfolio Manager Circuit Breaker Integration

Tests:
1. Circuit breaker receives trade records on close_position
2. Per-symbol circuit breaker receives trade records
3. DCA stop loss validation
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import os


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with portfolio manager"""

    @patch("src.portfolio.manager.get_circuit_breaker")
    @patch("src.portfolio.manager.CIRCUIT_BREAKER_AVAILABLE", True)
    def test_close_position_records_to_circuit_breaker(self, mock_get_cb):
        """Test that close_position records trade to circuit breaker"""
        from src.portfolio.manager import PortfolioManager

        # Setup mock circuit breaker
        mock_cb = MagicMock()
        mock_cb.stats = {"consecutive_losses": 0}
        mock_get_cb.return_value = mock_cb

        # Setup mock database
        with patch.object(PortfolioManager, "__init__", lambda x: None):
            manager = PortfolioManager()
            manager._lock = MagicMock()
            manager.db = MagicMock()
            manager.monitor = MagicMock()
            manager.signal_tracker = MagicMock()
            manager.config = MagicMock()

            # Mock position data
            mock_position = {
                "shares": 100,
                "avg_price": 50000,
                "entry_date": datetime.now().isoformat(),
                "metadata": {},
            }
            manager.db.get_positions.return_value = {"VCB": mock_position}
            manager.db.transaction.return_value.__enter__ = MagicMock()
            manager.db.transaction.return_value.__exit__ = MagicMock()

            # Close position with profit
            manager.close_position("VCB", 55000, "Test exit")

            # Verify circuit breaker was called
            mock_cb.record_trade.assert_called_once()
            call_args = mock_cb.record_trade.call_args[0]
            pnl = call_args[0]
            assert pnl > 0  # Should be profit (55000 - 50000) * 100

    @patch("src.portfolio.manager.get_per_symbol_circuit_breaker")
    @patch("src.portfolio.manager.PER_SYMBOL_CB_AVAILABLE", True)
    def test_close_position_records_to_per_symbol_cb(self, mock_get_per_symbol_cb):
        """Test that close_position records to per-symbol circuit breaker"""
        from src.portfolio.manager import PortfolioManager

        # Setup mock per-symbol circuit breaker
        mock_per_symbol_cb = MagicMock()
        mock_get_per_symbol_cb.return_value = mock_per_symbol_cb

        # Setup mock database
        with patch.object(PortfolioManager, "__init__", lambda x: None):
            manager = PortfolioManager()
            manager._lock = MagicMock()
            manager.db = MagicMock()
            manager.monitor = MagicMock()
            manager.signal_tracker = MagicMock()
            manager.config = MagicMock()

            # Mock position data (losing trade)
            mock_position = {
                "shares": 100,
                "avg_price": 50000,
                "entry_date": datetime.now().isoformat(),
                "metadata": {},
            }
            manager.db.get_positions.return_value = {"HPG": mock_position}
            manager.db.transaction.return_value.__enter__ = MagicMock()
            manager.db.transaction.return_value.__exit__ = MagicMock()

            # Close position with loss
            manager.close_position("HPG", 45000, "Stop loss")

            # Verify per-symbol circuit breaker was called
            mock_per_symbol_cb.record_trade.assert_called_once()
            call_args = mock_per_symbol_cb.record_trade.call_args
            assert call_args[0][0] == "HPG"  # symbol
            assert call_args[0][1] is False  # is_win (loss)
            assert call_args[0][2] < 0  # pnl_percent (negative)


class TestDCAStopLossValidation:
    """Test DCA stop loss validation improvements"""

    def test_dca_stop_loss_recalculated_when_above_new_avg(self):
        """Test stop loss is recalculated when old stop >= new avg price"""
        from src.portfolio.manager import PortfolioManager

        with patch.object(PortfolioManager, "__init__", lambda x: None):
            manager = PortfolioManager()
            manager._lock = MagicMock()
            manager.db = MagicMock()
            manager.db.transaction.return_value.__enter__ = MagicMock()
            manager.db.transaction.return_value.__exit__ = MagicMock()

            # Existing position: bought at 50000, stop at 46500 (-7%)
            existing_pos = {
                "shares": 100,
                "avg_price": 50000,
                "entry_date": datetime.now().isoformat(),
                "stop_loss": 46500,  # -7% of 50000
                "take_profit": 57500,
                "metadata": {},
            }

            # DCA at lower price: 40000
            # New avg = (100*50000 + 100*40000) / 200 = 45000
            # Old stop (46500) > new avg (45000) - INVALID!
            manager._average_up_position(
                symbol="TEST",
                existing_pos=existing_pos,
                shares_to_add=100,
                price_to_add=40000,
                metadata={},
            )

            # Verify save_position was called
            save_call = manager.db.save_position.call_args
            saved_stop_loss = save_call[1]["stop_loss"]
            saved_avg_price = save_call[1]["avg_price"]

            # Stop loss must be below new avg price
            assert (
                saved_stop_loss < saved_avg_price
            ), f"Stop loss {saved_stop_loss} should be < avg price {saved_avg_price}"

    def test_dca_stop_loss_kept_when_valid(self):
        """Test stop loss is kept when still valid after DCA"""
        from src.portfolio.manager import PortfolioManager

        with patch.object(PortfolioManager, "__init__", lambda x: None):
            manager = PortfolioManager()
            manager._lock = MagicMock()
            manager.db = MagicMock()
            manager.db.transaction.return_value.__enter__ = MagicMock()
            manager.db.transaction.return_value.__exit__ = MagicMock()

            # Existing position: bought at 50000, stop at 46500 (-7%)
            existing_pos = {
                "shares": 100,
                "avg_price": 50000,
                "entry_date": datetime.now().isoformat(),
                "stop_loss": 46500,
                "take_profit": 57500,
                "metadata": {},
            }

            # DCA at higher price: 52000
            # New avg = (100*50000 + 100*52000) / 200 = 51000
            # Old stop (46500) < new avg (51000) - VALID
            # New calculated stop = 51000 * 0.93 = 47430
            # Old stop (46500) < new stop (47430) - use new stop
            manager._average_up_position(
                symbol="TEST",
                existing_pos=existing_pos,
                shares_to_add=100,
                price_to_add=52000,
                metadata={},
            )

            # Verify save_position was called
            save_call = manager.db.save_position.call_args
            saved_stop_loss = save_call[1]["stop_loss"]
            saved_avg_price = save_call[1]["avg_price"]

            # Stop loss must be below new avg price
            assert saved_stop_loss < saved_avg_price
            # New avg should be 51000
            assert saved_avg_price == 51000


class TestReducePositionCircuitBreaker:
    """Test reduce_position also records to circuit breakers"""

    @patch("src.portfolio.manager.get_circuit_breaker")
    @patch("src.portfolio.manager.get_per_symbol_circuit_breaker")
    @patch("src.portfolio.manager.CIRCUIT_BREAKER_AVAILABLE", True)
    @patch("src.portfolio.manager.PER_SYMBOL_CB_AVAILABLE", True)
    def test_reduce_position_records_to_circuit_breakers(self, mock_get_per_symbol_cb, mock_get_cb):
        """Test that reduce_position records to both circuit breakers"""
        from src.portfolio.manager import PortfolioManager

        # Setup mocks
        mock_cb = MagicMock()
        mock_cb.stats = {"consecutive_losses": 0}
        mock_get_cb.return_value = mock_cb

        mock_per_symbol_cb = MagicMock()
        mock_get_per_symbol_cb.return_value = mock_per_symbol_cb

        with patch.object(PortfolioManager, "__init__", lambda x: None):
            manager = PortfolioManager()
            manager._lock = MagicMock()
            manager.db = MagicMock()
            manager.monitor = MagicMock()
            manager.signal_tracker = MagicMock()
            manager.config = MagicMock()

            # Mock position data
            mock_position = {
                "shares": 200,
                "avg_price": 50000,
                "entry_date": datetime.now().isoformat(),
                "stop_loss": 46500,
                "take_profit": 57500,
                "metadata": {},
            }
            manager.db.get_positions.return_value = {"VNM": mock_position}
            manager.db.transaction.return_value.__enter__ = MagicMock()
            manager.db.transaction.return_value.__exit__ = MagicMock()

            # Reduce position (partial sell with profit)
            manager.reduce_position("VNM", 100, 55000, "Take profit 50%")

            # Verify both circuit breakers were called
            mock_cb.record_trade.assert_called_once()
            mock_per_symbol_cb.record_trade.assert_called_once()

            # Verify per-symbol CB received correct data
            ps_call_args = mock_per_symbol_cb.record_trade.call_args[0]
            assert ps_call_args[0] == "VNM"  # symbol
            assert ps_call_args[1] is True  # is_win (profit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
