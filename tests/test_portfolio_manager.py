"""
Unit tests for Portfolio Manager
"""
import pytest
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPortfolioManager:
    """Test PortfolioManager class"""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Setup test database"""
        from database import TradingDB
        import database

        # Create fresh in-memory database for each test
        # :memory: database is always fresh, no need to delete data
        self.db = TradingDB(':memory:')
        self.db.create_tables()

        # Use monkeypatch to replace get_db
        monkeypatch.setattr(database, 'get_db', lambda: self.db)

        yield

    def test_add_position_valid(self):
        """Test adding valid position"""
        from portfolio_manager import PortfolioManager

        manager = PortfolioManager()

        # Should not raise exception
        manager.add_position(
            symbol='VCB',
            shares=100,
            entry_price=60000,
            stop_loss=57000,
            take_profit=66000
        )

        # Verify position was added
        positions = manager.get_positions()
        assert 'VCB' in positions
        assert positions['VCB']['shares'] == 100
        assert positions['VCB']['avg_price'] == 60000

    def test_add_position_invalid_symbol(self):
        """Test adding position with invalid symbol"""
        from portfolio_manager import PortfolioManager
        from exceptions import PortfolioError

        manager = PortfolioManager()

        # Empty symbol
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('', 100, 60000)
        assert 'Symbol must be a non-empty string' in str(exc_info.value)

        # Non-alphabetic symbol
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('VCB123', 100, 60000)
        assert 'Invalid symbol format' in str(exc_info.value)

        # Too long symbol
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('VERYLONGSYMBOL', 100, 60000)
        assert 'Invalid symbol format' in str(exc_info.value)

    def test_add_position_invalid_shares(self):
        """Test adding position with invalid shares"""
        from portfolio_manager import PortfolioManager
        from exceptions import PortfolioError

        manager = PortfolioManager()

        # Zero shares
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('VCB', 0, 60000)
        assert 'Shares must be a positive integer' in str(exc_info.value)

        # Negative shares
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('VCB', -100, 60000)
        assert 'Shares must be a positive integer' in str(exc_info.value)

        # Float shares
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('VCB', 100.5, 60000)
        assert 'Shares must be a positive integer' in str(exc_info.value)

    def test_add_position_invalid_price(self):
        """Test adding position with invalid price"""
        from portfolio_manager import PortfolioManager
        from exceptions import PortfolioError

        manager = PortfolioManager()

        # Zero price
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('VCB', 100, 0)
        assert 'Entry price must be a positive number' in str(exc_info.value)

        # Negative price
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('VCB', 100, -60000)
        assert 'Entry price must be a positive number' in str(exc_info.value)

    def test_add_position_invalid_stop_loss(self):
        """Test adding position with invalid stop loss"""
        from portfolio_manager import PortfolioManager
        from exceptions import PortfolioError

        manager = PortfolioManager()

        # Stop loss >= entry price
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('VCB', 100, 60000, stop_loss=60000)
        assert 'Stop loss must be below entry price' in str(exc_info.value)

        # Stop loss > entry price
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('VCB', 100, 60000, stop_loss=65000)
        assert 'Stop loss must be below entry price' in str(exc_info.value)

    def test_add_position_invalid_take_profit(self):
        """Test adding position with invalid take profit"""
        from portfolio_manager import PortfolioManager
        from exceptions import PortfolioError

        manager = PortfolioManager()

        # Take profit <= entry price
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('VCB', 100, 60000, take_profit=60000)
        assert 'Take profit must be above entry price' in str(exc_info.value)

        # Take profit < entry price
        with pytest.raises(PortfolioError) as exc_info:
            manager.add_position('VCB', 100, 60000, take_profit=55000)
        assert 'Take profit must be above entry price' in str(exc_info.value)

    def test_get_portfolio_value(self):
        """Test calculating portfolio value"""
        from portfolio_manager import PortfolioManager

        manager = PortfolioManager()

        # Add positions
        manager.add_position('VCB', 100, 60000)
        manager.add_position('HPG', 200, 25000)

        portfolio = manager.get_portfolio_value()

        assert portfolio['total_cost'] == 60000 * 100 + 25000 * 200
        assert portfolio['num_positions'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
