"""
Unit tests for WebSocket Module
"""

import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from websocket.live_portfolio import LivePortfolioMonitor, PositionAlert
from websocket.price_feed import PriceFeedClient, PriceUpdate, WebSocketConfig


class TestPriceFeedClient:
    """Test PriceFeedClient class"""

    def test_client_initialization(self):
        """Test client initializes correctly"""
        config = WebSocketConfig(
            url="wss://test.example.com", reconnect_delay=5, max_reconnect_attempts=3
        )

        client = PriceFeedClient(config)

        assert client.config.url == "wss://test.example.com"
        assert client.config.reconnect_delay == 5
        assert client.running is False
        assert client.connected is False
        assert len(client.prices) == 0

    def test_subscribe_symbols(self):
        """Test subscribing to symbols"""
        client = PriceFeedClient()

        client.subscribe(["VCB", "HPG"])

        assert "VCB" in client.config.subscribe_channels
        assert "HPG" in client.config.subscribe_channels
        assert len(client.config.subscribe_channels) == 2

    def test_subscribe_single_symbol(self):
        """Test subscribing to single symbol"""
        client = PriceFeedClient()

        client.subscribe("VCB")

        assert "VCB" in client.config.subscribe_channels

    def test_unsubscribe_symbols(self):
        """Test unsubscribing from symbols"""
        client = PriceFeedClient()

        client.subscribe(["VCB", "HPG", "VHM"])
        client.unsubscribe(["HPG"])

        assert "VCB" in client.config.subscribe_channels
        assert "VHM" in client.config.subscribe_channels
        assert "HPG" not in client.config.subscribe_channels

    def test_add_price_callback(self):
        """Test adding price callback"""
        client = PriceFeedClient()

        def test_callback(update):
            pass

        client.add_price_callback(test_callback)

        assert len(client.price_callbacks) == 1
        assert test_callback in client.price_callbacks

    def test_process_price_update(self):
        """Test processing price update"""
        client = PriceFeedClient()

        # Mock price data (SSI format)
        data = {
            "sym": "VCB",
            "lastPrice": 65000,
            "totalVol": 1000000,
            "change": 1000,
            "changePc": 1.56,
            "bid1": 64900,
            "ask1": 65100,
            "high": 66000,
            "low": 64000,
        }

        client.process_price_update(data)

        assert "VCB" in client.prices
        update = client.prices["VCB"]
        assert update.symbol == "VCB"
        assert update.price == 65000
        assert update.volume == 1000000
        assert update.change == 1000
        assert update.change_percent == 1.56

    def test_get_price(self):
        """Test getting cached price"""
        client = PriceFeedClient()

        # Add price to cache
        update = PriceUpdate(
            symbol="VCB",
            price=65000,
            volume=1000000,
            change=1000,
            change_percent=1.56,
            timestamp=datetime.now(),
        )
        client.prices["VCB"] = update

        # Get price
        cached_price = client.get_price("VCB")

        assert cached_price is not None
        assert cached_price.symbol == "VCB"
        assert cached_price.price == 65000

    def test_get_price_not_found(self):
        """Test getting price for unknown symbol"""
        client = PriceFeedClient()

        price = client.get_price("UNKNOWN")

        assert price is None


class TestLivePortfolioMonitor:
    """Test LivePortfolioMonitor class"""

    def setup_method(self):
        """Setup for each test"""
        # Mock portfolio manager
        self.mock_portfolio = Mock()
        self.mock_portfolio.get_positions.return_value = {
            "VCB": {
                "avg_price": 60000,
                "shares": 100,
                "entry_date": datetime.now(),
                "stop_loss": 57000,
                "take_profit": 66000,
            }
        }

    def test_monitor_initialization(self):
        """Test monitor initializes correctly"""
        monitor = LivePortfolioMonitor(
            portfolio_manager=self.mock_portfolio, enable_alerts=True
        )

        assert monitor.portfolio is not None
        assert monitor.price_feed is not None
        assert monitor.enable_alerts is True
        assert len(monitor.alert_callbacks) == 0

    def test_add_alert_callback(self):
        """Test adding alert callback"""
        monitor = LivePortfolioMonitor(portfolio_manager=self.mock_portfolio)

        def test_callback(alert):
            pass

        monitor.add_alert_callback(test_callback)

        assert len(monitor.alert_callbacks) == 1

    def test_send_alert(self):
        """Test sending alerts"""
        monitor = LivePortfolioMonitor(portfolio_manager=self.mock_portfolio)

        # Track if callback was called
        callback_called = [False]

        def test_callback(alert):
            callback_called[0] = True
            assert alert.symbol == "VCB"
            assert alert.alert_type == "STOP_LOSS"

        monitor.add_alert_callback(test_callback)

        # Create and send alert
        alert = PositionAlert(
            symbol="VCB",
            alert_type="STOP_LOSS",
            message="Test alert",
            current_price=56000,
            position_pnl=-400000,
            timestamp=datetime.now(),
            urgency=5,
        )

        monitor.send_alert(alert)

        assert callback_called[0] is True

    def test_alerts_disabled(self):
        """Test alerts can be disabled"""
        monitor = LivePortfolioMonitor(
            portfolio_manager=self.mock_portfolio, enable_alerts=False
        )

        callback_called = [False]

        def test_callback(alert):
            callback_called[0] = True

        monitor.add_alert_callback(test_callback)

        # Create alert
        alert = PositionAlert(
            symbol="VCB",
            alert_type="TEST",
            message="Test",
            current_price=60000,
            position_pnl=0,
            timestamp=datetime.now(),
            urgency=1,
        )

        monitor.send_alert(alert)

        # Callback should not be called when alerts disabled
        assert callback_called[0] is False


class TestPriceUpdate:
    """Test PriceUpdate dataclass"""

    def test_price_update_creation(self):
        """Test creating PriceUpdate"""
        update = PriceUpdate(
            symbol="VCB",
            price=65000,
            volume=1000000,
            change=1000,
            change_percent=1.56,
            timestamp=datetime.now(),
            bid=64900,
            ask=65100,
        )

        assert update.symbol == "VCB"
        assert update.price == 65000
        assert update.volume == 1000000
        assert update.bid == 64900
        assert update.ask == 65100


class TestPositionAlert:
    """Test PositionAlert dataclass"""

    def test_position_alert_creation(self):
        """Test creating PositionAlert"""
        alert = PositionAlert(
            symbol="VCB",
            alert_type="TAKE_PROFIT",
            message="Target reached",
            current_price=66000,
            position_pnl=600000,
            timestamp=datetime.now(),
            urgency=4,
        )

        assert alert.symbol == "VCB"
        assert alert.alert_type == "TAKE_PROFIT"
        assert alert.urgency == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
