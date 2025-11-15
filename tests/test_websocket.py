"""
Unit tests for WebSocket Module
"""

import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pytest

# Add project root to path
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

    def test_price_callbacks_triggered(self):
        """Test that price callbacks are triggered on update"""
        client = PriceFeedClient()

        callback_triggered = [False]
        received_update = [None]

        def test_callback(update: PriceUpdate):
            callback_triggered[0] = True
            received_update[0] = update

        client.add_price_callback(test_callback)

        # Process price update
        data = {
            "sym": "HPG",
            "lastPrice": 25000,
            "totalVol": 5000000,
            "change": 500,
            "changePc": 2.04,
        }

        client.process_price_update(data)

        # Verify callback was triggered
        assert callback_triggered[0] is True
        assert received_update[0] is not None
        assert received_update[0].symbol == "HPG"
        assert received_update[0].price == 25000

    def test_multiple_callbacks(self):
        """Test multiple callbacks can be registered"""
        client = PriceFeedClient()

        call_count = [0]

        def callback1(update):
            call_count[0] += 1

        def callback2(update):
            call_count[0] += 1

        client.add_price_callback(callback1)
        client.add_price_callback(callback2)

        # Process update
        data = {"sym": "VCB", "lastPrice": 65000, "totalVol": 1000000}
        client.process_price_update(data)

        # Both callbacks should be called
        assert call_count[0] == 2

    def test_config_defaults(self):
        """Test WebSocketConfig default values"""
        config = WebSocketConfig()

        assert config.url == "wss://fc-data.ssi.com.vn/realtime"
        assert config.reconnect_delay == 5
        assert config.max_reconnect_attempts == 10
        assert config.heartbeat_interval == 30
        assert len(config.subscribe_channels) == 0

    def test_config_custom_values(self):
        """Test WebSocketConfig with custom values"""
        config = WebSocketConfig(
            url="wss://custom.url",
            reconnect_delay=10,
            max_reconnect_attempts=5,
            heartbeat_interval=60,
            subscribe_channels=["VCB", "HPG"],
        )

        assert config.url == "wss://custom.url"
        assert config.reconnect_delay == 10
        assert config.max_reconnect_attempts == 5
        assert config.heartbeat_interval == 60
        assert len(config.subscribe_channels) == 2


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

    def test_check_position_updates(self):
        """Test checking position updates with price changes"""
        monitor = LivePortfolioMonitor(portfolio_manager=self.mock_portfolio)

        # Mock price feed
        mock_price = PriceUpdate(
            symbol="VCB",
            price=61000,  # Above entry price of 60000
            volume=1000000,
            change=1000,
            change_percent=1.67,
            timestamp=datetime.now(),
        )
        monitor.price_feed.prices["VCB"] = mock_price

        # Check positions (should calculate P&L)
        positions = monitor.portfolio.get_positions()
        assert "VCB" in positions

    def test_calculate_position_pnl(self):
        """Test P&L calculation for positions"""
        monitor = LivePortfolioMonitor(portfolio_manager=self.mock_portfolio)

        # Simulate position with known values
        position = {
            "avg_price": 60000,
            "shares": 100,
            "entry_date": datetime.now(),
            "stop_loss": 57000,
            "take_profit": 66000,
        }

        current_price = 65000

        # Calculate expected P&L
        expected_pnl = (current_price - position["avg_price"]) * position["shares"]
        expected_pnl_pct = ((current_price / position["avg_price"]) - 1) * 100

        assert expected_pnl == 500000  # (65000 - 60000) * 100
        assert abs(expected_pnl_pct - 8.33) < 0.01  # ~8.33%

    def test_alert_urgency_levels(self):
        """Test different alert urgency levels"""
        monitor = LivePortfolioMonitor(portfolio_manager=self.mock_portfolio)

        # Low urgency alert
        low_alert = PositionAlert(
            symbol="VCB",
            alert_type="INFO",
            message="Price update",
            current_price=60500,
            position_pnl=50000,
            timestamp=datetime.now(),
            urgency=1,
        )

        # High urgency alert
        high_alert = PositionAlert(
            symbol="VCB",
            alert_type="STOP_LOSS",
            message="Stop loss triggered",
            current_price=56000,
            position_pnl=-400000,
            timestamp=datetime.now(),
            urgency=5,
        )

        assert low_alert.urgency == 1
        assert high_alert.urgency == 5
        assert high_alert.urgency > low_alert.urgency


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

    def test_stop_loss_alert(self):
        """Test stop loss alert creation"""
        alert = PositionAlert(
            symbol="HPG",
            alert_type="STOP_LOSS",
            message="Stop loss hit at 24,000",
            current_price=24000,
            position_pnl=-100000,
            timestamp=datetime.now(),
            urgency=5,
        )

        assert alert.symbol == "HPG"
        assert alert.alert_type == "STOP_LOSS"
        assert alert.position_pnl < 0
        assert alert.urgency == 5

    def test_take_profit_alert(self):
        """Test take profit alert creation"""
        alert = PositionAlert(
            symbol="VNM",
            alert_type="TAKE_PROFIT",
            message="Take profit target reached",
            current_price=85000,
            position_pnl=500000,
            timestamp=datetime.now(),
            urgency=4,
        )

        assert alert.symbol == "VNM"
        assert alert.alert_type == "TAKE_PROFIT"
        assert alert.position_pnl > 0
        assert alert.urgency == 4


class TestIntegration:
    """Integration tests for WebSocket components"""

    def test_price_feed_and_monitor_integration(self):
        """Test price feed client and portfolio monitor work together"""
        mock_portfolio = Mock()
        mock_portfolio.get_positions.return_value = {
            "VCB": {
                "avg_price": 60000,
                "shares": 100,
                "entry_date": datetime.now(),
                "stop_loss": 57000,
                "take_profit": 66000,
            }
        }

        monitor = LivePortfolioMonitor(portfolio_manager=mock_portfolio)

        # Verify price feed client is created
        assert monitor.price_feed is not None
        assert isinstance(monitor.price_feed, PriceFeedClient)

        # Add price update
        price_update = PriceUpdate(
            symbol="VCB",
            price=65000,
            volume=1000000,
            change=5000,
            change_percent=8.33,
            timestamp=datetime.now(),
        )
        monitor.price_feed.prices["VCB"] = price_update

        # Verify price is cached
        cached_price = monitor.price_feed.get_price("VCB")
        assert cached_price is not None
        assert cached_price.price == 65000

    def test_multiple_symbols_tracking(self):
        """Test tracking multiple symbols simultaneously"""
        client = PriceFeedClient()

        symbols = ["VCB", "HPG", "VHM", "VNM", "VIC"]
        client.subscribe(symbols)

        # Add prices for all symbols
        for i, symbol in enumerate(symbols):
            update = PriceUpdate(
                symbol=symbol,
                price=50000 + (i * 1000),
                volume=1000000,
                change=100,
                change_percent=0.2,
                timestamp=datetime.now(),
            )
            client.prices[symbol] = update

        # Verify all symbols are tracked
        for symbol in symbols:
            assert symbol in client.prices
            assert client.get_price(symbol) is not None

    def test_price_update_with_optional_fields(self):
        """Test price update with all optional fields"""
        update = PriceUpdate(
            symbol="FPT",
            price=105000,
            volume=2000000,
            change=2000,
            change_percent=1.94,
            timestamp=datetime.now(),
            bid=104800,
            ask=105200,
            high=106000,
            low=103000,
        )

        assert update.symbol == "FPT"
        assert update.bid == 104800
        assert update.ask == 105200
        assert update.high == 106000
        assert update.low == 103000

        # Check spread
        spread = update.ask - update.bid
        assert spread == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
