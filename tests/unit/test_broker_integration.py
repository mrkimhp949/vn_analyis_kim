# -*- coding: utf-8 -*-
"""
Tests for Broker Integration

Tests:
- SSI Broker
- VNDirect Broker
- Broker Factory
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.broker import (
    BaseBroker,
    SimulatedBroker,
    Order,
    Position,
    AccountInfo,
    OrderSide,
    OrderType,
    OrderStatus,
    get_paper_broker,
    SSIBroker,
    VNDirectBroker,
    get_broker,
)


class TestSimulatedBroker:
    """Tests for SimulatedBroker"""

    def test_init(self):
        """Test broker initialization"""
        broker = SimulatedBroker(initial_cash=50_000_000)
        assert broker.broker_name == "SIMULATED"
        assert broker._cash == 50_000_000

    def test_connect(self):
        """Test connection"""
        broker = SimulatedBroker()
        assert broker.connect() is True
        assert broker._connected is True

    def test_get_account_info(self):
        """Test account info"""
        broker = SimulatedBroker(initial_cash=100_000_000)
        broker.connect()

        info = broker.get_account_info()
        assert info is not None
        assert info.cash_balance == 100_000_000
        assert info.broker == "SIMULATED"

    def test_place_buy_order(self):
        """Test placing buy order"""
        broker = SimulatedBroker(initial_cash=100_000_000)
        broker.connect()

        order = broker.buy("VNM", 100, 85000)

        assert order is not None
        assert order.symbol == "VNM"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.status == OrderStatus.FILLED

    def test_place_sell_order(self):
        """Test placing sell order"""
        broker = SimulatedBroker(initial_cash=100_000_000)
        broker.connect()

        # First buy
        broker.buy("VNM", 100, 85000)

        # Then sell
        order = broker.sell("VNM", 100, 86000)

        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.status == OrderStatus.FILLED

    def test_invalid_lot_size(self):
        """Test invalid lot size rejection"""
        broker = SimulatedBroker()
        broker.connect()

        order = broker.buy("VNM", 50, 85000)  # Not multiple of 100
        assert order is None

    def test_insufficient_buying_power(self):
        """Test insufficient buying power"""
        broker = SimulatedBroker(initial_cash=1_000_000)
        broker.connect()

        order = broker.buy("VNM", 1000, 85000)  # 85M > 1M
        assert order is None

    def test_get_positions(self):
        """Test getting positions"""
        broker = SimulatedBroker(initial_cash=100_000_000)
        broker.connect()

        broker.buy("VNM", 100, 85000)
        broker.buy("FPT", 200, 90000)

        positions = broker.get_positions()
        assert len(positions) == 2

        vnm_pos = broker.get_position("VNM")
        assert vnm_pos is not None
        assert vnm_pos.quantity == 100

    def test_cancel_order(self):
        """Test order cancellation"""
        broker = SimulatedBroker()
        broker.connect()

        # Note: SimulatedBroker fills immediately, so cancel won't work
        # This tests the cancel logic path
        result = broker.cancel_order("INVALID_ID")
        assert result is False


class TestSSIBroker:
    """Tests for SSI Broker"""

    def test_init(self):
        """Test SSI broker initialization"""
        broker = SSIBroker(
            account_id="TEST123",
            consumer_id="test_id",
            consumer_secret="test_secret",
            is_paper=True,
        )

        assert broker.account_id == "TEST123"
        assert broker.broker_name == "SSI_PAPER"
        assert broker.is_paper is True

    @patch("requests.Session.post")
    def test_connect_success(self, mock_post):
        """Test successful connection"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 200, "data": {"accessToken": "test_token"}}
        mock_post.return_value = mock_response

        broker = SSIBroker(
            account_id="TEST123", consumer_id="test_id", consumer_secret="test_secret"
        )

        result = broker.connect()
        assert result is True
        assert broker._connected is True

    @patch("requests.Session.post")
    def test_connect_failure(self, mock_post):
        """Test connection failure"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        broker = SSIBroker(account_id="TEST123", consumer_id="invalid", consumer_secret="invalid")

        result = broker.connect()
        assert result is False

    def test_invalid_lot_size(self):
        """Test invalid lot size rejection"""
        broker = SSIBroker(
            account_id="TEST123", consumer_id="test_id", consumer_secret="test_secret"
        )
        broker._connected = True
        broker._access_token = "test_token"

        order = broker.place_order("VNM", OrderSide.BUY, 50, 85000)
        assert order is None


class TestVNDirectBroker:
    """Tests for VNDirect Broker"""

    def test_init(self):
        """Test VNDirect broker initialization"""
        broker = VNDirectBroker(
            account_id="TEST123", username="test_user", password="test_pass", is_paper=True
        )

        assert broker.account_id == "TEST123"
        assert broker.broker_name == "VNDIRECT_PAPER"
        assert broker.is_paper is True

    @patch("requests.Session.post")
    def test_connect_success(self, mock_post):
        """Test successful connection"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {"accessToken": "test_token", "refreshToken": "refresh_token"},
        }
        mock_post.return_value = mock_response

        broker = VNDirectBroker(account_id="TEST123", username="test_user", password="test_pass")

        result = broker.connect()
        assert result is True

    def test_invalid_lot_size(self):
        """Test invalid lot size rejection"""
        broker = VNDirectBroker(account_id="TEST123", username="test_user", password="test_pass")
        broker._connected = True
        broker._access_token = "test_token"

        order = broker.place_order("VNM", OrderSide.BUY, 75, 85000)
        assert order is None


class TestBrokerFactory:
    """Tests for broker factory function"""

    def test_get_simulated_broker(self):
        """Test getting simulated broker"""
        broker = get_broker("SIMULATED", "TEST123", {"initial_cash": 50_000_000})

        assert isinstance(broker, SimulatedBroker)

    def test_get_ssi_broker(self):
        """Test getting SSI broker"""
        with patch.object(SSIBroker, "connect", return_value=True):
            broker = get_broker(
                "SSI", "TEST123", {"consumer_id": "test", "consumer_secret": "test"}, is_paper=True
            )

            assert isinstance(broker, SSIBroker)

    def test_get_vndirect_broker(self):
        """Test getting VNDirect broker"""
        with patch.object(VNDirectBroker, "connect", return_value=True):
            broker = get_broker(
                "VNDIRECT", "TEST123", {"username": "test", "password": "test"}, is_paper=True
            )

            assert isinstance(broker, VNDirectBroker)

    def test_invalid_broker_type(self):
        """Test invalid broker type"""
        with pytest.raises(ValueError):
            get_broker("INVALID", "TEST123", {})


class TestOrderDataclass:
    """Tests for Order dataclass"""

    def test_order_properties(self):
        """Test order properties"""
        order = Order(
            order_id="TEST001",
            symbol="VNM",
            side=OrderSide.BUY,
            order_type=OrderType.LO,
            quantity=100,
            price=85000,
            status=OrderStatus.PARTIAL,
            filled_quantity=50,
        )

        assert order.is_active is True
        assert order.is_complete is False
        assert order.remaining_quantity == 50
        assert order.fill_rate == 0.5


class TestPositionDataclass:
    """Tests for Position dataclass"""

    def test_position_creation(self):
        """Test position creation"""
        pos = Position(
            symbol="VNM",
            quantity=100,
            avg_price=85000,
            market_price=86000,
            unrealized_pnl=100000,
            market_value=8600000,
            cost_basis=8500000,
        )

        assert pos.symbol == "VNM"
        assert pos.quantity == 100
        assert pos.unrealized_pnl == 100000
