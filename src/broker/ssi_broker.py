# -*- coding: utf-8 -*-
"""
SSI Securities Broker Integration

Real broker implementation for SSI Securities (ssi.com.vn)
Supports:
- Order placement (Buy/Sell)
- Position management
- Account information
- Real-time order status

API Documentation: https://iboard.ssi.com.vn/

Author: Trading Bot Team
Version: 1.0.0
"""

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

import requests

from src.broker.base_broker import (
    BaseBroker,
    Order,
    Position,
    AccountInfo,
    OrderSide,
    OrderType,
    OrderStatus,
)

logger = logging.getLogger(__name__)


class SSIEndpoints:
    """SSI API Endpoints"""

    BASE_URL = "https://fc-tradeapi.ssi.com.vn"
    AUTH_URL = f"{BASE_URL}/api/v2/Auth/AccessToken"
    ACCOUNT_URL = f"{BASE_URL}/api/v2/Trading/Account"
    POSITIONS_URL = f"{BASE_URL}/api/v2/Trading/Positions"
    ORDERS_URL = f"{BASE_URL}/api/v2/Trading/Orders"
    NEW_ORDER_URL = f"{BASE_URL}/api/v2/Trading/NewOrder"
    CANCEL_ORDER_URL = f"{BASE_URL}/api/v2/Trading/CancelOrder"
    ORDER_HISTORY_URL = f"{BASE_URL}/api/v2/Trading/OrderHistory"


class SSIBroker(BaseBroker):
    """
    SSI Securities Broker Implementation

    Usage:
        broker = SSIBroker(
            account_id="YOUR_ACCOUNT",
            consumer_id="YOUR_CONSUMER_ID",
            consumer_secret="YOUR_SECRET",
            private_key="YOUR_PRIVATE_KEY"
        )
        broker.connect()

        # Place order
        order = broker.buy("VNM", 100, 85000)

        # Check positions
        positions = broker.get_positions()
    """

    def __init__(
        self,
        account_id: str,
        consumer_id: str,
        consumer_secret: str,
        private_key: str = "",
        is_paper: bool = True,  # Default to paper trading for safety
    ):
        """
        Initialize SSI Broker

        Args:
            account_id: SSI trading account ID
            consumer_id: SSI API consumer ID
            consumer_secret: SSI API consumer secret
            private_key: Private key for signing (optional)
            is_paper: Use paper trading mode (default True for safety)
        """
        super().__init__(account_id, consumer_id, consumer_secret)
        self.consumer_id = consumer_id
        self.consumer_secret = consumer_secret
        self.private_key = private_key
        self.is_paper = is_paper

        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._session = requests.Session()

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.2  # 200ms between requests

    @property
    def broker_name(self) -> str:
        return "SSI" + ("_PAPER" if self.is_paper else "")

    def connect(self) -> bool:
        """Connect and authenticate with SSI API"""
        try:
            self._access_token = self._authenticate()
            if self._access_token:
                self._connected = True
                logger.info(f"✅ Connected to SSI {'Paper' if self.is_paper else 'Live'} Trading")
                return True
        except Exception as e:
            logger.error(f"❌ SSI connection failed: {e}")

        return False

    def _authenticate(self) -> Optional[str]:
        """Authenticate with SSI API"""
        try:
            payload = {
                "consumerID": self.consumer_id,
                "consumerSecret": self.consumer_secret,
            }

            response = self._session.post(SSIEndpoints.AUTH_URL, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == 200:
                    token = data.get("data", {}).get("accessToken")
                    logger.info("✅ SSI authentication successful")
                    return token

            logger.warning(f"SSI auth failed: {response.text}")
            return None

        except Exception as e:
            logger.error(f"SSI authentication error: {e}")
            return None

    def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid token"""
        if not self._access_token:
            return self.connect()
        return True

    def _rate_limit(self):
        """Apply rate limiting"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self, method: str, url: str, data: Optional[Dict] = None, params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make authenticated API request"""
        if not self._ensure_authenticated():
            return None

        self._rate_limit()

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        try:
            if method.upper() == "GET":
                response = self._session.get(url, headers=headers, params=params, timeout=10)
            else:
                response = self._session.post(url, headers=headers, json=data, timeout=10)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                # Token expired, re-authenticate
                logger.info("Token expired, re-authenticating...")
                self._access_token = self._authenticate()
                return self._make_request(method, url, data, params)
            else:
                logger.warning(f"SSI API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"SSI request error: {e}")
            return None

    def disconnect(self) -> None:
        """Disconnect from SSI API"""
        self._connected = False
        self._access_token = None
        self._session.close()
        logger.info("Disconnected from SSI API")

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        data = self._make_request(
            "GET", SSIEndpoints.ACCOUNT_URL, params={"account": self.account_id}
        )

        if not data or data.get("status") != 200:
            return None

        account_data = data.get("data", {})

        return AccountInfo(
            account_id=self.account_id,
            account_name=account_data.get("accountName", ""),
            broker=self.broker_name,
            total_equity=float(account_data.get("equity", 0)),
            cash_balance=float(account_data.get("cashBalance", 0)),
            buying_power=float(account_data.get("buyingPower", 0)),
            margin_used=float(account_data.get("marginUsed", 0)),
            margin_available=float(account_data.get("marginAvailable", 0)),
            total_market_value=float(account_data.get("marketValue", 0)),
            total_unrealized_pnl=float(account_data.get("unrealizedPL", 0)),
            pending_settlements=float(account_data.get("pendingSettlement", 0)),
            t1_receivable=float(account_data.get("t1Receivable", 0)),
            t2_receivable=float(account_data.get("t2Receivable", 0)),
        )

    def get_positions(self) -> List[Position]:
        """Get all positions"""
        data = self._make_request(
            "GET", SSIEndpoints.POSITIONS_URL, params={"account": self.account_id}
        )

        if not data or data.get("status") != 200:
            return []

        positions = []
        for item in data.get("data", []):
            pos = Position(
                symbol=item.get("symbol", ""),
                quantity=int(item.get("quantity", 0)),
                avg_price=float(item.get("avgPrice", 0)),
                market_price=float(item.get("marketPrice", 0)),
                unrealized_pnl=float(item.get("unrealizedPL", 0)),
                unrealized_pnl_pct=float(item.get("unrealizedPLPercent", 0)),
                market_value=float(item.get("marketValue", 0)),
                cost_basis=float(item.get("costBasis", 0)),
                available_quantity=int(item.get("availableQty", 0)),
                account_id=self.account_id,
            )
            positions.append(pos)

        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol.upper() == symbol.upper():
                return pos
        return None

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.LO,
    ) -> Optional[Order]:
        """
        Place an order

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Number of shares (must be multiple of 100)
            price: Order price
            order_type: Order type (LO, ATO, ATC, MP)

        Returns:
            Order object or None if failed
        """
        # Validate lot size
        if quantity % 100 != 0:
            logger.warning(f"Invalid lot size: {quantity}. Must be multiple of 100.")
            return None

        # Safety check for live trading
        if not self.is_paper:
            logger.warning("⚠️ LIVE TRADING MODE - Order will be executed!")

        # Map order type to SSI format
        ssi_order_type = {
            OrderType.LO: "LO",
            OrderType.ATO: "ATO",
            OrderType.ATC: "ATC",
            OrderType.MP: "MP",
            OrderType.MTL: "MTL",
            OrderType.MOK: "MOK",
            OrderType.MAK: "MAK",
        }.get(order_type, "LO")

        payload = {
            "account": self.account_id,
            "symbol": symbol.upper(),
            "side": "B" if side == OrderSide.BUY else "S",
            "orderType": ssi_order_type,
            "quantity": quantity,
            "price": price,
        }

        data = self._make_request("POST", SSIEndpoints.NEW_ORDER_URL, data=payload)

        if not data or data.get("status") != 200:
            logger.error(f"Order placement failed: {data}")
            return None

        order_data = data.get("data", {})

        order = Order(
            order_id=order_data.get("orderID", ""),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.SUBMITTED,
            account_id=self.account_id,
            broker=self.broker_name,
            message=order_data.get("message", ""),
        )

        logger.info(f"📝 Order placed: {side.value} {quantity} {symbol} @ {price:,.0f}")
        self._notify_order_update(order)

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        payload = {
            "account": self.account_id,
            "orderID": order_id,
        }

        data = self._make_request("POST", SSIEndpoints.CANCEL_ORDER_URL, data=payload)

        if data and data.get("status") == 200:
            logger.info(f"✅ Order {order_id} cancelled")
            return True

        logger.warning(f"❌ Failed to cancel order {order_id}")
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        orders = self.get_orders()
        for order in orders:
            if order.order_id == order_id:
                return order
        return None

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get all orders"""
        data = self._make_request(
            "GET", SSIEndpoints.ORDERS_URL, params={"account": self.account_id}
        )

        if not data or data.get("status") != 200:
            return []

        orders = []
        for item in data.get("data", []):
            # Map SSI status to OrderStatus
            ssi_status = item.get("orderStatus", "").upper()
            order_status = {
                "PENDING": OrderStatus.PENDING,
                "SUBMITTED": OrderStatus.SUBMITTED,
                "PARTIAL": OrderStatus.PARTIAL,
                "FILLED": OrderStatus.FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
            }.get(ssi_status, OrderStatus.PENDING)

            if status and order_status != status:
                continue

            order = Order(
                order_id=item.get("orderID", ""),
                symbol=item.get("symbol", ""),
                side=OrderSide.BUY if item.get("side") == "B" else OrderSide.SELL,
                order_type=OrderType.LO,
                quantity=int(item.get("quantity", 0)),
                price=float(item.get("price", 0)),
                status=order_status,
                filled_quantity=int(item.get("filledQty", 0)),
                filled_price=float(item.get("avgPrice", 0)),
                account_id=self.account_id,
                broker=self.broker_name,
            )
            orders.append(order)

        return orders


# Factory function
def create_ssi_broker(
    account_id: str, consumer_id: str, consumer_secret: str, is_paper: bool = True
) -> SSIBroker:
    """Create and connect SSI broker"""
    broker = SSIBroker(
        account_id=account_id,
        consumer_id=consumer_id,
        consumer_secret=consumer_secret,
        is_paper=is_paper,
    )
    broker.connect()
    return broker
