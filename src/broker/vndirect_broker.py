# -*- coding: utf-8 -*-
"""
VNDirect Securities Broker Integration

Real broker implementation for VNDirect Securities (vndirect.com.vn)
Supports:
- Order placement (Buy/Sell)
- Position management
- Account information
- Real-time order status

API Documentation: https://trade.vndirect.com.vn/

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


class VNDirectEndpoints:
    """VNDirect API Endpoints"""

    BASE_URL = "https://trade-api.vndirect.com.vn"
    AUTH_URL = f"{BASE_URL}/accounts/v3/auth"
    ACCOUNT_URL = f"{BASE_URL}/accounts/v3/accounts"
    POSITIONS_URL = f"{BASE_URL}/accounts/v3/portfolio"
    ORDERS_URL = f"{BASE_URL}/orders/v2/orders"
    NEW_ORDER_URL = f"{BASE_URL}/orders/v2/place"
    CANCEL_ORDER_URL = f"{BASE_URL}/orders/v2/cancel"

    # OTP endpoints
    OTP_REQUEST_URL = f"{BASE_URL}/accounts/v3/otp/request"
    OTP_VERIFY_URL = f"{BASE_URL}/accounts/v3/otp/verify"


class VNDirectBroker(BaseBroker):
    """
    VNDirect Securities Broker Implementation

    Usage:
        broker = VNDirectBroker(
            account_id="YOUR_ACCOUNT",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD"
        )
        broker.connect()

        # Place order
        order = broker.buy("VNM", 100, 85000)

        # Check positions
        positions = broker.get_positions()

    Note: VNDirect requires OTP for order placement in live mode
    """

    def __init__(
        self,
        account_id: str,
        username: str,
        password: str,
        is_paper: bool = True,  # Default to paper trading for safety
        otp_callback: Optional[callable] = None,  # Callback to get OTP
    ):
        """
        Initialize VNDirect Broker

        Args:
            account_id: VNDirect trading account ID
            username: VNDirect username
            password: VNDirect password
            is_paper: Use paper trading mode (default True for safety)
            otp_callback: Callback function to get OTP for live trading
        """
        super().__init__(account_id, username, password)
        self.username = username
        self.password = password
        self.is_paper = is_paper
        self.otp_callback = otp_callback

        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._session = requests.Session()
        self._otp_token: Optional[str] = None

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.3  # 300ms between requests

        # Retry settings
        self._max_retries = 3
        self._retry_delay = 1.0  # seconds

    @property
    def broker_name(self) -> str:
        return "VNDIRECT" + ("_PAPER" if self.is_paper else "")

    def connect(self) -> bool:
        """Connect and authenticate with VNDirect API"""
        try:
            self._access_token = self._authenticate()
            if self._access_token:
                self._connected = True
                logger.info(
                    f"✅ Connected to VNDirect {'Paper' if self.is_paper else 'Live'} Trading"
                )
                return True
        except Exception as e:
            logger.error(f"❌ VNDirect connection failed: {e}")

        return False

    def _authenticate(self) -> Optional[str]:
        """Authenticate with VNDirect API"""
        for attempt in range(self._max_retries):
            try:
                payload = {
                    "username": self.username,
                    "password": self.password,
                }

                response = self._session.post(VNDirectEndpoints.AUTH_URL, json=payload, timeout=15)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        self._access_token = data.get("data", {}).get("accessToken")
                        self._refresh_token = data.get("data", {}).get("refreshToken")
                        logger.info("✅ VNDirect authentication successful")
                        return self._access_token

                logger.warning(f"VNDirect auth attempt {attempt + 1} failed: {response.text}")

                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))

            except requests.exceptions.Timeout:
                logger.warning(f"VNDirect auth timeout (attempt {attempt + 1})")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"VNDirect connection error: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"VNDirect authentication error: {e}")
                break

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
        self,
        method: str,
        url: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        retry: bool = True,
    ) -> Optional[Dict]:
        """Make authenticated API request with retry logic"""
        if not self._ensure_authenticated():
            return None

        self._rate_limit()

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        for attempt in range(self._max_retries if retry else 1):
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
                    if self._access_token:
                        headers["Authorization"] = f"Bearer {self._access_token}"
                        continue
                elif response.status_code == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                else:
                    logger.warning(f"VNDirect API error: {response.status_code} - {response.text}")

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"VNDirect request error: {e}")
                break

            if attempt < self._max_retries - 1:
                time.sleep(self._retry_delay * (attempt + 1))

        return None

    def disconnect(self) -> None:
        """Disconnect from VNDirect API"""
        self._connected = False
        self._access_token = None
        self._refresh_token = None
        self._session.close()
        logger.info("Disconnected from VNDirect API")

    def request_otp(self) -> bool:
        """Request OTP for order placement"""
        if self.is_paper:
            return True  # No OTP needed for paper trading

        data = self._make_request("POST", VNDirectEndpoints.OTP_REQUEST_URL)
        if data and data.get("status") == "success":
            logger.info("OTP requested successfully")
            return True
        return False

    def verify_otp(self, otp: str) -> bool:
        """Verify OTP"""
        if self.is_paper:
            return True

        data = self._make_request("POST", VNDirectEndpoints.OTP_VERIFY_URL, data={"otp": otp})
        if data and data.get("status") == "success":
            self._otp_token = data.get("data", {}).get("otpToken")
            logger.info("OTP verified successfully")
            return True
        return False

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        data = self._make_request(
            "GET", VNDirectEndpoints.ACCOUNT_URL, params={"accountNo": self.account_id}
        )

        if not data or data.get("status") != "success":
            return None

        account_data = data.get("data", {})

        return AccountInfo(
            account_id=self.account_id,
            account_name=account_data.get("accountName", ""),
            broker=self.broker_name,
            total_equity=float(account_data.get("totalEquity", 0)),
            cash_balance=float(account_data.get("cashBalance", 0)),
            buying_power=float(account_data.get("buyingPower", 0)),
            margin_used=float(account_data.get("marginDebt", 0)),
            margin_available=float(account_data.get("marginAvailable", 0)),
            total_market_value=float(account_data.get("stockValue", 0)),
            total_unrealized_pnl=float(account_data.get("unrealizedPL", 0)),
            pending_settlements=float(account_data.get("pendingSettlement", 0)),
            t1_receivable=float(account_data.get("t1Amount", 0)),
            t2_receivable=float(account_data.get("t2Amount", 0)),
        )

    def get_positions(self) -> List[Position]:
        """Get all positions"""
        data = self._make_request(
            "GET", VNDirectEndpoints.POSITIONS_URL, params={"accountNo": self.account_id}
        )

        if not data or data.get("status") != "success":
            return []

        positions = []
        for item in data.get("data", {}).get("stocks", []):
            pos = Position(
                symbol=item.get("symbol", ""),
                quantity=int(item.get("totalQuantity", 0)),
                avg_price=float(item.get("avgPrice", 0)),
                market_price=float(item.get("marketPrice", 0)),
                unrealized_pnl=float(item.get("unrealizedPL", 0)),
                unrealized_pnl_pct=float(item.get("unrealizedPLPercent", 0)),
                market_value=float(item.get("marketValue", 0)),
                cost_basis=float(item.get("costValue", 0)),
                available_quantity=int(item.get("availableQuantity", 0)),
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

            # Request OTP if needed
            if not self._otp_token:
                if self.otp_callback:
                    self.request_otp()
                    otp = self.otp_callback()
                    if not self.verify_otp(otp):
                        logger.error("OTP verification failed")
                        return None
                else:
                    logger.error("OTP required but no callback provided")
                    return None

        # Map order type to VNDirect format
        vnd_order_type = {
            OrderType.LO: "LO",
            OrderType.ATO: "ATO",
            OrderType.ATC: "ATC",
            OrderType.MP: "MP",
            OrderType.MTL: "MTL",
            OrderType.MOK: "MOK",
            OrderType.MAK: "MAK",
        }.get(order_type, "LO")

        payload = {
            "accountNo": self.account_id,
            "symbol": symbol.upper(),
            "side": "NB" if side == OrderSide.BUY else "NS",  # NB=Buy, NS=Sell
            "orderType": vnd_order_type,
            "quantity": quantity,
            "price": price,
        }

        # Add OTP token for live trading
        if not self.is_paper and self._otp_token:
            payload["otpToken"] = self._otp_token

        data = self._make_request("POST", VNDirectEndpoints.NEW_ORDER_URL, data=payload)

        if not data or data.get("status") != "success":
            logger.error(f"Order placement failed: {data}")
            return None

        order_data = data.get("data", {})

        order = Order(
            order_id=order_data.get("orderId", ""),
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
            "accountNo": self.account_id,
            "orderId": order_id,
        }

        # Add OTP for live trading
        if not self.is_paper and self._otp_token:
            payload["otpToken"] = self._otp_token

        data = self._make_request("POST", VNDirectEndpoints.CANCEL_ORDER_URL, data=payload)

        if data and data.get("status") == "success":
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
            "GET", VNDirectEndpoints.ORDERS_URL, params={"accountNo": self.account_id}
        )

        if not data or data.get("status") != "success":
            return []

        orders = []
        for item in data.get("data", []):
            # Map VNDirect status to OrderStatus
            vnd_status = item.get("orderStatus", "").upper()
            order_status = {
                "PENDING": OrderStatus.PENDING,
                "QUEUED": OrderStatus.SUBMITTED,
                "PARTIAL": OrderStatus.PARTIAL,
                "FILLED": OrderStatus.FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
                "EXPIRED": OrderStatus.EXPIRED,
            }.get(vnd_status, OrderStatus.PENDING)

            if status and order_status != status:
                continue

            order = Order(
                order_id=item.get("orderId", ""),
                symbol=item.get("symbol", ""),
                side=OrderSide.BUY if item.get("side") == "NB" else OrderSide.SELL,
                order_type=OrderType.LO,
                quantity=int(item.get("quantity", 0)),
                price=float(item.get("price", 0)),
                status=order_status,
                filled_quantity=int(item.get("filledQuantity", 0)),
                filled_price=float(item.get("avgPrice", 0)),
                account_id=self.account_id,
                broker=self.broker_name,
            )
            orders.append(order)

        return orders

    def get_order_history(
        self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> List[Order]:
        """Get order history"""
        params = {"accountNo": self.account_id}

        if from_date:
            params["fromDate"] = from_date.strftime("%Y-%m-%d")
        if to_date:
            params["toDate"] = to_date.strftime("%Y-%m-%d")

        data = self._make_request("GET", f"{VNDirectEndpoints.ORDERS_URL}/history", params=params)

        if not data or data.get("status") != "success":
            return []

        # Same parsing as get_orders
        orders = []
        for item in data.get("data", []):
            vnd_status = item.get("orderStatus", "").upper()
            order_status = {
                "FILLED": OrderStatus.FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
                "EXPIRED": OrderStatus.EXPIRED,
            }.get(vnd_status, OrderStatus.FILLED)

            order = Order(
                order_id=item.get("orderId", ""),
                symbol=item.get("symbol", ""),
                side=OrderSide.BUY if item.get("side") == "NB" else OrderSide.SELL,
                order_type=OrderType.LO,
                quantity=int(item.get("quantity", 0)),
                price=float(item.get("price", 0)),
                status=order_status,
                filled_quantity=int(item.get("filledQuantity", 0)),
                filled_price=float(item.get("avgPrice", 0)),
                account_id=self.account_id,
                broker=self.broker_name,
            )
            orders.append(order)

        return orders


# Factory function
def create_vndirect_broker(
    account_id: str,
    username: str,
    password: str,
    is_paper: bool = True,
    otp_callback: Optional[callable] = None,
) -> VNDirectBroker:
    """Create and connect VNDirect broker"""
    broker = VNDirectBroker(
        account_id=account_id,
        username=username,
        password=password,
        is_paper=is_paper,
        otp_callback=otp_callback,
    )
    broker.connect()
    return broker
