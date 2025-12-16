# -*- coding: utf-8 -*-
"""
VPS Securities Broker Integration

Real broker implementation for VPS Securities (vps.com.vn)
Supports:
- Order placement (Buy/Sell)
- Position management  
- Account information
- Real-time order status
- Market data streaming

API Documentation: https://openapi.vps.com.vn/

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
from urllib.parse import urlencode

import requests

from src.broker.base_broker import (
    AccountInfo,
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================
DEFAULT_TIMEOUT = 10  # seconds
MIN_REQUEST_INTERVAL = 0.2  # 200ms between requests
LOT_SIZE = 100  # Vietnam market lot size


class VPSEndpoints:
    """VPS API Endpoints"""

    # Production URLs
    BASE_URL = "https://openapi.vps.com.vn"

    # Authentication
    AUTH_URL = f"{BASE_URL}/auth/token"
    REFRESH_URL = f"{BASE_URL}/auth/refresh"

    # Account
    ACCOUNT_INFO_URL = f"{BASE_URL}/api/v1/account/info"
    BALANCE_URL = f"{BASE_URL}/api/v1/account/balance"
    POSITIONS_URL = f"{BASE_URL}/api/v1/account/positions"

    # Orders
    NEW_ORDER_URL = f"{BASE_URL}/api/v1/order/new"
    CANCEL_ORDER_URL = f"{BASE_URL}/api/v1/order/cancel"
    MODIFY_ORDER_URL = f"{BASE_URL}/api/v1/order/modify"
    ORDER_STATUS_URL = f"{BASE_URL}/api/v1/order/status"
    ORDER_HISTORY_URL = f"{BASE_URL}/api/v1/order/history"

    # Market Data
    QUOTE_URL = f"{BASE_URL}/api/v1/market/quote"
    DEPTH_URL = f"{BASE_URL}/api/v1/market/depth"

    # Paper Trading (Sandbox)
    SANDBOX_BASE_URL = "https://sandbox-openapi.vps.com.vn"


class VPSBroker(BaseBroker):
    """
    VPS Securities Broker Implementation

    Usage:
        broker = VPSBroker(
            account_id="YOUR_ACCOUNT",
            api_key="YOUR_API_KEY",
            api_secret="YOUR_SECRET",
            is_paper=True  # Use sandbox for testing
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
        api_key: str,
        api_secret: str,
        is_paper: bool = True,  # Default to paper trading for safety
        auto_refresh_token: bool = True,
    ):
        """
        Initialize VPS Broker

        Args:
            account_id: VPS trading account ID
            api_key: VPS API key
            api_secret: VPS API secret
            is_paper: Use paper trading mode (sandbox)
            auto_refresh_token: Auto refresh token before expiry
        """
        super().__init__(account_id, api_key, api_secret)
        self.is_paper = is_paper
        self.auto_refresh_token = auto_refresh_token

        # Set base URL based on mode
        self.base_url = VPSEndpoints.SANDBOX_BASE_URL if is_paper else VPSEndpoints.BASE_URL

        # Session management
        self._session = requests.Session()
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._last_request_time: float = 0

        # Rate limiting
        self._min_request_interval = MIN_REQUEST_INTERVAL

        logger.info(
            f"VPS Broker initialized - Account: {account_id}, "
            f"Mode: {'PAPER' if is_paper else 'LIVE'}"
        )

    @property
    def broker_name(self) -> str:
        return "VPS"

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _generate_signature(self, params: Dict, timestamp: str) -> str:
        """Generate HMAC signature for API request"""
        # Sort params and create query string
        sorted_params = sorted(params.items())
        query_string = urlencode(sorted_params)

        # Add timestamp to signature payload
        payload = f"{query_string}&timestamp={timestamp}"

        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return signature

    def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        authenticated: bool = True,
    ) -> Optional[Dict]:
        """Make API request with authentication and error handling"""
        self._rate_limit()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if authenticated:
            if not self._access_token:
                logger.error("Not authenticated - call connect() first")
                return None

            # Check token expiry
            if self.auto_refresh_token and self._is_token_expired():
                self._refresh_access_token()

            headers["Authorization"] = f"Bearer {self._access_token}"

        # Add timestamp and signature for authenticated requests
        if authenticated and params:
            timestamp = str(int(time.time() * 1000))
            params["timestamp"] = timestamp
            params["signature"] = self._generate_signature(params, timestamp)

        try:
            if method.upper() == "GET":
                response = self._session.get(
                    url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT
                )
            elif method.upper() == "POST":
                response = self._session.post(
                    url, params=params, json=data, headers=headers, timeout=DEFAULT_TIMEOUT
                )
            elif method.upper() == "DELETE":
                response = self._session.delete(
                    url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT
                )
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {e}")
            return None

    def _is_token_expired(self) -> bool:
        """Check if access token is expired or about to expire"""
        if not self._token_expiry:
            return True
        # Refresh 5 minutes before expiry
        return datetime.now() >= self._token_expiry

    def _refresh_access_token(self) -> bool:
        """Refresh access token using refresh token"""
        if not self._refresh_token:
            logger.error("No refresh token available")
            return False

        try:
            response = self._session.post(
                f"{self.base_url}/auth/refresh",
                json={"refresh_token": self._refresh_token},
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            self._access_token = data.get("access_token")
            self._refresh_token = data.get("refresh_token", self._refresh_token)

            # Set expiry (typically 30 minutes, minus 5 minute buffer)
            expires_in = data.get("expires_in", 1800)
            from datetime import timedelta

            self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)

            logger.info("Access token refreshed successfully")
            return True

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False

    def connect(self) -> bool:
        """Connect to VPS API and authenticate"""
        try:
            # Authenticate
            auth_data = {
                "api_key": self.api_key,
                "api_secret": self.api_secret,
                "account_id": self.account_id,
            }

            response = self._session.post(
                f"{self.base_url}/auth/token",
                json=auth_data,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "success":
                self._access_token = data.get("access_token")
                self._refresh_token = data.get("refresh_token")

                # Set expiry
                expires_in = data.get("expires_in", 1800)
                from datetime import timedelta

                self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)

                self._connected = True
                logger.info(f"✅ Connected to VPS API (Account: {self.account_id})")
                return True
            else:
                logger.error(f"Authentication failed: {data.get('message')}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from VPS API"""
        self._access_token = None
        self._refresh_token = None
        self._token_expiry = None
        self._connected = False
        self._session.close()
        logger.info("Disconnected from VPS API")

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        # Get account info
        account_data = self._make_request("GET", f"{self.base_url}/api/v1/account/info")
        if not account_data:
            return None

        # Get balance info
        balance_data = self._make_request("GET", f"{self.base_url}/api/v1/account/balance")

        try:
            account = account_data.get("data", {})
            balance = balance_data.get("data", {}) if balance_data else {}

            return AccountInfo(
                account_id=self.account_id,
                account_name=account.get("name", ""),
                broker=self.broker_name,
                total_equity=float(balance.get("total_equity", 0)),
                cash_balance=float(balance.get("cash", 0)),
                buying_power=float(balance.get("buying_power", 0)),
                margin_used=float(balance.get("margin_used", 0)),
                margin_available=float(balance.get("margin_available", 0)),
                margin_ratio=float(balance.get("margin_ratio", 0)),
                total_market_value=float(balance.get("market_value", 0)),
                total_unrealized_pnl=float(balance.get("unrealized_pnl", 0)),
                pending_settlements=float(balance.get("pending_t2", 0)),
                is_active=account.get("status") == "active",
                last_updated=datetime.now(),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse account info: {e}")
            return None

    def get_positions(self) -> List[Position]:
        """Get all positions"""
        response = self._make_request("GET", f"{self.base_url}/api/v1/account/positions")
        if not response:
            return []

        positions = []
        for pos_data in response.get("data", []):
            try:
                position = Position(
                    symbol=pos_data.get("symbol", ""),
                    quantity=int(pos_data.get("quantity", 0)),
                    avg_price=float(pos_data.get("avg_price", 0)),
                    market_price=float(pos_data.get("market_price", 0)),
                    unrealized_pnl=float(pos_data.get("unrealized_pnl", 0)),
                    unrealized_pnl_pct=float(pos_data.get("unrealized_pnl_pct", 0)),
                    market_value=float(pos_data.get("market_value", 0)),
                    cost_basis=float(pos_data.get("cost_basis", 0)),
                    available_quantity=int(pos_data.get("available_qty", 0)),
                    pending_buy=int(pos_data.get("pending_buy", 0)),
                    pending_sell=int(pos_data.get("pending_sell", 0)),
                    account_id=self.account_id,
                    last_updated=datetime.now(),
                )
                positions.append(position)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse position: {e}")
                continue

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
            symbol: Stock symbol (e.g., "VNM")
            side: BUY or SELL
            quantity: Number of shares (must be multiple of 100)
            price: Order price
            order_type: LO, ATO, ATC, MP, etc.

        Returns:
            Order object if successful, None otherwise
        """
        # Validate lot size
        if quantity % LOT_SIZE != 0:
            logger.error(f"Quantity must be multiple of {LOT_SIZE}")
            return None

        # Build order request
        order_data = {
            "account_id": self.account_id,
            "symbol": symbol.upper(),
            "side": side.value,
            "quantity": quantity,
            "price": price,
            "order_type": order_type.value,
        }

        response = self._make_request(
            "POST",
            f"{self.base_url}/api/v1/order/new",
            data=order_data,
        )

        if not response:
            return None

        if response.get("status") != "success":
            logger.error(f"Order rejected: {response.get('message')}")
            return None

        try:
            order_info = response.get("data", {})
            order = Order(
                order_id=order_info.get("order_id", ""),
                symbol=symbol.upper(),
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                status=OrderStatus.SUBMITTED,
                account_id=self.account_id,
                broker=self.broker_name,
                message="Order submitted successfully",
            )

            logger.info(
                f"✅ Order placed: {side.value} {quantity} {symbol} @ {price:,.0f} "
                f"(ID: {order.order_id})"
            )
            return order

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse order response: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        response = self._make_request(
            "POST",
            f"{self.base_url}/api/v1/order/cancel",
            data={"order_id": order_id, "account_id": self.account_id},
        )

        if response and response.get("status") == "success":
            logger.info(f"✅ Order cancelled: {order_id}")
            return True
        else:
            logger.error(f"Failed to cancel order: {order_id}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        response = self._make_request(
            "GET",
            f"{self.base_url}/api/v1/order/status",
            params={"order_id": order_id},
        )

        if not response:
            return None

        try:
            order_data = response.get("data", {})
            return Order(
                order_id=order_data.get("order_id", ""),
                symbol=order_data.get("symbol", ""),
                side=OrderSide(order_data.get("side", "BUY")),
                order_type=OrderType(order_data.get("order_type", "LO")),
                quantity=int(order_data.get("quantity", 0)),
                price=float(order_data.get("price", 0)),
                status=OrderStatus(order_data.get("status", "PENDING")),
                filled_quantity=int(order_data.get("filled_qty", 0)),
                filled_price=float(order_data.get("filled_price", 0)),
                account_id=self.account_id,
                broker=self.broker_name,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse order: {e}")
            return None

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get all orders, optionally filtered by status"""
        params = {}
        if status:
            params["status"] = status.value

        response = self._make_request(
            "GET",
            f"{self.base_url}/api/v1/order/history",
            params=params,
        )

        if not response:
            return []

        orders = []
        for order_data in response.get("data", []):
            try:
                order = Order(
                    order_id=order_data.get("order_id", ""),
                    symbol=order_data.get("symbol", ""),
                    side=OrderSide(order_data.get("side", "BUY")),
                    order_type=OrderType(order_data.get("order_type", "LO")),
                    quantity=int(order_data.get("quantity", 0)),
                    price=float(order_data.get("price", 0)),
                    status=OrderStatus(order_data.get("status", "PENDING")),
                    filled_quantity=int(order_data.get("filled_qty", 0)),
                    filled_price=float(order_data.get("filled_price", 0)),
                    account_id=self.account_id,
                    broker=self.broker_name,
                )
                orders.append(order)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse order: {e}")
                continue

        return orders

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.LO,
    ) -> Optional[Order]:
        """Place a buy order"""
        return self.place_order(symbol, OrderSide.BUY, quantity, price, order_type)

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.LO,
    ) -> Optional[Order]:
        """Place a sell order"""
        return self.place_order(symbol, OrderSide.SELL, quantity, price, order_type)

    def buy_market(self, symbol: str, quantity: int) -> Optional[Order]:
        """Place a market buy order (MP - Market Price)"""
        # Get current price for market order
        quote = self.get_quote(symbol)
        if not quote:
            logger.error(f"Cannot get quote for {symbol}")
            return None

        price = quote.get("ask_price", quote.get("last_price", 0))
        return self.place_order(symbol, OrderSide.BUY, quantity, price, OrderType.MP)

    def sell_market(self, symbol: str, quantity: int) -> Optional[Order]:
        """Place a market sell order"""
        quote = self.get_quote(symbol)
        if not quote:
            logger.error(f"Cannot get quote for {symbol}")
            return None

        price = quote.get("bid_price", quote.get("last_price", 0))
        return self.place_order(symbol, OrderSide.SELL, quantity, price, OrderType.MP)

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get current quote for a symbol"""
        response = self._make_request(
            "GET",
            f"{self.base_url}/api/v1/market/quote",
            params={"symbol": symbol.upper()},
            authenticated=False,  # Market data may not require auth
        )

        if response and response.get("status") == "success":
            return response.get("data", {})
        return None

    def get_order_book(self, symbol: str, depth: int = 10) -> Optional[Dict]:
        """Get order book depth for a symbol"""
        response = self._make_request(
            "GET",
            f"{self.base_url}/api/v1/market/depth",
            params={"symbol": symbol.upper(), "depth": depth},
            authenticated=False,
        )

        if response and response.get("status") == "success":
            return response.get("data", {})
        return None


# ============================================================================
# Factory function
# ============================================================================


def get_vps_broker(
    account_id: str,
    api_key: str,
    api_secret: str,
    is_paper: bool = True,
) -> VPSBroker:
    """
    Factory function to create VPS broker instance

    Args:
        account_id: VPS trading account ID
        api_key: VPS API key
        api_secret: VPS API secret
        is_paper: Use paper trading mode (default True for safety)

    Returns:
        VPSBroker instance
    """
    return VPSBroker(
        account_id=account_id,
        api_key=api_key,
        api_secret=api_secret,
        is_paper=is_paper,
    )
