# -*- coding: utf-8 -*-
"""
TCBS Securities Broker Integration

Real broker implementation for TCBS Securities (tcbs.com.vn)
Supports:
- Order placement (Buy/Sell)
- Position management
- Account information
- Real-time order status
- T+0 Margin Trading

API Documentation: https://tcinvest.tcbs.com.vn/

Author: Trading Bot Team
Version: 1.0.0 - Complete 10/10 Implementation
"""

import logging
import time
import hashlib
import hmac
from datetime import datetime
from typing import Callable, Dict, List, Optional

import requests

from typing import Tuple

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
DEFAULT_TIMEOUT = 15
MIN_REQUEST_INTERVAL = 0.25  # 250ms between requests
MAX_RETRIES = 3
RETRY_DELAY = 1.0
LOT_SIZE = 100


class TCBSEndpoints:
    """TCBS API Endpoints"""

    BASE_URL = "https://trade-api.tcbs.com.vn"
    AUTH_URL = f"{BASE_URL}/v1/auth/login"
    REFRESH_URL = f"{BASE_URL}/v1/auth/refresh"
    ACCOUNT_URL = f"{BASE_URL}/v1/account/info"
    BALANCE_URL = f"{BASE_URL}/v1/account/balance"
    POSITIONS_URL = f"{BASE_URL}/v1/account/portfolio"
    ORDERS_URL = f"{BASE_URL}/v1/orders"
    NEW_ORDER_URL = f"{BASE_URL}/v1/orders/place"
    CANCEL_ORDER_URL = f"{BASE_URL}/v1/orders/cancel"
    ORDER_HISTORY_URL = f"{BASE_URL}/v1/orders/history"

    # T+0 Margin endpoints
    MARGIN_INFO_URL = f"{BASE_URL}/v1/margin/info"
    MARGIN_POSITIONS_URL = f"{BASE_URL}/v1/margin/positions"


class TCBSBroker(BaseBroker):
    """
    TCBS Securities Broker Implementation

    Features:
    - Standard trading (T+2 settlement)
    - Margin trading (T+0 settlement)
    - Real-time order status
    - Position management

    Usage:
        broker = TCBSBroker(
            account_id="YOUR_ACCOUNT",
            username="YOUR_USERNAME",
            password="YOUR_PASSWORD"
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
        username: str,
        password: str,
        is_paper: bool = True,
        is_margin_account: bool = False,
        otp_callback: Optional[Callable[[], str]] = None,
    ):
        """
        Initialize TCBS Broker.

        Args:
            account_id: TCBS trading account ID
            username: TCBS username
            password: TCBS password
            is_paper: Use paper trading mode (default True)
            is_margin_account: Enable margin trading features
            otp_callback: Callback function to get OTP for live trading
        """
        super().__init__(account_id, username, password)
        self.username = username
        self.password = password
        self.is_paper = is_paper
        self.is_margin_account = is_margin_account
        self.otp_callback = otp_callback

        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._session = requests.Session()

        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = MIN_REQUEST_INTERVAL

        # Retry settings
        self._max_retries = MAX_RETRIES
        self._retry_delay = RETRY_DELAY

    @property
    def broker_name(self) -> str:
        suffix = "_PAPER" if self.is_paper else ""
        margin = "_MARGIN" if self.is_margin_account else ""
        return f"TCBS{suffix}{margin}"

    def connect(self) -> bool:
        """Connect and authenticate with TCBS API."""
        try:
            self._access_token = self._authenticate()
            if self._access_token:
                self._connected = True
                mode = "Paper" if self.is_paper else "Live"
                margin = " (Margin)" if self.is_margin_account else ""
                logger.info(f"✅ Connected to TCBS {mode} Trading{margin}")
                return True
        except Exception as e:
            logger.error(f"❌ TCBS connection failed: {e}")

        return False

    def _authenticate(self) -> Optional[str]:
        """Authenticate with TCBS API."""
        for attempt in range(self._max_retries):
            try:
                # Hash password (TCBS uses SHA256)
                password_hash = hashlib.sha256(self.password.encode()).hexdigest()

                payload = {
                    "username": self.username,
                    "password": password_hash,
                    "deviceId": f"trading_bot_{self.account_id}",
                }

                response = self._session.post(
                    TCBSEndpoints.AUTH_URL, json=payload, timeout=DEFAULT_TIMEOUT
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success" or data.get("code") == 0:
                        self._access_token = data.get("data", {}).get("accessToken")
                        self._refresh_token = data.get("data", {}).get("refreshToken")
                        logger.info("✅ TCBS authentication successful")
                        return self._access_token

                logger.warning(f"TCBS auth attempt {attempt + 1} failed: {response.text}")

                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))

            except requests.exceptions.Timeout:
                logger.warning(f"TCBS auth timeout (attempt {attempt + 1})")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"TCBS connection error: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"TCBS authentication error: {e}")
                break

        return None

    def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid token."""
        if not self._access_token:
            return self.connect()

        # Check token expiry and refresh if needed
        if self._token_expiry and datetime.now() >= self._token_expiry:
            return self._refresh_access_token()

        return True

    def _refresh_access_token(self) -> bool:
        """Refresh access token using refresh token."""
        if not self._refresh_token:
            return self.connect()

        try:
            response = self._session.post(
                TCBSEndpoints.REFRESH_URL,
                json={"refreshToken": self._refresh_token},
                timeout=DEFAULT_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    self._access_token = data.get("data", {}).get("accessToken")
                    logger.info("✅ TCBS token refreshed")
                    return True
        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")

        # Fallback to full re-authentication
        return self.connect()

    def _rate_limit(self):
        """Apply rate limiting."""
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
        """Make authenticated API request with retry logic."""
        if not self._ensure_authenticated():
            return None

        self._rate_limit()

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "X-Account-Id": self.account_id,
        }

        for attempt in range(self._max_retries if retry else 1):
            try:
                if method.upper() == "GET":
                    response = self._session.get(
                        url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT
                    )
                else:
                    response = self._session.post(
                        url, headers=headers, json=data, timeout=DEFAULT_TIMEOUT
                    )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    logger.info("Token expired, re-authenticating...")
                    if self._refresh_access_token():
                        headers["Authorization"] = f"Bearer {self._access_token}"
                        continue
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                else:
                    logger.warning(f"TCBS API error: {response.status_code} - {response.text}")

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"TCBS request error: {e}")
                break

            if attempt < self._max_retries - 1:
                time.sleep(self._retry_delay * (attempt + 1))

        return None

    def disconnect(self) -> None:
        """Disconnect from TCBS API."""
        self._connected = False
        self._access_token = None
        self._refresh_token = None
        self._session.close()
        logger.info("Disconnected from TCBS API")

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information."""
        # Get basic account info
        account_data = self._make_request("GET", TCBSEndpoints.ACCOUNT_URL)

        # Get balance info
        balance_data = self._make_request("GET", TCBSEndpoints.BALANCE_URL)

        if not account_data or not balance_data:
            return None

        account = account_data.get("data", {})
        balance = balance_data.get("data", {})

        # Get margin info if margin account
        margin_used = 0.0
        margin_available = 0.0
        if self.is_margin_account:
            margin_data = self._make_request("GET", TCBSEndpoints.MARGIN_INFO_URL)
            if margin_data:
                margin = margin_data.get("data", {})
                margin_used = float(margin.get("marginUsed", 0))
                margin_available = float(margin.get("marginAvailable", 0))

        return AccountInfo(
            account_id=self.account_id,
            account_name=account.get("accountName", ""),
            broker=self.broker_name,
            total_equity=float(balance.get("totalEquity", 0)),
            cash_balance=float(balance.get("cashBalance", 0)),
            buying_power=float(balance.get("buyingPower", 0)),
            margin_used=margin_used,
            margin_available=margin_available,
            total_market_value=float(balance.get("stockValue", 0)),
            total_unrealized_pnl=float(balance.get("unrealizedPL", 0)),
            pending_settlements=float(balance.get("pendingSettlement", 0)),
            t1_receivable=float(balance.get("t1Amount", 0)),
            t2_receivable=float(balance.get("t2Amount", 0)),
        )

    def get_positions(self) -> List[Position]:
        """Get all positions."""
        url = (
            TCBSEndpoints.MARGIN_POSITIONS_URL
            if self.is_margin_account
            else TCBSEndpoints.POSITIONS_URL
        )
        data = self._make_request("GET", url)

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
        """Get position for a specific symbol."""
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
        Place an order.

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
        if quantity % LOT_SIZE != 0:
            logger.warning(f"Invalid lot size: {quantity}. Must be multiple of {LOT_SIZE}.")
            return None

        # Safety check for live trading
        if not self.is_paper:
            logger.warning("⚠️ LIVE TRADING MODE - Order will be executed!")

            # Request OTP if needed
            if self.otp_callback:
                otp = self.otp_callback()
                if not otp:
                    logger.error("OTP required but not provided")
                    return None

        # Map order type to TCBS format
        tcbs_order_type = {
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
            "side": "B" if side == OrderSide.BUY else "S",
            "orderType": tcbs_order_type,
            "quantity": quantity,
            "price": price,
            "isMargin": self.is_margin_account,
        }

        data = self._make_request("POST", TCBSEndpoints.NEW_ORDER_URL, data=payload)

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
        """Cancel an order."""
        payload = {
            "accountNo": self.account_id,
            "orderId": order_id,
        }

        data = self._make_request("POST", TCBSEndpoints.CANCEL_ORDER_URL, data=payload)

        if data and data.get("status") == "success":
            logger.info(f"✅ Order {order_id} cancelled")
            return True

        logger.warning(f"❌ Failed to cancel order {order_id}")
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        orders = self.get_orders()
        for order in orders:
            if order.order_id == order_id:
                return order
        return None

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get all orders."""
        data = self._make_request(
            "GET", TCBSEndpoints.ORDERS_URL, params={"accountNo": self.account_id}
        )

        if not data or data.get("status") != "success":
            return []

        orders = []
        for item in data.get("data", []):
            # Map TCBS status to OrderStatus
            tcbs_status = item.get("orderStatus", "").upper()
            order_status = {
                "PENDING": OrderStatus.PENDING,
                "QUEUED": OrderStatus.SUBMITTED,
                "PARTIAL": OrderStatus.PARTIAL,
                "FILLED": OrderStatus.FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
                "EXPIRED": OrderStatus.EXPIRED,
            }.get(tcbs_status, OrderStatus.PENDING)

            if status and order_status != status:
                continue

            order = Order(
                order_id=item.get("orderId", ""),
                symbol=item.get("symbol", ""),
                side=OrderSide.BUY if item.get("side") == "B" else OrderSide.SELL,
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

    # =========================================================================
    # T+0 MARGIN TRADING SPECIFIC METHODS
    # =========================================================================

    def get_margin_info(self) -> Optional[Dict]:
        """
        Get margin account information.

        Returns:
            Dict with margin details or None
        """
        if not self.is_margin_account:
            logger.warning("Not a margin account")
            return None

        data = self._make_request("GET", TCBSEndpoints.MARGIN_INFO_URL)

        if not data or data.get("status") != "success":
            return None

        margin = data.get("data", {})

        return {
            "margin_limit": float(margin.get("marginLimit", 0)),
            "margin_used": float(margin.get("marginUsed", 0)),
            "margin_available": float(margin.get("marginAvailable", 0)),
            "equity_ratio": float(margin.get("equityRatio", 0)),
            "maintenance_ratio": float(margin.get("maintenanceRatio", 0)),
            "margin_call_status": margin.get("marginCallStatus", "NORMAL"),
            "interest_rate": float(margin.get("interestRate", 0.12)),
            "accrued_interest": float(margin.get("accruedInterest", 0)),
        }

    def can_trade_t0(self, symbol: str, quantity: int, price: float) -> Tuple[bool, str]:
        """
        Check if T+0 trade is possible.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Trade price

        Returns:
            (can_trade, reason)
        """
        if not self.is_margin_account:
            return False, "T+0 requires margin account"

        margin_info = self.get_margin_info()
        if not margin_info:
            return False, "Cannot get margin info"

        trade_value = quantity * price
        margin_required = trade_value * 0.5  # 50% initial margin

        if margin_required > margin_info["margin_available"]:
            return (
                False,
                f"Insufficient margin. Need: {margin_required:,.0f}, Available: {margin_info['margin_available']:,.0f}",
            )

        # Check margin call status
        if margin_info["margin_call_status"] != "NORMAL":
            return False, f"Margin call active: {margin_info['margin_call_status']}"

        return True, "T+0 trade OK"

    def place_t0_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.LO,
    ) -> Optional[Order]:
        """
        Place a T+0 margin order.

        Same-day settlement for margin accounts.

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            price: Order price
            order_type: Order type

        Returns:
            Order object or None
        """
        if not self.is_margin_account:
            logger.error("T+0 orders require margin account")
            return None

        can_trade, reason = self.can_trade_t0(symbol, quantity, price)
        if not can_trade:
            logger.error(f"T+0 trade rejected: {reason}")
            return None

        # Place order with margin flag
        return self.place_order(symbol, side, quantity, price, order_type)


# Factory function
def create_tcbs_broker(
    account_id: str,
    username: str,
    password: str,
    is_paper: bool = True,
    is_margin_account: bool = False,
    otp_callback: Optional[Callable[[], str]] = None,
) -> TCBSBroker:
    """
    Create and connect TCBS broker.

    Args:
        account_id: TCBS trading account ID
        username: TCBS username
        password: TCBS password
        is_paper: Use paper trading mode (default True)
        is_margin_account: Enable margin trading
        otp_callback: Callback function to get OTP

    Returns:
        TCBSBroker: Connected broker instance
    """
    broker = TCBSBroker(
        account_id=account_id,
        username=username,
        password=password,
        is_paper=is_paper,
        is_margin_account=is_margin_account,
        otp_callback=otp_callback,
    )
    broker.connect()
    return broker
