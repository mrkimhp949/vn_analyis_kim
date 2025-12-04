# -*- coding: utf-8 -*-
"""
Broker Integration Base Classes

Abstract base classes for broker API integration:
- Order placement
- Position management
- Account information
- Real-time order status

Supported brokers:
- SSI Securities
- VNDirect
- TCBS
- MBS

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side"""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order types for Vietnam market"""

    LO = "LO"  # Limit Order
    ATO = "ATO"  # At The Open
    ATC = "ATC"  # At The Close
    MP = "MP"  # Market Price
    MTL = "MTL"  # Market To Limit
    MOK = "MOK"  # Match Or Kill
    MAK = "MAK"  # Match And Kill


class OrderStatus(Enum):
    """Order status"""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """Order data"""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float

    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None

    # Metadata
    account_id: str = ""
    broker: str = ""
    message: str = ""

    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]

    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0


@dataclass
class Position:
    """Position data"""

    symbol: str
    quantity: int
    avg_price: float
    market_price: float

    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0

    # Market data
    market_value: float = 0.0
    cost_basis: float = 0.0

    # Settlement
    available_quantity: int = 0  # Available for selling (T+2 settled)
    pending_buy: int = 0
    pending_sell: int = 0

    # Metadata
    account_id: str = ""
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AccountInfo:
    """Account information"""

    account_id: str
    account_name: str
    broker: str

    # Balances
    total_equity: float = 0.0
    cash_balance: float = 0.0
    buying_power: float = 0.0

    # Margin
    margin_used: float = 0.0
    margin_available: float = 0.0
    margin_ratio: float = 0.0

    # Positions
    total_market_value: float = 0.0
    total_cost_basis: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0

    # Settlement
    pending_settlements: float = 0.0
    t1_receivable: float = 0.0
    t2_receivable: float = 0.0

    # Status
    is_active: bool = True
    last_updated: datetime = field(default_factory=datetime.now)


class BaseBroker(ABC):
    """
    Abstract base class for broker integration

    Implement this class for each broker:
    - SSIBroker
    - VNDirectBroker
    - TCBSBroker
    """

    def __init__(self, account_id: str, api_key: str = "", api_secret: str = ""):
        self.account_id = account_id
        self.api_key = api_key
        self.api_secret = api_secret
        self._connected = False
        self._order_callbacks: List[Callable[[Order], None]] = []

    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Return broker name"""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker API"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker API"""
        pass

    @abstractmethod
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all positions"""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.LO,
    ) -> Optional[Order]:
        """Place an order"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        pass

    @abstractmethod
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get all orders, optionally filtered by status"""
        pass

    def register_order_callback(self, callback: Callable[[Order], None]) -> None:
        """Register callback for order updates"""
        self._order_callbacks.append(callback)

    def _notify_order_update(self, order: Order) -> None:
        """Notify all callbacks of order update"""
        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Order callback error: {e}")

    # Convenience methods
    def buy(
        self, symbol: str, quantity: int, price: float, order_type: OrderType = OrderType.LO
    ) -> Optional[Order]:
        """Place buy order"""
        return self.place_order(symbol, OrderSide.BUY, quantity, price, order_type)

    def sell(
        self, symbol: str, quantity: int, price: float, order_type: OrderType = OrderType.LO
    ) -> Optional[Order]:
        """Place sell order"""
        return self.place_order(symbol, OrderSide.SELL, quantity, price, order_type)

    def get_buying_power(self) -> float:
        """Get available buying power"""
        account = self.get_account_info()
        return account.buying_power if account else 0.0

    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        account = self.get_account_info()
        return account.total_equity if account else 0.0


class SimulatedBroker(BaseBroker):
    """
    Simulated broker for paper trading and testing
    """

    def __init__(self, account_id: str = "PAPER", initial_cash: float = 100_000_000):  # 100M VND
        super().__init__(account_id)
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        self._connected = True

    @property
    def broker_name(self) -> str:
        return "SIMULATED"

    def connect(self) -> bool:
        self._connected = True
        logger.info("✅ Connected to simulated broker")
        return True

    def disconnect(self) -> None:
        self._connected = False

    def get_account_info(self) -> AccountInfo:
        total_market_value = sum(p.market_value for p in self._positions.values())
        total_cost = sum(p.cost_basis for p in self._positions.values())
        total_pnl = sum(p.unrealized_pnl for p in self._positions.values())

        return AccountInfo(
            account_id=self.account_id,
            account_name="Paper Trading Account",
            broker=self.broker_name,
            total_equity=self._cash + total_market_value,
            cash_balance=self._cash,
            buying_power=self._cash,
            total_market_value=total_market_value,
            total_cost_basis=total_cost,
            total_unrealized_pnl=total_pnl,
        )

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        order_type: OrderType = OrderType.LO,
    ) -> Optional[Order]:
        # Validate lot size
        if quantity % 100 != 0:
            logger.warning(f"Invalid lot size: {quantity}")
            return None

        # Check buying power for buy orders
        if side == OrderSide.BUY:
            required = quantity * price
            if required > self._cash:
                logger.warning(f"Insufficient buying power: {required:,.0f} > {self._cash:,.0f}")
                return None

        # Check position for sell orders
        if side == OrderSide.SELL:
            position = self._positions.get(symbol)
            if not position or position.quantity < quantity:
                logger.warning(f"Insufficient position for {symbol}")
                return None

        # Create order
        self._order_counter += 1
        order_id = f"SIM{self._order_counter:06d}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.SUBMITTED,
            account_id=self.account_id,
            broker=self.broker_name,
        )

        self._orders[order_id] = order

        # Simulate immediate fill for paper trading
        self._simulate_fill(order)

        return order

    def _simulate_fill(self, order: Order) -> None:
        """Simulate order fill"""
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = order.price
        order.filled_at = datetime.now()
        order.updated_at = datetime.now()

        # Update position and cash
        if order.side == OrderSide.BUY:
            self._cash -= order.quantity * order.price

            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                total_qty = pos.quantity + order.quantity
                total_cost = pos.cost_basis + (order.quantity * order.price)
                pos.quantity = total_qty
                pos.avg_price = total_cost / total_qty
                pos.cost_basis = total_cost
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=order.price,
                    market_price=order.price,
                    cost_basis=order.quantity * order.price,
                    market_value=order.quantity * order.price,
                    available_quantity=order.quantity,
                    account_id=self.account_id,
                )

        else:  # SELL
            self._cash += order.quantity * order.price

            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                pos.quantity -= order.quantity
                pos.realized_pnl += (order.price - pos.avg_price) * order.quantity

                if pos.quantity <= 0:
                    del self._positions[order.symbol]
                else:
                    pos.cost_basis = pos.quantity * pos.avg_price

        # Notify callbacks
        self._notify_order_update(order)

        logger.info(
            f"📝 Order filled: {order.side.value} {order.quantity} {order.symbol} "
            f"@ {order.price:,.0f} VND"
        )

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.is_active:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                self._notify_order_update(order)
                return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return orders

    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """Update market prices for positions"""
        for symbol, price in prices.items():
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.market_price = price
                pos.market_value = pos.quantity * price
                pos.unrealized_pnl = pos.market_value - pos.cost_basis
                pos.unrealized_pnl_pct = (
                    (pos.unrealized_pnl / pos.cost_basis * 100) if pos.cost_basis > 0 else 0
                )
                pos.last_updated = datetime.now()

    def reset(self) -> None:
        """Reset paper trading account"""
        self._cash = self._initial_cash
        self._positions.clear()
        self._orders.clear()
        self._order_counter = 0
        logger.info("Paper trading account reset")


# Singleton for paper trading
_paper_broker: Optional[SimulatedBroker] = None


def get_paper_broker(initial_cash: float = 100_000_000) -> SimulatedBroker:
    """Get singleton paper trading broker"""
    global _paper_broker
    if _paper_broker is None:
        _paper_broker = SimulatedBroker(initial_cash=initial_cash)
    return _paper_broker
