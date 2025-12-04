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
import random
import time
import threading

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


@dataclass
class SimulationConfig:
    """Configuration for realistic order simulation"""

    # Slippage settings
    enable_slippage: bool = True
    slippage_min_pct: float = 0.001  # 0.1%
    slippage_max_pct: float = 0.003  # 0.3%

    # Partial fill settings
    enable_partial_fills: bool = True
    fill_batch_size: int = 100  # Fill in batches of 100 shares
    partial_fill_probability: float = 0.3  # 30% chance of partial fill

    # Latency settings
    enable_latency: bool = True
    latency_min_ms: int = 1000  # 1 second
    latency_max_ms: int = 3000  # 3 seconds

    # Market impact (for large orders)
    enable_market_impact: bool = True
    market_impact_threshold: int = 10000  # Orders > 10k shares
    market_impact_factor: float = 0.001  # 0.1% per 10k shares


class SimulatedBroker(BaseBroker):
    """
    Simulated broker for paper trading and testing

    Features:
    - Slippage simulation (±0.1-0.3% random)
    - Partial fills (fill in batches of 100 shares)
    - Order latency (1-3s delay before fill)
    - Market impact for large orders
    """

    def __init__(
        self,
        account_id: str = "PAPER",
        initial_cash: float = 100_000_000,  # 100M VND
        simulation_config: Optional[SimulationConfig] = None,
        realistic_mode: bool = False,  # Enable all realistic features
    ):
        super().__init__(account_id)
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        self._connected = True
        self._pending_fills: List[Dict] = []  # For async partial fills

        # Simulation config
        if simulation_config:
            self._sim_config = simulation_config
        elif realistic_mode:
            self._sim_config = SimulationConfig()  # All features enabled by default
        else:
            # Legacy mode - immediate fills, no slippage
            self._sim_config = SimulationConfig(
                enable_slippage=False,
                enable_partial_fills=False,
                enable_latency=False,
                enable_market_impact=False,
            )

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

    def _calculate_slippage(self, order: Order) -> float:
        """Calculate slippage based on order side and config"""
        if not self._sim_config.enable_slippage:
            return 0.0

        slippage_pct = random.uniform(
            self._sim_config.slippage_min_pct, self._sim_config.slippage_max_pct
        )

        # BUY orders get worse (higher) price, SELL orders get worse (lower) price
        if order.side == OrderSide.BUY:
            return slippage_pct
        else:
            return -slippage_pct

    def _calculate_market_impact(self, order: Order) -> float:
        """Calculate market impact for large orders"""
        if not self._sim_config.enable_market_impact:
            return 0.0

        if order.quantity <= self._sim_config.market_impact_threshold:
            return 0.0

        # Impact increases with order size
        size_multiplier = order.quantity / self._sim_config.market_impact_threshold
        impact_pct = self._sim_config.market_impact_factor * size_multiplier

        # BUY pushes price up, SELL pushes price down
        if order.side == OrderSide.BUY:
            return impact_pct
        else:
            return -impact_pct

    def _get_fill_price(self, order: Order) -> float:
        """Calculate realistic fill price with slippage and market impact"""
        base_price = order.price

        slippage = self._calculate_slippage(order)
        market_impact = self._calculate_market_impact(order)

        total_adjustment = slippage + market_impact
        fill_price = base_price * (1 + total_adjustment)

        # Round to valid price tick (Vietnam market: 10 VND for most stocks)
        fill_price = round(fill_price / 10) * 10

        if total_adjustment != 0:
            logger.debug(
                f"Price adjustment for {order.symbol}: "
                f"slippage={slippage*100:.2f}%, impact={market_impact*100:.2f}%, "
                f"base={base_price:,.0f} -> fill={fill_price:,.0f}"
            )

        return fill_price

    def _simulate_fill(self, order: Order, async_mode: bool = False) -> None:
        """
        Simulate order fill with realistic behavior

        Features:
        - Slippage: ±0.1-0.3% random price adjustment
        - Partial fills: Fill in batches of 100 shares
        - Latency: 1-3s delay before fill
        - Market impact: Additional slippage for large orders
        """
        # Simulate latency
        if self._sim_config.enable_latency and not async_mode:
            latency_ms = random.randint(
                self._sim_config.latency_min_ms, self._sim_config.latency_max_ms
            )
            logger.debug(f"Simulating {latency_ms}ms latency for order {order.order_id}")
            time.sleep(latency_ms / 1000.0)

        # Calculate fill price with slippage and market impact
        fill_price = self._get_fill_price(order)

        # Determine fill quantity (partial vs full)
        if (
            self._sim_config.enable_partial_fills
            and order.quantity > self._sim_config.fill_batch_size
        ):
            # Chance of partial fill
            if random.random() < self._sim_config.partial_fill_probability:
                # Fill in batches
                fill_qty = self._sim_config.fill_batch_size
                self._execute_partial_fill(order, fill_qty, fill_price)

                # Schedule remaining fills
                self._schedule_remaining_fills(order)
                return

        # Full fill
        self._execute_full_fill(order, fill_price)

    def _execute_partial_fill(self, order: Order, fill_qty: int, fill_price: float) -> None:
        """Execute a partial fill"""
        order.status = OrderStatus.PARTIAL
        order.filled_quantity += fill_qty

        # Weighted average fill price
        if order.filled_quantity == fill_qty:
            order.filled_price = fill_price
        else:
            prev_value = (order.filled_quantity - fill_qty) * order.filled_price
            new_value = fill_qty * fill_price
            order.filled_price = (prev_value + new_value) / order.filled_quantity

        order.updated_at = datetime.now()

        # Update position and cash for this partial fill
        self._update_position_and_cash(order.symbol, order.side, fill_qty, fill_price)

        self._notify_order_update(order)

        logger.info(
            f"📝 Partial fill: {order.side.value} {fill_qty}/{order.quantity} {order.symbol} "
            f"@ {fill_price:,.0f} VND (total filled: {order.filled_quantity})"
        )

    def _execute_full_fill(self, order: Order, fill_price: float) -> None:
        """Execute a full fill (or complete remaining quantity)"""
        remaining = order.remaining_quantity

        # Update weighted average price
        if order.filled_quantity > 0:
            prev_value = order.filled_quantity * order.filled_price
            new_value = remaining * fill_price
            order.filled_price = (prev_value + new_value) / order.quantity
        else:
            order.filled_price = fill_price

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_at = datetime.now()
        order.updated_at = datetime.now()

        # Update position and cash
        self._update_position_and_cash(order.symbol, order.side, remaining, fill_price)

        self._notify_order_update(order)

        slippage_info = ""
        if order.filled_price != order.price:
            slippage_pct = (order.filled_price - order.price) / order.price * 100
            slippage_info = f" (slippage: {slippage_pct:+.2f}%)"

        logger.info(
            f"📝 Order filled: {order.side.value} {order.quantity} {order.symbol} "
            f"@ {order.filled_price:,.0f} VND{slippage_info}"
        )

    def _update_position_and_cash(
        self, symbol: str, side: OrderSide, quantity: int, price: float
    ) -> None:
        """Update position and cash for a fill"""
        if side == OrderSide.BUY:
            self._cash -= quantity * price

            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos.quantity + quantity
                total_cost = pos.cost_basis + (quantity * price)
                pos.quantity = total_qty
                pos.avg_price = total_cost / total_qty
                pos.cost_basis = total_cost
                pos.market_value = total_qty * pos.market_price
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    market_price=price,
                    cost_basis=quantity * price,
                    market_value=quantity * price,
                    available_quantity=quantity,
                    account_id=self.account_id,
                )
        else:  # SELL
            self._cash += quantity * price

            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.realized_pnl += (price - pos.avg_price) * quantity
                pos.quantity -= quantity

                if pos.quantity <= 0:
                    del self._positions[symbol]
                else:
                    pos.cost_basis = pos.quantity * pos.avg_price
                    pos.market_value = pos.quantity * pos.market_price

    def _schedule_remaining_fills(self, order: Order) -> None:
        """Schedule remaining fills for partial fill orders"""

        def fill_remaining():
            while order.remaining_quantity > 0 and order.status == OrderStatus.PARTIAL:
                # Random delay between partial fills
                delay = random.uniform(0.5, 2.0)
                time.sleep(delay)

                fill_qty = min(self._sim_config.fill_batch_size, order.remaining_quantity)
                fill_price = self._get_fill_price(order)

                if order.remaining_quantity <= fill_qty:
                    self._execute_full_fill(order, fill_price)
                else:
                    self._execute_partial_fill(order, fill_qty, fill_price)

        # Run in background thread
        thread = threading.Thread(target=fill_remaining, daemon=True)
        thread.start()

    def process_pending_fills(self) -> None:
        """Process any pending partial fills (for synchronous testing)"""
        for order in self._orders.values():
            if order.status == OrderStatus.PARTIAL:
                fill_price = self._get_fill_price(order)
                self._execute_full_fill(order, fill_price)

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
