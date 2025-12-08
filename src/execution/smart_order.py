# -*- coding: utf-8 -*-
"""
Smart Order Execution for Vietnam Stock Market

Implements intelligent order execution strategies:
- TWAP (Time-Weighted Average Price): Split orders over time
- VWAP (Volume-Weighted Average Price): Execute based on volume profile
- Participation Rate: Limit market impact
- Iceberg Orders: Hide large order size

Vietnam Market Considerations:
- Lot size: 100 shares minimum
- Trading hours: 9:00-11:30, 13:00-14:45
- ATO (9:00-9:15) and ATC (14:30-14:45) auction sessions
- ±7% daily price limit (HOSE)
- Lower liquidity than developed markets

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
from threading import Thread, Event
import math

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Order execution strategies"""

    MARKET = "MARKET"  # Single market order
    LIMIT = "LIMIT"  # Single limit order
    TWAP = "TWAP"  # Time-weighted average price
    VWAP = "VWAP"  # Volume-weighted average price
    ICEBERG = "ICEBERG"  # Hidden size orders
    PARTICIPATION = "PARTICIPATION"  # % of volume


@dataclass
class ExecutionSlice:
    """Individual order slice in execution plan"""

    slice_id: int
    quantity: int
    target_time: datetime
    price_limit: Optional[float] = None
    executed: bool = False
    executed_quantity: int = 0
    executed_price: float = 0.0
    executed_time: Optional[datetime] = None
    order_id: Optional[str] = None
    status: str = "PENDING"


@dataclass
class ExecutionPlan:
    """Complete execution plan for an order"""

    symbol: str
    side: str  # BUY or SELL
    total_quantity: int
    strategy: ExecutionStrategy
    slices: List[ExecutionSlice] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_participation_rate: float = 0.05  # Max 5% of volume
    price_limit: Optional[float] = None
    urgency: str = "NORMAL"  # LOW, NORMAL, HIGH

    @property
    def executed_quantity(self) -> int:
        return sum(s.executed_quantity for s in self.slices)

    @property
    def remaining_quantity(self) -> int:
        return self.total_quantity - self.executed_quantity

    @property
    def avg_executed_price(self) -> float:
        total_value = sum(s.executed_quantity * s.executed_price for s in self.slices)
        total_qty = self.executed_quantity
        return total_value / total_qty if total_qty > 0 else 0.0

    @property
    def completion_pct(self) -> float:
        return (
            (self.executed_quantity / self.total_quantity * 100) if self.total_quantity > 0 else 0.0
        )

    @property
    def is_complete(self) -> bool:
        return self.executed_quantity >= self.total_quantity


@dataclass
class VolumeProfile:
    """Intraday volume profile for VWAP calculation"""

    # Vietnam market typical volume distribution
    # Morning session: 9:00-11:30 (~55% of daily volume)
    # Afternoon session: 13:00-14:45 (~45% of daily volume)

    intervals: Dict[str, float] = field(
        default_factory=lambda: {
            "09:00-09:15": 0.12,  # ATO - high volume
            "09:15-09:30": 0.08,
            "09:30-10:00": 0.10,
            "10:00-10:30": 0.08,
            "10:30-11:00": 0.08,
            "11:00-11:30": 0.09,  # Pre-lunch
            "13:00-13:30": 0.10,  # Post-lunch
            "13:30-14:00": 0.08,
            "14:00-14:30": 0.12,  # Pre-ATC
            "14:30-14:45": 0.15,  # ATC - highest volume
        }
    )

    def get_weight(self, time_str: str) -> float:
        """Get volume weight for a time interval"""
        return self.intervals.get(time_str, 0.05)


class SmartOrderExecutor:
    """
    Smart Order Execution Engine

    Splits large orders into smaller slices to minimize market impact.
    Supports TWAP, VWAP, Iceberg, and Participation strategies.

    Usage:
        executor = SmartOrderExecutor(broker=my_broker)

        # TWAP execution over 2 hours
        plan = executor.create_twap_plan("VNM", "BUY", 1000, duration_minutes=120)
        executor.execute_plan(plan)

        # VWAP execution
        plan = executor.create_vwap_plan("HPG", "BUY", 2000)
        executor.execute_plan(plan)
    """

    # Vietnam market constants
    LOT_SIZE = 100
    MIN_SLICE_VALUE = 10_000_000  # 10M VND minimum per slice
    MAX_SLICES = 20  # Maximum number of slices

    # Trading session times (Vietnam)
    MORNING_START = (9, 0)
    MORNING_END = (11, 30)
    AFTERNOON_START = (13, 0)
    AFTERNOON_END = (14, 45)

    def __init__(
        self,
        broker=None,
        get_price_func: Optional[Callable[[str], float]] = None,
        get_volume_func: Optional[Callable[[str], float]] = None,
        default_participation_rate: float = 0.05,
        min_interval_seconds: int = 30,
    ):
        """
        Initialize Smart Order Executor.

        Args:
            broker: Broker instance for order execution
            get_price_func: Function to get current price
            get_volume_func: Function to get current/average volume
            default_participation_rate: Default max % of volume (5%)
            min_interval_seconds: Minimum time between slices
        """
        self.broker = broker
        self.get_price = get_price_func or self._default_get_price
        self.get_volume = get_volume_func or self._default_get_volume
        self.default_participation_rate = default_participation_rate
        self.min_interval_seconds = min_interval_seconds

        self.volume_profile = VolumeProfile()
        self._active_plans: Dict[str, ExecutionPlan] = {}
        self._stop_event = Event()

    def _default_get_price(self, symbol: str) -> float:
        """Default price getter - should be overridden"""
        try:
            from src.data.loader import load_data

            df = load_data(symbol, lookback=5)
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    def _default_get_volume(self, symbol: str) -> float:
        """Default volume getter - returns average daily volume"""
        try:
            from src.data.loader import load_data

            df = load_data(symbol, lookback=20)
            if df is not None and not df.empty:
                return float(df["volume"].mean())
        except Exception:
            pass
        return 100000  # Default 100K shares

    def _round_to_lot(self, shares: int) -> int:
        """Round shares to valid lot size"""
        if shares <= 0:
            return 0
        rounded = (shares // self.LOT_SIZE) * self.LOT_SIZE
        return max(self.LOT_SIZE, rounded)

    def _is_trading_time(self, dt: Optional[datetime] = None) -> bool:
        """Check if given time is within trading hours"""
        if dt is None:
            dt = datetime.now()

        hour, minute = dt.hour, dt.minute
        time_val = hour * 60 + minute

        morning_start = self.MORNING_START[0] * 60 + self.MORNING_START[1]
        morning_end = self.MORNING_END[0] * 60 + self.MORNING_END[1]
        afternoon_start = self.AFTERNOON_START[0] * 60 + self.AFTERNOON_START[1]
        afternoon_end = self.AFTERNOON_END[0] * 60 + self.AFTERNOON_END[1]

        return (
            morning_start <= time_val <= morning_end or afternoon_start <= time_val <= afternoon_end
        )

    def _get_next_trading_time(self, dt: datetime) -> datetime:
        """Get next valid trading time"""
        hour, minute = dt.hour, dt.minute
        time_val = hour * 60 + minute

        morning_start = self.MORNING_START[0] * 60 + self.MORNING_START[1]
        morning_end = self.MORNING_END[0] * 60 + self.MORNING_END[1]
        afternoon_start = self.AFTERNOON_START[0] * 60 + self.AFTERNOON_START[1]

        if time_val < morning_start:
            return dt.replace(hour=self.MORNING_START[0], minute=self.MORNING_START[1])
        elif morning_end < time_val < afternoon_start:
            return dt.replace(hour=self.AFTERNOON_START[0], minute=self.AFTERNOON_START[1])
        elif time_val > self.AFTERNOON_END[0] * 60 + self.AFTERNOON_END[1]:
            # Next day
            next_day = dt + timedelta(days=1)
            return next_day.replace(hour=self.MORNING_START[0], minute=self.MORNING_START[1])

        return dt

    def _calculate_optimal_slices(
        self,
        total_quantity: int,
        price: float,
        avg_volume: float,
        duration_minutes: int,
    ) -> int:
        """Calculate optimal number of slices based on order size and liquidity"""
        # Order value
        order_value = total_quantity * price

        # Minimum slices based on value (1 slice per 10M VND)
        value_based_slices = max(1, int(order_value / self.MIN_SLICE_VALUE))

        # Slices based on participation rate
        # If order is 5% of daily volume, need ~20 slices to spread evenly
        volume_ratio = total_quantity / avg_volume if avg_volume > 0 else 0.1
        participation_slices = max(1, int(volume_ratio / self.default_participation_rate))

        # Slices based on duration
        time_based_slices = max(1, duration_minutes // 10)  # 1 slice per 10 minutes

        # Take maximum but cap at MAX_SLICES
        optimal = min(
            self.MAX_SLICES, max(value_based_slices, participation_slices, time_based_slices)
        )

        # Ensure each slice has at least 1 lot
        max_by_quantity = total_quantity // self.LOT_SIZE

        return max(1, min(optimal, max_by_quantity))

    # =========================================================================
    # TWAP (Time-Weighted Average Price)
    # =========================================================================

    def create_twap_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        duration_minutes: int = 60,
        start_time: Optional[datetime] = None,
        price_limit: Optional[float] = None,
    ) -> ExecutionPlan:
        """
        Create TWAP execution plan.

        Splits order into equal-sized slices executed at regular intervals.

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Total shares to execute
            duration_minutes: Time to complete execution
            start_time: When to start (default: now)
            price_limit: Maximum/minimum price limit

        Returns:
            ExecutionPlan with TWAP slices
        """
        quantity = self._round_to_lot(quantity)
        if quantity <= 0:
            raise ValueError("Quantity must be positive and at least 1 lot")

        price = self.get_price(symbol)
        avg_volume = self.get_volume(symbol)

        # Calculate number of slices
        num_slices = self._calculate_optimal_slices(quantity, price, avg_volume, duration_minutes)

        # Calculate quantity per slice
        base_qty = quantity // num_slices
        base_qty = self._round_to_lot(base_qty)
        remainder = quantity - (base_qty * num_slices)

        # Calculate time interval
        interval_seconds = (duration_minutes * 60) // num_slices
        interval_seconds = max(interval_seconds, self.min_interval_seconds)

        # Create slices
        start = start_time or datetime.now()
        start = self._get_next_trading_time(start)

        slices = []
        current_time = start

        for i in range(num_slices):
            # Add remainder to last slice
            slice_qty = base_qty + (remainder if i == num_slices - 1 else 0)
            slice_qty = self._round_to_lot(slice_qty)

            if slice_qty > 0:
                slices.append(
                    ExecutionSlice(
                        slice_id=i + 1,
                        quantity=slice_qty,
                        target_time=current_time,
                        price_limit=price_limit,
                    )
                )

            # Move to next interval, skipping lunch break
            current_time = current_time + timedelta(seconds=interval_seconds)
            current_time = self._get_next_trading_time(current_time)

        plan = ExecutionPlan(
            symbol=symbol,
            side=side.upper(),
            total_quantity=quantity,
            strategy=ExecutionStrategy.TWAP,
            slices=slices,
            start_time=start,
            end_time=current_time,
            price_limit=price_limit,
        )

        logger.info(
            f"📊 TWAP Plan created: {symbol} {side} {quantity} shares "
            f"in {num_slices} slices over {duration_minutes} minutes"
        )

        return plan

    # =========================================================================
    # VWAP (Volume-Weighted Average Price)
    # =========================================================================

    def create_vwap_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        price_limit: Optional[float] = None,
    ) -> ExecutionPlan:
        """
        Create VWAP execution plan.

        Distributes order based on historical volume profile.
        More shares during high-volume periods (ATO, ATC).

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Total shares to execute
            start_time: When to start (default: market open)
            end_time: When to finish (default: market close)
            price_limit: Maximum/minimum price limit

        Returns:
            ExecutionPlan with VWAP slices
        """
        quantity = self._round_to_lot(quantity)
        if quantity <= 0:
            raise ValueError("Quantity must be positive and at least 1 lot")

        now = datetime.now()
        start = start_time or now.replace(hour=9, minute=0, second=0)
        end = end_time or now.replace(hour=14, minute=45, second=0)

        # Create slices based on volume profile
        slices = []
        slice_id = 1
        remaining = quantity

        for interval, weight in self.volume_profile.intervals.items():
            # Parse interval time
            start_str, end_str = interval.split("-")
            start_hour, start_min = map(int, start_str.split(":"))

            # Calculate quantity for this interval
            interval_qty = int(quantity * weight)
            interval_qty = self._round_to_lot(interval_qty)

            if interval_qty > 0 and remaining > 0:
                actual_qty = min(interval_qty, remaining)
                actual_qty = self._round_to_lot(actual_qty)

                if actual_qty > 0:
                    target_time = now.replace(hour=start_hour, minute=start_min, second=0)

                    slices.append(
                        ExecutionSlice(
                            slice_id=slice_id,
                            quantity=actual_qty,
                            target_time=target_time,
                            price_limit=price_limit,
                        )
                    )

                    remaining -= actual_qty
                    slice_id += 1

        # Add any remainder to last slice
        if remaining > 0 and slices:
            slices[-1].quantity += self._round_to_lot(remaining)

        plan = ExecutionPlan(
            symbol=symbol,
            side=side.upper(),
            total_quantity=quantity,
            strategy=ExecutionStrategy.VWAP,
            slices=slices,
            start_time=start,
            end_time=end,
            price_limit=price_limit,
        )

        logger.info(
            f"📊 VWAP Plan created: {symbol} {side} {quantity} shares "
            f"in {len(slices)} slices following volume profile"
        )

        return plan

    # =========================================================================
    # ICEBERG Orders
    # =========================================================================

    def create_iceberg_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        visible_quantity: int,
        price_limit: float,
        refresh_interval_seconds: int = 60,
    ) -> ExecutionPlan:
        """
        Create Iceberg execution plan.

        Shows only a portion of the order (visible_quantity) at a time.
        Refreshes when filled.

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Total shares to execute
            visible_quantity: Visible portion per slice
            price_limit: Limit price for all slices
            refresh_interval_seconds: Time between refreshes

        Returns:
            ExecutionPlan with Iceberg slices
        """
        quantity = self._round_to_lot(quantity)
        visible_quantity = self._round_to_lot(visible_quantity)

        if visible_quantity <= 0:
            visible_quantity = self.LOT_SIZE

        num_slices = math.ceil(quantity / visible_quantity)

        slices = []
        current_time = datetime.now()
        remaining = quantity

        for i in range(num_slices):
            slice_qty = min(visible_quantity, remaining)
            slice_qty = self._round_to_lot(slice_qty)

            if slice_qty > 0:
                slices.append(
                    ExecutionSlice(
                        slice_id=i + 1,
                        quantity=slice_qty,
                        target_time=current_time,
                        price_limit=price_limit,
                    )
                )
                remaining -= slice_qty
                current_time = current_time + timedelta(seconds=refresh_interval_seconds)

        plan = ExecutionPlan(
            symbol=symbol,
            side=side.upper(),
            total_quantity=quantity,
            strategy=ExecutionStrategy.ICEBERG,
            slices=slices,
            price_limit=price_limit,
        )

        logger.info(
            f"🧊 Iceberg Plan created: {symbol} {side} {quantity} shares "
            f"with {visible_quantity} visible per slice"
        )

        return plan

    # =========================================================================
    # Participation Rate Strategy
    # =========================================================================

    def create_participation_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        participation_rate: float = 0.05,
        price_limit: Optional[float] = None,
    ) -> ExecutionPlan:
        """
        Create Participation Rate execution plan.

        Limits execution to a percentage of market volume.
        Reduces market impact for large orders.

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Total shares to execute
            participation_rate: Max % of volume (default 5%)
            price_limit: Maximum/minimum price limit

        Returns:
            ExecutionPlan with participation-based slices
        """
        quantity = self._round_to_lot(quantity)
        avg_volume = self.get_volume(symbol)

        # Calculate how many intervals needed
        # If avg daily volume is 1M and we want 50K at 5% rate
        # We need 50K / (1M * 5%) = 1 interval
        max_per_interval = avg_volume * participation_rate / 10  # 10 intervals per day
        max_per_interval = self._round_to_lot(int(max_per_interval))

        if max_per_interval <= 0:
            max_per_interval = self.LOT_SIZE

        num_slices = math.ceil(quantity / max_per_interval)
        num_slices = min(num_slices, self.MAX_SLICES)

        # Distribute evenly
        base_qty = quantity // num_slices
        base_qty = self._round_to_lot(base_qty)

        slices = []
        current_time = datetime.now()
        interval = timedelta(minutes=30)  # 30 minutes between slices
        remaining = quantity

        for i in range(num_slices):
            slice_qty = min(base_qty, remaining)
            if i == num_slices - 1:
                slice_qty = remaining  # Last slice gets remainder
            slice_qty = self._round_to_lot(slice_qty)

            if slice_qty > 0:
                current_time = self._get_next_trading_time(current_time)
                slices.append(
                    ExecutionSlice(
                        slice_id=i + 1,
                        quantity=slice_qty,
                        target_time=current_time,
                        price_limit=price_limit,
                    )
                )
                remaining -= slice_qty
                current_time = current_time + interval

        plan = ExecutionPlan(
            symbol=symbol,
            side=side.upper(),
            total_quantity=quantity,
            strategy=ExecutionStrategy.PARTICIPATION,
            slices=slices,
            max_participation_rate=participation_rate,
            price_limit=price_limit,
        )

        logger.info(
            f"📈 Participation Plan created: {symbol} {side} {quantity} shares "
            f"at {participation_rate*100:.1f}% participation rate"
        )

        return plan

    # =========================================================================
    # Execution Engine
    # =========================================================================

    def execute_slice(self, plan: ExecutionPlan, slice_: ExecutionSlice) -> bool:
        """
        Execute a single slice.

        Args:
            plan: Parent execution plan
            slice_: Slice to execute

        Returns:
            True if executed successfully
        """
        if slice_.executed:
            return True

        if self.broker is None:
            logger.warning("No broker configured - simulating execution")
            # Simulate execution
            slice_.executed = True
            slice_.executed_quantity = slice_.quantity
            slice_.executed_price = self.get_price(plan.symbol)
            slice_.executed_time = datetime.now()
            slice_.status = "FILLED"
            return True

        try:
            # Place order via broker
            from src.broker.base_broker import OrderSide, OrderType

            side = OrderSide.BUY if plan.side == "BUY" else OrderSide.SELL
            order_type = OrderType.LO if slice_.price_limit else OrderType.MP

            order = self.broker.place_order(
                symbol=plan.symbol,
                side=side,
                quantity=slice_.quantity,
                price=slice_.price_limit or self.get_price(plan.symbol),
                order_type=order_type,
            )

            if order:
                slice_.order_id = order.order_id
                slice_.status = "SUBMITTED"

                # Wait for fill (simplified - real implementation would poll)
                time.sleep(2)

                # Check order status
                updated_order = self.broker.get_order(order.order_id)
                if updated_order and updated_order.filled_quantity > 0:
                    slice_.executed = True
                    slice_.executed_quantity = updated_order.filled_quantity
                    slice_.executed_price = updated_order.filled_price
                    slice_.executed_time = datetime.now()
                    slice_.status = "FILLED"
                    return True

        except Exception as e:
            logger.error(f"Slice execution failed: {e}")
            slice_.status = "FAILED"

        return False

    def execute_plan(
        self,
        plan: ExecutionPlan,
        async_execution: bool = False,
    ) -> ExecutionPlan:
        """
        Execute an entire plan.

        Args:
            plan: Execution plan to run
            async_execution: Run in background thread

        Returns:
            Updated execution plan
        """
        if async_execution:
            thread = Thread(target=self._execute_plan_sync, args=(plan,))
            thread.daemon = True
            thread.start()
            self._active_plans[plan.symbol] = plan
            return plan

        return self._execute_plan_sync(plan)

    def _execute_plan_sync(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Synchronous plan execution"""
        logger.info(f"🚀 Starting {plan.strategy.value} execution for {plan.symbol}")

        for slice_ in plan.slices:
            if self._stop_event.is_set():
                logger.info("Execution stopped by user")
                break

            # Wait until target time
            now = datetime.now()
            if slice_.target_time > now:
                wait_seconds = (slice_.target_time - now).total_seconds()
                if wait_seconds > 0:
                    logger.debug(f"Waiting {wait_seconds:.0f}s for slice {slice_.slice_id}")
                    time.sleep(min(wait_seconds, 60))  # Max 60s wait

            # Check if trading time
            if not self._is_trading_time():
                logger.info("Outside trading hours - pausing execution")
                continue

            # Execute slice
            success = self.execute_slice(plan, slice_)

            if success:
                logger.info(
                    f"✅ Slice {slice_.slice_id}/{len(plan.slices)} executed: "
                    f"{slice_.executed_quantity} @ {slice_.executed_price:,.0f} | "
                    f"Progress: {plan.completion_pct:.1f}%"
                )
            else:
                logger.warning(f"⚠️ Slice {slice_.slice_id} failed")

        logger.info(
            f"📊 Execution complete: {plan.symbol} {plan.side} "
            f"{plan.executed_quantity}/{plan.total_quantity} shares "
            f"@ avg {plan.avg_executed_price:,.0f}"
        )

        return plan

    def stop_execution(self, symbol: Optional[str] = None):
        """Stop execution for a symbol or all"""
        self._stop_event.set()
        if symbol and symbol in self._active_plans:
            del self._active_plans[symbol]
        logger.info(f"Execution stopped for {symbol or 'all symbols'}")

    def get_execution_status(self, symbol: str) -> Optional[Dict]:
        """Get execution status for a symbol"""
        plan = self._active_plans.get(symbol)
        if not plan:
            return None

        return {
            "symbol": plan.symbol,
            "side": plan.side,
            "strategy": plan.strategy.value,
            "total_quantity": plan.total_quantity,
            "executed_quantity": plan.executed_quantity,
            "remaining_quantity": plan.remaining_quantity,
            "avg_price": plan.avg_executed_price,
            "completion_pct": plan.completion_pct,
            "slices_total": len(plan.slices),
            "slices_executed": sum(1 for s in plan.slices if s.executed),
        }


# Singleton instance
_executor_instance: Optional[SmartOrderExecutor] = None


def get_smart_executor(broker=None) -> SmartOrderExecutor:
    """Get singleton instance of smart order executor"""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = SmartOrderExecutor(broker=broker)
    return _executor_instance
