# -*- coding: utf-8 -*-
"""
Order Guard - Duplicate Prevention & Order Management

Features:
- Prevent duplicate orders within time window
- Track pending orders
- Auto-cancel stale orders
- Order idempotency

Author: Trading Bot Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from threading import RLock
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class OrderGuardError(Exception):
    """Base exception for OrderGuard"""

    pass


class DuplicateOrderError(OrderGuardError):
    """Raised when a duplicate order is detected"""

    pass


class OrderTimeoutError(OrderGuardError):
    """Raised when order times out"""

    pass


class PendingOrderStatus(Enum):
    """Pending order status"""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    REJECTED = "REJECTED"


@dataclass
class PendingOrder:
    """Tracked pending order"""

    order_id: str
    symbol: str
    side: str  # BUY or SELL
    shares: int
    price: float
    order_type: str  # MARKET, LIMIT
    created_at: str
    status: PendingOrderStatus = PendingOrderStatus.PENDING
    filled_shares: int = 0
    filled_price: float = 0.0
    updated_at: str = ""
    timeout_at: str = ""
    idempotency_key: str = ""

    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.idempotency_key:
            self.idempotency_key = self._generate_idempotency_key()

    def _generate_idempotency_key(self) -> str:
        """Generate idempotency key for this order"""
        # Key based on symbol, side, and hour (prevent same order within hour)
        dt = datetime.fromisoformat(self.created_at)
        key_str = f"{self.symbol}_{self.side}_{dt.strftime('%Y%m%d_%H')}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]


class OrderGuard:
    """
    Order Guard - Prevents duplicate orders and manages pending orders

    Features:
    - Duplicate detection within configurable time window
    - Pending order tracking
    - Auto-timeout for stale orders
    - Idempotency key generation

    Usage:
        guard = get_order_guard()

        # Before placing order
        can_place, reason = guard.can_place_order("VNM", "BUY", 100)
        if not can_place:
            raise DuplicateOrderError(reason)

        # Register order
        order = guard.register_order("VNM", "BUY", 100, 85000)

        # Update status
        guard.update_order_status(order.order_id, PendingOrderStatus.FILLED)
    """

    STATE_FILE = "order_guard_state.json"

    def __init__(
        self,
        duplicate_window_minutes: int = 5,  # Block same order within 5 min
        order_timeout_minutes: int = 30,  # Cancel pending after 30 min
        max_pending_per_symbol: int = 2,  # Max pending orders per symbol
        state_file: str = None,
    ):
        self.duplicate_window_minutes = duplicate_window_minutes
        self.order_timeout_minutes = order_timeout_minutes
        self.max_pending_per_symbol = max_pending_per_symbol
        self.state_file = state_file or self.STATE_FILE

        self._lock = RLock()
        self._pending_orders: Dict[str, PendingOrder] = {}  # order_id -> PendingOrder
        self._recent_keys: Dict[str, datetime] = {}  # idempotency_key -> timestamp
        self._order_counter = 0

        self._load_state()

    def _load_state(self):
        """Load state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Restore pending orders
                    for order_data in data.get("pending_orders", []):
                        order_data["status"] = PendingOrderStatus(order_data["status"])
                        order = PendingOrder(**order_data)
                        if order.status == PendingOrderStatus.PENDING:
                            self._pending_orders[order.order_id] = order

                    # Restore recent keys (clean old ones)
                    cutoff = datetime.now() - timedelta(minutes=self.duplicate_window_minutes)
                    for key, ts_str in data.get("recent_keys", {}).items():
                        ts = datetime.fromisoformat(ts_str)
                        if ts > cutoff:
                            self._recent_keys[key] = ts

                    self._order_counter = data.get("order_counter", 0)

            except Exception as e:
                logger.error(f"Failed to load order guard state: {e}")

    def _save_state(self):
        """Save state to file"""
        try:
            # Convert pending orders to dicts
            pending_list = []
            for order in self._pending_orders.values():
                order_dict = asdict(order)
                order_dict["status"] = order.status.value
                pending_list.append(order_dict)

            # Convert recent keys
            recent_keys = {k: v.isoformat() for k, v in self._recent_keys.items()}

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pending_orders": pending_list,
                        "recent_keys": recent_keys,
                        "order_counter": self._order_counter,
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.error(f"Failed to save order guard state: {e}")

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self._order_counter += 1
        return f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._order_counter:04d}"

    def _get_idempotency_key(self, symbol: str, side: str) -> str:
        """Generate idempotency key"""
        dt = datetime.now()
        # Key includes minute for finer granularity in paper trading
        key_str = f"{symbol}_{side}_{dt.strftime('%Y%m%d_%H%M')}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _cleanup_old_entries(self):
        """Remove expired entries"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=self.duplicate_window_minutes)

        # Clean recent keys
        expired_keys = [k for k, ts in self._recent_keys.items() if ts < cutoff]
        for key in expired_keys:
            del self._recent_keys[key]

        # Check for timed out orders
        timeout_cutoff = now - timedelta(minutes=self.order_timeout_minutes)
        for order_id, order in list(self._pending_orders.items()):
            if order.status == PendingOrderStatus.PENDING:
                created = datetime.fromisoformat(order.created_at)
                if created < timeout_cutoff:
                    order.status = PendingOrderStatus.TIMEOUT
                    order.updated_at = now.isoformat()
                    logger.warning(f"⏰ Order timeout: {order_id} ({order.symbol} {order.side})")

    def can_place_order(
        self,
        symbol: str,
        side: str,
        shares: int,
    ) -> Tuple[bool, str]:
        """
        Check if order can be placed

        Returns:
            (can_place, reason)
        """
        with self._lock:
            self._cleanup_old_entries()

            # Check 1: Duplicate within time window
            key = self._get_idempotency_key(symbol, side)
            if key in self._recent_keys:
                elapsed = datetime.now() - self._recent_keys[key]
                remaining = self.duplicate_window_minutes * 60 - elapsed.total_seconds()
                return False, f"Duplicate order blocked. Wait {remaining:.0f}s"

            # Check 2: Max pending orders per symbol
            pending_count = sum(
                1
                for o in self._pending_orders.values()
                if o.symbol == symbol and o.status == PendingOrderStatus.PENDING
            )
            if pending_count >= self.max_pending_per_symbol:
                return False, f"Max {self.max_pending_per_symbol} pending orders for {symbol}"

            return True, "OK"

    def register_order(
        self,
        symbol: str,
        side: str,
        shares: int,
        price: float,
        order_type: str = "MARKET",
    ) -> PendingOrder:
        """
        Register a new order

        Raises:
            DuplicateOrderError if duplicate detected

        Returns:
            PendingOrder object
        """
        with self._lock:
            # Validate first
            can_place, reason = self.can_place_order(symbol, side, shares)
            if not can_place:
                raise DuplicateOrderError(reason)

            # Create order
            now = datetime.now()
            order_id = self._generate_order_id()
            idempotency_key = self._get_idempotency_key(symbol, side)

            order = PendingOrder(
                order_id=order_id,
                symbol=symbol.upper(),
                side=side.upper(),
                shares=shares,
                price=price,
                order_type=order_type.upper(),
                created_at=now.isoformat(),
                timeout_at=(now + timedelta(minutes=self.order_timeout_minutes)).isoformat(),
                idempotency_key=idempotency_key,
            )

            # Register
            self._pending_orders[order_id] = order
            self._recent_keys[idempotency_key] = now

            self._save_state()

            logger.info(f"📝 Order registered: {order_id} | {symbol} {side} {shares}@{price:,.0f}")

            return order

    def update_order_status(
        self,
        order_id: str,
        status: PendingOrderStatus,
        filled_shares: int = 0,
        filled_price: float = 0.0,
    ) -> Optional[PendingOrder]:
        """
        Update order status

        Returns:
            Updated PendingOrder or None if not found
        """
        with self._lock:
            if order_id not in self._pending_orders:
                logger.warning(f"Order not found: {order_id}")
                return None

            order = self._pending_orders[order_id]
            order.status = status
            order.updated_at = datetime.now().isoformat()

            if filled_shares > 0:
                order.filled_shares = filled_shares
            if filled_price > 0:
                order.filled_price = filled_price

            # Remove from tracking if completed
            if status in (
                PendingOrderStatus.FILLED,
                PendingOrderStatus.CANCELLED,
                PendingOrderStatus.TIMEOUT,
                PendingOrderStatus.REJECTED,
            ):
                # Keep in history but mark as complete
                pass

            self._save_state()

            logger.info(f"📝 Order updated: {order_id} -> {status.value}")

            return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        with self._lock:
            if order_id not in self._pending_orders:
                return False

            order = self._pending_orders[order_id]
            if order.status != PendingOrderStatus.PENDING:
                return False

            order.status = PendingOrderStatus.CANCELLED
            order.updated_at = datetime.now().isoformat()

            self._save_state()

            logger.info(f"❌ Order cancelled: {order_id}")
            return True

    def get_pending_orders(self, symbol: str = None) -> List[PendingOrder]:
        """Get pending orders, optionally filtered by symbol"""
        with self._lock:
            self._cleanup_old_entries()

            orders = [
                o for o in self._pending_orders.values() if o.status == PendingOrderStatus.PENDING
            ]

            if symbol:
                orders = [o for o in orders if o.symbol == symbol.upper()]

            return orders

    def get_timed_out_orders(self) -> List[PendingOrder]:
        """Get orders that have timed out"""
        with self._lock:
            self._cleanup_old_entries()
            return [
                o for o in self._pending_orders.values() if o.status == PendingOrderStatus.TIMEOUT
            ]

    def cancel_all_pending(self, symbol: str = None) -> int:
        """Cancel all pending orders"""
        with self._lock:
            count = 0
            for order in self._pending_orders.values():
                if order.status == PendingOrderStatus.PENDING:
                    if symbol is None or order.symbol == symbol.upper():
                        order.status = PendingOrderStatus.CANCELLED
                        order.updated_at = datetime.now().isoformat()
                        count += 1

            self._save_state()

            logger.info(f"❌ Cancelled {count} pending orders")
            return count

    def get_statistics(self) -> Dict:
        """Get order guard statistics"""
        with self._lock:
            self._cleanup_old_entries()

            status_counts = {}
            for order in self._pending_orders.values():
                status = order.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "total_orders": len(self._pending_orders),
                "pending_count": status_counts.get("PENDING", 0),
                "filled_count": status_counts.get("FILLED", 0),
                "cancelled_count": status_counts.get("CANCELLED", 0),
                "timeout_count": status_counts.get("TIMEOUT", 0),
                "recent_keys_count": len(self._recent_keys),
                "duplicate_window_minutes": self.duplicate_window_minutes,
                "order_timeout_minutes": self.order_timeout_minutes,
            }


# Singleton instance
_order_guard_instance: Optional[OrderGuard] = None
_order_guard_lock = RLock()


def get_order_guard() -> OrderGuard:
    """Get singleton order guard instance"""
    global _order_guard_instance

    with _order_guard_lock:
        if _order_guard_instance is None:
            _order_guard_instance = OrderGuard()
        return _order_guard_instance


def reset_order_guard():
    """Reset order guard (for testing)"""
    global _order_guard_instance
    with _order_guard_lock:
        _order_guard_instance = None
