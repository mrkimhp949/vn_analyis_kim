# -*- coding: utf-8 -*-
"""
T+0 Intraday Trading Support for Vietnam Stock Market

Vietnam T+0 Rules (as of 2024):
- T+0 is allowed for MARGIN ACCOUNTS only
- You can sell stocks bought today using margin buying power
- Cash accounts must wait T+2 for settlement
- Intraday buying power = Cash + (Margin limit - Used margin)

Key differences from regular T+2:
- Can exit same-day positions (day trading)
- Higher transaction costs (more trades)
- Requires margin account with sufficient limit
- Subject to intraday margin calls

Broker-specific T+0 rules:
- SSI: T+0 with margin, min 50M VND account
- VPS: T+0 with margin, min 100M VND account
- TCBS: T+0 with margin, automatic
- VNDirect: T+0 with margin, min 30M VND account

This module provides:
1. Intraday position tracking
2. Same-day exit capability check
3. Intraday P&L tracking
4. Day trading statistics
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time
from enum import Enum
from typing import Dict, List, Optional, Tuple
from threading import RLock

logger = logging.getLogger(__name__)


class IntradayPositionStatus(Enum):
    """Status of an intraday position"""

    OPEN = "OPEN"  # Position opened today, still holding
    PARTIALLY_CLOSED = "PARTIAL"  # Partially closed intraday
    CLOSED = "CLOSED"  # Fully closed intraday (day trade)
    CARRIED_OVER = "CARRIED"  # Will be carried to T+1 (not closed today)


class TradingMode(Enum):
    """Trading mode for the account"""

    CASH_ONLY = "CASH"  # Cash account - T+2 settlement
    MARGIN_T2 = "MARGIN_T2"  # Margin account - T+2 settlement
    MARGIN_T0 = "MARGIN_T0"  # Margin account with T+0 enabled


@dataclass
class IntradayTrade:
    """Record of a single intraday trade"""

    trade_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    price: float
    value: float
    timestamp: datetime
    is_intraday_close: bool = False  # True if this closes an intraday position
    commission: float = 0.0


@dataclass
class IntradayPosition:
    """
    Intraday position tracking.

    Tracks positions opened today that may be closed same-day (T+0).
    """

    symbol: str
    open_quantity: int  # Shares bought today
    current_quantity: int  # Remaining shares (after partial sells)
    avg_open_price: float  # Average buy price
    current_price: float  # Latest price
    open_time: datetime  # When position was opened
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    realized_pnl: float = 0.0  # P&L from closed portion
    unrealized_pnl: float = 0.0  # P&L from open portion
    status: IntradayPositionStatus = IntradayPositionStatus.OPEN
    trades: List[IntradayTrade] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_pnl_pct(self) -> float:
        """Total P&L percentage"""
        cost_basis = self.open_quantity * self.avg_open_price
        if cost_basis <= 0:
            return 0.0
        return self.total_pnl / cost_basis

    @property
    def holding_minutes(self) -> int:
        """Minutes since position was opened"""
        end_time = self.close_time or datetime.now()
        delta = end_time - self.open_time
        return int(delta.total_seconds() / 60)


@dataclass
class IntradayStats:
    """Daily intraday trading statistics"""

    date: date
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    day_trades: int = 0  # Complete round trips (buy+sell same day)
    total_volume: int = 0
    total_value: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commission: float = 0.0
    win_trades: int = 0
    loss_trades: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_minutes: float = 0.0

    @property
    def win_rate(self) -> float:
        """Win rate for closed day trades"""
        total = self.win_trades + self.loss_trades
        if total == 0:
            return 0.0
        return self.win_trades / total

    @property
    def net_pnl(self) -> float:
        """Net P&L after commissions"""
        return self.realized_pnl - self.total_commission


class IntradayTracker:
    """
    Track intraday (T+0) trading for Vietnam stock market.

    Vietnam intraday trading characteristics:
    - Only available for margin accounts
    - Same transaction costs as regular trades
    - Subject to margin requirements
    - ATO/ATC sessions have higher volatility
    - Price limits (±7%) still apply

    Usage:
        tracker = IntradayTracker(mode=TradingMode.MARGIN_T0)
        tracker.record_buy("VNM", 1000, 80000)
        # Later same day...
        can_sell, reason = tracker.can_sell_intraday("VNM", 1000)
        if can_sell:
            tracker.record_sell("VNM", 1000, 82000)
    """

    # Vietnam trading hours
    MARKET_OPEN = time(9, 0)
    MARKET_CLOSE = time(14, 45)
    LUNCH_START = time(11, 30)
    LUNCH_END = time(13, 0)

    # Intraday limits
    MAX_INTRADAY_TRADES = 20  # Max trades per day (prevent overtrading)
    MAX_INTRADAY_LOSS_PCT = 0.02  # Stop trading if -2% intraday loss
    MIN_HOLDING_MINUTES = 5  # Minimum holding time (avoid wash trades)

    def __init__(
        self,
        mode: TradingMode = TradingMode.MARGIN_T0,
        margin_buying_power: float = 0.0,
        commission_rate: float = 0.0015,  # 0.15% per trade
        enable_t0: bool = True,
    ):
        """
        Initialize intraday tracker.

        Args:
            mode: Trading mode (CASH_ONLY, MARGIN_T2, MARGIN_T0)
            margin_buying_power: Available margin for T+0 trading
            commission_rate: Commission rate per trade
            enable_t0: Whether T+0 is enabled for this account
        """
        self.mode = mode
        self.margin_buying_power = margin_buying_power
        self.commission_rate = commission_rate
        self.enable_t0 = enable_t0 and mode == TradingMode.MARGIN_T0

        # Position tracking
        self._positions: Dict[str, IntradayPosition] = {}
        self._today_trades: List[IntradayTrade] = []
        self._stats: IntradayStats = IntradayStats(date=date.today())

        # Thread safety
        self._lock = RLock()

        # Daily reset tracking
        self._last_reset_date: Optional[date] = None

        logger.info(
            f"✅ IntradayTracker initialized: mode={mode.value}, "
            f"T+0={'enabled' if self.enable_t0 else 'disabled'}"
        )

    def _check_new_day(self):
        """Reset tracking for new trading day."""
        today = date.today()
        if self._last_reset_date != today:
            self._positions.clear()
            self._today_trades.clear()
            self._stats = IntradayStats(date=today)
            self._last_reset_date = today
            logger.info(f"📅 Intraday tracker reset for new day: {today}")

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        import uuid

        return f"T{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    def is_market_open(self) -> bool:
        """Check if market is currently open for trading."""
        now = datetime.now().time()

        # Check if within trading hours
        if now < self.MARKET_OPEN or now > self.MARKET_CLOSE:
            return False

        # Check if lunch break
        if self.LUNCH_START <= now <= self.LUNCH_END:
            return False

        return True

    def record_buy(
        self, symbol: str, quantity: int, price: float, timestamp: Optional[datetime] = None
    ) -> IntradayTrade:
        """
        Record a buy trade.

        Args:
            symbol: Stock symbol
            quantity: Number of shares bought
            price: Purchase price
            timestamp: Trade timestamp (default: now)

        Returns:
            IntradayTrade record
        """
        with self._lock:
            self._check_new_day()

            timestamp = timestamp or datetime.now()
            value = quantity * price
            commission = value * self.commission_rate

            trade = IntradayTrade(
                trade_id=self._generate_trade_id(),
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                price=price,
                value=value,
                timestamp=timestamp,
                commission=commission,
            )

            self._today_trades.append(trade)

            # Update or create position
            if symbol in self._positions:
                pos = self._positions[symbol]
                # Average up/down
                total_cost = (pos.current_quantity * pos.avg_open_price) + value
                total_qty = pos.current_quantity + quantity
                pos.avg_open_price = total_cost / total_qty
                pos.current_quantity = total_qty
                pos.open_quantity = total_qty
                pos.trades.append(trade)
            else:
                self._positions[symbol] = IntradayPosition(
                    symbol=symbol,
                    open_quantity=quantity,
                    current_quantity=quantity,
                    avg_open_price=price,
                    current_price=price,
                    open_time=timestamp,
                    trades=[trade],
                )

            # Update stats
            self._stats.total_trades += 1
            self._stats.buy_trades += 1
            self._stats.total_volume += quantity
            self._stats.total_value += value
            self._stats.total_commission += commission

            logger.info(
                f"📈 BUY recorded: {symbol} {quantity} @ {price:,.0f} " f"(value: {value:,.0f} VND)"
            )

            return trade

    def can_sell_intraday(self, symbol: str, quantity: int) -> Tuple[bool, str]:
        """
        Check if an intraday sell is allowed.

        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell

        Returns:
            Tuple of (can_sell, reason)
        """
        with self._lock:
            self._check_new_day()

            # Check if T+0 is enabled
            if not self.enable_t0:
                return False, "T+0 trading is not enabled for this account"

            # Check if we have the position
            if symbol not in self._positions:
                return False, f"No intraday position found for {symbol}"

            pos = self._positions[symbol]

            # Check quantity
            if quantity > pos.current_quantity:
                return (
                    False,
                    f"Insufficient shares. Have: {pos.current_quantity}, "
                    f"Want to sell: {quantity}",
                )

            # Check minimum holding time (prevent wash trades)
            if pos.holding_minutes < self.MIN_HOLDING_MINUTES:
                return (
                    False,
                    f"Minimum holding time not met. "
                    f"Hold for {self.MIN_HOLDING_MINUTES - pos.holding_minutes} more minutes",
                )

            # Check daily trade limit
            if self._stats.total_trades >= self.MAX_INTRADAY_TRADES:
                return False, f"Daily trade limit reached ({self.MAX_INTRADAY_TRADES})"

            # Check daily loss limit
            if self._stats.net_pnl < -self.margin_buying_power * self.MAX_INTRADAY_LOSS_PCT:
                return (False, f"Daily loss limit reached. Net P&L: {self._stats.net_pnl:,.0f} VND")

            return True, "OK"

    def record_sell(
        self, symbol: str, quantity: int, price: float, timestamp: Optional[datetime] = None
    ) -> Optional[IntradayTrade]:
        """
        Record a sell trade (T+0 intraday exit).

        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell
            price: Sale price
            timestamp: Trade timestamp (default: now)

        Returns:
            IntradayTrade record if successful, None if failed
        """
        with self._lock:
            can_sell, reason = self.can_sell_intraday(symbol, quantity)
            if not can_sell:
                logger.warning(f"❌ Cannot sell intraday: {reason}")
                return None

            timestamp = timestamp or datetime.now()
            value = quantity * price
            commission = value * self.commission_rate

            pos = self._positions[symbol]

            # Calculate P&L for this sale
            cost_basis = quantity * pos.avg_open_price
            realized_pnl = value - cost_basis - commission

            trade = IntradayTrade(
                trade_id=self._generate_trade_id(),
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=price,
                value=value,
                timestamp=timestamp,
                is_intraday_close=True,
                commission=commission,
            )

            self._today_trades.append(trade)
            pos.trades.append(trade)

            # Update position
            pos.current_quantity -= quantity
            pos.realized_pnl += realized_pnl

            if pos.current_quantity == 0:
                pos.status = IntradayPositionStatus.CLOSED
                pos.close_time = timestamp
                pos.close_price = price
            else:
                pos.status = IntradayPositionStatus.PARTIALLY_CLOSED

            # Update stats
            self._stats.total_trades += 1
            self._stats.sell_trades += 1
            self._stats.total_volume += quantity
            self._stats.total_value += value
            self._stats.total_commission += commission
            self._stats.realized_pnl += realized_pnl

            if pos.status == IntradayPositionStatus.CLOSED:
                self._stats.day_trades += 1
                self._stats.avg_holding_minutes = (
                    self._stats.avg_holding_minutes * (self._stats.day_trades - 1)
                    + pos.holding_minutes
                ) / self._stats.day_trades

            if realized_pnl > 0:
                self._stats.win_trades += 1
                self._stats.largest_win = max(self._stats.largest_win, realized_pnl)
            else:
                self._stats.loss_trades += 1
                self._stats.largest_loss = min(self._stats.largest_loss, realized_pnl)

            logger.info(
                f"📉 SELL recorded: {symbol} {quantity} @ {price:,.0f} "
                f"(P&L: {realized_pnl:+,.0f} VND)"
            )

            return trade

    def update_price(self, symbol: str, current_price: float):
        """Update current price for unrealized P&L calculation."""
        with self._lock:
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.current_price = current_price

                if pos.current_quantity > 0:
                    cost_basis = pos.current_quantity * pos.avg_open_price
                    market_value = pos.current_quantity * current_price
                    pos.unrealized_pnl = market_value - cost_basis

                    self._stats.unrealized_pnl = sum(
                        p.unrealized_pnl for p in self._positions.values()
                    )

    def get_position(self, symbol: str) -> Optional[IntradayPosition]:
        """Get intraday position for a symbol."""
        with self._lock:
            self._check_new_day()
            return self._positions.get(symbol)

    def get_all_positions(self) -> List[IntradayPosition]:
        """Get all intraday positions."""
        with self._lock:
            self._check_new_day()
            return list(self._positions.values())

    def get_open_positions(self) -> List[IntradayPosition]:
        """Get only open intraday positions."""
        with self._lock:
            self._check_new_day()
            return [
                p
                for p in self._positions.values()
                if p.status
                in [IntradayPositionStatus.OPEN, IntradayPositionStatus.PARTIALLY_CLOSED]
            ]

    def get_stats(self) -> IntradayStats:
        """Get today's intraday trading statistics."""
        with self._lock:
            self._check_new_day()
            return self._stats

    def close_all_positions(self, get_price_func) -> List[IntradayTrade]:
        """
        Close all open intraday positions (end of day).

        Args:
            get_price_func: Function to get current price for a symbol

        Returns:
            List of sell trades executed
        """
        with self._lock:
            trades = []
            open_positions = self.get_open_positions()

            for pos in open_positions:
                current_price = get_price_func(pos.symbol)
                if current_price and pos.current_quantity > 0:
                    trade = self.record_sell(pos.symbol, pos.current_quantity, current_price)
                    if trade:
                        trades.append(trade)

            return trades

    def get_status_message(self) -> str:
        """Get formatted status message."""
        stats = self.get_stats()
        positions = self.get_open_positions()

        lines = [
            "=" * 50,
            f"📊 INTRADAY TRADING STATUS ({stats.date})",
            "=" * 50,
            f"Mode: {self.mode.value} | T+0: {'✅ Enabled' if self.enable_t0 else '❌ Disabled'}",
            "-" * 50,
            f"Total Trades: {stats.total_trades} (Buy: {stats.buy_trades}, Sell: {stats.sell_trades})",
            f"Day Trades:   {stats.day_trades}",
            f"Total Volume: {stats.total_volume:,}",
            f"Total Value:  {stats.total_value:,.0f} VND",
            "-" * 50,
            f"Realized P&L:   {stats.realized_pnl:+,.0f} VND",
            f"Unrealized P&L: {stats.unrealized_pnl:+,.0f} VND",
            f"Commissions:    {stats.total_commission:,.0f} VND",
            f"Net P&L:        {stats.net_pnl:+,.0f} VND",
            "-" * 50,
            f"Win Rate: {stats.win_rate:.1%} ({stats.win_trades}W / {stats.loss_trades}L)",
            f"Largest Win:  {stats.largest_win:+,.0f} VND",
            f"Largest Loss: {stats.largest_loss:+,.0f} VND",
            f"Avg Holding:  {stats.avg_holding_minutes:.0f} minutes",
        ]

        if positions:
            lines.append("-" * 50)
            lines.append("📈 OPEN POSITIONS:")
            for pos in positions:
                pnl_pct = pos.total_pnl_pct * 100
                emoji = "🟢" if pos.total_pnl >= 0 else "🔴"
                lines.append(
                    f"   {emoji} {pos.symbol}: {pos.current_quantity} shares @ "
                    f"{pos.avg_open_price:,.0f} → {pos.current_price:,.0f} "
                    f"({pnl_pct:+.2f}%)"
                )

        lines.append("=" * 50)
        return "\n".join(lines)


# Singleton instance
_intraday_tracker: Optional[IntradayTracker] = None


def get_intraday_tracker(
    mode: TradingMode = TradingMode.MARGIN_T0, margin_buying_power: float = 100_000_000
) -> IntradayTracker:
    """Get singleton intraday tracker instance."""
    global _intraday_tracker
    if _intraday_tracker is None:
        _intraday_tracker = IntradayTracker(mode=mode, margin_buying_power=margin_buying_power)
    return _intraday_tracker


# Test
if __name__ == "__main__":
    print("Testing Intraday Tracker...")

    tracker = IntradayTracker(
        mode=TradingMode.MARGIN_T0, margin_buying_power=100_000_000, enable_t0=True
    )

    # Simulate intraday trading
    print("\n1️⃣ Recording BUY...")
    tracker.record_buy("VNM", 1000, 80_000)
    tracker.record_buy("HPG", 2000, 25_000)

    print("\n2️⃣ Update prices...")
    tracker.update_price("VNM", 81_500)  # +1.875%
    tracker.update_price("HPG", 24_500)  # -2%

    print("\n3️⃣ Check if can sell intraday...")
    can_sell, reason = tracker.can_sell_intraday("VNM", 500)
    print(f"   Can sell VNM: {can_sell} - {reason}")

    # Wait minimum holding time (simulated)
    import time

    print("\n4️⃣ Waiting for minimum holding time...")
    time.sleep(1)  # In real scenario, wait 5 minutes

    # For testing, temporarily set MIN_HOLDING_MINUTES to 0
    tracker.MIN_HOLDING_MINUTES = 0

    print("\n5️⃣ Recording SELL (partial)...")
    tracker.record_sell("VNM", 500, 81_500)

    print("\n6️⃣ Recording SELL (full)...")
    tracker.record_sell("VNM", 500, 82_000)

    print("\n" + tracker.get_status_message())

    print("\n✅ Test completed!")
