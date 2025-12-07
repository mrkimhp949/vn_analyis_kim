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
from datetime import datetime, date, time, timedelta
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


class WashTradeDetector:
    """
    Wash Trade Detection and Prevention for Vietnam Market.
    
    Wash trading is illegal and involves buying and selling the same security
    to create artificial activity. This detector prevents:
    - Rapid buy-sell cycles (< 5 minutes)
    - Same price buy-sell (no economic purpose)
    - Excessive round trips in short period
    - Pattern-based wash trade detection
    
    Vietnam regulations:
    - Circular 203/2015/TT-BTC prohibits wash trading
    - Penalties: Fines up to 5x profit + trading ban
    """
    
    # Detection thresholds
    MIN_HOLDING_MINUTES = 5          # Minimum hold time
    MIN_PRICE_CHANGE_PCT = 0.005     # Minimum 0.5% price change for valid trade
    MAX_ROUND_TRIPS_PER_HOUR = 3     # Max buy-sell cycles per hour
    MAX_ROUND_TRIPS_PER_DAY = 10     # Max buy-sell cycles per day
    PATTERN_LOOKBACK_TRADES = 20    # Trades to analyze for patterns
    
    def __init__(self):
        self._trade_history: List[Dict] = []
        self._round_trip_count: int = 0
        self._hourly_round_trips: Dict[int, int] = {}  # hour -> count
        self._flagged_symbols: set = set()
        self._last_reset_date: Optional[date] = None
    
    def _reset_if_new_day(self):
        """Reset counters for new trading day."""
        today = date.today()
        if self._last_reset_date != today:
            self._trade_history.clear()
            self._round_trip_count = 0
            self._hourly_round_trips.clear()
            self._flagged_symbols.clear()
            self._last_reset_date = today
    
    def check_wash_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        last_trade_time: Optional[datetime] = None,
        last_trade_price: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Check if a trade might be a wash trade.
        
        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            quantity: Number of shares
            price: Trade price
            last_trade_time: Time of last trade for this symbol
            last_trade_price: Price of last trade for this symbol
            
        Returns:
            (is_wash_trade, reason)
        """
        self._reset_if_new_day()
        
        # Check 1: Symbol already flagged
        if symbol in self._flagged_symbols:
            return True, f"⚠️ {symbol} flagged for suspicious activity - trading paused"
        
        # Check 2: Minimum holding time
        if last_trade_time:
            minutes_held = (datetime.now() - last_trade_time).total_seconds() / 60
            if minutes_held < self.MIN_HOLDING_MINUTES:
                return True, (
                    f"⚠️ Wash trade risk: Only {minutes_held:.1f} min since last trade "
                    f"(min: {self.MIN_HOLDING_MINUTES} min)"
                )
        
        # Check 3: Same price (no economic purpose)
        if last_trade_price and abs(price - last_trade_price) / last_trade_price < self.MIN_PRICE_CHANGE_PCT:
            return True, (
                f"⚠️ Wash trade risk: Price unchanged from last trade "
                f"({price:,.0f} ≈ {last_trade_price:,.0f})"
            )
        
        # Check 4: Hourly round trip limit
        current_hour = datetime.now().hour
        hourly_count = self._hourly_round_trips.get(current_hour, 0)
        if hourly_count >= self.MAX_ROUND_TRIPS_PER_HOUR:
            return True, (
                f"⚠️ Wash trade risk: {hourly_count} round trips this hour "
                f"(max: {self.MAX_ROUND_TRIPS_PER_HOUR})"
            )
        
        # Check 5: Daily round trip limit
        if self._round_trip_count >= self.MAX_ROUND_TRIPS_PER_DAY:
            return True, (
                f"⚠️ Wash trade risk: {self._round_trip_count} round trips today "
                f"(max: {self.MAX_ROUND_TRIPS_PER_DAY})"
            )
        
        # Check 6: Pattern detection (alternating buy-sell on same symbol)
        if len(self._trade_history) >= 4:
            recent = [t for t in self._trade_history[-10:] if t['symbol'] == symbol]
            if len(recent) >= 4:
                # Check for alternating pattern
                sides = [t['side'] for t in recent[-4:]]
                if sides == ['BUY', 'SELL', 'BUY', 'SELL'] or sides == ['SELL', 'BUY', 'SELL', 'BUY']:
                    self._flagged_symbols.add(symbol)
                    return True, f"⚠️ Wash trade pattern detected for {symbol} - flagged"
        
        return False, "OK"
    
    def record_trade(self, symbol: str, side: str, quantity: int, price: float):
        """Record a trade for pattern analysis."""
        self._reset_if_new_day()
        
        self._trade_history.append({
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(),
        })
        
        # Trim history
        if len(self._trade_history) > self.PATTERN_LOOKBACK_TRADES:
            self._trade_history = self._trade_history[-self.PATTERN_LOOKBACK_TRADES:]
        
        # Update round trip count if this completes a cycle
        if side == 'SELL':
            # Check if there was a recent buy
            recent_buys = [
                t for t in self._trade_history[:-1]
                if t['symbol'] == symbol and t['side'] == 'BUY'
            ]
            if recent_buys:
                self._round_trip_count += 1
                current_hour = datetime.now().hour
                self._hourly_round_trips[current_hour] = self._hourly_round_trips.get(current_hour, 0) + 1
    
    def get_stats(self) -> Dict:
        """Get wash trade detection statistics."""
        return {
            'total_trades_tracked': len(self._trade_history),
            'round_trips_today': self._round_trip_count,
            'hourly_round_trips': dict(self._hourly_round_trips),
            'flagged_symbols': list(self._flagged_symbols),
            'max_round_trips_per_day': self.MAX_ROUND_TRIPS_PER_DAY,
        }


class IntradayTracker:
    """
    Track intraday (T+0) trading for Vietnam stock market.

    Vietnam intraday trading characteristics:
    - Only available for margin accounts
    - Same transaction costs as regular trades
    - Subject to margin requirements
    - ATO/ATC sessions have higher volatility
    - Price limits (±7%) still apply
    
    IMPROVED v2.0:
    - Wash trade detection and prevention
    - Pattern-based suspicious activity detection
    - Hourly and daily round trip limits
    - Minimum holding time enforcement

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

    # Intraday limits - TIGHTENED v2.0
    MAX_INTRADAY_TRADES = 20  # Max trades per day (prevent overtrading)
    MAX_INTRADAY_LOSS_PCT = 0.02  # Stop trading if -2% intraday loss
    MIN_HOLDING_MINUTES = 5  # Minimum holding time (avoid wash trades)
    
    # NEW v2.0: Additional limits
    MAX_SYMBOL_TRADES_PER_DAY = 6  # Max trades per symbol per day
    COOLING_OFF_MINUTES = 15  # Cooling off after loss

    def __init__(
        self,
        mode: TradingMode = TradingMode.MARGIN_T0,
        margin_buying_power: float = 0.0,
        commission_rate: float = 0.0015,  # 0.15% per trade
        enable_t0: bool = True,
        enable_wash_trade_detection: bool = True,  # NEW v2.0
    ):
        """
        Initialize intraday tracker.

        Args:
            mode: Trading mode (CASH_ONLY, MARGIN_T2, MARGIN_T0)
            margin_buying_power: Available margin for T+0 trading
            commission_rate: Commission rate per trade
            enable_t0: Whether T+0 is enabled for this account
            enable_wash_trade_detection: Enable wash trade prevention (NEW v2.0)
        """
        self.mode = mode
        self.margin_buying_power = margin_buying_power
        self.commission_rate = commission_rate
        self.enable_t0 = enable_t0 and mode == TradingMode.MARGIN_T0
        self.enable_wash_trade_detection = enable_wash_trade_detection

        # Position tracking
        self._positions: Dict[str, IntradayPosition] = {}
        self._today_trades: List[IntradayTrade] = []
        self._stats: IntradayStats = IntradayStats(date=date.today())
        
        # NEW v2.0: Wash trade detector
        self._wash_trade_detector = WashTradeDetector() if enable_wash_trade_detection else None
        
        # NEW v2.0: Per-symbol trade count
        self._symbol_trade_counts: Dict[str, int] = {}
        
        # NEW v2.0: Cooling off tracking (after losses)
        self._cooling_off_until: Optional[datetime] = None
        self._last_loss_symbol: Optional[str] = None

        # Thread safety
        self._lock = RLock()

        # Daily reset tracking
        self._last_reset_date: Optional[date] = None

        logger.info(
            f"✅ IntradayTracker initialized: mode={mode.value}, "
            f"T+0={'enabled' if self.enable_t0 else 'disabled'}, "
            f"wash_trade_detection={'enabled' if enable_wash_trade_detection else 'disabled'}"
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

    def can_sell_intraday(self, symbol: str, quantity: int, price: float = 0) -> Tuple[bool, str]:
        """
        Check if an intraday sell is allowed.
        
        IMPROVED v2.0: Added wash trade detection and cooling off period.

        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell
            price: Expected sell price (for wash trade detection)

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
            
            # NEW v2.0: Check per-symbol trade limit
            symbol_trades = self._symbol_trade_counts.get(symbol, 0)
            if symbol_trades >= self.MAX_SYMBOL_TRADES_PER_DAY:
                return (
                    False,
                    f"Per-symbol trade limit reached for {symbol} "
                    f"({symbol_trades}/{self.MAX_SYMBOL_TRADES_PER_DAY})"
                )
            
            # NEW v2.0: Check cooling off period
            if self._cooling_off_until and datetime.now() < self._cooling_off_until:
                remaining = (self._cooling_off_until - datetime.now()).total_seconds() / 60
                return (
                    False,
                    f"Cooling off period active. Wait {remaining:.0f} more minutes "
                    f"(after loss on {self._last_loss_symbol})"
                )
            
            # NEW v2.0: Wash trade detection
            if self._wash_trade_detector and price > 0:
                # Get last trade info for this symbol
                last_trade = None
                for trade in reversed(pos.trades):
                    if trade.side == "BUY":
                        last_trade = trade
                        break
                
                is_wash, reason = self._wash_trade_detector.check_wash_trade(
                    symbol=symbol,
                    side="SELL",
                    quantity=quantity,
                    price=price,
                    last_trade_time=last_trade.timestamp if last_trade else None,
                    last_trade_price=last_trade.price if last_trade else None,
                )
                
                if is_wash:
                    return False, reason

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
                # Clear cooling off on win
                self._cooling_off_until = None
                self._last_loss_symbol = None
            else:
                self._stats.loss_trades += 1
                self._stats.largest_loss = min(self._stats.largest_loss, realized_pnl)
                # NEW v2.0: Activate cooling off period after loss
                self._cooling_off_until = datetime.now() + timedelta(minutes=self.COOLING_OFF_MINUTES)
                self._last_loss_symbol = symbol
                logger.warning(
                    f"⏸️ Cooling off activated for {self.COOLING_OFF_MINUTES} min after loss on {symbol}"
                )
            
            # NEW v2.0: Update per-symbol trade count
            self._symbol_trade_counts[symbol] = self._symbol_trade_counts.get(symbol, 0) + 1
            
            # NEW v2.0: Record trade for wash trade detection
            if self._wash_trade_detector:
                self._wash_trade_detector.record_trade(symbol, "SELL", quantity, price)

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
