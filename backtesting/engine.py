# -*- coding: utf-8 -*-
"""
Backtesting Engine - Test Trading Strategies with Historical Data

IMPROVEMENTS V4 (2025-01):
- Vietnam lot size (100 shares) and tick size enforcement
- Synchronized transaction costs from constants.py
- Partial exit support
- Circuit breaker and risk guards
- Position size multiplier support
- VALIDATED SLIPPAGE based on real VN market data (NEW v4)
- Session-aware slippage estimation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

from src.config.constants import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_SLIPPAGE,
    TOTAL_TRANSACTION_COST,
    VIETNAM_LOT_SIZE,
    VIETNAM_TICK_SIZE,
    VN_SLIPPAGE_VN30,
    VN_SLIPPAGE_LIQUID,
    VN_SLIPPAGE_MEDIUM,
    VN_SLIPPAGE_ILLIQUID,
    get_dynamic_slippage,
    get_order_impact_slippage,
)

logger = logging.getLogger(__name__)

# Try to import validated slippage from vn_market_data
VN_VALIDATED_SLIPPAGE_AVAILABLE = False
try:
    from src.utils.vn_market_data import (
        get_validated_slippage,
        VALIDATED_SLIPPAGE_VN,
    )

    VN_VALIDATED_SLIPPAGE_AVAILABLE = True
    logger.info("✅ Using validated VN market slippage data")
except ImportError:
    pass

# Try to import VN30 fetcher for dynamic VN30 check
VN30_DYNAMIC_AVAILABLE = False
try:
    from src.utils.vietnam_market import is_vn30_dynamic, get_vn30_symbols_dynamic

    VN30_DYNAMIC_AVAILABLE = True
except ImportError:
    pass


def round_to_lot_size(shares: int, lot_size: int = VIETNAM_LOT_SIZE) -> int:
    """Round shares down to nearest lot size (Vietnam: 100 shares)"""
    return (shares // lot_size) * lot_size


def round_to_tick_size(price: float, tick_size: float = VIETNAM_TICK_SIZE) -> float:
    """Round price to nearest tick size (Vietnam: 10 VND for most stocks)"""
    return round(price / tick_size) * tick_size


@dataclass
class BacktestConfig:
    """Backtesting configuration"""

    initial_capital: float = 100_000_000  # 100M VND
    # Use centralized transaction costs from constants.py
    commission_rate: float = DEFAULT_COMMISSION_RATE  # 0.35% (brokerage + tax + fees)
    slippage: float = DEFAULT_SLIPPAGE  # 0.1% price slippage (fallback)
    max_positions: int = 5  # Max concurrent positions
    position_size_pct: float = 0.20  # 20% per position (default, can be overridden by multiplier)

    # Risk management
    max_loss_per_trade: float = 0.03  # 3% max loss per trade
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_portfolio_risk: float = 0.15  # 15% max total portfolio risk

    # Vietnam market specific
    lot_size: int = VIETNAM_LOT_SIZE  # 100 shares per lot
    tick_size: float = VIETNAM_TICK_SIZE  # 10 VND tick size

    # Circuit breaker settings
    max_consecutive_losses: int = 3  # Block trading after N consecutive losses
    daily_loss_circuit_breaker: bool = True  # Enable daily loss circuit breaker

    # IMPROVED v4: Validated slippage settings based on real VN market data
    use_dynamic_slippage: bool = True  # Use dynamic slippage based on liquidity
    use_validated_slippage: bool = True  # NEW v4: Use validated slippage from real market data
    slippage_vn30: float = VN_SLIPPAGE_VN30  # 0.3% for VN30 blue chips
    slippage_liquid: float = VN_SLIPPAGE_LIQUID  # 0.4% for liquid stocks
    slippage_medium: float = VN_SLIPPAGE_MEDIUM  # 0.6% for medium liquidity
    slippage_illiquid: float = VN_SLIPPAGE_ILLIQUID  # 1.0% for illiquid stocks


def calculate_dynamic_slippage(
    symbol: str,
    order_value: float,
    avg_daily_volume: float = 0,
    avg_price: float = 0,
    config: Optional[BacktestConfig] = None,
) -> float:
    """
    Calculate dynamic slippage based on symbol liquidity and order size.

    IMPROVED v4: Uses VALIDATED slippage from real VN market trading data.

    Key improvements:
    - Session-aware slippage (higher during lunch break, ATO/ATC)
    - Order size impact using square-root model
    - Validated against actual trading execution data

    Args:
        symbol: Stock symbol
        order_value: Order value in VND
        avg_daily_volume: Average daily volume in shares
        avg_price: Average price per share
        config: BacktestConfig (optional)

    Returns:
        Slippage rate (0.002 - 0.030)
    """
    if config is None:
        config = BacktestConfig()

    if not config.use_dynamic_slippage:
        return config.slippage

    # NEW v4: Use validated slippage if available
    if config.use_validated_slippage and VN_VALIDATED_SLIPPAGE_AVAILABLE:
        try:
            avg_daily_value = (
                avg_daily_volume * avg_price if avg_daily_volume > 0 and avg_price > 0 else 0
            )
            result = get_validated_slippage(
                symbol=symbol,
                order_value=order_value,
                avg_daily_value=avg_daily_value,
                price=avg_price,
                is_market_order=True,
            )
            return result.get("total_slippage", config.slippage)
        except Exception as e:
            logger.debug(f"Validated slippage failed for {symbol}: {e}")

    # Check if VN30 using dynamic fetcher
    is_vn30 = False
    if VN30_DYNAMIC_AVAILABLE:
        try:
            is_vn30 = is_vn30_dynamic(symbol)
        except Exception:
            pass

    # Calculate liquidity value if possible
    liquidity_value = avg_daily_volume * avg_price if avg_daily_volume > 0 and avg_price > 0 else 0

    # Use the centralized slippage calculation from constants.py
    if liquidity_value > 0:
        try:
            return get_order_impact_slippage(
                symbol=symbol,
                order_value=order_value,
                avg_daily_volume=avg_daily_volume,
                avg_price=avg_price,
                is_market_order=True,
            )
        except Exception:
            pass

    # Fallback: tiered slippage
    if is_vn30:
        return config.slippage_vn30  # 0.3%
    elif liquidity_value > 3_000_000_000:  # > 3B VND
        return config.slippage_liquid  # 0.4%
    elif liquidity_value > 1_000_000_000:  # 1-3B VND
        return config.slippage_medium  # 0.6%
    else:
        return config.slippage_illiquid  # 1.0%


@dataclass
class Trade:
    """Individual trade record"""

    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
    initial_shares: int = 0  # Track original shares for partial exits
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Performance
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0

    # Metadata
    entry_reason: str = ""
    exit_reason: str = ""
    holding_days: int = 0

    # Partial exit tracking
    partial_exits: List[Dict] = field(default_factory=list)
    realized_pnl: float = 0.0  # PnL from partial exits

    def close_trade(
        self,
        exit_date: datetime,
        exit_price: float,
        exit_reason: str,
        commission_rate: float,
        slippage: float,
    ):
        """Close the trade and calculate PnL"""
        # CRITICAL FIX: Validate exit_price to prevent calculation errors
        if exit_price is None or exit_price <= 0:
            raise ValueError(
                f"Invalid exit_price: {exit_price}. " f"Must be a positive number for {self.symbol}"
            )

        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.holding_days = (exit_date - self.entry_date).days

        # Calculate costs for remaining shares
        entry_commission = self.shares * self.entry_price * commission_rate
        exit_commission = self.shares * self.exit_price * commission_rate
        self.commission = entry_commission + exit_commission

        # Slippage cost (assumes adverse price movement)
        self.slippage_cost = self.shares * self.entry_price * slippage

        # PnL calculation (including realized PnL from partial exits)
        gross_pnl = self.shares * (self.exit_price - self.entry_price)
        final_pnl = gross_pnl - self.commission - self.slippage_cost
        self.pnl = final_pnl + self.realized_pnl  # Add partial exit PnL

        # Calculate total PnL percent based on initial investment
        total_investment = (
            self.initial_shares * self.entry_price
            if self.initial_shares > 0
            else self.shares * self.entry_price
        )
        self.pnl_percent = (self.pnl / total_investment) * 100 if total_investment > 0 else 0

    def partial_close(
        self,
        exit_date: datetime,
        exit_price: float,
        exit_shares: int,
        exit_reason: str,
        commission_rate: float,
        slippage: float,
    ) -> float:
        """
        Partially close the trade

        Returns:
            PnL from this partial exit
        """
        if exit_shares > self.shares:
            exit_shares = self.shares

        if exit_shares <= 0:
            return 0.0

        # Calculate costs for partial exit
        entry_commission = exit_shares * self.entry_price * commission_rate
        exit_commission = exit_shares * exit_price * commission_rate
        partial_commission = entry_commission + exit_commission
        partial_slippage = exit_shares * self.entry_price * slippage

        # Calculate partial PnL
        gross_pnl = exit_shares * (exit_price - self.entry_price)
        partial_pnl = gross_pnl - partial_commission - partial_slippage

        # Record partial exit
        self.partial_exits.append(
            {
                "date": exit_date,
                "price": exit_price,
                "shares": exit_shares,
                "pnl": partial_pnl,
                "reason": exit_reason,
            }
        )

        # Update trade state
        self.shares -= exit_shares
        self.realized_pnl += partial_pnl
        self.commission += partial_commission
        self.slippage_cost += partial_slippage

        logger.info(
            f"📊 PARTIAL CLOSE {self.symbol}: {exit_shares} shares @ {exit_price:,.0f} | "
            f"PnL: {partial_pnl:+,.0f} | Remaining: {self.shares} shares"
        )

        return partial_pnl


@dataclass
class BacktestResult:
    """Backtesting results and metrics"""

    # Basic info
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Performance metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Win/Loss metrics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Other metrics
    avg_holding_days: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # Trade history
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


class BacktestEngine:
    """
    Backtesting engine for trading strategies

    Features:
    - Historical data simulation
    - Realistic transaction costs (commission + slippage)
    - DYNAMIC SLIPPAGE based on symbol liquidity (IMPROVED v3)
    - Position sizing and risk management
    - Performance metrics calculation
    - Trade history tracking
    - Vietnam market: lot size (100) and tick size (10 VND) enforcement
    - Circuit breaker and risk guards
    - Partial exit support
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.capital = self.config.initial_capital
        self.initial_capital = self.config.initial_capital

        # Active positions
        self.positions: Dict[str, Trade] = {}

        # Completed trades
        self.completed_trades: List[Trade] = []

        # Equity tracking
        self.equity_curve = []
        self.daily_returns = []

        # Circuit breaker state
        self.consecutive_losses: Dict[str, int] = {}  # Per-symbol consecutive losses
        self.daily_pnl: float = 0.0  # Track daily P&L
        self.current_date: Optional[datetime] = None
        self.circuit_breaker_triggered: bool = False

        # Risk tracking
        self.total_portfolio_risk: float = 0.0

        # IMPROVED v3: Liquidity data cache for dynamic slippage
        self._liquidity_cache: Dict[str, Dict] = {}

        slippage_mode = "DYNAMIC" if self.config.use_dynamic_slippage else "FIXED"
        logger.info(f"Backtesting engine initialized with {self.capital:,.0f} VND")
        logger.info(
            f"  Transaction cost: {self.config.commission_rate*100:.2f}% + {slippage_mode} slippage"
        )
        logger.info(f"  Lot size: {self.config.lot_size}, Tick size: {self.config.tick_size} VND")

    def set_liquidity_data(self, symbol: str, avg_daily_volume: float, avg_price: float):
        """
        Set liquidity data for a symbol to enable dynamic slippage calculation.

        IMPROVED v3: Call this before trading to get accurate slippage.

        Args:
            symbol: Stock symbol
            avg_daily_volume: Average daily trading volume in shares
            avg_price: Average price per share in VND
        """
        self._liquidity_cache[symbol] = {
            "avg_daily_volume": avg_daily_volume,
            "avg_price": avg_price,
            "liquidity_value": avg_daily_volume * avg_price,
        }

    def _get_dynamic_slippage(self, symbol: str, order_value: float) -> float:
        """
        Get dynamic slippage for a symbol based on liquidity.

        Args:
            symbol: Stock symbol
            order_value: Order value in VND

        Returns:
            Slippage rate
        """
        if not self.config.use_dynamic_slippage:
            return self.config.slippage

        liquidity = self._liquidity_cache.get(symbol, {})
        avg_daily_volume = liquidity.get("avg_daily_volume", 0)
        avg_price = liquidity.get("avg_price", 0)

        return calculate_dynamic_slippage(
            symbol=symbol,
            order_value=order_value,
            avg_daily_volume=avg_daily_volume,
            avg_price=avg_price,
            config=self.config,
        )

    def can_open_position(self, symbol: str, required_capital: float) -> bool:
        """
        Check if we can open a new position

        Includes circuit breaker and risk guard checks
        """
        # Check circuit breaker
        if self.circuit_breaker_triggered:
            logger.warning(f"Cannot open {symbol}: Circuit breaker triggered (daily loss limit)")
            return False

        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            logger.debug(
                f"Cannot open {symbol}: Max positions ({self.config.max_positions}) reached"
            )
            return False

        # Check available capital
        if required_capital > self.capital:
            logger.debug(
                f"Cannot open {symbol}: Insufficient capital ({self.capital:,.0f} < {required_capital:,.0f})"
            )
            return False

        # Check if already have position
        if symbol in self.positions:
            logger.debug(f"Cannot open {symbol}: Position already exists")
            return False

        # Check per-symbol consecutive losses
        if symbol in self.consecutive_losses:
            if self.consecutive_losses[symbol] >= self.config.max_consecutive_losses:
                logger.warning(
                    f"Cannot open {symbol}: {self.consecutive_losses[symbol]} consecutive losses "
                    f"(max: {self.config.max_consecutive_losses})"
                )
                return False

        # Check max portfolio risk
        current_exposure = sum(t.shares * t.entry_price for t in self.positions.values())
        new_exposure = current_exposure + required_capital
        exposure_pct = new_exposure / self.initial_capital

        if exposure_pct > (1 - self.config.max_portfolio_risk):
            logger.debug(
                f"Cannot open {symbol}: Would exceed max portfolio exposure "
                f"({exposure_pct*100:.1f}% > {(1-self.config.max_portfolio_risk)*100:.1f}%)"
            )
            return False

        return True

    def open_position(
        self,
        symbol: str,
        date: datetime,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        reason: str = "",
        position_size_multiplier: float = 1.0,  # NEW: Support dynamic position sizing
    ) -> Optional[Trade]:
        """
        Open a new position

        Args:
            symbol: Stock symbol
            date: Entry date
            entry_price: Entry price (will be rounded to tick size)
            stop_loss: Stop loss price
            take_profit: Take profit price
            reason: Entry reason
            position_size_multiplier: Multiplier for position size (0.3 to 1.5)
        """
        # Round entry price to tick size (Vietnam market)
        entry_price = round_to_tick_size(entry_price, self.config.tick_size)

        # Calculate position size with multiplier
        # Clamp multiplier to valid range
        multiplier = max(0.3, min(1.5, position_size_multiplier))
        position_capital = self.capital * self.config.position_size_pct * multiplier

        # Calculate shares and round to lot size (Vietnam: 100 shares)
        raw_shares = int(position_capital / entry_price)
        shares = round_to_lot_size(raw_shares, self.config.lot_size)

        if shares <= 0:
            logger.warning(f"Cannot open {symbol}: Shares = 0 after lot size rounding")
            return None

        actual_cost = shares * entry_price

        # Check if can open
        if not self.can_open_position(symbol, actual_cost):
            return None

        # Round stop loss and take profit to tick size
        stop_loss = round_to_tick_size(stop_loss, self.config.tick_size)
        take_profit = round_to_tick_size(take_profit, self.config.tick_size)

        # Create trade
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=entry_price,
            shares=shares,
            initial_shares=shares,  # Track original shares for partial exits
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_reason=reason,
        )

        # Update capital
        self.capital -= actual_cost
        self.positions[symbol] = trade

        # Update daily tracking
        self._update_daily_tracking(date)

        logger.info(
            f"✅ OPEN {symbol} @ {entry_price:,.0f} x {shares} shares "
            f"(lot-rounded from {raw_shares}) = {actual_cost:,.0f} VND"
        )
        if multiplier != 1.0:
            logger.info(f"   Position size multiplier: {multiplier:.2f}x")

        return trade

    def _update_daily_tracking(self, date: datetime):
        """Update daily P&L tracking and check circuit breaker"""
        if self.current_date is None or date.date() != self.current_date.date():
            # New day - reset daily P&L
            self.current_date = date
            self.daily_pnl = 0.0
            self.circuit_breaker_triggered = False

    def _check_daily_circuit_breaker(self):
        """Check if daily loss circuit breaker should trigger"""
        if not self.config.daily_loss_circuit_breaker:
            return

        daily_loss_pct = abs(self.daily_pnl) / self.initial_capital
        if self.daily_pnl < 0 and daily_loss_pct >= self.config.max_daily_loss:
            self.circuit_breaker_triggered = True
            logger.warning(
                f"🚨 CIRCUIT BREAKER TRIGGERED: Daily loss {daily_loss_pct*100:.2f}% "
                f">= {self.config.max_daily_loss*100:.1f}%"
            )

    def close_position(
        self, symbol: str, date: datetime, exit_price: float, reason: str = ""
    ) -> Optional[Trade]:
        """Close an existing position (full exit) with DYNAMIC SLIPPAGE"""
        if symbol not in self.positions:
            logger.warning(f"Cannot close {symbol}: No open position")
            return None

        # CRITICAL FIX: Validate exit_price before processing
        if exit_price is None or exit_price <= 0:
            logger.error(f"❌ [{symbol}] Invalid exit_price: {exit_price}. Cannot close position.")
            return None

        trade = self.positions[symbol]

        # Round exit price to tick size
        exit_price = round_to_tick_size(exit_price, self.config.tick_size)

        # IMPROVED v3: Calculate dynamic slippage based on order size
        order_value = trade.shares * exit_price
        dynamic_slippage = self._get_dynamic_slippage(symbol, order_value)

        # Close the trade with dynamic slippage
        trade.close_trade(
            exit_date=date,
            exit_price=exit_price,
            exit_reason=reason,
            commission_rate=self.config.commission_rate,
            slippage=dynamic_slippage,
        )

        # Update capital (for remaining shares after any partial exits)
        exit_value = trade.shares * exit_price
        # Note: commission and slippage for final exit already calculated in close_trade
        self.capital += exit_value

        # Update daily P&L and check circuit breaker
        self._update_daily_tracking(date)
        self.daily_pnl += trade.pnl
        self._check_daily_circuit_breaker()

        # Update consecutive losses tracking
        if trade.pnl < 0:
            self.consecutive_losses[symbol] = self.consecutive_losses.get(symbol, 0) + 1
        else:
            self.consecutive_losses[symbol] = 0  # Reset on win

        # Move to completed trades
        del self.positions[symbol]
        self.completed_trades.append(trade)

        pnl_str = f"+{trade.pnl:,.0f}" if trade.pnl >= 0 else f"{trade.pnl:,.0f}"
        logger.info(
            f"❌ CLOSE {symbol} @ {exit_price:,.0f} | PnL: {pnl_str} ({trade.pnl_percent:+.2f}%) | {reason}"
        )

        return trade

    def partial_close_position(
        self,
        symbol: str,
        date: datetime,
        exit_price: float,
        exit_percent: float,
        reason: str = "",
    ) -> Optional[float]:
        """
        Partially close a position

        Args:
            symbol: Stock symbol
            date: Exit date
            exit_price: Exit price
            exit_percent: Percentage of position to close (0.0 to 1.0)
            reason: Exit reason

        Returns:
            PnL from partial exit, or None if failed
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot partial close {symbol}: No open position")
            return None

        # CRITICAL FIX: Validate exit_price before processing
        if exit_price is None or exit_price <= 0:
            logger.error(
                f"❌ [{symbol}] Invalid exit_price: {exit_price}. Cannot partial close position."
            )
            return None

        trade = self.positions[symbol]

        # Round exit price to tick size
        exit_price = round_to_tick_size(exit_price, self.config.tick_size)

        # Calculate shares to exit (rounded to lot size)
        exit_shares = int(trade.shares * exit_percent)
        exit_shares = round_to_lot_size(exit_shares, self.config.lot_size)

        if exit_shares <= 0:
            logger.warning(f"Cannot partial close {symbol}: Exit shares = 0 after lot rounding")
            return None

        # If exit_shares equals remaining shares, do full close instead
        if exit_shares >= trade.shares:
            logger.info(
                f"Partial close {symbol}: Converting to full close (exit_shares >= remaining)"
            )
            result = self.close_position(symbol, date, exit_price, reason)
            return result.pnl if result else None

        # IMPROVED v3: Calculate dynamic slippage for partial exit
        order_value = exit_shares * exit_price
        dynamic_slippage = self._get_dynamic_slippage(symbol, order_value)

        # Execute partial close with dynamic slippage
        partial_pnl = trade.partial_close(
            exit_date=date,
            exit_price=exit_price,
            exit_shares=exit_shares,
            exit_reason=reason,
            commission_rate=self.config.commission_rate,
            slippage=dynamic_slippage,
        )

        # Update capital with proceeds from partial exit
        exit_value = exit_shares * exit_price
        self.capital += exit_value - (
            exit_shares * exit_price * (self.config.commission_rate + dynamic_slippage)
        )

        # Update daily P&L
        self._update_daily_tracking(date)
        self.daily_pnl += partial_pnl

        return partial_pnl

    def emergency_close_all(
        self, date: datetime, current_prices: Dict[str, float], reason: str = "Emergency Exit"
    ):
        """
        Emergency close all positions (circuit breaker triggered)

        Args:
            date: Exit date
            current_prices: Dict of symbol -> current price
            reason: Exit reason
        """
        logger.warning(f"🚨 EMERGENCY CLOSE ALL: {reason}")

        for symbol in list(self.positions.keys()):
            price = current_prices.get(symbol)
            if price is None:
                price = self.positions[symbol].entry_price  # Fallback to entry price
            self.close_position(symbol, date, price, reason)

    def update_equity(self, date: datetime, current_prices: Dict[str, float]):
        """Update equity curve with current prices"""
        # Calculate portfolio value
        positions_value = sum(
            trade.shares * current_prices.get(symbol, trade.entry_price)
            for symbol, trade in self.positions.items()
        )

        total_equity = self.capital + positions_value

        self.equity_curve.append(
            {
                "date": date,
                "equity": total_equity,
                "cash": self.capital,
                "positions_value": positions_value,
                "num_positions": len(self.positions),
            }
        )

        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]["equity"]
            daily_return = (total_equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)

    def calculate_results(self) -> BacktestResult:
        """Calculate final backtest results and metrics"""
        if not self.equity_curve:
            raise ValueError("No equity curve data - run backtest first")

        equity_df = pd.DataFrame(self.equity_curve)

        # Basic metrics
        start_date = equity_df["date"].iloc[0]
        end_date = equity_df["date"].iloc[-1]
        final_capital = equity_df["equity"].iloc[-1]

        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (
            ((final_capital / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        )

        # Sharpe ratio (assuming risk-free rate = 0)
        if len(self.daily_returns) > 0:
            daily_returns_std = np.std(self.daily_returns)
            avg_daily_return = np.mean(self.daily_returns)
            sharpe_ratio = (
                (avg_daily_return / daily_returns_std) * np.sqrt(252)
                if daily_returns_std > 0
                else 0
            )
        else:
            sharpe_ratio = 0

        # Max drawdown
        equity_series = equity_df["equity"]
        rolling_max = equity_series.expanding().max()
        drawdown = equity_series - rolling_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (
            (max_drawdown / rolling_max[drawdown.idxmin()]) * 100 if len(rolling_max) > 0 else 0
        )

        # Trade statistics
        trades = self.completed_trades
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Win/Loss metrics
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        # Other metrics
        avg_holding_days = np.mean([t.holding_days for t in trades]) if trades else 0
        total_commission = sum(t.commission for t in trades)
        total_slippage = sum(t.slippage_cost for t in trades)

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_days=avg_holding_days,
            total_commission=total_commission,
            total_slippage=total_slippage,
            trades=trades,
            equity_curve=equity_df,
        )

    def print_results(self, result: BacktestResult):
        """Print formatted backtest results"""
        print("\n" + "=" * 80)
        print("BACKTESTING RESULTS")
        print("=" * 80)

        print(f"\n📅 Period: {result.start_date.date()} → {result.end_date.date()}")
        print(f"💰 Initial Capital: {result.initial_capital:,.0f} VND")
        print(f"💰 Final Capital: {result.final_capital:,.0f} VND")

        print("\n📊 Performance:")
        print(f"  Total Return: {result.total_return:+,.0f} VND ({result.total_return_pct:+.2f}%)")
        print(f"  Annualized Return: {result.annualized_return:.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:,.0f} VND ({result.max_drawdown_pct:.2f}%)")

        print("\n📈 Trade Statistics:")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Winning Trades: {result.winning_trades} ({result.win_rate:.1f}%)")
        print(f"  Losing Trades: {result.losing_trades}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")

        print("\n💵 Win/Loss Analysis:")
        print(f"  Average Win: {result.avg_win:+,.0f} VND")
        print(f"  Average Loss: {result.avg_loss:+,.0f} VND")
        print(f"  Largest Win: {result.largest_win:+,.0f} VND")
        print(f"  Largest Loss: {result.largest_loss:+,.0f} VND")

        print("\n⏱️  Other Metrics:")
        print(f"  Avg Holding Days: {result.avg_holding_days:.1f}")
        print(f"  Total Commission: {result.total_commission:,.0f} VND")
        print(f"  Total Slippage: {result.total_slippage:,.0f} VND")

        # Transaction cost summary
        total_costs = result.total_commission + result.total_slippage
        cost_pct = (total_costs / result.initial_capital) * 100 if result.initial_capital > 0 else 0
        print(f"  Total Transaction Costs: {total_costs:,.0f} VND ({cost_pct:.2f}% of capital)")

        print("=" * 80 + "\n")
