# -*- coding: utf-8 -*-
"""
Backtesting Engine - Test Trading Strategies with Historical Data
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting configuration"""

    initial_capital: float = 100_000_000  # 100M VND
    commission_rate: float = 0.0015  # 0.15% per trade
    slippage: float = 0.001  # 0.1% price slippage
    max_positions: int = 5  # Max concurrent positions
    position_size_pct: float = 0.20  # 20% per position

    # Risk management
    max_loss_per_trade: float = 0.03  # 3% max loss per trade
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_portfolio_risk: float = 0.15  # 15% max total portfolio risk


@dataclass
class Trade:
    """Individual trade record"""

    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
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

    def close_trade(
        self,
        exit_date: datetime,
        exit_price: float,
        exit_reason: str,
        commission_rate: float,
        slippage: float,
    ):
        """Close the trade and calculate PnL"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.holding_days = (exit_date - self.entry_date).days

        # Calculate costs
        entry_commission = self.shares * self.entry_price * commission_rate
        exit_commission = self.shares * self.exit_price * commission_rate
        self.commission = entry_commission + exit_commission

        # Slippage cost (assumes adverse price movement)
        self.slippage_cost = self.shares * self.entry_price * slippage

        # PnL calculation
        gross_pnl = self.shares * (self.exit_price - self.entry_price)
        self.pnl = gross_pnl - self.commission - self.slippage_cost
        self.pnl_percent = (self.pnl / (self.shares * self.entry_price)) * 100


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
    - Position sizing and risk management
    - Performance metrics calculation
    - Trade history tracking
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

        logger.info(f"Backtesting engine initialized with {self.capital:,.0f} VND")

    def can_open_position(self, symbol: str, required_capital: float) -> bool:
        """Check if we can open a new position"""
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

        return True

    def open_position(
        self,
        symbol: str,
        date: datetime,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        reason: str = "",
    ) -> Optional[Trade]:
        """Open a new position"""
        # Calculate position size
        position_capital = self.capital * self.config.position_size_pct
        shares = int(position_capital / entry_price)

        if shares <= 0:
            logger.warning(f"Cannot open {symbol}: Shares = 0")
            return None

        actual_cost = shares * entry_price

        # Check if can open
        if not self.can_open_position(symbol, actual_cost):
            return None

        # Create trade
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=entry_price,
            shares=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_reason=reason,
        )

        # Update capital
        self.capital -= actual_cost
        self.positions[symbol] = trade

        logger.info(
            f"✅ OPEN {symbol} @ {entry_price:,.0f} x {shares} shares = {actual_cost:,.0f} VND"
        )
        return trade

    def close_position(
        self, symbol: str, date: datetime, exit_price: float, reason: str = ""
    ) -> Optional[Trade]:
        """Close an existing position"""
        if symbol not in self.positions:
            logger.warning(f"Cannot close {symbol}: No open position")
            return None

        trade = self.positions[symbol]

        # Close the trade
        trade.close_trade(
            exit_date=date,
            exit_price=exit_price,
            exit_reason=reason,
            commission_rate=self.config.commission_rate,
            slippage=self.config.slippage,
        )

        # Update capital
        exit_value = trade.shares * exit_price
        self.capital += exit_value - trade.commission - trade.slippage_cost

        # Move to completed trades
        del self.positions[symbol]
        self.completed_trades.append(trade)

        pnl_str = f"+{trade.pnl:,.0f}" if trade.pnl >= 0 else f"{trade.pnl:,.0f}"
        logger.info(
            f"❌ CLOSE {symbol} @ {exit_price:,.0f} | PnL: {pnl_str} ({trade.pnl_percent:+.2f}%) | {reason}"
        )

        return trade

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
        print("💰 Initial Capital: {result.initial_capital:,.0f} VND")
        print("💰 Final Capital: {result.final_capital:,.0f} VND")

        print("\n📊 Performance:")
        print(f"  Total Return: {result.total_return:+,.0f} VND ({result.total_return_pct:+.2f}%)")
        print("  Annualized Return: {result.annualized_return:.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:,.0f} VND ({result.max_drawdown_pct:.2f}%)")

        print("\n📈 Trade Statistics:")
        print(f"  Total Trades: {result.total_trades}")
        print("  Winning Trades: {result.winning_trades} ({result.win_rate:.1f}%)")
        print(f"  Losing Trades: {result.losing_trades}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")

        print("\n💵 Win/Loss Analysis:")
        print("  Average Win: {result.avg_win:+,.0f} VND")
        print("  Average Loss: {result.avg_loss:+,.0f} VND")
        print("  Largest Win: {result.largest_win:+,.0f} VND")
        print("  Largest Loss: {result.largest_loss:+,.0f} VND")

        print("\n⏱️  Other Metrics:")
        print(f"  Avg Holding Days: {result.avg_holding_days:.1f}")
        print("  Total Commission: {result.total_commission:,.0f} VND")
        print("  Total Slippage: {result.total_slippage:,.0f} VND")

        print("=" * 80 + "\n")
