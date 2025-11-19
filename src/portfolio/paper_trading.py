"""
Paper Trading System
Mô phỏng thực thi lệnh để test strategy mà không cần tiền thật
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.portfolio.manager import get_portfolio_manager

logger = logging.getLogger(__name__)

PAPER_TRADING_FILE = "paper_trading.json"


@dataclass
class PaperTrade:
    """Một giao dịch paper trading"""

    symbol: str
    action: str  # 'BUY' or 'SELL'
    shares: int
    price: float
    timestamp: str
    signal_confidence: Optional[float] = None
    signal_reason: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str = "PENDING"  # PENDING, FILLED, CANCELLED, REJECTED


class PaperTradingAccount:
    """
    Tài khoản paper trading

    Mô phỏng:
    - Vốn ban đầu
    - Thực thi lệnh với slippage và fees
    - Tracking PnL
    - So sánh với real portfolio
    """

    def __init__(
        self,
        initial_capital: float = 100_000_000,
        commission_rate: float = 0.0015,  # 0.15% phí giao dịch
        slippage_pct: float = 0.001,  # 0.1% slippage
        account_file: str = PAPER_TRADING_FILE,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        self.account_file = account_file
        self.portfolio_manager = get_portfolio_manager()

        self.load_account()

    def load_account(self):
        """Load tài khoản từ file"""
        if not os.path.exists(self.account_file):
            self.account = {
                "initial_capital": self.initial_capital,
                "cash": self.initial_capital,
                "positions": {},  # DEPRECATED
                "trades": [],  # List of PaperTrade
                "daily_pnl": [],  # [{date, pnl, equity}]
                "created_at": datetime.now().isoformat(),
            }
            self.save_account()
        else:
            try:
                with open(self.account_file, "r", encoding="utf-8") as f:
                    self.account = json.load(f)
            except Exception:
                self.account = {
                    "initial_capital": self.initial_capital,
                    "cash": self.initial_capital,
                    "positions": {},
                    "trades": [],
                    "daily_pnl": [],
                    "created_at": datetime.now().isoformat(),
                }

    def save_account(self):
        """Lưu tài khoản vào file"""
        with open(self.account_file, "w", encoding="utf-8") as f:
            json.dump(self.account, f, indent=2, ensure_ascii=False)

    def execute_buy(
        self,
        symbol: str,
        shares: int,
        price: float,
        signal_confidence: Optional[float] = None,
        signal_reason: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[PaperTrade]]:
        """
        Thực thi lệnh mua

        Returns:
            (success, message, trade)
        """
        # Apply slippage
        execution_price = price * (1 + self.slippage_pct)

        # Calculate cost
        gross_cost = execution_price * shares
        commission = gross_cost * self.commission_rate
        total_cost = gross_cost + commission

        # Check if enough cash
        if total_cost > self.account["cash"]:
            return (
                False,
                f"Không đủ tiền. Cần {total_cost:,.0f} VNĐ, có {self.account['cash']:,.0f} VNĐ",
                None,
            )

        # Create trade object for logging
        trade = PaperTrade(
            symbol=symbol.upper(),
            action="BUY",
            shares=shares,
            price=execution_price,
            timestamp=datetime.now().isoformat(),
            signal_confidence=signal_confidence,
            signal_reason=signal_reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="FILLED",
        )

        # Use PortfolioManager to add the position to the database
        try:
            self.portfolio_manager.add_position(
                symbol=symbol.upper(),
                shares=shares,
                entry_price=execution_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "signal_confidence": signal_confidence,
                    "signal_reason": signal_reason,
                },
            )
        except Exception:
            return False, f"Lỗi DB khi thêm vị thế {symbol}", None

        # Update cash
        self.account["cash"] -= total_cost

        # Record trade log (in-memory for backward compatibility)
        self.account["trades"].append(asdict(trade))

        # Calculate slippage cost
        slippage_cost = price * self.slippage_pct * shares

        # Save to database for persistence and querying
        try:
            if self.portfolio_manager:
                self.portfolio_manager.db.save_trade(
                    symbol=symbol.upper(),
                    action="BUY",
                    shares=shares,
                    price=execution_price,
                    total_value=total_cost,
                    trade_date=datetime.now().isoformat(),
                    reason=signal_reason or "Paper trading",
                    metadata={
                        "signal_confidence": signal_confidence,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "commission": commission,
                        "slippage": slippage_cost,
                    },
                )
        except Exception as e:
            logger.warning(f"Failed to save trade to DB: {e}")

        self.save_account()

        return (
            True,
            f"✅ Mua {shares} CP {symbol} @ {execution_price:,.0f} VNĐ (phí: {commission:,.0f})",
            trade,
        )

    def execute_sell(
        self,
        symbol: str,
        shares: Optional[int] = None,  # This is now for logging/verification, not logic
        price: Optional[float] = None,
        exit_type: str = "FULL",
        reason: str = "Exit Signal",
    ) -> Tuple[bool, str, Optional[PaperTrade]]:
        """
        Thực thi lệnh bán (toàn bộ hoặc một phần).

        Args:
            exit_type: 'FULL', 'PARTIAL_50%', 'PARTIAL_30%', etc.
        """
        symbol = symbol.upper()

        if not self.portfolio_manager:
            return False, "Portfolio Manager không khả dụng", None

        positions = self.portfolio_manager.get_positions()
        if symbol not in positions:
            return False, f"Không có {symbol} trong portfolio", None

        position = positions[symbol]
        available_shares = position["shares"]

        # Get current price if not provided
        if price is None:
            try:
                from src.data.loader import load_data
                from src.utils.dataframe_utils import safe_get_latest

                df = load_data(symbol, lookback=5, use_cache=True)
                if df is None or df.empty:
                    # Fallback: use average entry price if market data unavailable
                    price = position.get("avg_price", 0)
                else:
                    price = safe_get_latest(df, "close", 0)
            except Exception:
                # Fallback: use average entry price if fetch fails
                price = position.get("avg_price", 0)

            if price is None or price <= 0:
                return False, f"Không thể lấy giá {symbol} (hãy cung cấp PRICE)", None

        # Apply slippage
        execution_price = price * (1 - self.slippage_pct)

        try:
            # Determine shares to sell and perform DB update
            direct_partial = (
                shares is not None and isinstance(shares, int) and 0 < shares < available_shares
            )

            if direct_partial:
                shares_to_sell = shares
                # Update cash based on exact shares
                gross_proceeds = execution_price * shares_to_sell
                commission = gross_proceeds * self.commission_rate
                net_proceeds = gross_proceeds - commission
                self.account["cash"] += net_proceeds

                # Use PortfolioManager to reduce exact shares
                self.portfolio_manager.reduce_position(
                    symbol=symbol,
                    shares_to_sell=shares_to_sell,
                    exit_price=execution_price,
                    reason=reason or "Manual partial sell",
                )
            else:
                # Derive shares_to_sell via exit_type
                if exit_type == "FULL" or shares is None or shares >= available_shares:
                    shares_to_sell = available_shares
                    exit_type = "FULL"
                else:
                    # If user passed shares >= available, treat as FULL
                    shares_to_sell = available_shares
                    exit_type = "FULL"

                gross_proceeds = execution_price * shares_to_sell
                commission = gross_proceeds * self.commission_rate
                net_proceeds = gross_proceeds - commission
                self.account["cash"] += net_proceeds

                # Delegate to PortfolioManager to handle exit
                self.portfolio_manager.handle_exit(
                    symbol=symbol,
                    exit_price=execution_price,
                    exit_type=exit_type,
                    reason=reason,
                )

            # Create a representative trade object for logging
            trade = PaperTrade(
                symbol=symbol,
                action=(f"SELL_PARTIAL" if direct_partial else f"SELL_{exit_type}"),
                shares=shares_to_sell,
                price=execution_price,
                timestamp=datetime.now().isoformat(),
                status="FILLED",
            )
            self.account["trades"].append(asdict(trade))

            # Calculate exit slippage cost
            exit_slippage_cost = price * self.slippage_pct * shares_to_sell

            # Save to database for persistence and querying
            try:
                if self.portfolio_manager:
                    self.portfolio_manager.db.save_trade(
                        symbol=symbol.upper(),
                        action=f"SELL_{exit_type}" if not direct_partial else "SELL_PARTIAL",
                        shares=shares_to_sell,
                        price=execution_price,
                        total_value=gross_proceeds,
                        trade_date=datetime.now().isoformat(),
                        reason=reason or "Paper trading exit",
                        metadata={
                            "exit_type": exit_type,
                            "commission": commission,
                            "slippage": exit_slippage_cost,
                            "partial": direct_partial,
                        },
                    )
            except Exception as e:
                logger.warning(f"Failed to save trade to DB: {e}")

            self.save_account()

            return (
                True,
                f"✅ Xử lý lệnh bán {'PARTIAL' if direct_partial else exit_type} cho {symbol} @ {execution_price:,.0f} VNĐ",
                trade,
            )

        except Exception:
            return False, f"Lỗi khi xử lý lệnh bán {symbol}", None

    def get_portfolio_value(self) -> Dict:
        """
        Tính giá trị portfolio hiện tại
        """
        if not self.portfolio_manager:
            return {}

        return self.portfolio_manager.get_portfolio_value()

    def record_daily_pnl(self):
        """Ghi lại PnL trong ngày"""
        if not self.portfolio_manager:
            return

        portfolio_value = self.get_portfolio_value()

        today = datetime.now().date().isoformat()

        # The cash value in portfolio_value from manager might be more accurate
        # but for now, we use the cash managed by this paper account class.
        # A future refactor could consolidate cash management into the PortfolioManager.
        current_cash = self.account.get("cash", 0)

        daily_record = {
            "date": today,
            "equity": portfolio_value.get("total_positions_value", 0) + current_cash,
            "pnl": portfolio_value.get("total_pnl", 0),
            "return_pct": portfolio_value.get("total_pnl_percent", 0),
            "cash": current_cash,
            "positions_value": portfolio_value.get("total_positions_value", 0),
        }

        # Check if already recorded today
        existing_index = -1
        for i, record in enumerate(self.account["daily_pnl"]):
            if record.get("date") == today:
                existing_index = i
                break

        if existing_index != -1:
            self.account["daily_pnl"][existing_index] = daily_record
        else:
            self.account["daily_pnl"].append(daily_record)

        self.save_account()

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Dict]:
        """Lấy lịch sử giao dịch từ database"""
        if not self.portfolio_manager:
            # Fallback to in-memory trades if DB not available
            if symbol:
                return [
                    t
                    for t in self.account.get("trades", [])
                    if t.get("symbol", "").upper() == symbol.upper()
                ]
            return self.account.get("trades", [])

        try:
            trades = self.portfolio_manager.db.get_trades(symbol, limit=100)
            # Ensure all trades have required fields
            for trade in trades:
                if "timestamp" not in trade and "trade_date" in trade:
                    trade["timestamp"] = trade["trade_date"]
            return trades
        except Exception as e:
            logger.error(f"Error getting trades from DB: {e}, falling back to in-memory")
            # Fallback to in-memory trades
            if symbol:
                return [
                    t
                    for t in self.account.get("trades", [])
                    if t.get("symbol", "").upper() == symbol.upper()
                ]
            return self.account.get("trades", [])

    def get_statistics(self) -> Dict:
        """Lấy thống kê trading"""
        if not self.portfolio_manager:
            return {}

        trades = self.get_trade_history()

        buy_trades = [t for t in trades if t.get("action") == "BUY"]
        sell_trades = [t for t in trades if "SELL" in t.get("action", "")]

        total_commission = sum(
            (t.get("price", 0) * t.get("shares", 0) * self.commission_rate) for t in trades
        )

        portfolio_value = self.get_portfolio_value()
        current_cash = self.account.get("cash", 0)

        return {
            "total_trades": len(trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total_commission": total_commission,
            "current_portfolio_value": portfolio_value.get("total_positions_value", 0)
            + current_cash,
            "current_pnl": portfolio_value.get("total_pnl", 0),
            "current_return_pct": portfolio_value.get("total_pnl_percent", 0),
            "num_positions": portfolio_value.get("num_positions", 0),
        }

    def format_account_summary(self) -> str:
        """Format tóm tắt tài khoản"""
        if not self.portfolio_manager:
            return "Portfolio Manager không khả dụng."

        stats = self.get_statistics()
        portfolio_value = self.get_portfolio_value()
        current_cash = self.account.get("cash", 0)
        total_value = portfolio_value.get("total_positions_value", 0) + current_cash

        lines = [
            "📊 *Paper Trading Account (DB-backed):*\n",
            f"💰 Vốn ban đầu: {self.initial_capital:,.0f} VNĐ",
            f"💵 Tiền mặt (ước tính): {current_cash:,.0f} VNĐ",
            f"📈 Giá trị vị thế: {portfolio_value.get('total_positions_value', 0):,.0f} VNĐ",
            f"💼 Tổng giá trị: {total_value:,.0f} VNĐ",
            f"{'📈' if portfolio_value.get('total_pnl', 0) >= 0 else '📉'} P&L: "
            f"{portfolio_value.get('total_pnl', 0):+,.0f} VNĐ "
            f"({portfolio_value.get('total_pnl_percent', 0):+.2f}%)",
            f"📊 Số vị thế: {portfolio_value.get('num_positions', 0)}",
            f"🔄 Tổng giao dịch: {stats.get('total_trades', 0)}",
            f"💸 Tổng phí (ước tính): {stats.get('total_commission', 0):,.0f} VNĐ",
        ]

        return "\n".join(lines)


# Global instance
_paper_account = None


def get_paper_account(initial_capital: float = 100_000_000) -> PaperTradingAccount:
    """Get or create paper trading account"""
    global _paper_account
    if _paper_account is None:
        _paper_account = PaperTradingAccount(initial_capital=initial_capital)
    return _paper_account
