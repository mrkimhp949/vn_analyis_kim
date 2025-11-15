"""
Paper Trading System
Mô phỏng thực thi lệnh để test strategy mà không cần tiền thật
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from portfolio_manager import get_portfolio_manager

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
        except Exception as e:
            return False, f"Lỗi DB khi thêm vị thế {symbol}: {e}", None

        # Update cash
        self.account["cash"] -= total_cost

        # Record trade log
        self.account["trades"].append(asdict(trade))

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
                from data_loader import load_data

                df = load_data(symbol, lookback=5, use_cache=True)
                if df.empty:
                    return False, f"Không thể lấy giá {symbol}", None
                price = df["close"].iloc[-1]
            except Exception:
                return False, f"Lỗi lấy giá {symbol}", None

        # Apply slippage
        execution_price = price * (1 - self.slippage_pct)

        try:
            # Determine shares to sell for logging
            if exit_type == "FULL":
                shares_to_sell = available_shares
            else:
                try:
                    percentage = (
                        float(exit_type.replace("PARTIAL_", "").replace("%", ""))
                        / 100.0
                    )
                    shares_to_sell = int(available_shares * percentage)
                except (ValueError, TypeError):
                    shares_to_sell = available_shares  # Fallback to full sell on error

            # Calculate proceeds for cash update
            gross_proceeds = execution_price * shares_to_sell
            commission = gross_proceeds * self.commission_rate
            net_proceeds = gross_proceeds - commission
            self.account["cash"] += net_proceeds

            # Use PortfolioManager to handle the database logic
            self.portfolio_manager.handle_exit(
                symbol=symbol,
                exit_price=execution_price,
                exit_type=exit_type,
                reason=reason,
            )

            # Create a representative trade object for logging
            trade = PaperTrade(
                symbol=symbol,
                action=f"SELL_{exit_type}",
                shares=shares_to_sell,
                price=execution_price,
                timestamp=datetime.now().isoformat(),
                status="FILLED",
            )
            self.account["trades"].append(asdict(trade))
            self.save_account()

            return (
                True,
                f"✅ Xử lý lệnh bán {exit_type} cho {symbol} @ {execution_price:,.0f} VNĐ",
                trade,
            )

        except Exception as e:
            return False, f"Lỗi khi xử lý lệnh bán {symbol}: {e}", None

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
        """Lấy lịch sử giao dịch"""
        if not self.portfolio_manager:
            return []
        return self.portfolio_manager.db.get_trades(symbol)

    def get_statistics(self) -> Dict:
        """Lấy thống kê trading"""
        if not self.portfolio_manager:
            return {}

        trades = self.get_trade_history()

        buy_trades = [t for t in trades if t.get("action") == "BUY"]
        sell_trades = [t for t in trades if "SELL" in t.get("action", "")]

        total_commission = sum(
            (t.get("price", 0) * t.get("shares", 0) * self.commission_rate)
            for t in trades
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
            f"{'📈' if portfolio_value.get('total_pnl', 0) >= 0 else '📉'} P&L: {portfolio_value.get('total_pnl', 0):+,.0f} VNĐ ({portfolio_value.get('total_pnl_percent', 0):+.2f}%)",
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
