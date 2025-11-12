"""
Paper Trading System
Mô phỏng thực thi lệnh để test strategy mà không cần tiền thật
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

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
        account_file: str = PAPER_TRADING_FILE
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        self.account_file = account_file
        
        self.load_account()
    
    def load_account(self):
        """Load tài khoản từ file"""
        if not os.path.exists(self.account_file):
            self.account = {
                "initial_capital": self.initial_capital,
                "cash": self.initial_capital,
                "positions": {},  # {symbol: {shares, avg_price, entry_date}}
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
        take_profit: Optional[float] = None
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
            return False, f"Không đủ tiền. Cần {total_cost:,.0f} VNĐ, có {self.account['cash']:,.0f} VNĐ", None
        
        # Create trade
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
            status="FILLED"
        )
        
        # Update positions
        if symbol.upper() in self.account["positions"]:
            pos = self.account["positions"][symbol.upper()]
            total_shares = pos["shares"] + shares
            total_cost = (pos["shares"] * pos["avg_price"]) + gross_cost
            pos["shares"] = total_shares
            pos["avg_price"] = total_cost / total_shares
        else:
            self.account["positions"][symbol.upper()] = {
                "shares": shares,
                "avg_price": execution_price,
                "entry_date": datetime.now().isoformat(),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
        
        # Update cash
        self.account["cash"] -= total_cost
        
        # Record trade
        self.account["trades"].append(asdict(trade))
        
        self.save_account()
        
        return True, f"✅ Mua {shares} CP {symbol} @ {execution_price:,.0f} VNĐ (phí: {commission:,.0f})", trade
    
    def execute_sell(
        self,
        symbol: str,
        shares: Optional[int] = None,
        price: Optional[float] = None
    ) -> Tuple[bool, str, Optional[PaperTrade]]:
        """
        Thực thi lệnh bán
        
        Returns:
            (success, message, trade)
        """
        symbol = symbol.upper()
        
        if symbol not in self.account["positions"]:
            return False, f"Không có {symbol} trong portfolio", None
        
        position = self.account["positions"][symbol]
        available_shares = position["shares"]
        
        if shares is None:
            shares = available_shares  # Bán hết
        
        if shares > available_shares:
            return False, f"Không đủ cổ phiếu. Có {available_shares} CP, muốn bán {shares} CP", None
        
        # Get current price if not provided
        if price is None:
            try:
                from data_loader import load_data
                df = load_data(symbol, lookback=5, use_cache=True)
                if df.empty:
                    return False, f"Không thể lấy giá {symbol}", None
                price = df['close'].iloc[-1]
            except Exception:
                return False, f"Lỗi lấy giá {symbol}", None
        
        # Apply slippage
        execution_price = price * (1 - self.slippage_pct)
        
        # Calculate proceeds
        gross_proceeds = execution_price * shares
        commission = gross_proceeds * self.commission_rate
        net_proceeds = gross_proceeds - commission
        
        # Create trade
        trade = PaperTrade(
            symbol=symbol,
            action="SELL",
            shares=shares,
            price=execution_price,
            timestamp=datetime.now().isoformat(),
            status="FILLED"
        )
        
        # Update positions
        if shares == available_shares:
            # Bán hết
            del self.account["positions"][symbol]
        else:
            position["shares"] -= shares
        
        # Update cash
        self.account["cash"] += net_proceeds
        
        # Record trade
        self.account["trades"].append(asdict(trade))
        
        self.save_account()
        
        return True, f"✅ Bán {shares} CP {symbol} @ {execution_price:,.0f} VNĐ (phí: {commission:,.0f})", trade
    
    def get_portfolio_value(self) -> Dict:
        """
        Tính giá trị portfolio hiện tại
        
        Returns:
            Dict với:
            - cash: Tiền mặt
            - positions_value: Giá trị các vị thế
            - total_value: Tổng giá trị
            - total_pnl: Lợi nhuận/lỗ
            - total_return_pct: % return
        """
        positions_value = 0.0
        positions_cost = 0.0
        
        for symbol, position in self.account["positions"].items():
            try:
                from data_loader import load_data
                df = load_data(symbol, lookback=5, use_cache=True)
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    shares = position["shares"]
                    avg_price = position["avg_price"]
                    
                    positions_value += current_price * shares
                    positions_cost += avg_price * shares
            except Exception:
                continue
        
        total_value = self.account["cash"] + positions_value
        total_invested = self.initial_capital - self.account["cash"] + positions_cost
        total_pnl = total_value - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        return {
            "cash": self.account["cash"],
            "positions_value": positions_value,
            "total_value": total_value,
            "total_invested": total_invested,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "num_positions": len(self.account["positions"]),
        }
    
    def record_daily_pnl(self):
        """Ghi lại PnL trong ngày"""
        portfolio_value = self.get_portfolio_value()
        
        today = datetime.now().date().isoformat()
        
        daily_record = {
            "date": today,
            "equity": portfolio_value["total_value"],
            "pnl": portfolio_value["total_pnl"],
            "return_pct": portfolio_value["total_return_pct"],
            "cash": portfolio_value["cash"],
            "positions_value": portfolio_value["positions_value"],
        }
        
        # Check if already recorded today
        existing = None
        for record in self.account["daily_pnl"]:
            if record.get("date") == today:
                existing = record
                break
        
        if existing:
            idx = self.account["daily_pnl"].index(existing)
            self.account["daily_pnl"][idx] = daily_record
        else:
            self.account["daily_pnl"].append(daily_record)
        
        self.save_account()
    
    def get_trade_history(self, symbol: Optional[str] = None) -> List[Dict]:
        """Lấy lịch sử giao dịch"""
        trades = self.account.get("trades", [])
        
        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol.upper()]
        
        return sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê trading"""
        trades = self.account.get("trades", [])
        
        buy_trades = [t for t in trades if t.get("action") == "BUY"]
        sell_trades = [t for t in trades if t.get("action") == "SELL"]
        
        total_commission = sum(
            (t.get("price", 0) * t.get("shares", 0) * self.commission_rate)
            for t in trades
        )
        
        portfolio_value = self.get_portfolio_value()
        
        return {
            "total_trades": len(trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total_commission": total_commission,
            "current_portfolio_value": portfolio_value["total_value"],
            "current_pnl": portfolio_value["total_pnl"],
            "current_return_pct": portfolio_value["total_return_pct"],
            "num_positions": portfolio_value["num_positions"],
        }
    
    def format_account_summary(self) -> str:
        """Format tóm tắt tài khoản"""
        stats = self.get_statistics()
        portfolio_value = self.get_portfolio_value()
        
        lines = [
            "📊 *Paper Trading Account:*\n",
            f"💰 Vốn ban đầu: {self.initial_capital:,.0f} VNĐ",
            f"💵 Tiền mặt: {portfolio_value['cash']:,.0f} VNĐ",
            f"📈 Giá trị vị thế: {portfolio_value['positions_value']:,.0f} VNĐ",
            f"💼 Tổng giá trị: {portfolio_value['total_value']:,.0f} VNĐ",
            f"{'📈' if portfolio_value['total_pnl'] >= 0 else '📉'} P&L: {portfolio_value['total_pnl']:+,.0f} VNĐ ({portfolio_value['total_return_pct']:+.2f}%)",
            f"📊 Số vị thế: {portfolio_value['num_positions']}",
            f"🔄 Tổng giao dịch: {stats['total_trades']}",
            f"💸 Tổng phí: {stats['total_commission']:,.0f} VNĐ",
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

