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
    order_type: str = "MARKET"  # MARKET or LIMIT
    limit_price: Optional[float] = None  # For LIMIT orders


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
                "pending_orders": [],  # NEW: List of pending limit orders
                "daily_pnl": [],  # [{date, pnl, equity}]
                "created_at": datetime.now().isoformat(),
            }
            self.save_account()
        else:
            try:
                with open(self.account_file, "r", encoding="utf-8") as f:
                    self.account = json.load(f)
                    # Ensure pending_orders exists
                    if "pending_orders" not in self.account:
                        self.account["pending_orders"] = []
            except Exception:
                self.account = {
                    "initial_capital": self.initial_capital,
                    "cash": self.initial_capital,
                    "positions": {},
                    "trades": [],
                    "pending_orders": [],
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
        is_limit_order: bool = False,
        limit_price: Optional[float] = None,
        metadata: Optional[Dict] = None,  # NEW: Trade metadata for tracking
    ) -> Tuple[bool, str, Optional[PaperTrade]]:
        """
        Thực thi lệnh mua (Market hoặc Limit order)

        Args:
            is_limit_order: True nếu là limit order
            limit_price: Giá limit nếu is_limit_order = True

        Returns:
            (success, message, trade)
        """
        # ENHANCEMENT: Handle limit orders
        if is_limit_order and limit_price is not None:
            return self._create_limit_order(
                symbol=symbol,
                shares=shares,
                limit_price=limit_price,
                signal_confidence=signal_confidence,
                signal_reason=signal_reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        # Market order: Apply slippage
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
            order_type="MARKET",
            limit_price=None,
        )

        # Use PortfolioManager to add the position to the database
        try:
            # Merge default metadata with provided metadata
            trade_metadata = {
                "signal_confidence": signal_confidence,
                "signal_reason": signal_reason,
            }
            if metadata:
                trade_metadata.update(metadata)

            self.portfolio_manager.add_position(
                symbol=symbol.upper(),
                shares=shares,
                entry_price=execution_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=trade_metadata,
            )
        except Exception:
            return False, f"Lỗi DB khi thêm vị thế {symbol}", None

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
            "equity": portfolio_value.get("total_value", 0) + current_cash,
            "pnl": portfolio_value.get("pnl", 0),
            "return_pct": portfolio_value.get("pnl_percent", 0),
            "cash": current_cash,
            "positions_value": portfolio_value.get("total_value", 0),
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
            (t.get("price", 0) * t.get("shares", 0) * self.commission_rate) for t in trades
        )

        portfolio_value = self.get_portfolio_value()
        current_cash = self.account.get("cash", 0)

        return {
            "total_trades": len(trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total_commission": total_commission,
            "current_portfolio_value": portfolio_value.get("total_value", 0) + current_cash,
            "current_pnl": portfolio_value.get("pnl", 0),
            "current_return_pct": portfolio_value.get("pnl_percent", 0),
            "num_positions": portfolio_value.get("num_positions", 0),
        }

    def format_account_summary(self) -> str:
        """Format tóm tắt tài khoản"""
        if not self.portfolio_manager:
            return "Portfolio Manager không khả dụng."

        stats = self.get_statistics()
        portfolio_value = self.get_portfolio_value()
        current_cash = self.account.get("cash", 0)
        total_value = portfolio_value.get("total_value", 0) + current_cash

        lines = [
            "📊 *Paper Trading Account (DB-backed):*\n",
            f"💰 Vốn ban đầu: {self.initial_capital:,.0f} VNĐ",
            f"💵 Tiền mặt (ước tính): {current_cash:,.0f} VNĐ",
            f"📈 Giá trị vị thế: {portfolio_value.get('total_value', 0):,.0f} VNĐ",
            f"💼 Tổng giá trị: {total_value:,.0f} VNĐ",
            f"{'📈' if portfolio_value.get('pnl', 0) >= 0 else '📉'} P&L: "
            f"{portfolio_value.get('pnl', 0):+,.0f} VNĐ "
            f"({portfolio_value.get('pnl_percent', 0):+.2f}%)",
            f"📊 Số vị thế: {portfolio_value.get('num_positions', 0)}",
            f"🔄 Tổng giao dịch: {stats.get('total_trades', 0)}",
            f"💸 Tổng phí (ước tính): {stats.get('total_commission', 0):,.0f} VNĐ",
        ]

        # Show pending limit orders if any
        pending_orders = self.account.get("pending_orders", [])
        if pending_orders:
            lines.append(f"\n⏳ Đang chờ {len(pending_orders)} limit order(s):")
            for order in pending_orders[:5]:  # Show max 5
                lines.append(
                    f"  • {order['symbol']}: {order['shares']} CP @ {order.get('limit_price', order['price']):,.0f} VNĐ"
                )

        return "\n".join(lines)

    def _create_limit_order(
        self,
        symbol: str,
        shares: int,
        limit_price: float,
        signal_confidence: Optional[float] = None,
        signal_reason: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[PaperTrade]]:
        """
        NEW: Tạo limit order (pending order)

        Limit order sẽ được check và fill khi giá đạt limit_price
        """
        # Check if enough cash for limit order
        total_cost = limit_price * shares * (1 + self.commission_rate)
        if total_cost > self.account["cash"]:
            return (
                False,
                f"Không đủ tiền cho limit order. Cần {total_cost:,.0f} VNĐ, có {self.account['cash']:,.0f} VNĐ",
                None,
            )

        # Create pending limit order
        limit_order = PaperTrade(
            symbol=symbol.upper(),
            action="BUY",
            shares=shares,
            price=limit_price,
            timestamp=datetime.now().isoformat(),
            signal_confidence=signal_confidence,
            signal_reason=signal_reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="PENDING",
            order_type="LIMIT",
            limit_price=limit_price,
        )

        # Add to pending orders
        self.account["pending_orders"].append(asdict(limit_order))
        self.save_account()

        return (
            True,
            f"⏳ Limit order đặt: {shares} CP {symbol} @ {limit_price:,.0f} VNĐ (chờ fill)",
            limit_order,
        )

    def check_limit_orders(
        self,
        current_prices: Dict[str, float],
        timeout_hours: int = 24,
        fallback_to_market: bool = True,
    ) -> List[Tuple[bool, str, PaperTrade]]:
        """
        ENHANCED: Check và fill limit orders với timeout và market order fallback

        Args:
            current_prices: Dict mapping symbol -> current_price
            timeout_hours: Hours after which limit order expires (default: 24h)
            fallback_to_market: Convert to market order on timeout (default: True)

        Returns:
            List of (filled, message, trade) tuples
        """
        filled_orders = []
        remaining_orders = []
        current_time = datetime.now()

        for order_dict in self.account.get("pending_orders", []):
            symbol = order_dict["symbol"]
            limit_price = order_dict.get("limit_price", order_dict["price"])

            # Check for timeout
            order_time = datetime.fromisoformat(order_dict["timestamp"])
            time_elapsed = (current_time - order_time).total_seconds() / 3600  # hours

            if time_elapsed > timeout_hours:
                # Order timed out
                if fallback_to_market and symbol in current_prices:
                    # Convert to market order
                    logger.info(
                        f"⏰ Limit order {symbol} timed out after {time_elapsed:.1f}h, "
                        "converting to market order"
                    )
                    # Will be processed as market order below
                    order_dict["timed_out"] = True
                    order_dict["original_limit_price"] = limit_price
                    # Use current price for market order execution
                else:
                    # Cancel expired order
                    logger.info(
                        f"⏰ Limit order {symbol} expired after {time_elapsed:.1f}h, cancelling"
                    )
                    cancelled_order = PaperTrade(**order_dict)
                    cancelled_order.status = "EXPIRED"
                    filled_orders.append(
                        (
                            False,
                            f"⏰ Limit order {symbol} expired after {time_elapsed:.1f}h",
                            cancelled_order,
                        )
                    )
                    continue

            if symbol not in current_prices:
                # No price data - keep order pending (if not timed out)
                if time_elapsed <= timeout_hours:
                    remaining_orders.append(order_dict)
                continue

            current_price = current_prices[symbol]

            # Check if limit order should be filled OR timed out with fallback
            timed_out_with_fallback = order_dict.get("timed_out", False)
            should_fill = order_dict["action"] == "BUY" and (
                current_price <= limit_price or timed_out_with_fallback
            )

            if should_fill:
                # Fill the order
                filled_order = PaperTrade(**order_dict)
                filled_order.status = "FILLED"

                # Calculate execution price (with slippage)
                # If timed out with fallback, use current market price
                if timed_out_with_fallback:
                    execution_price = current_price * (1 + self.slippage_pct)
                    filled_order.order_type = "MARKET (from LIMIT timeout)"
                else:
                    # Normal limit order fill
                    execution_price = current_price * (1 + self.slippage_pct)

                filled_order.price = execution_price

                # Calculate cost
                gross_cost = execution_price * filled_order.shares
                commission = gross_cost * self.commission_rate
                total_cost = gross_cost + commission

                # Check if still enough cash
                if total_cost > self.account["cash"]:
                    # Not enough cash - cancel order
                    filled_order.status = "CANCELLED"
                    filled_orders.append(
                        (
                            False,
                            f"❌ Limit order {symbol} cancelled - không đủ tiền",
                            filled_order,
                        )
                    )
                    continue

                # Deduct cash
                self.account["cash"] -= total_cost

                # Add position to portfolio
                try:
                    self.portfolio_manager.add_position(
                        symbol=symbol.upper(),
                        shares=filled_order.shares,
                        entry_price=execution_price,
                        stop_loss=filled_order.stop_loss,
                        take_profit=filled_order.take_profit,
                        metadata={
                            "signal_confidence": filled_order.signal_confidence,
                            "signal_reason": filled_order.signal_reason,
                            "order_type": "LIMIT",
                        },
                    )
                except Exception:
                    # Failed to add position - refund cash and cancel
                    self.account["cash"] += total_cost
                    filled_order.status = "CANCELLED"
                    filled_orders.append(
                        (
                            False,
                            f"❌ Limit order {symbol} cancelled - lỗi DB",
                            filled_order,
                        )
                    )
                    continue

                # Record filled trade
                self.account["trades"].append(asdict(filled_order))

                # Generate message based on fill type
                if timed_out_with_fallback:
                    message = (
                        f"✅ Market order (timeout fallback): {filled_order.shares} CP {symbol} "
                        f"@ {execution_price:,.0f} VNĐ (limit was {limit_price:,.0f})"
                    )
                else:
                    message = (
                        f"✅ Limit order filled: {filled_order.shares} CP {symbol} "
                        f"@ {execution_price:,.0f} VNĐ"
                    )

                filled_orders.append((True, message, filled_order))
            else:
                # Order not yet fillable - keep pending (if not timed out)
                if not timed_out_with_fallback:
                    remaining_orders.append(order_dict)

        # Update pending orders
        self.account["pending_orders"] = remaining_orders
        self.save_account()

        return filled_orders


# Global instance
_paper_account = None


def get_paper_account(initial_capital: float = 100_000_000) -> PaperTradingAccount:
    """Get or create paper trading account"""
    global _paper_account
    if _paper_account is None:
        _paper_account = PaperTradingAccount(initial_capital=initial_capital)
    return _paper_account
