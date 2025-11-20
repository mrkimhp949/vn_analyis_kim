"""
Portfolio Manager - Quản lý portfolio với SQLite
Thay thế JSON files bằng database
Thread-safe với locking mechanism
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from threading import RLock
from typing import Dict, Optional

from src.data.database import get_db
from src.config.trading_config import get_config

from src.monitoring.performance import get_performance_monitor

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Quản lý portfolio với SQLite database

    Features:
    - Lưu positions vào database thay vì JSON
    - Track performance metrics
    - Portfolio history
    """

    def __init__(self):
        self.db = get_db()
        self.monitor = get_performance_monitor()
        self.config = get_config()

        # Thread safety
        self._lock = RLock()  # Reentrant lock for nested calls

        logger.info("✅ Portfolio Manager initialized with thread safety")

    @contextmanager
    def _transaction(self):
        """Context manager for thread-safe transactions"""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def get_positions(self) -> Dict:
        """Get all active positions (thread-safe)"""
        with self._lock:
            return self.db.get_positions()

    def add_position(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Add new position or average up an existing one (thread-safe).
        Tự động kiểm tra và xử lý mua mới hoặc trung bình giá.
        """
        from src.config.exceptions import PortfolioError

        # Validate inputs
        if not symbol or not isinstance(symbol, str):
            raise PortfolioError("Symbol must be a non-empty string", context={"symbol": symbol})
        if not isinstance(shares, int) or shares <= 0:
            raise PortfolioError("Shares must be a positive integer", context={"shares": shares})
        if not isinstance(entry_price, (int, float)) or entry_price <= 0:
            raise PortfolioError(
                "Entry price must be a positive number",
                context={"entry_price": entry_price},
            )

        # Thread-safe transaction
        with self._transaction():
            existing_positions = self.db.get_positions()

            if symbol in existing_positions:
                # --- LOGIC TRUNG BÌNH GIÁ (DCA) ---
                self._average_up_position(
                    symbol=symbol,
                    existing_pos=existing_positions[symbol],
                    shares_to_add=shares,
                    price_to_add=entry_price,
                    metadata=metadata,
                )
            else:
                # --- LOGIC THÊM VỊ THẾ MỚI ---
                self._create_new_position(
                    symbol=symbol,
                    shares=shares,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata=metadata,
                )

    def _create_new_position(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        metadata: Optional[Dict],
    ):
        """Helper function to create a completely new position."""
        entry_date = datetime.now().isoformat()
        entry_value = shares * entry_price

        self.db.save_position(
            symbol=symbol,
            shares=shares,
            avg_price=entry_price,
            entry_date=entry_date,
            entry_value=entry_value,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
        )

        self.db.save_trade(
            symbol=symbol,
            action="BUY_NEW",
            shares=shares,
            price=entry_price,
            total_value=entry_value,
            trade_date=entry_date,
            reason="New entry signal",
            metadata=metadata,
        )

        print(f"✅ Added new position: {symbol} - {shares} shares @ {entry_price:,.0f}")

    def _average_up_position(
        self,
        symbol: str,
        existing_pos: Dict,
        shares_to_add: int,
        price_to_add: float,
        metadata: Optional[Dict],
    ):
        """Helper function to average up an existing position."""
        current_shares = existing_pos["shares"]
        current_avg_price = existing_pos["avg_price"]

        # Calculate new average price
        total_shares = current_shares + shares_to_add
        total_value = (current_shares * current_avg_price) + (shares_to_add * price_to_add)
        new_avg_price = total_value / total_shares

        # Log the additional buy
        trade_date = datetime.now().isoformat()
        self.db.save_trade(
            symbol=symbol,
            action="BUY_ADD",
            shares=shares_to_add,
            price=price_to_add,
            total_value=shares_to_add * price_to_add,
            trade_date=trade_date,
            reason="Averaging up",
            metadata=metadata,
        )

        # Update the position with new values
        # Note: Stop loss and take profit might need re-evaluation, but for now we keep them
        updated_metadata = existing_pos.get("metadata", {})
        if metadata:
            updated_metadata.update(metadata)

        self.db.save_position(
            symbol=symbol,
            shares=total_shares,
            avg_price=new_avg_price,
            entry_date=existing_pos["entry_date"],  # Keep original entry date
            entry_value=total_value,  # Update total cost basis
            stop_loss=existing_pos.get("stop_loss"),  # Should be re-evaluated
            take_profit=existing_pos.get("take_profit"),  # Should be re-evaluated
            metadata=updated_metadata,
        )

        print(
            f"✅ Averaged up: {symbol}. Added {shares_to_add} shares. New avg price: {new_avg_price:,.0f}"
        )

    def reduce_position(
        self,
        symbol: str,
        shares_to_sell: int,
        exit_price: float,
        reason: str = "Partial Exit",
    ):
        """Reduce position size (partial sell)."""
        from src.config.exceptions import PortfolioError

        positions = self.db.get_positions()

        if symbol not in positions:
            raise PortfolioError(
                f"Position {symbol} not found for partial sell",
                context={"symbol": symbol},
            )

        pos = positions[symbol]
        current_shares = pos["shares"]

        if not isinstance(shares_to_sell, int) or shares_to_sell <= 0:
            raise PortfolioError(
                "Shares to sell must be a positive integer",
                context={"shares_to_sell": shares_to_sell},
            )

        if shares_to_sell >= current_shares:
            # If selling all or more shares, it's a full closure.
            self.close_position(symbol, exit_price, f"Full exit via reduce: {reason}")
            return

        entry_price = pos["avg_price"]

        # Calculate P&L for the sold portion
        exit_value = shares_to_sell * exit_price
        entry_value_of_sold_part = shares_to_sell * entry_price
        pnl = exit_value - entry_value_of_sold_part
        pnl_percent = (pnl / entry_value_of_sold_part) * 100 if entry_value_of_sold_part > 0 else 0

        # Log the partial sell trade
        self.db.save_trade(
            symbol=symbol,
            action="SELL_PARTIAL",
            shares=shares_to_sell,
            price=exit_price,
            total_value=exit_value,
            trade_date=datetime.now().isoformat(),
            reason=reason,
            metadata={"pnl": pnl, "pnl_percent": pnl_percent},
        )

        # Update the existing position
        remaining_shares = current_shares - shares_to_sell
        remaining_value = remaining_shares * entry_price

        self.db.save_position(
            symbol=symbol,
            shares=remaining_shares,
            avg_price=entry_price,
            entry_date=pos["entry_date"],
            entry_value=remaining_value,  # Update entry value
            stop_loss=pos.get("stop_loss"),
            take_profit=pos.get("take_profit"),
            metadata=pos.get("metadata", {}),
        )

        print(
            f"✅ Reduced position: {symbol} - Sold {shares_to_sell} shares. Remaining: {remaining_shares}"
        )

    def handle_exit(self, symbol: str, exit_price: float, exit_type: str, reason: str):
        """
        Dispatches to the correct exit method based on exit_type.
        exit_type can be 'FULL', 'PARTIAL_50%', 'PARTIAL_30%', etc.
        """
        positions = self.db.get_positions()
        if symbol not in positions:
            print(f"⚠️ Cannot handle exit for non-existent position {symbol}")
            return

        if exit_type == "FULL":
            self.close_position(symbol, exit_price, reason)
        else:
            # Handle partial exits
            try:
                # e.g., 'PARTIAL_50%' -> 50
                percentage_str = exit_type.replace("PARTIAL_", "").replace("%", "")
                percentage = float(percentage_str) / 100.0

                current_shares = positions[symbol]["shares"]
                shares_to_sell = int(current_shares * percentage)

                # Ensure we sell at least 1 share and in lots of 100 if possible
                if shares_to_sell > 0:
                    shares_to_sell = max(
                        (shares_to_sell // 100) * 100,
                        100 if current_shares > 100 else shares_to_sell,
                    )
                    self.reduce_position(symbol, shares_to_sell, exit_price, reason)
                else:
                    print(f"⚠️ Calculated 0 shares to sell for {symbol} with type {exit_type}")

            except (ValueError, TypeError):
                print("❌ Invalid exit_type format: {exit_type}.")

    def close_position(self, symbol: str, exit_price: float, reason: str = "Exit signal"):
        """Close a position entirely."""
        positions = self.db.get_positions()

        if symbol not in positions:
            print("⚠️ Position {symbol} not found to close.")
            return

        pos = positions[symbol]
        shares = pos["shares"]
        entry_price = pos["avg_price"]
        entry_date = pos["entry_date"]

        # Calculate P&L
        exit_value = shares * exit_price
        entry_value = shares * entry_price
        pnl = exit_value - entry_value
        pnl_percent = (pnl / entry_value) * 100 if entry_value > 0 else 0

        # Track in performance monitor
        self.monitor.track_trade(
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares,
            entry_date=entry_date,
            exit_date=datetime.now().isoformat(),
        )

        # Log trade
        self.db.save_trade(
            symbol=symbol,
            action="SELL_FULL",
            shares=shares,
            price=exit_price,
            total_value=exit_value,
            trade_date=datetime.now().isoformat(),
            reason=reason,
            metadata={"pnl": pnl, "pnl_percent": pnl_percent},
        )

        # Delete position
        self.db.delete_position(symbol)

        print("✅ Closed position: {symbol} - P&L: {pnl:+,.0f} ({pnl_percent:+.1f}%)")

    def update_position_price(self, symbol: str, current_price: float):
        """Update current price for position"""
        positions = self.db.get_positions()

        if symbol not in positions:
            return

        pos = positions[symbol]
        metadata = pos.get("metadata", {})
        metadata["last_price"] = current_price
        metadata["last_updated"] = datetime.now().isoformat()

        self.db.save_position(
            symbol=symbol,
            shares=pos["shares"],
            avg_price=pos["avg_price"],
            entry_date=pos["entry_date"],
            entry_value=pos["entry_value"],
            stop_loss=pos.get("stop_loss"),
            take_profit=pos.get("take_profit"),
            metadata=metadata,
        )

    def refresh_all_prices(self, lookback: int = 5, force_refresh: bool = True) -> Dict[str, float]:
        """
        Load latest close price for each position and update metadata.

        Args:
            lookback: Number of days to look back
            force_refresh: If True, bypass cache and fetch fresh data from API (default: True)

        Returns:
            dict mapping symbol -> updated price (or existing avg_price if failed).
        """
        positions = self.db.get_positions()
        updated = {}
        if not positions:
            return updated
        try:
            from src.data.loader import load_data
        except Exception:
            # Data loader unavailable; skip refresh
            return updated

        logger.info(
            f"🔄 Refreshing prices for {len(positions)} positions "
            f"(force_refresh={force_refresh})..."
        )

        for symbol, pos in positions.items():
            try:
                # CRITICAL FIX: Use use_cache=False when force_refresh=True
                # This ensures we always get the LATEST price from API, not stale cache
                use_cache = not force_refresh
                df = load_data(
                    symbol,
                    lookback=lookback,
                    use_cache=use_cache,
                    required_bars=1,
                )
                if df is not None and not df.empty:
                    latest = float(df.iloc[-1]["close"])
                    self.update_position_price(symbol, latest)
                    updated[symbol] = latest
                    logger.debug(f"  ✅ {symbol}: {latest:,.0f} VNĐ (fresh from API)")
                else:
                    updated[symbol] = pos["avg_price"]
                    logger.warning(
                        f"  ⚠️ {symbol}: No data, using avg price {pos['avg_price']:,.0f}"
                    )
            except Exception as e:
                # Keep existing price on failure
                updated[symbol] = pos["avg_price"]
                logger.warning(f"  ❌ {symbol}: Error - {e}, using avg price")

        logger.info(f"✅ Price refresh completed: {len(updated)}/{len(positions)} updated")
        return updated

    def get_portfolio_value(self) -> Dict:
        """Calculate current portfolio value"""
        positions = self.db.get_positions()

        total_value = 0
        total_cost = 0

        for symbol, pos in positions.items():
            shares = pos["shares"]
            entry_price = pos["avg_price"]
            current_price = pos.get("metadata", {}).get("last_price", entry_price)

            total_cost += shares * entry_price
            total_value += shares * current_price

        pnl = total_value - total_cost
        pnl_percent = (pnl / total_cost * 100) if total_cost > 0 else 0

        return {
            "total_value": total_value,
            "total_cost": total_cost,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "num_positions": len(positions),
        }

    def get_daily_pnl_pct(self) -> float:
        """
        Tính toán P&L trong ngày dưới dạng phần trăm của tổng vốn.
        Hữu ích cho Circuit Breaker.
        """
        # Lấy snapshot gần nhất từ DB (thường là cuối ngày hôm qua)
        last_snapshot = self.db.get_last_portfolio_snapshot()
        if not last_snapshot:
            return 0.0  # Không có snapshot, không thể tính PNL trong ngày

        # Lấy giá trị hiện tại của portfolio
        current_portfolio_value = self.get_portfolio_value().get("total_value", 0.0)

        # Lấy tổng vốn từ config (hoặc có thể lấy từ snapshot nếu muốn)
        total_capital = self.config.trading.total_capital
        if total_capital == 0:
            return 0.0

        # PNL trong ngày = (Giá trị hiện tại - Giá trị cuối ngày hôm qua)
        daily_pnl = current_portfolio_value - last_snapshot["total_value"]

        # PNL % so với tổng vốn
        daily_pnl_pct = daily_pnl / total_capital

        return daily_pnl_pct

    def save_portfolio_snapshot(self):
        """Save current portfolio snapshot"""
        portfolio = self.get_portfolio_value()

        self.db.save_portfolio_snapshot(
            date=datetime.now().isoformat(),
            total_value=portfolio["total_value"],
            total_cost=portfolio["total_cost"],
            pnl=portfolio["pnl"],
            pnl_percent=portfolio["pnl_percent"],
            num_positions=portfolio["num_positions"],
        )

        print("📸 Saved portfolio snapshot: {portfolio['total_value']:,.0f} VNĐ")

    def get_rebalancing_suggestions(self, target_position_size: float = 0.10) -> Dict:
        """
        ENHANCEMENT: Get rebalancing suggestions for portfolio

        Args:
            target_position_size: Target size for each position (e.g., 0.10 = 10%)

        Returns:
            Dict with rebalancing recommendations
        """
        positions = self.db.get_positions()
        if not positions:
            return {"needs_rebalancing": False, "suggestions": []}

        portfolio = self.get_portfolio_value()
        total_value = portfolio["total_value"]

        if total_value == 0:
            return {"needs_rebalancing": False, "suggestions": []}

        suggestions = []
        needs_rebalancing = False

        for symbol, pos in positions.items():
            shares = pos["shares"]
            entry_price = pos["avg_price"]
            current_price = pos.get("metadata", {}).get("last_price", entry_price)

            position_value = shares * current_price
            position_weight = position_value / total_value

            # Check if position is too large or too small
            deviation = abs(position_weight - target_position_size)

            if position_weight > target_position_size * 1.5:
                # Overweight - suggest reducing
                target_value = total_value * target_position_size
                excess_value = position_value - target_value
                shares_to_sell = int(excess_value / current_price)

                suggestions.append(
                    {
                        "symbol": symbol,
                        "action": "REDUCE",
                        "current_weight": position_weight,
                        "target_weight": target_position_size,
                        "shares_to_sell": shares_to_sell,
                        "reason": f"Overweight: {position_weight:.1%} vs target {target_position_size:.1%}",
                    }
                )
                needs_rebalancing = True

            elif position_weight < target_position_size * 0.5 and position_weight > 0.02:
                # Underweight (but not too small) - suggest increasing
                target_value = total_value * target_position_size
                deficit_value = target_value - position_value
                shares_to_buy = int(deficit_value / current_price)

                suggestions.append(
                    {
                        "symbol": symbol,
                        "action": "INCREASE",
                        "current_weight": position_weight,
                        "target_weight": target_position_size,
                        "shares_to_buy": shares_to_buy,
                        "reason": f"Underweight: {position_weight:.1%} vs target {target_position_size:.1%}",
                    }
                )
                needs_rebalancing = True

            elif position_weight < 0.02:
                # Too small - suggest closing
                suggestions.append(
                    {
                        "symbol": symbol,
                        "action": "CLOSE",
                        "current_weight": position_weight,
                        "reason": f"Position too small: {position_weight:.1%}",
                    }
                )
                needs_rebalancing = True

        return {
            "needs_rebalancing": needs_rebalancing,
            "num_positions": len(positions),
            "total_value": total_value,
            "suggestions": suggestions,
        }

    def get_detailed_analysis(self) -> str:
        """Get detailed portfolio analysis"""
        # CRITICAL: Force refresh prices from API (bypass cache) to get REAL-TIME prices
        self.refresh_all_prices(lookback=10, force_refresh=True)

        # Re-fetch positions after price update
        positions = self.db.get_positions()
        portfolio = self.get_portfolio_value()
        metrics = self.monitor.get_metrics()

        lines = []
        lines.append("📊 *PORTFOLIO ANALYSIS*")
        lines.append("=" * 40)

        # Portfolio summary
        lines.append(f"\n💰 *Portfolio Value:* {portfolio['total_value']:,.0f} VNĐ")
        lines.append(f"💵 *Total Cost:* {portfolio['total_cost']:,.0f} VNĐ")
        lines.append(f"📈 *P&L:* {portfolio['pnl']:+,.0f} VNĐ ({portfolio['pnl_percent']:+.1f}%)")
        lines.append(f"📦 *Positions:* {portfolio['num_positions']}")

        # Individual positions
        if positions:
            lines.append("\n🎯 *POSITIONS:*")
            for symbol, pos in positions.items():
                shares = pos["shares"]
                entry_price = pos["avg_price"]
                current_price = pos.get("metadata", {}).get("last_price", entry_price)

                pos_value = shares * current_price
                pos_cost = shares * entry_price
                pos_pnl = pos_value - pos_cost
                pos_pnl_pct = (pos_pnl / pos_cost * 100) if pos_cost > 0 else 0

                lines.append(f"• {symbol}: {shares:,} CP @ {entry_price:,.0f}")
                lines.append(
                    f"  Current: {current_price:,.0f} | P&L: {pos_pnl:+,.0f} ({pos_pnl_pct:+.1f}%)"
                )

        # Performance metrics
        if metrics["total_trades"] > 0:
            lines.append("\n📊 *PERFORMANCE:*")
            lines.append(f"• Total Trades: {metrics['total_trades']}")
            lines.append(f"• Win Rate: {metrics['win_rate']:.1f}%")
            lines.append(f"• Avg Profit: {metrics['avg_profit']:,.0f} VNĐ")
            lines.append(f"• Avg Loss: {metrics['avg_loss']:,.0f} VNĐ")
            lines.append(f"• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        return "\n".join(lines)

    def export_positions_detail(self, force_refresh: bool = True) -> Dict[str, Dict]:
        """
        Structured positions with current price & PnL for API JSON usage.

        Args:
            force_refresh: If True, refresh prices from API before exporting (default: True)
        """
        # Force refresh prices if requested
        if force_refresh:
            self.refresh_all_prices(lookback=5, force_refresh=True)

        positions = self.db.get_positions()
        detail = {}
        for symbol, pos in positions.items():
            shares = pos["shares"]
            entry_price = pos["avg_price"]
            current_price = pos.get("metadata", {}).get("last_price", entry_price)
            value = shares * current_price
            cost = shares * entry_price
            pnl = value - cost
            pnl_pct = (pnl / cost * 100) if cost > 0 else 0
            detail[symbol] = {
                "shares": shares,
                "avg_price": entry_price,
                "current_price": current_price,
                "value": value,
                "cost": cost,
                "pnl": pnl,
                "pnl_percent": pnl_pct,
                "last_updated": pos.get("metadata", {}).get("last_updated"),
            }
        return detail


# Singleton
_manager = None


def get_portfolio_manager() -> PortfolioManager:
    """Get portfolio manager singleton"""
    global _manager
    if _manager is None:
        _manager = PortfolioManager()
    return _manager


if __name__ == "__main__":
    print("Testing Portfolio Manager...")

    manager = PortfolioManager()

    # Test add position
    manager.add_position("VCB", 100, 60000, stop_loss=57000, take_profit=66000)

    # Test get positions
    positions = manager.get_positions()
    print(f"Positions: {positions}")

    # Test portfolio value
    portfolio = manager.get_portfolio_value()
    print(f"Portfolio: {portfolio}")

    # Test analysis
    analysis = manager.get_detailed_analysis()
    print(analysis)

    print("\n✅ Portfolio Manager test completed!")
