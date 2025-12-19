"""
Portfolio Manager - Quản lý portfolio với SQLite
Thay thế JSON files bằng database
Thread-safe với locking mechanism

IMPROVEMENTS V2:
- Integration with Circuit Breaker (record_trade on close)
- Integration with Per-Symbol Circuit Breaker (record_trade on close)
- Improved DCA stop loss validation
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from threading import RLock
from typing import Dict, Optional

from src.data.database import get_db
from src.config.trading_config import get_config

from src.monitoring.performance import get_performance_monitor
from src.monitoring.signal_performance import get_signal_performance_tracker

# Import circuit breakers
try:
    from src.risk.circuit_breaker import get_circuit_breaker

    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False

try:
    from src.risk.per_symbol_circuit_breaker import get_per_symbol_circuit_breaker

    PER_SYMBOL_CB_AVAILABLE = True
except ImportError:
    PER_SYMBOL_CB_AVAILABLE = False

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
        self.signal_tracker = get_signal_performance_tracker()

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

    def _record_to_circuit_breakers(
        self, symbol: str, pnl: float, is_win: bool, pnl_percent: float
    ):
        """
        CRITICAL FIX #1 & #2: Record trade results to circuit breakers

        This ensures:
        1. Global circuit breaker tracks daily P&L and consecutive losses
        2. Per-symbol circuit breaker tracks symbol-specific performance
        3. Alert when approaching thresholds (v4 improvement)

        Args:
            symbol: Stock symbol
            pnl: Profit/Loss amount in VND
            is_win: True if trade was profitable
            pnl_percent: P&L as percentage
        """
        # Record to global circuit breaker
        if CIRCUIT_BREAKER_AVAILABLE:
            try:
                circuit_breaker = get_circuit_breaker()
                circuit_breaker.record_trade(pnl)

                # IMPROVEMENT v4: Alert when approaching thresholds
                self._check_circuit_breaker_thresholds(circuit_breaker, symbol)

                logger.debug(
                    f"📊 Recorded to circuit breaker: {symbol} PnL={pnl:+,.0f} "
                    f"(consecutive_losses={circuit_breaker.stats.get('consecutive_losses', 0)})"
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to record to circuit breaker: {e}")

        # Record to per-symbol circuit breaker
        if PER_SYMBOL_CB_AVAILABLE:
            try:
                per_symbol_cb = get_per_symbol_circuit_breaker()
                per_symbol_cb.record_trade(symbol, is_win, pnl_percent)
                logger.debug(
                    f"📊 Recorded to per-symbol CB: {symbol} win={is_win} pnl={pnl_percent:+.2f}%"
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to record to per-symbol circuit breaker: {e}")

    def _check_circuit_breaker_thresholds(self, circuit_breaker, symbol: str):
        """
        IMPROVEMENT v4: Check and alert when approaching circuit breaker thresholds.

        Alerts at 70% and 90% of thresholds to give early warning.
        """
        try:
            stats = circuit_breaker.get_daily_stats()
            consecutive_losses = circuit_breaker.stats.get("consecutive_losses", 0)
            max_consecutive = circuit_breaker.max_consecutive_losses
            max_trades = circuit_breaker.max_trades_per_day

            # Alert 1: Consecutive losses approaching limit
            if consecutive_losses >= max_consecutive - 1:  # 1 away from limit
                logger.warning(
                    f"🚨 ALERT: {consecutive_losses}/{max_consecutive} consecutive losses! "
                    f"Circuit breaker sẽ kích hoạt sau 1 lệnh thua nữa."
                )
            elif consecutive_losses >= max_consecutive * 0.7:  # 70% of limit
                logger.warning(
                    f"⚠️ WARNING: {consecutive_losses}/{max_consecutive} consecutive losses. "
                    f"Cân nhắc giảm position size."
                )

            # Alert 2: Trade count approaching limit
            trades_today = stats.trades_count
            if trades_today >= max_trades - 1:  # 1 away from limit
                logger.warning(
                    f"🚨 ALERT: {trades_today}/{max_trades} trades today! "
                    f"Chỉ còn 1 lệnh trước khi circuit breaker kích hoạt."
                )
            elif trades_today >= max_trades * 0.8:  # 80% of limit
                logger.info(
                    f"📊 INFO: {trades_today}/{max_trades} trades today. "
                    f"Còn {max_trades - trades_today} lệnh."
                )

            # Alert 3: Daily loss approaching limit
            if circuit_breaker.total_capital > 0:
                daily_loss_pct = stats.total_loss / circuit_breaker.total_capital
                max_loss_pct = circuit_breaker.max_loss_per_day_pct

                if daily_loss_pct >= max_loss_pct * 0.9:  # 90% of limit
                    logger.warning(
                        f"🚨 ALERT: Daily loss {daily_loss_pct:.2%} gần ngưỡng {max_loss_pct:.2%}! "
                        f"Circuit breaker sắp kích hoạt."
                    )
                elif daily_loss_pct >= max_loss_pct * 0.7:  # 70% of limit
                    logger.warning(
                        f"⚠️ WARNING: Daily loss {daily_loss_pct:.2%} / {max_loss_pct:.2%}. "
                        f"Cân nhắc dừng trading."
                    )

        except Exception as e:
            logger.debug(f"Circuit breaker threshold check error: {e}")

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

        CRITICAL: stop_loss is REQUIRED for new positions. If None, will use default -7%.
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

        # CRITICAL FIX: Ensure stop_loss is NEVER None
        # If stop_loss not provided, calculate default -7% below entry
        if stop_loss is None or stop_loss <= 0 or stop_loss >= entry_price:
            default_stop_loss = entry_price * 0.93  # -7% default
            logger.warning(
                f"⚠️ Stop loss missing/invalid for {symbol}. "
                f"Using default -7%: {default_stop_loss:,.0f} VND"
            )
            stop_loss = default_stop_loss

        # Validate stop_loss is below entry_price
        if stop_loss >= entry_price:
            raise PortfolioError(
                f"Stop loss ({stop_loss:,.0f}) must be below entry price ({entry_price:,.0f})",
                context={
                    "symbol": symbol,
                    "stop_loss": stop_loss,
                    "entry_price": entry_price,
                    "suggestion": f"Use stop_loss < {entry_price:,.0f}",
                },
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
        """
        Helper function to create a completely new position.

        ENHANCEMENT: Uses database transaction for atomicity.
        Both save_position and save_trade succeed together or fail together.
        """
        entry_date = datetime.now().isoformat()
        entry_value = shares * entry_price

        # CRITICAL: Wrap in database transaction for atomicity
        # If save_trade fails, save_position will be rolled back
        with self.db.transaction() as conn:
            self.db.save_position(
                symbol=symbol,
                shares=shares,
                avg_price=entry_price,
                entry_date=entry_date,
                entry_value=entry_value,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
                conn=conn,
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
                conn=conn,
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
        """
        Helper function to average up an existing position.

        ENHANCEMENT: Uses database transaction for atomicity.
        Both save_trade and save_position succeed together or fail together.
        """
        current_shares = existing_pos["shares"]
        current_avg_price = existing_pos["avg_price"]

        # Calculate new average price
        total_shares = current_shares + shares_to_add
        total_value = (current_shares * current_avg_price) + (shares_to_add * price_to_add)
        new_avg_price = total_value / total_shares

        # CRITICAL FIX: Recalculate stop loss and take profit based on new average price
        # Use default -7% stop loss and 15% take profit from config
        old_stop_loss = existing_pos.get("stop_loss")
        old_take_profit = existing_pos.get("take_profit")

        # CRITICAL FIX #3: Improved DCA stop loss validation (v4)
        # Stop loss thresholds
        default_stop_pct = 0.93  # -7% default
        min_stop_pct = 0.95  # -5% minimum (never closer than this)
        floor_warning_pct = 0.94  # -6% warning zone (close to floor)

        new_stop_loss = new_avg_price * default_stop_pct
        min_safe_stop = new_avg_price * min_stop_pct  # -5% minimum
        floor_warning_level = new_avg_price * floor_warning_pct  # -6%

        # VALIDATION: Ensure stop loss is ALWAYS below new avg price
        # This is critical when DCA-ing into a falling stock
        if old_stop_loss and old_stop_loss >= new_avg_price:
            # Old stop loss is above new avg price - INVALID after DCA
            logger.warning(
                f"⚠️ {symbol}: Old stop loss {old_stop_loss:,.0f} >= new avg price {new_avg_price:,.0f}. "
                f"Recalculating to {new_stop_loss:,.0f}"
            )
            final_stop_loss = new_stop_loss
        elif old_stop_loss and old_stop_loss > min_safe_stop:
            # Old stop loss is too close to new avg price (within 5%)
            # This is dangerous - easy to get stopped out
            logger.warning(
                f"⚠️ {symbol}: Old stop loss {old_stop_loss:,.0f} quá gần avg price {new_avg_price:,.0f} "
                f"(chỉ {((new_avg_price - old_stop_loss) / new_avg_price * 100):.1f}%). "
                f"Điều chỉnh xuống {new_stop_loss:,.0f} (-7%)"
            )
            final_stop_loss = new_stop_loss
        elif old_stop_loss and old_stop_loss > new_stop_loss:
            # Old stop loss is valid and higher - keep it but warn if in danger zone
            final_stop_loss = old_stop_loss
            if old_stop_loss > floor_warning_level:
                logger.warning(
                    f"⚠️ {symbol}: Stop loss {old_stop_loss:,.0f} trong vùng nguy hiểm "
                    f"(gần floor -7%). Cân nhắc hạ xuống {new_stop_loss:,.0f}"
                )
            else:
                logger.info(
                    f"🔒 {symbol}: Giữ stop loss {old_stop_loss:,.0f} "
                    f"(valid và cao hơn {new_stop_loss:,.0f})"
                )
        else:
            final_stop_loss = new_stop_loss
            logger.info(
                f"📊 {symbol}: Cập nhật stop loss {final_stop_loss:,.0f} "
                f"dựa trên avg price mới {new_avg_price:,.0f}"
            )

        # Final validation: stop loss must be at least 5% below entry price
        if final_stop_loss > min_safe_stop:
            final_stop_loss = new_stop_loss
            logger.error(
                f"🚨 {symbol}: Stop loss quá gần! Bắt buộc điều chỉnh xuống {final_stop_loss:,.0f} (-7%)"
            )

        # Additional warning: check if stop loss is near Vietnam floor limit
        floor_price = new_avg_price * 0.93  # Vietnam floor is -7%
        if final_stop_loss <= floor_price * 1.01:  # Within 1% of floor
            logger.warning(
                f"🚨 {symbol}: Stop loss {final_stop_loss:,.0f} RẤT GẦN FLOOR {floor_price:,.0f}! "
                f"Nếu giá chạm sàn, có thể không thoát được lệnh."
            )

        # Calculate new take profit: 15% above new avg price
        default_tp_pct = 1.15  # +15%
        new_take_profit = new_avg_price * default_tp_pct

        # Use new take profit (always update to reflect new avg price)
        final_take_profit = new_take_profit

        # CRITICAL: Wrap in database transaction for atomicity
        trade_date = datetime.now().isoformat()
        with self.db.transaction() as conn:
            # Log the additional buy
            self.db.save_trade(
                symbol=symbol,
                action="BUY_ADD",
                shares=shares_to_add,
                price=price_to_add,
                total_value=shares_to_add * price_to_add,
                trade_date=trade_date,
                reason="Averaging up",
                metadata=metadata,
                conn=conn,
            )

            # Update the position with new values
            updated_metadata = existing_pos.get("metadata", {})
            if metadata:
                updated_metadata.update(metadata)

            self.db.save_position(
                symbol=symbol,
                shares=total_shares,
                avg_price=new_avg_price,
                entry_date=existing_pos["entry_date"],  # Keep original entry date
                entry_value=total_value,  # Update total cost basis
                stop_loss=final_stop_loss,  # UPDATED: Recalculated based on new avg price
                take_profit=final_take_profit,  # UPDATED: Recalculated based on new avg price
                metadata=updated_metadata,
                conn=conn,
            )

        print(
            f"✅ Averaged up: {symbol}. Added {shares_to_add} shares. "
            f"New avg price: {new_avg_price:,.0f}"
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

        # Track partial exit in performance monitor
        self.monitor.track_trade(
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares_to_sell,
            entry_date=pos["entry_date"],
            exit_date=datetime.now().isoformat(),
        )

        # Track signal performance (ML vs Technical)
        metadata = pos.get("metadata", {})
        is_ml_signal = metadata.get("signal_source") == "ml"
        self.signal_tracker.track_trade(
            is_ml_signal=is_ml_signal,
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares_to_sell,
        )

        # CRITICAL FIX #1 & #2: Record trade to circuit breakers
        is_win = pnl > 0
        self._record_to_circuit_breakers(symbol, pnl, is_win, pnl_percent)

        # CRITICAL: Wrap in database transaction for atomicity
        # Both save_trade and save_position succeed together or fail together
        with self.db.transaction() as conn:
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
                conn=conn,
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
                conn=conn,
            )

        print(
            f"✅ Reduced position: {symbol} - Sold {shares_to_sell} shares. "
            f"Remaining: {remaining_shares}"
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
                print(f"❌ Invalid exit_type format: {exit_type}.")

    def close_position(self, symbol: str, exit_price: float, reason: str = "Exit signal"):
        """Close a position entirely."""
        # CRITICAL FIX: Validate exit_price before processing
        if exit_price is None or exit_price <= 0:
            print(f"❌ Invalid exit_price: {exit_price}. Cannot close position for {symbol}.")
            return

        positions = self.db.get_positions()

        if symbol not in positions:
            print(f"⚠️ Position {symbol} not found to close.")
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

        # Track signal performance (ML vs Technical)
        metadata = pos.get("metadata", {})
        is_ml_signal = metadata.get("signal_source") == "ml"
        self.signal_tracker.track_trade(
            is_ml_signal=is_ml_signal,
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares,
        )

        # CRITICAL FIX #1 & #2: Record trade to circuit breakers
        is_win = pnl > 0
        self._record_to_circuit_breakers(symbol, pnl, is_win, pnl_percent)

        # CRITICAL: Wrap in database transaction for atomicity
        # Both save_trade and delete_position succeed together or fail together
        with self.db.transaction() as conn:
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
                conn=conn,
            )

            # Delete position
            self.db.delete_position(symbol, conn=conn)

        logger.info(f"✅ Closed position: {symbol} - P&L: {pnl:+,.0f} ({pnl_percent:+.1f}%)")

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

        print(f"📸 Saved portfolio snapshot: {portfolio['total_value']:,.0f} VNĐ")

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
                        "reason": (
                            f"Overweight: {position_weight:.1%} "
                            f"vs target {target_position_size:.1%}"
                        ),
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
                        "reason": (
                            f"Underweight: {position_weight:.1%} "
                            f"vs target {target_position_size:.1%}"
                        ),
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

    def get_sector_exposure(self) -> Dict:
        """
        IMPROVEMENT #8: Track sector exposure của portfolio

        Tính toán phân bổ theo ngành để tránh over-concentration

        Returns:
            Dict with sector exposure analysis
        """
        positions = self.db.get_positions()
        if not positions:
            return {
                "sectors": {},
                "max_sector_exposure": 0.0,
                "is_concentrated": False,
                "warnings": [],
            }

        try:
            # Get sector mapping for symbols
            sector_mapping = self._get_sector_mapping(list(positions.keys()))

            portfolio = self.get_portfolio_value()
            total_value = portfolio["total_value"]

            if total_value == 0:
                return {
                    "sectors": {},
                    "max_sector_exposure": 0.0,
                    "is_concentrated": False,
                    "warnings": [],
                }

            # Calculate sector exposure
            sector_values = {}
            for symbol, pos in positions.items():
                shares = pos["shares"]
                current_price = pos.get("metadata", {}).get("last_price", pos["avg_price"])
                position_value = shares * current_price

                sector = sector_mapping.get(symbol, "Unknown")
                sector_values[sector] = sector_values.get(sector, 0) + position_value

            # Calculate percentages
            sector_exposure = {}
            for sector, value in sector_values.items():
                pct = (value / total_value) * 100 if total_value > 0 else 0
                sector_exposure[sector] = {
                    "value": value,
                    "percentage": pct,
                    "symbols": [
                        s
                        for s, p in positions.items()
                        if sector_mapping.get(s, "Unknown") == sector
                    ],
                }

            # Find max exposure
            max_exposure = max((s["percentage"] for s in sector_exposure.values()), default=0)

            # Check for concentration (>40% in one sector)
            is_concentrated = max_exposure > 40
            warnings = []

            if is_concentrated:
                concentrated_sectors = [
                    sector for sector, data in sector_exposure.items() if data["percentage"] > 40
                ]
                warnings.append(
                    f"⚠️ Over-concentrated in: {', '.join(concentrated_sectors)} "
                    f"(>{max_exposure:.1f}%)"
                )

            # Warn if >30% in any sector
            for sector, data in sector_exposure.items():
                if 30 < data["percentage"] <= 40:
                    warnings.append(f"⚠️ High exposure to {sector}: {data['percentage']:.1f}%")

            return {
                "sectors": sector_exposure,
                "max_sector_exposure": max_exposure,
                "is_concentrated": is_concentrated,
                "warnings": warnings,
                "total_value": total_value,
            }

        except Exception as e:
            logger.warning(f"⚠️ Error calculating sector exposure: {e}")
            return {
                "sectors": {},
                "max_sector_exposure": 0.0,
                "is_concentrated": False,
                "warnings": [f"Error: {e}"],
            }

    def _get_sector_mapping(self, symbols: list) -> Dict[str, str]:
        """
        Get sector mapping for symbols

        Returns:
            Dict mapping symbol -> sector name
        """
        # Default sector mapping for common VN stocks
        # In production, this should come from a database or API
        SECTOR_MAPPING = {
            # Banking
            "VCB": "Banking",
            "BID": "Banking",
            "CTG": "Banking",
            "TCB": "Banking",
            "MBB": "Banking",
            "ACB": "Banking",
            "VPB": "Banking",
            "HDB": "Banking",
            "TPB": "Banking",
            "STB": "Banking",
            "SHB": "Banking",
            "EIB": "Banking",
            # Real Estate
            "VHM": "Real Estate",
            "VIC": "Real Estate",
            "NVL": "Real Estate",
            "KDH": "Real Estate",
            "DXG": "Real Estate",
            "PDR": "Real Estate",
            "NLG": "Real Estate",
            "DIG": "Real Estate",
            "CEO": "Real Estate",
            # Steel & Materials
            "HPG": "Steel",
            "HSG": "Steel",
            "NKG": "Steel",
            "TLH": "Steel",
            "POM": "Steel",
            # Consumer
            "VNM": "Consumer",
            "MSN": "Consumer",
            "SAB": "Consumer",
            "MWG": "Consumer",
            "PNJ": "Consumer",
            "FRT": "Consumer",
            # Technology
            "FPT": "Technology",
            "CMG": "Technology",
            # Oil & Gas
            "GAS": "Oil & Gas",
            "PLX": "Oil & Gas",
            "PVD": "Oil & Gas",
            "PVS": "Oil & Gas",
            "BSR": "Oil & Gas",
            # Utilities
            "POW": "Utilities",
            "REE": "Utilities",
            "PC1": "Utilities",
            "GEG": "Utilities",
            "NT2": "Utilities",
            # Aviation
            "HVN": "Aviation",
            "VJC": "Aviation",
            # Securities
            "SSI": "Securities",
            "VND": "Securities",
            "HCM": "Securities",
            "VCI": "Securities",
            "SHS": "Securities",
            # Insurance
            "BVH": "Insurance",
            "BMI": "Insurance",
        }

        result = {}
        for symbol in symbols:
            result[symbol] = SECTOR_MAPPING.get(symbol, "Other")

        return result

    def check_sector_before_entry(self, symbol: str, max_sector_pct: float = 40.0) -> tuple:
        """
        Check if adding a symbol would exceed sector concentration limit

        Args:
            symbol: Symbol to check
            max_sector_pct: Maximum allowed sector percentage (default 40%)

        Returns:
            (can_add, warning_message)
        """
        sector_exposure = self.get_sector_exposure()
        sector_mapping = self._get_sector_mapping([symbol])
        target_sector = sector_mapping.get(symbol, "Other")

        current_exposure = sector_exposure["sectors"].get(target_sector, {}).get("percentage", 0)

        # Estimate new exposure (rough estimate assuming 10% position)
        estimated_new_exposure = current_exposure + 10  # Assume 10% position

        if estimated_new_exposure > max_sector_pct:
            return (
                False,
                f"Adding {symbol} would increase {target_sector} exposure to ~{estimated_new_exposure:.1f}% "
                f"(limit: {max_sector_pct}%)",
            )

        if current_exposure > max_sector_pct * 0.8:  # Warning at 80% of limit
            return (
                True,
                f"⚠️ {target_sector} sector already at {current_exposure:.1f}% - approaching limit",
            )

        return (True, None)

    def get_detailed_analysis(self) -> str:
        """Get detailed portfolio analysis"""
        # CRITICAL: Force refresh prices from API (bypass cache) to get REAL-TIME prices
        self.refresh_all_prices(lookback=60, force_refresh=True)

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
            self.refresh_all_prices(lookback=60, force_refresh=True)

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
