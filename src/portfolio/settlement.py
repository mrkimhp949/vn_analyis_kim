# -*- coding: utf-8 -*-
"""
T+2.5 Settlement Tracker for Vietnam Stock Market

Vietnam market settlement rules:
- T+2: Stocks available for trading after 2 business days
- T+2.5: Cash available for withdrawal after 2.5 business days
- Buy on T0 → Cash locked until T+2
- Sell on T0 → Cash available on T+2 (trading) or T+2.5 (withdrawal)

This module tracks pending settlements to prevent over-buying.

Author: Trading Bot Team
Version: 1.0.0
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from threading import RLock

logger = logging.getLogger(__name__)


@dataclass
class SettlementRecord:
    """Individual settlement record"""

    trade_date: str
    settlement_date: str
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    amount: float
    status: str = "PENDING"  # PENDING, SETTLED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SettlementTracker:
    """
    Track T+2 settlement for Vietnam market.

    Prevents over-buying by tracking pending settlements and
    calculating available cash for new trades.

    Usage:
        tracker = get_settlement_tracker()

        # Record a buy
        tracker.record_trade("VNM", "BUY", 100, 8_500_000)

        # Check available cash
        available = tracker.get_available_cash(total_cash=100_000_000)
    """

    SETTLEMENT_DAYS = 2  # T+2 for Vietnam
    STATE_FILE = "settlement_state.json"
    INITIAL_MARGIN_RATIO = 0.50  # 50% initial margin for position sizing

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self._records: List[SettlementRecord] = []
        self._lock = RLock()

        # Load persisted state
        self._load_state()

        # Clean up old settled records
        self._cleanup_old_records()

        logger.info(f"✅ SettlementTracker initialized with {len(self._records)} pending records")

    def _load_state(self):
        """Load persisted settlement state"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._records = [SettlementRecord(**r) for r in data.get("records", [])]
                    logger.info(f"📂 Loaded {len(self._records)} settlement records")
            except Exception as e:
                logger.warning(f"Failed to load settlement state: {e}")
                self._records = []

    def _save_state(self):
        """Persist settlement state"""
        try:
            data = {
                "records": [asdict(r) for r in self._records],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save settlement state: {e}")

    def _get_settlement_date(self, trade_date: date) -> date:
        """
        Calculate settlement date (T+2 business days).

        Skips weekends and Vietnam holidays.
        """
        try:
            from src.utils.vietnam_market import get_next_trading_day

            result = get_next_trading_day(trade_date, days_ahead=self.SETTLEMENT_DAYS)
            # Ensure we return a date object, not datetime
            if hasattr(result, "date"):
                return result.date()
            return result
        except ImportError:
            # Fallback: simple T+2 without holiday check
            settlement = trade_date + timedelta(days=self.SETTLEMENT_DAYS)
            # Skip weekends
            while settlement.weekday() >= 5:
                settlement += timedelta(days=1)
            return settlement

    def _cleanup_old_records(self):
        """Remove settled records older than 7 days"""
        with self._lock:
            today = date.today()
            cutoff = today - timedelta(days=7)

            # Mark settled records
            for record in self._records:
                settlement_date = date.fromisoformat(record.settlement_date)
                if settlement_date <= today and record.status == "PENDING":
                    record.status = "SETTLED"

            # Remove old settled records
            original_count = len(self._records)
            self._records = [
                r
                for r in self._records
                if r.status == "PENDING" or date.fromisoformat(r.settlement_date) > cutoff
            ]

            removed = original_count - len(self._records)
            if removed > 0:
                logger.info(f"🧹 Cleaned up {removed} old settlement records")
                self._save_state()

    def _parse_settlement_date(self, settlement_str: str) -> Optional[date]:
        """Parse settlement date string to date object"""
        try:
            if "T" in settlement_str:
                # datetime format: "2025-12-10T00:00:00"
                return datetime.fromisoformat(settlement_str).date()
            else:
                # date format: "2025-12-10"
                return date.fromisoformat(settlement_str)
        except (ValueError, AttributeError):
            return None

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        amount: float,
        trade_date: Optional[date] = None,
    ) -> SettlementRecord:
        """
        Record a trade for settlement tracking.

        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            quantity: Number of shares
            amount: Total trade value in VND
            trade_date: Trade date (default: today)

        Returns:
            SettlementRecord
        """
        with self._lock:
            if trade_date is None:
                trade_date = date.today()

            settlement_date = self._get_settlement_date(trade_date)

            record = SettlementRecord(
                trade_date=trade_date.isoformat(),
                settlement_date=settlement_date.isoformat(),
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                amount=amount,
            )

            self._records.append(record)
            self._save_state()

            logger.info(
                f"📝 Settlement recorded: {side} {symbol} {quantity} @ {amount:,.0f} VND | "
                f"Trade: {trade_date} → Settlement: {settlement_date}"
            )

            return record

    def get_pending_settlements(self) -> Dict[str, float]:
        """
        Get pending settlements grouped by settlement date.

        Returns:
            Dict of {settlement_date: total_amount}
        """
        with self._lock:
            today = date.today()
            pending = {}

            for record in self._records:
                if record.status != "PENDING":
                    continue

                # Parse settlement date
                settlement_date = self._parse_settlement_date(record.settlement_date)
                if settlement_date is None:
                    continue

                if settlement_date > today and record.side == "BUY":
                    # Only BUY orders lock cash
                    pending[record.settlement_date] = (
                        pending.get(record.settlement_date, 0) + record.amount
                    )

            return pending

    def get_pending_buy_amount(self) -> float:
        """Get total pending BUY settlement amount"""
        pending = self.get_pending_settlements()
        return sum(pending.values())

    def get_available_cash(self, total_cash: float) -> Dict[str, float]:
        """
        Calculate available cash for new trades.

        Args:
            total_cash: Total cash balance in account

        Returns:
            Dict with:
            - available_cash: Cash available for new trades
            - pending_settlements: Total pending settlement amount
            - buffer: Safety buffer (10% of pending)
        """
        with self._lock:
            pending = self.get_pending_buy_amount()

            # Add 10% safety buffer
            buffer = pending * 0.10

            available = max(0, total_cash - pending - buffer)

            return {
                "total_cash": total_cash,
                "pending_settlements": pending,
                "buffer": buffer,
                "available_cash": available,
                "utilization_pct": (pending / total_cash * 100) if total_cash > 0 else 0,
            }

    def can_buy(self, amount: float, total_cash: float) -> Tuple[bool, str]:
        """
        Check if can execute a buy order.

        Args:
            amount: Buy order amount
            total_cash: Total cash balance

        Returns:
            (can_buy, reason)
        """
        cash_info = self.get_available_cash(total_cash)
        available = cash_info["available_cash"]

        if amount > available:
            return (
                False,
                f"Insufficient available cash. "
                f"Need: {amount:,.0f}, Available: {available:,.0f} "
                f"(Pending settlements: {cash_info['pending_settlements']:,.0f})",
            )

        return True, "OK"

    def get_settlement_summary(self) -> Dict:
        """Get summary of all settlements"""
        with self._lock:
            today = date.today()

            pending_count = sum(1 for r in self._records if r.status == "PENDING")
            settled_count = sum(1 for r in self._records if r.status == "SETTLED")

            pending_buy = sum(
                r.amount for r in self._records if r.status == "PENDING" and r.side == "BUY"
            )
            pending_sell = sum(
                r.amount for r in self._records if r.status == "PENDING" and r.side == "SELL"
            )

            # Next settlement date
            next_settlement = None
            for record in self._records:
                if record.status == "PENDING":
                    settlement_date = self._parse_settlement_date(record.settlement_date)
                    if settlement_date and settlement_date > today:
                        if next_settlement is None or settlement_date < next_settlement:
                            next_settlement = settlement_date

            return {
                "pending_count": pending_count,
                "settled_count": settled_count,
                "pending_buy_amount": pending_buy,
                "pending_sell_amount": pending_sell,
                "net_pending": pending_buy - pending_sell,
                "next_settlement_date": next_settlement.isoformat() if next_settlement else None,
                "records": [asdict(r) for r in self._records if r.status == "PENDING"],
            }

    def clear_all(self) -> int:
        """Clear all records (for testing)"""
        with self._lock:
            count = len(self._records)
            self._records = []
            self._save_state()
            return count

    # =========================================================================
    # NEW v7.0: CASH FLOW PREDICTION
    # =========================================================================

    def predict_cash_availability(
        self,
        total_cash: float,
        days_ahead: int = 5,
    ) -> Dict[str, Dict]:
        """
        Predict cash availability for each day in the next N days.

        IMPROVED v7.0: Cash flow prediction for better position planning.

        This helps traders plan position sizes by knowing when cash
        will become available from pending settlements.

        Args:
            total_cash: Current total cash balance
            days_ahead: Number of days to predict (default: 5)

        Returns:
            Dict of {date_str: {
                "available_cash": float,
                "settling_today": float,
                "pending_after": float,
                "can_trade_value": float,
            }}
        """
        with self._lock:
            today = date.today()
            predictions = {}

            # Get all pending settlements
            pending_by_date = {}
            for record in self._records:
                if record.status != "PENDING" or record.side != "BUY":
                    continue

                settlement_date = self._parse_settlement_date(record.settlement_date)
                if settlement_date is None:
                    continue

                if settlement_date not in pending_by_date:
                    pending_by_date[settlement_date] = 0
                pending_by_date[settlement_date] += record.amount

            # Calculate for each day
            cumulative_settled = 0
            total_pending = sum(pending_by_date.values())

            for i in range(days_ahead + 1):
                check_date = today + timedelta(days=i)

                # Skip weekends
                if check_date.weekday() >= 5:
                    continue

                # Amount settling on this day
                settling_today = pending_by_date.get(check_date, 0)
                cumulative_settled += settling_today

                # Remaining pending after this day
                pending_after = total_pending - cumulative_settled

                # Available cash = total - pending + buffer
                buffer = pending_after * 0.10
                available = max(0, total_cash - pending_after - buffer)

                # Can trade value (with initial margin)
                can_trade_value = available / self.INITIAL_MARGIN_RATIO

                predictions[check_date.isoformat()] = {
                    "date": check_date.isoformat(),
                    "day_name": check_date.strftime("%A"),
                    "available_cash": available,
                    "settling_today": settling_today,
                    "pending_after": pending_after,
                    "can_trade_value": can_trade_value,
                    "utilization_pct": (
                        (total_cash - available) / total_cash * 100 if total_cash > 0 else 0
                    ),
                }

            return predictions

    def get_optimal_entry_day(
        self,
        total_cash: float,
        required_amount: float,
        max_days_wait: int = 5,
    ) -> Tuple[Optional[str], str]:
        """
        Find the optimal day to enter a position based on cash availability.

        IMPROVED v7.0: Smart entry timing based on settlement schedule.

        Args:
            total_cash: Current total cash balance
            required_amount: Amount needed for the position
            max_days_wait: Maximum days willing to wait

        Returns:
            (optimal_date, reason)
            - (date_str, "OK") if found
            - (None, reason) if not possible within timeframe
        """
        predictions = self.predict_cash_availability(total_cash, max_days_wait)

        for date_str, info in predictions.items():
            if info["available_cash"] >= required_amount:
                if date_str == date.today().isoformat():
                    return (date_str, "Cash available today")
                else:
                    return (
                        date_str,
                        f"Cash available on {info['day_name']} "
                        f"(settling: {info['settling_today']:,.0f} VND)",
                    )

        # Not possible within timeframe
        max_available = max(p["available_cash"] for p in predictions.values())
        return (
            None,
            f"Insufficient cash within {max_days_wait} days. "
            f"Max available: {max_available:,.0f}, Need: {required_amount:,.0f}",
        )

    def get_cash_flow_report(self, total_cash: float) -> str:
        """
        Generate a formatted cash flow report.

        Args:
            total_cash: Current total cash balance

        Returns:
            Formatted string report
        """
        predictions = self.predict_cash_availability(total_cash, days_ahead=5)
        summary = self.get_settlement_summary()

        lines = [
            "=" * 60,
            "💰 T+2 CASH FLOW PREDICTION REPORT",
            "=" * 60,
            f"Total Cash Balance: {total_cash:>15,.0f} VND",
            f"Pending Settlements: {summary['pending_buy_amount']:>14,.0f} VND",
            f"Net Pending: {summary['net_pending']:>22,.0f} VND",
            "-" * 60,
            "📅 DAILY CASH AVAILABILITY:",
            "-" * 60,
        ]

        for date_str, info in predictions.items():
            emoji = "🟢" if info["available_cash"] > total_cash * 0.5 else "🟡"
            lines.append(
                f"   {emoji} {info['day_name'][:3]} {date_str}: "
                f"{info['available_cash']:>12,.0f} VND available "
                f"(+{info['settling_today']:,.0f} settling)"
            )

        lines.append("-" * 60)
        lines.append("📊 PENDING SETTLEMENTS:")

        for record in self._records:
            if record.status == "PENDING":
                lines.append(
                    f"   • {record.side} {record.symbol}: {record.amount:,.0f} VND "
                    f"→ {record.settlement_date}"
                )

        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# MARGIN TRADING SETTLEMENT - T+0 Support (IMPROVED v10.0)
# =============================================================================


@dataclass
class MarginSettlementRecord:
    """Settlement record for margin trades (T+0)"""

    trade_date: str
    symbol: str
    side: str
    quantity: int
    amount: float
    margin_ratio: float  # 0.5 = 50% margin used
    borrowed_amount: float
    settlement_type: str = "T+0"  # T+0 for margin, T+2 for cash
    interest_rate: float = 0.12  # 12% annual
    status: str = "OPEN"  # OPEN, CLOSED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MarginSettlementManager:
    """
    Margin Settlement Manager for T+0 Trading.

    IMPROVED v10.0: Complete margin settlement support.

    Vietnam Margin Rules:
    - T+0 settlement for margin accounts
    - Can buy and sell same day
    - Interest charged on borrowed amount
    - Initial margin: 50%
    - Maintenance margin: 35%

    Usage:
        manager = get_margin_settlement_manager()

        # Record margin buy
        manager.record_margin_trade("VNM", "BUY", 1000, 85_000_000, margin_ratio=0.5)

        # Check buying power
        buying_power = manager.get_buying_power(equity=100_000_000)
    """

    INITIAL_MARGIN = 0.50  # 50% margin requirement
    MAINTENANCE_MARGIN = 0.35  # 35% maintenance
    INTEREST_RATE_ANNUAL = 0.12  # 12% annual interest
    INTEREST_RATE_DAILY = INTEREST_RATE_ANNUAL / 365

    def __init__(self, state_file: str = "margin_settlement_state.json"):
        self.state_file = state_file
        self._records: List[MarginSettlementRecord] = []
        self._lock = RLock()
        self._load_state()
        logger.info("✅ MarginSettlementManager initialized")

    def _load_state(self):
        """Load persisted margin state"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._records = [MarginSettlementRecord(**r) for r in data.get("records", [])]
            except Exception as e:
                logger.warning(f"Failed to load margin state: {e}")
                self._records = []

    def _save_state(self):
        """Persist margin state"""
        try:
            data = {
                "records": [asdict(r) for r in self._records],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save margin state: {e}")

    def record_margin_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        amount: float,
        margin_ratio: float = 0.5,
    ) -> MarginSettlementRecord:
        """
        Record a margin trade (T+0 settlement).

        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            quantity: Number of shares
            amount: Total trade value
            margin_ratio: Margin used (0.5 = 50%)

        Returns:
            MarginSettlementRecord
        """
        with self._lock:
            borrowed = amount * margin_ratio if side == "BUY" else 0

            record = MarginSettlementRecord(
                trade_date=date.today().isoformat(),
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                amount=amount,
                margin_ratio=margin_ratio,
                borrowed_amount=borrowed,
            )

            self._records.append(record)
            self._save_state()

            logger.info(
                f"📝 Margin trade: {side} {symbol} {quantity} @ {amount:,.0f} VND | "
                f"Borrowed: {borrowed:,.0f} (T+0 settlement)"
            )

            return record

    def get_total_borrowed(self) -> float:
        """Get total borrowed amount across all open positions"""
        with self._lock:
            return sum(
                r.borrowed_amount for r in self._records if r.status == "OPEN" and r.side == "BUY"
            )

    def get_daily_interest(self) -> float:
        """Calculate daily interest on borrowed amount"""
        borrowed = self.get_total_borrowed()
        return borrowed * self.INTEREST_RATE_DAILY

    def get_buying_power(self, equity: float) -> Dict:
        """
        Calculate buying power for margin account.

        Args:
            equity: Current account equity

        Returns:
            Dict with buying power details
        """
        borrowed = self.get_total_borrowed()
        daily_interest = self.get_daily_interest()

        # Max borrowing = equity * (1 / INITIAL_MARGIN - 1)
        # e.g., 100M equity with 50% margin = can borrow up to 100M
        max_borrowing = equity * (1 / self.INITIAL_MARGIN - 1)
        available_margin = max(0, max_borrowing - borrowed)

        # Total buying power = equity + available_margin
        buying_power = equity + available_margin

        # T+0 same-day trading power (can use sell proceeds immediately)
        t0_power = buying_power  # Same as buying power for margin

        return {
            "equity": equity,
            "borrowed": borrowed,
            "max_borrowing": max_borrowing,
            "available_margin": available_margin,
            "buying_power": buying_power,
            "t0_trading_power": t0_power,
            "daily_interest": daily_interest,
            "margin_utilization_pct": (borrowed / max_borrowing * 100) if max_borrowing > 0 else 0,
            "maintenance_margin_value": equity * self.MAINTENANCE_MARGIN,
        }

    def can_trade_t0(
        self,
        symbol: str,
        side: str,
        amount: float,
        equity: float,
    ) -> Tuple[bool, str]:
        """
        Check if T+0 trade is possible with margin account.

        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            amount: Trade amount
            equity: Current equity

        Returns:
            (can_trade, reason)
        """
        power = self.get_buying_power(equity)

        if side == "BUY":
            if amount > power["buying_power"]:
                return (
                    False,
                    f"Insufficient buying power. "
                    f"Need: {amount:,.0f}, Available: {power['buying_power']:,.0f}",
                )
            return True, f"T+0 BUY OK. Buying power: {power['buying_power']:,.0f}"

        # SELL - check if we have the position
        return True, "T+0 SELL OK (immediate settlement)"

    def get_t0_vs_t2_comparison(self, equity: float, cash_balance: float) -> Dict:
        """
        Compare T+0 (margin) vs T+2 (cash) trading capabilities.

        Args:
            equity: Account equity (for margin)
            cash_balance: Cash balance (for cash account)

        Returns:
            Comparison dict
        """
        margin_power = self.get_buying_power(equity)

        # T+2 cash account
        tracker = get_settlement_tracker()
        cash_info = tracker.get_available_cash(cash_balance)

        return {
            "margin_account": {
                "settlement": "T+0",
                "buying_power": margin_power["buying_power"],
                "same_day_sell": True,
                "interest_cost": margin_power["daily_interest"],
                "advantage": "Can trade same day, use leverage",
            },
            "cash_account": {
                "settlement": "T+2",
                "buying_power": cash_info["available_cash"],
                "same_day_sell": False,
                "interest_cost": 0,
                "advantage": "No interest cost, no margin calls",
            },
            "recommendation": ("Use margin for active trading, cash for long-term holds"),
        }


# Singleton instances
_tracker_instance: Optional[SettlementTracker] = None
_margin_manager_instance: Optional[MarginSettlementManager] = None


def get_settlement_tracker() -> SettlementTracker:
    """Get singleton instance of settlement tracker"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = SettlementTracker()
    return _tracker_instance


def get_margin_settlement_manager() -> MarginSettlementManager:
    """Get singleton instance of margin settlement manager"""
    global _margin_manager_instance
    if _margin_manager_instance is None:
        _margin_manager_instance = MarginSettlementManager()
    return _margin_manager_instance


def reset_settlement_tracker() -> None:
    """Reset singleton instance (useful for testing)"""
    global _tracker_instance
    _tracker_instance = None


def reset_margin_settlement_manager() -> None:
    """Reset margin singleton instance (useful for testing)"""
    global _margin_manager_instance
    _margin_manager_instance = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def check_settlement_type(is_margin_account: bool = False) -> str:
    """
    Get settlement type based on account type.

    Args:
        is_margin_account: True if margin account

    Returns:
        "T+0" for margin, "T+2" for cash
    """
    return "T+0" if is_margin_account else "T+2"


def get_available_buying_power(
    cash_balance: float,
    equity: float = None,
    is_margin_account: bool = False,
) -> Dict:
    """
    Get available buying power based on account type.

    Args:
        cash_balance: Cash balance for cash account
        equity: Equity for margin account
        is_margin_account: True if margin account

    Returns:
        Dict with buying power details
    """
    if is_margin_account and equity is not None:
        manager = get_margin_settlement_manager()
        return manager.get_buying_power(equity)
    else:
        tracker = get_settlement_tracker()
        cash_info = tracker.get_available_cash(cash_balance)
        return {
            "equity": cash_balance,
            "buying_power": cash_info["available_cash"],
            "settlement_type": "T+2",
            "pending_settlements": cash_info["pending_settlements"],
        }
