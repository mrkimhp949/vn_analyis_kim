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

                # Parse settlement date (handle both date and datetime formats)
                try:
                    settlement_str = record.settlement_date
                    if "T" in settlement_str:
                        # datetime format: "2025-12-10T00:00:00"
                        settlement_date = datetime.fromisoformat(settlement_str).date()
                    else:
                        # date format: "2025-12-10"
                        settlement_date = date.fromisoformat(settlement_str)
                except (ValueError, AttributeError):
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
                    settlement_date = date.fromisoformat(record.settlement_date)
                    if settlement_date > today:
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


# Singleton instance
_tracker_instance: Optional[SettlementTracker] = None


def get_settlement_tracker() -> SettlementTracker:
    """Get singleton instance of settlement tracker"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = SettlementTracker()
    return _tracker_instance
