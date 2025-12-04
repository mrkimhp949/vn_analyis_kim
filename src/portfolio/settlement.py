"""
T+2 / T+2.5 Settlement Tracker for Vietnam Stock Market

Vietnam market operates on T+2 settlement with T+2.5 cash withdrawal:
- Buy order on Day 0 → Stock settled on Day 2 (can sell)
- Sell order on Day 0 → Cash settled on Day 2 (can use for trading)
- Cash withdrawal → Available on Day 2.5 (afternoon of T+2)

IMPORTANT: T+2 vs T+2.5 distinction
- T+2: Cash available for TRADING (buying new stocks)
- T+2.5: Cash available for WITHDRAWAL (transfer to bank)

This module tracks pending settlements to ensure:
1. Cash availability for new purchases (T+2)
2. Cash availability for withdrawal (T+2.5)
3. Stock availability for sales
4. Proper accounting of unsettled positions
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SettlementRecord:
    """Record of a pending settlement"""

    trade_id: str
    symbol: str
    trade_type: str  # 'BUY' or 'SELL'
    trade_date: datetime
    settlement_date: datetime
    shares: int
    price: float
    value: float
    is_settled: bool = False


class T2SettlementTracker:
    """
    Track T+2 settlements for Vietnam market

    Responsibilities:
    - Track pending stock settlements from buys
    - Track pending cash settlements from sells
    - Calculate available cash (settled + pending)
    - Calculate sellable shares (settled only)
    - Auto-mark settlements as settled after T+2
    """

    def __init__(self, settlement_days: int = 2):
        """
        Initialize T+2 settlement tracker

        Args:
            settlement_days: Number of days for settlement (default: 2 for Vietnam T+2)
        """
        from src.config.constants import VIETNAM_SETTLEMENT_DAYS

        self.settlement_days = settlement_days or VIETNAM_SETTLEMENT_DAYS

        # Pending settlements
        self.pending_settlements: List[SettlementRecord] = []

        # Track by symbol for quick lookup
        self.pending_by_symbol: Dict[str, List[SettlementRecord]] = defaultdict(list)

        logger.info(f"✅ T+{self.settlement_days} Settlement Tracker initialized")

    def record_buy(
        self,
        trade_id: str,
        symbol: str,
        shares: int,
        price: float,
        trade_date: Optional[datetime] = None,
    ) -> SettlementRecord:
        """
        Record a buy trade for T+2 settlement tracking

        Args:
            trade_id: Unique trade identifier
            symbol: Stock symbol
            shares: Number of shares bought
            price: Purchase price
            trade_date: Trade execution date (default: now)

        Returns:
            SettlementRecord for the buy trade
        """
        if trade_date is None:
            trade_date = datetime.now()

        settlement_date = self._calculate_settlement_date(trade_date)
        value = shares * price

        record = SettlementRecord(
            trade_id=trade_id,
            symbol=symbol,
            trade_type="BUY",
            trade_date=trade_date,
            settlement_date=settlement_date,
            shares=shares,
            price=price,
            value=value,
            is_settled=False,
        )

        self.pending_settlements.append(record)
        self.pending_by_symbol[symbol].append(record)

        logger.info(
            f"📝 Recorded BUY settlement: {symbol} {shares} shares @ {price:,.0f} "
            f"(settles on {settlement_date.strftime('%Y-%m-%d')})"
        )

        return record

    def record_sell(
        self,
        trade_id: str,
        symbol: str,
        shares: int,
        price: float,
        trade_date: Optional[datetime] = None,
    ) -> SettlementRecord:
        """
        Record a sell trade for T+2 cash settlement tracking

        Args:
            trade_id: Unique trade identifier
            symbol: Stock symbol
            shares: Number of shares sold
            price: Sale price
            trade_date: Trade execution date (default: now)

        Returns:
            SettlementRecord for the sell trade
        """
        if trade_date is None:
            trade_date = datetime.now()

        settlement_date = self._calculate_settlement_date(trade_date)
        value = shares * price

        record = SettlementRecord(
            trade_id=trade_id,
            symbol=symbol,
            trade_type="SELL",
            trade_date=trade_date,
            settlement_date=settlement_date,
            shares=shares,
            price=price,
            value=value,
            is_settled=False,
        )

        self.pending_settlements.append(record)
        self.pending_by_symbol[symbol].append(record)

        logger.info(
            f"📝 Recorded SELL settlement: {symbol} {shares} shares @ {price:,.0f} "
            f"(cash settles on {settlement_date.strftime('%Y-%m-%d')})"
        )

        return record

    def _calculate_settlement_date(self, trade_date: datetime) -> datetime:
        """
        Calculate settlement date (T+2 trading days)

        IMPROVED: Now properly accounts for:
        - Weekends (Saturday, Sunday)
        - Vietnam public holidays (Tết, 30/4, 1/5, 2/9, Hung Kings' Day)

        Uses Vietnam market calendar from src/utils/vietnam_market.py
        """
        from src.utils.vietnam_market import get_next_trading_day

        # Get T+2 trading days (skips weekends and VN holidays)
        settlement_date = get_next_trading_day(trade_date, days_ahead=self.settlement_days)

        return settlement_date

    def update_settlements(self, current_date: Optional[datetime] = None) -> int:
        """
        Update settlement status - mark settled if settlement_date has passed

        Args:
            current_date: Current date (default: now)

        Returns:
            Number of settlements marked as settled
        """
        if current_date is None:
            current_date = datetime.now()

        settled_count = 0

        for record in self.pending_settlements:
            if not record.is_settled and current_date >= record.settlement_date:
                record.is_settled = True
                settled_count += 1

                logger.debug(
                    f"✅ Settlement completed: {record.symbol} {record.trade_type} "
                    f"{record.shares} shares (trade_id: {record.trade_id})"
                )

        if settled_count > 0:
            logger.info(f"✅ Marked {settled_count} settlements as completed")

        return settled_count

    def get_available_cash(
        self, total_cash: float, current_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate available cash accounting for unsettled sells

        Args:
            total_cash: Total cash including unsettled
            current_date: Current date (default: now)

        Returns:
            Dict with settled_cash, pending_cash, and available_cash
        """
        if current_date is None:
            current_date = datetime.now()

        # Update settlements first
        self.update_settlements(current_date)

        # Calculate pending cash from unsettled sells
        pending_cash = sum(
            record.value
            for record in self.pending_settlements
            if record.trade_type == "SELL" and not record.is_settled
        )

        # Available cash = total - pending
        # (assuming total_cash includes pending settlements)
        settled_cash = total_cash - pending_cash
        available_cash = settled_cash  # Only use settled cash for new buys

        return {
            "total_cash": total_cash,
            "settled_cash": settled_cash,
            "pending_cash": pending_cash,
            "available_cash": available_cash,
        }

    def get_sellable_shares(
        self, symbol: str, total_shares: int, current_date: Optional[datetime] = None
    ) -> Dict[str, int]:
        """
        Calculate sellable shares accounting for unsettled buys

        Args:
            symbol: Stock symbol
            total_shares: Total shares including unsettled
            current_date: Current date (default: now)

        Returns:
            Dict with total_shares, settled_shares, pending_shares, sellable_shares
        """
        if current_date is None:
            current_date = datetime.now()

        # Update settlements first
        self.update_settlements(current_date)

        # Calculate pending shares from unsettled buys
        pending_shares = sum(
            record.shares
            for record in self.pending_by_symbol[symbol]
            if record.trade_type == "BUY" and not record.is_settled
        )

        # Sellable shares = total - pending
        settled_shares = total_shares - pending_shares
        sellable_shares = max(0, settled_shares)  # Can't sell negative shares

        return {
            "total_shares": total_shares,
            "settled_shares": settled_shares,
            "pending_shares": pending_shares,
            "sellable_shares": sellable_shares,
        }

    def get_pending_settlements(self, settled: Optional[bool] = None) -> List[SettlementRecord]:
        """
        Get pending settlements, optionally filtered by settled status

        Args:
            settled: Filter by settled status (None = all, True = settled only, False = unsettled only)

        Returns:
            List of SettlementRecord
        """
        if settled is None:
            return self.pending_settlements.copy()

        return [record for record in self.pending_settlements if record.is_settled == settled]

    def get_settlement_summary(self, current_date: Optional[datetime] = None) -> Dict:
        """
        Get summary of settlement status

        Returns:
            Dict with settlement statistics
        """
        if current_date is None:
            current_date = datetime.now()

        self.update_settlements(current_date)

        unsettled = [r for r in self.pending_settlements if not r.is_settled]
        settled = [r for r in self.pending_settlements if r.is_settled]

        unsettled_buys = [r for r in unsettled if r.trade_type == "BUY"]
        unsettled_sells = [r for r in unsettled if r.trade_type == "SELL"]

        pending_stock_value = sum(r.value for r in unsettled_buys)
        pending_cash_value = sum(r.value for r in unsettled_sells)

        return {
            "total_settlements": len(self.pending_settlements),
            "settled_count": len(settled),
            "unsettled_count": len(unsettled),
            "unsettled_buys": len(unsettled_buys),
            "unsettled_sells": len(unsettled_sells),
            "pending_stock_value": pending_stock_value,
            "pending_cash_value": pending_cash_value,
            "oldest_unsettled": min([r.trade_date for r in unsettled], default=None),
        }

    def get_withdrawable_cash(
        self, total_cash: float, current_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate withdrawable cash (T+2.5 rule) for Vietnam market.

        IMPORTANT: In Vietnam:
        - T+2: Cash is available for TRADING (buying new stocks)
        - T+2.5: Cash is available for WITHDRAWAL (transfer to bank)

        The 0.5 day difference means:
        - Morning of T+2: Cash available for trading only
        - Afternoon of T+2 (after 13:00): Cash available for withdrawal

        Args:
            total_cash: Total cash including unsettled
            current_date: Current date/time (default: now)

        Returns:
            Dict with trading_cash, withdrawable_cash, pending_withdrawal
        """
        if current_date is None:
            current_date = datetime.now()

        # First get trading cash (T+2 settled)
        trading_cash_info = self.get_available_cash(total_cash, current_date)
        trading_cash = trading_cash_info["available_cash"]

        # Check time of day for T+2.5 withdrawal availability
        current_hour = current_date.hour

        # T+2.5: Withdrawal available after 13:00 on T+2
        withdrawal_cutoff_hour = 13

        # Calculate pending cash from sells that are T+2 settled but not yet T+2.5
        pending_withdrawal = 0.0

        for record in self.pending_settlements:
            if record.trade_type == "SELL" and record.is_settled:
                # Check if it's the settlement day
                if record.settlement_date.date() == current_date.date():
                    if current_hour < withdrawal_cutoff_hour:
                        # Before 13:00 - not yet withdrawable
                        pending_withdrawal += record.value

        # Withdrawable cash = trading cash - pending withdrawal
        withdrawable_cash = max(0, trading_cash - pending_withdrawal)

        return {
            "total_cash": total_cash,
            "trading_cash": trading_cash,  # T+2 - available for buying
            "withdrawable_cash": withdrawable_cash,  # T+2.5 - available for bank transfer
            "pending_withdrawal": pending_withdrawal,  # Will be withdrawable after 13:00
            "can_withdraw_after": "13:00" if pending_withdrawal > 0 else None,
            "note": (
                "⚠️ Cash pending T+2.5 settlement - available for trading but not withdrawal"
                if pending_withdrawal > 0
                else "✅ All settled cash is withdrawable"
            ),
        }

    def get_settlement_timeline(
        self, symbol: Optional[str] = None, current_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get detailed settlement timeline for planning.

        Shows when each pending settlement will complete for:
        - Stock availability (T+2)
        - Cash for trading (T+2)
        - Cash for withdrawal (T+2.5)

        Args:
            symbol: Optional filter by symbol
            current_date: Current date (default: now)

        Returns:
            Dict with settlement timeline
        """
        if current_date is None:
            current_date = datetime.now()

        self.update_settlements(current_date)

        timeline = {
            "today": [],
            "tomorrow": [],
            "day_after": [],
            "later": [],
        }

        for record in self.pending_settlements:
            if record.is_settled:
                continue

            if symbol and record.symbol != symbol:
                continue

            days_until = (record.settlement_date.date() - current_date.date()).days

            entry = {
                "trade_id": record.trade_id,
                "symbol": record.symbol,
                "type": record.trade_type,
                "shares": record.shares,
                "value": record.value,
                "trade_date": record.trade_date.strftime("%Y-%m-%d"),
                "settlement_date": record.settlement_date.strftime("%Y-%m-%d"),
                "days_until": days_until,
            }

            if record.trade_type == "SELL":
                # Add withdrawal availability (T+2.5)
                entry["trading_available"] = record.settlement_date.strftime("%Y-%m-%d 09:00")
                entry["withdrawal_available"] = record.settlement_date.strftime("%Y-%m-%d 13:00")

            if days_until <= 0:
                timeline["today"].append(entry)
            elif days_until == 1:
                timeline["tomorrow"].append(entry)
            elif days_until == 2:
                timeline["day_after"].append(entry)
            else:
                timeline["later"].append(entry)

        return {
            "current_date": current_date.strftime("%Y-%m-%d %H:%M"),
            "timeline": timeline,
            "total_pending": sum(len(v) for v in timeline.values()),
        }

    def cleanup_old_settlements(self, days_to_keep: int = 30) -> int:
        """
        Remove old settled records to prevent memory growth

        Args:
            days_to_keep: Keep settled records for this many days (default: 30)

        Returns:
            Number of records cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Filter out old settled records
        before_count = len(self.pending_settlements)

        self.pending_settlements = [
            record
            for record in self.pending_settlements
            if not record.is_settled or record.settlement_date >= cutoff_date
        ]

        # Rebuild by_symbol index
        self.pending_by_symbol.clear()
        for record in self.pending_settlements:
            self.pending_by_symbol[record.symbol].append(record)

        after_count = len(self.pending_settlements)
        cleaned = before_count - after_count

        if cleaned > 0:
            logger.info(
                f"🧹 Cleaned up {cleaned} old settlement records (kept last {days_to_keep} days)"
            )

        return cleaned


# Singleton
_settlement_tracker = None


def get_settlement_tracker() -> T2SettlementTracker:
    """Get T+2 settlement tracker singleton"""
    global _settlement_tracker
    if _settlement_tracker is None:
        _settlement_tracker = T2SettlementTracker()
    return _settlement_tracker
