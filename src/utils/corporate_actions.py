# -*- coding: utf-8 -*-
"""
Corporate Action Handler - Stock Split, Dividend, etc.

Handles corporate actions that affect position tracking:
- Stock splits (forward and reverse)
- Cash dividends
- Stock dividends
- Rights issues
- Mergers & acquisitions

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class CorporateActionType(Enum):
    """Types of corporate actions"""

    STOCK_SPLIT = "STOCK_SPLIT"  # Forward split (e.g., 2:1)
    REVERSE_SPLIT = "REVERSE_SPLIT"  # Reverse split (e.g., 1:5)
    CASH_DIVIDEND = "CASH_DIVIDEND"
    STOCK_DIVIDEND = "STOCK_DIVIDEND"
    RIGHTS_ISSUE = "RIGHTS_ISSUE"
    BONUS_ISSUE = "BONUS_ISSUE"
    MERGER = "MERGER"
    SPINOFF = "SPINOFF"


@dataclass
class CorporateAction:
    """Corporate action event"""

    symbol: str
    action_type: CorporateActionType
    ex_date: date
    record_date: date
    payment_date: Optional[date] = None

    # Split/Dividend ratios
    ratio_from: float = 1.0  # e.g., 1 for 2:1 split
    ratio_to: float = 1.0  # e.g., 2 for 2:1 split

    # Cash dividend
    dividend_per_share: float = 0.0

    # Description
    description: str = ""

    @property
    def adjustment_factor(self) -> float:
        """Get price adjustment factor for splits/dividends"""
        if self.ratio_to == 0:
            return 1.0
        return self.ratio_from / self.ratio_to


class CorporateActionHandler:
    """
    Handle corporate actions for position adjustments

    Usage:
        handler = CorporateActionHandler()

        # Register a stock split
        handler.register_split("VNM", 2, 1, date(2025, 1, 15))  # 2:1 split

        # Adjust positions after ex-date
        adjusted_qty, adjusted_price = handler.adjust_position(
            "VNM", 1000, 85000, date(2025, 1, 16)
        )
    """

    def __init__(self):
        self._actions: Dict[str, List[CorporateAction]] = {}
        self._applied_actions: Dict[str, set] = {}  # Track applied actions per position

    def register_split(
        self,
        symbol: str,
        ratio_to: float,
        ratio_from: float,
        ex_date: date,
        record_date: Optional[date] = None,
        description: str = "",
    ) -> CorporateAction:
        """
        Register a stock split

        Args:
            symbol: Stock symbol
            ratio_to: New shares per old share (e.g., 2 for 2:1 split)
            ratio_from: Old shares (usually 1)
            ex_date: Ex-dividend date
            record_date: Record date (defaults to ex_date - 1)
            description: Optional description

        Returns:
            CorporateAction object

        Example:
            # 2:1 split (each share becomes 2)
            register_split("VNM", 2, 1, date(2025, 1, 15))

            # 1:5 reverse split (5 shares become 1)
            register_split("VNM", 1, 5, date(2025, 1, 15))
        """
        action_type = (
            CorporateActionType.STOCK_SPLIT
            if ratio_to > ratio_from
            else CorporateActionType.REVERSE_SPLIT
        )

        action = CorporateAction(
            symbol=symbol.upper(),
            action_type=action_type,
            ex_date=ex_date,
            record_date=record_date or ex_date,
            ratio_from=ratio_from,
            ratio_to=ratio_to,
            description=description or f"{int(ratio_to)}:{int(ratio_from)} split",
        )

        if symbol not in self._actions:
            self._actions[symbol] = []
        self._actions[symbol].append(action)

        logger.info(
            f"📋 Registered {action_type.value} for {symbol}: "
            f"{int(ratio_to)}:{int(ratio_from)} on {ex_date}"
        )

        return action

    def register_dividend(
        self,
        symbol: str,
        dividend_per_share: float,
        ex_date: date,
        payment_date: Optional[date] = None,
        is_stock_dividend: bool = False,
        stock_ratio: float = 0.0,
    ) -> CorporateAction:
        """
        Register a dividend (cash or stock)

        Args:
            symbol: Stock symbol
            dividend_per_share: Cash dividend per share (VND)
            ex_date: Ex-dividend date
            payment_date: Payment date
            is_stock_dividend: True if stock dividend
            stock_ratio: Stock dividend ratio (e.g., 0.1 for 10% bonus)

        Returns:
            CorporateAction object
        """
        action_type = (
            CorporateActionType.STOCK_DIVIDEND
            if is_stock_dividend
            else CorporateActionType.CASH_DIVIDEND
        )

        action = CorporateAction(
            symbol=symbol.upper(),
            action_type=action_type,
            ex_date=ex_date,
            record_date=ex_date,
            payment_date=payment_date,
            dividend_per_share=dividend_per_share,
            ratio_to=1 + stock_ratio if is_stock_dividend else 1.0,
            ratio_from=1.0,
            description=(
                f"Dividend: {dividend_per_share:,.0f} VND/share"
                if not is_stock_dividend
                else f"Stock dividend: {stock_ratio*100:.0f}%"
            ),
        )

        if symbol not in self._actions:
            self._actions[symbol] = []
        self._actions[symbol].append(action)

        logger.info(
            f"📋 Registered {action_type.value} for {symbol}: "
            f"{dividend_per_share:,.0f} VND on {ex_date}"
        )

        return action

    def get_pending_actions(
        self,
        symbol: str,
        as_of_date: date,
    ) -> List[CorporateAction]:
        """Get pending corporate actions for a symbol after a date"""
        if symbol.upper() not in self._actions:
            return []

        return [action for action in self._actions[symbol.upper()] if action.ex_date >= as_of_date]

    def adjust_position(
        self,
        symbol: str,
        quantity: int,
        avg_price: float,
        position_date: date,
        current_date: Optional[date] = None,
    ) -> Tuple[int, float]:
        """
        Adjust position for any corporate actions between position_date and current_date

        Args:
            symbol: Stock symbol
            quantity: Current share quantity
            avg_price: Current average price
            position_date: Date position was opened
            current_date: Current date (defaults to today)

        Returns:
            Tuple of (adjusted_quantity, adjusted_avg_price)
        """
        current_date = current_date or date.today()
        symbol = symbol.upper()

        if symbol not in self._actions:
            return quantity, avg_price

        adjusted_qty = quantity
        adjusted_price = avg_price

        # Sort actions by ex_date
        actions = sorted(self._actions[symbol], key=lambda x: x.ex_date)

        for action in actions:
            # Only apply actions that occurred after position was opened
            if action.ex_date <= position_date or action.ex_date > current_date:
                continue

            # Check if already applied (track by action hash)
            action_id = f"{action.symbol}_{action.ex_date}_{action.action_type.value}"
            if symbol not in self._applied_actions:
                self._applied_actions[symbol] = set()

            if action_id in self._applied_actions[symbol]:
                continue

            # Apply adjustment
            if action.action_type in [
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.REVERSE_SPLIT,
                CorporateActionType.STOCK_DIVIDEND,
            ]:
                factor = action.ratio_to / action.ratio_from if action.ratio_from > 0 else 1.0
                adjusted_qty = int(adjusted_qty * factor)
                adjusted_price = adjusted_price / factor if factor > 0 else adjusted_price

                logger.info(
                    f"📊 Applied {action.action_type.value} to {symbol}: "
                    f"qty {quantity} -> {adjusted_qty}, "
                    f"price {avg_price:,.0f} -> {adjusted_price:,.0f}"
                )

            self._applied_actions[symbol].add(action_id)

        return adjusted_qty, adjusted_price

    def adjust_historical_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        split_adjust: bool = True,
        dividend_adjust: bool = False,
    ) -> pd.DataFrame:
        """
        Adjust historical price data for corporate actions

        Args:
            df: DataFrame with OHLCV data (must have 'time' column)
            symbol: Stock symbol
            split_adjust: Adjust for splits
            dividend_adjust: Adjust for dividends

        Returns:
            Adjusted DataFrame
        """
        if symbol.upper() not in self._actions:
            return df

        df = df.copy()

        # Ensure time column is datetime
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])

        actions = sorted(self._actions[symbol.upper()], key=lambda x: x.ex_date, reverse=True)

        for action in actions:
            if not split_adjust and action.action_type in [
                CorporateActionType.STOCK_SPLIT,
                CorporateActionType.REVERSE_SPLIT,
            ]:
                continue

            if not dividend_adjust and action.action_type == CorporateActionType.CASH_DIVIDEND:
                continue

            # Get adjustment factor
            factor = action.adjustment_factor

            if action.action_type == CorporateActionType.CASH_DIVIDEND:
                # Dividend adjustment: subtract dividend from historical prices
                mask = df["time"].dt.date < action.ex_date
                df.loc[mask, ["open", "high", "low", "close"]] -= action.dividend_per_share
            else:
                # Split adjustment: multiply prices by factor, divide volume
                mask = df["time"].dt.date < action.ex_date
                df.loc[mask, ["open", "high", "low", "close"]] *= factor
                if "volume" in df.columns:
                    df.loc[mask, "volume"] = (df.loc[mask, "volume"] / factor).astype(int)

        return df

    def get_upcoming_events(
        self,
        symbols: Optional[List[str]] = None,
        days_ahead: int = 30,
    ) -> List[CorporateAction]:
        """Get upcoming corporate actions"""
        today = date.today()
        from datetime import timedelta

        end_date = today + timedelta(days=days_ahead)

        upcoming = []

        symbols_to_check = symbols or list(self._actions.keys())

        for symbol in symbols_to_check:
            symbol = symbol.upper()
            if symbol not in self._actions:
                continue

            for action in self._actions[symbol]:
                if today <= action.ex_date <= end_date:
                    upcoming.append(action)

        return sorted(upcoming, key=lambda x: x.ex_date)


# Singleton instance
_corporate_action_handler: Optional[CorporateActionHandler] = None


def get_corporate_action_handler() -> CorporateActionHandler:
    """Get singleton corporate action handler"""
    global _corporate_action_handler
    if _corporate_action_handler is None:
        _corporate_action_handler = CorporateActionHandler()
    return _corporate_action_handler
