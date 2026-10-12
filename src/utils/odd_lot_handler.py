# -*- coding: utf-8 -*-
"""
Odd-Lot Handler - Handle odd-lot trading for Vietnam market

Vietnam market rules for odd-lots (<100 shares):
1. Cannot trade odd-lots on main board (HOSE/HNX continuous session)
2. Must use odd-lot board with different rules:
   - Only sell orders allowed (no buying odd-lots)
   - Price must be at or below reference price
   - Matched at end of session only
   - Lower liquidity, may not fill

This module helps:
- Detect odd-lot positions
- Calculate optimal exit strategies
- Track odd-lot cleanup performance

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Vietnam lot size
VN_LOT_SIZE = 100


@dataclass
class OddLotPosition:
    """Represents an odd-lot position."""

    symbol: str
    odd_lot_shares: int  # Shares < 100
    full_lot_shares: int  # Shares in full lots
    total_shares: int
    avg_price: float
    current_price: float
    unrealized_pnl_pct: float

    @property
    def odd_lot_value(self) -> float:
        """Value of odd-lot portion."""
        return self.odd_lot_shares * self.current_price

    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable."""
        return self.unrealized_pnl_pct > 0


@dataclass
class OddLotExitRecommendation:
    """Recommendation for handling odd-lot."""

    symbol: str
    action: str  # "SELL_NOW", "WAIT", "COMBINE", "IGNORE"
    reason: str
    recommended_price: float
    expected_fill_probability: float  # 0-1
    estimated_cost_pct: float  # Cost as % of odd-lot value
    priority: int  # 1-5, 5 = highest priority


class OddLotHandler:
    """
    Handle odd-lot positions for Vietnam market.

    Key features:
    - Detect odd-lots in portfolio
    - Recommend optimal exit timing
    - Calculate true cost of odd-lot cleanup
    - Track odd-lot fill rates
    """

    # Thresholds
    MIN_ODD_LOT_VALUE_TO_CARE = 500_000  # 500K VND - below this, may not be worth effort
    MAX_ODD_LOT_HOLD_DAYS = 5  # Max days to hold odd-lot before forced cleanup
    ODD_LOT_DISCOUNT_PCT = 0.02  # Typical 2% discount for odd-lot sells

    # Fill probability by market conditions
    FILL_PROB_BULL = 0.85  # Higher fill rate in bull market
    FILL_PROB_SIDEWAYS = 0.70
    FILL_PROB_BEAR = 0.50  # Lower fill rate in bear market

    def __init__(self):
        self._odd_lot_history: Dict[str, List[Dict]] = {}
        self._fill_stats: Dict[str, Dict] = {}
        logger.info("✅ OddLotHandler initialized")

    def detect_odd_lots(self, positions: Dict[str, Dict]) -> List[OddLotPosition]:
        """
        Detect odd-lot positions in portfolio.

        Args:
            positions: Dict of symbol -> position data

        Returns:
            List of OddLotPosition objects
        """
        odd_lots = []

        for symbol, pos in positions.items():
            shares = pos.get("shares", 0)

            if shares <= 0:
                continue

            odd_lot_shares = shares % VN_LOT_SIZE
            full_lot_shares = shares - odd_lot_shares

            if odd_lot_shares > 0:
                avg_price = pos.get("avg_price", 0)
                current_price = pos.get("metadata", {}).get("last_price", avg_price)

                if avg_price > 0:
                    pnl_pct = (current_price - avg_price) / avg_price
                else:
                    pnl_pct = 0

                odd_lots.append(
                    OddLotPosition(
                        symbol=symbol,
                        odd_lot_shares=odd_lot_shares,
                        full_lot_shares=full_lot_shares,
                        total_shares=shares,
                        avg_price=avg_price,
                        current_price=current_price,
                        unrealized_pnl_pct=pnl_pct,
                    )
                )

        return odd_lots

    def get_exit_recommendation(
        self,
        odd_lot: OddLotPosition,
        market_regime: str = "SIDEWAYS",
        days_held: int = 0,
        reference_price: float = 0,
    ) -> OddLotExitRecommendation:
        """
        Get recommendation for handling an odd-lot position.

        Args:
            odd_lot: OddLotPosition to analyze
            market_regime: Current market regime
            days_held: Days position has been held
            reference_price: Today's reference price (for odd-lot board)

        Returns:
            OddLotExitRecommendation
        """
        symbol = odd_lot.symbol
        odd_value = odd_lot.odd_lot_value

        # Get fill probability based on market
        if market_regime == "BULL":
            fill_prob = self.FILL_PROB_BULL
        elif market_regime == "BEAR":
            fill_prob = self.FILL_PROB_BEAR
        else:
            fill_prob = self.FILL_PROB_SIDEWAYS

        # Adjust fill probability based on historical data
        if symbol in self._fill_stats:
            historical_fill_rate = self._fill_stats[symbol].get("fill_rate", fill_prob)
            fill_prob = (fill_prob + historical_fill_rate) / 2

        # Calculate recommended price (at or below reference)
        if reference_price > 0:
            recommended_price = reference_price * (1 - self.ODD_LOT_DISCOUNT_PCT)
        else:
            recommended_price = odd_lot.current_price * (1 - self.ODD_LOT_DISCOUNT_PCT)

        # Estimate cost (discount + potential non-fill cost)
        estimated_cost_pct = self.ODD_LOT_DISCOUNT_PCT + (1 - fill_prob) * 0.02

        # Determine action
        if odd_value < self.MIN_ODD_LOT_VALUE_TO_CARE:
            action = "IGNORE"
            reason = f"Odd-lot value {odd_value:,.0f} VND too small to prioritize"
            priority = 1

        elif days_held >= self.MAX_ODD_LOT_HOLD_DAYS:
            action = "SELL_NOW"
            reason = f"Held {days_held} days, exceeds max hold period"
            priority = 5

        elif odd_lot.is_profitable and odd_lot.unrealized_pnl_pct > 0.02:
            action = "SELL_NOW"
            reason = f"Profitable ({odd_lot.unrealized_pnl_pct:.1%}), lock in gains"
            priority = 4

        elif market_regime == "BEAR":
            action = "SELL_NOW"
            reason = "Bear market - reduce exposure"
            priority = 4

        elif odd_lot.full_lot_shares == 0:
            # Only odd-lot remaining
            action = "SELL_NOW"
            reason = "Only odd-lot remaining, cleanup position"
            priority = 3

        elif odd_lot.unrealized_pnl_pct < -0.05:
            # Losing position
            action = "SELL_NOW"
            reason = f"Losing position ({odd_lot.unrealized_pnl_pct:.1%}), cut losses"
            priority = 4

        else:
            action = "WAIT"
            reason = "No urgency, wait for better conditions"
            priority = 2

        return OddLotExitRecommendation(
            symbol=symbol,
            action=action,
            reason=reason,
            recommended_price=recommended_price,
            expected_fill_probability=fill_prob,
            estimated_cost_pct=estimated_cost_pct,
            priority=priority,
        )

    def calculate_cleanup_cost(
        self,
        odd_lots: List[OddLotPosition],
        market_regime: str = "SIDEWAYS",
    ) -> Dict:
        """
        Calculate total cost to cleanup all odd-lots.

        Args:
            odd_lots: List of odd-lot positions
            market_regime: Current market regime

        Returns:
            Dict with cost breakdown
        """
        total_odd_value = sum(ol.odd_lot_value for ol in odd_lots)
        total_discount_cost = total_odd_value * self.ODD_LOT_DISCOUNT_PCT

        # Estimate non-fill risk cost
        if market_regime == "BULL":
            non_fill_risk = 0.15
        elif market_regime == "BEAR":
            non_fill_risk = 0.50
        else:
            non_fill_risk = 0.30

        non_fill_cost = total_odd_value * non_fill_risk * 0.02  # 2% additional if not filled

        return {
            "total_odd_lot_value": total_odd_value,
            "num_odd_lots": len(odd_lots),
            "estimated_discount_cost": total_discount_cost,
            "estimated_non_fill_cost": non_fill_cost,
            "total_estimated_cost": total_discount_cost + non_fill_cost,
            "cost_as_pct_of_value": (
                (total_discount_cost + non_fill_cost) / total_odd_value * 100
                if total_odd_value > 0
                else 0
            ),
            "market_regime": market_regime,
        }

    def record_odd_lot_fill(
        self,
        symbol: str,
        shares: int,
        order_price: float,
        fill_price: float,
        filled: bool,
        fill_time_minutes: int = 0,
    ) -> None:
        """
        Record odd-lot fill result for tracking.

        Args:
            symbol: Stock symbol
            shares: Number of odd-lot shares
            order_price: Price order was placed at
            fill_price: Actual fill price (0 if not filled)
            filled: Whether order was filled
            fill_time_minutes: Time to fill in minutes
        """
        if symbol not in self._odd_lot_history:
            self._odd_lot_history[symbol] = []

        record = {
            "timestamp": datetime.now().isoformat(),
            "shares": shares,
            "order_price": order_price,
            "fill_price": fill_price,
            "filled": filled,
            "fill_time_minutes": fill_time_minutes,
            "slippage_pct": (
                (order_price - fill_price) / order_price if filled and order_price > 0 else 0
            ),
        }

        self._odd_lot_history[symbol].append(record)

        # Update fill stats
        self._update_fill_stats(symbol)

        logger.info(
            f"📊 Odd-lot {'filled' if filled else 'not filled'}: "
            f"{symbol} {shares} shares @ {fill_price:,.0f}"
        )

    def _update_fill_stats(self, symbol: str) -> None:
        """Update fill statistics for a symbol."""
        history = self._odd_lot_history.get(symbol, [])

        if not history:
            return

        fills = [h for h in history if h["filled"]]

        self._fill_stats[symbol] = {
            "total_attempts": len(history),
            "successful_fills": len(fills),
            "fill_rate": len(fills) / len(history) if history else 0,
            "avg_slippage_pct": sum(f["slippage_pct"] for f in fills) / len(fills) if fills else 0,
            "avg_fill_time_minutes": (
                sum(f["fill_time_minutes"] for f in fills) / len(fills) if fills else 0
            ),
        }

    def get_fill_stats(self, symbol: Optional[str] = None) -> Dict:
        """Get fill statistics."""
        if symbol:
            return self._fill_stats.get(symbol, {})
        return self._fill_stats

    def should_avoid_creating_odd_lot(
        self,
        current_shares: int,
        shares_to_sell: int,
    ) -> Tuple[bool, int]:
        """
        Check if a partial sell would create an odd-lot.

        Args:
            current_shares: Current position size
            shares_to_sell: Shares planning to sell

        Returns:
            Tuple of (would_create_odd_lot, recommended_shares_to_sell)
        """
        remaining = current_shares - shares_to_sell

        if remaining <= 0:
            return False, shares_to_sell

        odd_lot_remaining = remaining % VN_LOT_SIZE

        if odd_lot_remaining > 0:
            # Would create odd-lot
            # Option 1: Sell more to avoid odd-lot
            sell_more = shares_to_sell + odd_lot_remaining

            # Option 2: Sell less to keep full lots
            sell_less = shares_to_sell - (VN_LOT_SIZE - odd_lot_remaining)
            sell_less = max(0, (sell_less // VN_LOT_SIZE) * VN_LOT_SIZE)

            # Recommend the option closer to original intent
            if abs(sell_more - shares_to_sell) <= abs(sell_less - shares_to_sell):
                recommended = sell_more
            else:
                recommended = sell_less if sell_less > 0 else sell_more

            return True, recommended

        return False, shares_to_sell

    def round_to_lot_size(self, shares: int, round_up: bool = False) -> int:
        """
        Round shares to nearest lot size.

        Args:
            shares: Number of shares
            round_up: If True, round up; otherwise round down

        Returns:
            Rounded share count
        """
        if round_up:
            return ((shares + VN_LOT_SIZE - 1) // VN_LOT_SIZE) * VN_LOT_SIZE
        else:
            return (shares // VN_LOT_SIZE) * VN_LOT_SIZE


# Singleton instance
_handler_instance: Optional[OddLotHandler] = None


def get_odd_lot_handler() -> OddLotHandler:
    """Get singleton OddLotHandler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = OddLotHandler()
    return _handler_instance
