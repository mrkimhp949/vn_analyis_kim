# -*- coding: utf-8 -*-
"""
Margin Call Tracking Module for Vietnam Stock Market

Vietnam margin rules:
- Initial margin: 50% (can borrow up to 50% of position value)
- Maintenance margin: 30-35% (broker dependent)
- Margin call trigger: When equity falls below maintenance level
- Force liquidation: When equity falls below 25% (emergency level)

Broker-specific margin rates (2024):
- VPS: 50% initial, 35% maintenance
- SSI: 50% initial, 33% maintenance
- TCBS: 50% initial, 30% maintenance
- VNDirect: 50% initial, 35% maintenance

This module tracks:
1. Current margin level
2. Margin call warnings
3. Force liquidation alerts
4. Position-specific margin requirements
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from threading import RLock

logger = logging.getLogger(__name__)


class MarginStatus(Enum):
    """Margin account status levels"""
    SAFE = "SAFE"                    # Margin ratio > 50%
    WARNING = "WARNING"              # 40% < Margin ratio <= 50%
    MARGIN_CALL = "MARGIN_CALL"      # 30% < Margin ratio <= 40%
    CRITICAL = "CRITICAL"            # 25% < Margin ratio <= 30%
    FORCE_LIQUIDATION = "FORCE_LIQUIDATION"  # Margin ratio <= 25%


@dataclass
class MarginPosition:
    """Individual position margin information"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    loan_amount: float
    equity: float
    margin_ratio: float
    marginable: bool = True
    margin_rate: float = 0.50  # 50% marginable by default
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        return (self.current_price - self.avg_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.avg_price <= 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price


@dataclass
class MarginAccountSummary:
    """Overall margin account summary"""
    total_market_value: float
    total_loan: float
    total_equity: float
    margin_ratio: float
    status: MarginStatus
    available_margin: float
    buying_power: float
    maintenance_requirement: float
    margin_call_amount: float = 0.0
    positions: List[MarginPosition] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class MarginTracker:
    """
    Track margin status and generate alerts for Vietnam stock market.

    Vietnam margin trading characteristics:
    - T+2 settlement affects margin calculations
    - Price limits (±7% HOSE) can trap positions
    - No short selling (except covered warrants)
    - Margin rates vary by stock (blue chips get better rates)

    Usage:
        tracker = MarginTracker(initial_margin=0.50, maintenance_margin=0.35)
        tracker.update_position("VNM", 1000, 80000, 85000, loan=40_000_000)
        summary = tracker.get_account_summary()
        if summary.status == MarginStatus.MARGIN_CALL:
            # Handle margin call
    """

    # Margin rates by stock tier (Vietnam specific)
    MARGIN_RATES = {
        "VN30": 0.50,      # Blue chips: 50% margin
        "MIDCAP": 0.40,    # Mid caps: 40% margin
        "SMALLCAP": 0.30,  # Small caps: 30% margin
        "SPECULATIVE": 0.0  # No margin for speculative stocks
    }

    # VN30 symbols (highest margin rate)
    VN30_SYMBOLS = {
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
    }

    # Non-marginable stocks (speculative, penny stocks)
    NON_MARGINABLE_PATTERNS = ["_WFT", "OTC", "-WR"]  # Warrants, OTC, rights

    def __init__(
        self,
        initial_margin: float = 0.50,
        maintenance_margin: float = 0.35,
        warning_margin: float = 0.40,
        critical_margin: float = 0.30,
        force_liquidation_margin: float = 0.25,
        broker: str = "SSI"
    ):
        """
        Initialize margin tracker.

        Args:
            initial_margin: Initial margin requirement (default 50%)
            maintenance_margin: Maintenance margin level (default 35%)
            warning_margin: Warning level (default 40%)
            critical_margin: Critical level before force liquidation (default 30%)
            force_liquidation_margin: Force liquidation trigger (default 25%)
            broker: Broker name for specific rules
        """
        self.initial_margin = initial_margin
        self.maintenance_margin = maintenance_margin
        self.warning_margin = warning_margin
        self.critical_margin = critical_margin
        self.force_liquidation_margin = force_liquidation_margin
        self.broker = broker

        # Position tracking
        self._positions: Dict[str, MarginPosition] = {}
        self._total_loan: float = 0.0
        self._cash_balance: float = 0.0

        # Thread safety
        self._lock = RLock()

        # Alert history
        self._alert_history: List[Dict] = []
        self._last_status: Optional[MarginStatus] = None

        logger.info(
            f"✅ MarginTracker initialized: "
            f"initial={initial_margin:.0%}, maintenance={maintenance_margin:.0%}, "
            f"broker={broker}"
        )

    def get_margin_rate(self, symbol: str) -> float:
        """
        Get margin rate for a specific symbol.

        Vietnam rules:
        - VN30 blue chips: 50% margin
        - Mid caps: 40% margin
        - Small caps: 30% margin
        - Speculative/warrants: 0% (no margin)
        """
        symbol = symbol.upper()

        # Check if non-marginable
        for pattern in self.NON_MARGINABLE_PATTERNS:
            if pattern in symbol:
                return 0.0

        # VN30 gets highest margin
        if symbol in self.VN30_SYMBOLS:
            return self.MARGIN_RATES["VN30"]

        # TODO: Integrate with stock list for proper tier classification
        # For now, default to mid-cap rate
        return self.MARGIN_RATES["MIDCAP"]

    def update_position(
        self,
        symbol: str,
        quantity: int,
        avg_price: float,
        current_price: float,
        loan_amount: float = 0.0
    ) -> MarginPosition:
        """
        Update or add a margin position.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            avg_price: Average purchase price
            current_price: Current market price
            loan_amount: Loan amount for this position

        Returns:
            Updated MarginPosition
        """
        with self._lock:
            market_value = quantity * current_price
            equity = market_value - loan_amount
            margin_ratio = equity / market_value if market_value > 0 else 1.0
            margin_rate = self.get_margin_rate(symbol)

            position = MarginPosition(
                symbol=symbol,
                quantity=quantity,
                avg_price=avg_price,
                current_price=current_price,
                market_value=market_value,
                loan_amount=loan_amount,
                equity=equity,
                margin_ratio=margin_ratio,
                marginable=margin_rate > 0,
                margin_rate=margin_rate,
                last_updated=datetime.now()
            )

            self._positions[symbol] = position

            logger.debug(
                f"📊 Updated margin position: {symbol} "
                f"qty={quantity}, price={current_price:,.0f}, "
                f"margin_ratio={margin_ratio:.1%}"
            )

            return position

    def remove_position(self, symbol: str) -> bool:
        """Remove a position from tracking."""
        with self._lock:
            if symbol in self._positions:
                del self._positions[symbol]
                logger.info(f"🗑️ Removed margin position: {symbol}")
                return True
            return False

    def set_cash_balance(self, cash: float):
        """Set current cash balance."""
        with self._lock:
            self._cash_balance = cash

    def set_total_loan(self, loan: float):
        """Set total loan amount."""
        with self._lock:
            self._total_loan = loan

    def get_margin_status(self, margin_ratio: float) -> MarginStatus:
        """
        Determine margin status based on ratio.

        Args:
            margin_ratio: Current margin ratio (equity/market_value)

        Returns:
            MarginStatus enum
        """
        if margin_ratio <= self.force_liquidation_margin:
            return MarginStatus.FORCE_LIQUIDATION
        elif margin_ratio <= self.critical_margin:
            return MarginStatus.CRITICAL
        elif margin_ratio <= self.maintenance_margin:
            return MarginStatus.MARGIN_CALL
        elif margin_ratio <= self.warning_margin:
            return MarginStatus.WARNING
        else:
            return MarginStatus.SAFE

    def get_account_summary(self) -> MarginAccountSummary:
        """
        Calculate and return account margin summary.

        Returns:
            MarginAccountSummary with all margin metrics
        """
        with self._lock:
            positions = list(self._positions.values())

            if not positions:
                return MarginAccountSummary(
                    total_market_value=0,
                    total_loan=self._total_loan,
                    total_equity=self._cash_balance,
                    margin_ratio=1.0,
                    status=MarginStatus.SAFE,
                    available_margin=self._cash_balance,
                    buying_power=self._cash_balance * 2,  # 50% margin = 2x buying power
                    maintenance_requirement=0,
                    positions=[],
                    last_updated=datetime.now()
                )

            # Calculate totals
            total_market_value = sum(p.market_value for p in positions)
            total_loan = self._total_loan or sum(p.loan_amount for p in positions)
            total_equity = total_market_value - total_loan + self._cash_balance

            # Overall margin ratio
            if total_market_value > 0:
                margin_ratio = total_equity / total_market_value
            else:
                margin_ratio = 1.0

            # Determine status
            status = self.get_margin_status(margin_ratio)

            # Calculate maintenance requirement
            maintenance_requirement = total_market_value * self.maintenance_margin

            # Calculate margin call amount (if applicable)
            margin_call_amount = 0.0
            if status in [MarginStatus.MARGIN_CALL, MarginStatus.CRITICAL, MarginStatus.FORCE_LIQUIDATION]:
                margin_call_amount = maintenance_requirement - total_equity

            # Calculate available margin and buying power
            excess_equity = total_equity - maintenance_requirement
            available_margin = max(0, excess_equity)
            buying_power = available_margin / self.initial_margin if self.initial_margin > 0 else 0

            # Generate warnings
            warnings = self._generate_warnings(margin_ratio, positions, status)

            # Track status change
            self._track_status_change(status, margin_ratio)

            return MarginAccountSummary(
                total_market_value=total_market_value,
                total_loan=total_loan,
                total_equity=total_equity,
                margin_ratio=margin_ratio,
                status=status,
                available_margin=available_margin,
                buying_power=buying_power,
                maintenance_requirement=maintenance_requirement,
                margin_call_amount=margin_call_amount,
                positions=positions,
                warnings=warnings,
                last_updated=datetime.now()
            )

    def _generate_warnings(
        self,
        margin_ratio: float,
        positions: List[MarginPosition],
        status: MarginStatus
    ) -> List[str]:
        """Generate warning messages based on margin status."""
        warnings = []

        if status == MarginStatus.FORCE_LIQUIDATION:
            warnings.append(
                f"🚨 FORCE LIQUIDATION IMMINENT! "
                f"Margin ratio {margin_ratio:.1%} below {self.force_liquidation_margin:.0%}. "
                f"Broker will liquidate positions!"
            )
        elif status == MarginStatus.CRITICAL:
            warnings.append(
                f"🔴 CRITICAL: Margin ratio {margin_ratio:.1%} approaching force liquidation. "
                f"Deposit cash or close positions immediately!"
            )
        elif status == MarginStatus.MARGIN_CALL:
            warnings.append(
                f"⚠️ MARGIN CALL: Margin ratio {margin_ratio:.1%} below maintenance "
                f"({self.maintenance_margin:.0%}). Action required within T+2!"
            )
        elif status == MarginStatus.WARNING:
            warnings.append(
                f"🟡 WARNING: Margin ratio {margin_ratio:.1%} approaching maintenance level. "
                f"Monitor closely."
            )

        # Check individual positions near floor (Vietnam specific)
        for pos in positions:
            if pos.unrealized_pnl_pct < -0.05:  # -5% loss
                warnings.append(
                    f"⚠️ {pos.symbol}: Unrealized loss {pos.unrealized_pnl_pct:.1%}, "
                    f"may hit floor if continues"
                )

        return warnings

    def _track_status_change(self, new_status: MarginStatus, margin_ratio: float):
        """Track status changes for alerting."""
        if self._last_status is not None and new_status != self._last_status:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "previous_status": self._last_status.value,
                "new_status": new_status.value,
                "margin_ratio": margin_ratio,
            }
            self._alert_history.append(alert)

            # Log status change
            if new_status.value in ["MARGIN_CALL", "CRITICAL", "FORCE_LIQUIDATION"]:
                logger.warning(
                    f"⚠️ MARGIN STATUS CHANGE: {self._last_status.value} → {new_status.value} "
                    f"(ratio: {margin_ratio:.1%})"
                )
            else:
                logger.info(
                    f"📊 Margin status: {self._last_status.value} → {new_status.value}"
                )

        self._last_status = new_status

    def can_open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        use_margin: bool = True
    ) -> Tuple[bool, str, float]:
        """
        Check if a new position can be opened within margin limits.

        Args:
            symbol: Stock symbol to buy
            quantity: Number of shares
            price: Purchase price
            use_margin: Whether to use margin for this purchase

        Returns:
            Tuple of (can_open, reason, max_quantity_allowed)
        """
        with self._lock:
            summary = self.get_account_summary()

            # Check if already in margin call
            if summary.status in [MarginStatus.MARGIN_CALL, MarginStatus.CRITICAL,
                                  MarginStatus.FORCE_LIQUIDATION]:
                return (
                    False,
                    f"Cannot open new positions - margin status: {summary.status.value}",
                    0
                )

            position_value = quantity * price
            margin_rate = self.get_margin_rate(symbol)

            if not use_margin or margin_rate == 0:
                # Cash purchase only
                required_cash = position_value
            else:
                # Margin purchase
                required_cash = position_value * (1 - margin_rate)

            if required_cash > summary.available_margin + self._cash_balance:
                max_affordable = (summary.available_margin + self._cash_balance)
                if use_margin and margin_rate > 0:
                    max_affordable = max_affordable / (1 - margin_rate)
                max_quantity = int(max_affordable / price)

                return (
                    False,
                    f"Insufficient margin. Required: {required_cash:,.0f} VND, "
                    f"Available: {summary.available_margin + self._cash_balance:,.0f} VND",
                    max_quantity
                )

            # Check if new position would breach margin limits
            new_total_value = summary.total_market_value + position_value
            new_loan = summary.total_loan + (position_value * margin_rate if use_margin else 0)
            new_equity = summary.total_equity + (position_value * (1 - margin_rate) if use_margin else position_value)
            new_margin_ratio = new_equity / new_total_value if new_total_value > 0 else 1.0

            if new_margin_ratio < self.warning_margin:
                return (
                    False,
                    f"Position would breach warning margin level "
                    f"(projected ratio: {new_margin_ratio:.1%})",
                    0
                )

            return (True, "OK", quantity)

    def get_liquidation_candidates(self, target_ratio: float = None) -> List[Dict]:
        """
        Get list of positions to liquidate to restore margin.

        Strategy: Liquidate positions with worst P&L first to minimize realized losses.

        Args:
            target_ratio: Target margin ratio (default: maintenance + 5%)

        Returns:
            List of positions to liquidate with recommended quantities
        """
        with self._lock:
            summary = self.get_account_summary()

            if summary.status == MarginStatus.SAFE:
                return []

            target = target_ratio or (self.maintenance_margin + 0.05)

            # Calculate how much equity we need to add
            required_equity = summary.total_market_value * target
            equity_shortfall = required_equity - summary.total_equity

            if equity_shortfall <= 0:
                return []

            # Sort positions by unrealized P&L (worst first)
            positions_sorted = sorted(
                summary.positions,
                key=lambda p: p.unrealized_pnl_pct
            )

            candidates = []
            remaining_shortfall = equity_shortfall

            for pos in positions_sorted:
                if remaining_shortfall <= 0:
                    break

                # Calculate how much of this position to sell
                # Selling releases: position_value * margin_rate as loan reduction
                # And keeps: position_value * (1 - margin_rate) as equity
                loan_reduction_per_share = pos.current_price * pos.margin_rate
                shares_needed = int(remaining_shortfall / loan_reduction_per_share) + 1
                shares_to_sell = min(shares_needed, pos.quantity)

                if shares_to_sell > 0:
                    candidates.append({
                        "symbol": pos.symbol,
                        "current_quantity": pos.quantity,
                        "sell_quantity": shares_to_sell,
                        "current_price": pos.current_price,
                        "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                        "value_to_sell": shares_to_sell * pos.current_price,
                        "reason": "Margin restoration"
                    })

                    remaining_shortfall -= shares_to_sell * loan_reduction_per_share

            return candidates

    def get_status_message(self) -> str:
        """Get formatted status message for display."""
        summary = self.get_account_summary()

        status_emoji = {
            MarginStatus.SAFE: "✅",
            MarginStatus.WARNING: "🟡",
            MarginStatus.MARGIN_CALL: "⚠️",
            MarginStatus.CRITICAL: "🔴",
            MarginStatus.FORCE_LIQUIDATION: "🚨"
        }

        lines = [
            "=" * 50,
            f"{status_emoji.get(summary.status, '❓')} MARGIN STATUS: {summary.status.value}",
            "=" * 50,
            f"📊 Total Market Value: {summary.total_market_value:>15,.0f} VND",
            f"💳 Total Loan:         {summary.total_loan:>15,.0f} VND",
            f"💰 Total Equity:       {summary.total_equity:>15,.0f} VND",
            f"📈 Margin Ratio:       {summary.margin_ratio:>14.1%}",
            "-" * 50,
            f"🔒 Maintenance Req:    {summary.maintenance_requirement:>15,.0f} VND",
            f"💵 Available Margin:   {summary.available_margin:>15,.0f} VND",
            f"🛒 Buying Power:       {summary.buying_power:>15,.0f} VND",
        ]

        if summary.margin_call_amount > 0:
            lines.append(f"⚠️ Margin Call Amount: {summary.margin_call_amount:>15,.0f} VND")

        if summary.warnings:
            lines.append("-" * 50)
            lines.append("⚠️ WARNINGS:")
            for warning in summary.warnings:
                lines.append(f"   {warning}")

        lines.append("=" * 50)

        return "\n".join(lines)


# Singleton instance
_margin_tracker: Optional[MarginTracker] = None


def get_margin_tracker(
    initial_margin: float = 0.50,
    maintenance_margin: float = 0.35,
    broker: str = "SSI"
) -> MarginTracker:
    """Get singleton margin tracker instance."""
    global _margin_tracker
    if _margin_tracker is None:
        _margin_tracker = MarginTracker(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            broker=broker
        )
    return _margin_tracker


# Test
if __name__ == "__main__":
    print("Testing Margin Tracker...")

    tracker = MarginTracker(
        initial_margin=0.50,
        maintenance_margin=0.35,
        broker="SSI"
    )

    # Add some positions
    tracker.set_cash_balance(50_000_000)  # 50M VND cash

    tracker.update_position(
        symbol="VNM",
        quantity=1000,
        avg_price=80_000,
        current_price=75_000,  # Lost 6.25%
        loan_amount=40_000_000  # 50% margin
    )

    tracker.update_position(
        symbol="HPG",
        quantity=2000,
        avg_price=25_000,
        current_price=23_000,  # Lost 8%
        loan_amount=25_000_000
    )

    # Get summary
    print(tracker.get_status_message())

    # Check if can open new position
    can_open, reason, max_qty = tracker.can_open_position("FPT", 500, 100_000)
    print(f"\nCan open FPT position: {can_open} - {reason}")

    # Get liquidation candidates
    candidates = tracker.get_liquidation_candidates()
    if candidates:
        print("\n📉 Liquidation Candidates:")
        for c in candidates:
            print(f"   {c['symbol']}: Sell {c['sell_quantity']} shares ({c['unrealized_pnl_pct']:.1%} P&L)")

    print("\n✅ Test completed!")
