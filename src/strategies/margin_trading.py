# -*- coding: utf-8 -*-
"""
Margin Trading Module - Vietnam Market

Handles margin trading specific logic:
- Margin call detection and handling
- Maintenance margin monitoring
- Force liquidation prevention
- Margin utilization optimization

Vietnam Margin Rules:
- Initial margin: 50-70% depending on stock (VN30 = 50%, others = 60-70%)
- Maintenance margin: 30-40% 
- Margin call trigger: When equity falls below maintenance margin
- Force liquidation: When equity falls below 25% or no response to margin call

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


class MarginTier(Enum):
    """Margin tier classification based on stock quality"""

    TIER_1 = "TIER_1"  # VN30 - 50% initial margin
    TIER_2 = "TIER_2"  # Large caps outside VN30 - 60% initial margin
    TIER_3 = "TIER_3"  # Mid caps - 70% initial margin
    NO_MARGIN = "NO_MARGIN"  # Stocks not eligible for margin


class MarginStatus(Enum):
    """Margin account status levels"""

    SAFE = "SAFE"  # Equity > 50% of loan
    WARNING = "WARNING"  # Equity 40-50% of loan
    MARGIN_CALL = "MARGIN_CALL"  # Equity 30-40% of loan
    FORCE_SELL = "FORCE_SELL"  # Equity < 30% of loan
    CRITICAL = "CRITICAL"  # Equity < 25% of loan - immediate liquidation


@dataclass
class MarginConfig:
    """
    Margin trading configuration for Vietnam market

    Based on typical VN broker margin requirements:
    - SSI, VNDirect, TCBS have similar margin policies
    """

    # Initial margin requirements by tier
    initial_margin_tier1: float = 0.50  # VN30: 50% (can borrow up to 2x)
    initial_margin_tier2: float = 0.60  # Large cap: 60%
    initial_margin_tier3: float = 0.70  # Mid cap: 70%

    # Maintenance margin thresholds
    maintenance_margin: float = 0.35  # 35% maintenance margin
    margin_call_threshold: float = 0.40  # Margin call at 40%
    warning_threshold: float = 0.50  # Warning at 50%
    force_sell_threshold: float = 0.30  # Force sell at 30%
    critical_threshold: float = 0.25  # Critical/immediate liquidation at 25%

    # Interest rates (annual)
    margin_interest_rate: float = 0.12  # 12% annual interest typical in VN

    # Risk management
    max_margin_utilization: float = 0.80  # Max 80% of available margin
    position_concentration_limit: float = 0.40  # Max 40% of margin in single stock

    # Margin call handling
    margin_call_response_hours: int = 24  # Hours to respond to margin call (T+1)
    auto_reduce_on_warning: bool = True  # Auto-reduce position on warning
    warning_reduce_pct: float = 0.20  # Reduce 20% on warning

    # VN30 symbols for tier classification
    vn30_symbols: List[str] = field(
        default_factory=lambda: [
            "ACB",
            "BCM",
            "BID",
            "BVH",
            "CTG",
            "FPT",
            "GAS",
            "GVR",
            "HDB",
            "HPG",
            "MBB",
            "MSN",
            "MWG",
            "PLX",
            "POW",
            "SAB",
            "SHB",
            "SSB",
            "SSI",
            "STB",
            "TCB",
            "TPB",
            "VCB",
            "VHM",
            "VIB",
            "VIC",
            "VJC",
            "VNM",
            "VPB",
            "VRE",
        ]
    )


@dataclass
class MarginPosition:
    """Single margin position details"""

    symbol: str
    shares: int
    avg_cost: float
    current_price: float
    margin_tier: MarginTier
    loan_amount: float  # Amount borrowed

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def equity(self) -> float:
        return self.market_value - self.loan_amount

    @property
    def equity_ratio(self) -> float:
        """Equity as percentage of market value"""
        if self.market_value <= 0:
            return 0
        return self.equity / self.market_value

    @property
    def loan_to_value(self) -> float:
        """Loan as percentage of market value (inverse of equity ratio)"""
        return 1 - self.equity_ratio

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - (self.shares * self.avg_cost)

    @property
    def unrealized_pnl_pct(self) -> float:
        cost = self.shares * self.avg_cost
        if cost <= 0:
            return 0
        return self.unrealized_pnl / cost


@dataclass
class MarginCallEvent:
    """Margin call event details"""

    timestamp: datetime
    symbol: str
    equity_ratio: float
    required_equity_ratio: float
    shortfall_amount: float
    deadline: datetime
    status: str  # "PENDING", "RESOLVED", "FORCE_SOLD"
    resolution_action: Optional[str] = None


@dataclass
class MarginAccountSummary:
    """Overall margin account summary"""

    total_equity: float
    total_market_value: float
    total_loan: float
    available_margin: float
    margin_utilization: float
    overall_status: MarginStatus
    positions_at_risk: List[str]
    margin_calls: List[MarginCallEvent]


class MarginTradingManager:
    """
    Manages margin trading operations for Vietnam market

    Key responsibilities:
    1. Track margin positions and equity ratios
    2. Detect margin call conditions
    3. Generate margin call alerts
    4. Suggest position reductions to avoid force sell
    5. Calculate optimal margin utilization
    """

    def __init__(self, config: Optional[MarginConfig] = None):
        self.config = config or MarginConfig()
        self.positions: Dict[str, MarginPosition] = {}
        self.margin_calls: List[MarginCallEvent] = []
        self.call_history: List[Dict] = []

        # Try to load dynamic VN30 symbols
        try:
            from src.utils.vietnam_market import get_vn30_symbols_dynamic

            self.config.vn30_symbols = list(get_vn30_symbols_dynamic())
            logger.info(f"Loaded {len(self.config.vn30_symbols)} VN30 symbols dynamically")
        except ImportError:
            logger.warning("Using static VN30 list for margin tier classification")

    def get_margin_tier(self, symbol: str) -> MarginTier:
        """
        Determine margin tier for a symbol

        Tier 1: VN30 stocks - highest margin availability
        Tier 2: Large caps outside VN30
        Tier 3: Mid caps
        No Margin: Small caps, penny stocks
        """
        symbol = symbol.upper()

        if symbol in self.config.vn30_symbols:
            return MarginTier.TIER_1

        # Check market cap tier (would need market data integration)
        # For now, assume large caps based on common knowledge
        large_caps = {"DGC", "PNJ", "REE", "DPM", "DCM", "GMD", "NLG", "KDH", "VND", "HCM"}
        if symbol in large_caps:
            return MarginTier.TIER_2

        # Default to Tier 3 for other marginable stocks
        # In production, would check against broker's marginable list
        return MarginTier.TIER_3

    def get_initial_margin_requirement(self, symbol: str) -> float:
        """Get initial margin requirement for a symbol"""
        tier = self.get_margin_tier(symbol)

        if tier == MarginTier.TIER_1:
            return self.config.initial_margin_tier1
        elif tier == MarginTier.TIER_2:
            return self.config.initial_margin_tier2
        elif tier == MarginTier.TIER_3:
            return self.config.initial_margin_tier3
        else:
            return 1.0  # 100% - no margin available

    def calculate_max_buyable_shares(
        self, symbol: str, price: float, available_cash: float, use_margin: bool = True
    ) -> Tuple[int, float, float]:
        """
        Calculate maximum shares buyable with available margin

        Args:
            symbol: Stock symbol
            price: Current price
            available_cash: Cash available for trading
            use_margin: Whether to use margin

        Returns:
            (max_shares, loan_amount, equity_required)
        """
        if not use_margin:
            max_shares = int(available_cash / price)
            return (max_shares // 100) * 100, 0, max_shares * price

        initial_margin = self.get_initial_margin_requirement(symbol)

        # With 50% margin, can buy 2x cash value
        # With 60% margin, can buy 1.67x cash value
        # With 70% margin, can buy 1.43x cash value
        leverage = 1 / initial_margin
        max_utilization = self.config.max_margin_utilization

        max_buying_power = available_cash * leverage * max_utilization
        max_shares = int(max_buying_power / price)
        max_shares = (max_shares // 100) * 100  # Round to lot size

        total_value = max_shares * price
        equity_required = total_value * initial_margin
        loan_amount = total_value - equity_required

        return max_shares, loan_amount, equity_required

    def add_position(
        self, symbol: str, shares: int, avg_cost: float, current_price: float, loan_amount: float
    ) -> MarginPosition:
        """Add or update a margin position"""
        tier = self.get_margin_tier(symbol)

        position = MarginPosition(
            symbol=symbol,
            shares=shares,
            avg_cost=avg_cost,
            current_price=current_price,
            margin_tier=tier,
            loan_amount=loan_amount,
        )

        self.positions[symbol] = position
        return position

    def update_price(self, symbol: str, current_price: float) -> Optional[MarginPosition]:
        """Update current price for a position"""
        if symbol not in self.positions:
            return None

        self.positions[symbol].current_price = current_price
        return self.positions[symbol]

    def check_margin_status(self, symbol: str) -> Tuple[MarginStatus, Dict]:
        """
        Check margin status for a position

        Returns:
            (status, details_dict)
        """
        if symbol not in self.positions:
            return MarginStatus.SAFE, {"error": "Position not found"}

        position = self.positions[symbol]
        equity_ratio = position.equity_ratio

        details = {
            "symbol": symbol,
            "equity_ratio": equity_ratio,
            "equity": position.equity,
            "loan": position.loan_amount,
            "market_value": position.market_value,
            "unrealized_pnl_pct": position.unrealized_pnl_pct,
        }

        if equity_ratio < self.config.critical_threshold:
            status = MarginStatus.CRITICAL
            details["action"] = "IMMEDIATE_LIQUIDATION"
            details["message"] = (
                f"Critical: Equity {equity_ratio:.1%} below {self.config.critical_threshold:.0%}"
            )
        elif equity_ratio < self.config.force_sell_threshold:
            status = MarginStatus.FORCE_SELL
            details["action"] = "FORCE_SELL_INITIATED"
            details["message"] = (
                f"Force sell: Equity {equity_ratio:.1%} below {self.config.force_sell_threshold:.0%}"
            )
        elif equity_ratio < self.config.margin_call_threshold:
            status = MarginStatus.MARGIN_CALL
            details["action"] = "DEPOSIT_OR_REDUCE"
            details["message"] = (
                f"Margin call: Equity {equity_ratio:.1%} below {self.config.margin_call_threshold:.0%}"
            )
            details["shortfall"] = self._calculate_shortfall(position)
        elif equity_ratio < self.config.warning_threshold:
            status = MarginStatus.WARNING
            details["action"] = "MONITOR_CLOSELY"
            details["message"] = f"Warning: Equity {equity_ratio:.1%} approaching margin call"
        else:
            status = MarginStatus.SAFE
            details["action"] = "NONE"
            details["message"] = f"Safe: Equity ratio {equity_ratio:.1%}"

        return status, details

    def _calculate_shortfall(self, position: MarginPosition) -> float:
        """Calculate amount needed to restore maintenance margin"""
        required_equity = position.market_value * self.config.maintenance_margin
        current_equity = position.equity
        return max(0, required_equity - current_equity)

    def check_all_positions(self) -> MarginAccountSummary:
        """Check margin status for all positions"""
        total_equity = 0
        total_market_value = 0
        total_loan = 0
        positions_at_risk = []
        active_margin_calls = []

        for symbol, position in self.positions.items():
            total_equity += position.equity
            total_market_value += position.market_value
            total_loan += position.loan_amount

            status, details = self.check_margin_status(symbol)

            if status in [
                MarginStatus.WARNING,
                MarginStatus.MARGIN_CALL,
                MarginStatus.FORCE_SELL,
                MarginStatus.CRITICAL,
            ]:
                positions_at_risk.append(symbol)

            if status == MarginStatus.MARGIN_CALL:
                # Create margin call event
                call_event = MarginCallEvent(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    equity_ratio=position.equity_ratio,
                    required_equity_ratio=self.config.margin_call_threshold,
                    shortfall_amount=details.get("shortfall", 0),
                    deadline=datetime.now()
                    + timedelta(hours=self.config.margin_call_response_hours),
                    status="PENDING",
                )
                active_margin_calls.append(call_event)

        # Calculate overall status
        if total_market_value > 0:
            overall_equity_ratio = total_equity / total_market_value
            margin_utilization = total_loan / total_market_value if total_market_value > 0 else 0
        else:
            overall_equity_ratio = 1.0
            margin_utilization = 0

        if overall_equity_ratio < self.config.critical_threshold:
            overall_status = MarginStatus.CRITICAL
        elif overall_equity_ratio < self.config.force_sell_threshold:
            overall_status = MarginStatus.FORCE_SELL
        elif overall_equity_ratio < self.config.margin_call_threshold:
            overall_status = MarginStatus.MARGIN_CALL
        elif overall_equity_ratio < self.config.warning_threshold:
            overall_status = MarginStatus.WARNING
        else:
            overall_status = MarginStatus.SAFE

        # Calculate available margin
        max_loan = total_market_value * (1 - self.config.maintenance_margin)
        available_margin = max(0, max_loan - total_loan)

        return MarginAccountSummary(
            total_equity=total_equity,
            total_market_value=total_market_value,
            total_loan=total_loan,
            available_margin=available_margin,
            margin_utilization=margin_utilization,
            overall_status=overall_status,
            positions_at_risk=positions_at_risk,
            margin_calls=active_margin_calls,
        )

    def suggest_reduction(self, symbol: str, target_equity_ratio: float = 0.50) -> Dict:
        """
        Suggest position reduction to achieve target equity ratio

        Args:
            symbol: Position symbol
            target_equity_ratio: Target equity ratio (default 50%)

        Returns:
            Dict with reduction recommendation
        """
        if symbol not in self.positions:
            return {"error": "Position not found"}

        position = self.positions[symbol]
        current_equity_ratio = position.equity_ratio

        if current_equity_ratio >= target_equity_ratio:
            return {
                "symbol": symbol,
                "action": "NO_REDUCTION_NEEDED",
                "current_ratio": current_equity_ratio,
                "target_ratio": target_equity_ratio,
            }

        # Calculate shares to sell to achieve target ratio
        # After selling S shares at price P:
        # New equity = old_equity + S*P (selling repays loan)
        # New market value = old_market_value - S*P
        # Target: (old_equity + S*P) / (old_market_value - S*P) = target_ratio

        P = position.current_price
        E = position.equity
        MV = position.market_value
        target = target_equity_ratio

        # Solve: (E + SP) / (MV - SP) = target
        # E + SP = target * MV - target * SP
        # SP + target * SP = target * MV - E
        # SP * (1 + target) = target * MV - E
        # S = (target * MV - E) / (P * (1 + target))

        shares_to_sell = (target * MV - E) / (P * (1 + target))
        shares_to_sell = int(shares_to_sell)
        shares_to_sell = ((shares_to_sell // 100) + 1) * 100  # Round up to lot
        shares_to_sell = min(shares_to_sell, position.shares)  # Can't sell more than owned

        # Calculate new state after reduction
        proceeds = shares_to_sell * P
        new_loan = position.loan_amount - proceeds
        new_shares = position.shares - shares_to_sell
        new_market_value = new_shares * P
        new_equity = new_market_value - max(0, new_loan)
        new_equity_ratio = new_equity / new_market_value if new_market_value > 0 else 1.0

        return {
            "symbol": symbol,
            "action": "REDUCE_POSITION",
            "shares_to_sell": shares_to_sell,
            "current_shares": position.shares,
            "remaining_shares": new_shares,
            "current_equity_ratio": current_equity_ratio,
            "new_equity_ratio": new_equity_ratio,
            "target_equity_ratio": target_equity_ratio,
            "estimated_proceeds": proceeds,
            "loan_repayment": min(proceeds, position.loan_amount),
            "urgency": "HIGH" if current_equity_ratio < 0.35 else "MEDIUM",
        }

    def handle_margin_call(self, symbol: str, action: str, amount: float = 0) -> Dict:
        """
        Handle a margin call with specified action

        Args:
            symbol: Position symbol
            action: "DEPOSIT", "REDUCE", or "CLOSE"
            amount: Amount to deposit (for DEPOSIT action)

        Returns:
            Result of margin call handling
        """
        if symbol not in self.positions:
            return {"error": "Position not found"}

        position = self.positions[symbol]

        if action == "DEPOSIT":
            # Simulate deposit - increases equity, reduces loan
            new_loan = position.loan_amount - amount
            position.loan_amount = max(0, new_loan)

            new_status, details = self.check_margin_status(symbol)

            return {
                "action": "DEPOSIT",
                "amount": amount,
                "new_equity_ratio": position.equity_ratio,
                "new_status": new_status.value,
                "margin_call_resolved": new_status in [MarginStatus.SAFE, MarginStatus.WARNING],
            }

        elif action == "REDUCE":
            reduction = self.suggest_reduction(symbol)
            return {"action": "REDUCE_SUGGESTED", "recommendation": reduction}

        elif action == "CLOSE":
            # Close entire position
            proceeds = position.market_value
            loan_repayment = min(proceeds, position.loan_amount)
            remaining_cash = proceeds - loan_repayment

            # Remove position
            del self.positions[symbol]

            return {
                "action": "POSITION_CLOSED",
                "proceeds": proceeds,
                "loan_repaid": loan_repayment,
                "cash_returned": remaining_cash,
            }

        return {"error": f"Unknown action: {action}"}

    def calculate_margin_interest(self, days: int = 1) -> Dict[str, float]:
        """Calculate daily margin interest for all positions"""
        daily_rate = self.config.margin_interest_rate / 365

        interest_by_symbol = {}
        total_interest = 0

        for symbol, position in self.positions.items():
            if position.loan_amount > 0:
                interest = position.loan_amount * daily_rate * days
                interest_by_symbol[symbol] = interest
                total_interest += interest

        return {
            "by_symbol": interest_by_symbol,
            "total": total_interest,
            "daily_rate": daily_rate,
            "annual_rate": self.config.margin_interest_rate,
            "days": days,
        }

    def get_margin_health_score(self) -> float:
        """
        Calculate overall margin health score (0-100)

        Factors:
        - Average equity ratio across positions
        - Number of positions at risk
        - Margin utilization level
        """
        if not self.positions:
            return 100.0

        summary = self.check_all_positions()

        # Base score from overall equity ratio
        if summary.total_market_value > 0:
            equity_ratio = summary.total_equity / summary.total_market_value
        else:
            equity_ratio = 1.0

        base_score = min(100, equity_ratio * 200)  # 50% ratio = 100 score

        # Penalty for positions at risk
        risk_penalty = len(summary.positions_at_risk) * 10

        # Penalty for high margin utilization
        util_penalty = max(0, (summary.margin_utilization - 0.5) * 50)

        # Penalty for margin calls
        call_penalty = len(summary.margin_calls) * 20

        final_score = max(0, base_score - risk_penalty - util_penalty - call_penalty)
        return round(final_score, 1)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_margin_manager: Optional[MarginTradingManager] = None


def get_margin_manager(config: Optional[MarginConfig] = None) -> MarginTradingManager:
    """Get singleton margin manager instance"""
    global _margin_manager
    if _margin_manager is None:
        _margin_manager = MarginTradingManager(config)
    return _margin_manager


def check_margin_before_trade(
    symbol: str, shares: int, price: float, available_cash: float, use_margin: bool = True
) -> Dict:
    """
    Pre-trade margin check

    Returns recommendation on whether trade is safe from margin perspective
    """
    manager = get_margin_manager()

    if not use_margin:
        can_afford = available_cash >= shares * price
        return {
            "approved": can_afford,
            "reason": "Cash trade" if can_afford else "Insufficient cash",
            "margin_required": 0,
            "loan_amount": 0,
        }

    initial_margin = manager.get_initial_margin_requirement(symbol)
    trade_value = shares * price
    equity_required = trade_value * initial_margin
    loan_needed = trade_value - equity_required

    # Check if we have enough cash for initial margin
    can_afford = available_cash >= equity_required

    # Check concentration limit
    summary = manager.check_all_positions()
    concentration_ok = (
        trade_value / (summary.total_market_value + trade_value)
    ) <= manager.config.position_concentration_limit

    # Check overall utilization
    new_total_loan = summary.total_loan + loan_needed
    new_total_mv = summary.total_market_value + trade_value
    new_utilization = new_total_loan / new_total_mv if new_total_mv > 0 else 0
    utilization_ok = new_utilization <= manager.config.max_margin_utilization

    approved = can_afford and concentration_ok and utilization_ok

    reasons = []
    if not can_afford:
        reasons.append(f"Need {equity_required:,.0f} VND equity, have {available_cash:,.0f}")
    if not concentration_ok:
        reasons.append("Would exceed position concentration limit")
    if not utilization_ok:
        reasons.append(f"Would exceed max margin utilization ({new_utilization:.1%})")

    return {
        "approved": approved,
        "reason": "; ".join(reasons) if reasons else "Trade approved",
        "margin_tier": manager.get_margin_tier(symbol).value,
        "initial_margin_pct": initial_margin,
        "equity_required": equity_required,
        "loan_amount": loan_needed,
        "new_utilization": new_utilization,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Margin Trading Module...\n")

    # Create manager
    manager = MarginTradingManager()

    # Test tier classification
    print("Margin Tiers:")
    for symbol in ["VNM", "FPT", "DGC", "ABC"]:
        tier = manager.get_margin_tier(symbol)
        margin = manager.get_initial_margin_requirement(symbol)
        print(f"  {symbol}: {tier.value} - {margin:.0%} initial margin")

    # Test max buyable calculation
    print("\nMax Buyable Shares (100M VND cash):")
    for symbol in ["VNM", "FPT"]:
        shares, loan, equity = manager.calculate_max_buyable_shares(
            symbol, 100000, 100_000_000, use_margin=True
        )
        print(f"  {symbol}: {shares:,} shares, loan {loan:,.0f} VND, equity {equity:,.0f} VND")

    # Test position tracking
    print("\nPosition Tracking:")
    manager.add_position("VNM", 1000, 85000, 80000, 40_000_000)  # In loss
    manager.add_position("FPT", 500, 100000, 110000, 25_000_000)  # In profit

    summary = manager.check_all_positions()
    print(f"  Total equity: {summary.total_equity:,.0f} VND")
    print(f"  Total loan: {summary.total_loan:,.0f} VND")
    print(f"  Margin utilization: {summary.margin_utilization:.1%}")
    print(f"  Status: {summary.overall_status.value}")

    # Test margin call scenario
    print("\nMargin Call Scenario:")
    manager.add_position("HPG", 2000, 30000, 20000, 35_000_000)  # Heavy loss
    status, details = manager.check_margin_status("HPG")
    print(f"  HPG status: {status.value}")
    print(f"  Details: {details['message']}")

    if status == MarginStatus.MARGIN_CALL:
        reduction = manager.suggest_reduction("HPG")
        print(f"  Suggested: Sell {reduction['shares_to_sell']} shares")

    # Test health score
    print(f"\nMargin Health Score: {manager.get_margin_health_score()}/100")

    print("\n✅ Margin Trading Module test completed!")
