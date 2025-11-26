# -*- coding: utf-8 -*-
"""
T+2 Settlement Timing Awareness
Tối ưu hóa timing cho vào lệnh dựa trên T+2 settlement cycle của Vietnam
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SettlementPhase(Enum):
    """Giai đoạn trong T+2 settlement cycle"""
    T0_OPTIMAL = "T0_OPTIMAL"  # Best day to buy (settlement in 2 days)
    T0_GOOD = "T0_GOOD"  # Good day to buy
    T0_CAUTION = "T0_CAUTION"  # Can buy but need cash reserve
    T1_RESERVED = "T1_RESERVED"  # T+1 - cash reserved, limited buying
    T2_SETTLEMENT = "T2_SETTLEMENT"  # T+2 - settlement day, very limited buying


@dataclass
class SettlementAnalysis:
    """Kết quả phân tích settlement timing"""

    phase: SettlementPhase
    days_until_next_settlement: int  # Days until next settlement
    pending_settlements: List[Dict]  # List of pending settlements
    total_pending_amount: float  # Total VND pending settlement
    available_buying_power: float  # Available cash for new trades

    can_trade: bool  # Can make new trades
    recommended_position_reduction: float  # % reduction (0-1)
    confidence_adjustment: int  # -10 to +5

    reasons: list = None
    warnings: list = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.warnings is None:
            self.warnings = []


class SettlementTimingAnalyzer:
    """
    Phân tích timing cho vào lệnh dựa trên T+2 settlement

    VIETNAM MARKET SPECIFICS:
    - T+2 settlement: Buy on T0, pay on T+2
    - Need to reserve cash for T+1 and T+2 settlements
    - Cannot use pending sell proceeds until T+2

    STRATEGY:
    - T0 (Monday): Best day - full 2 days until settlement
    - T0 (Tuesday-Thursday): Good - manage cash flow
    - T0 (Friday): CAUTION - weekend + 2 days = Tuesday settlement
    - Avoid buying when large settlements pending
    - Reserve cash for upcoming settlements
    """

    def __init__(
        self,
        settlement_days: int = 2,  # T+2 for Vietnam
        min_cash_reserve_pct: float = 0.30,  # Keep 30% cash for settlements
        max_pending_settlement_pct: float = 0.60,  # Max 60% pending
        friday_caution: bool = True,  # Extra caution for Friday trades
    ):
        """
        Args:
            settlement_days: Number of days for settlement (2 for Vietnam)
            min_cash_reserve_pct: Minimum cash to keep for settlements (0-1)
            max_pending_settlement_pct: Max % of capital in pending settlements
            friday_caution: Apply extra caution to Friday trades
        """
        self.settlement_days = settlement_days
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self.max_pending_settlement_pct = max_pending_settlement_pct
        self.friday_caution = friday_caution

    def analyze(
        self,
        total_capital: float,
        current_cash: float,
        portfolio_manager=None,
        current_date: Optional[datetime] = None,
    ) -> SettlementAnalysis:
        """
        Phân tích settlement timing và available buying power

        Args:
            total_capital: Total portfolio capital
            current_cash: Current available cash
            portfolio_manager: Portfolio manager instance (for pending settlements)
            current_date: Current date (default: today)

        Returns:
            SettlementAnalysis object
        """
        if current_date is None:
            current_date = datetime.now()

        reasons = []
        warnings = []

        try:
            # 1. GET PENDING SETTLEMENTS
            pending_settlements = self._get_pending_settlements(
                portfolio_manager, current_date
            )

            # 2. CALCULATE TOTAL PENDING AMOUNT
            total_pending = sum(
                settlement.get('amount', 0) for settlement in pending_settlements
            )

            # 3. DETERMINE SETTLEMENT PHASE
            phase = self._determine_phase(
                current_date, pending_settlements, total_pending, total_capital
            )

            # 4. CALCULATE DAYS UNTIL NEXT SETTLEMENT
            days_until_next = self._days_until_next_settlement(
                pending_settlements, current_date
            )

            # 5. CALCULATE AVAILABLE BUYING POWER
            # Need to reserve cash for:
            # - Upcoming settlements (pending buys)
            # - Minimum cash reserve
            min_cash_reserve = total_capital * self.min_cash_reserve_pct
            cash_for_settlements = total_pending
            required_reserve = max(min_cash_reserve, cash_for_settlements)

            available_buying_power = max(0, current_cash - required_reserve)

            # 6. CHECK IF CAN TRADE
            pending_pct = total_pending / total_capital if total_capital > 0 else 0
            can_trade = (
                pending_pct < self.max_pending_settlement_pct and
                available_buying_power > 0 and
                phase != SettlementPhase.T2_SETTLEMENT
            )

            # 7. RECOMMENDED POSITION SIZE REDUCTION
            reduction = self._calculate_reduction(phase, pending_pct)

            # 8. CONFIDENCE ADJUSTMENT
            confidence_adj = self._calculate_confidence_adjustment(
                phase, pending_pct, available_buying_power, total_capital
            )

            # 9. BUILD MESSAGES
            self._build_messages(
                reasons, warnings, phase, pending_settlements, total_pending,
                available_buying_power, total_capital, days_until_next
            )

            return SettlementAnalysis(
                phase=phase,
                days_until_next_settlement=days_until_next,
                pending_settlements=pending_settlements,
                total_pending_amount=total_pending,
                available_buying_power=available_buying_power,
                can_trade=can_trade,
                recommended_position_reduction=reduction,
                confidence_adjustment=confidence_adj,
                reasons=reasons,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Error in settlement timing analysis: {e}", exc_info=True)
            return self._default_result()

    def _get_pending_settlements(
        self,
        portfolio_manager,
        current_date: datetime
    ) -> List[Dict]:
        """
        Get pending settlements from portfolio manager

        Returns: List of pending settlements with date and amount
        """
        pending = []

        if portfolio_manager is None:
            return pending

        try:
            # Get all positions
            positions = portfolio_manager.get_positions()

            for symbol, pos in positions.items():
                # Get entry date
                entry_date_str = pos.get('entry_date')
                if not entry_date_str:
                    continue

                entry_date = datetime.fromisoformat(entry_date_str)

                # Calculate settlement date (T+2)
                settlement_date = entry_date + timedelta(days=self.settlement_days)

                # If settlement is in future, add to pending
                if settlement_date > current_date:
                    amount = pos.get('shares', 0) * pos.get('avg_price', 0)
                    pending.append({
                        'symbol': symbol,
                        'entry_date': entry_date,
                        'settlement_date': settlement_date,
                        'amount': amount,
                        'days_remaining': (settlement_date - current_date).days
                    })

            # Sort by settlement date
            pending.sort(key=lambda x: x['settlement_date'])

        except Exception as e:
            logger.error(f"Error getting pending settlements: {e}")

        return pending

    def _determine_phase(
        self,
        current_date: datetime,
        pending_settlements: List[Dict],
        total_pending: float,
        total_capital: float
    ) -> SettlementPhase:
        """
        Determine current settlement phase

        Logic:
        - Monday-Thursday with low pending: OPTIMAL
        - Monday-Thursday with medium pending: GOOD
        - Friday or high pending: CAUTION
        - Tomorrow = settlement day: T1_RESERVED
        - Today = settlement day: T2_SETTLEMENT
        """
        # Check if today is settlement day
        if self._is_settlement_day(pending_settlements, current_date):
            return SettlementPhase.T2_SETTLEMENT

        # Check if tomorrow is settlement day
        tomorrow = current_date + timedelta(days=1)
        if self._is_settlement_day(pending_settlements, tomorrow):
            return SettlementPhase.T1_RESERVED

        # Calculate pending %
        pending_pct = total_pending / total_capital if total_capital > 0 else 0

        # Day of week check
        day_of_week = current_date.weekday()  # 0=Monday, 4=Friday

        # Friday check
        if self.friday_caution and day_of_week == 4:  # Friday
            return SettlementPhase.T0_CAUTION

        # Pending amount check
        if pending_pct >= 0.50:
            return SettlementPhase.T0_CAUTION
        elif pending_pct >= 0.30:
            return SettlementPhase.T0_GOOD
        else:
            return SettlementPhase.T0_OPTIMAL

    def _is_settlement_day(
        self,
        pending_settlements: List[Dict],
        check_date: datetime
    ) -> bool:
        """Check if check_date is a settlement day"""
        for settlement in pending_settlements:
            settlement_date = settlement['settlement_date']
            if settlement_date.date() == check_date.date():
                return True
        return False

    def _days_until_next_settlement(
        self,
        pending_settlements: List[Dict],
        current_date: datetime
    ) -> int:
        """Calculate days until next settlement"""
        if not pending_settlements:
            return 999  # No pending settlements

        # Get earliest settlement
        next_settlement = pending_settlements[0]['settlement_date']
        days = (next_settlement - current_date).days

        return max(0, days)

    def _calculate_reduction(
        self,
        phase: SettlementPhase,
        pending_pct: float
    ) -> float:
        """
        Calculate recommended position size reduction

        Returns: 0.0 (no reduction) to 1.0 (100% reduction)
        """
        # Base reduction by phase
        base_reduction = {
            SettlementPhase.T0_OPTIMAL: 0.0,  # No reduction
            SettlementPhase.T0_GOOD: 0.10,  # 10% reduction
            SettlementPhase.T0_CAUTION: 0.30,  # 30% reduction
            SettlementPhase.T1_RESERVED: 0.50,  # 50% reduction
            SettlementPhase.T2_SETTLEMENT: 1.0,  # Block new trades
        }[phase]

        # Additional reduction based on pending %
        if pending_pct >= 0.50:
            base_reduction += 0.20
        elif pending_pct >= 0.40:
            base_reduction += 0.10

        return min(1.0, base_reduction)

    def _calculate_confidence_adjustment(
        self,
        phase: SettlementPhase,
        pending_pct: float,
        available_buying_power: float,
        total_capital: float
    ) -> int:
        """
        Calculate confidence adjustment

        Returns: -10 to +5
        """
        # Base adjustment by phase
        adjustment = {
            SettlementPhase.T0_OPTIMAL: +5,  # Bonus for optimal timing
            SettlementPhase.T0_GOOD: 0,  # Neutral
            SettlementPhase.T0_CAUTION: -5,  # Penalty for caution
            SettlementPhase.T1_RESERVED: -8,  # Strong penalty
            SettlementPhase.T2_SETTLEMENT: -10,  # Maximum penalty
        }[phase]

        # Additional penalty for high pending
        if pending_pct >= 0.50:
            adjustment -= 5

        # Bonus for high buying power
        buying_power_pct = available_buying_power / total_capital if total_capital > 0 else 0
        if buying_power_pct >= 0.50:
            adjustment += 3

        return max(-10, min(5, adjustment))

    def _build_messages(
        self,
        reasons: list,
        warnings: list,
        phase: SettlementPhase,
        pending_settlements: List[Dict],
        total_pending: float,
        available_buying_power: float,
        total_capital: float,
        days_until_next: int
    ):
        """Build reason and warning messages"""

        # Phase message
        if phase == SettlementPhase.T0_OPTIMAL:
            reasons.append("✅ Optimal settlement timing (low pending, not Friday)")
        elif phase == SettlementPhase.T0_GOOD:
            reasons.append("✅ Good settlement timing")
        elif phase == SettlementPhase.T0_CAUTION:
            warnings.append("⚠️ Settlement caution (Friday or high pending)")
        elif phase == SettlementPhase.T1_RESERVED:
            warnings.append("🔶 T+1 phase - cash reserved for tomorrow's settlement")
        elif phase == SettlementPhase.T2_SETTLEMENT:
            warnings.append("🚫 T+2 settlement day - avoid new trades")

        # Pending amount
        pending_pct = total_pending / total_capital if total_capital > 0 else 0
        if total_pending > 0:
            if pending_pct >= 0.50:
                warnings.append(
                    f"⚠️ High pending settlements: {total_pending:,.0f} VND "
                    f"({pending_pct:.0%})"
                )
            else:
                reasons.append(
                    f"ℹ️ Pending settlements: {total_pending:,.0f} VND "
                    f"({pending_pct:.0%})"
                )

        # Buying power
        buying_power_pct = available_buying_power / total_capital if total_capital > 0 else 0
        if buying_power_pct >= 0.40:
            reasons.append(
                f"✅ Good buying power: {available_buying_power:,.0f} VND "
                f"({buying_power_pct:.0%})"
            )
        elif buying_power_pct < 0.10:
            warnings.append(
                f"⚠️ Low buying power: {available_buying_power:,.0f} VND "
                f"({buying_power_pct:.0%})"
            )

        # Next settlement
        if days_until_next <= 1:
            warnings.append(f"🔔 Next settlement in {days_until_next} day(s)")

    def _default_result(self) -> SettlementAnalysis:
        """Return default result on error"""
        return SettlementAnalysis(
            phase=SettlementPhase.T0_CAUTION,
            days_until_next_settlement=999,
            pending_settlements=[],
            total_pending_amount=0,
            available_buying_power=0,
            can_trade=True,
            recommended_position_reduction=0.0,
            confidence_adjustment=0,
            reasons=[],
            warnings=["⚠️ Could not analyze settlement timing"],
        )


# Singleton instance
_settlement_analyzer = None


def get_settlement_analyzer() -> SettlementTimingAnalyzer:
    """Get singleton instance"""
    global _settlement_analyzer
    if _settlement_analyzer is None:
        _settlement_analyzer = SettlementTimingAnalyzer()
    return _settlement_analyzer
