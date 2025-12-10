# -*- coding: utf-8 -*-
"""
Entry Signal Module - Data classes for entry signal representation.

This module contains:
- SignalStrength enum for signal classification
- EntrySignal dataclass for comprehensive entry signal data

Extracted from entry_logic.py for better modularity.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalStrength(Enum):
    """Signal strength classification for position sizing decisions."""

    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NO_SIGNAL = 0

    @classmethod
    def from_score(cls, score: int) -> "SignalStrength":
        """
        Convert numeric score to SignalStrength.

        Args:
            score: Numeric score (0-5+)

        Returns:
            Corresponding SignalStrength enum value
        """
        if score >= 5:
            return cls.VERY_STRONG
        elif score >= 4:
            return cls.STRONG
        elif score >= 3:
            return cls.MODERATE
        elif score >= 2:
            return cls.WEAK
        elif score >= 1:
            return cls.VERY_WEAK
        else:
            return cls.NO_SIGNAL

    def to_position_multiplier(self) -> float:
        """
        Convert signal strength to position size multiplier.

        Returns:
            Multiplier for position sizing (0.0 - 1.5)
        """
        multipliers = {
            SignalStrength.VERY_STRONG: 1.5,
            SignalStrength.STRONG: 1.2,
            SignalStrength.MODERATE: 1.0,
            SignalStrength.WEAK: 0.7,
            SignalStrength.VERY_WEAK: 0.5,
            SignalStrength.NO_SIGNAL: 0.0,
        }
        return multipliers.get(self, 1.0)


@dataclass
class EntrySignal:
    """
    Container for entry signal analysis results.

    Attributes:
        should_enter: Whether to enter the position
        signal_type: Signal direction ('BUY', 'SELL', 'HOLD')
        confidence: Confidence score 0-100
        strength: Signal strength classification
        position_size_multiplier: Position size adjustment (0.0-1.5)
        position_multiplier: Alias for position_size_multiplier (used by entry_logic)
        reasons: List of positive factors supporting entry
        warnings: List of risk factors or concerns
        entry_price: Recommended entry price
        stop_loss: Stop loss price level
        take_profit_targets: List of take profit price levels
        risk_reward: Calculated risk/reward ratio
        is_limit_order: Whether to use limit order
        limit_price: Limit price if using limit order
        entry_type: Entry strategy type
        adjustment_breakdown: Detailed confidence adjustment breakdown
        telemetry: Detailed scoring breakdown for debugging
    """

    should_enter: bool
    signal_type: str
    confidence: int
    strength: SignalStrength
    reasons: List[str]
    warnings: List[str]
    entry_price: float
    stop_loss: float
    take_profit_targets: List[float]
    # Optional fields with defaults
    position_size_multiplier: float = 1.0
    position_multiplier: float = 1.0  # Alias used by entry_logic
    risk_reward: float = 0.0
    is_limit_order: bool = False
    limit_price: Optional[float] = None
    entry_type: str = "MARKET"
    adjustment_breakdown: Optional[List[Dict]] = field(default=None)
    telemetry: Optional[Dict] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "should_enter": self.should_enter,
            "signal_type": self.signal_type,
            "confidence": self.confidence,
            "strength": self.strength.name if self.strength else "UNKNOWN",
            "strength_value": self.strength.value if self.strength else 0,
            "position_size_multiplier": self.position_size_multiplier or self.position_multiplier,
            "position_multiplier": self.position_multiplier,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_targets": self.take_profit_targets,
            "is_limit_order": self.is_limit_order,
            "limit_price": self.limit_price,
            "entry_type": self.entry_type,
            "risk_reward": self.risk_reward,
            "risk_reward_ratio": self.get_risk_reward_ratio(),
        }

    def get_risk_reward_ratio(self) -> float:
        """
        Calculate risk/reward ratio from entry, stop loss and first take profit.

        Returns:
            Risk/reward ratio (e.g., 2.0 means reward is 2x risk)
        """
        # Use stored risk_reward if available
        if self.risk_reward > 0:
            return self.risk_reward

        if not self.take_profit_targets or self.entry_price <= 0:
            return 0.0

        risk = self.entry_price - self.stop_loss
        if risk <= 0:
            return 0.0

        # Use first take profit target for R:R calculation
        reward = self.take_profit_targets[0] - self.entry_price
        return round(reward / risk, 2) if risk > 0 else 0.0

    def is_high_confidence(self) -> bool:
        """
        Check if this is a high confidence signal worth taking.

        High confidence criteria:
        - Should enter is True
        - Confidence >= 70
        - Strength is STRONG or VERY_STRONG
        - Max 2 warnings

        Returns:
            True if signal meets high confidence criteria
        """
        return (
            self.should_enter
            and self.confidence >= 70
            and self.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]
            and len(self.warnings) <= 2
        )

    def is_actionable(self) -> bool:
        """
        Check if signal is actionable (should enter with positive confidence).

        Returns:
            True if signal can be acted upon
        """
        return self.should_enter and self.confidence > 0 and self.entry_price > 0

    def get_stop_loss_percent(self) -> float:
        """
        Calculate stop loss as percentage below entry.

        Returns:
            Stop loss percentage (e.g., 5.0 for 5%)
        """
        if self.entry_price <= 0:
            return 0.0
        return round((self.entry_price - self.stop_loss) / self.entry_price * 100, 2)

    def get_first_target_percent(self) -> float:
        """
        Calculate first take profit target as percentage above entry.

        Returns:
            Take profit percentage (e.g., 10.0 for 10%)
        """
        if not self.take_profit_targets or self.entry_price <= 0:
            return 0.0
        return round((self.take_profit_targets[0] - self.entry_price) / self.entry_price * 100, 2)

    def summary(self) -> str:
        """
        Generate human-readable summary of the signal.

        Returns:
            Formatted summary string
        """
        if not self.should_enter:
            return f"❌ NO ENTRY - {self.signal_type}"

        emoji = "🟢" if self.is_high_confidence() else "🟡"
        return (
            f"{emoji} {self.signal_type} | "
            f"Conf: {self.confidence}% | "
            f"Entry: {self.entry_price:,.0f} | "
            f"SL: {self.stop_loss:,.0f} ({self.get_stop_loss_percent():.1f}%) | "
            f"R:R: {self.get_risk_reward_ratio():.1f} | "
            f"Warnings: {len(self.warnings)}"
        )


def create_no_signal(reason: str, telemetry: Optional[Dict] = None) -> EntrySignal:
    """
    Factory function to create a "no entry" signal.

    Args:
        reason: Reason for not entering
        telemetry: Optional telemetry data

    Returns:
        EntrySignal with should_enter=False
    """
    return EntrySignal(
        should_enter=False,
        signal_type="HOLD",
        confidence=0,
        strength=SignalStrength.NO_SIGNAL,
        reasons=[],
        warnings=[reason],
        entry_price=0.0,
        stop_loss=0.0,
        take_profit_targets=[],
        position_size_multiplier=0.0,
        position_multiplier=0.0,
        risk_reward=0.0,
        is_limit_order=False,
        limit_price=None,
        entry_type="NONE",
        adjustment_breakdown=None,
        telemetry=telemetry,
    )
