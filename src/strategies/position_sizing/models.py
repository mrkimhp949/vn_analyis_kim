# -*- coding: utf-8 -*-
"""
Position Sizing Models

Data classes for position sizing results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config.constants import VIETNAM_LOT_SIZE


@dataclass
class EnhancedPositionSize:
    """Container cho kết quả position sizing với Kelly."""

    shares: int
    value: float
    risk_amount: float
    risk_percent: float
    max_loss: float
    position_percent: float
    kelly_percent: float
    recommended_entries: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adjustments: Dict[str, float] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if position is valid for trading."""
        return self.shares > 0 and self.shares >= VIETNAM_LOT_SIZE


@dataclass
class MarketRegimeInfo:
    """Structured market regime information."""

    regime: str = "SIDEWAYS"
    confidence: float = 50.0
    tradeable: bool = True
    description: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "MarketRegimeInfo":
        """Create from dictionary."""
        if not data:
            return cls()
        return cls(
            regime=data.get("regime", "SIDEWAYS"),
            confidence=data.get("confidence", 50.0),
            tradeable=data.get("tradeable", True),
            description=data.get("description", ""),
        )


# Alias for backward compatibility
PositionSize = EnhancedPositionSize
