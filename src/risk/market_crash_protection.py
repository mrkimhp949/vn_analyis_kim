# -*- coding: utf-8 -*-
"""
Market Crash Protection Module - Priority 1 Improvement

Real-time VNINDEX monitoring and automatic exposure reduction during market crashes.

Features:
- Real-time VNINDEX change monitoring
- Automatic position reduction when market drops > 5%
- Full exit when market drops > 7%
- Integration with PortfolioManager for position management
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MarketCrashStatus:
    """Market crash protection status"""

    is_crash_mode: bool
    vnindex_change_pct: float
    protection_level: str  # NORMAL, WARNING, CAUTION, CRASH
    recommended_action: str
    exposure_reduction_pct: float  # 0.0-1.0 (how much to reduce)
    message: str
    timestamp: datetime


class MarketCrashProtector:
    """
    Market Crash Protection System

    Monitors VNINDEX in real-time and automatically reduces portfolio exposure
    during market crashes to protect capital.

    Protection Levels:
    - NORMAL: VNINDEX > -2% → No action
    - WARNING: VNINDEX -2% to -5% → Monitor closely, reduce new positions
    - CAUTION: VNINDEX -5% to -7% → Reduce all positions by 50%
    - CRASH: VNINDEX < -7% → Exit all positions immediately
    """

    # Protection thresholds
    WARNING_THRESHOLD = -2.0  # -2% VNINDEX: Warning level
    CAUTION_THRESHOLD = -5.0  # -5% VNINDEX: Reduce exposure by 50%
    CRASH_THRESHOLD = -7.0  # -7% VNINDEX: Exit all positions

    def __init__(self):
        self._last_vnindex_value: Optional[float] = None
        self._last_check_time: Optional[datetime] = None
        self._crash_mode_active: bool = False

    def check_market_status(
        self, vnindex_current: float, vnindex_previous: Optional[float] = None
    ) -> MarketCrashStatus:
        """
        Check current market status and return protection recommendations.

        Args:
            vnindex_current: Current VNINDEX value
            vnindex_previous: Previous VNINDEX value (for change calculation)

        Returns:
            MarketCrashStatus with recommendations
        """
        # Calculate VNINDEX change
        if vnindex_previous is None:
            vnindex_previous = self._last_vnindex_value

        if vnindex_previous is None or vnindex_previous <= 0:
            # No previous data - assume normal
            return MarketCrashStatus(
                is_crash_mode=False,
                vnindex_change_pct=0.0,
                protection_level="NORMAL",
                recommended_action="MONITOR",
                exposure_reduction_pct=0.0,
                message="✅ Market normal - no previous data for comparison",
                timestamp=datetime.now(),
            )

        vnindex_change_pct = ((vnindex_current - vnindex_previous) / vnindex_previous) * 100

        # Update cache
        self._last_vnindex_value = vnindex_current
        self._last_check_time = datetime.now()

        # Determine protection level
        if vnindex_change_pct >= self.WARNING_THRESHOLD:
            # NORMAL: Market is fine
            self._crash_mode_active = False
            return MarketCrashStatus(
                is_crash_mode=False,
                vnindex_change_pct=vnindex_change_pct,
                protection_level="NORMAL",
                recommended_action="MONITOR",
                exposure_reduction_pct=0.0,
                message=f"✅ Market normal: VNINDEX {vnindex_change_pct:+.2f}%",
                timestamp=datetime.now(),
            )

        elif vnindex_change_pct >= self.CAUTION_THRESHOLD:
            # WARNING: Market down 2-5% - monitor closely
            self._crash_mode_active = False
            return MarketCrashStatus(
                is_crash_mode=False,
                vnindex_change_pct=vnindex_change_pct,
                protection_level="WARNING",
                recommended_action="REDUCE_NEW_POSITIONS",
                exposure_reduction_pct=0.0,
                message=(
                    f"⚠️ Market warning: VNINDEX {vnindex_change_pct:+.2f}% - "
                    f"Reduce new position sizes by 50%"
                ),
                timestamp=datetime.now(),
            )

        elif vnindex_change_pct >= self.CRASH_THRESHOLD:
            # CAUTION: Market down 5-7% - reduce all positions by 50%
            self._crash_mode_active = True
            return MarketCrashStatus(
                is_crash_mode=True,
                vnindex_change_pct=vnindex_change_pct,
                protection_level="CAUTION",
                recommended_action="REDUCE_ALL_POSITIONS",
                exposure_reduction_pct=0.5,  # Reduce by 50%
                message=(
                    f"🟡 Market caution: VNINDEX {vnindex_change_pct:+.2f}% - "
                    f"REDUCE ALL POSITIONS BY 50%"
                ),
                timestamp=datetime.now(),
            )

        else:
            # CRASH: Market down > 7% - exit all positions
            self._crash_mode_active = True
            return MarketCrashStatus(
                is_crash_mode=True,
                vnindex_change_pct=vnindex_change_pct,
                protection_level="CRASH",
                recommended_action="EXIT_ALL_POSITIONS",
                exposure_reduction_pct=1.0,  # Exit 100%
                message=(
                    f"🚨 MARKET CRASH: VNINDEX {vnindex_change_pct:+.2f}% - "
                    f"EXIT ALL POSITIONS IMMEDIATELY"
                ),
                timestamp=datetime.now(),
            )

    def get_exposure_reduction_plan(
        self, status: MarketCrashStatus, current_positions: Dict
    ) -> List[Dict]:
        """
        Generate exposure reduction plan based on market crash status.

        Args:
            status: MarketCrashStatus from check_market_status
            current_positions: Dict of current positions {symbol: position_data}

        Returns:
            List of reduction actions [{symbol, action, shares_to_sell, reason}]
        """
        if status.exposure_reduction_pct == 0.0:
            return []  # No reduction needed

        reduction_plan = []

        for symbol, position in current_positions.items():
            shares = position.get("shares", 0)
            if shares <= 0:
                continue

            if status.exposure_reduction_pct >= 1.0:
                # Full exit
                reduction_plan.append(
                    {
                        "symbol": symbol,
                        "action": "EXIT_FULL",
                        "shares_to_sell": shares,
                        "reason": f"Market crash protection: {status.protection_level}",
                    }
                )
            else:
                # Partial reduction
                shares_to_sell = int(shares * status.exposure_reduction_pct)
                # Round to lot size (100 shares)
                shares_to_sell = (shares_to_sell // 100) * 100
                if shares_to_sell < 100:
                    shares_to_sell = 100  # Minimum 1 lot

                if shares_to_sell >= shares:
                    # If reduction would be full, just exit fully
                    reduction_plan.append(
                        {
                            "symbol": symbol,
                            "action": "EXIT_FULL",
                            "shares_to_sell": shares,
                            "reason": f"Market crash protection: {status.protection_level}",
                        }
                    )
                else:
                    reduction_plan.append(
                        {
                            "symbol": symbol,
                            "action": "REDUCE_PARTIAL",
                            "shares_to_sell": shares_to_sell,
                            "reason": f"Market crash protection: {status.protection_level}",
                        }
                    )

        return reduction_plan

    def should_block_new_entries(self, status: MarketCrashStatus) -> Tuple[bool, str]:
        """
        Check if new entries should be blocked based on market status.

        Args:
            status: MarketCrashStatus

        Returns:
            Tuple of (should_block, reason)
        """
        if status.protection_level == "CRASH":
            return (
                True,
                f"🚫 Market crash: VNINDEX {status.vnindex_change_pct:+.2f}% - Blocking all new entries",
            )

        if status.protection_level == "CAUTION":
            return (
                True,
                f"⚠️ Market caution: VNINDEX {status.vnindex_change_pct:+.2f}% - Blocking new entries",
            )

        if status.protection_level == "WARNING":
            return (
                False,
                f"⚠️ Market warning: VNINDEX {status.vnindex_change_pct:+.2f}% - Reduce position sizes",
            )

        return False, "Market normal - entries allowed"

    def get_position_size_multiplier(self, status: MarketCrashStatus) -> float:
        """
        Get position size multiplier based on market status.

        Args:
            status: MarketCrashStatus

        Returns:
            Multiplier (0.0-1.0) for position sizing
        """
        if status.protection_level == "CRASH":
            return 0.0  # No new positions
        elif status.protection_level == "CAUTION":
            return 0.0  # No new positions
        elif status.protection_level == "WARNING":
            return 0.5  # Reduce by 50%
        else:
            return 1.0  # Normal sizing

    def is_crash_mode_active(self) -> bool:
        """Check if crash mode is currently active."""
        return self._crash_mode_active


# Singleton instance
_crash_protector: Optional[MarketCrashProtector] = None


def get_market_crash_protector() -> MarketCrashProtector:
    """Get singleton instance of MarketCrashProtector."""
    global _crash_protector
    if _crash_protector is None:
        _crash_protector = MarketCrashProtector()
    return _crash_protector


# Convenience functions
def check_market_crash_status(
    vnindex_current: float, vnindex_previous: Optional[float] = None
) -> MarketCrashStatus:
    """Quick check of market crash status."""
    protector = get_market_crash_protector()
    return protector.check_market_status(vnindex_current, vnindex_previous)


def should_block_entries(
    vnindex_current: float, vnindex_previous: Optional[float] = None
) -> Tuple[bool, str]:
    """Quick check if entries should be blocked."""
    status = check_market_crash_status(vnindex_current, vnindex_previous)
    protector = get_market_crash_protector()
    return protector.should_block_new_entries(status)


# Test
if __name__ == "__main__":
    print("Testing Market Crash Protection...")

    protector = MarketCrashProtector()

    # Test scenarios
    test_cases = [
        (1500.0, 1500.0, "Normal market"),
        (1470.0, 1500.0, "Warning: -2%"),
        (1425.0, 1500.0, "Caution: -5%"),
        (1395.0, 1500.0, "Crash: -7%"),
        (1350.0, 1500.0, "Severe crash: -10%"),
    ]

    for current, previous, description in test_cases:
        status = protector.check_market_status(current, previous)
        print(f"\n{description}:")
        print(f"  Level: {status.protection_level}")
        print(f"  Change: {status.vnindex_change_pct:+.2f}%")
        print(f"  Action: {status.recommended_action}")
        print(f"  Reduction: {status.exposure_reduction_pct*100:.0f}%")
        print(f"  Message: {status.message}")

    print("\n✅ Market Crash Protection test completed!")
