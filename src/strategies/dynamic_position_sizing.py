"""
Dynamic Position Sizing Module
Adjusts position sizes based on:
1. Market regime (Bull/Bear/Sideways)
2. Market volatility (ATR-based)
3. Recent trading performance (consecutive wins/losses)
4. Drawdown state

IMPROVEMENT: Addresses critique point #3 in trading logic evaluation
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DynamicPositionSizeAdjuster:
    """
    Dynamically adjusts position sizes based on market conditions and performance

    Base position size is modified by multipliers from:
    - Market regime: Bull (1.0-1.2x), Sideways (0.8-1.0x), Bear (0.5-0.8x)
    - Volatility: Low (<1.5%) = 1.0x, Medium (1.5-3%) = 0.85x, High (>3%) = 0.6x
    - Recent performance: Winning streak (1.0-1.1x), Losing streak (0.6-0.9x)
    - Drawdown: <5% = 1.0x, 5-10% = 0.8x, >10% = 0.5x
    """

    def __init__(
        self,
        # Regime multipliers
        bull_multiplier: float = 1.1,  # Increase size in bull market
        sideways_multiplier: float = 0.9,  # Slightly reduce in sideways
        bear_multiplier: float = 0.5,  # Significantly reduce in bear market
        # Volatility thresholds and multipliers
        low_vol_threshold: float = 0.015,  # 1.5% volatility threshold
        high_vol_threshold: float = 0.030,  # 3.0% volatility threshold
        high_vol_multiplier: float = 0.6,  # Reduce to 60% in high volatility
        medium_vol_multiplier: float = 0.85,  # Reduce to 85% in medium volatility
        # Performance-based adjustments
        consecutive_wins_boost: float = 1.1,  # 10% boost after 3+ wins
        consecutive_losses_penalty: float = 0.6,  # Reduce to 60% after 2+ losses
        min_trades_for_adjustment: int = 3,  # Minimum trades before adjusting
        # Drawdown-based adjustments
        warning_drawdown_pct: float = 0.05,  # 5% drawdown warning
        critical_drawdown_pct: float = 0.10,  # 10% drawdown critical
        drawdown_warning_multiplier: float = 0.8,  # Reduce to 80% at warning
        drawdown_critical_multiplier: float = 0.5,  # Reduce to 50% at critical
    ):
        # Regime settings
        self.bull_multiplier = bull_multiplier
        self.sideways_multiplier = sideways_multiplier
        self.bear_multiplier = bear_multiplier

        # Volatility settings
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.high_vol_multiplier = high_vol_multiplier
        self.medium_vol_multiplier = medium_vol_multiplier

        # Performance settings
        self.consecutive_wins_boost = consecutive_wins_boost
        self.consecutive_losses_penalty = consecutive_losses_penalty
        self.min_trades_for_adjustment = min_trades_for_adjustment

        # Drawdown settings
        self.warning_drawdown_pct = warning_drawdown_pct
        self.critical_drawdown_pct = critical_drawdown_pct
        self.drawdown_warning_multiplier = drawdown_warning_multiplier
        self.drawdown_critical_multiplier = drawdown_critical_multiplier

    def get_regime_multiplier(self, market_regime: Optional[Dict]) -> tuple:
        """
        Get position size multiplier based on market regime

        Args:
            market_regime: Market regime dict with 'regime' key

        Returns:
            (multiplier, reason)
        """
        if market_regime is None:
            return 1.0, "No regime data"

        regime = market_regime.get("regime", "SIDEWAYS")

        if regime == "BULL":
            return self.bull_multiplier, f"Bull market (+{(self.bull_multiplier-1)*100:.0f}%)"
        elif regime == "BEAR":
            return self.bear_multiplier, f"Bear market ({(self.bear_multiplier-1)*100:.0f}%)"
        else:  # SIDEWAYS
            return (
                self.sideways_multiplier,
                f"Sideways market ({(self.sideways_multiplier-1)*100:.0f}%)",
            )

    def get_volatility_multiplier(self, market_volatility: float) -> tuple:
        """
        Get position size multiplier based on market volatility

        Args:
            market_volatility: Market volatility (ATR/Price ratio)

        Returns:
            (multiplier, reason)
        """
        if market_volatility <= 0:
            return 1.0, "No volatility data"

        vol_pct = market_volatility * 100

        if vol_pct > self.high_vol_threshold * 100:
            return (
                self.high_vol_multiplier,
                f"High volatility ({vol_pct:.1f}%) - reduce to {self.high_vol_multiplier*100:.0f}%",
            )
        elif vol_pct > self.low_vol_threshold * 100:
            return (
                self.medium_vol_multiplier,
                f"Medium volatility ({vol_pct:.1f}%) - reduce to {self.medium_vol_multiplier*100:.0f}%",
            )
        else:
            return 1.0, f"Low volatility ({vol_pct:.1f}%) - no adjustment"

    def get_performance_multiplier(
        self, consecutive_wins: int = 0, consecutive_losses: int = 0, total_trades: int = 0
    ) -> tuple:
        """
        Get position size multiplier based on recent trading performance

        Args:
            consecutive_wins: Number of consecutive winning trades
            consecutive_losses: Number of consecutive losing trades
            total_trades: Total number of trades (to check minimum threshold)

        Returns:
            (multiplier, reason)
        """
        # Don't adjust until we have enough trade history
        if total_trades < self.min_trades_for_adjustment:
            return 1.0, f"Insufficient trades ({total_trades} < {self.min_trades_for_adjustment})"

        # Losing streak - reduce size significantly
        if consecutive_losses >= 2:
            return (
                self.consecutive_losses_penalty,
                f"Losing streak ({consecutive_losses} losses) - reduce to {self.consecutive_losses_penalty*100:.0f}%",
            )

        # Winning streak - slightly increase size
        if consecutive_wins >= 3:
            return (
                self.consecutive_wins_boost,
                f"Winning streak ({consecutive_wins} wins) - boost to {self.consecutive_wins_boost*100:.0f}%",
            )

        return 1.0, "Performance neutral"

    def get_drawdown_multiplier(self, current_drawdown_pct: float) -> tuple:
        """
        Get position size multiplier based on portfolio drawdown

        Args:
            current_drawdown_pct: Current drawdown as percentage (0-1)

        Returns:
            (multiplier, reason)
        """
        if current_drawdown_pct >= self.critical_drawdown_pct:
            return (
                self.drawdown_critical_multiplier,
                f"CRITICAL drawdown ({current_drawdown_pct*100:.1f}%) - reduce to {self.drawdown_critical_multiplier*100:.0f}%",
            )
        elif current_drawdown_pct >= self.warning_drawdown_pct:
            return (
                self.drawdown_warning_multiplier,
                f"Warning drawdown ({current_drawdown_pct*100:.1f}%) - reduce to {self.drawdown_warning_multiplier*100:.0f}%",
            )

        return 1.0, "Drawdown OK"

    def calculate_dynamic_multiplier(
        self,
        market_regime: Optional[Dict] = None,
        market_volatility: float = 0.0,
        consecutive_wins: int = 0,
        consecutive_losses: int = 0,
        total_trades: int = 0,
        current_drawdown_pct: float = 0.0,
    ) -> Dict:
        """
        Calculate overall dynamic position size multiplier

        Combines all factors multiplicatively (conservative approach)

        Args:
            market_regime: Market regime dict
            market_volatility: Market volatility (ATR/Price)
            consecutive_wins: Consecutive winning trades
            consecutive_losses: Consecutive losing trades
            total_trades: Total number of trades
            current_drawdown_pct: Current drawdown percentage

        Returns:
            Dict with:
            - multiplier: Overall multiplier to apply to base position size
            - regime_mult: Regime component
            - vol_mult: Volatility component
            - perf_mult: Performance component
            - dd_mult: Drawdown component
            - reasons: List of adjustment reasons
            - final_recommendation: Human-readable recommendation
        """
        # Get individual multipliers
        regime_mult, regime_reason = self.get_regime_multiplier(market_regime)
        vol_mult, vol_reason = self.get_volatility_multiplier(market_volatility)
        perf_mult, perf_reason = self.get_performance_multiplier(
            consecutive_wins, consecutive_losses, total_trades
        )
        dd_mult, dd_reason = self.get_drawdown_multiplier(current_drawdown_pct)

        # Combine multiplicatively (conservative)
        overall_mult = regime_mult * vol_mult * perf_mult * dd_mult

        # Cap at reasonable bounds (0.3x to 1.3x)
        overall_mult = max(0.3, min(1.3, overall_mult))

        reasons = []
        if regime_mult != 1.0:
            reasons.append(regime_reason)
        if vol_mult != 1.0:
            reasons.append(vol_reason)
        if perf_mult != 1.0:
            reasons.append(perf_reason)
        if dd_mult != 1.0:
            reasons.append(dd_reason)

        # Generate recommendation
        if overall_mult >= 1.1:
            recommendation = f"INCREASE position size to {overall_mult*100:.0f}% of base"
        elif overall_mult <= 0.8:
            recommendation = f"REDUCE position size to {overall_mult*100:.0f}% of base"
        else:
            recommendation = "MAINTAIN base position size"

        result = {
            "multiplier": overall_mult,
            "regime_mult": regime_mult,
            "vol_mult": vol_mult,
            "perf_mult": perf_mult,
            "dd_mult": dd_mult,
            "reasons": reasons,
            "final_recommendation": recommendation,
            "breakdown": {
                "regime": regime_reason,
                "volatility": vol_reason,
                "performance": perf_reason,
                "drawdown": dd_reason,
            },
        }

        # Log the decision
        if reasons:
            logger.info(f"📊 Dynamic Position Sizing: {recommendation}")
            for reason in reasons:
                logger.info(f"   - {reason}")

        return result


# Singleton instance
_dynamic_adjuster = None


def get_dynamic_adjuster() -> DynamicPositionSizeAdjuster:
    """Get singleton instance of dynamic position size adjuster"""
    global _dynamic_adjuster
    if _dynamic_adjuster is None:
        _dynamic_adjuster = DynamicPositionSizeAdjuster()
    return _dynamic_adjuster


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TESTING DYNAMIC POSITION SIZING")
    print("=" * 70 + "\n")

    adjuster = DynamicPositionSizeAdjuster()

    # Test 1: Bull market, low volatility, winning streak
    print("1️⃣ Test: Bull + Low Vol + Winning Streak")
    result = adjuster.calculate_dynamic_multiplier(
        market_regime={"regime": "BULL"},
        market_volatility=0.012,  # 1.2%
        consecutive_wins=3,
        total_trades=5,
    )
    print(f"   Multiplier: {result['multiplier']:.2f}x")
    print(f"   Recommendation: {result['final_recommendation']}")
    print()

    # Test 2: Bear market, high volatility, losing streak
    print("2️⃣ Test: Bear + High Vol + Losing Streak")
    result = adjuster.calculate_dynamic_multiplier(
        market_regime={"regime": "BEAR"},
        market_volatility=0.035,  # 3.5%
        consecutive_losses=2,
        total_trades=5,
    )
    print(f"   Multiplier: {result['multiplier']:.2f}x")
    print(f"   Recommendation: {result['final_recommendation']}")
    print()

    # Test 3: Critical drawdown
    print("3️⃣ Test: Critical Drawdown (12%)")
    result = adjuster.calculate_dynamic_multiplier(
        current_drawdown_pct=0.12,
        total_trades=10,
    )
    print(f"   Multiplier: {result['multiplier']:.2f}x")
    print(f"   Recommendation: {result['final_recommendation']}")
    print()

    print("=" * 70)
    print("✅ Testing complete!")
