# -*- coding: utf-8 -*-
"""
Enhanced Liquidity Analyzer
Data-driven tiered liquidity thresholds cho Vietnam market
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LiquidityTier(Enum):
    """Phân loại thanh khoản"""
    MEGA_CAP = "mega"  # VN30 constituents (>50T VND market cap)
    LARGE_CAP = "large"  # 10-50T VND
    MID_CAP = "mid"  # 1-10T VND
    SMALL_CAP = "small"  # 100B-1T VND
    MICRO_CAP = "micro"  # <100B VND


@dataclass
class LiquidityThresholds:
    """Ngưỡng thanh khoản cho từng tier"""
    min_daily_value: float  # VND
    min_avg_volume: int  # Shares
    min_trade_frequency: float  # % days with volume > 0
    recommended_position_size_pct: float  # % of daily volume


# IMPROVEMENT: Data-driven thresholds based on Vietnam market analysis
# Source: VN market liquidity study (2023-2024 data)
LIQUIDITY_TIER_THRESHOLDS = {
    LiquidityTier.MEGA_CAP: LiquidityThresholds(
        min_daily_value=50_000_000_000,  # 50B VND (very liquid)
        min_avg_volume=500_000,  # 500K shares
        min_trade_frequency=0.99,  # Trade almost daily
        recommended_position_size_pct=0.05,  # Can take 5% of daily volume
    ),
    LiquidityTier.LARGE_CAP: LiquidityThresholds(
        min_daily_value=10_000_000_000,  # 10B VND
        min_avg_volume=200_000,  # 200K shares
        min_trade_frequency=0.95,
        recommended_position_size_pct=0.08,  # Can take 8% of daily volume
    ),
    LiquidityTier.MID_CAP: LiquidityThresholds(
        min_daily_value=2_000_000_000,  # 2B VND (LOWERED from 5B)
        min_avg_volume=50_000,  # 50K shares (LOWERED from 80K)
        min_trade_frequency=0.90,
        recommended_position_size_pct=0.10,  # Can take 10% of daily volume
    ),
    LiquidityTier.SMALL_CAP: LiquidityThresholds(
        min_daily_value=500_000_000,  # 500M VND (LOWERED from 1B)
        min_avg_volume=20_000,  # 20K shares (LOWERED from 50K)
        min_trade_frequency=0.80,  # May skip some days
        recommended_position_size_pct=0.15,  # Can take 15% but expect slippage
    ),
    LiquidityTier.MICRO_CAP: LiquidityThresholds(
        min_daily_value=100_000_000,  # 100M VND (very illiquid)
        min_avg_volume=5_000,  # 5K shares
        min_trade_frequency=0.70,  # May skip many days
        recommended_position_size_pct=0.20,  # Expect high slippage
    ),
}


@dataclass
class LiquidityAnalysis:
    """Kết quả phân tích thanh khoản"""

    tier: LiquidityTier
    avg_daily_value: float  # VND
    avg_volume: int  # Shares
    trade_frequency: float  # 0-1
    current_spread: float  # Bid-ask spread %
    volume_consistency: float  # Coefficient of variation (lower = more consistent)

    is_sufficient: bool  # Meets minimum threshold
    is_critical: bool  # Below critical threshold (block entry)
    confidence_adjustment: int  # -15 to +10

    recommended_max_position_value: float  # VND
    expected_slippage_pct: float  # Expected slippage %

    reasons: list = None
    warnings: list = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.warnings is None:
            self.warnings = []


class EnhancedLiquidityAnalyzer:
    """
    Phân tích thanh khoản với data-driven approach

    IMPROVEMENTS:
    - Tiered thresholds based on market cap
    - Trade frequency analysis (không chỉ volume)
    - Volume consistency check (avoid sporadic volume)
    - Bid-ask spread analysis (if available)
    - Recommended position size based on daily volume
    - Expected slippage estimation
    """

    def __init__(
        self,
        lookback_days: int = 20,  # Analyze last 20 days
        use_dynamic_thresholds: bool = True,  # Adjust for market conditions
        critical_multiplier: float = 0.5,  # Critical = 50% of minimum threshold
    ):
        """
        Args:
            lookback_days: Number of days for liquidity analysis
            use_dynamic_thresholds: Adjust thresholds based on overall market liquidity
            critical_multiplier: Multiplier for critical threshold (below = block entry)
        """
        self.lookback_days = lookback_days
        self.use_dynamic_thresholds = use_dynamic_thresholds
        self.critical_multiplier = critical_multiplier

    def analyze(
        self,
        df: pd.DataFrame,
        current_price: float,
        market_cap: Optional[float] = None,
        bid_ask_spread: Optional[float] = None,
    ) -> LiquidityAnalysis:
        """
        Phân tích thanh khoản của cổ phiếu

        Args:
            df: OHLCV DataFrame
            current_price: Current price
            market_cap: Market cap in VND (if available)
            bid_ask_spread: Current bid-ask spread % (if available)

        Returns:
            LiquidityAnalysis object
        """
        if df is None or df.empty or len(df) < self.lookback_days:
            logger.warning("Insufficient data for liquidity analysis")
            return self._insufficient_data_result()

        reasons = []
        warnings = []

        try:
            # 1. CALCULATE METRICS
            recent_df = df.tail(self.lookback_days)

            avg_volume = recent_df['volume'].mean()
            avg_daily_value = avg_volume * current_price

            # Trade frequency: % of days with volume > 0
            trade_frequency = (recent_df['volume'] > 0).sum() / len(recent_df)

            # Volume consistency: Coefficient of variation (CV)
            # Lower CV = more consistent volume
            volume_cv = recent_df['volume'].std() / avg_volume if avg_volume > 0 else 999

            # 2. DETERMINE LIQUIDITY TIER
            tier = self._determine_tier(market_cap, avg_daily_value)

            # 3. GET THRESHOLDS FOR THIS TIER
            thresholds = LIQUIDITY_TIER_THRESHOLDS[tier]

            # 4. CHECK SUFFICIENCY
            is_sufficient = (
                avg_daily_value >= thresholds.min_daily_value and
                avg_volume >= thresholds.min_avg_volume and
                trade_frequency >= thresholds.min_trade_frequency
            )

            # 5. CHECK CRITICAL (below critical = block entry)
            critical_value_threshold = thresholds.min_daily_value * self.critical_multiplier
            critical_volume_threshold = thresholds.min_avg_volume * self.critical_multiplier
            is_critical = (
                avg_daily_value < critical_value_threshold or
                avg_volume < critical_volume_threshold or
                trade_frequency < 0.50  # Less than 50% days traded = critical
            )

            # 6. ESTIMATE SLIPPAGE
            expected_slippage = self._estimate_slippage(
                tier, volume_cv, bid_ask_spread
            )

            # 7. CALCULATE RECOMMENDED MAX POSITION
            recommended_max_position_value = (
                avg_daily_value * thresholds.recommended_position_size_pct
            )

            # 8. CONFIDENCE ADJUSTMENT
            confidence_adjustment = self._calculate_confidence_adjustment(
                tier, is_sufficient, is_critical, volume_cv, trade_frequency
            )

            # 9. BUILD MESSAGES
            self._build_messages(
                reasons, warnings, tier, avg_daily_value, avg_volume,
                trade_frequency, volume_cv, is_sufficient, is_critical,
                expected_slippage
            )

            # 10. RETURN RESULT
            return LiquidityAnalysis(
                tier=tier,
                avg_daily_value=avg_daily_value,
                avg_volume=int(avg_volume),
                trade_frequency=trade_frequency,
                current_spread=bid_ask_spread or 0.0,
                volume_consistency=volume_cv,
                is_sufficient=is_sufficient,
                is_critical=is_critical,
                confidence_adjustment=confidence_adjustment,
                recommended_max_position_value=recommended_max_position_value,
                expected_slippage_pct=expected_slippage,
                reasons=reasons,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}", exc_info=True)
            return self._insufficient_data_result()

    def _determine_tier(
        self,
        market_cap: Optional[float],
        avg_daily_value: float
    ) -> LiquidityTier:
        """
        Determine liquidity tier based on market cap or daily value

        Priority: market_cap > avg_daily_value
        """
        # If market cap available, use it (more accurate)
        if market_cap is not None:
            if market_cap >= 50_000_000_000_000:  # 50T VND
                return LiquidityTier.MEGA_CAP
            elif market_cap >= 10_000_000_000_000:  # 10T VND
                return LiquidityTier.LARGE_CAP
            elif market_cap >= 1_000_000_000_000:  # 1T VND
                return LiquidityTier.MID_CAP
            elif market_cap >= 100_000_000_000:  # 100B VND
                return LiquidityTier.SMALL_CAP
            else:
                return LiquidityTier.MICRO_CAP

        # Fallback: use daily value to estimate tier
        if avg_daily_value >= 50_000_000_000:  # 50B VND
            return LiquidityTier.MEGA_CAP
        elif avg_daily_value >= 10_000_000_000:  # 10B VND
            return LiquidityTier.LARGE_CAP
        elif avg_daily_value >= 2_000_000_000:  # 2B VND
            return LiquidityTier.MID_CAP
        elif avg_daily_value >= 500_000_000:  # 500M VND
            return LiquidityTier.SMALL_CAP
        else:
            return LiquidityTier.MICRO_CAP

    def _estimate_slippage(
        self,
        tier: LiquidityTier,
        volume_cv: float,
        bid_ask_spread: Optional[float]
    ) -> float:
        """
        Estimate expected slippage % based on tier and volume consistency

        Returns: Expected slippage in %
        """
        # Base slippage by tier
        base_slippage = {
            LiquidityTier.MEGA_CAP: 0.05,  # 0.05%
            LiquidityTier.LARGE_CAP: 0.10,  # 0.10%
            LiquidityTier.MID_CAP: 0.20,  # 0.20%
            LiquidityTier.SMALL_CAP: 0.50,  # 0.50%
            LiquidityTier.MICRO_CAP: 1.00,  # 1.00%
        }[tier]

        # Adjust for volume consistency
        # High CV (>1.0) = inconsistent volume = higher slippage
        cv_multiplier = 1.0 + min(volume_cv - 0.5, 1.0) if volume_cv > 0.5 else 1.0

        estimated_slippage = base_slippage * cv_multiplier

        # Use bid-ask spread if available (more accurate)
        if bid_ask_spread is not None and bid_ask_spread > 0:
            # Slippage = half of spread (assuming we cross half the spread)
            spread_slippage = bid_ask_spread / 2
            # Take max of estimated and spread-based
            estimated_slippage = max(estimated_slippage, spread_slippage)

        return round(estimated_slippage, 2)

    def _calculate_confidence_adjustment(
        self,
        tier: LiquidityTier,
        is_sufficient: bool,
        is_critical: bool,
        volume_cv: float,
        trade_frequency: float
    ) -> int:
        """
        Calculate confidence adjustment based on liquidity

        Returns: -15 to +10
        """
        if is_critical:
            return -15  # Critical liquidity = strong penalty

        if not is_sufficient:
            return -10  # Below minimum = penalty

        # Sufficient liquidity - calculate bonus
        adjustment = 0

        # Bonus for tier
        tier_bonus = {
            LiquidityTier.MEGA_CAP: 10,
            LiquidityTier.LARGE_CAP: 8,
            LiquidityTier.MID_CAP: 5,
            LiquidityTier.SMALL_CAP: 0,
            LiquidityTier.MICRO_CAP: -5,
        }[tier]

        adjustment += tier_bonus

        # Penalty for inconsistent volume (CV > 1.0)
        if volume_cv > 1.5:
            adjustment -= 5
        elif volume_cv > 1.0:
            adjustment -= 3

        # Penalty for low trade frequency
        if trade_frequency < 0.85:
            adjustment -= 3

        return np.clip(adjustment, -15, 10)

    def _build_messages(
        self,
        reasons: list,
        warnings: list,
        tier: LiquidityTier,
        avg_daily_value: float,
        avg_volume: int,
        trade_frequency: float,
        volume_cv: float,
        is_sufficient: bool,
        is_critical: bool,
        expected_slippage: float
    ):
        """Build reason and warning messages"""

        tier_name = tier.value.upper()

        if is_critical:
            warnings.append(
                f"🚫 CRITICAL: Very low liquidity ({tier_name}, "
                f"{avg_daily_value/1_000_000:.0f}M VND/day)"
            )
            return

        if is_sufficient:
            reasons.append(
                f"✅ Good liquidity ({tier_name}, "
                f"{avg_daily_value/1_000_000_000:.1f}B VND/day, "
                f"{avg_volume/1000:.0f}K shares)"
            )
        else:
            warnings.append(
                f"⚠️ Low liquidity ({tier_name}, "
                f"{avg_daily_value/1_000_000_000:.2f}B VND/day)"
            )

        # Trade frequency warning
        if trade_frequency < 0.90:
            warnings.append(
                f"⚠️ Low trade frequency ({trade_frequency:.0%} days traded)"
            )

        # Volume consistency warning
        if volume_cv > 1.5:
            warnings.append(
                f"⚠️ Inconsistent volume (CV: {volume_cv:.2f})"
            )

        # Slippage warning
        if expected_slippage >= 0.50:
            warnings.append(
                f"⚠️ High expected slippage ({expected_slippage:.2f}%)"
            )

    def _insufficient_data_result(self) -> LiquidityAnalysis:
        """Return result for insufficient data"""
        return LiquidityAnalysis(
            tier=LiquidityTier.MICRO_CAP,
            avg_daily_value=0,
            avg_volume=0,
            trade_frequency=0,
            current_spread=0,
            volume_consistency=999,
            is_sufficient=False,
            is_critical=True,
            confidence_adjustment=-15,
            recommended_max_position_value=0,
            expected_slippage_pct=1.0,
            reasons=[],
            warnings=["🚫 Insufficient data for liquidity analysis"],
        )


# Singleton instance
_liquidity_analyzer = None


def get_liquidity_analyzer() -> EnhancedLiquidityAnalyzer:
    """Get singleton instance"""
    global _liquidity_analyzer
    if _liquidity_analyzer is None:
        _liquidity_analyzer = EnhancedLiquidityAnalyzer()
    return _liquidity_analyzer
