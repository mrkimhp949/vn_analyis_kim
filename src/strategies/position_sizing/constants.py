# -*- coding: utf-8 -*-
"""
Position Sizing Constants

Centralized constants for position sizing calculations.
Extracted from position_sizing.py for better organization.
"""


class PositionSizingConstants:
    """Centralized constants for position sizing calculations."""

    # Risk thresholds
    MIN_RISK_PERCENT: float = 0.01  # 1% minimum risk per share
    DEFAULT_RISK_PERCENT: float = 0.02  # 2% default risk if stop too tight

    # Kelly Criterion
    MAX_KELLY_PERCENT: float = 0.25  # Max 25% of capital via Kelly
    DEFAULT_KELLY_FRACTION: float = 0.5  # Half-Kelly for safety
    MIN_KELLY_FALLBACK: float = 0.01  # 1% minimum for negative Kelly (v2.0 behavior)

    # Correlation
    HIGH_CORRELATION_THRESHOLD: float = 0.70
    MEDIUM_CORRELATION_THRESHOLD: float = 0.50
    HIGH_CORRELATION_ADJUSTMENT: float = 0.50  # Reduce 50%
    MEDIUM_CORRELATION_ADJUSTMENT: float = 0.75  # Reduce 25%

    # Sector limits
    SECTOR_HIGH_COUNT: int = 3  # 3+ positions = high concentration
    SECTOR_MEDIUM_COUNT: int = 2  # 2 positions = medium concentration
    SECTOR_HIGH_ADJUSTMENT: float = 0.70  # Reduce 30%
    SECTOR_MEDIUM_ADJUSTMENT: float = 0.85  # Reduce 15%

    # =========================================================================
    # LIQUIDITY TIERS - IMPROVED v5.0 for Vietnam Market
    # =========================================================================
    # 4-tier system based on stock liquidity and market cap
    # Higher liquidity = larger position sizes allowed
    #
    # VN30: Blue chips with highest liquidity (HPG, VNM, VCB, etc.)
    # LARGE_CAP: Large caps outside VN30 (> 5B VND daily)
    # MID_CAP: Mid caps (3-5B VND daily)
    # SMALL_CAP: Small caps (< 3B VND daily)
    #
    # Position limits prevent excessive exposure to illiquid stocks
    LIQUIDITY_TIERS = {
        "VN30": {
            "max_position_pct": 0.15,  # 15% max position for VN30
            "min_daily_value": 10_000_000_000,  # 10B VND
            "slippage": 0.003,  # 0.3% slippage
        },
        "LARGE_CAP": {
            "max_position_pct": 0.12,  # 12% max position
            "min_daily_value": 5_000_000_000,  # 5B VND
            "slippage": 0.004,  # 0.4% slippage
        },
        "MID_CAP": {
            "max_position_pct": 0.10,  # 10% max position
            "min_daily_value": 3_000_000_000,  # 3B VND
            "slippage": 0.006,  # 0.6% slippage
        },
        "SMALL_CAP": {
            "max_position_pct": 0.06,  # 6% max position
            "min_daily_value": 0,  # Any liquidity
            "slippage": 0.010,  # 1.0% slippage
        },
    }

    # =========================================================================
    # REGIME-AWARE KELLY ADJUSTMENTS - IMPROVED v5.0
    # =========================================================================
    # Kelly fraction adjusted by market regime for risk management
    # BULL: Full half-Kelly (aggressive)
    # SIDEWAYS: Standard half-Kelly
    # BEAR: Quarter-Kelly (defensive)
    # HIGH_VOL: Eighth-Kelly (very defensive)
    REGIME_KELLY_FRACTIONS = {
        "BULL": 0.50,  # Full half-Kelly
        "SIDEWAYS": 0.40,  # Slightly reduced
        "BEAR": 0.25,  # Quarter-Kelly
        "HIGH_VOLATILITY": 0.125,  # Eighth-Kelly
    }

    # Transaction cost for Kelly adjustment (Vietnam market)
    VN_TRANSACTION_COST: float = 0.0148  # 1.48% round trip

    # DCA levels - IMPROVED v4.2 for Vietnam Market
    # Vietnam market characteristics:
    # - ±7% daily price limit means 1-3% DCA levels can hit within same day
    # - Transaction cost ~1.48% round trip reduces DCA effectiveness
    # - Wider levels (2%, 4%, 6%) provide better cost-adjusted entries
    # - Each DCA level should exceed transaction cost to be profitable
    #
    # RECOMMENDATION: Consider disabling DCA for VN market due to:
    # 1. High transaction costs (1.48% round trip)
    # 2. T+2 settlement ties up capital
    # 3. Narrow DCA levels get hit too quickly in volatile market
    DCA_LEVEL_1_PERCENT: float = 0.50  # 50% at first level
    DCA_LEVEL_2_PERCENT: float = 0.30  # 30% at second level
    DCA_LEVEL_3_PERCENT: float = 0.20  # 20% at third level
    DCA_LEVEL_1_DISCOUNT: float = 0.98  # WIDENED: 2% below entry (was 1%)
    DCA_LEVEL_2_DISCOUNT: float = 0.96  # WIDENED: 4% below entry (was 2%)
    DCA_LEVEL_3_DISCOUNT: float = 0.94  # WIDENED: 6% below entry (was 3%)

    # =========================================================================
    # DCA (Dollar Cost Averaging) Configuration - IMPROVED v6.0
    # =========================================================================
    # DCA is DISABLED for Vietnam market due to unfavorable cost structure
    #
    # RATIONALE - Why DCA doesn't work well for VN market:
    #
    # 1. HIGH TRANSACTION COSTS (1.48% round trip)
    #    - Each DCA buy costs ~0.70% (brokerage + fees + slippage)
    #    - Each sell costs ~0.78% (includes 0.1% government tax)
    #    - Total round trip: 1.48%
    #    - DCA level must exceed 1.48% to be profitable
    #
    # 2. T+2 SETTLEMENT TIES UP CAPITAL
    #    - Buy on T0 → Cash locked until T+2
    #    - Multiple DCA buys = multiple capital lockups
    #    - Reduces flexibility for other opportunities
    #
    # 3. ±7% DAILY LIMIT MAKES DCA LEVELS HIT TOO QUICKLY
    #    - Traditional DCA levels (1%, 2%, 3%) can all hit in one day
    #    - No time to assess if drop is temporary or trend change
    #    - Wider levels (2%, 4%, 6%) recommended if DCA is used
    #
    # 4. COST-BENEFIT ANALYSIS
    #    - 3 DCA buys at 2%, 4%, 6% below entry
    #    - Total transaction cost: 3 × 0.70% = 2.1% on buys
    #    - Plus 0.78% on final sell = 2.88% total
    #    - Average entry improvement: ~4%
    #    - Net benefit: 4% - 2.88% = 1.12% (marginal)
    #
    # RECOMMENDATION: Use single entry with proper position sizing instead
    # =========================================================================
    DCA_ENABLED: bool = False  # DISABLED: High transaction costs make DCA unprofitable
    DCA_MIN_PROFIT_THRESHOLD: float = 0.03  # TIGHTENED: Min 3% expected profit after costs

    # Cache settings
    CORRELATION_CACHE_TTL: int = 3600  # 1 hour
    CORRELATION_CACHE_MAXSIZE: int = 500

    # Risk multiplier bounds - DOCUMENTED v4.2
    # These bounds control position size scaling based on signal quality
    #
    # MIN_RISK_MULTIPLIER = 0.5 rationale:
    # - Even weak signals get 50% of base position
    # - Prevents over-reduction that makes positions too small
    # - 50% of 1.5% risk = 0.75% risk per trade (still meaningful)
    # - Allows participation in uncertain markets with reduced exposure
    #
    # MAX_RISK_MULTIPLIER = 1.2 rationale:
    # - Strong signals get max 20% boost over base position
    # - Conservative cap prevents overconfidence in any single trade
    # - 120% of 1.5% risk = 1.8% risk per trade (within 2% guideline)
    # - Balances conviction with risk management
    #
    # Combined with Kelly Criterion, actual position sizes are further
    # constrained by win rate and risk/reward statistics.
    MIN_RISK_MULTIPLIER: float = 0.5
    MAX_RISK_MULTIPLIER: float = 1.2

    # Circuit breaker
    CAUTION_MODE_MULTIPLIER: float = 0.5  # Reduce 50% in caution mode
