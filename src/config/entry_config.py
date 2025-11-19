"""
Entry Logic Configuration
Centralized configuration for all entry signal filters and thresholds
"""

from dataclasses import dataclass
from typing import Dict, List
import os


@dataclass
class FilterThresholds:
    """Thresholds for entry signal filters"""

    # Market Regime Adjustments
    regime_confidence_threshold: int = 70  # Min confidence to apply regime adjustments

    # Trend Alignment
    trend_perfect_bonus: int = 5  # Bonus for perfect uptrend (Price > EMA20 > EMA50 > EMA200)
    trend_weak_penalty: int = -10  # Penalty for weak trend

    # Support/Resistance
    support_distance_percent: float = 3.0  # Max distance to support (%)
    resistance_proximity_percent: float = 2.0  # Too close to resistance threshold (%)
    support_bounce_distance: float = 0.02  # Distance to detect support bounce (2%)
    support_near_bonus: int = 10  # Bonus for entry near support
    support_bounce_bonus: int = 15  # Bonus for bouncing from support (reversal)
    resistance_close_penalty: int = -15  # Penalty for being close to resistance

    # Volume Confirmation
    volume_ratio_threshold: float = 1.2  # Minimum volume ratio (current/avg)
    volume_surge_threshold: float = 1.5  # Volume surge threshold
    volume_low_penalty: int = -10  # Penalty for low volume
    volume_surge_bonus: int = 5  # Bonus for volume surge

    # Liquidity Tiers (VND daily value)
    liquidity_large_cap: float = 5_000_000_000  # 5B VND
    liquidity_mid_cap: float = 2_000_000_000  # 2B VND
    liquidity_small_cap: float = 1_000_000_000  # 1B VND
    liquidity_volume_large: int = 150_000
    liquidity_volume_mid: int = 80_000
    liquidity_volume_small: int = 50_000
    liquidity_low_penalty: int = -15
    liquidity_good_bonus: int = 5

    # Volatility
    volatility_optimal_min: float = 2.0  # Min optimal volatility (%)
    volatility_optimal_max: float = 3.0  # Max optimal volatility (%)
    volatility_too_high: float = 4.0  # Too high volatility threshold (%)
    volatility_high_penalty: int = -15
    volatility_optimal_bonus: int = 5

    # RSI
    rsi_oversold: float = 30  # RSI oversold level (strong buy)
    rsi_optimal_min: float = 30  # RSI optimal range min
    rsi_optimal_max: float = 60  # RSI optimal range max
    rsi_overbought: float = 70  # RSI overbought level (reject)
    rsi_oversold_bonus: int = 15  # Bonus for oversold RSI (strong signal)
    rsi_optimal_bonus: int = 5  # Bonus for optimal RSI
    rsi_overbought_penalty: int = -10  # Penalty for overbought RSI

    # Price Action
    price_action_bullish_bonus: int = 10
    price_action_bearish_penalty: int = -10

    # Sector Strength
    sector_rs_threshold: float = 1.0  # RS > 1.0 = leading sector
    sector_weak_threshold: float = 0.95  # RS < 0.95 = lagging sector
    sector_leading_bonus: int = 10
    sector_lagging_penalty: int = -15

    # Multi-Timeframe
    mtf_weekly_penalty: int = -5
    mtf_monthly_penalty: int = -5

    # Market Breadth
    breadth_strong_threshold: float = 0.6  # 60% advancing = strong
    breadth_weak_threshold: float = 0.4  # 40% advancing = weak
    breadth_strong_bonus: int = 5
    breadth_weak_penalty: int = -10

    # Portfolio Correlation
    correlation_max_threshold: float = 0.70  # Max correlation with portfolio
    correlation_diversification_threshold: float = 0.30  # Good diversification
    correlation_avg_threshold: float = 0.25  # Avg correlation threshold
    correlation_high_penalty: int = -20  # Large penalty for high correlation
    correlation_good_bonus: int = 5

    # Earnings/Events
    earnings_days_before: int = 5  # Don't enter 5 days before earnings
    earnings_penalty: int = -25  # Large penalty for entering near earnings

    # Fundamentals
    pe_ratio_max: float = 30  # Max acceptable P/E ratio
    pe_ratio_min: float = 5  # Min acceptable P/E ratio
    pe_ratio_optimal_min: float = 8  # Optimal P/E range min
    pe_ratio_optimal_max: float = 20  # Optimal P/E range max
    debt_ratio_max: float = 0.70  # Max debt ratio (70%)
    debt_ratio_optimal: float = 0.30  # Optimal debt ratio (30%)
    fundamentals_poor_penalty: int = -15
    fundamentals_good_bonus: int = 5


@dataclass
class MarketRegimeAdjustments:
    """Dynamic adjustments based on market regime"""

    # BULL Market
    bull_confidence_adjustment: int = -5  # Lower threshold in bull market
    bull_penalty_scale: float = 0.7  # Scale penalties down (70%)
    bull_min_regime_confidence: int = 70  # Min confidence to apply bull adjustments

    # BEAR Market
    bear_confidence_adjustment: int = 10  # Higher threshold in bear market
    bear_penalty_scale: float = 1.2  # Scale penalties up (120%)

    # HIGH VOLATILITY
    high_vol_confidence_adjustment: int = 15  # Much higher threshold
    high_vol_penalty_scale: float = 1.3  # Scale penalties up (130%)

    # SIDEWAYS
    sideways_penalty_scale: float = 1.0  # Normal (100%)

    # Portfolio Heat Adjustments
    portfolio_heat_threshold_1: int = 5  # First threshold (5 positions)
    portfolio_heat_adjustment_1: int = 5  # Raise confidence by +5
    portfolio_heat_threshold_2: int = 8  # Second threshold (8 positions)
    portfolio_heat_adjustment_2: int = 10  # Raise confidence by +10


@dataclass
class VolumeConfirmationConfig:
    """Volume confirmation multi-indicator settings"""

    # Dynamic thresholds by market regime
    bull_threshold: float = 0.4  # Lower threshold in bull market
    bear_threshold: float = 0.6  # Higher threshold in bear market
    sideways_threshold: float = 0.5  # Normal threshold

    # Weights for scoring
    volume_ratio_weight: float = 0.4  # 40% weight
    volume_trend_weight: float = 0.3  # 30% weight
    obv_weight: float = 0.3  # 30% weight

    # Volume ratio thresholds
    volume_ratio_strong: float = 1.5  # 1.5x = strong
    volume_ratio_good: float = 1.2  # 1.2x = good
    volume_ratio_neutral: float = 1.0  # 1.0x = neutral


@dataclass
class EntryOptimizationConfig:
    """Entry price optimization settings"""

    # Pullback Entry
    pullback_max_from_high: float = 5.0  # Max pullback % from recent high
    pullback_min_from_high: float = 1.0  # Min pullback % to consider
    pullback_distance_threshold: float = 1.0  # Distance to EMA/support (%)
    pullback_min_improvement: float = 0.5  # Min improvement vs market order (%)

    # Breakout Entry
    breakout_resistance_threshold: float = 1.0  # Distance to resistance (%)
    breakout_volume_multiplier: float = 1.2  # Min volume for breakout
    breakout_pullback_pct: float = 0.5  # Pullback % after breakout

    # Limit vs Market Order
    limit_order_min_diff: float = 0.5  # Min price diff to use limit order (%)

    # Entry Types
    enable_pullback_entry: bool = True
    enable_breakout_entry: bool = True
    enable_rsi_oversold_entry: bool = True


@dataclass
class RiskManagementConfig:
    """Risk management for entry signals"""

    # Stop Loss
    stop_loss_atr_multiplier: float = 2.0  # 2x ATR
    stop_loss_min_percent: float = 3.0  # Min 3% stop loss
    stop_loss_max_percent: float = 10.0  # Max 10% stop loss

    # Take Profit
    take_profit_ratios: List[float] = None  # Will be set in __post_init__

    # Risk/Reward
    min_risk_reward: float = 2.0  # Minimum 2:1 R:R ratio

    # Position Sizing
    position_multiplier_min: float = 0.3
    position_multiplier_max: float = 1.5

    # Signal Strength Thresholds
    strength_very_strong: float = 5.0
    strength_strong: float = 4.0
    strength_moderate: float = 3.0
    strength_weak: float = 2.0

    # Max Drawdown
    max_portfolio_drawdown: float = 0.15  # 15% max drawdown
    max_position_drawdown: float = 0.20  # 20% max position drawdown

    def __post_init__(self):
        if self.take_profit_ratios is None:
            self.take_profit_ratios = [1.5, 3.0, 5.0]


@dataclass
class PerformanceFeedbackConfig:
    """Performance feedback loop settings"""

    min_trades_for_feedback: int = 20  # Min trades before applying feedback
    win_rate_good_threshold: float = 60.0  # Win rate >= 60% = good
    win_rate_poor_threshold: float = 45.0  # Win rate <= 45% = poor
    confidence_adjustment_good: int = 5  # Adjustment for good performance
    confidence_adjustment_poor: int = -5  # Adjustment for poor performance


@dataclass
class FilterWeights:
    """
    SIMPLIFIED: Core filter priorities
    These weights determine which filters are MANDATORY vs OPTIONAL
    """

    # MANDATORY FILTERS (must pass or signal rejected)
    mandatory_filters: List[str] = None

    # HIGH PRIORITY FILTERS (large impact on confidence)
    high_priority: Dict[str, int] = None

    # MEDIUM PRIORITY FILTERS (moderate impact)
    medium_priority: Dict[str, int] = None

    # LOW PRIORITY FILTERS (small impact, can be ignored)
    low_priority: Dict[str, int] = None

    def __post_init__(self):
        if self.mandatory_filters is None:
            # These MUST pass or signal is rejected
            self.mandatory_filters = [
                "market_regime",  # Market must be tradeable
                "liquidity",  # Must have minimum liquidity
                "risk_reward",  # Must meet R:R ratio
            ]

        if self.high_priority is None:
            # These have biggest impact on signal quality
            self.high_priority = {
                "trend_alignment": 10,
                "support_resistance": 15,
                "volume_confirmation": 10,
                "rsi": 15,
            }

        if self.medium_priority is None:
            # These are important but not critical
            self.medium_priority = {
                "volatility": 8,
                "sector_strength": 10,
                "portfolio_correlation": 12,
            }

        if self.low_priority is None:
            # These are nice-to-have
            self.low_priority = {
                "price_action": 5,
                "multi_timeframe": 5,
                "market_breadth": 5,
            }


@dataclass
class EntryLogicConfig:
    """Master configuration for entry logic"""

    # Sub-configurations
    filters: FilterThresholds = None
    regime: MarketRegimeAdjustments = None
    volume: VolumeConfirmationConfig = None
    entry_optimization: EntryOptimizationConfig = None
    risk: RiskManagementConfig = None
    performance: PerformanceFeedbackConfig = None
    weights: FilterWeights = None

    # Base settings
    min_confidence: int = 60  # Base minimum confidence
    base_min_confidence: int = 60  # Original threshold (for dynamic adjustment)
    min_confidence_lower_bound: int = 40  # Min allowed after adjustments
    min_confidence_upper_bound: int = 80  # Max allowed after adjustments

    # Filter simplification
    use_simplified_filters: bool = True  # Use 8-10 core filters instead of 14

    # Tiered liquidity
    use_tiered_liquidity: bool = True

    # Requirements
    require_trend_alignment: bool = False  # Changed to optional
    require_volume_confirmation: bool = False  # Changed to optional

    def __post_init__(self):
        if self.filters is None:
            self.filters = FilterThresholds()
        if self.regime is None:
            self.regime = MarketRegimeAdjustments()
        if self.volume is None:
            self.volume = VolumeConfirmationConfig()
        if self.entry_optimization is None:
            self.entry_optimization = EntryOptimizationConfig()
        if self.risk is None:
            self.risk = RiskManagementConfig()
        if self.performance is None:
            self.performance = PerformanceFeedbackConfig()
        if self.weights is None:
            self.weights = FilterWeights()

    @classmethod
    def from_env(cls):
        """Load from environment variables"""
        return cls(
            min_confidence=int(os.getenv("ENTRY_MIN_CONFIDENCE", 60)),
            use_simplified_filters=os.getenv("USE_SIMPLIFIED_FILTERS", "true").lower() == "true",
            use_tiered_liquidity=os.getenv("USE_TIERED_LIQUIDITY", "true").lower() == "true",
            require_trend_alignment=os.getenv("REQUIRE_TREND_ALIGNMENT", "false").lower() == "true",
            require_volume_confirmation=os.getenv("REQUIRE_VOLUME_CONFIRMATION", "false").lower() == "true",
        )

    def validate(self):
        """Validate configuration"""
        if not (0 <= self.min_confidence <= 100):
            raise ValueError(f"min_confidence must be 0-100, got {self.min_confidence}")

        if self.risk.min_risk_reward < 1.0:
            raise ValueError(f"min_risk_reward must be >= 1.0, got {self.risk.min_risk_reward}")

        if not (0 < self.risk.position_multiplier_min < self.risk.position_multiplier_max):
            raise ValueError(
                f"Invalid position multipliers: {self.risk.position_multiplier_min} - {self.risk.position_multiplier_max}"
            )

        if not (0 < self.risk.max_portfolio_drawdown <= 1.0):
            raise ValueError(f"max_portfolio_drawdown must be 0-1, got {self.risk.max_portfolio_drawdown}")

    def summary(self) -> str:
        """Get configuration summary"""
        lines = []
        lines.append("⚙️ ENTRY LOGIC CONFIGURATION")
        lines.append("=" * 60)

        lines.append("\n📊 Base Settings:")
        lines.append(f"  Min Confidence: {self.min_confidence}%")
        lines.append(f"  Use Simplified Filters: {self.use_simplified_filters}")
        lines.append(f"  Require Trend Alignment: {self.require_trend_alignment}")
        lines.append(f"  Require Volume Confirmation: {self.require_volume_confirmation}")

        lines.append("\n🎯 Risk Management:")
        lines.append(f"  Min R:R Ratio: {self.risk.min_risk_reward}")
        lines.append(f"  Stop Loss ATR: {self.risk.stop_loss_atr_multiplier}x")
        lines.append(f"  Max Portfolio Drawdown: {self.risk.max_portfolio_drawdown*100}%")
        lines.append(f"  Position Multiplier: {self.risk.position_multiplier_min} - {self.risk.position_multiplier_max}")

        lines.append("\n📈 Market Regime Adjustments:")
        lines.append(f"  Bull Adjustment: {self.regime.bull_confidence_adjustment:+d}")
        lines.append(f"  Bear Adjustment: {self.regime.bear_confidence_adjustment:+d}")
        lines.append(f"  High Vol Adjustment: {self.regime.high_vol_confidence_adjustment:+d}")

        lines.append("\n💧 Liquidity Tiers:")
        lines.append(f"  Large Cap: {self.filters.liquidity_large_cap/1e9:.1f}B VND")
        lines.append(f"  Mid Cap: {self.filters.liquidity_mid_cap/1e9:.1f}B VND")
        lines.append(f"  Small Cap: {self.filters.liquidity_small_cap/1e9:.1f}B VND")

        if self.use_simplified_filters:
            lines.append("\n🔍 SIMPLIFIED FILTERS (8-10 Core Filters):")
            lines.append(f"  Mandatory: {', '.join(self.weights.mandatory_filters)}")
            lines.append(f"  High Priority: {len(self.weights.high_priority)} filters")
            lines.append(f"  Medium Priority: {len(self.weights.medium_priority)} filters")

        return "\n".join(lines)


# Singleton instance
_entry_config = None


def get_entry_config(validate: bool = True) -> EntryLogicConfig:
    """
    Get entry logic configuration singleton

    Args:
        validate: Whether to validate config on load

    Returns:
        EntryLogicConfig instance
    """
    global _entry_config
    if _entry_config is None:
        _entry_config = EntryLogicConfig.from_env()
        if validate:
            _entry_config.validate()
    return _entry_config


# For backward compatibility
def get_default_config() -> EntryLogicConfig:
    """Get default entry logic configuration"""
    return EntryLogicConfig()


if __name__ == "__main__":
    print("Testing Entry Logic Configuration...")

    config = EntryLogicConfig()
    config.validate()

    print(config.summary())
    print("\n✅ Configuration valid!")
