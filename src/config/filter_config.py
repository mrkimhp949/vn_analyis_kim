# -*- coding: utf-8 -*-
"""
Filter Configuration for Entry Logic
Centralized configuration for all entry filter thresholds and adjustments

IMPROVEMENT (2025-12-01):
- Extracted magic numbers from entry_logic.py
- Makes thresholds visible and easier to tune
- Enables A/B testing and optimization
- Single source of truth for filter parameters
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RSIConfig:
    """RSI Filter Configuration"""

    # RSI thresholds
    overbought: int = 70
    oversold: int = 30
    warning_threshold: int = 65

    # Confidence adjustments
    overbought_penalty: int = -20  # Strong penalty for overbought
    oversold_bonus: int = +10  # Bonus for oversold
    neutral_bonus: int = +5  # Small bonus for neutral RSI


@dataclass
class VolatilityConfig:
    """Volatility Filter Configuration"""

    # ATR/Price ratio thresholds
    min_atr_ratio: float = 0.02  # 2%
    max_atr_ratio: float = 0.08  # 8%
    optimal_min: float = 0.03  # 3%
    optimal_max: float = 0.05  # 5%

    # Confidence adjustments
    too_high_penalty: int = -15
    optimal_bonus: int = +5
    too_low_penalty: int = -10  # Low volatility can also be bad (no movement)


@dataclass
class LiquidityConfig:
    """Liquidity Filter Configuration (Vietnam Market)"""

    # Tiered liquidity thresholds (VND daily value)
    large_cap: int = 10_000_000_000  # 10B VND
    mid_cap: int = 5_000_000_000  # 5B VND
    small_cap: int = 1_000_000_000  # 1B VND
    micro_cap: int = 500_000_000  # 500M VND

    # Minimum requirements
    min_avg_volume: int = 50_000  # shares
    min_daily_value: int = 1_000_000_000  # 1B VND (hard minimum)

    # Confidence adjustments
    critical_penalty: int = -999  # Block entry (insufficient liquidity)
    low_penalty: int = -15
    good_bonus: int = +5
    excellent_bonus: int = +10  # For large cap with great liquidity


@dataclass
class TrendConfig:
    """Trend Alignment Filter Configuration"""

    # EMA periods (read-only, calculated in indicators)
    ema_short: int = 20
    ema_long: int = 50
    ema_trend: int = 200

    # Trend strength thresholds (EMA separation %)
    weak_separation: float = 0.5  # < 0.5% separation = weak trend
    moderate_separation: float = 2.0  # 0.5-2% = moderate
    strong_separation: float = 5.0  # > 5% = strong trend

    # Confidence adjustments
    misalignment_penalty: int = -10
    weak_alignment_bonus: int = 0
    moderate_alignment_bonus: int = +5
    strong_alignment_bonus: int = +10


@dataclass
class VolumeConfig:
    """Volume Confirmation Filter Configuration"""

    # Volume surge thresholds (vs 20-day average)
    surge_threshold: float = 1.5  # 1.5x average = surge
    weak_threshold: float = 0.7  # < 0.7x average = weak

    # Confidence adjustments
    no_confirmation_penalty: int = -10
    weak_volume_penalty: int = -5
    surge_bonus: int = +5
    confirmed_bonus: int = +3

    # Small cap relaxation
    small_cap_penalty_reduction: int = 5  # Reduce penalty for small caps


@dataclass
class SupportResistanceConfig:
    """Support/Resistance Filter Configuration"""

    # Distance thresholds (% from current price)
    support_near_pct: float = 3.0  # Within 3% of support
    support_bounce_pct: float = 1.5  # Within 1.5% = bouncing
    resistance_near_pct: float = 3.0  # Within 3% of resistance

    # Confidence adjustments
    near_resistance_penalty: int = -15
    near_support_bonus: int = +10
    bouncing_from_support_bonus: int = +15  # Strong reversal signal


@dataclass
class VietnamPriceLimitConfig:
    """Vietnam Market Price Floor/Ceiling Configuration"""

    # Vietnam price limits are ±7% from reference price
    # We check at ±6.5% to avoid triggering right at limit
    floor_threshold_pct: float = 6.5  # Approaching floor
    ceiling_threshold_pct: float = 6.5  # Approaching ceiling

    # Actions
    ceiling_action: str = "BLOCK"  # Block entry at ceiling (too risky)
    floor_action: str = "WARN"  # Warn but allow entry at floor (reversal opportunity)

    # Confidence adjustments
    ceiling_penalty: int = -999  # Block entry
    floor_penalty: int = -20  # Strong penalty but don't block


@dataclass
class CorrelationConfig:
    """Portfolio Correlation Filter Configuration"""

    # Correlation thresholds
    max_correlation: float = 0.70  # > 0.7 = too correlated
    good_diversification: float = 0.30  # < 0.3 = well diversified
    excellent_diversification: float = 0.15  # < 0.15 = excellent

    # Lookback period for correlation calculation
    lookback_days: int = 60

    # Cache settings
    cache_ttl_seconds: int = 300  # 5 minutes

    # Confidence adjustments
    high_correlation_penalty: int = -20
    good_diversification_bonus: int = +10
    excellent_diversification_bonus: int = +15


@dataclass
class MarketRegimeConfig:
    """Market Regime Filter Configuration"""

    # Regime-based adjustment scaling
    # In BULL markets: scale penalties down (more signals)
    # In BEAR markets: scale penalties up (fewer, higher quality signals)
    bull_penalty_scale: float = 0.7  # 30% lighter penalties
    sideways_penalty_scale: float = 1.0  # No change
    bear_penalty_scale: float = 1.2  # 20% heavier penalties
    high_volatility_penalty_scale: float = 1.3  # 30% heavier penalties

    # Minimum regime confidence for regime-aware filtering
    min_regime_confidence: int = 70


@dataclass
class ConfidenceConfig:
    """Overall Confidence Adjustment Configuration"""

    # Minimum confidence thresholds
    min_confidence: int = 45  # Base minimum (relaxed from 55%)
    min_confidence_technical_only: int = 55  # Higher threshold for technical-only signals

    # Minimum risk/reward ratio
    min_risk_reward: float = 1.0  # After transaction costs

    # Maximum warnings before rejection
    max_warnings: int = 5

    # Confidence adjustment limits
    max_total_adjustment: int = 30  # Cap total adjustment at ±30 points


@dataclass
class FilterPerformanceConfig:
    """Filter Performance Tracking Configuration"""

    # Enable/disable filter performance tracking
    enable_tracking: bool = True

    # Minimum trades before adjusting filter based on performance
    min_trades_for_feedback: int = 50

    # Performance feedback strength
    # If filter performance is poor, how much to penalize
    poor_performance_penalty: int = -5  # Per 10% below 50% win rate
    good_performance_bonus: int = +3  # Per 10% above 60% win rate


class FilterConfigManager:
    """Manager for filter configurations"""

    def __init__(self):
        self.rsi = RSIConfig()
        self.volatility = VolatilityConfig()
        self.liquidity = LiquidityConfig()
        self.trend = TrendConfig()
        self.volume = VolumeConfig()
        self.support_resistance = SupportResistanceConfig()
        self.vietnam_limits = VietnamPriceLimitConfig()
        self.correlation = CorrelationConfig()
        self.regime = MarketRegimeConfig()
        self.confidence = ConfidenceConfig()
        self.performance = FilterPerformanceConfig()

    def get_all_configs(self) -> dict:
        """Get all configuration values as dict"""
        return {
            "rsi": self.rsi,
            "volatility": self.volatility,
            "liquidity": self.liquidity,
            "trend": self.trend,
            "volume": self.volume,
            "support_resistance": self.support_resistance,
            "vietnam_limits": self.vietnam_limits,
            "correlation": self.correlation,
            "regime": self.regime,
            "confidence": self.confidence,
            "performance": self.performance,
        }

    def export_to_dict(self) -> dict:
        """Export all configs to flat dict for logging/serialization"""
        result = {}
        for category, config in self.get_all_configs().items():
            for key, value in config.__dict__.items():
                result[f"{category}.{key}"] = value
        return result


# Singleton instance
_filter_config = None


def get_filter_config() -> FilterConfigManager:
    """Get filter configuration singleton"""
    global _filter_config
    if _filter_config is None:
        _filter_config = FilterConfigManager()
    return _filter_config


# Convenience function to reset config (for testing)
def reset_filter_config():
    """Reset filter configuration to defaults"""
    global _filter_config
    _filter_config = None
