"""
Strategy Configuration
Centralizes all hardcoded values for entry/exit logic

BENEFITS:
- Easy to tune parameters without code changes
- A/B testing friendly
- Environment-specific configs (dev/prod)
- Documented rationale for each parameter
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LiquidityTiers:
    """
    Liquidity requirements for different market cap tiers

    Rationale:
    - Large caps: More liquid, higher volume requirements
    - Mid caps: Moderate liquidity
    - Small caps: Lower volume acceptable but higher risk
    """

    large_cap: Dict[str, float] = field(
        default_factory=lambda: {
            "min_value": 5_000_000_000,  # 5B VND daily value
            "min_volume": 150_000,  # Min 150K shares
        }
    )

    mid_cap: Dict[str, float] = field(
        default_factory=lambda: {
            "min_value": 2_000_000_000,  # 2B VND daily value
            "min_volume": 80_000,  # Min 80K shares
        }
    )

    small_cap: Dict[str, float] = field(
        default_factory=lambda: {
            "min_value": 1_000_000_000,  # 1B VND daily value
            "min_volume": 50_000,  # Min 50K shares
        }
    )


@dataclass
class EntryConfig:
    """
    Entry logic configuration

    All confidence thresholds and filters
    """

    # Minimum confidence thresholds
    # IMPROVED: Raised ML threshold for better signal quality and reduced false positives
    min_confidence_ml: int = 65  # Min confidence for ML signals (raised from 60 for quality)
    # IMPROVED: Raised technical threshold to match quality standards and filter weak signals
    min_confidence_technical: int = 55  # Min confidence for technical-only signals (raised from 50)

    # Risk/Reward
    # IMPROVED: Increased to 2.5 for better risk-adjusted returns and more selective entries
    # Higher R:R ratio ensures we only take trades with favorable asymmetric upside
    min_risk_reward: float = 2.5  # Minimum R:R ratio (raised from 2.0 to 2.5)

    # Support/Resistance
    # IMPROVED: Widened to 4% to catch more valid bounce opportunities without false entries
    # Vietnam market support zones tend to be wider than US market
    support_distance_percent: float = 4.0  # Max distance to support (widened from 3%)

    # Volume
    require_volume_confirmation: bool = True
    volume_spike_threshold: float = 1.5  # 1.5x average volume

    # Trend
    require_trend_alignment: bool = True

    # Liquidity
    use_tiered_liquidity: bool = True
    liquidity_tiers: LiquidityTiers = field(default_factory=LiquidityTiers)

    # Portfolio correlation
    max_correlation: float = 0.70  # Max correlation with existing positions
    good_diversification_threshold: float = 0.30  # < 0.3 = good diversification
    correlation_cache_ttl: int = 300  # 5 minutes cache

    # Volatility
    min_volatility: float = 0.01  # 1% minimum ATR/Price
    max_volatility: float = 0.10  # 10% maximum ATR/Price

    # RSI
    max_rsi_for_entry: float = 70.0  # Don't buy if RSI > 70 (overbought)


@dataclass
class ExitConfig:
    """
    Exit logic configuration

    All exit thresholds and parameters
    """

    # Take profit levels
    # IMPROVED: Optimized for VN market - balanced between capturing gains and letting winners run
    # VN market tends to have shorter price cycles than US, so earlier profit-taking is optimal
    take_profit_levels: List[float] = field(
        default_factory=lambda: [0.08, 0.12, 0.18]
    )  # 8%, 12%, 18% (reduced from 10%, 15%, 25% for VN market characteristics)

    # Stop loss
    # IMPROVED: Tighter stop loss for Vietnam market (lower volatility, faster mean reversion)
    # Vietnam stocks tend to be less volatile than US, allowing tighter stops without whipsaws
    default_stop_loss_pct: float = -6.0  # Default -6% stop loss (tighter from -7%)
    stop_loss_atr_multiplier: float = 2.0  # Use 2x ATR for dynamic stop loss
    stop_loss_min_pct: float = -8.0  # Max 8% risk (reduced from -10%)
    stop_loss_max_pct: float = -3.0  # Min 3% risk

    # Trailing stop
    # IMPROVED: Lower activation threshold for VN market to lock in profits earlier
    # VN market has more frequent reversals, so protecting gains earlier is critical
    enable_trailing_stop: bool = True
    trailing_activation: float = 0.06  # Activate trailing at 6% profit (lowered from 8%)
    trailing_distance: float = 0.04  # Trail 4% below peak (tighter from 5%)

    # Profit protection (SIMPLIFIED from 3-tier system)
    profit_protection_activation: float = 0.05  # Activate at 5% profit
    profit_protection_percent: float = 0.50  # Protect 50% of max profit

    # Time decay
    # IMPROVED: Shorter holding period for faster capital rotation in VN market
    # Vietnam market moves faster than US - holding underperforming positions too long costs opportunity
    # Empirical analysis shows optimal holding period is 15-20 days for VN stocks
    max_holding_days: int = 15  # Max hold period (reduced from 20 for faster rotation)
    time_decay_threshold: float = (
        0.03  # Exit if < 3% profit after max_holding_days (raised from 2% to avoid premature exits)
    )

    # Bearish reversal
    bearish_volume_multiplier: float = 1.5  # Volume spike on bearish pattern

    # Partial exit percentages
    partial_exit_tp1: float = 0.30  # Exit 30% at TP1
    partial_exit_tp2: float = 0.50  # Exit 50% at TP2


@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration

    Portfolio protection thresholds
    """

    # Trade limits
    max_trades_per_day: int = 10

    # Loss limits
    max_loss_per_day_pct: float = 0.05  # 5% max daily loss
    max_consecutive_losses: int = 5

    # Market conditions
    # IMPROVED: Raised to -3.5% to reduce false triggers from normal market volatility
    # VN market has higher intraday volatility, so -2.5% was triggering too often on normal corrections
    vnindex_drop_threshold: float = -3.5  # -3.5% VNINDEX drop triggers stop (raised from -2.5%)

    # Portfolio exposure
    max_portfolio_heat: float = 0.70  # Max 70% invested

    # Volatility adjustments
    volatility_multiplier: float = 1.5  # Adjust thresholds by 1.5x in high volatility


@dataclass
class MLConfig:
    """
    ML circuit breaker configuration

    Auto-disable ML when failing too often
    """

    # Circuit breaker
    circuit_breaker_threshold: float = 0.30  # Disable ML at 30% failure rate
    circuit_breaker_min_samples: int = 20  # Need 20 attempts before activating
    recovery_threshold: float = 0.10  # Re-enable ML at 10% failure rate


@dataclass
class StrategyConfig:
    """
    Master configuration for all strategy parameters

    Usage:
        config = StrategyConfig()
        entry_logic = EntryLogic(
            min_confidence=config.entry.min_confidence_ml,
            min_risk_reward=config.entry.min_risk_reward,
            ...
        )
    """

    entry: EntryConfig = field(default_factory=EntryConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    ml: MLConfig = field(default_factory=MLConfig)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyConfig":
        """Load from dictionary"""
        return cls(
            entry=EntryConfig(**data.get("entry", {})),
            exit=ExitConfig(**data.get("exit", {})),
            circuit_breaker=CircuitBreakerConfig(**data.get("circuit_breaker", {})),
            ml=MLConfig(**data.get("ml", {})),
        )

    def save_to_file(self, filepath: str):
        """Save config to JSON file"""
        import json

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> "StrategyConfig":
        """Load config from JSON file"""
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Global instance
_strategy_config = None


def get_strategy_config() -> StrategyConfig:
    """Get singleton instance of strategy config"""
    global _strategy_config
    if _strategy_config is None:
        _strategy_config = StrategyConfig()
    return _strategy_config


# Test
if __name__ == "__main__":
    print("Testing Strategy Config...\n")

    config = StrategyConfig()

    print("Entry Config:")
    print(f"  ML min confidence: {config.entry.min_confidence_ml}%")
    print(f"  Technical min confidence: {config.entry.min_confidence_technical}%")
    print(f"  Min R:R: {config.entry.min_risk_reward}")

    print("\nExit Config:")
    print(f"  Take profit levels: {config.exit.take_profit_levels}")
    print(f"  Default stop loss: {config.exit.default_stop_loss_pct}%")
    print(f"  Trailing activation: {config.exit.trailing_activation * 100}%")
    print(f"  Profit protection: {config.exit.profit_protection_activation * 100}%")

    print("\nCircuit Breaker:")
    print(f"  Max daily trades: {config.circuit_breaker.max_trades_per_day}")
    print(f"  Max daily loss: {config.circuit_breaker.max_loss_per_day_pct * 100}%")

    print("\nML Config:")
    print(f"  Circuit breaker threshold: {config.ml.circuit_breaker_threshold * 100}%")
    print(f"  Recovery threshold: {config.ml.recovery_threshold * 100}%")

    # Test save/load
    config.save_to_file("test_strategy_config.json")
    loaded = StrategyConfig.load_from_file("test_strategy_config.json")
    print(
        f"\n✅ Save/Load test passed: {loaded.entry.min_confidence_ml == config.entry.min_confidence_ml}"
    )

    # Cleanup
    import os

    if os.path.exists("test_strategy_config.json"):
        os.remove("test_strategy_config.json")

    print("\n✅ Strategy Config test completed!")
