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
    min_confidence_ml: int = 60  # Min confidence for ML signals
    min_confidence_technical: int = 50  # Min confidence for technical-only signals (RAISED from 40)

    # Risk/Reward
    min_risk_reward: float = 2.0  # Minimum R:R ratio

    # Support/Resistance
    support_distance_percent: float = 3.0  # Max distance to support

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
    take_profit_levels: List[float] = field(
        default_factory=lambda: [0.10, 0.15, 0.25]
    )  # 10%, 15%, 25%

    # Stop loss
    default_stop_loss_pct: float = -7.0  # Default -7% stop loss
    stop_loss_atr_multiplier: float = 2.0  # Use 2x ATR for dynamic stop loss
    stop_loss_min_pct: float = -10.0  # Max 10% risk
    stop_loss_max_pct: float = -3.0  # Min 3% risk

    # Trailing stop
    enable_trailing_stop: bool = True
    trailing_activation: float = 0.08  # Activate trailing at 8% profit
    trailing_distance: float = 0.05  # Trail 5% below peak

    # Profit protection (SIMPLIFIED from 3-tier system)
    profit_protection_activation: float = 0.05  # Activate at 5% profit
    profit_protection_percent: float = 0.50  # Protect 50% of max profit

    # Time decay
    max_holding_days: int = 20  # Max hold period
    time_decay_threshold: float = 0.02  # Exit if < 2% profit after max_holding_days

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
    vnindex_drop_threshold: float = -2.5  # -2.5% VNINDEX drop triggers stop

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
