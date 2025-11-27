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
    Liquidity requirements for different market cap tiers - OPTIMIZED VERSION

    Rationale:
    - Large caps: High liquidity, strict requirements for better execution
    - Mid caps: Moderate liquidity, balanced requirements
    - Small caps: Lower volume acceptable but higher risk premium

    IMPROVED v3.0: Tightened thresholds for better trade quality
    """

    large_cap: Dict[str, float] = field(
        default_factory=lambda: {
            "min_value": 5_000_000_000,  # TIGHTENED: 5B VND minimum
            "min_volume": 150_000,  # TIGHTENED: 150K shares for better execution
        }
    )

    mid_cap: Dict[str, float] = field(
        default_factory=lambda: {
            "min_value": 2_000_000_000,  # TIGHTENED: 2B VND minimum
            "min_volume": 80_000,  # TIGHTENED: 80K shares
        }
    )

    small_cap: Dict[str, float] = field(
        default_factory=lambda: {
            "min_value": 1_000_000_000,  # TIGHTENED: 1B VND minimum
            "min_volume": 50_000,  # TIGHTENED: 50K shares
        }
    )


@dataclass
class EntryConfig:
    """
    Entry logic configuration - OPTIMIZED VERSION v3.0

    All confidence thresholds and filters - TIGHTENED for better win rate
    """

    # Minimum confidence thresholds - TIGHTENED for quality
    min_confidence_ml: int = 60  # TIGHTENED: ML signals need 60%+ confidence
    min_confidence_technical: int = 55  # TIGHTENED: Technical-only needs 55%+

    # Risk/Reward - TIGHTENED for profitability
    min_risk_reward: float = 2.0  # TIGHTENED: Minimum 2:1 R:R ratio

    # Support/Resistance - TIGHTENED
    support_distance_percent: float = 4.0  # TIGHTENED: Max 4% from support

    # Volume - ENABLED for confirmation
    require_volume_confirmation: bool = True  # ENABLED: Volume confirms momentum
    volume_spike_threshold: float = 1.3  # TIGHTENED: 1.3x average volume

    # Trend - ENABLED for trend following
    require_trend_alignment: bool = True  # ENABLED: Trade with the trend

    # Liquidity
    use_tiered_liquidity: bool = True
    liquidity_tiers: LiquidityTiers = field(default_factory=LiquidityTiers)

    # Portfolio correlation - TIGHTENED for diversification
    max_correlation: float = 0.70  # TIGHTENED: Max 0.70 correlation
    good_diversification_threshold: float = 0.30  # TIGHTENED: Good if < 0.30
    correlation_cache_ttl: int = 300  # 5 minutes cache

    # Volatility - TIGHTENED
    min_volatility: float = 0.01  # TIGHTENED: Min 1% volatility (avoid dead stocks)
    max_volatility: float = 0.08  # TIGHTENED: Max 8% volatility (avoid too risky)

    # RSI - TIGHTENED
    max_rsi_for_entry: float = 70.0  # TIGHTENED: Don't buy overbought stocks


@dataclass
class ExitConfig:
    """
    Exit logic configuration - OPTIMIZED VERSION v3.0

    All exit thresholds and parameters - Optimized for VN market characteristics
    """

    # Take profit levels - OPTIMIZED for VN market cycles
    # VN market has shorter cycles, earlier profit-taking is optimal
    take_profit_levels: List[float] = field(
        default_factory=lambda: [0.08, 0.15, 0.25]
    )  # 8%, 15%, 25% - balanced between capturing gains and letting winners run

    # Stop loss - TIGHTENED for capital preservation
    # VN market has ±7% daily limit, so -5% stop is reasonable
    default_stop_loss_pct: float = -5.0  # TIGHTENED: -5% stop loss
    stop_loss_atr_multiplier: float = 1.5  # TIGHTENED: 1.5x ATR (was 2.0)
    stop_loss_min_pct: float = -7.0  # Max 7% risk (matches VN price limit)
    stop_loss_max_pct: float = -3.0  # Min 3% risk

    # Trailing stop - TIGHTENED to lock profits earlier
    enable_trailing_stop: bool = True
    trailing_activation: float = 0.05  # TIGHTENED: Activate at 5% profit
    trailing_distance: float = 0.03  # TIGHTENED: Trail 3% below peak

    # Profit protection - IMPROVED
    profit_protection_activation: float = 0.04  # TIGHTENED: Activate at 4% profit
    profit_protection_percent: float = 0.60  # IMPROVED: Protect 60% of max profit

    # Time decay - OPTIMIZED for VN market rotation
    max_holding_days: int = 20  # BALANCED: 20 days max (not too short, not too long)
    time_decay_threshold: float = 0.02  # Exit if < 2% profit after max_holding_days

    # Bearish reversal
    bearish_volume_multiplier: float = 1.5  # Volume spike on bearish pattern

    # Partial exit percentages - IMPROVED for better risk management
    partial_exit_tp1: float = 0.50  # IMPROVED: Exit 50% at TP1 (lock in profits)
    partial_exit_tp2: float = 0.30  # Exit 30% at TP2
    # Remaining 20% rides to TP3 or trailing stop


@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration - OPTIMIZED VERSION v3.0

    Portfolio protection thresholds - TIGHTENED for better risk management
    """

    # Trade limits - TIGHTENED
    max_trades_per_day: int = 8  # TIGHTENED: Max 8 trades/day (avoid overtrading)

    # Loss limits - TIGHTENED
    max_loss_per_day_pct: float = 0.03  # TIGHTENED: 3% max daily loss (was 5%)
    max_consecutive_losses: int = 3  # TIGHTENED: 3 consecutive losses triggers pause

    # Market conditions - BALANCED
    # VN market: -2.5% is significant, -3% is serious
    vnindex_drop_threshold: float = -2.5  # TIGHTENED: -2.5% VNINDEX drop triggers stop
    vnindex_warning_threshold: float = -1.5  # NEW: Warning at -1.5%
    vnindex_caution_threshold: float = -2.0  # NEW: Caution mode at -2.0%

    # Portfolio exposure - TIGHTENED
    max_portfolio_heat: float = 0.60  # TIGHTENED: Max 60% invested (was 70%)
    max_single_position: float = 0.15  # NEW: Max 15% in single position
    max_sector_exposure: float = 0.30  # NEW: Max 30% in single sector

    # Volatility adjustments
    volatility_multiplier: float = 1.5  # Adjust thresholds by 1.5x in high volatility

    # Drawdown protection - NEW
    max_drawdown_pct: float = 0.12  # NEW: 12% max drawdown from peak
    drawdown_warning_pct: float = 0.08  # NEW: Warning at 8% drawdown


@dataclass
class MLConfig:
    """
    ML circuit breaker configuration - OPTIMIZED VERSION v3.0

    Auto-disable ML when failing too often - TIGHTENED for reliability
    """

    # Circuit breaker - TIGHTENED
    circuit_breaker_threshold: float = 0.15  # TIGHTENED: Disable ML at 15% failure rate
    circuit_breaker_min_samples: int = 30  # TIGHTENED: Need 30 attempts for reliability
    recovery_threshold: float = 0.05  # TIGHTENED: Re-enable ML at 5% failure rate

    # NEW: ML signal quality thresholds
    min_ml_confidence: float = 0.60  # Minimum ML confidence to consider
    ml_signal_weight: float = 0.70  # Weight of ML signal vs technical (70/30)
    technical_fallback_threshold: float = 0.55  # Technical confidence needed when ML fails


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
