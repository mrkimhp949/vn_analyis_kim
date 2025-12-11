"""
Strategies Package - Trading strategy components for Vietnam stock market.

This package contains:
- Entry logic and filters for trade signals
- Exit logic for position management
- Position sizing calculations
- Risk management

Modular Structure (v9.0+):
- entry_signal.py: EntrySignal dataclass and SignalStrength enum
- entry_filters.py: Filter implementations (price limit, trend, volume, etc.)
- entry_validators.py: Data validation and technical calculations
- technical_checks.py: Technical analysis check methods
- technical_scorers.py: Technical indicator scoring
- price_optimizer.py: Price and risk calculations
- sentiment_analyzer.py: Sentiment analysis
- entry_logic.py: Main orchestration logic (ImprovedEntryLogic) - Refactored version
"""

# Core entry components
from src.strategies.entry_signal import (
    SignalStrength,
    EntrySignal,
    create_no_signal,
)

from src.strategies.entry_filters import (
    FilterResult,
    PriceLimitResult,
    TrendResult,
    VolumeResult,
    RSIResult,
    BaseEntryFilter,
    VietnamPriceLimitFilter,
    TrendAlignmentFilter,
    VolumeConfirmationFilter,
    RSIFilter,
    VolatilityFilter,
    EntryFilterManager,
)

from src.strategies.entry_validators import (
    DataValidationResult,
    SignalValidationResult,
    RiskRewardResult,
    DataFrameValidator,
    MLSignalValidator,
    TechnicalConfidenceCalculator,
    SupportResistanceCalculator,
    RiskRewardCalculator,
)

# Technical analysis modules
from src.strategies.technical_checks import TechnicalChecker

from src.strategies.technical_scorers import TechnicalScorer

from src.strategies.price_optimizer import (
    PriceOptimizer,
    PriceCalculationResult,
    RiskRewardCalculator as PriceRiskRewardCalculator,
)

from src.strategies.sentiment_analyzer import (
    SentimentAnalyzer,
    VolumeAnalyzer,
)

# Main entry logic
from src.strategies.entry_logic import ImprovedEntryLogic

# Exit logic
from src.strategies.exit_logic import ImprovedExitStrategy

__all__ = [
    # Entry Signal
    "SignalStrength",
    "EntrySignal",
    "create_no_signal",
    # Filters
    "FilterResult",
    "BaseEntryFilter",
    "VietnamPriceLimitFilter",
    "TrendAlignmentFilter",
    "VolumeConfirmationFilter",
    "RSIFilter",
    "EntryFilterManager",
    # Validators
    "DataFrameValidator",
    "MLSignalValidator",
    "TechnicalConfidenceCalculator",
    "RiskRewardCalculator",
    # Technical Analysis
    "TechnicalChecker",
    "TechnicalScorer",
    # Price Optimization
    "PriceOptimizer",
    "PriceCalculationResult",
    "PriceRiskRewardCalculator",
    # Sentiment Analysis
    "SentimentAnalyzer",
    "VolumeAnalyzer",
    # Main logic
    "ImprovedEntryLogic",
    "ImprovedExitStrategy",
]
