"""
Abstract Base Classes for Trading Strategies

This module defines interfaces (ABCs) for various trading strategies,
enabling polymorphism and easier testing/swapping of implementations.

Benefits:
- Loose coupling: Easy to swap strategy implementations
- Polymorphism: Can use different strategies interchangeably
- Testing: Easy to create mock strategies
- Documentation: Clear contract for strategy implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import pandas as pd


# =============================================================================
# DATA CLASSES (Shared across strategies)
# =============================================================================


@dataclass
class EntryAnalysisResult:
    """Result from entry strategy analysis"""

    should_enter: bool
    confidence: int  # 0-100
    entry_price: float
    stop_loss: float
    take_profit_targets: List[float]
    position_size_recommendation: Optional[int] = None
    reasons: List[str] = None
    warnings: List[str] = None
    ml_confidence: Optional[float] = None
    news_sentiment: Optional[float] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class ExitSignal:
    """Signal from exit strategy"""

    should_exit: bool
    exit_type: str  # "FULL", "PARTIAL"
    exit_percent: float  # Percentage to exit (0-100)
    reason: str
    confidence: float  # 0-1
    urgency: int  # 1-5 (5 = most urgent)
    recommended_price: Optional[float] = None


@dataclass
class PositionSizeResult:
    """Result from position sizing strategy"""

    shares: int
    value: float
    risk_amount: float
    risk_percent: float
    max_loss: float
    position_percent: float
    recommended_entries: List[Dict] = None
    warnings: List[str] = None
    adjustments: Dict[str, float] = None

    def __post_init__(self):
        if self.recommended_entries is None:
            self.recommended_entries = []
        if self.warnings is None:
            self.warnings = []
        if self.adjustments is None:
            self.adjustments = {}


# =============================================================================
# ENTRY STRATEGY INTERFACE
# =============================================================================


class EntryStrategy(ABC):
    """
    Abstract base class for entry strategies

    Implementations must define how to analyze entry conditions
    for a given stock.
    """

    @abstractmethod
    def analyze_entry(
        self,
        symbol: str,
        df: pd.DataFrame,
        ml_signal: Optional[Dict] = None,
        news_sentiment: Optional[float] = None,
        market_regime: Optional[Dict] = None,
    ) -> EntryAnalysisResult:
        """
        Analyze whether to enter a position

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV and indicators
            ml_signal: Optional ML prediction signal
            news_sentiment: Optional news sentiment score (-1 to 1)
            market_regime: Optional market regime info

        Returns:
            EntryAnalysisResult with entry decision and details

        Example:
            >>> strategy = TrendFollowingEntry()
            >>> result = strategy.analyze_entry("VNM", df)
            >>> if result.should_enter:
            >>>     print(f"Enter at {result.entry_price}, SL: {result.stop_loss}")
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging/identification"""
        pass


class TrendFollowingEntry(EntryStrategy):
    """
    Example implementation: Trend Following Entry Strategy

    Enters positions when:
    - Price above EMAs (20, 50, 200)
    - RSI in healthy range
    - Volume confirmation
    - Near support level
    """

    def __init__(self, min_confidence: int = 60):
        self.min_confidence = min_confidence

    def analyze_entry(
        self,
        symbol: str,
        df: pd.DataFrame,
        ml_signal: Optional[Dict] = None,
        news_sentiment: Optional[float] = None,
        market_regime: Optional[Dict] = None,
    ) -> EntryAnalysisResult:
        """Implement trend following logic"""
        # This would contain actual implementation
        # For now,示范 structure
        return EntryAnalysisResult(
            should_enter=False,
            confidence=50,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit_targets=[],
            reasons=["Example implementation"],
        )

    def get_strategy_name(self) -> str:
        return "TrendFollowingEntry"


class MeanReversionEntry(EntryStrategy):
    """
    Example implementation: Mean Reversion Entry Strategy

    Enters positions when:
    - Price significantly below moving average
    - RSI oversold
    - Near strong support level
    """

    def analyze_entry(
        self,
        symbol: str,
        df: pd.DataFrame,
        ml_signal: Optional[Dict] = None,
        news_sentiment: Optional[float] = None,
        market_regime: Optional[Dict] = None,
    ) -> EntryAnalysisResult:
        """Implement mean reversion logic"""
        return EntryAnalysisResult(
            should_enter=False,
            confidence=50,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit_targets=[],
            reasons=["Example implementation"],
        )

    def get_strategy_name(self) -> str:
        return "MeanReversionEntry"


# =============================================================================
# EXIT STRATEGY INTERFACE
# =============================================================================


class ExitStrategy(ABC):
    """
    Abstract base class for exit strategies

    Implementations must define how to determine exit signals
    for existing positions.
    """

    @abstractmethod
    def check_exit(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float,
        df: pd.DataFrame,
        ml_signal: Optional[Dict] = None,
        market_regime: Optional[Dict] = None,
    ) -> ExitSignal:
        """
        Check if position should be exited

        Args:
            symbol: Stock symbol
            position: Position data dict
            current_price: Current market price
            df: DataFrame with OHLCV and indicators
            ml_signal: Optional ML prediction signal
            market_regime: Optional market regime info

        Returns:
            ExitSignal with exit decision

        Example:
            >>> strategy = StopLossExit()
            >>> signal = strategy.check_exit("VNM", position, current_price, df)
            >>> if signal.should_exit:
            >>>     print(f"Exit: {signal.reason}")
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging/identification"""
        pass


class StopLossTakeProfitExit(ExitStrategy):
    """
    Example implementation: Stop Loss & Take Profit Exit

    Exits when:
    - Price hits stop loss
    - Price hits take profit targets
    - Trailing stop activated
    """

    def __init__(
        self, stop_loss_pct: float = 0.05, take_profit_pct: float = 0.10
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def check_exit(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float,
        df: pd.DataFrame,
        ml_signal: Optional[Dict] = None,
        market_regime: Optional[Dict] = None,
    ) -> ExitSignal:
        """Implement stop loss / take profit logic"""
        return ExitSignal(
            should_exit=False,
            exit_type="FULL",
            exit_percent=100.0,
            reason="Example implementation",
            confidence=0.5,
            urgency=3,
        )

    def get_strategy_name(self) -> str:
        return "StopLossTakeProfitExit"


class TimeBasedExit(ExitStrategy):
    """
    Example implementation: Time-based Exit

    Exits when:
    - Position held for maximum days
    - Profit below threshold after certain time
    """

    def __init__(self, max_holding_days: int = 30):
        self.max_holding_days = max_holding_days

    def check_exit(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float,
        df: pd.DataFrame,
        ml_signal: Optional[Dict] = None,
        market_regime: Optional[Dict] = None,
    ) -> ExitSignal:
        """Implement time-based exit logic"""
        return ExitSignal(
            should_exit=False,
            exit_type="FULL",
            exit_percent=100.0,
            reason="Example implementation",
            confidence=0.5,
            urgency=2,
        )

    def get_strategy_name(self) -> str:
        return "TimeBasedExit"


# =============================================================================
# POSITION SIZING STRATEGY INTERFACE
# =============================================================================


class PositionSizingStrategy(ABC):
    """
    Abstract base class for position sizing strategies

    Implementations must define how to calculate optimal position size.
    """

    @abstractmethod
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: int,
        total_capital: float,
        current_portfolio_risk: float,
        correlation_with_portfolio: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size

        Args:
            symbol: Stock symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price
            take_profit: Take profit target
            confidence: Entry confidence (0-100)
            total_capital: Total trading capital
            current_portfolio_risk: Current portfolio risk percentage
            correlation_with_portfolio: Correlation with existing positions

        Returns:
            PositionSizeResult with size and risk details

        Example:
            >>> strategy = FixedRiskSizing(risk_per_trade=0.02)
            >>> result = strategy.calculate_position_size(
            >>>     "VNM", 100000, 95000, 110000, 80, 100_000_000, 0.05
            >>> )
            >>> print(f"Buy {result.shares} shares")
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging/identification"""
        pass


class FixedRiskSizing(PositionSizingStrategy):
    """
    Example implementation: Fixed Risk Position Sizing

    Sizes positions to risk fixed percentage of capital per trade.
    """

    def __init__(self, risk_per_trade: float = 0.02):
        """
        Args:
            risk_per_trade: Percentage of capital to risk (0.02 = 2%)
        """
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: int,
        total_capital: float,
        current_portfolio_risk: float,
        correlation_with_portfolio: Optional[float] = None,
    ) -> PositionSizeResult:
        """Implement fixed risk sizing"""
        risk_amount = total_capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share > 0:
            shares = int(risk_amount / risk_per_share)
            shares = (shares // 100) * 100  # Round to lots of 100
        else:
            shares = 0

        return PositionSizeResult(
            shares=shares,
            value=shares * entry_price,
            risk_amount=shares * risk_per_share,
            risk_percent=(shares * risk_per_share / total_capital) * 100,
            max_loss=shares * risk_per_share,
            position_percent=(shares * entry_price / total_capital) * 100,
        )

    def get_strategy_name(self) -> str:
        return "FixedRiskSizing"


class KellyCriterionSizing(PositionSizingStrategy):
    """
    Example implementation: Kelly Criterion Position Sizing

    Sizes positions using Kelly Criterion formula based on win rate.
    """

    def __init__(self, kelly_fraction: float = 0.5):
        """
        Args:
            kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly for safety)
        """
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: int,
        total_capital: float,
        current_portfolio_risk: float,
        correlation_with_portfolio: Optional[float] = None,
    ) -> PositionSizeResult:
        """Implement Kelly criterion sizing"""
        # Simplified example - would need historical win rate data
        return PositionSizeResult(
            shares=0,
            value=0.0,
            risk_amount=0.0,
            risk_percent=0.0,
            max_loss=0.0,
            position_percent=0.0,
            warnings=["Example implementation - requires historical data"],
        )

    def get_strategy_name(self) -> str:
        return "KellyCriterionSizing"


# =============================================================================
# STRATEGY COMBINER (Composite Pattern)
# =============================================================================


class CompositeExitStrategy(ExitStrategy):
    """
    Composite pattern: Combine multiple exit strategies

    Checks all strategies and exits if ANY returns a strong signal.

    Example:
        >>> composite = CompositeExitStrategy([
        >>>     StopLossTakeProfitExit(),
        >>>     TimeBasedExit(),
        >>>     MLSignalExit(),
        >>> ])
        >>> signal = composite.check_exit(...)
    """

    def __init__(self, strategies: List[ExitStrategy]):
        self.strategies = strategies

    def check_exit(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float,
        df: pd.DataFrame,
        ml_signal: Optional[Dict] = None,
        market_regime: Optional[Dict] = None,
    ) -> ExitSignal:
        """Check all strategies, return first strong exit signal"""

        for strategy in self.strategies:
            signal = strategy.check_exit(
                symbol, position, current_price, df, ml_signal, market_regime
            )

            if signal.should_exit and signal.urgency >= 4:
                # High urgency signal - exit immediately
                return signal

        # If no high-urgency signals, check for medium urgency
        for strategy in self.strategies:
            signal = strategy.check_exit(
                symbol, position, current_price, df, ml_signal, market_regime
            )

            if signal.should_exit and signal.urgency >= 3:
                return signal

        # No exit signals
        return ExitSignal(
            should_exit=False,
            exit_type="NONE",
            exit_percent=0.0,
            reason="No exit conditions met",
            confidence=0.0,
            urgency=0,
        )

    def get_strategy_name(self) -> str:
        strategy_names = [s.get_strategy_name() for s in self.strategies]
        return f"Composite({', '.join(strategy_names)})"
