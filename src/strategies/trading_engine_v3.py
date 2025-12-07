# -*- coding: utf-8 -*-
"""
Trading Engine v3.0 - Unified Entry + Position Sizing

Combines Entry Logic v3.0 and Position Sizing v3.0 into a single
trading engine optimized for Vietnam market.

Features:
- Simplified decision flow
- Full Vietnam market compliance
- Transaction cost awareness
- Adaptive behavior by market regime
- Real-time integration ready

Author: Trading Bot Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.strategies.entry_logic_v3 import (
    SimplifiedEntryLogicV3,
    EntrySignalV3,
    FilterPriority,
    SignalStrength,
    get_entry_logic_v3,
)
from src.strategies.position_sizing_v3 import (
    EnhancedPositionSizerV3,
    PositionSizeResult,
    PortfolioContext,
    get_position_sizer_v3,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """Complete trade decision with entry and position sizing."""

    # Decision
    should_trade: bool
    action: str  # BUY, SELL, HOLD

    # Entry details
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    # Position details
    shares: int
    position_value: float
    risk_amount: float

    # Metrics
    confidence: int
    risk_reward_ratio: float
    position_pct: float
    risk_pct: float

    # Signal info
    signal_strength: SignalStrength
    entry_signal: Optional[EntrySignalV3] = None
    position_result: Optional[PositionSizeResult] = None

    # Actionable info
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class TradingEngineV3:
    """
    Unified Trading Engine v3.0.

    Combines entry analysis and position sizing into a single
    decision-making engine.

    Usage:
        engine = TradingEngineV3(total_capital=500_000_000)
        decision = engine.analyze_trade(
            symbol="VCB",
            df=price_data,
            market_regime=regime,
            ml_signal=ml_signal,
        )

        if decision.should_trade:
            broker.buy(decision.symbol, decision.shares, decision.entry_price)
    """

    def __init__(
        self,
        total_capital: float = 100_000_000,
        # Entry config
        base_min_confidence: int = 55,
        min_risk_reward: float = 2.0,
        # Position config
        max_risk_per_trade: float = 0.015,
        max_position_pct: float = 0.12,
        # Features
        use_adaptive_thresholds: bool = True,
        use_kelly: bool = True,
        include_transaction_costs: bool = True,
    ):
        """
        Initialize trading engine.

        Args:
            total_capital: Total portfolio capital in VND
            base_min_confidence: Base minimum confidence for entry
            min_risk_reward: Minimum risk/reward ratio
            max_risk_per_trade: Maximum risk per trade (1.5% default)
            max_position_pct: Maximum position size (12% default)
            use_adaptive_thresholds: Adjust thresholds by regime
            use_kelly: Use Kelly Criterion for sizing
            include_transaction_costs: Include costs in calculations
        """
        self.total_capital = total_capital

        # Initialize entry logic
        self._entry_logic = SimplifiedEntryLogicV3(
            base_min_confidence=base_min_confidence,
            min_risk_reward=min_risk_reward,
            max_position_pct=max_position_pct,
            use_adaptive_thresholds=use_adaptive_thresholds,
            include_transaction_costs=include_transaction_costs,
        )

        # Initialize position sizer
        self._position_sizer = EnhancedPositionSizerV3(
            total_capital=total_capital,
            max_risk_per_trade=max_risk_per_trade,
            max_position_pct=max_position_pct,
            use_kelly=use_kelly,
            use_volatility_adjustment=True,
            use_correlation_check=True,
        )

        # Portfolio tracking
        self._portfolio_context: Optional[PortfolioContext] = None

    def analyze_trade(
        self,
        symbol: str,
        df: pd.DataFrame,
        market_regime: Optional[Dict] = None,
        ml_signal: Optional[Dict] = None,
        foreign_flow: Optional[Dict] = None,
        sector: Optional[str] = None,
        current_positions: Optional[Dict] = None,
    ) -> TradeDecision:
        """
        Analyze trade opportunity and return complete decision.

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame with indicators
            market_regime: Market regime info
            ml_signal: ML model signal
            foreign_flow: Foreign flow data
            sector: Stock sector
            current_positions: Current portfolio positions

        Returns:
            TradeDecision with complete analysis
        """
        # Step 1: Entry Analysis
        entry_signal = self._entry_logic.analyze_entry(
            symbol=symbol,
            df=df,
            ml_signal=ml_signal,
            market_regime=market_regime,
            foreign_flow=foreign_flow,
            current_positions=current_positions,
        )

        # If no entry signal, return hold decision
        if not entry_signal.should_enter:
            return self._create_hold_decision(
                symbol=symbol,
                entry_signal=entry_signal,
            )

        # Step 2: Position Sizing
        position_result = self._position_sizer.calculate_position_size(
            symbol=symbol,
            entry_price=entry_signal.entry_price,
            stop_loss=entry_signal.stop_loss,
            confidence=entry_signal.confidence,
            market_regime=market_regime,
            sector=sector,
            df=df,
            portfolio_context=self._portfolio_context,
        )

        # If position invalid, return hold decision
        if not position_result.is_valid:
            return self._create_hold_decision(
                symbol=symbol,
                entry_signal=entry_signal,
                reason=position_result.validation_message,
            )

        # Step 3: Build trade decision
        return self._create_trade_decision(
            symbol=symbol,
            entry_signal=entry_signal,
            position_result=position_result,
        )

    def _create_trade_decision(
        self,
        symbol: str,
        entry_signal: EntrySignalV3,
        position_result: PositionSizeResult,
    ) -> TradeDecision:
        """Create trade decision from entry and position results."""
        return TradeDecision(
            should_trade=True,
            action="BUY",
            symbol=symbol,
            entry_price=entry_signal.entry_price,
            stop_loss=entry_signal.stop_loss,
            take_profit_1=entry_signal.take_profit_1,
            take_profit_2=entry_signal.take_profit_2,
            shares=position_result.shares,
            position_value=position_result.value,
            risk_amount=position_result.risk_amount,
            confidence=entry_signal.confidence,
            risk_reward_ratio=entry_signal.risk_reward_ratio,
            position_pct=position_result.position_pct,
            risk_pct=position_result.risk_pct,
            signal_strength=entry_signal.strength,
            entry_signal=entry_signal,
            position_result=position_result,
            reasons=entry_signal.reasons,
            warnings=entry_signal.warnings + position_result.warnings,
            recommendations=entry_signal.recommendations,
        )

    def _create_hold_decision(
        self,
        symbol: str,
        entry_signal: EntrySignalV3,
        reason: str = None,
    ) -> TradeDecision:
        """Create hold decision."""
        warnings = entry_signal.warnings.copy()
        if reason:
            warnings.append(reason)

        return TradeDecision(
            should_trade=False,
            action="HOLD",
            symbol=symbol,
            entry_price=entry_signal.entry_price,
            stop_loss=0,
            take_profit_1=0,
            take_profit_2=0,
            shares=0,
            position_value=0,
            risk_amount=0,
            confidence=entry_signal.confidence,
            risk_reward_ratio=0,
            position_pct=0,
            risk_pct=0,
            signal_strength=entry_signal.strength,
            entry_signal=entry_signal,
            warnings=warnings,
        )

    # =========================================================================
    # PORTFOLIO MANAGEMENT
    # =========================================================================

    def update_portfolio(
        self,
        total_capital: float,
        available_cash: float,
        current_exposure: float,
        positions: Dict[str, Dict] = None,
        sector_exposures: Dict[str, float] = None,
    ) -> None:
        """
        Update portfolio context for position sizing.

        Args:
            total_capital: Total portfolio value
            available_cash: Available cash for trading
            current_exposure: Current invested amount
            positions: Current positions {symbol: {value, sector, ...}}
            sector_exposures: Sector exposure percentages
        """
        self._portfolio_context = PortfolioContext(
            total_capital=total_capital,
            available_cash=available_cash,
            current_exposure=current_exposure,
            current_positions=positions or {},
            sector_exposures=sector_exposures or {},
        )

        self._position_sizer.update_portfolio_context(self._portfolio_context)
        self.total_capital = total_capital

    def record_trade_result(self, trade: Dict) -> None:
        """
        Record trade result for Kelly calculation.

        Args:
            trade: Dict with symbol, entry_price, exit_price, pnl, pnl_pct
        """
        self._position_sizer.record_trade(trade)

    def get_trade_statistics(self) -> Dict:
        """Get trade statistics for performance analysis."""
        return self._position_sizer.get_trade_statistics()

    # =========================================================================
    # BATCH ANALYSIS
    # =========================================================================

    def analyze_watchlist(
        self,
        watchlist: List[Dict],  # [{symbol, df, sector}, ...]
        market_regime: Optional[Dict] = None,
        foreign_flow: Optional[Dict] = None,
        max_trades: int = 5,
    ) -> List[TradeDecision]:
        """
        Analyze multiple symbols and return top trade opportunities.

        Args:
            watchlist: List of {symbol, df, sector} dicts
            market_regime: Market regime info
            foreign_flow: Foreign flow data
            max_trades: Maximum number of trades to return

        Returns:
            List of TradeDecision sorted by confidence
        """
        decisions = []

        for item in watchlist:
            symbol = item.get("symbol")
            df = item.get("df")
            sector = item.get("sector")
            ml_signal = item.get("ml_signal")

            if df is None or df.empty:
                continue

            decision = self.analyze_trade(
                symbol=symbol,
                df=df,
                market_regime=market_regime,
                ml_signal=ml_signal,
                foreign_flow=foreign_flow,
                sector=sector,
            )

            if decision.should_trade:
                decisions.append(decision)

        # Sort by confidence and return top N
        decisions.sort(key=lambda d: d.confidence, reverse=True)
        return decisions[:max_trades]


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================

_trading_engine: Optional[TradingEngineV3] = None


def get_trading_engine_v3(total_capital: float = 100_000_000) -> TradingEngineV3:
    """Get singleton trading engine v3."""
    global _trading_engine
    if _trading_engine is None:
        _trading_engine = TradingEngineV3(total_capital=total_capital)
    return _trading_engine


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_analyze(
    symbol: str,
    df: pd.DataFrame,
    total_capital: float = 100_000_000,
    market_regime: Optional[Dict] = None,
    ml_signal: Optional[Dict] = None,
) -> TradeDecision:
    """
    Quick trade analysis for a single symbol.

    Convenience function for simple use cases.
    """
    engine = get_trading_engine_v3(total_capital)
    return engine.analyze_trade(
        symbol=symbol,
        df=df,
        market_regime=market_regime,
        ml_signal=ml_signal,
    )
