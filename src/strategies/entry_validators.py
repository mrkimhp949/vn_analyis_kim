# -*- coding: utf-8 -*-
"""
Entry Validators Module - Data validation and technical checks for entry analysis.

This module contains validation logic used by ImprovedEntryLogic:
- DataFrame validation
- ML signal validation
- Technical confidence calculation
- Support/Resistance calculations
- Stop loss and take profit calculations

Extracted from entry_logic.py for better modularity and testability.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.config.constants import (
    TECH_ONLY_MIN_CONFIDENCE,
    TECH_SCORE_GOOD,
    TECH_SCORE_HIGH,
    TECH_SCORE_LOW,
    TECH_SCORE_MODERATE,
    TECH_SCORE_POOR,
    TOTAL_TRANSACTION_COST,
    DEFAULT_SLIPPAGE,
)
from src.config.exceptions import DataQualityError
from src.utils.validation import DataValidator
from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT DATA CLASSES
# =============================================================================


@dataclass
class DataValidationResult:
    """Result from DataFrame validation."""

    is_valid: bool
    error_message: str = ""
    row_count: int = 0
    has_required_columns: bool = True
    missing_columns: List[str] = None

    def __post_init__(self):
        if self.missing_columns is None:
            self.missing_columns = []


@dataclass
class SignalValidationResult:
    """Result from ML signal validation."""

    is_valid: bool
    signal_type: str = "HOLD"
    base_confidence: float = 0.0
    current_price: float = 0.0
    error_message: str = ""
    is_technical_only: bool = False


@dataclass
class RiskRewardResult:
    """Result from risk/reward calculation."""

    is_valid: bool
    stop_loss: float = 0.0
    take_profit_targets: List[float] = None
    risk_reward_ratio: float = 0.0
    risk_amount: float = 0.0
    reward_amount: float = 0.0
    error_message: str = ""

    def __post_init__(self):
        if self.take_profit_targets is None:
            self.take_profit_targets = []


@dataclass
class SupportResistanceLevel:
    """Support and resistance levels."""

    support: float
    resistance: float
    distance_to_support_pct: float
    distance_to_resistance_pct: float


# =============================================================================
# DATA VALIDATORS
# =============================================================================


class DataFrameValidator:
    """Validates DataFrame for entry analysis requirements."""

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
    OPTIONAL_COLUMNS = ["rsi", "atr", "ema_20", "ema_50", "ema_200", "obv"]
    MIN_ROWS = 50

    @classmethod
    def validate(cls, df: pd.DataFrame, min_rows: int = None) -> DataValidationResult:
        """
        Validate DataFrame for entry analysis.

        Args:
            df: DataFrame to validate
            min_rows: Minimum required rows (default: 50)

        Returns:
            DataValidationResult with validation outcome
        """
        min_rows = min_rows or cls.MIN_ROWS

        if df is None:
            return DataValidationResult(
                is_valid=False,
                error_message="DataFrame is None",
            )

        if df.empty:
            return DataValidationResult(
                is_valid=False,
                error_message="DataFrame is empty",
                row_count=0,
            )

        # Check row count
        row_count = len(df)
        if row_count < min_rows:
            return DataValidationResult(
                is_valid=False,
                error_message=f"Insufficient data: {row_count} < {min_rows} rows",
                row_count=row_count,
            )

        # Check required columns
        missing_columns = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]

        if missing_columns:
            return DataValidationResult(
                is_valid=False,
                error_message=f"Missing required columns: {missing_columns}",
                row_count=row_count,
                has_required_columns=False,
                missing_columns=missing_columns,
            )

        # Check for NaN in critical columns
        nan_columns = [col for col in cls.REQUIRED_COLUMNS if df[col].isna().any()]

        if nan_columns:
            logger.warning(f"NaN values found in columns: {nan_columns}")

        return DataValidationResult(
            is_valid=True,
            row_count=row_count,
            has_required_columns=True,
        )


class MLSignalValidator:
    """Validates ML signals for entry analysis."""

    @classmethod
    def validate(
        cls,
        ml_signal: Optional[Dict],
        df: pd.DataFrame,
        min_confidence: int = 45,
    ) -> SignalValidationResult:
        """
        Validate ML signal with technical analysis fallback.

        Args:
            ml_signal: ML signal dictionary or None
            df: DataFrame for technical fallback
            min_confidence: Minimum confidence threshold

        Returns:
            SignalValidationResult with validation outcome
        """
        current_price = safe_get_latest(df, "close", 0)

        if ml_signal is None:
            # Use technical analysis fallback
            return cls._technical_fallback(df, current_price, min_confidence)

        # Validate ML signal
        signal_type = ml_signal.get("signal", "HOLD")
        confidence = ml_signal.get("confidence", 0)

        if signal_type != "BUY":
            return SignalValidationResult(
                is_valid=False,
                signal_type=signal_type,
                error_message=f"Signal = {signal_type}",
                current_price=current_price,
            )

        if confidence < min_confidence:
            return SignalValidationResult(
                is_valid=False,
                signal_type=signal_type,
                base_confidence=confidence,
                error_message=f"Confidence too low: {confidence}% < {min_confidence}%",
                current_price=current_price,
            )

        return SignalValidationResult(
            is_valid=True,
            signal_type=signal_type,
            base_confidence=confidence,
            current_price=current_price,
            is_technical_only=False,
        )

    @classmethod
    def _technical_fallback(
        cls,
        df: pd.DataFrame,
        current_price: float,
        min_confidence: int,
    ) -> SignalValidationResult:
        """Calculate technical signal when ML is unavailable."""
        logger.info("ML signal unavailable - using technical analysis fallback")

        tech_confidence = TechnicalConfidenceCalculator.calculate(df)

        if tech_confidence < TECH_ONLY_MIN_CONFIDENCE:
            return SignalValidationResult(
                is_valid=False,
                error_message=f"Technical confidence low: {tech_confidence:.0f}%",
                current_price=current_price,
                is_technical_only=True,
            )

        # Determine signal from technicals
        signal_type = TechnicalConfidenceCalculator.get_signal_type(df)

        if signal_type != "BUY":
            return SignalValidationResult(
                is_valid=False,
                signal_type=signal_type,
                error_message=f"Technical signal = {signal_type}",
                current_price=current_price,
                is_technical_only=True,
            )

        return SignalValidationResult(
            is_valid=True,
            signal_type="BUY",
            base_confidence=tech_confidence,
            current_price=current_price,
            is_technical_only=True,
        )


# =============================================================================
# TECHNICAL CALCULATIONS
# =============================================================================


class TechnicalConfidenceCalculator:
    """Calculates technical analysis confidence score."""

    @classmethod
    def calculate(cls, df: pd.DataFrame) -> float:
        """
        Calculate technical confidence score (0-100).

        Factors:
        - Trend strength (EMA alignment)
        - RSI level
        - Volume trend
        - Price action

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            Confidence score 0-100
        """
        if len(df) < 20:
            return 50.0  # Neutral

        score = 0.0
        max_score = 0.0

        # 1. Trend strength (30 points)
        trend_score = cls._calculate_trend_score(df)
        score += trend_score
        max_score += 30

        # 2. RSI score (20 points)
        rsi_score = cls._calculate_rsi_score(df)
        score += rsi_score
        max_score += 20

        # 3. Volume score (20 points)
        volume_score = cls._calculate_volume_score(df)
        score += volume_score
        max_score += 20

        # 4. Price action score (30 points)
        price_action_score = cls._calculate_price_action_score(df)
        score += price_action_score
        max_score += 30

        # Normalize to 0-100
        confidence = (score / max_score) * 100 if max_score > 0 else 50.0

        return round(confidence, 1)

    @classmethod
    def get_signal_type(cls, df: pd.DataFrame) -> str:
        """
        Determine signal type from technicals.

        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        if len(df) < 20:
            return "HOLD"

        close = df["close"]
        ema20 = close.ewm(span=20, adjust=False).mean()

        latest_price = close.iloc[-1]
        latest_ema20 = ema20.iloc[-1]
        prev_price = close.iloc[-2]
        prev_ema20 = ema20.iloc[-2]

        # Price crossing above EMA20
        if prev_price <= prev_ema20 and latest_price > latest_ema20:
            return "BUY"

        # Price crossing below EMA20
        if prev_price >= prev_ema20 and latest_price < latest_ema20:
            return "SELL"

        # Trend continuation
        if latest_price > latest_ema20:
            rsi = safe_get_latest(df, "rsi", 50)
            if rsi < 70:  # Not overbought
                return "BUY"

        return "HOLD"

    @classmethod
    def _calculate_trend_score(cls, df: pd.DataFrame) -> float:
        """Calculate trend score (max 30 points)."""
        close = df["close"]
        ema20 = close.ewm(span=20, adjust=False).mean()

        latest_price = close.iloc[-1]
        latest_ema20 = ema20.iloc[-1]

        if latest_price > latest_ema20:
            # Check if EMA20 is rising
            if latest_ema20 > ema20.iloc[-5]:
                return 30.0  # Strong uptrend
            return 20.0  # Weak uptrend
        else:
            return 5.0  # Downtrend

    @classmethod
    def _calculate_rsi_score(cls, df: pd.DataFrame) -> float:
        """Calculate RSI score (max 20 points)."""
        if "rsi" not in df.columns:
            return 10.0

        rsi = safe_get_latest(df, "rsi", 50)

        if rsi < 30:
            return 20.0  # Oversold
        elif rsi < 50:
            return 15.0  # Good for entry
        elif rsi < 70:
            return 10.0  # Neutral
        else:
            return 0.0  # Overbought

    @classmethod
    def _calculate_volume_score(cls, df: pd.DataFrame) -> float:
        """Calculate volume score (max 20 points)."""
        current_volume = safe_get_latest(df, "volume", 0)
        avg_volume = safe_rolling_operation(df, "volume", 20, "mean", 1)

        if avg_volume == 0:
            return 10.0

        ratio = current_volume / avg_volume

        if ratio >= 1.5:
            return 20.0  # Volume surge
        elif ratio >= 1.0:
            return 15.0  # Normal volume
        elif ratio >= 0.7:
            return 10.0  # Low volume
        else:
            return 5.0  # Very low volume

    @classmethod
    def _calculate_price_action_score(cls, df: pd.DataFrame) -> float:
        """Calculate price action score (max 30 points)."""
        if len(df) < 5:
            return 15.0

        # Recent price movement
        close = df["close"]
        price_5d_ago = close.iloc[-5]
        current_price = close.iloc[-1]

        change_pct = ((current_price - price_5d_ago) / price_5d_ago) * 100

        if change_pct > 5:
            return 30.0  # Strong up
        elif change_pct > 2:
            return 25.0  # Moderate up
        elif change_pct > 0:
            return 20.0  # Slight up
        elif change_pct > -2:
            return 10.0  # Flat
        else:
            return 5.0  # Down


class SupportResistanceCalculator:
    """Calculates support and resistance levels."""

    @classmethod
    def calculate(
        cls, df: pd.DataFrame, current_price: float, lookback: int = 20
    ) -> SupportResistanceLevel:
        """
        Calculate support and resistance levels.

        Args:
            df: DataFrame with OHLCV
            current_price: Current price
            lookback: Number of periods to look back

        Returns:
            SupportResistanceLevel with calculated levels
        """
        if len(df) < lookback:
            return SupportResistanceLevel(
                support=current_price * 0.95,
                resistance=current_price * 1.05,
                distance_to_support_pct=5.0,
                distance_to_resistance_pct=5.0,
            )

        support = df["low"].tail(lookback).min()
        resistance = df["high"].tail(lookback).max()

        distance_to_support = ((current_price - support) / support) * 100
        distance_to_resistance = ((resistance - current_price) / current_price) * 100

        return SupportResistanceLevel(
            support=support,
            resistance=resistance,
            distance_to_support_pct=distance_to_support,
            distance_to_resistance_pct=distance_to_resistance,
        )

    @classmethod
    def is_bouncing_from_support(
        cls,
        df: pd.DataFrame,
        support_level: float,
        current_price: float,
        threshold_pct: float = 2.0,
    ) -> bool:
        """
        Check if price is bouncing from support level.

        Args:
            df: DataFrame with OHLCV
            support_level: Support price level
            current_price: Current price
            threshold_pct: Proximity threshold percentage

        Returns:
            True if bouncing from support
        """
        if len(df) < 5:
            return False

        recent_low = df["low"].tail(5).min()

        # Check if recent low touched support
        if abs(recent_low - support_level) / support_level > threshold_pct / 100:
            return False

        # Check if price is moving up from support
        prev_close = df["close"].iloc[-2]
        if current_price > prev_close * 1.01:  # 1% up
            return True

        return False


class RiskRewardCalculator:
    """Calculates stop loss, take profit, and risk/reward ratios."""

    @classmethod
    def calculate(
        cls,
        entry_price: float,
        atr: float,
        support_level: Optional[float] = None,
        atr_multiplier: float = 2.0,
        min_stop_pct: float = 3.0,
        max_stop_pct: float = 10.0,
        take_profit_ratios: List[float] = None,
        transaction_cost_pct: float = None,
    ) -> RiskRewardResult:
        """
        Calculate stop loss and take profit levels.

        Args:
            entry_price: Entry price
            atr: Average True Range
            support_level: Optional support level for stop loss
            atr_multiplier: ATR multiplier for stop loss
            min_stop_pct: Minimum stop loss percentage
            max_stop_pct: Maximum stop loss percentage
            take_profit_ratios: R:R ratios for take profit targets
            transaction_cost_pct: Transaction cost percentage

        Returns:
            RiskRewardResult with calculated levels
        """
        if take_profit_ratios is None:
            take_profit_ratios = [1.5, 3.0, 5.0]

        if transaction_cost_pct is None:
            transaction_cost_pct = TOTAL_TRANSACTION_COST + DEFAULT_SLIPPAGE

        if entry_price <= 0:
            return RiskRewardResult(
                is_valid=False,
                error_message="Invalid entry price",
            )

        # Calculate stop loss
        atr_stop = entry_price - (atr * atr_multiplier)

        # Use support level if available and reasonable
        if support_level and support_level < entry_price:
            support_stop = support_level * 0.99  # 1% below support
            stop_loss = max(atr_stop, support_stop)
        else:
            stop_loss = atr_stop

        # Enforce min/max stop loss distance
        min_stop_distance = entry_price * (min_stop_pct / 100)
        max_stop_distance = entry_price * (max_stop_pct / 100)

        stop_distance = entry_price - stop_loss

        if stop_distance < min_stop_distance:
            stop_loss = entry_price - min_stop_distance
        elif stop_distance > max_stop_distance:
            stop_loss = entry_price - max_stop_distance

        # Validate stop loss
        if stop_loss >= entry_price:
            return RiskRewardResult(
                is_valid=False,
                error_message=f"Stop loss {stop_loss:.0f} must be below entry {entry_price:.0f}",
            )

        # Calculate risk
        price_risk = entry_price - stop_loss
        entry_cost = entry_price * transaction_cost_pct
        stop_exit_cost = stop_loss * transaction_cost_pct
        total_risk = price_risk + entry_cost + stop_exit_cost

        if total_risk <= 0:
            return RiskRewardResult(
                is_valid=False,
                error_message="Risk calculation error",
            )

        # Calculate take profit targets
        take_profit_targets = []
        for ratio in take_profit_ratios:
            # Target = Entry + (Risk * Ratio)
            target_before_costs = entry_price + (price_risk * ratio)
            take_profit_targets.append(round(target_before_costs, 0))

        # Calculate R:R using second target (TP2)
        if len(take_profit_targets) >= 2:
            tp2 = take_profit_targets[1]
            exit_cost = tp2 * transaction_cost_pct
            reward = (tp2 - entry_price) - entry_cost - exit_cost
            risk_reward_ratio = reward / total_risk if total_risk > 0 else 0
        else:
            risk_reward_ratio = 0

        return RiskRewardResult(
            is_valid=True,
            stop_loss=round(stop_loss, 0),
            take_profit_targets=take_profit_targets,
            risk_reward_ratio=round(risk_reward_ratio, 2),
            risk_amount=round(total_risk, 0),
            reward_amount=round(reward, 0) if "reward" in locals() else 0,
        )
