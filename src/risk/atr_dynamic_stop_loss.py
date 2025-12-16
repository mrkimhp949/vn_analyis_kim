# -*- coding: utf-8 -*-
"""
Per-Symbol ATR-Based Dynamic Stop Loss Manager

This module provides sophisticated stop loss management with:
- Per-symbol volatility tracking using ATR
- Dynamic stop loss adjustment based on market conditions
- Vietnam market-specific adjustments (price limits, sessions)
- Beta-adjusted stop loss for high-volatility stocks
- Integration with existing exit logic

Usage:
    from src.risk.atr_dynamic_stop_loss import (
        get_dynamic_stop_loss_manager,
        ATRDynamicStopLossManager,
    )
    
    manager = get_dynamic_stop_loss_manager()
    
    # Get stop loss for a position
    stop_info = manager.calculate_stop_loss(
        symbol="VNM",
        entry_price=90000,
        df=stock_data,
    )
    
    # Update trailing stop
    new_stop = manager.update_trailing_stop(
        symbol="VNM",
        current_price=95000,
        highest_price=97000,
    )
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


class StopLossType(Enum):
    """Types of stop loss strategies"""

    ATR_BASED = "atr_based"  # Based on ATR volatility
    SUPPORT_BASED = "support_based"  # Based on support levels
    PERCENTAGE = "percentage"  # Fixed percentage
    HYBRID = "hybrid"  # Combination of above
    BETA_ADJUSTED = "beta_adjusted"  # Adjusted for stock beta


class MarketCondition(Enum):
    """Current market condition"""

    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"


# Default configurations
DEFAULT_CONFIG = {
    # ATR settings
    "atr_period": 14,
    "atr_multiplier_base": 2.0,
    "atr_multiplier_volatile": 2.5,
    "atr_multiplier_calm": 1.5,
    # Percentage bounds
    "min_stop_pct": 0.03,  # 3% minimum
    "max_stop_pct": 0.10,  # 10% maximum
    "upcom_max_stop_pct": 0.12,  # 12% for UPCoM (higher volatility)
    # Trailing stop
    "trailing_activation_pct": 0.05,  # Activate at 5% profit
    "trailing_atr_multiplier": 1.5,
    # Beta adjustments
    "high_beta_threshold": 1.2,
    "very_high_beta_threshold": 1.5,
    "low_beta_threshold": 0.8,
    "high_beta_multiplier": 1.3,
    "low_beta_multiplier": 0.8,
    # Vietnam market specifics
    "price_limit_buffer": 0.005,  # 0.5% buffer from price limits
    "session_adjustments": {
        "ATO": 1.2,  # Wider stop during ATO
        "ATC": 1.2,  # Wider stop during ATC
        "CONTINUOUS": 1.0,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class StopLossResult:
    """Result of stop loss calculation"""

    stop_loss_price: float
    stop_loss_type: StopLossType
    stop_loss_pct: float

    # Details
    atr_value: float = 0.0
    atr_multiplier: float = 2.0
    beta: Optional[float] = None
    support_level: Optional[float] = None

    # Risk metrics
    risk_amount: float = 0.0
    position_risk_pct: float = 0.0

    # Adjustments applied
    adjustments: List[str] = field(default_factory=list)

    # Validation
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "stop_loss_price": self.stop_loss_price,
            "stop_loss_type": self.stop_loss_type.value,
            "stop_loss_pct": self.stop_loss_pct,
            "atr_value": self.atr_value,
            "atr_multiplier": self.atr_multiplier,
            "beta": self.beta,
            "risk_amount": self.risk_amount,
            "adjustments": self.adjustments,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
        }


@dataclass
class TrailingStopState:
    """State of trailing stop for a position"""

    symbol: str
    entry_price: float
    initial_stop: float
    current_stop: float
    highest_price: float

    # Tracking
    times_updated: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    activation_triggered: bool = False

    # ATR info
    current_atr: float = 0.0
    atr_multiplier: float = 1.5


@dataclass
class SymbolVolatilityProfile:
    """Volatility profile for a symbol"""

    symbol: str

    # ATR stats
    current_atr: float = 0.0
    avg_atr_20d: float = 0.0
    atr_percentile: float = 50.0  # Current ATR percentile

    # Volatility metrics
    daily_volatility: float = 0.0
    weekly_volatility: float = 0.0

    # Beta
    beta: float = 1.0
    beta_calculation_date: Optional[datetime] = None

    # Classification
    volatility_class: str = "normal"  # low, normal, high, extreme

    # Trend
    trend_direction: str = "neutral"  # up, down, neutral
    trend_strength: float = 0.0

    last_updated: datetime = field(default_factory=datetime.now)


# =============================================================================
# ATR CALCULATOR
# =============================================================================


class ATRCalculator:
    """Calculate ATR and related volatility metrics"""

    @staticmethod
    def calculate_atr(
        df: pd.DataFrame,
        period: int = 14,
    ) -> float:
        """
        Calculate Average True Range.

        Args:
            df: DataFrame with high, low, close columns
            period: ATR period (default 14)

        Returns:
            Current ATR value
        """
        if df is None or len(df) < period + 1:
            return 0.0

        try:
            # Check required columns
            required = ["high", "low", "close"]
            if not all(col in df.columns for col in required):
                return 0.0

            # Calculate True Range components
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate ATR using EMA (Wilder's smoothing)
            atr = true_range.ewm(span=period, adjust=False).mean()

            return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

        except Exception as e:
            logger.debug(f"ATR calculation error: {e}")
            return 0.0

    @staticmethod
    def calculate_atr_percentile(
        df: pd.DataFrame,
        period: int = 14,
        lookback: int = 252,
    ) -> float:
        """
        Calculate current ATR's percentile rank over lookback period.

        Returns:
            Percentile rank (0-100)
        """
        if df is None or len(df) < lookback:
            return 50.0

        try:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.ewm(span=period, adjust=False).mean()

            current_atr = atr.iloc[-1]
            historical_atr = atr.tail(lookback)

            percentile = (historical_atr < current_atr).mean() * 100
            return float(percentile)

        except Exception:
            return 50.0

    @staticmethod
    def calculate_volatility_metrics(
        df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate various volatility metrics."""
        if df is None or len(df) < 20:
            return {}

        try:
            returns = df["close"].pct_change().dropna()

            daily_vol = returns.std() * 100
            weekly_vol = returns.rolling(5).std().iloc[-1] * 100 if len(returns) >= 5 else daily_vol
            monthly_vol = (
                returns.rolling(20).std().iloc[-1] * 100 if len(returns) >= 20 else daily_vol
            )

            return {
                "daily_volatility": daily_vol,
                "weekly_volatility": weekly_vol,
                "monthly_volatility": monthly_vol,
                "annualized_volatility": daily_vol * np.sqrt(252),
            }

        except Exception:
            return {}


# =============================================================================
# BETA CALCULATOR
# =============================================================================


class BetaCalculator:
    """Calculate stock beta relative to market index"""

    @staticmethod
    def calculate_beta(
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        lookback: int = 60,
    ) -> Optional[float]:
        """
        Calculate stock beta against market index.

        Args:
            stock_df: Stock price DataFrame
            index_df: Index (e.g., VNINDEX) DataFrame
            lookback: Lookback period in days

        Returns:
            Beta value or None if cannot calculate
        """
        if stock_df is None or index_df is None:
            return None

        if len(stock_df) < lookback or len(index_df) < lookback:
            return None

        try:
            stock_returns = stock_df["close"].pct_change().tail(lookback).dropna()
            index_returns = index_df["close"].pct_change().tail(lookback).dropna()

            # Align data
            common_dates = stock_returns.index.intersection(index_returns.index)
            if len(common_dates) < 30:
                return None

            stock_r = stock_returns.loc[common_dates]
            index_r = index_returns.loc[common_dates]

            # Calculate beta using covariance method
            covariance = np.cov(stock_r, index_r)[0, 1]
            variance = np.var(index_r)

            if variance == 0:
                return None

            beta = covariance / variance

            # Clamp to reasonable range
            return max(0.0, min(3.0, beta))

        except Exception as e:
            logger.debug(f"Beta calculation error: {e}")
            return None

    @staticmethod
    def get_beta_adjustment_factor(
        beta: Optional[float],
        config: Dict = None,
    ) -> Tuple[float, str]:
        """
        Get stop loss adjustment factor based on beta.

        Returns:
            (multiplier, reason) tuple
        """
        if config is None:
            config = DEFAULT_CONFIG

        if beta is None:
            return 1.0, "No beta data"

        if beta >= config["very_high_beta_threshold"]:
            return 1.4, f"Very high beta ({beta:.2f})"
        elif beta >= config["high_beta_threshold"]:
            return config["high_beta_multiplier"], f"High beta ({beta:.2f})"
        elif beta <= config["low_beta_threshold"]:
            return config["low_beta_multiplier"], f"Low beta ({beta:.2f})"
        else:
            return 1.0, f"Normal beta ({beta:.2f})"


# =============================================================================
# SUPPORT LEVEL DETECTOR
# =============================================================================


class SupportLevelDetector:
    """Detect support levels for stop loss placement"""

    @staticmethod
    def find_nearest_support(
        df: pd.DataFrame,
        current_price: float,
        lookback: int = 60,
    ) -> Optional[float]:
        """
        Find nearest support level below current price.

        Args:
            df: Price DataFrame
            current_price: Current stock price
            lookback: Days to look back

        Returns:
            Support level price or None
        """
        if df is None or len(df) < lookback:
            return None

        try:
            recent = df.tail(lookback)
            lows = recent["low"].values

            # Find local minima
            supports = []
            for i in range(2, len(lows) - 2):
                if (
                    lows[i] < lows[i - 1]
                    and lows[i] < lows[i + 1]
                    and lows[i] < lows[i - 2]
                    and lows[i] < lows[i + 2]
                ):
                    supports.append(lows[i])

            # Add recent swing lows
            for i in range(5, len(lows)):
                window_low = min(lows[i - 5 : i])
                if window_low < current_price * 0.97:  # At least 3% below
                    supports.append(window_low)

            if not supports:
                return None

            # Find nearest support below current price
            valid_supports = [s for s in supports if s < current_price * 0.98]
            if not valid_supports:
                return None

            return max(valid_supports)  # Highest support below current price

        except Exception as e:
            logger.debug(f"Support detection error: {e}")
            return None

    @staticmethod
    def find_resistance(
        df: pd.DataFrame,
        current_price: float,
        lookback: int = 60,
    ) -> Optional[float]:
        """Find nearest resistance level above current price."""
        if df is None or len(df) < lookback:
            return None

        try:
            recent = df.tail(lookback)
            highs = recent["high"].values

            # Find local maxima
            resistances = []
            for i in range(2, len(highs) - 2):
                if (
                    highs[i] > highs[i - 1]
                    and highs[i] > highs[i + 1]
                    and highs[i] > highs[i - 2]
                    and highs[i] > highs[i + 2]
                ):
                    resistances.append(highs[i])

            if not resistances:
                return None

            # Find nearest resistance above current price
            valid_resistances = [r for r in resistances if r > current_price * 1.02]
            if not valid_resistances:
                return None

            return min(valid_resistances)

        except Exception:
            return None


# =============================================================================
# MAIN DYNAMIC STOP LOSS MANAGER
# =============================================================================


class ATRDynamicStopLossManager:
    """
    Per-Symbol ATR-Based Dynamic Stop Loss Manager

    Features:
    - Individual volatility tracking per symbol
    - Multiple stop loss calculation methods
    - Trailing stop management
    - Vietnam market adjustments
    - Integration with existing risk management

    Usage:
        manager = ATRDynamicStopLossManager()

        # Calculate initial stop loss
        result = manager.calculate_stop_loss(
            symbol="VNM",
            entry_price=90000,
            df=stock_data,
            index_df=vnindex_data,
        )

        # Update trailing stop
        new_stop = manager.update_trailing_stop(
            symbol="VNM",
            current_price=95000,
            highest_price=97000,
        )
    """

    CACHE_FILE = "data_cache/stop_loss_profiles.json"

    def __init__(
        self,
        config: Optional[Dict] = None,
        storage_dir: Optional[str] = None,
    ):
        self._config = {**DEFAULT_CONFIG, **(config or {})}
        self._storage_dir = Path(storage_dir) if storage_dir else Path(".")
        self._lock = RLock()

        # Per-symbol volatility profiles
        self._profiles: Dict[str, SymbolVolatilityProfile] = {}

        # Active trailing stops
        self._trailing_stops: Dict[str, TrailingStopState] = {}

        # ATR calculator
        self._atr_calc = ATRCalculator()

        # Beta calculator
        self._beta_calc = BetaCalculator()

        # Support detector
        self._support_detector = SupportLevelDetector()

        # Load cached profiles
        self._load_profiles()

        logger.info("📉 ATR Dynamic Stop Loss Manager initialized")

    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        df: Optional[pd.DataFrame] = None,
        index_df: Optional[pd.DataFrame] = None,
        exchange: str = "HOSE",
        market_regime: Optional[str] = None,
        session: Optional[str] = None,
        use_support: bool = True,
        force_refresh: bool = False,
    ) -> StopLossResult:
        """
        Calculate optimal stop loss for a position.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            df: Price DataFrame for the stock
            index_df: Index DataFrame for beta calculation
            exchange: Exchange (HOSE, HNX, UPCOM)
            market_regime: Current market regime
            session: Trading session (ATO, ATC, CONTINUOUS)
            use_support: Whether to consider support levels
            force_refresh: Force recalculation of profile

        Returns:
            StopLossResult with stop loss details
        """
        symbol = symbol.upper()
        adjustments = []
        warnings = []

        # Update/get volatility profile
        profile = self._get_or_update_profile(symbol, df, index_df, force_refresh)

        # Calculate base ATR stop
        atr = profile.current_atr if profile.current_atr > 0 else self._atr_calc.calculate_atr(df)

        if atr <= 0:
            # Fallback to percentage-based stop
            stop_pct = self._config["min_stop_pct"]
            if exchange == "UPCOM":
                stop_pct = min(stop_pct * 1.5, self._config["upcom_max_stop_pct"])

            stop_price = entry_price * (1 - stop_pct)
            warnings.append("No ATR data, using minimum percentage stop")

            return StopLossResult(
                stop_loss_price=stop_price,
                stop_loss_type=StopLossType.PERCENTAGE,
                stop_loss_pct=stop_pct,
                is_valid=True,
                warnings=warnings,
            )

        # Base ATR multiplier
        atr_multiplier = self._config["atr_multiplier_base"]

        # Adjust for volatility class
        if profile.volatility_class == "high":
            atr_multiplier = self._config["atr_multiplier_volatile"]
            adjustments.append("High volatility: wider stop")
        elif profile.volatility_class == "low":
            atr_multiplier = self._config["atr_multiplier_calm"]
            adjustments.append("Low volatility: tighter stop")

        # Beta adjustment
        beta = profile.beta
        beta_factor, beta_reason = self._beta_calc.get_beta_adjustment_factor(beta, self._config)
        if beta_factor != 1.0:
            atr_multiplier *= beta_factor
            adjustments.append(beta_reason)

        # Market regime adjustment
        if market_regime == "BEAR":
            atr_multiplier *= 0.85  # Tighter in bear market
            adjustments.append("Bear market: tighter stop")
        elif market_regime == "HIGH_VOLATILITY":
            atr_multiplier *= 1.2  # Wider in high vol
            adjustments.append("High vol regime: wider stop")

        # Session adjustment
        if session and session in self._config["session_adjustments"]:
            session_factor = self._config["session_adjustments"][session]
            if session_factor != 1.0:
                atr_multiplier *= session_factor
                adjustments.append(f"{session} session adjustment")

        # Calculate ATR-based stop
        atr_stop = entry_price - (atr * atr_multiplier)

        # Find support level
        support_level = None
        if use_support and df is not None:
            support_level = self._support_detector.find_nearest_support(df, entry_price)

        # Choose final stop loss
        stop_type = StopLossType.ATR_BASED
        if support_level and support_level > atr_stop:
            # Support level provides better protection
            stop_price = support_level
            stop_type = StopLossType.HYBRID
            adjustments.append(f"Using support at {support_level:,.0f}")
        else:
            stop_price = atr_stop

        # Apply min/max constraints
        min_stop = entry_price * (1 - self._config["min_stop_pct"])
        max_stop = entry_price * (1 - self._config["max_stop_pct"])

        if exchange == "UPCOM":
            max_stop = entry_price * (1 - self._config["upcom_max_stop_pct"])

        if stop_price > min_stop:
            stop_price = min_stop
            adjustments.append(f"Minimum {self._config['min_stop_pct']*100:.0f}% applied")
        elif stop_price < max_stop:
            stop_price = max_stop
            adjustments.append(f"Maximum {self._config['max_stop_pct']*100:.0f}% applied")

        # Calculate metrics
        stop_pct = (entry_price - stop_price) / entry_price
        risk_amount = entry_price - stop_price

        # Validate
        if stop_price >= entry_price:
            stop_price = entry_price * 0.95
            warnings.append("Stop price adjusted (was above entry)")

        if stop_price <= 0:
            stop_price = entry_price * 0.90
            warnings.append("Stop price adjusted (was <= 0)")

        return StopLossResult(
            stop_loss_price=float(stop_price),
            stop_loss_type=stop_type,
            stop_loss_pct=float(stop_pct),
            atr_value=float(atr),
            atr_multiplier=float(atr_multiplier),
            beta=float(beta) if beta else None,
            support_level=support_level,
            risk_amount=float(risk_amount),
            adjustments=adjustments,
            is_valid=True,
            warnings=warnings,
        )

    def init_trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        initial_stop: float,
        df: Optional[pd.DataFrame] = None,
    ) -> TrailingStopState:
        """
        Initialize trailing stop for a position.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            initial_stop: Initial stop loss price
            df: Price DataFrame for ATR calculation

        Returns:
            TrailingStopState
        """
        symbol = symbol.upper()

        atr = self._atr_calc.calculate_atr(df) if df is not None else 0.0

        state = TrailingStopState(
            symbol=symbol,
            entry_price=entry_price,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            highest_price=entry_price,
            current_atr=atr,
            atr_multiplier=self._config["trailing_atr_multiplier"],
        )

        with self._lock:
            self._trailing_stops[symbol] = state

        return state

    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        highest_price: Optional[float] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, bool, str]:
        """
        Update trailing stop for a position.

        Args:
            symbol: Stock symbol
            current_price: Current price
            highest_price: Highest price since entry (optional)
            df: Updated price DataFrame

        Returns:
            (new_stop_price, stop_triggered, message)
        """
        symbol = symbol.upper()

        with self._lock:
            if symbol not in self._trailing_stops:
                return 0.0, False, "No trailing stop initialized"

            state = self._trailing_stops[symbol]

        # Update highest price
        if highest_price and highest_price > state.highest_price:
            state.highest_price = highest_price
        elif current_price > state.highest_price:
            state.highest_price = current_price

        # Check if trailing should be activated
        profit_pct = (current_price - state.entry_price) / state.entry_price
        activation_threshold = self._config["trailing_activation_pct"]

        if not state.activation_triggered:
            if profit_pct >= activation_threshold:
                state.activation_triggered = True
                logger.debug(
                    f"Trailing stop activated for {symbol} at {profit_pct*100:.1f}% profit"
                )
            else:
                # Not yet activated, return initial stop
                if current_price <= state.initial_stop:
                    return (
                        state.initial_stop,
                        True,
                        f"Initial stop hit at {state.initial_stop:,.0f}",
                    )
                return state.current_stop, False, "Trailing not yet activated"

        # Calculate new trailing stop
        if df is not None and len(df) >= 14:
            atr = self._atr_calc.calculate_atr(df)
            if atr > 0:
                state.current_atr = atr

        if state.current_atr > 0:
            # ATR-based trailing
            new_stop = state.highest_price - (state.current_atr * state.atr_multiplier)
        else:
            # Percentage-based fallback
            trailing_pct = 0.05  # 5% trailing
            new_stop = state.highest_price * (1 - trailing_pct)

        # Only move stop up, never down
        if new_stop > state.current_stop:
            state.current_stop = new_stop
            state.times_updated += 1
            state.last_update = datetime.now()

        # Ensure stop is above entry (lock in profit)
        if profit_pct >= 0.10:  # > 10% profit
            min_stop = state.entry_price * 1.02  # At least 2% profit
            if state.current_stop < min_stop:
                state.current_stop = min_stop

        # Check if stop is hit
        if current_price <= state.current_stop:
            return state.current_stop, True, f"Trailing stop hit at {state.current_stop:,.0f}"

        with self._lock:
            self._trailing_stops[symbol] = state

        return state.current_stop, False, f"Trailing stop: {state.current_stop:,.0f}"

    def remove_trailing_stop(self, symbol: str):
        """Remove trailing stop for a position."""
        symbol = symbol.upper()
        with self._lock:
            if symbol in self._trailing_stops:
                del self._trailing_stops[symbol]

    def get_volatility_profile(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        force_refresh: bool = False,
    ) -> SymbolVolatilityProfile:
        """Get volatility profile for a symbol."""
        return self._get_or_update_profile(symbol, df, None, force_refresh)

    def get_stop_loss_recommendation(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        account_value: float,
        df: Optional[pd.DataFrame] = None,
        max_risk_pct: float = 0.02,  # 2% max risk per trade
    ) -> Dict[str, Any]:
        """
        Get comprehensive stop loss recommendation.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            position_size: Position size (shares or value)
            account_value: Total account value
            df: Price DataFrame
            max_risk_pct: Maximum risk percentage per trade

        Returns:
            Recommendation dict with stop loss options
        """
        result = self.calculate_stop_loss(symbol, entry_price, df)

        # Calculate risk metrics
        position_value = position_size * entry_price
        risk_per_share = entry_price - result.stop_loss_price
        total_risk = position_size * risk_per_share
        risk_of_account = total_risk / account_value if account_value > 0 else 0

        # Check if risk is acceptable
        is_acceptable = risk_of_account <= max_risk_pct

        # Calculate alternative positions sizes
        max_position_risk = account_value * max_risk_pct
        suggested_position_size = max_position_risk / risk_per_share if risk_per_share > 0 else 0

        return {
            "symbol": symbol,
            "entry_price": entry_price,
            "stop_loss": result.to_dict(),
            "risk_metrics": {
                "risk_per_share": risk_per_share,
                "total_risk": total_risk,
                "risk_of_account": risk_of_account,
                "is_acceptable": is_acceptable,
            },
            "position_sizing": {
                "current_size": position_size,
                "suggested_size": suggested_position_size,
                "size_adjustment": suggested_position_size - position_size,
            },
            "recommendation": "ACCEPTABLE" if is_acceptable else "REDUCE_SIZE",
        }

    def _get_or_update_profile(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        index_df: Optional[pd.DataFrame] = None,
        force_refresh: bool = False,
    ) -> SymbolVolatilityProfile:
        """Get or update volatility profile for a symbol."""
        symbol = symbol.upper()

        with self._lock:
            if symbol in self._profiles and not force_refresh:
                profile = self._profiles[symbol]
                # Check freshness
                if (datetime.now() - profile.last_updated).total_seconds() < 3600:
                    return profile

        # Create or update profile
        profile = SymbolVolatilityProfile(symbol=symbol)

        if df is not None and len(df) >= 14:
            # Calculate ATR
            profile.current_atr = self._atr_calc.calculate_atr(df)
            profile.avg_atr_20d = self._atr_calc.calculate_atr(df.tail(20))
            profile.atr_percentile = self._atr_calc.calculate_atr_percentile(df)

            # Calculate volatility metrics
            vol_metrics = self._atr_calc.calculate_volatility_metrics(df)
            profile.daily_volatility = vol_metrics.get("daily_volatility", 0)
            profile.weekly_volatility = vol_metrics.get("weekly_volatility", 0)

            # Classify volatility
            if profile.atr_percentile >= 80:
                profile.volatility_class = "extreme"
            elif profile.atr_percentile >= 65:
                profile.volatility_class = "high"
            elif profile.atr_percentile <= 20:
                profile.volatility_class = "low"
            else:
                profile.volatility_class = "normal"

            # Calculate trend
            if len(df) >= 20:
                sma20 = df["close"].tail(20).mean()
                current = df["close"].iloc[-1]
                if current > sma20 * 1.02:
                    profile.trend_direction = "up"
                    profile.trend_strength = (current / sma20 - 1) * 100
                elif current < sma20 * 0.98:
                    profile.trend_direction = "down"
                    profile.trend_strength = (1 - current / sma20) * 100
                else:
                    profile.trend_direction = "neutral"
                    profile.trend_strength = 0

        # Calculate beta if index data available
        if df is not None and index_df is not None:
            beta = self._beta_calc.calculate_beta(df, index_df)
            if beta is not None:
                profile.beta = beta
                profile.beta_calculation_date = datetime.now()

        profile.last_updated = datetime.now()

        with self._lock:
            self._profiles[symbol] = profile
            self._save_profiles()

        return profile

    def _load_profiles(self):
        """Load cached volatility profiles."""
        cache_path = self._storage_dir / self.CACHE_FILE
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    for symbol, profile_data in data.items():
                        self._profiles[symbol] = SymbolVolatilityProfile(
                            symbol=symbol,
                            current_atr=profile_data.get("current_atr", 0),
                            avg_atr_20d=profile_data.get("avg_atr_20d", 0),
                            atr_percentile=profile_data.get("atr_percentile", 50),
                            daily_volatility=profile_data.get("daily_volatility", 0),
                            weekly_volatility=profile_data.get("weekly_volatility", 0),
                            beta=profile_data.get("beta", 1.0),
                            volatility_class=profile_data.get("volatility_class", "normal"),
                            trend_direction=profile_data.get("trend_direction", "neutral"),
                            trend_strength=profile_data.get("trend_strength", 0),
                        )
                logger.debug(f"Loaded {len(self._profiles)} volatility profiles")
            except Exception as e:
                logger.debug(f"Could not load profiles: {e}")

    def _save_profiles(self):
        """Save volatility profiles to cache."""
        try:
            cache_path = self._storage_dir / self.CACHE_FILE
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for symbol, profile in self._profiles.items():
                data[symbol] = {
                    "current_atr": profile.current_atr,
                    "avg_atr_20d": profile.avg_atr_20d,
                    "atr_percentile": profile.atr_percentile,
                    "daily_volatility": profile.daily_volatility,
                    "weekly_volatility": profile.weekly_volatility,
                    "beta": profile.beta,
                    "volatility_class": profile.volatility_class,
                    "trend_direction": profile.trend_direction,
                    "trend_strength": profile.trend_strength,
                    "last_updated": profile.last_updated.isoformat(),
                }

            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.debug(f"Could not save profiles: {e}")


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_manager_instance: Optional[ATRDynamicStopLossManager] = None
_manager_lock = RLock()


def get_dynamic_stop_loss_manager() -> ATRDynamicStopLossManager:
    """Get singleton instance of the stop loss manager."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = ATRDynamicStopLossManager()
        return _manager_instance


def reset_stop_loss_manager():
    """Reset the singleton instance (for testing)."""
    global _manager_instance
    with _manager_lock:
        _manager_instance = None


def calculate_atr_stop_loss(
    symbol: str, entry_price: float, df: pd.DataFrame, **kwargs
) -> StopLossResult:
    """Convenience function to calculate stop loss."""
    return get_dynamic_stop_loss_manager().calculate_stop_loss(symbol, entry_price, df, **kwargs)


def update_trailing_stop(
    symbol: str,
    current_price: float,
    highest_price: Optional[float] = None,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[float, bool, str]:
    """Convenience function to update trailing stop."""
    return get_dynamic_stop_loss_manager().update_trailing_stop(
        symbol, current_price, highest_price, df
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING ATR DYNAMIC STOP LOSS MANAGER")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n = 100

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n)
    trend = np.linspace(90000, 100000, n)
    noise = np.random.normal(0, 1000, n)
    close = trend + noise

    df = pd.DataFrame(
        {
            "date": dates,
            "open": close + np.random.normal(0, 500, n),
            "high": close + np.abs(np.random.normal(500, 200, n)),
            "low": close - np.abs(np.random.normal(500, 200, n)),
            "close": close,
            "volume": np.random.uniform(100000, 500000, n),
        }
    )

    manager = get_dynamic_stop_loss_manager()

    # Test stop loss calculation
    entry_price = 100000
    result = manager.calculate_stop_loss("VNM", entry_price, df)

    print(f"\n📊 Stop Loss Calculation for VNM @ {entry_price:,}")
    print("-" * 60)
    print(f"  Stop Price: {result.stop_loss_price:,.0f}")
    print(f"  Stop %: {result.stop_loss_pct*100:.1f}%")
    print(f"  Type: {result.stop_loss_type.value}")
    print(f"  ATR: {result.atr_value:,.0f}")
    print(f"  Adjustments: {result.adjustments}")

    # Test volatility profile
    profile = manager.get_volatility_profile("VNM", df)

    print(f"\n📈 Volatility Profile:")
    print("-" * 60)
    print(f"  ATR: {profile.current_atr:,.0f}")
    print(f"  ATR Percentile: {profile.atr_percentile:.0f}%")
    print(f"  Volatility Class: {profile.volatility_class}")
    print(f"  Daily Vol: {profile.daily_volatility:.2f}%")
    print(f"  Trend: {profile.trend_direction} ({profile.trend_strength:.1f}%)")

    # Test trailing stop
    print(f"\n🔄 Trailing Stop Test:")
    print("-" * 60)

    manager.init_trailing_stop("VNM", entry_price, result.stop_loss_price, df)

    # Simulate price movement
    test_prices = [101000, 103000, 105000, 103000, 102000, 100000]
    highest = entry_price

    for price in test_prices:
        if price > highest:
            highest = price
        new_stop, triggered, msg = manager.update_trailing_stop("VNM", price, highest, df)
        print(f"  Price: {price:,} | Highest: {highest:,} | Stop: {new_stop:,.0f} | {msg}")
        if triggered:
            break

    print("\n" + "=" * 60)
