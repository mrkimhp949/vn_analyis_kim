# -*- coding: utf-8 -*-
"""
Entry Filters Module - Filter implementations for entry signal analysis.

This module contains all filter logic used by ImprovedEntryLogic:
- Price limit filter (Vietnam ±7%)
- Trend alignment filter
- Support/Resistance filter
- Volume confirmation filter
- Liquidity filter
- Volatility filter
- RSI filter
- Correlation filter
- Optional filters (sector, breadth, price action)

Extracted from entry_logic.py for better modularity and testability.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config.constants import (
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    VN_CEILING_DISTANCE_THRESHOLD,
    VN_FLOOR_DISTANCE_THRESHOLD,
    VN_FLOOR_PENALTY,
    VIETNAM_PRICE_LIMIT_PERCENT,
)
from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

logger = logging.getLogger(__name__)


# =============================================================================
# FILTER RESULT DATA CLASSES
# =============================================================================


@dataclass
class FilterResult:
    """Base result from any filter check."""

    passed: bool
    adjustment: int = 0
    reason: str = ""
    is_blocking: bool = False  # If True, blocks entry entirely
    details: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "adjustment": self.adjustment,
            "reason": self.reason,
            "is_blocking": self.is_blocking,
            "details": self.details,
        }


@dataclass
class PriceLimitResult(FilterResult):
    """Result from Vietnam price limit check."""

    near_limit: bool = False
    limit_type: Optional[str] = None  # 'CEILING' or 'FLOOR'
    distance_to_ceiling: float = 0.0
    distance_to_floor: float = 0.0
    ceiling_price: float = 0.0
    floor_price: float = 0.0


@dataclass
class TrendResult(FilterResult):
    """Result from trend alignment check."""

    aligned: bool = False
    strength: int = 0  # 0-100
    ema_alignment: str = ""  # 'perfect', 'good', 'ok', 'none'


@dataclass
class SupportResistanceResult(FilterResult):
    """Result from support/resistance check."""

    near_support: bool = False
    bouncing_from_support: bool = False
    too_close_to_resistance: bool = False
    support_level: float = 0.0
    resistance_level: float = 0.0
    distance_to_support: float = 0.0
    distance_to_resistance: float = 0.0


@dataclass
class VolumeResult(FilterResult):
    """Result from volume confirmation check."""

    confirmed: bool = False
    surge: bool = False
    volume_ratio: float = 1.0
    is_manipulation: bool = False
    obv_bullish: bool = False


@dataclass
class LiquidityResult(FilterResult):
    """Result from liquidity check."""

    sufficient: bool = False
    tier: str = "unknown"  # 'large', 'mid', 'small', 'penny'
    avg_value: float = 0.0
    critical: bool = False


@dataclass
class VolatilityResult(FilterResult):
    """Result from volatility check."""

    too_high: bool = False
    optimal: bool = False
    value: float = 0.0  # ATR/Price percentage


@dataclass
class RSIResult(FilterResult):
    """Result from RSI check."""

    overbought: bool = False
    oversold: bool = False
    optimal: bool = False
    value: float = 50.0


# =============================================================================
# BASE FILTER CLASS
# =============================================================================


class BaseEntryFilter(ABC):
    """Abstract base class for entry filters."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def check(self, df: pd.DataFrame, **kwargs) -> FilterResult:
        """
        Run the filter check.

        Args:
            df: DataFrame with OHLCV data
            **kwargs: Additional parameters specific to the filter

        Returns:
            FilterResult with check outcome
        """
        pass

    def _log_result(self, result: FilterResult) -> None:
        """Log filter result for debugging."""
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        if result.is_blocking:
            status = "🚫 BLOCKED"
        self.logger.debug(f"{self.name}: {status} - {result.reason}")


# =============================================================================
# FILTER IMPLEMENTATIONS
# =============================================================================


class VietnamPriceLimitFilter(BaseEntryFilter):
    """
    Check if price is near Vietnam market floor/ceiling limits (±7%).

    Vietnam stock market has daily price limits:
    - Ceiling (trần): +7% from reference price
    - Floor (sàn): -7% from reference price

    Trading near these limits is risky:
    - Near ceiling: Block entry (may not be able to buy, may get trapped)
    - Near floor: Warn but allow (potential reversal opportunity)
    """

    def __init__(self):
        super().__init__("vietnam_price_limits")

    def check(self, df: pd.DataFrame, current_price: float = None, **kwargs) -> PriceLimitResult:
        """
        Check Vietnam price limits.

        Args:
            df: DataFrame with OHLCV data
            current_price: Current stock price

        Returns:
            PriceLimitResult with limit check details
        """
        if current_price is None:
            current_price = safe_get_latest(df, "close", 0)

        ceiling_mult = 1 + VIETNAM_PRICE_LIMIT_PERCENT
        floor_mult = 1 - VIETNAM_PRICE_LIMIT_PERCENT

        if len(df) < 2:
            return PriceLimitResult(
                passed=True,
                reason="Insufficient data for price limit check",
                near_limit=False,
            )

        reference_price = df["close"].iloc[-2]
        ceiling_price = reference_price * ceiling_mult
        floor_price = reference_price * floor_mult

        distance_to_ceiling = ((ceiling_price - current_price) / reference_price) * 100
        distance_to_floor = ((current_price - floor_price) / reference_price) * 100

        near_ceiling = distance_to_ceiling <= VN_CEILING_DISTANCE_THRESHOLD
        near_floor = distance_to_floor <= VN_FLOOR_DISTANCE_THRESHOLD

        # Determine result
        if near_ceiling:
            result = PriceLimitResult(
                passed=False,
                is_blocking=True,
                adjustment=0,
                reason=f"Near ceiling ({current_price:,.0f} / {ceiling_price:,.0f}), only {distance_to_ceiling:.2f}% away",
                near_limit=True,
                limit_type="CEILING",
                distance_to_ceiling=distance_to_ceiling,
                distance_to_floor=distance_to_floor,
                ceiling_price=ceiling_price,
                floor_price=floor_price,
            )
        elif near_floor:
            result = PriceLimitResult(
                passed=True,  # Allow but warn
                is_blocking=False,
                adjustment=VN_FLOOR_PENALTY,
                reason=f"Near floor ({current_price:,.0f} / {floor_price:,.0f}), potential reversal",
                near_limit=True,
                limit_type="FLOOR",
                distance_to_ceiling=distance_to_ceiling,
                distance_to_floor=distance_to_floor,
                ceiling_price=ceiling_price,
                floor_price=floor_price,
            )
        else:
            result = PriceLimitResult(
                passed=True,
                reason="Price within safe limits",
                near_limit=False,
                distance_to_ceiling=distance_to_ceiling,
                distance_to_floor=distance_to_floor,
                ceiling_price=ceiling_price,
                floor_price=floor_price,
            )

        self._log_result(result)
        return result


class TrendAlignmentFilter(BaseEntryFilter):
    """
    Check trend alignment using EMA crossovers.

    Perfect alignment: Price > EMA20 > EMA50 > EMA200
    Good alignment: Price > EMA20 > EMA50
    Ok alignment: Price > EMA20
    """

    def __init__(self):
        super().__init__("trend_alignment")

    def check(self, df: pd.DataFrame, signal_type: str = "BUY", **kwargs) -> TrendResult:
        """
        Check trend alignment.

        Args:
            df: DataFrame with OHLCV data
            signal_type: 'BUY' or 'SELL'

        Returns:
            TrendResult with alignment details
        """
        if len(df) < 20:
            return TrendResult(
                passed=True,
                reason="Insufficient data for trend check",
                aligned=True,
                strength=50,
            )

        # Get EMAs
        close = df["close"]
        ema20 = close.ewm(span=20, adjust=False).mean()

        latest_price = close.iloc[-1]
        latest_ema20 = ema20.iloc[-1]

        # Check data availability for longer EMAs
        use_ema50 = len(df) >= 50
        use_ema200 = len(df) >= 200

        if use_ema50:
            ema50 = close.ewm(span=50, adjust=False).mean()
            latest_ema50 = ema50.iloc[-1]
        else:
            latest_ema50 = latest_ema20

        if use_ema200:
            ema200 = close.ewm(span=200, adjust=False).mean()
            latest_ema200 = ema200.iloc[-1]
        else:
            latest_ema200 = latest_ema50

        # Check alignment for BUY signals
        if signal_type == "BUY":
            ok = latest_price > latest_ema20

            if use_ema200:
                perfect = latest_price > latest_ema20 > latest_ema50 > latest_ema200
                good = latest_price > latest_ema20 > latest_ema50
            elif use_ema50:
                perfect = False
                good = latest_price > latest_ema20 > latest_ema50
            else:
                perfect = False
                good = False

            if perfect:
                return TrendResult(
                    passed=True,
                    adjustment=+5,
                    reason="Perfect uptrend (EMA20>50>200)",
                    aligned=True,
                    strength=100,
                    ema_alignment="perfect",
                )
            elif good:
                return TrendResult(
                    passed=True,
                    adjustment=+3,
                    reason="Strong uptrend (EMA20>50)",
                    aligned=True,
                    strength=75,
                    ema_alignment="good",
                )
            elif ok:
                return TrendResult(
                    passed=True,
                    adjustment=0,
                    reason="Short-term uptrend",
                    aligned=True,
                    strength=50,
                    ema_alignment="ok",
                )
            else:
                return TrendResult(
                    passed=False,
                    adjustment=-10,
                    reason="Downtrend or sideways",
                    aligned=False,
                    strength=0,
                    ema_alignment="none",
                )

        # For SELL signals (inverse logic)
        return TrendResult(
            passed=True,
            reason="Sell signal alignment check",
            aligned=True,
            strength=50,
        )


class VolumeConfirmationFilter(BaseEntryFilter):
    """
    Check volume confirmation with multiple indicators.

    Checks:
    1. Volume ratio (current vs average)
    2. Volume trend (5-day vs 20-day MA)
    3. OBV (On-Balance Volume) trend
    4. Manipulation detection (abnormal spikes)
    """

    def __init__(self):
        super().__init__("volume_confirmation")

    def check(
        self, df: pd.DataFrame, market_regime: Optional[Dict] = None, **kwargs
    ) -> VolumeResult:
        """
        Check volume confirmation.

        Args:
            df: DataFrame with OHLCV data
            market_regime: Market regime information

        Returns:
            VolumeResult with volume analysis
        """
        if len(df) < 20:
            return VolumeResult(
                passed=True,
                reason="Insufficient data",
                confirmed=True,
            )

        current_volume = safe_get_latest(df, "volume", 0)
        avg_volume_20 = safe_rolling_operation(df, "volume", 20, "mean", 0)

        if avg_volume_20 == 0:
            return VolumeResult(
                passed=True,
                reason="No volume data",
                confirmed=True,
            )

        volume_ratio = current_volume / avg_volume_20

        # Check for manipulation first
        if volume_ratio > 8.0:
            return VolumeResult(
                passed=False,
                is_blocking=True,
                reason=f"Abnormal volume spike {volume_ratio:.1f}x - possible manipulation",
                confirmed=False,
                is_manipulation=True,
                volume_ratio=volume_ratio,
            )

        # Check for wash trading (volume spike with no price movement)
        if volume_ratio > 5.0 and len(df) >= 2:
            current_close = df["close"].iloc[-1]
            prev_close = df["close"].iloc[-2]
            price_change = abs((current_close - prev_close) / prev_close * 100)

            if price_change < 2.0:
                return VolumeResult(
                    passed=False,
                    is_blocking=True,
                    reason=f"Volume {volume_ratio:.1f}x but price only {price_change:.1f}% - wash trading",
                    confirmed=False,
                    is_manipulation=True,
                    volume_ratio=volume_ratio,
                )

        # Dynamic threshold based on market regime
        base_threshold = 0.5
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            if regime == "BULL":
                base_threshold = 0.4
            elif regime in ["BEAR", "HIGH_VOLATILITY"]:
                base_threshold = 0.6

        # Calculate confidence score
        confidence_score = 0.0

        # Volume ratio contributes 40%
        if volume_ratio >= 1.5:
            confidence_score += 0.4
        elif volume_ratio >= 1.2:
            confidence_score += 0.3
        elif volume_ratio >= 1.0:
            confidence_score += 0.2

        # Volume trend contributes 30%
        vol_5d = df["volume"].tail(5).mean()
        vol_10d = df["volume"].tail(10).mean()
        volume_trending_up = vol_5d > vol_10d
        if volume_trending_up:
            confidence_score += 0.3

        # OBV trend contributes 30%
        obv_bullish = self._check_obv_trend(df)
        if obv_bullish:
            confidence_score += 0.3

        confirmed = confidence_score >= base_threshold
        surge = volume_ratio >= 1.5

        if confirmed:
            adjustment = +5 if surge else 0
            reason = f"Volume {volume_ratio:.1f}x" + (" surge" if surge else "")
        else:
            adjustment = -10
            reason = f"Weak volume {volume_ratio:.1f}x"

        return VolumeResult(
            passed=confirmed,
            adjustment=adjustment,
            reason=reason,
            confirmed=confirmed,
            surge=surge,
            volume_ratio=volume_ratio,
            is_manipulation=False,
            obv_bullish=obv_bullish,
        )

    def _check_obv_trend(self, df: pd.DataFrame, periods: int = 5) -> bool:
        """Check if OBV is trending up."""
        if len(df) < periods + 1:
            return True

        try:
            # Calculate OBV
            obv = [0]
            for i in range(1, len(df)):
                if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                    obv.append(obv[-1] + df["volume"].iloc[i])
                elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                    obv.append(obv[-1] - df["volume"].iloc[i])
                else:
                    obv.append(obv[-1])

            # Check trend
            obv_recent = obv[-periods:]
            return obv_recent[-1] > obv_recent[0]
        except Exception:
            return True


class RSIFilter(BaseEntryFilter):
    """
    Check RSI levels for entry signals.

    Levels:
    - > 70: Overbought (warning)
    - 60-70: Neutral
    - 30-60: Optimal for entry
    - < 30: Oversold (strong buy signal, but needs confirmation in bear market)
    """

    def __init__(self):
        super().__init__("rsi")

    def check(self, df: pd.DataFrame, market_regime: Optional[Dict] = None, **kwargs) -> RSIResult:
        """
        Check RSI levels.

        Args:
            df: DataFrame with OHLCV and RSI
            market_regime: Market regime for context

        Returns:
            RSIResult with RSI analysis
        """
        if "rsi" not in df.columns:
            return RSIResult(
                passed=True,
                reason="No RSI data",
                optimal=True,
                value=50,
            )

        rsi = safe_get_latest(df, "rsi", 50)

        if pd.isna(rsi):
            return RSIResult(
                passed=True,
                reason="RSI data invalid",
                optimal=True,
                value=50,
            )

        regime = "SIDEWAYS"
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")

        # Overbought
        if rsi > RSI_OVERBOUGHT:
            return RSIResult(
                passed=False,
                adjustment=-10,
                reason=f"RSI overbought: {rsi:.1f}",
                overbought=True,
                value=rsi,
            )

        # Oversold - needs special handling in bear market
        if rsi < RSI_OVERSOLD:
            if regime == "BEAR":
                # In bear market, oversold can go more oversold
                return RSIResult(
                    passed=True,
                    adjustment=+3,
                    reason=f"RSI oversold in bear market: {rsi:.1f} (cautious)",
                    oversold=True,
                    value=rsi,
                )
            elif regime == "HIGH_VOLATILITY":
                return RSIResult(
                    passed=True,
                    adjustment=+5,
                    reason=f"RSI oversold in high vol: {rsi:.1f}",
                    oversold=True,
                    value=rsi,
                )
            else:
                return RSIResult(
                    passed=True,
                    adjustment=+15,
                    reason=f"RSI oversold: {rsi:.1f} (strong buy)",
                    oversold=True,
                    value=rsi,
                )

        # Optimal range
        if 30 <= rsi <= 60:
            return RSIResult(
                passed=True,
                adjustment=+5,
                reason=f"RSI optimal: {rsi:.1f}",
                optimal=True,
                value=rsi,
            )

        # Neutral (60-70)
        return RSIResult(
            passed=True,
            adjustment=0,
            reason=f"RSI neutral: {rsi:.1f}",
            value=rsi,
        )


class VolatilityFilter(BaseEntryFilter):
    """
    Check volatility using ATR/Price ratio.

    Levels:
    - < 2%: Too low (no momentum)
    - 2-3%: Optimal
    - > 4%: Too high (risky)
    """

    def __init__(self):
        super().__init__("volatility")

    def check(self, df: pd.DataFrame, **kwargs) -> VolatilityResult:
        """
        Check volatility levels.

        Args:
            df: DataFrame with OHLCV and ATR

        Returns:
            VolatilityResult with volatility analysis
        """
        atr = safe_get_latest(df, "atr", 0)
        price = safe_get_latest(df, "close", 0)

        if price == 0:
            return VolatilityResult(
                passed=True,
                reason="No price data",
                optimal=True,
                value=0,
            )

        volatility = (atr / price) * 100

        if volatility > 4:
            return VolatilityResult(
                passed=False,
                adjustment=-15,
                reason=f"High volatility: {volatility:.2f}%",
                too_high=True,
                value=volatility,
            )
        elif 2 <= volatility <= 3:
            return VolatilityResult(
                passed=True,
                adjustment=+5,
                reason=f"Optimal volatility: {volatility:.2f}%",
                optimal=True,
                value=volatility,
            )
        else:
            return VolatilityResult(
                passed=True,
                adjustment=0,
                reason=f"Low volatility: {volatility:.2f}%",
                value=volatility,
            )


# =============================================================================
# FILTER MANAGER
# =============================================================================


class EntryFilterManager:
    """
    Manages and orchestrates multiple entry filters.

    Usage:
        manager = EntryFilterManager()
        results = manager.run_all_filters(df, current_price, market_regime)

        if results.blocked:
            print(f"Entry blocked: {results.block_reason}")
        else:
            print(f"Total adjustment: {results.total_adjustment}")
    """

    def __init__(self):
        self.filters: Dict[str, BaseEntryFilter] = {
            "price_limit": VietnamPriceLimitFilter(),
            "trend": TrendAlignmentFilter(),
            "volume": VolumeConfirmationFilter(),
            "rsi": RSIFilter(),
            "volatility": VolatilityFilter(),
        }
        self.logger = logging.getLogger(__name__)

    def run_filter(self, filter_name: str, df: pd.DataFrame, **kwargs) -> Optional[FilterResult]:
        """Run a single filter by name."""
        if filter_name not in self.filters:
            self.logger.warning(f"Unknown filter: {filter_name}")
            return None

        return self.filters[filter_name].check(df, **kwargs)

    def run_blocking_filters(
        self,
        df: pd.DataFrame,
        current_price: float,
        market_regime: Optional[Dict] = None,
    ) -> Tuple[bool, Optional[str], List[FilterResult]]:
        """
        Run filters that can block entry.

        Returns:
            Tuple of (is_blocked, block_reason, filter_results)
        """
        results = []

        # Price limit filter
        price_result = self.filters["price_limit"].check(df, current_price=current_price)
        results.append(price_result)
        if price_result.is_blocking:
            return (True, price_result.reason, results)

        # Volume manipulation check
        volume_result = self.filters["volume"].check(df, market_regime=market_regime)
        results.append(volume_result)
        if volume_result.is_blocking:
            return (True, volume_result.reason, results)

        return (False, None, results)

    def calculate_total_adjustment(self, results: List[FilterResult]) -> int:
        """Calculate total confidence adjustment from filter results."""
        return sum(r.adjustment for r in results if r is not None)
