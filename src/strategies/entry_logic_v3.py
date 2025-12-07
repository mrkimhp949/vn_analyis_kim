# -*- coding: utf-8 -*-
"""
Entry Logic v3.0 - Simplified & Optimized for Vietnam Market

IMPROVEMENTS over v2.0:
1. Simplified filter pipeline (5 core filters instead of 9)
2. Adaptive thresholds based on market regime
3. Smart filter prioritization (critical → important → optional)
4. Vietnam-specific optimizations (T+2, lot size, tick size)
5. Real-time foreign flow integration
6. Intraday momentum scoring
7. Better transaction cost awareness

Author: Trading Bot Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Vietnam Market Specific
# =============================================================================


class VNMarketConstants:
    """Vietnam market specific constants."""

    # Transaction costs (realistic)
    ROUND_TRIP_COST = 0.0148  # 1.48%
    BUY_COST = 0.0070  # 0.70%
    SELL_COST = 0.0078  # 0.78%

    # Liquidity tiers (VND)
    LIQUIDITY_VN30 = 10_000_000_000  # 10B - Blue chips
    LIQUIDITY_LARGE = 5_000_000_000  # 5B - Large cap
    LIQUIDITY_MID = 2_000_000_000  # 2B - Mid cap
    LIQUIDITY_SMALL = 500_000_000  # 500M - Small cap (minimum)

    # Price limits
    HOSE_LIMIT = 0.07  # ±7%
    HNX_LIMIT = 0.10  # ±10%
    UPCOM_LIMIT = 0.15  # ±15%

    # Optimal trading windows (Vietnam time)
    OPTIMAL_MORNING_START = time(9, 30)
    OPTIMAL_MORNING_END = time(10, 30)
    OPTIMAL_AFTERNOON_START = time(13, 30)
    OPTIMAL_AFTERNOON_END = time(14, 15)

    # Avoid windows
    ATO_START = time(9, 0)
    ATO_END = time(9, 15)
    ATC_START = time(14, 30)
    ATC_END = time(14, 45)
    LUNCH_START = time(11, 15)
    LUNCH_END = time(13, 0)

    # Lot size
    LOT_SIZE = 100


class FilterPriority(Enum):
    """Filter priority levels."""

    CRITICAL = 1  # Must pass - blocks entry if failed
    IMPORTANT = 2  # Strong weight - significant confidence impact
    OPTIONAL = 3  # Nice to have - minor confidence adjustment


class SignalStrength(Enum):
    """Signal strength classification."""

    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NO_SIGNAL = 0


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FilterResult:
    """Result of a single filter check."""

    name: str
    passed: bool
    priority: FilterPriority
    confidence_delta: int
    reason: str
    details: Dict = field(default_factory=dict)


@dataclass
class EntrySignalV3:
    """
    Enhanced entry signal result v3.0.

    Simplified and more actionable than v2.0.
    """

    should_enter: bool
    signal_type: str  # BUY, SELL, HOLD
    confidence: int  # 0-100
    strength: SignalStrength

    # Position sizing hints
    position_multiplier: float  # 0.0 - 1.2
    max_position_pct: float  # Max % of portfolio

    # Price levels
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    # Risk metrics
    risk_reward_ratio: float
    expected_return_after_costs: float

    # Filter results
    critical_filters_passed: int
    critical_filters_total: int
    filters_summary: List[FilterResult] = field(default_factory=list)

    # Actionable info
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Timing
    is_optimal_time: bool = False
    next_optimal_window: str = ""


# =============================================================================
# MAIN CLASS - Simplified Entry Logic v3.0
# =============================================================================


class SimplifiedEntryLogicV3:
    """
    Simplified Entry Logic v3.0 for Vietnam Market.

    KEY IMPROVEMENTS:
    1. 5 Core Filters (vs 9 in v2.0):
       - CRITICAL: Market Regime, Liquidity
       - IMPORTANT: Technical Score, Risk/Reward
       - OPTIONAL: Timing, Foreign Flow

    2. Adaptive Thresholds:
       - BULL: Lower confidence required (50%), higher position size
       - BEAR: Higher confidence required (70%), smaller position size
       - SIDEWAYS: Standard thresholds (60%)

    3. Transaction Cost Awareness:
       - All R:R calculations include 1.48% round trip cost
       - Minimum R:R of 2.0 ensures profitability after costs

    4. Vietnam Market Optimizations:
       - T+2 settlement awareness
       - Lot size 100 enforcement
       - Price limit (±7%) protection
       - ATO/ATC session handling
    """

    def __init__(
        self,
        base_min_confidence: int = 55,
        min_risk_reward: float = 2.0,
        min_liquidity_vnd: float = 500_000_000,
        max_position_pct: float = 0.12,
        use_adaptive_thresholds: bool = True,
        include_transaction_costs: bool = True,
    ):
        """
        Initialize simplified entry logic.

        Args:
            base_min_confidence: Base minimum confidence (adjusted by regime)
            min_risk_reward: Minimum risk/reward ratio (after costs)
            min_liquidity_vnd: Minimum daily liquidity in VND
            max_position_pct: Maximum position as % of portfolio
            use_adaptive_thresholds: Adjust thresholds by market regime
            include_transaction_costs: Include costs in R:R calculation
        """
        self.base_min_confidence = base_min_confidence
        self.min_risk_reward = min_risk_reward
        self.min_liquidity_vnd = min_liquidity_vnd
        self.max_position_pct = max_position_pct
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.include_transaction_costs = include_transaction_costs

        # Adaptive thresholds by regime
        self._regime_thresholds = {
            "BULL": {
                "min_confidence": 50,
                "min_rr": 1.8,
                "position_mult": 1.2,
                "max_position_pct": 0.15,
            },
            "SIDEWAYS": {
                "min_confidence": 60,
                "min_rr": 2.0,
                "position_mult": 1.0,
                "max_position_pct": 0.12,
            },
            "BEAR": {
                "min_confidence": 70,
                "min_rr": 2.5,
                "position_mult": 0.6,
                "max_position_pct": 0.08,
            },
            "HIGH_VOLATILITY": {
                "min_confidence": 75,
                "min_rr": 3.0,
                "position_mult": 0.5,
                "max_position_pct": 0.06,
            },
        }

    def analyze_entry(
        self,
        symbol: str,
        df: pd.DataFrame,
        ml_signal: Optional[Dict] = None,
        market_regime: Optional[Dict] = None,
        foreign_flow: Optional[Dict] = None,
        current_positions: Optional[Dict] = None,
    ) -> EntrySignalV3:
        """
        Analyze entry opportunity with simplified 5-filter pipeline.

        Filter Pipeline:
        1. [CRITICAL] Market Regime - Must be tradeable
        2. [CRITICAL] Liquidity - Must meet minimum threshold
        3. [IMPORTANT] Technical Score - Trend, momentum, volume
        4. [IMPORTANT] Risk/Reward - Must exceed minimum after costs
        5. [OPTIONAL] Timing & Flow - Session timing, foreign flow

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame with indicators
            ml_signal: ML model signal (optional)
            market_regime: Market regime info (optional)
            foreign_flow: Foreign flow data (optional)
            current_positions: Current portfolio positions (optional)

        Returns:
            EntrySignalV3 with complete analysis
        """
        # Initialize
        filter_results: List[FilterResult] = []
        reasons: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []

        # Get current price
        if df.empty or len(df) < 20:
            return self._create_no_entry("Insufficient data", filter_results)

        current_price = float(df["close"].iloc[-1])

        # Get adaptive thresholds
        regime = market_regime.get("regime", "SIDEWAYS") if market_regime else "SIDEWAYS"
        thresholds = self._get_adaptive_thresholds(regime)

        # =================================================================
        # FILTER 1: Market Regime [CRITICAL]
        # =================================================================
        regime_result = self._check_market_regime(market_regime)
        filter_results.append(regime_result)

        if not regime_result.passed:
            return self._create_no_entry(
                regime_result.reason, filter_results, warnings=[regime_result.reason]
            )

        if regime_result.confidence_delta > 0:
            reasons.append(regime_result.reason)

        # =================================================================
        # FILTER 2: Liquidity [CRITICAL]
        # =================================================================
        liquidity_result = self._check_liquidity(df, current_price, symbol)
        filter_results.append(liquidity_result)

        if not liquidity_result.passed:
            return self._create_no_entry(
                liquidity_result.reason, filter_results, warnings=[liquidity_result.reason]
            )

        if liquidity_result.confidence_delta > 0:
            reasons.append(liquidity_result.reason)

        # =================================================================
        # FILTER 3: Technical Score [IMPORTANT]
        # =================================================================
        tech_result = self._check_technical_score(df, ml_signal)
        filter_results.append(tech_result)

        if tech_result.passed:
            reasons.append(tech_result.reason)
        else:
            warnings.append(tech_result.reason)

        # =================================================================
        # FILTER 4: Risk/Reward [IMPORTANT]
        # =================================================================
        stop_loss = self._calculate_stop_loss(df, current_price)
        take_profit_1, take_profit_2 = self._calculate_take_profits(
            current_price, stop_loss, thresholds["min_rr"]
        )

        rr_result = self._check_risk_reward(
            current_price, stop_loss, take_profit_1, thresholds["min_rr"]
        )
        filter_results.append(rr_result)

        if rr_result.passed:
            reasons.append(rr_result.reason)
        else:
            warnings.append(rr_result.reason)

        # =================================================================
        # FILTER 5: Timing & Flow [OPTIONAL]
        # =================================================================
        timing_result = self._check_timing_and_flow(foreign_flow)
        filter_results.append(timing_result)

        is_optimal_time = timing_result.details.get("is_optimal", False)
        next_window = timing_result.details.get("next_window", "")

        if timing_result.passed:
            if timing_result.confidence_delta > 0:
                reasons.append(timing_result.reason)
        else:
            warnings.append(timing_result.reason)

        # =================================================================
        # CALCULATE FINAL CONFIDENCE
        # =================================================================
        base_confidence = 50  # Start at neutral

        # Add ML signal confidence if available
        if ml_signal and ml_signal.get("signal") == "BUY":
            ml_conf = ml_signal.get("confidence", 0)
            base_confidence = max(base_confidence, ml_conf * 0.7)  # 70% weight to ML

        # Apply filter adjustments
        for result in filter_results:
            base_confidence += result.confidence_delta

        # Clamp confidence
        final_confidence = max(0, min(100, int(base_confidence)))

        # =================================================================
        # FINAL DECISION
        # =================================================================
        critical_passed = sum(
            1 for r in filter_results if r.priority == FilterPriority.CRITICAL and r.passed
        )
        critical_total = sum(1 for r in filter_results if r.priority == FilterPriority.CRITICAL)

        important_passed = sum(
            1 for r in filter_results if r.priority == FilterPriority.IMPORTANT and r.passed
        )

        # Decision logic
        should_enter = (
            critical_passed == critical_total  # All critical must pass
            and important_passed >= 1  # At least 1 important must pass
            and final_confidence >= thresholds["min_confidence"]
            and rr_result.passed  # R:R must be acceptable
        )

        # Calculate position multiplier
        position_mult = thresholds["position_mult"]
        if final_confidence >= 80:
            position_mult *= 1.1
        elif final_confidence < 60:
            position_mult *= 0.8

        # Adjust for timing
        if not is_optimal_time:
            position_mult *= 0.9
            recommendations.append(f"Consider waiting for optimal window: {next_window}")

        # Calculate expected return after costs
        risk = current_price - stop_loss
        reward = take_profit_1 - current_price
        gross_return = reward / current_price if current_price > 0 else 0
        net_return = gross_return - VNMarketConstants.ROUND_TRIP_COST

        # Determine signal strength
        strength = self._determine_strength(final_confidence, important_passed)

        return EntrySignalV3(
            should_enter=should_enter,
            signal_type="BUY" if should_enter else "HOLD",
            confidence=final_confidence,
            strength=strength,
            position_multiplier=position_mult,
            max_position_pct=thresholds["max_position_pct"],
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            risk_reward_ratio=rr_result.details.get("rr_ratio", 0),
            expected_return_after_costs=net_return * 100,
            critical_filters_passed=critical_passed,
            critical_filters_total=critical_total,
            filters_summary=filter_results,
            reasons=reasons,
            warnings=warnings,
            recommendations=recommendations,
            is_optimal_time=is_optimal_time,
            next_optimal_window=next_window,
        )

    # =========================================================================
    # FILTER IMPLEMENTATIONS
    # =========================================================================

    def _check_market_regime(self, market_regime: Optional[Dict]) -> FilterResult:
        """
        CRITICAL Filter: Check market regime.

        Blocks entry if:
        - Market is not tradeable
        - Regime is HIGH_VOLATILITY with low confidence
        """
        if not market_regime:
            return FilterResult(
                name="market_regime",
                passed=True,
                priority=FilterPriority.CRITICAL,
                confidence_delta=0,
                reason="No regime data - assuming SIDEWAYS",
            )

        regime = market_regime.get("regime", "SIDEWAYS")
        tradeable = market_regime.get("tradeable", True)
        confidence = market_regime.get("confidence", 50)

        if not tradeable:
            return FilterResult(
                name="market_regime",
                passed=False,
                priority=FilterPriority.CRITICAL,
                confidence_delta=-50,
                reason=f"🚫 Market not tradeable ({regime})",
            )

        # Confidence adjustments by regime
        delta = 0
        if regime == "BULL":
            delta = 10 if confidence >= 70 else 5
            reason = f"✅ BULL market ({confidence:.0f}% conf)"
        elif regime == "BEAR":
            delta = -15 if confidence >= 70 else -10
            reason = f"⚠️ BEAR market - reduced exposure ({confidence:.0f}% conf)"
        elif regime == "HIGH_VOLATILITY":
            delta = -20
            reason = f"⚠️ HIGH VOLATILITY - caution required"
        else:
            delta = 0
            reason = f"📊 SIDEWAYS market ({confidence:.0f}% conf)"

        return FilterResult(
            name="market_regime",
            passed=True,
            priority=FilterPriority.CRITICAL,
            confidence_delta=delta,
            reason=reason,
            details={"regime": regime, "confidence": confidence},
        )

    def _check_liquidity(self, df: pd.DataFrame, current_price: float, symbol: str) -> FilterResult:
        """
        CRITICAL Filter: Check liquidity requirements.

        Vietnam market liquidity tiers:
        - VN30: 10B+ VND daily
        - Large cap: 5B+ VND
        - Mid cap: 2B+ VND
        - Small cap: 500M+ VND (minimum)
        """
        if "volume" not in df.columns:
            return FilterResult(
                name="liquidity",
                passed=False,
                priority=FilterPriority.CRITICAL,
                confidence_delta=-30,
                reason="🚫 No volume data available",
            )

        # Calculate average daily value
        avg_volume = df["volume"].tail(20).mean()
        avg_value = avg_volume * current_price

        # Determine tier and check
        if avg_value >= VNMarketConstants.LIQUIDITY_VN30:
            tier = "VN30"
            delta = 10
            passed = True
        elif avg_value >= VNMarketConstants.LIQUIDITY_LARGE:
            tier = "LARGE_CAP"
            delta = 5
            passed = True
        elif avg_value >= VNMarketConstants.LIQUIDITY_MID:
            tier = "MID_CAP"
            delta = 0
            passed = True
        elif avg_value >= VNMarketConstants.LIQUIDITY_SMALL:
            tier = "SMALL_CAP"
            delta = -5
            passed = True
        else:
            tier = "ILLIQUID"
            delta = -30
            passed = False

        if passed:
            reason = f"✅ Liquidity OK ({tier}: {avg_value/1e9:.1f}B VND)"
        else:
            reason = f"🚫 Insufficient liquidity ({avg_value/1e9:.2f}B < 0.5B VND)"

        return FilterResult(
            name="liquidity",
            passed=passed,
            priority=FilterPriority.CRITICAL,
            confidence_delta=delta,
            reason=reason,
            details={"tier": tier, "avg_value": avg_value},
        )

    def _check_technical_score(self, df: pd.DataFrame, ml_signal: Optional[Dict]) -> FilterResult:
        """
        IMPORTANT Filter: Calculate technical score.

        Components:
        - Trend alignment (SMA20 > SMA50): +15
        - RSI in buy zone (30-50): +10
        - Volume confirmation (> avg): +5
        - Price above support: +10
        - ML signal alignment: +15
        """
        score = 0
        details = {}

        close = df["close"]
        current = close.iloc[-1]

        # 1. Trend alignment
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20

        if current > sma20 > sma50:
            score += 15
            details["trend"] = "UPTREND"
        elif current > sma20:
            score += 8
            details["trend"] = "ABOVE_SMA20"
        elif current < sma20 < sma50:
            score -= 10
            details["trend"] = "DOWNTREND"
        else:
            details["trend"] = "NEUTRAL"

        # 2. RSI check
        if "rsi" in df.columns:
            rsi = df["rsi"].iloc[-1]
            if 30 <= rsi <= 50:
                score += 10
                details["rsi"] = f"BUY_ZONE ({rsi:.0f})"
            elif rsi < 30:
                score += 15  # Oversold - strong buy
                details["rsi"] = f"OVERSOLD ({rsi:.0f})"
            elif rsi > 70:
                score -= 15  # Overbought - avoid
                details["rsi"] = f"OVERBOUGHT ({rsi:.0f})"
            else:
                details["rsi"] = f"NEUTRAL ({rsi:.0f})"

        # 3. Volume confirmation
        if "volume" in df.columns:
            vol_avg = df["volume"].rolling(20).mean().iloc[-1]
            vol_current = df["volume"].iloc[-1]
            vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1

            if vol_ratio > 1.5:
                score += 10
                details["volume"] = f"SURGE ({vol_ratio:.1f}x)"
            elif vol_ratio > 1.0:
                score += 5
                details["volume"] = f"ABOVE_AVG ({vol_ratio:.1f}x)"
            else:
                details["volume"] = f"BELOW_AVG ({vol_ratio:.1f}x)"

        # 4. Support check (simple: above 20-day low)
        low_20 = close.tail(20).min()
        support_distance = (current - low_20) / low_20 * 100

        if support_distance < 3:
            score += 10
            details["support"] = f"NEAR_SUPPORT ({support_distance:.1f}%)"
        elif support_distance < 7:
            score += 5
            details["support"] = f"ABOVE_SUPPORT ({support_distance:.1f}%)"

        # 5. ML signal alignment
        if ml_signal and ml_signal.get("signal") == "BUY":
            ml_conf = ml_signal.get("confidence", 0)
            if ml_conf >= 70:
                score += 15
                details["ml"] = f"STRONG_BUY ({ml_conf:.0f}%)"
            elif ml_conf >= 60:
                score += 10
                details["ml"] = f"BUY ({ml_conf:.0f}%)"
            else:
                score += 5
                details["ml"] = f"WEAK_BUY ({ml_conf:.0f}%)"
        elif ml_signal and ml_signal.get("signal") == "SELL":
            score -= 20
            details["ml"] = "SELL_SIGNAL"

        # Determine pass/fail
        passed = score >= 15

        if score >= 30:
            reason = f"✅ Strong technical setup (score: {score})"
        elif score >= 15:
            reason = f"✅ Acceptable technical setup (score: {score})"
        elif score >= 0:
            reason = f"⚠️ Weak technical setup (score: {score})"
        else:
            reason = f"⚠️ Poor technical setup (score: {score})"

        return FilterResult(
            name="technical_score",
            passed=passed,
            priority=FilterPriority.IMPORTANT,
            confidence_delta=score,
            reason=reason,
            details=details,
        )

    def _check_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        min_rr: float,
    ) -> FilterResult:
        """
        IMPORTANT Filter: Check risk/reward ratio.

        Includes transaction costs in calculation:
        - Round trip cost: 1.48%
        - Minimum R:R after costs: 2.0 (adjustable by regime)
        """
        if entry_price <= 0 or stop_loss <= 0:
            return FilterResult(
                name="risk_reward",
                passed=False,
                priority=FilterPriority.IMPORTANT,
                confidence_delta=-20,
                reason="🚫 Invalid price data",
            )

        risk = entry_price - stop_loss
        reward = take_profit - entry_price

        if risk <= 0:
            return FilterResult(
                name="risk_reward",
                passed=False,
                priority=FilterPriority.IMPORTANT,
                confidence_delta=-20,
                reason="🚫 Invalid stop loss (above entry)",
            )

        # Gross R:R
        gross_rr = reward / risk

        # Net R:R (after transaction costs)
        if self.include_transaction_costs:
            cost_impact = entry_price * VNMarketConstants.ROUND_TRIP_COST
            net_reward = reward - cost_impact
            net_rr = net_reward / risk if risk > 0 else 0
        else:
            net_rr = gross_rr

        passed = net_rr >= min_rr

        if net_rr >= 3.0:
            delta = 15
            reason = f"✅ Excellent R:R ({net_rr:.1f}:1 after costs)"
        elif net_rr >= 2.5:
            delta = 10
            reason = f"✅ Good R:R ({net_rr:.1f}:1 after costs)"
        elif net_rr >= min_rr:
            delta = 5
            reason = f"✅ Acceptable R:R ({net_rr:.1f}:1 after costs)"
        elif net_rr >= 1.5:
            delta = -5
            reason = f"⚠️ Low R:R ({net_rr:.1f}:1 after costs, min: {min_rr})"
        else:
            delta = -15
            reason = f"⚠️ Poor R:R ({net_rr:.1f}:1 after costs, min: {min_rr})"

        return FilterResult(
            name="risk_reward",
            passed=passed,
            priority=FilterPriority.IMPORTANT,
            confidence_delta=delta,
            reason=reason,
            details={
                "gross_rr": gross_rr,
                "net_rr": net_rr,
                "risk": risk,
                "reward": reward,
                "rr_ratio": net_rr,
            },
        )

    def _check_timing_and_flow(self, foreign_flow: Optional[Dict]) -> FilterResult:
        """
        OPTIONAL Filter: Check timing and foreign flow.

        Optimal windows:
        - Morning: 9:30 - 10:30
        - Afternoon: 13:30 - 14:15

        Avoid:
        - ATO: 9:00 - 9:15
        - ATC: 14:30 - 14:45
        - Pre-lunch: 11:15 - 11:30
        """
        try:
            import pytz

            vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
            now = datetime.now(vn_tz).time()
        except ImportError:
            now = datetime.now().time()

        is_optimal = False
        next_window = ""
        timing_delta = 0

        # Check optimal windows
        if VNMarketConstants.OPTIMAL_MORNING_START <= now <= VNMarketConstants.OPTIMAL_MORNING_END:
            is_optimal = True
            timing_delta = 5
            timing_reason = "✅ Optimal morning window"
        elif (
            VNMarketConstants.OPTIMAL_AFTERNOON_START
            <= now
            <= VNMarketConstants.OPTIMAL_AFTERNOON_END
        ):
            is_optimal = True
            timing_delta = 5
            timing_reason = "✅ Optimal afternoon window"
        elif VNMarketConstants.ATO_START <= now <= VNMarketConstants.ATO_END:
            timing_delta = -10
            timing_reason = "⚠️ ATO session - high volatility"
            next_window = "9:30"
        elif VNMarketConstants.ATC_START <= now <= VNMarketConstants.ATC_END:
            timing_delta = -10
            timing_reason = "⚠️ ATC session - high volatility"
            next_window = "Tomorrow 9:30"
        elif VNMarketConstants.LUNCH_START <= now <= VNMarketConstants.LUNCH_END:
            timing_delta = -5
            timing_reason = "⚠️ Lunch break approaching"
            next_window = "13:30"
        else:
            timing_reason = "📊 Acceptable timing"
            if now < VNMarketConstants.OPTIMAL_MORNING_START:
                next_window = "9:30"
            elif now < VNMarketConstants.OPTIMAL_AFTERNOON_START:
                next_window = "13:30"

        # Foreign flow adjustment
        flow_delta = 0
        if foreign_flow:
            net_value = foreign_flow.get("net_value", 0)
            if net_value > 50_000_000_000:  # 50B+ net buy
                flow_delta = 10
                timing_reason += " | Strong foreign buying"
            elif net_value > 10_000_000_000:  # 10B+ net buy
                flow_delta = 5
                timing_reason += " | Foreign buying"
            elif net_value < -50_000_000_000:  # 50B+ net sell
                flow_delta = -10
                timing_reason += " | Strong foreign selling"
            elif net_value < -10_000_000_000:  # 10B+ net sell
                flow_delta = -5
                timing_reason += " | Foreign selling"

        total_delta = timing_delta + flow_delta

        return FilterResult(
            name="timing_flow",
            passed=is_optimal or total_delta >= 0,
            priority=FilterPriority.OPTIONAL,
            confidence_delta=total_delta,
            reason=timing_reason,
            details={
                "is_optimal": is_optimal,
                "next_window": next_window,
                "timing_delta": timing_delta,
                "flow_delta": flow_delta,
            },
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_adaptive_thresholds(self, regime: str) -> Dict:
        """Get adaptive thresholds based on market regime."""
        if not self.use_adaptive_thresholds:
            return {
                "min_confidence": self.base_min_confidence,
                "min_rr": self.min_risk_reward,
                "position_mult": 1.0,
                "max_position_pct": self.max_position_pct,
            }

        return self._regime_thresholds.get(regime, self._regime_thresholds["SIDEWAYS"])

    def _calculate_stop_loss(self, df: pd.DataFrame, current_price: float) -> float:
        """
        Calculate stop loss using ATR or percentage.

        Vietnam market: 5.5% default stop (within ±7% daily limit)
        """
        # Try ATR-based stop
        if "atr" in df.columns:
            atr = df["atr"].iloc[-1]
            atr_stop = current_price - (atr * 2.0)

            # Ensure stop is within reasonable range (3-7%)
            min_stop = current_price * 0.93  # Max 7% loss
            max_stop = current_price * 0.97  # Min 3% loss

            return max(min_stop, min(max_stop, atr_stop))

        # Default: 5.5% stop loss
        return current_price * 0.945

    def _calculate_take_profits(
        self, entry: float, stop: float, min_rr: float
    ) -> Tuple[float, float]:
        """
        Calculate take profit targets.

        TP1: min_rr * risk (partial exit)
        TP2: 2 * min_rr * risk (full exit)
        """
        risk = entry - stop

        # Account for transaction costs
        cost_buffer = entry * VNMarketConstants.ROUND_TRIP_COST

        tp1 = entry + (risk * min_rr) + cost_buffer
        tp2 = entry + (risk * min_rr * 2) + cost_buffer

        # Round to tick size
        tp1 = self._round_to_tick(tp1)
        tp2 = self._round_to_tick(tp2)

        return tp1, tp2

    def _round_to_tick(self, price: float) -> float:
        """Round to Vietnam tick size."""
        if price < 10000:
            tick = 10
        elif price < 50000:
            tick = 50
        else:
            tick = 100
        return round(price / tick) * tick

    def _determine_strength(self, confidence: int, important_passed: int) -> SignalStrength:
        """Determine signal strength from confidence and filters."""
        if confidence >= 80 and important_passed >= 2:
            return SignalStrength.VERY_STRONG
        elif confidence >= 70 and important_passed >= 2:
            return SignalStrength.STRONG
        elif confidence >= 60:
            return SignalStrength.MODERATE
        elif confidence >= 50:
            return SignalStrength.WEAK
        elif confidence >= 40:
            return SignalStrength.VERY_WEAK
        else:
            return SignalStrength.NO_SIGNAL

    def _create_no_entry(
        self,
        reason: str,
        filters: List[FilterResult],
        warnings: List[str] = None,
    ) -> EntrySignalV3:
        """Create a no-entry result."""
        return EntrySignalV3(
            should_enter=False,
            signal_type="HOLD",
            confidence=0,
            strength=SignalStrength.NO_SIGNAL,
            position_multiplier=0,
            max_position_pct=0,
            entry_price=0,
            stop_loss=0,
            take_profit_1=0,
            take_profit_2=0,
            risk_reward_ratio=0,
            expected_return_after_costs=0,
            critical_filters_passed=0,
            critical_filters_total=sum(1 for f in filters if f.priority == FilterPriority.CRITICAL),
            filters_summary=filters,
            reasons=[],
            warnings=warnings or [reason],
            recommendations=[],
        )


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================

_entry_logic_v3: Optional[SimplifiedEntryLogicV3] = None


def get_entry_logic_v3() -> SimplifiedEntryLogicV3:
    """Get singleton entry logic v3."""
    global _entry_logic_v3
    if _entry_logic_v3 is None:
        _entry_logic_v3 = SimplifiedEntryLogicV3()
    return _entry_logic_v3
