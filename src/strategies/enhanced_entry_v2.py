# -*- coding: utf-8 -*-
"""
Enhanced Entry Logic v2.0 - Full Integration

Integrates all new modules for 10/10 trading logic:
- Real-time data integration
- Fundamental analysis filters
- Earnings calendar awareness
- Portfolio VaR checks
- Alert notifications

Author: Trading Bot Team
Version: 2.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EnhancedEntryResult:
    """Enhanced entry analysis result"""

    should_enter: bool
    signal_type: str  # BUY, SELL, HOLD
    confidence: float  # 0-100

    # Position sizing
    position_multiplier: float  # 0.0 to 1.0
    recommended_shares: int
    recommended_value: float

    # Price levels
    entry_price: float
    stop_loss: float
    take_profit_targets: List[float]

    # Risk metrics
    risk_reward_ratio: float
    var_impact: float  # VaR impact of this trade
    portfolio_risk_after: float

    # Filters passed
    filters_passed: List[str] = field(default_factory=list)
    filters_failed: List[str] = field(default_factory=list)

    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Detailed breakdown
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    timing_score: float = 0.0
    risk_score: float = 0.0


class EnhancedEntryLogicV2:
    """
    Enhanced Entry Logic with full integration

    New Features (v2.0):
    1. Real-time data integration
    2. Fundamental analysis (P/E, P/B, ROE)
    3. Earnings calendar check
    4. Ex-dividend awareness
    5. Portfolio VaR integration
    6. Foreign flow real-time
    7. Alert notifications

    Filter Pipeline:
    1. Market Regime (must be tradeable)
    2. Session Timing (optimal windows)
    3. Fundamental Score (min 40/100)
    4. Earnings Risk (avoid 5 days before)
    5. Dividend Risk (avoid 3 days before ex-date)
    6. Technical Filters (trend, RSI, volume)
    7. Liquidity Check (min 2B VND)
    8. Portfolio Risk (VaR check)
    9. Position Correlation
    """

    def __init__(
        self,
        min_confidence: int = 50,
        min_fundamental_score: float = 40.0,
        max_portfolio_var_pct: float = 5.0,
        enable_realtime: bool = True,
        enable_fundamental: bool = True,
        enable_earnings_check: bool = True,
        enable_alerts: bool = True,
    ):
        self.min_confidence = min_confidence
        self.min_fundamental_score = min_fundamental_score
        self.max_portfolio_var_pct = max_portfolio_var_pct

        # Feature flags
        self.enable_realtime = enable_realtime
        self.enable_fundamental = enable_fundamental
        self.enable_earnings_check = enable_earnings_check
        self.enable_alerts = enable_alerts

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all components"""
        # Real-time data
        self._realtime_manager = None
        if self.enable_realtime:
            try:
                from src.data.realtime_provider import get_realtime_manager

                self._realtime_manager = get_realtime_manager()
            except ImportError:
                logger.warning("Real-time provider not available")

        # Fundamental analyzer
        self._fundamental_analyzer = None
        if self.enable_fundamental:
            try:
                from src.data.fundamental_analyzer import get_fundamental_analyzer

                self._fundamental_analyzer = get_fundamental_analyzer()
            except ImportError:
                logger.warning("Fundamental analyzer not available")

        # Earnings calendar
        self._earnings_manager = None
        if self.enable_earnings_check:
            try:
                from src.data.earnings_calendar import get_earnings_manager

                self._earnings_manager = get_earnings_manager()
            except ImportError:
                logger.warning("Earnings calendar not available")

        # VaR calculator
        self._var_calculator = None
        try:
            from src.risk.portfolio_var import get_var_calculator

            self._var_calculator = get_var_calculator()
        except ImportError:
            logger.warning("VaR calculator not available")

        # Alert manager
        self._alert_manager = None
        if self.enable_alerts:
            try:
                from src.notifications.alert_manager import get_alert_manager

                self._alert_manager = get_alert_manager()
            except ImportError:
                logger.warning("Alert manager not available")

    def _normalize_price(self, price: float) -> float:
        """Normalize price to VND (vnstock returns prices in thousands)."""
        try:
            from src.utils.vietnam_market import normalize_price_to_vnd
            return normalize_price_to_vnd(price)
        except ImportError:
            # Fallback: if price < 1000, assume it's in thousands
            if 0 < price < 1000:
                return price * 1000
            return price

    def analyze_entry(
        self,
        symbol: str,
        df: pd.DataFrame,
        ml_signal: Optional[Dict] = None,
        market_regime: Optional[Dict] = None,
        portfolio_value: float = 100_000_000,
        current_positions: Dict[str, Dict] = None,
        sector: str = None,
    ) -> EnhancedEntryResult:
        """
        Comprehensive entry analysis

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            ml_signal: ML signal dict
            market_regime: Market regime info
            portfolio_value: Total portfolio value
            current_positions: Current positions dict
            sector: Stock sector

        Returns:
            EnhancedEntryResult
        """
        current_positions = current_positions or {}

        filters_passed = []
        filters_failed = []
        warnings = []
        recommendations = []

        # Get current price and normalize to VND
        current_price = df["close"].iloc[-1] if not df.empty else 0
        current_price = self._normalize_price(current_price)

        # Initialize scores
        technical_score = 50.0
        fundamental_score = 50.0
        timing_score = 50.0
        risk_score = 50.0

        # Position multiplier starts at 1.0
        position_multiplier = 1.0

        # =====================================================================
        # FILTER 1: Market Regime
        # =====================================================================
        if market_regime:
            if not market_regime.get("tradeable", True):
                filters_failed.append("MARKET_REGIME")
                return self._create_no_entry_result(
                    symbol,
                    current_price,
                    "Market regime not tradeable",
                    filters_passed,
                    filters_failed,
                    warnings,
                )

            regime = market_regime.get("regime", "SIDEWAYS")
            if regime == "BULL":
                position_multiplier *= 1.1
                filters_passed.append("MARKET_REGIME_BULL")
            elif regime == "BEAR":
                position_multiplier *= 0.6
                warnings.append("⚠️ Bear market - reduced position size")
            elif regime == "HIGH_VOLATILITY":
                position_multiplier *= 0.5
                warnings.append("⚠️ High volatility - reduced position size")

            filters_passed.append("MARKET_REGIME")

        # =====================================================================
        # FILTER 2: Session Timing
        # =====================================================================
        try:
            from src.market.session_trading import analyze_entry_timing

            timing = analyze_entry_timing()

            timing_score = timing.quality_score
            position_multiplier *= timing.position_size_multiplier

            if timing.is_optimal:
                filters_passed.append("SESSION_TIMING")
                recommendations.append(
                    f"✅ {timing.reasons[0] if timing.reasons else 'Good timing'}"
                )
            else:
                if timing.quality_score < 40:
                    filters_failed.append("SESSION_TIMING")
                    warnings.extend(timing.warnings)
                else:
                    filters_passed.append("SESSION_TIMING_ACCEPTABLE")
                    warnings.extend(timing.warnings)
        except Exception as e:
            logger.debug(f"Session timing check failed: {e}")

        # =====================================================================
        # FILTER 3: Fundamental Analysis
        # =====================================================================
        if self._fundamental_analyzer and self.enable_fundamental:
            try:
                fund_score = self._fundamental_analyzer.calculate_fundamental_score(symbol, sector)
                fundamental_score = fund_score.total_score

                if fundamental_score < self.min_fundamental_score:
                    filters_failed.append("FUNDAMENTAL_SCORE")
                    warnings.append(
                        f"⚠️ Low fundamental score: {fundamental_score:.0f}/100 "
                        f"(min: {self.min_fundamental_score})"
                    )
                    position_multiplier *= 0.7
                else:
                    filters_passed.append("FUNDAMENTAL_SCORE")
                    if fund_score.recommendation in ["STRONG_BUY", "BUY"]:
                        position_multiplier *= 1.1
                        recommendations.append(
                            f"✅ Strong fundamentals: {fund_score.recommendation}"
                        )

                warnings.extend(fund_score.warnings)

            except Exception as e:
                logger.debug(f"Fundamental analysis failed: {e}")

        # =====================================================================
        # FILTER 4: Earnings Calendar
        # =====================================================================
        if self._earnings_manager and self.enable_earnings_check:
            try:
                earnings_risk = self._earnings_manager.assess_earnings_risk(symbol)

                if earnings_risk.has_upcoming_earnings:
                    position_multiplier *= earnings_risk.position_multiplier

                    if earnings_risk.should_avoid_entry:
                        filters_failed.append("EARNINGS_RISK")
                        warnings.append(
                            f"⚠️ Earnings in {earnings_risk.days_until_earnings} days - "
                            f"avoid entry"
                        )
                    else:
                        filters_passed.append("EARNINGS_CHECK")

                    warnings.extend(earnings_risk.warnings)
                    recommendations.extend(earnings_risk.recommendations)
                else:
                    filters_passed.append("EARNINGS_CHECK")

            except Exception as e:
                logger.debug(f"Earnings check failed: {e}")

        # =====================================================================
        # FILTER 5: Ex-Dividend Check
        # =====================================================================
        if self._fundamental_analyzer:
            try:
                div_multiplier, div_reason = (
                    self._fundamental_analyzer.get_dividend_risk_adjustment(symbol)
                )
                position_multiplier *= div_multiplier

                if div_multiplier < 1.0:
                    warnings.append(f"⚠️ {div_reason}")
                elif div_multiplier > 1.0:
                    recommendations.append(f"✅ {div_reason}")

            except Exception as e:
                logger.debug(f"Dividend check failed: {e}")

        # =====================================================================
        # FILTER 6: Technical Analysis
        # =====================================================================
        technical_score, tech_reasons, tech_warnings = self._analyze_technical(df, ml_signal)

        if technical_score >= 60:
            filters_passed.append("TECHNICAL_ANALYSIS")
            recommendations.extend(tech_reasons)
        elif technical_score >= 40:
            filters_passed.append("TECHNICAL_ACCEPTABLE")
            warnings.extend(tech_warnings)
        else:
            filters_failed.append("TECHNICAL_ANALYSIS")
            warnings.extend(tech_warnings)
            position_multiplier *= 0.5

        # =====================================================================
        # FILTER 7: Liquidity Check
        # =====================================================================
        liquidity_ok, liquidity_msg = self._check_liquidity(df, current_price)

        if liquidity_ok:
            filters_passed.append("LIQUIDITY")
        else:
            filters_failed.append("LIQUIDITY")
            warnings.append(f"⚠️ {liquidity_msg}")
            position_multiplier *= 0.5

        # =====================================================================
        # FILTER 8: Portfolio VaR Check
        # =====================================================================
        if self._var_calculator and current_positions:
            try:
                # Calculate current VaR
                current_var = self._var_calculator.calculate_var(portfolio_value, current_positions)

                if current_var.var_percent > self.max_portfolio_var_pct:
                    filters_failed.append("PORTFOLIO_VAR")
                    warnings.append(
                        f"⚠️ Portfolio VaR too high: {current_var.var_percent:.1f}% "
                        f"(max: {self.max_portfolio_var_pct}%)"
                    )
                    position_multiplier *= 0.5
                else:
                    filters_passed.append("PORTFOLIO_VAR")
                    risk_score = 100 - (current_var.var_percent / self.max_portfolio_var_pct * 50)

            except Exception as e:
                logger.debug(f"VaR check failed: {e}")

        # =====================================================================
        # FILTER 9: Real-time Foreign Flow
        # =====================================================================
        if self._realtime_manager and self.enable_realtime:
            try:
                foreign_flow = self._realtime_manager.get_foreign_flow_realtime()

                if foreign_flow:
                    vnindex_flow = foreign_flow.get("VNINDEX", {})
                    net_volume = vnindex_flow.get("net_volume", 0)

                    if net_volume > 1_000_000:  # Strong foreign buying
                        position_multiplier *= 1.1
                        recommendations.append("✅ Strong foreign buying detected")
                    elif net_volume < -1_000_000:  # Strong foreign selling
                        position_multiplier *= 0.8
                        warnings.append("⚠️ Foreign selling pressure")

            except Exception as e:
                logger.debug(f"Foreign flow check failed: {e}")

        # =====================================================================
        # CALCULATE FINAL DECISION
        # =====================================================================

        # Calculate composite confidence
        confidence = (
            technical_score * 0.35
            + fundamental_score * 0.25
            + timing_score * 0.20
            + risk_score * 0.20
        )

        # Adjust for failed filters
        critical_failures = ["MARKET_REGIME", "LIQUIDITY", "EARNINGS_RISK"]
        has_critical_failure = any(f in filters_failed for f in critical_failures)

        if has_critical_failure:
            confidence *= 0.5

        # Final decision
        should_enter = (
            confidence >= self.min_confidence
            and not has_critical_failure
            and position_multiplier >= 0.3
        )

        # Calculate position size
        max_position_value = portfolio_value * 0.15  # Max 15% per position
        recommended_value = max_position_value * position_multiplier
        recommended_shares = self._calculate_shares(recommended_value, current_price)

        # Calculate stop loss and take profit
        stop_loss = self._calculate_stop_loss(df, current_price)
        take_profit_targets = self._calculate_take_profits(current_price, stop_loss)

        # Risk/reward ratio
        risk = current_price - stop_loss
        reward = take_profit_targets[0] - current_price if take_profit_targets else risk * 2
        risk_reward = reward / risk if risk > 0 else 0

        # Send alert if should enter
        if should_enter and self._alert_manager:
            self._alert_manager.signal_buy(
                symbol, current_price, f"Confidence: {confidence:.0f}%, R:R: {risk_reward:.1f}"
            )

        return EnhancedEntryResult(
            should_enter=should_enter,
            signal_type="BUY" if should_enter else "HOLD",
            confidence=confidence,
            position_multiplier=position_multiplier,
            recommended_shares=recommended_shares,
            recommended_value=recommended_value,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_targets=take_profit_targets,
            risk_reward_ratio=risk_reward,
            var_impact=0,  # TODO: Calculate
            portfolio_risk_after=0,  # TODO: Calculate
            filters_passed=filters_passed,
            filters_failed=filters_failed,
            warnings=warnings,
            recommendations=recommendations,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            timing_score=timing_score,
            risk_score=risk_score,
        )

    def _analyze_technical(
        self, df: pd.DataFrame, ml_signal: Optional[Dict]
    ) -> Tuple[float, List[str], List[str]]:
        """Analyze technical indicators"""
        score = 50.0
        reasons = []
        warnings = []

        if df.empty or len(df) < 50:
            return 30.0, [], ["Insufficient data"]

        close = df["close"]

        # Trend analysis
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]

        if current > sma20 > sma50:
            score += 15
            reasons.append("✅ Uptrend (Price > SMA20 > SMA50)")
        elif current < sma20 < sma50:
            score -= 15
            warnings.append("⚠️ Downtrend")

        # RSI
        if "rsi" in df.columns:
            rsi = df["rsi"].iloc[-1]
            if rsi < 30:
                score += 10
                reasons.append(f"✅ RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                score -= 10
                warnings.append(f"⚠️ RSI overbought ({rsi:.0f})")

        # Volume
        if "volume" in df.columns:
            vol_avg = df["volume"].rolling(20).mean().iloc[-1]
            vol_current = df["volume"].iloc[-1]

            if vol_current > vol_avg * 1.5:
                score += 5
                reasons.append("✅ Volume surge")

        # ML signal
        if ml_signal:
            ml_conf = ml_signal.get("confidence", 0)
            if ml_signal.get("signal") == "BUY" and ml_conf >= 60:
                score += 15
                reasons.append(f"✅ ML BUY signal ({ml_conf:.0f}%)")
            elif ml_signal.get("signal") == "SELL":
                score -= 20
                warnings.append("⚠️ ML SELL signal")

        return min(100, max(0, score)), reasons, warnings

    def _check_liquidity(self, df: pd.DataFrame, current_price: float) -> Tuple[bool, str]:
        """Check liquidity requirements"""
        if "volume" not in df.columns or len(df) < 20:
            return False, "Insufficient volume data"

        avg_volume = df["volume"].tail(20).mean()
        # current_price is already normalized to VND
        avg_value = avg_volume * current_price

        min_liquidity = 2_000_000_000  # 2B VND

        if avg_value < min_liquidity:
            return False, f"Low liquidity: {avg_value/1e9:.1f}B VND (min: 2B)"

        return True, f"Good liquidity: {avg_value/1e9:.1f}B VND"

    def _calculate_shares(self, value: float, price: float) -> int:
        """Calculate shares rounded to lot size"""
        if price <= 0:
            return 0
        shares = int(value / price)
        return (shares // 100) * 100  # Round to lot of 100

    def _calculate_stop_loss(self, df: pd.DataFrame, current_price: float) -> float:
        """Calculate stop loss price"""
        # Use ATR if available
        if "atr" in df.columns:
            atr = df["atr"].iloc[-1]
            stop_loss = current_price - (atr * 2)
        else:
            # Default 5.5% stop loss
            stop_loss = current_price * 0.945

        # Round to tick
        return self._round_to_tick(stop_loss)

    def _calculate_take_profits(self, current_price: float, stop_loss: float) -> List[float]:
        """Calculate take profit targets"""
        risk = current_price - stop_loss

        # TP1: 1.5R, TP2: 2.5R, TP3: 4R
        tp1 = current_price + (risk * 1.5)
        tp2 = current_price + (risk * 2.5)
        tp3 = current_price + (risk * 4.0)

        return [self._round_to_tick(tp1), self._round_to_tick(tp2), self._round_to_tick(tp3)]

    def _round_to_tick(self, price: float) -> float:
        """Round to Vietnam tick size"""
        if price < 10000:
            tick = 10
        elif price < 50000:
            tick = 50
        else:
            tick = 100
        return round(price / tick) * tick

    def _create_no_entry_result(
        self,
        symbol: str,
        price: float,
        reason: str,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> EnhancedEntryResult:
        """Create no-entry result"""
        return EnhancedEntryResult(
            should_enter=False,
            signal_type="HOLD",
            confidence=0,
            position_multiplier=0,
            recommended_shares=0,
            recommended_value=0,
            entry_price=price,
            stop_loss=0,
            take_profit_targets=[],
            risk_reward_ratio=0,
            var_impact=0,
            portfolio_risk_after=0,
            filters_passed=passed,
            filters_failed=failed,
            warnings=warnings + [f"❌ {reason}"],
            recommendations=[],
        )


# Singleton
_enhanced_entry: Optional[EnhancedEntryLogicV2] = None


def get_enhanced_entry_v2() -> EnhancedEntryLogicV2:
    """Get singleton enhanced entry logic"""
    global _enhanced_entry
    if _enhanced_entry is None:
        _enhanced_entry = EnhancedEntryLogicV2()
    return _enhanced_entry
