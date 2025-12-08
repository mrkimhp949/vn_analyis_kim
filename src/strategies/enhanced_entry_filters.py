# -*- coding: utf-8 -*-
"""
Enhanced Entry Filters Integration
Tích hợp các module cải thiện vào entry logic

INTEGRATIONS:
1. Enhanced Market Regime (VN30, HNX, margin debt)
2. Session Trading (ATO/ATC timing)
3. Fundamental Analysis (P/E, earnings calendar)
4. Entry Timing Optimization
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EnhancedEntryResult:
    """Enhanced entry analysis result"""

    should_enter: bool
    confidence_adjustment: int
    position_size_multiplier: float

    # Component results
    regime_check: Dict
    timing_check: Dict
    fundamental_check: Dict

    # Aggregated
    reasons: List[str]
    warnings: List[str]

    # Recommended order type
    order_type: str  # MARKET, LIMIT, ATO, ATC


class EnhancedEntryFilters:
    """
    Enhanced entry filters integrating:
    - Multi-index regime detection
    - ATO/ATC session timing
    - Fundamental analysis
    - Earnings calendar
    """

    def __init__(
        self,
        use_enhanced_regime: bool = True,
        use_session_timing: bool = True,
        use_fundamentals: bool = True,
        use_earnings_calendar: bool = True,
    ):
        self.use_enhanced_regime = use_enhanced_regime
        self.use_session_timing = use_session_timing
        self.use_fundamentals = use_fundamentals
        self.use_earnings_calendar = use_earnings_calendar

        # Initialize components
        self._regime_detector = None
        self._session_manager = None
        self._fundamental_analyzer = None

    def _get_regime_detector(self):
        """Lazy load regime detector"""
        if self._regime_detector is None:
            try:
                from src.market.regime_detector import get_enhanced_regime_detector

                self._regime_detector = get_enhanced_regime_detector()
            except ImportError:
                logger.warning("Enhanced regime detector not available")
        return self._regime_detector

    def _get_session_manager(self):
        """Lazy load session manager"""
        if self._session_manager is None:
            try:
                from src.market.session_trading import get_session_manager

                self._session_manager = get_session_manager()
            except ImportError:
                logger.warning("Session trading manager not available")
        return self._session_manager

    def _get_fundamental_analyzer(self):
        """Lazy load fundamental analyzer"""
        if self._fundamental_analyzer is None:
            try:
                from src.data.fundamental_analyzer import get_fundamental_analyzer

                self._fundamental_analyzer = get_fundamental_analyzer()
            except ImportError:
                logger.warning("Fundamental analyzer not available")
        return self._fundamental_analyzer

    def analyze(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        vnindex_df: Optional[pd.DataFrame] = None,
        vn30_df: Optional[pd.DataFrame] = None,
        hnx_df: Optional[pd.DataFrame] = None,
        sector: Optional[str] = None,
        urgency: str = "NORMAL",
    ) -> EnhancedEntryResult:
        """
        Run enhanced entry analysis

        Args:
            symbol: Stock symbol
            df: Stock OHLCV data (reserved for future technical confirmation)
            vnindex_df: VNINDEX data for regime detection
            vn30_df: VN30 data (optional)
            hnx_df: HNX data (optional)
            sector: Stock sector
            urgency: Trade urgency (LOW, NORMAL, HIGH)

        Returns:
            EnhancedEntryResult
        """
        # Note: df parameter reserved for future technical analysis integration
        _ = df  # Suppress unused variable warning

        reasons = []
        warnings = []
        confidence_adjustment = 0
        position_multiplier = 1.0

        # 1. Enhanced Regime Check
        regime_check = self._check_enhanced_regime(vnindex_df, vn30_df, hnx_df)
        if regime_check["adjustment"] != 0:
            confidence_adjustment += regime_check["adjustment"]
        reasons.extend(regime_check.get("reasons", []))
        warnings.extend(regime_check.get("warnings", []))

        # 2. Session Timing Check
        timing_check = self._check_session_timing(urgency)
        if timing_check["adjustment"] != 0:
            confidence_adjustment += timing_check["adjustment"]
        position_multiplier *= timing_check.get("size_multiplier", 1.0)
        reasons.extend(timing_check.get("reasons", []))
        warnings.extend(timing_check.get("warnings", []))

        # 3. Fundamental Check
        fundamental_check = self._check_fundamentals(symbol, sector)
        if fundamental_check["adjustment"] != 0:
            confidence_adjustment += fundamental_check["adjustment"]
        reasons.extend(fundamental_check.get("reasons", []))
        warnings.extend(fundamental_check.get("warnings", []))

        # 4. Earnings Calendar Check
        if self.use_earnings_calendar:
            earnings_check = self._check_earnings_calendar(symbol)
            if earnings_check["adjustment"] != 0:
                confidence_adjustment += earnings_check["adjustment"]
            position_multiplier *= earnings_check.get("size_multiplier", 1.0)
            warnings.extend(earnings_check.get("warnings", []))

        # 5. Ex-Dividend Check (NEW)
        dividend_check = self._check_ex_dividend(symbol)
        if dividend_check["adjustment"] != 0:
            confidence_adjustment += dividend_check["adjustment"]
        position_multiplier *= dividend_check.get("size_multiplier", 1.0)
        warnings.extend(dividend_check.get("warnings", []))
        reasons.extend(dividend_check.get("reasons", []))

        # Determine if should enter
        should_enter = True

        # Block if regime not tradeable
        if not regime_check.get("tradeable", True):
            should_enter = False
            warnings.append("🚫 Market regime not tradeable")

        # Block if timing is AVOID
        if timing_check.get("entry_quality") == "AVOID" and urgency != "HIGH":
            should_enter = False
            warnings.append("🚫 Entry timing not favorable")

        # Determine order type
        order_type = timing_check.get("order_type", "MARKET")

        return EnhancedEntryResult(
            should_enter=should_enter,
            confidence_adjustment=confidence_adjustment,
            position_size_multiplier=position_multiplier,
            regime_check=regime_check,
            timing_check=timing_check,
            fundamental_check=fundamental_check,
            reasons=reasons,
            warnings=warnings,
            order_type=order_type,
        )

    def _check_enhanced_regime(
        self,
        vnindex_df: Optional[pd.DataFrame],
        vn30_df: Optional[pd.DataFrame],
        hnx_df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Check enhanced market regime"""
        result = {
            "adjustment": 0,
            "tradeable": True,
            "reasons": [],
            "warnings": [],
        }

        if not self.use_enhanced_regime:
            return result

        detector = self._get_regime_detector()
        if detector is None or vnindex_df is None:
            return result

        try:
            regime = detector.detect(vnindex_df, vn30_df, hnx_df)

            result["regime"] = regime.regime
            result["confidence"] = regime.confidence
            result["tradeable"] = regime.tradeable

            # Adjustments based on regime
            if regime.regime == "BULL":
                result["adjustment"] = 5
                result["reasons"].append(f"✅ Bull market ({regime.confidence:.0f}% conf)")
            elif regime.regime == "BEAR":
                result["adjustment"] = -15
                result["warnings"].append(f"⚠️ Bear market ({regime.confidence:.0f}% conf)")
            elif regime.regime == "HIGH_VOLATILITY":
                result["adjustment"] = -20
                result["warnings"].append("⚠️ High volatility regime")
            elif regime.regime == "CORRECTION":
                result["adjustment"] = -5
                result["warnings"].append("📉 Market correction - watch for reversal")

            # Foreign flow signal
            if regime.foreign_flow_signal == "BUYING":
                result["adjustment"] += 5
                result["reasons"].append("🌍 Foreign buying detected")
            elif regime.foreign_flow_signal == "SELLING":
                result["adjustment"] -= 5
                result["warnings"].append("⚠️ Foreign selling pressure")

            # Margin debt warning
            if regime.margin_debt_signal == "HIGH_RISK":
                result["adjustment"] -= 10
                result["warnings"].append("⚠️ High margin debt - market risk elevated")

            # Correlation breakdown
            if regime.correlation_breakdown:
                result["adjustment"] -= 10
                result["warnings"].append("⚠️ Index correlation breakdown - market stress")

            # Add recommendations
            result["recommendations"] = regime.recommendations

        except Exception as e:
            logger.warning(f"Enhanced regime check failed: {e}")

        return result

    def _check_session_timing(self, urgency: str) -> Dict[str, Any]:
        """Check session timing"""
        result = {
            "adjustment": 0,
            "size_multiplier": 1.0,
            "entry_quality": "ACCEPTABLE",
            "order_type": "MARKET",
            "reasons": [],
            "warnings": [],
        }

        if not self.use_session_timing:
            return result

        manager = self._get_session_manager()
        if manager is None:
            return result

        try:
            # Get current session
            session = manager.get_current_session()
            result["session"] = session.session_type.value
            result["entry_quality"] = session.entry_quality
            result["order_type"] = session.recommended_order_type.value

            # Analyze entry timing
            timing = manager.analyze_entry_timing(urgency=urgency)
            result["size_multiplier"] = timing.position_size_multiplier

            # Adjustments
            if timing.is_optimal:
                result["adjustment"] = 5
                result["reasons"].append(
                    f"✅ Optimal entry timing ({timing.quality_score:.0f}/100)"
                )
            elif session.entry_quality == "AVOID":
                result["adjustment"] = -15
                result["warnings"].append(f"⚠️ Suboptimal timing: {session.session_type.value}")
            elif session.entry_quality == "ACCEPTABLE":
                result["adjustment"] = 0

            # Add session warnings
            result["warnings"].extend(session.warnings)
            result["reasons"].extend(timing.reasons)

            # Next optimal window
            next_window = manager.get_next_optimal_window()
            if next_window and not timing.is_optimal:
                if "minutes_until" in next_window:
                    result["warnings"].append(
                        f"💡 Next optimal window in {next_window['minutes_until']} mins "
                        f"({next_window['start']}-{next_window['end']})"
                    )

        except Exception as e:
            logger.warning(f"Session timing check failed: {e}")

        return result

    def _check_fundamentals(self, symbol: str, sector: Optional[str]) -> Dict[str, Any]:
        """Check fundamental analysis"""
        result = {
            "adjustment": 0,
            "reasons": [],
            "warnings": [],
        }

        if not self.use_fundamentals:
            return result

        analyzer = self._get_fundamental_analyzer()
        if analyzer is None:
            return result

        try:
            score = analyzer.calculate_fundamental_score(symbol, sector)

            result["total_score"] = score.total_score
            result["recommendation"] = score.recommendation

            # Adjustments based on score
            if score.recommendation == "STRONG_BUY":
                result["adjustment"] = 10
                result["reasons"].append(f"✅ Strong fundamentals ({score.total_score:.0f}/100)")
            elif score.recommendation == "BUY":
                result["adjustment"] = 5
                result["reasons"].append(f"✅ Good fundamentals ({score.total_score:.0f}/100)")
            elif score.recommendation == "AVOID":
                result["adjustment"] = -10
                result["warnings"].append(f"⚠️ Weak fundamentals ({score.total_score:.0f}/100)")
            elif score.recommendation == "STRONG_AVOID":
                result["adjustment"] = -20
                result["warnings"].append(f"🚫 Poor fundamentals ({score.total_score:.0f}/100)")

            # Add specific warnings
            result["warnings"].extend(score.warnings)

            # Component scores
            result["valuation_score"] = score.valuation_score
            result["profitability_score"] = score.profitability_score
            result["health_score"] = score.financial_health_score
            result["growth_score"] = score.growth_score

        except Exception as e:
            logger.warning(f"Fundamental check failed for {symbol}: {e}")

        return result

    def _check_earnings_calendar(self, symbol: str) -> Dict[str, Any]:
        """Check earnings calendar"""
        result = {
            "adjustment": 0,
            "size_multiplier": 1.0,
            "warnings": [],
        }

        if not self.use_earnings_calendar:
            return result

        analyzer = self._get_fundamental_analyzer()
        if analyzer is None:
            return result

        try:
            is_near, event = analyzer.is_near_earnings(symbol)

            if is_near and event:
                multiplier, reason = analyzer.get_earnings_risk_adjustment(symbol)
                result["size_multiplier"] = multiplier
                result["adjustment"] = -10  # Penalty for earnings uncertainty
                result["warnings"].append(f"📅 {reason}")
                result["earnings_date"] = event.announcement_date.strftime("%Y-%m-%d")
                result["fiscal_quarter"] = event.fiscal_quarter

        except Exception as e:
            logger.warning(f"Earnings calendar check failed for {symbol}: {e}")

        return result

    def _check_ex_dividend(self, symbol: str) -> Dict[str, Any]:
        """
        Check ex-dividend calendar.

        Avoid buying right before ex-dividend date as price will drop.
        Can be opportunity after ex-date.
        """
        result = {
            "adjustment": 0,
            "size_multiplier": 1.0,
            "warnings": [],
            "reasons": [],
        }

        analyzer = self._get_fundamental_analyzer()
        if analyzer is None:
            return result

        try:
            is_near, info = analyzer.is_near_ex_dividend(symbol)

            if is_near and info:
                multiplier, reason = analyzer.get_dividend_risk_adjustment(symbol)
                result["size_multiplier"] = multiplier

                position = info.get("position")

                if position == "BEFORE_EX_DATE":
                    days_until = info.get("days_until", 0)
                    result["adjustment"] = -15 if days_until <= 1 else -10
                    result["warnings"].append(f"💰 {reason}")
                    result["ex_date"] = info.get("ex_date")

                elif position == "AFTER_EX_DATE":
                    result["adjustment"] = 5  # Slight bonus for post-ex opportunity
                    result["reasons"].append(f"💰 {reason}")

        except Exception as e:
            logger.debug(f"Ex-dividend check failed for {symbol}: {e}")

        return result


# Singleton instance
_enhanced_filters = None


def get_enhanced_entry_filters() -> EnhancedEntryFilters:
    """Get singleton instance"""
    global _enhanced_filters
    if _enhanced_filters is None:
        _enhanced_filters = EnhancedEntryFilters()
    return _enhanced_filters


def reset_enhanced_filters() -> None:
    """Reset singleton instance (useful for testing)"""
    global _enhanced_filters
    _enhanced_filters = None


def run_enhanced_entry_analysis(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
    vnindex_df: Optional[pd.DataFrame] = None,
    vn30_df: Optional[pd.DataFrame] = None,
    hnx_df: Optional[pd.DataFrame] = None,
    sector: Optional[str] = None,
    urgency: str = "NORMAL",
) -> EnhancedEntryResult:
    """
    Run enhanced entry analysis (convenience function).

    Args:
        symbol: Stock symbol
        df: Stock OHLCV data (reserved for future use)
        vnindex_df: VNINDEX data for regime detection
        vn30_df: VN30 data (optional)
        hnx_df: HNX data (optional)
        sector: Stock sector
        urgency: Trade urgency (LOW, NORMAL, HIGH)

    Returns:
        EnhancedEntryResult
    """
    filters = get_enhanced_entry_filters()
    return filters.analyze(
        symbol=symbol,
        df=df,
        vnindex_df=vnindex_df,
        vn30_df=vn30_df,
        hnx_df=hnx_df,
        sector=sector,
        urgency=urgency,
    )


# Test
if __name__ == "__main__":
    print("Testing Enhanced Entry Filters...")

    import numpy as np

    # Create dummy data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    prices = 50000 + np.cumsum(np.random.randn(100) * 500)

    df = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(100000, 500000, 100),
        },
        index=dates,
    )

    # Test
    filters = EnhancedEntryFilters(
        use_enhanced_regime=False,  # Skip for test
        use_session_timing=True,
        use_fundamentals=False,  # Skip for test
        use_earnings_calendar=False,
    )

    result = filters.analyze("VNM", df)

    print(f"\nShould Enter: {result.should_enter}")
    print(f"Confidence Adjustment: {result.confidence_adjustment:+d}")
    print(f"Position Multiplier: {result.position_size_multiplier:.2f}")
    print(f"Order Type: {result.order_type}")

    print("\nReasons:")
    for r in result.reasons:
        print(f"  {r}")

    print("\nWarnings:")
    for w in result.warnings:
        print(f"  {w}")

    print("\n✅ Enhanced Entry Filters test completed!")
