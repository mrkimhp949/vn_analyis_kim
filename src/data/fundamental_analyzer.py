# -*- coding: utf-8 -*-
"""
Fundamental Analysis Integration
P/E, Debt Ratio, Earnings Calendar, Sector Valuation

FEATURES:
- P/E ratio analysis with sector comparison
- Debt ratio and financial health
- Earnings calendar integration
- Sector-specific valuation metrics
- Fundamental score calculation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FundamentalMetrics:
    """Fundamental metrics for a stock"""

    symbol: str

    # Valuation
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None

    # Profitability
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    profit_margin: Optional[float] = None

    # Financial Health
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None

    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None

    # Dividend
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

    # Market data
    market_cap: Optional[float] = None

    # Timestamps
    last_updated: Optional[datetime] = None


@dataclass
class EarningsEvent:
    """Earnings announcement event"""

    symbol: str
    announcement_date: datetime
    fiscal_quarter: str  # Q1, Q2, Q3, Q4
    fiscal_year: int

    # Estimates (if available)
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None

    # Actual (after announcement)
    eps_actual: Optional[float] = None
    revenue_actual: Optional[float] = None

    # Surprise
    eps_surprise_pct: Optional[float] = None

    # Status
    is_announced: bool = False


@dataclass
class FundamentalScore:
    """Composite fundamental score"""

    symbol: str
    total_score: float  # 0-100

    # Component scores
    valuation_score: float = 0.0
    profitability_score: float = 0.0
    financial_health_score: float = 0.0
    growth_score: float = 0.0

    # Flags
    is_undervalued: bool = False
    is_profitable: bool = False
    is_financially_healthy: bool = False
    is_growing: bool = False

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Recommendation
    recommendation: str = "NEUTRAL"  # STRONG_BUY, BUY, NEUTRAL, AVOID, STRONG_AVOID


# Vietnam market sector P/E benchmarks (approximate)
SECTOR_PE_BENCHMARKS = {
    "Ngân hàng": {"low": 6, "median": 10, "high": 15},
    "Bất động sản": {"low": 8, "median": 15, "high": 25},
    "Chứng khoán": {"low": 8, "median": 12, "high": 20},
    "Thép": {"low": 5, "median": 8, "high": 12},
    "Dầu khí": {"low": 6, "median": 10, "high": 15},
    "Điện": {"low": 10, "median": 15, "high": 20},
    "Bán lẻ": {"low": 12, "median": 18, "high": 30},
    "Công nghệ": {"low": 15, "median": 25, "high": 40},
    "Dược phẩm": {"low": 12, "median": 18, "high": 25},
    "Thực phẩm": {"low": 10, "median": 15, "high": 22},
    "Xây dựng": {"low": 6, "median": 10, "high": 15},
    "Vận tải": {"low": 8, "median": 12, "high": 18},
    "Default": {"low": 8, "median": 15, "high": 25},
}


class FundamentalAnalyzer:
    """
    Fundamental Analysis for Vietnam stocks

    Features:
    - P/E ratio analysis with sector comparison
    - Financial health assessment
    - Earnings calendar tracking
    - Composite fundamental score
    """

    def __init__(self):
        self._metrics_cache: Dict[str, FundamentalMetrics] = {}
        self._earnings_calendar: Dict[str, List[EarningsEvent]] = {}
        self._cache_ttl = 3600 * 24  # 24 hours

    def get_fundamental_metrics(
        self,
        symbol: str,
        force_refresh: bool = False,
    ) -> Optional[FundamentalMetrics]:
        """
        Get fundamental metrics for a stock

        Args:
            symbol: Stock symbol
            force_refresh: Force refresh from source

        Returns:
            FundamentalMetrics or None
        """
        # Check cache
        if not force_refresh and symbol in self._metrics_cache:
            cached = self._metrics_cache[symbol]
            if cached.last_updated:
                age = (datetime.now() - cached.last_updated).seconds
                if age < self._cache_ttl:
                    return cached

        # Try to fetch from data source
        metrics = self._fetch_fundamental_data(symbol)

        if metrics:
            self._metrics_cache[symbol] = metrics

        return metrics

    def _fetch_fundamental_data(self, symbol: str) -> Optional[FundamentalMetrics]:
        """
        Fetch fundamental data from source

        This would typically integrate with:
        - TCBS API
        - SSI API
        - VNDirect API
        - Cafef/Vietstock scraping
        """
        try:
            # Try TCBS provider first
            from src.data.tcbs_provider import get_tcbs_provider

            provider = get_tcbs_provider()
            data = provider.get_fundamental_data(symbol)

            if data:
                return FundamentalMetrics(
                    symbol=symbol,
                    pe_ratio=data.get("pe"),
                    pb_ratio=data.get("pb"),
                    roe=data.get("roe"),
                    debt_to_equity=data.get("debt_to_equity"),
                    dividend_yield=data.get("dividend_yield"),
                    market_cap=data.get("market_cap"),
                    last_updated=datetime.now(),
                )

        except ImportError:
            logger.debug("TCBS provider not available")
        except Exception as e:
            logger.warning(f"Error fetching fundamental data for {symbol}: {e}")

        # Return placeholder with None values
        return FundamentalMetrics(
            symbol=symbol,
            last_updated=datetime.now(),
        )

    def calculate_fundamental_score(
        self,
        symbol: str,
        sector: Optional[str] = None,
    ) -> FundamentalScore:
        """
        Calculate composite fundamental score

        Args:
            symbol: Stock symbol
            sector: Stock sector for comparison

        Returns:
            FundamentalScore object
        """
        metrics = self.get_fundamental_metrics(symbol)

        if metrics is None:
            return FundamentalScore(
                symbol=symbol,
                total_score=50.0,
                warnings=["No fundamental data available"],
                recommendation="NEUTRAL",
            )

        warnings = []

        # 1. Valuation Score (25%)
        valuation_score = self._calculate_valuation_score(metrics, sector, warnings)

        # 2. Profitability Score (25%)
        profitability_score = self._calculate_profitability_score(metrics, warnings)

        # 3. Financial Health Score (25%)
        health_score = self._calculate_health_score(metrics, warnings)

        # 4. Growth Score (25%)
        growth_score = self._calculate_growth_score(metrics, warnings)

        # Composite score
        total_score = (
            valuation_score * 0.25
            + profitability_score * 0.25
            + health_score * 0.25
            + growth_score * 0.25
        )

        # Determine flags
        is_undervalued = valuation_score >= 60
        is_profitable = profitability_score >= 60
        is_healthy = health_score >= 60
        is_growing = growth_score >= 60

        # Recommendation
        recommendation = self._get_recommendation(
            total_score, is_undervalued, is_profitable, is_healthy, is_growing
        )

        return FundamentalScore(
            symbol=symbol,
            total_score=total_score,
            valuation_score=valuation_score,
            profitability_score=profitability_score,
            financial_health_score=health_score,
            growth_score=growth_score,
            is_undervalued=is_undervalued,
            is_profitable=is_profitable,
            is_financially_healthy=is_healthy,
            is_growing=is_growing,
            warnings=warnings,
            recommendation=recommendation,
        )

    def _calculate_valuation_score(
        self,
        metrics: FundamentalMetrics,
        sector: Optional[str],
        warnings: List[str],
    ) -> float:
        """Calculate valuation score based on P/E, P/B"""
        score = 50.0  # Neutral default

        # Get sector benchmarks
        benchmarks = SECTOR_PE_BENCHMARKS.get(sector, SECTOR_PE_BENCHMARKS["Default"])

        # P/E analysis
        if metrics.pe_ratio is not None:
            pe = metrics.pe_ratio

            if pe <= 0:
                warnings.append("⚠️ Negative P/E (company losing money)")
                score -= 20
            elif pe < benchmarks["low"]:
                score += 20  # Potentially undervalued
            elif pe < benchmarks["median"]:
                score += 10  # Fair value
            elif pe < benchmarks["high"]:
                score -= 5  # Slightly expensive
            else:
                score -= 15  # Expensive
                warnings.append(f"⚠️ High P/E ({pe:.1f}) vs sector median ({benchmarks['median']})")

        # P/B analysis
        if metrics.pb_ratio is not None:
            pb = metrics.pb_ratio

            if pb < 1.0:
                score += 10  # Trading below book value
            elif pb < 2.0:
                score += 5  # Reasonable
            elif pb > 4.0:
                score -= 10  # Expensive
                warnings.append(f"⚠️ High P/B ({pb:.1f})")

        return max(0, min(100, score))

    def _calculate_profitability_score(
        self,
        metrics: FundamentalMetrics,
        warnings: List[str],
    ) -> float:
        """Calculate profitability score based on ROE, ROA, margins"""
        score = 50.0

        # ROE analysis
        if metrics.roe is not None:
            roe = metrics.roe

            if roe >= 20:
                score += 25  # Excellent
            elif roe >= 15:
                score += 15  # Good
            elif roe >= 10:
                score += 5  # Acceptable
            elif roe >= 0:
                score -= 5  # Below average
            else:
                score -= 20  # Losing money
                warnings.append(f"⚠️ Negative ROE ({roe:.1f}%)")

        # ROA analysis
        if metrics.roa is not None:
            roa = metrics.roa

            if roa >= 10:
                score += 15
            elif roa >= 5:
                score += 5
            elif roa < 0:
                score -= 10

        # Profit margin
        if metrics.profit_margin is not None:
            margin = metrics.profit_margin

            if margin >= 20:
                score += 10
            elif margin >= 10:
                score += 5
            elif margin < 0:
                score -= 15
                warnings.append("⚠️ Negative profit margin")

        return max(0, min(100, score))

    def _calculate_health_score(
        self,
        metrics: FundamentalMetrics,
        warnings: List[str],
    ) -> float:
        """Calculate financial health score"""
        score = 50.0

        # Debt to Equity
        if metrics.debt_to_equity is not None:
            de = metrics.debt_to_equity

            if de < 0.3:
                score += 20  # Very low debt
            elif de < 0.5:
                score += 10  # Low debt
            elif de < 1.0:
                score += 0  # Moderate
            elif de < 2.0:
                score -= 10  # High debt
            else:
                score -= 25  # Very high debt
                warnings.append(f"⚠️ High debt ratio ({de:.1f})")

        # Current ratio
        if metrics.current_ratio is not None:
            cr = metrics.current_ratio

            if cr >= 2.0:
                score += 15  # Strong liquidity
            elif cr >= 1.5:
                score += 10
            elif cr >= 1.0:
                score += 0  # Adequate
            else:
                score -= 15  # Liquidity risk
                warnings.append(f"⚠️ Low current ratio ({cr:.2f})")

        return max(0, min(100, score))

    def _calculate_growth_score(
        self,
        metrics: FundamentalMetrics,
        warnings: List[str],
    ) -> float:
        """Calculate growth score"""
        score = 50.0

        # Revenue growth
        if metrics.revenue_growth is not None:
            rg = metrics.revenue_growth

            if rg >= 20:
                score += 20  # High growth
            elif rg >= 10:
                score += 10  # Good growth
            elif rg >= 0:
                score += 0  # Stable
            else:
                score -= 15  # Declining
                warnings.append(f"⚠️ Negative revenue growth ({rg:.1f}%)")

        # Earnings growth
        if metrics.earnings_growth is not None:
            eg = metrics.earnings_growth

            if eg >= 25:
                score += 20
            elif eg >= 15:
                score += 10
            elif eg >= 0:
                score += 0
            else:
                score -= 15

        return max(0, min(100, score))

    def _get_recommendation(
        self,
        total_score: float,
        is_undervalued: bool,
        is_profitable: bool,
        is_healthy: bool,
        is_growing: bool,
    ) -> str:
        """Get recommendation based on scores"""
        positive_flags = sum([is_undervalued, is_profitable, is_healthy, is_growing])

        if total_score >= 75 and positive_flags >= 3:
            return "STRONG_BUY"
        elif total_score >= 65 and positive_flags >= 2:
            return "BUY"
        elif total_score >= 45:
            return "NEUTRAL"
        elif total_score >= 35:
            return "AVOID"
        else:
            return "STRONG_AVOID"

    # ========================================================================
    # EARNINGS CALENDAR
    # ========================================================================

    def get_upcoming_earnings(
        self,
        symbol: str,
        days_ahead: int = 30,
    ) -> List[EarningsEvent]:
        """
        Get upcoming earnings announcements

        Args:
            symbol: Stock symbol
            days_ahead: Days to look ahead

        Returns:
            List of EarningsEvent
        """
        if symbol in self._earnings_calendar:
            events = self._earnings_calendar[symbol]
            cutoff = datetime.now() + timedelta(days=days_ahead)
            return [e for e in events if e.announcement_date <= cutoff and not e.is_announced]

        # Try to fetch from source
        events = self._fetch_earnings_calendar(symbol)
        if events:
            self._earnings_calendar[symbol] = events
            cutoff = datetime.now() + timedelta(days=days_ahead)
            return [e for e in events if e.announcement_date <= cutoff and not e.is_announced]

        return []

    def _fetch_earnings_calendar(self, symbol: str) -> List[EarningsEvent]:
        """
        Fetch earnings calendar from source

        Vietnam companies typically report:
        - Q1: April-May
        - Q2: July-August
        - Q3: October-November
        - Q4/Annual: January-March (next year)
        """
        # Placeholder - would integrate with actual data source
        # Sources: Cafef, Vietstock, company IR pages
        return []

    def is_near_earnings(
        self,
        symbol: str,
        days_before: int = 5,
        days_after: int = 2,
    ) -> Tuple[bool, Optional[EarningsEvent]]:
        """
        Check if stock is near earnings announcement

        Args:
            symbol: Stock symbol
            days_before: Days before earnings to flag
            days_after: Days after earnings to flag

        Returns:
            (is_near, event)
        """
        events = self.get_upcoming_earnings(symbol, days_ahead=days_before + 30)

        now = datetime.now()

        for event in events:
            days_until = (event.announcement_date - now).days

            # Before earnings
            if 0 <= days_until <= days_before:
                return True, event

            # After earnings (if just announced)
            if event.is_announced:
                days_since = (now - event.announcement_date).days
                if days_since <= days_after:
                    return True, event

        return False, None

    def get_earnings_risk_adjustment(
        self,
        symbol: str,
    ) -> Tuple[float, str]:
        """
        Get position size adjustment for earnings risk

        Returns:
            (multiplier, reason)
        """
        is_near, event = self.is_near_earnings(symbol)

        if not is_near:
            return 1.0, "No upcoming earnings"

        if event:
            days_until = (event.announcement_date - datetime.now()).days

            if days_until <= 2:
                return 0.5, f"Earnings in {days_until} days - reduce position 50%"
            elif days_until <= 5:
                return 0.7, f"Earnings in {days_until} days - reduce position 30%"
            else:
                return 0.85, f"Earnings in {days_until} days - slight reduction"

        return 1.0, "No adjustment needed"

    # ========================================================================
    # EX-DIVIDEND CALENDAR (NEW)
    # ========================================================================

    def is_near_ex_dividend(
        self,
        symbol: str,
        days_before: int = 3,
        days_after: int = 1,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if stock is near ex-dividend date.

        Important for Vietnam market:
        - Price typically drops by dividend amount on ex-date
        - Avoid buying right before ex-date (will lose dividend value)
        - Can be opportunity to buy after ex-date drop

        Args:
            symbol: Stock symbol
            days_before: Days before ex-date to flag
            days_after: Days after ex-date to flag

        Returns:
            (is_near, dividend_info)
        """
        dividend_info = self._get_dividend_calendar(symbol)

        if not dividend_info:
            return False, None

        now = datetime.now()
        ex_date = dividend_info.get("ex_date")

        if not ex_date:
            return False, None

        days_until = (ex_date - now).days

        # Before ex-date
        if 0 <= days_until <= days_before:
            return True, {
                **dividend_info,
                "position": "BEFORE_EX_DATE",
                "days_until": days_until,
                "warning": f"Ex-dividend in {days_until} days - price may drop after",
            }

        # After ex-date
        if -days_after <= days_until < 0:
            return True, {
                **dividend_info,
                "position": "AFTER_EX_DATE",
                "days_since": abs(days_until),
                "info": "Just went ex-dividend - may be buying opportunity",
            }

        return False, None

    def _get_dividend_calendar(self, symbol: str) -> Optional[Dict]:
        """
        Get dividend calendar for a stock.

        Sources to integrate:
        - TCBS dividend data
        - Company announcements
        - Exchange filings
        """
        try:
            from src.data.tcbs_provider import get_tcbs_provider

            provider = get_tcbs_provider()
            dividend_data = provider.get_dividend_info(symbol)

            if dividend_data:
                return dividend_data

        except ImportError:
            logger.debug("TCBS provider not available for dividend data")
        except Exception as e:
            logger.debug(f"Dividend data fetch failed for {symbol}: {e}")

        return None

    def get_dividend_risk_adjustment(
        self,
        symbol: str,
    ) -> Tuple[float, str]:
        """
        Get position size adjustment for dividend risk.

        Returns:
            (multiplier, reason)
        """
        is_near, info = self.is_near_ex_dividend(symbol)

        if not is_near:
            return 1.0, "No upcoming ex-dividend"

        if info:
            position = info.get("position")

            if position == "BEFORE_EX_DATE":
                days_until = info.get("days_until", 0)
                dividend_yield = info.get("dividend_yield", 0)

                if days_until <= 1:
                    return (
                        0.5,
                        f"Ex-dividend tomorrow - avoid buying (will lose ~{dividend_yield:.1f}%)",
                    )
                elif days_until <= 3:
                    return 0.7, f"Ex-dividend in {days_until} days - reduce position"

            elif position == "AFTER_EX_DATE":
                return 1.1, "Just went ex-dividend - potential buying opportunity"

        return 1.0, "No adjustment needed"


# Singleton instance
_fundamental_analyzer = None


def get_fundamental_analyzer() -> FundamentalAnalyzer:
    """Get singleton instance"""
    global _fundamental_analyzer
    if _fundamental_analyzer is None:
        _fundamental_analyzer = FundamentalAnalyzer()
    return _fundamental_analyzer


# Convenience functions
def get_fundamental_score(symbol: str, sector: Optional[str] = None) -> FundamentalScore:
    """Get fundamental score for a stock"""
    return get_fundamental_analyzer().calculate_fundamental_score(symbol, sector)


def is_near_earnings(symbol: str) -> Tuple[bool, Optional[EarningsEvent]]:
    """Check if stock is near earnings"""
    return get_fundamental_analyzer().is_near_earnings(symbol)


# Test
if __name__ == "__main__":
    print("Testing Fundamental Analyzer...")

    analyzer = FundamentalAnalyzer()

    # Test with sample data
    test_metrics = FundamentalMetrics(
        symbol="VNM",
        pe_ratio=18.5,
        pb_ratio=3.2,
        roe=25.0,
        roa=12.0,
        debt_to_equity=0.4,
        current_ratio=1.8,
        revenue_growth=8.0,
        earnings_growth=12.0,
        last_updated=datetime.now(),
    )

    # Cache the test metrics
    analyzer._metrics_cache["VNM"] = test_metrics

    # Calculate score
    score = analyzer.calculate_fundamental_score("VNM", sector="Thực phẩm")

    print(f"\nFundamental Score for VNM:")
    print(f"  Total Score: {score.total_score:.1f}")
    print(f"  Valuation: {score.valuation_score:.1f}")
    print(f"  Profitability: {score.profitability_score:.1f}")
    print(f"  Financial Health: {score.financial_health_score:.1f}")
    print(f"  Growth: {score.growth_score:.1f}")
    print(f"  Recommendation: {score.recommendation}")

    if score.warnings:
        print(f"\nWarnings:")
        for w in score.warnings:
            print(f"  {w}")

    print("\n✅ Fundamental Analyzer test completed!")
