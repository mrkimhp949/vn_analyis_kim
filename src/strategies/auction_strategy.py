# -*- coding: utf-8 -*-
"""
ATO/ATC Auction Strategy for Vietnam Stock Market

Strategies for trading during auction sessions:
- ATO (At The Open): 9:00-9:15 - Opening auction
- ATC (At The Close): 14:30-14:45 - Closing auction

Auction sessions have unique characteristics:
- Single price matching (not continuous)
- Higher volatility
- Institutional participation
- Gap opportunities

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class AuctionSession(Enum):
    """Auction session types"""

    ATO = "ATO"  # At The Open (9:00-9:15)
    ATC = "ATC"  # At The Close (14:30-14:45)
    NONE = "NONE"  # Not in auction


class AuctionSignal(Enum):
    """Auction trading signals"""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class AuctionAnalysis:
    """Analysis result for auction session"""

    session: AuctionSession
    signal: AuctionSignal
    confidence: float  # 0-100
    expected_gap: float  # Expected gap % from previous close
    recommended_price: float
    order_type: str  # "ATO", "ATC", "LO"
    reasons: List[str]
    warnings: List[str]
    foreign_flow_impact: float  # -1 to +1
    overnight_news_impact: float  # -1 to +1


class AuctionStrategy:
    """
    Strategy for ATO/ATC auction sessions.

    ATO Strategy (Opening Auction):
    - Analyze overnight news and global markets
    - Check pre-market foreign flow indicators
    - Identify gap opportunities
    - Use ATO orders for strong conviction

    ATC Strategy (Closing Auction):
    - Analyze intraday momentum
    - Check institutional order flow
    - Position for overnight holding or exit
    - Use ATC orders for closing price execution

    Usage:
        strategy = AuctionStrategy()

        # Check if should use ATO
        analysis = strategy.analyze_ato("VNM", df, foreign_flow)
        if analysis.signal in [AuctionSignal.BUY, AuctionSignal.STRONG_BUY]:
            place_ato_order(...)
    """

    # Session times
    ATO_START = time(9, 0)
    ATO_END = time(9, 15)
    ATC_START = time(14, 30)
    ATC_END = time(14, 45)

    # Thresholds
    STRONG_GAP_THRESHOLD = 0.03  # 3% gap = strong signal
    MODERATE_GAP_THRESHOLD = 0.015  # 1.5% gap = moderate signal
    HIGH_CONFIDENCE_THRESHOLD = 75
    MIN_CONFIDENCE_FOR_AUCTION = 60

    def __init__(
        self,
        use_foreign_flow: bool = True,
        use_global_markets: bool = True,
        use_news_sentiment: bool = True,
        min_confidence: float = 60,
    ):
        """
        Initialize Auction Strategy.

        Args:
            use_foreign_flow: Consider foreign investor flow
            use_global_markets: Consider overnight global market moves
            use_news_sentiment: Consider news sentiment
            min_confidence: Minimum confidence for auction orders
        """
        self.use_foreign_flow = use_foreign_flow
        self.use_global_markets = use_global_markets
        self.use_news_sentiment = use_news_sentiment
        self.min_confidence = min_confidence

    def get_current_session(self) -> AuctionSession:
        """Get current auction session"""
        now = datetime.now().time()

        if self.ATO_START <= now <= self.ATO_END:
            return AuctionSession.ATO
        elif self.ATC_START <= now <= self.ATC_END:
            return AuctionSession.ATC
        else:
            return AuctionSession.NONE

    def is_auction_time(self) -> bool:
        """Check if currently in auction session"""
        return self.get_current_session() != AuctionSession.NONE

    def time_to_next_auction(self) -> Tuple[AuctionSession, int]:
        """
        Get time to next auction session.

        Returns:
            (session_type, minutes_until)
        """
        now = datetime.now()
        current_time = now.time()

        # Check ATO
        ato_start = datetime.combine(now.date(), self.ATO_START)
        if current_time < self.ATO_START:
            minutes = int((ato_start - now).total_seconds() / 60)
            return AuctionSession.ATO, minutes

        # Check ATC
        atc_start = datetime.combine(now.date(), self.ATC_START)
        if current_time < self.ATC_START:
            minutes = int((atc_start - now).total_seconds() / 60)
            return AuctionSession.ATC, minutes

        # Next day ATO
        next_ato = datetime.combine(now.date() + timedelta(days=1), self.ATO_START)
        minutes = int((next_ato - now).total_seconds() / 60)
        return AuctionSession.ATO, minutes

    # =========================================================================
    # ATO (Opening Auction) Strategy
    # =========================================================================

    def analyze_ato(
        self,
        symbol: str,
        df: pd.DataFrame,
        foreign_flow: Optional[Dict] = None,
        global_markets: Optional[Dict] = None,
        news_sentiment: Optional[Dict] = None,
        existing_position: Optional[Dict] = None,
    ) -> AuctionAnalysis:
        """
        Analyze whether to use ATO order.

        ATO is recommended when:
        1. Strong overnight catalyst (news, earnings)
        2. Significant foreign flow signal
        3. Global markets strongly directional
        4. Gap expected in favorable direction

        Args:
            symbol: Stock symbol
            df: Historical price data
            foreign_flow: Foreign investor flow data
            global_markets: Overnight global market data
            news_sentiment: News sentiment analysis
            existing_position: Current position if any

        Returns:
            AuctionAnalysis with recommendation
        """
        reasons = []
        warnings = []
        confidence = 50.0  # Base confidence

        # Get previous close
        if df is None or df.empty:
            return self._default_analysis(AuctionSession.ATO, "Insufficient data")

        prev_close = float(df["close"].iloc[-1])

        # 1. Analyze Foreign Flow
        foreign_impact = 0.0
        if self.use_foreign_flow and foreign_flow:
            foreign_impact = self._analyze_foreign_flow_for_ato(foreign_flow)
            if foreign_impact > 0.3:
                confidence += 15
                reasons.append(f"Strong foreign buying signal ({foreign_impact:.2f})")
            elif foreign_impact < -0.3:
                confidence -= 15
                warnings.append(f"Foreign selling pressure ({foreign_impact:.2f})")

        # 2. Analyze Global Markets
        global_impact = 0.0
        if self.use_global_markets and global_markets:
            global_impact = self._analyze_global_markets(global_markets)
            if global_impact > 0.02:  # >2% up
                confidence += 10
                reasons.append(f"Global markets positive ({global_impact*100:.1f}%)")
            elif global_impact < -0.02:  # >2% down
                confidence -= 10
                warnings.append(f"Global markets negative ({global_impact*100:.1f}%)")

        # 3. Analyze News Sentiment
        news_impact = 0.0
        if self.use_news_sentiment and news_sentiment:
            news_impact = self._analyze_news_sentiment(news_sentiment, symbol)
            if news_impact > 0.5:
                confidence += 15
                reasons.append("Positive news catalyst")
            elif news_impact < -0.5:
                confidence -= 15
                warnings.append("Negative news impact")

        # 4. Estimate Expected Gap
        expected_gap = self._estimate_gap(foreign_impact, global_impact, news_impact, df)

        if abs(expected_gap) > self.STRONG_GAP_THRESHOLD:
            confidence += 10
            reasons.append(f"Strong gap expected ({expected_gap*100:+.1f}%)")

        # 5. Check Technical Setup
        tech_score = self._check_technical_setup(df)
        confidence += tech_score * 10

        # 6. Determine Signal
        signal = self._determine_signal(confidence, expected_gap, "BUY")

        # 7. Calculate Recommended Price
        if expected_gap > 0:
            # Gap up expected - bid slightly below expected open
            recommended_price = prev_close * (1 + expected_gap * 0.8)
        else:
            # Gap down expected - bid at support
            recommended_price = prev_close * (1 + expected_gap * 1.2)

        # Round to tick
        recommended_price = self._round_to_tick(recommended_price)

        # 8. Determine Order Type
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            order_type = "ATO"  # Use ATO order for high confidence
        else:
            order_type = "LO"  # Use limit order for lower confidence

        return AuctionAnalysis(
            session=AuctionSession.ATO,
            signal=signal,
            confidence=confidence,
            expected_gap=expected_gap,
            recommended_price=recommended_price,
            order_type=order_type,
            reasons=reasons,
            warnings=warnings,
            foreign_flow_impact=foreign_impact,
            overnight_news_impact=news_impact,
        )

    # =========================================================================
    # ATC (Closing Auction) Strategy
    # =========================================================================

    def analyze_atc(
        self,
        symbol: str,
        df: pd.DataFrame,
        intraday_data: Optional[pd.DataFrame] = None,
        existing_position: Optional[Dict] = None,
        foreign_flow: Optional[Dict] = None,
    ) -> AuctionAnalysis:
        """
        Analyze whether to use ATC order.

        ATC is recommended when:
        1. Want to close position at closing price
        2. Avoid overnight risk (Friday, before holidays)
        3. Strong intraday momentum to capture
        4. Institutional order flow detected

        Args:
            symbol: Stock symbol
            df: Historical price data
            intraday_data: Today's intraday data
            existing_position: Current position if any
            foreign_flow: Foreign investor flow data

        Returns:
            AuctionAnalysis with recommendation
        """
        reasons = []
        warnings = []
        confidence = 50.0

        if df is None or df.empty:
            return self._default_analysis(AuctionSession.ATC, "Insufficient data")

        current_price = float(df["close"].iloc[-1])

        # 1. Check if Friday (weekend risk)
        is_friday = datetime.now().weekday() == 4
        if is_friday:
            confidence += 10
            reasons.append("Friday - consider closing to avoid weekend risk")

        # 2. Analyze Intraday Momentum
        intraday_momentum = 0.0
        if intraday_data is not None and not intraday_data.empty:
            intraday_momentum = self._analyze_intraday_momentum(intraday_data)
            if intraday_momentum > 0.02:  # >2% up today
                confidence += 10
                reasons.append(f"Strong intraday momentum ({intraday_momentum*100:+.1f}%)")
            elif intraday_momentum < -0.02:
                confidence -= 10
                warnings.append(f"Weak intraday momentum ({intraday_momentum*100:+.1f}%)")

        # 3. Check Existing Position
        position_pnl = 0.0
        if existing_position:
            entry_price = existing_position.get("entry_price", current_price)
            position_pnl = (current_price - entry_price) / entry_price

            if position_pnl > 0.03:  # >3% profit
                confidence += 15
                reasons.append(
                    f"Profitable position ({position_pnl*100:+.1f}%) - consider taking profits"
                )
            elif position_pnl < -0.02:  # >2% loss
                warnings.append(f"Position in loss ({position_pnl*100:+.1f}%)")

        # 4. Analyze Foreign Flow (end of day)
        foreign_impact = 0.0
        if foreign_flow:
            foreign_impact = foreign_flow.get("score", 0)
            if foreign_impact < -0.3:
                confidence += 10
                reasons.append("Foreign selling at close - consider exiting")

        # 5. Check Volume Profile
        volume_surge = self._check_volume_surge(df)
        if volume_surge > 2.0:
            warnings.append(f"High volume ({volume_surge:.1f}x) - potential distribution")

        # 6. Determine Signal
        # For ATC, we're usually deciding whether to SELL/EXIT
        if existing_position:
            signal = self._determine_signal(confidence, -position_pnl, "SELL")
        else:
            signal = self._determine_signal(confidence, intraday_momentum, "BUY")

        # 7. Recommended Price (ATC uses closing price)
        recommended_price = current_price

        # 8. Order Type
        if confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            order_type = "ATC"
        else:
            order_type = "LO"

        return AuctionAnalysis(
            session=AuctionSession.ATC,
            signal=signal,
            confidence=confidence,
            expected_gap=0.0,  # No gap for ATC
            recommended_price=recommended_price,
            order_type=order_type,
            reasons=reasons,
            warnings=warnings,
            foreign_flow_impact=foreign_impact,
            overnight_news_impact=0.0,
        )

    # =========================================================================
    # Entry/Exit Recommendations
    # =========================================================================

    def should_use_ato_entry(
        self,
        symbol: str,
        signal: Dict,
        df: pd.DataFrame,
        foreign_flow: Optional[Dict] = None,
    ) -> Tuple[bool, str, float]:
        """
        Determine if should use ATO for entry.

        Returns:
            (should_use_ato, reason, recommended_price)
        """
        analysis = self.analyze_ato(symbol, df, foreign_flow)

        # Use ATO when:
        # 1. High confidence signal (>80%)
        # 2. Strong foreign buying
        # 3. Positive gap expected

        signal_confidence = signal.get("confidence", 0)

        if (
            analysis.confidence >= self.HIGH_CONFIDENCE_THRESHOLD
            and signal_confidence >= 75
            and analysis.expected_gap > 0.01
        ):
            return (
                True,
                f"ATO recommended: {', '.join(analysis.reasons[:2])}",
                analysis.recommended_price,
            )

        return (False, "Use limit order instead of ATO", analysis.recommended_price)

    def should_use_atc_exit(
        self,
        symbol: str,
        position: Dict,
        df: pd.DataFrame,
        foreign_flow: Optional[Dict] = None,
    ) -> Tuple[bool, str, float]:
        """
        Determine if should use ATC for exit.

        Returns:
            (should_use_atc, reason, recommended_price)
        """
        analysis = self.analyze_atc(
            symbol, df, existing_position=position, foreign_flow=foreign_flow
        )

        # Use ATC when:
        # 1. Friday afternoon (weekend risk)
        # 2. Position profitable and want closing price
        # 3. Foreign selling detected

        is_friday = datetime.now().weekday() == 4
        position_pnl = position.get("unrealized_pnl_pct", 0)

        if is_friday and position_pnl > 0.02:
            return (
                True,
                "ATC exit recommended: Friday with profit - avoid weekend risk",
                analysis.recommended_price,
            )

        if analysis.foreign_flow_impact < -0.5 and position_pnl > 0:
            return (
                True,
                "ATC exit recommended: Foreign selling with profit",
                analysis.recommended_price,
            )

        return (False, "No ATC exit needed", analysis.recommended_price)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _analyze_foreign_flow_for_ato(self, foreign_flow: Dict) -> float:
        """Analyze foreign flow for ATO decision"""
        score = foreign_flow.get("score", 0)
        consecutive_days = foreign_flow.get("consecutive_days", 0)

        # Boost score if consecutive buying/selling
        if consecutive_days >= 3:
            score *= 1.2

        return max(-1, min(1, score))

    def _analyze_global_markets(self, global_markets: Dict) -> float:
        """Analyze overnight global market moves"""
        # Weight different markets
        weights = {
            "US_SP500": 0.3,
            "US_NASDAQ": 0.2,
            "ASIA_NIKKEI": 0.15,
            "ASIA_HSI": 0.15,
            "ASIA_KOSPI": 0.1,
            "EUROPE_STOXX": 0.1,
        }

        total_impact = 0.0
        total_weight = 0.0

        for market, weight in weights.items():
            change = global_markets.get(market, 0)
            if change != 0:
                total_impact += change * weight
                total_weight += weight

        return total_impact / total_weight if total_weight > 0 else 0.0

    def _analyze_news_sentiment(self, news_sentiment: Dict, symbol: str) -> float:
        """Analyze news sentiment for symbol"""
        # Check symbol-specific news
        symbol_sentiment = news_sentiment.get(symbol, {})
        if symbol_sentiment:
            return symbol_sentiment.get("score", 0)

        # Fall back to market sentiment
        return news_sentiment.get("market_sentiment", 0)

    def _estimate_gap(
        self,
        foreign_impact: float,
        global_impact: float,
        news_impact: float,
        df: pd.DataFrame,
    ) -> float:
        """Estimate expected gap at open"""
        # Weighted combination
        gap = (
            foreign_impact * 0.02  # Foreign flow can move 2%
            + global_impact * 0.5  # Global markets correlation ~50%
            + news_impact * 0.03  # News can move 3%
        )

        # Cap at daily limit
        return max(-0.07, min(0.07, gap))

    def _check_technical_setup(self, df: pd.DataFrame) -> float:
        """Check technical setup score (-1 to +1)"""
        if df is None or len(df) < 20:
            return 0.0

        try:
            close = df["close"].iloc[-1]
            sma20 = df["close"].tail(20).mean()
            sma50 = df["close"].tail(50).mean() if len(df) >= 50 else sma20

            score = 0.0

            # Price above MAs
            if close > sma20:
                score += 0.3
            if close > sma50:
                score += 0.3

            # Trend
            if sma20 > sma50:
                score += 0.4

            return score

        except Exception:
            return 0.0

    def _analyze_intraday_momentum(self, intraday_data: pd.DataFrame) -> float:
        """Analyze intraday momentum"""
        if intraday_data is None or intraday_data.empty:
            return 0.0

        try:
            open_price = intraday_data["open"].iloc[0]
            current_price = intraday_data["close"].iloc[-1]
            return (current_price - open_price) / open_price
        except Exception:
            return 0.0

    def _check_volume_surge(self, df: pd.DataFrame) -> float:
        """Check if volume is surging"""
        if df is None or len(df) < 20:
            return 1.0

        try:
            current_vol = df["volume"].iloc[-1]
            avg_vol = df["volume"].tail(20).mean()
            return current_vol / avg_vol if avg_vol > 0 else 1.0
        except Exception:
            return 1.0

    def _determine_signal(
        self,
        confidence: float,
        directional_factor: float,
        default_side: str,
    ) -> AuctionSignal:
        """Determine auction signal based on confidence and direction"""
        if confidence < self.min_confidence:
            return AuctionSignal.HOLD

        if default_side == "BUY":
            if confidence >= 80 and directional_factor > 0.02:
                return AuctionSignal.STRONG_BUY
            elif confidence >= 65 and directional_factor > 0:
                return AuctionSignal.BUY
            elif confidence >= 65 and directional_factor < -0.02:
                return AuctionSignal.SELL
            else:
                return AuctionSignal.HOLD
        else:  # SELL
            if confidence >= 80 and directional_factor < -0.02:
                return AuctionSignal.STRONG_SELL
            elif confidence >= 65 and directional_factor < 0:
                return AuctionSignal.SELL
            elif confidence >= 65 and directional_factor > 0.02:
                return AuctionSignal.BUY
            else:
                return AuctionSignal.HOLD

    def _round_to_tick(self, price: float) -> float:
        """Round price to valid tick size"""
        try:
            from src.utils.vietnam_market import round_to_tick

            return round_to_tick(price)
        except ImportError:
            # Fallback
            if price < 10000:
                return round(price / 10) * 10
            elif price < 50000:
                return round(price / 50) * 50
            else:
                return round(price / 100) * 100

    def _default_analysis(self, session: AuctionSession, reason: str) -> AuctionAnalysis:
        """Return default HOLD analysis"""
        return AuctionAnalysis(
            session=session,
            signal=AuctionSignal.HOLD,
            confidence=0,
            expected_gap=0,
            recommended_price=0,
            order_type="LO",
            reasons=[],
            warnings=[reason],
            foreign_flow_impact=0,
            overnight_news_impact=0,
        )


# Singleton instance
_strategy_instance: Optional[AuctionStrategy] = None


def get_auction_strategy() -> AuctionStrategy:
    """Get singleton instance of auction strategy"""
    global _strategy_instance
    if _strategy_instance is None:
        _strategy_instance = AuctionStrategy()
    return _strategy_instance
