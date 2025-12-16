# -*- coding: utf-8 -*-
"""
Dividend Capture Strategy for Vietnam Market

Implements dividend capture trading strategy:
1. Track ex-dividend dates
2. Analyze dividend yield and payout history
3. Generate buy signals before ex-date
4. Manage holding period around dividend events
5. Account for T+2 settlement in Vietnam

Vietnam Dividend Characteristics:
- Most companies pay annual dividends (1-2 times/year)
- Ex-date typically 2-3 weeks after announcement
- Cash dividend: Direct payment
- Stock dividend: Bonus shares
- Settlement: T+2 (must own before ex-date to receive dividend)

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS & ENUMS
# =============================================================================


class DividendType(Enum):
    """Type of dividend"""

    CASH = "CASH"  # Cổ tức tiền mặt
    STOCK = "STOCK"  # Cổ tức cổ phiếu
    MIXED = "MIXED"  # Kết hợp tiền mặt và cổ phiếu


class DividendSignal(Enum):
    """Dividend capture signal"""

    STRONG_BUY = "STRONG_BUY"  # High yield, good history
    BUY = "BUY"  # Reasonable yield
    HOLD = "HOLD"  # Already own, hold for dividend
    AVOID = "AVOID"  # Low yield or risky
    SELL_AFTER = "SELL_AFTER"  # Sell after ex-date


class DividendQuality(Enum):
    """Dividend quality rating"""

    EXCELLENT = "EXCELLENT"  # Consistent payer, growing dividends
    GOOD = "GOOD"  # Regular payer
    AVERAGE = "AVERAGE"  # Irregular but pays
    POOR = "POOR"  # Unreliable


@dataclass
class DividendEvent:
    """Single dividend event"""

    symbol: str
    announcement_date: date
    ex_date: date  # Ngày giao dịch không hưởng quyền
    record_date: date  # Ngày đăng ký cuối cùng
    payment_date: date
    dividend_type: DividendType
    cash_amount: float  # VND per share
    stock_ratio: float  # e.g., 10:1 means 10% stock dividend
    yield_percent: float  # Dividend yield at announcement


@dataclass
class DividendHistory:
    """Historical dividend data for a stock"""

    symbol: str
    total_dividends_5y: float  # Total dividends last 5 years
    avg_yield_5y: float  # Average yield last 5 years
    payout_ratio: float  # Dividend / EPS
    consecutive_years: int  # Years of consecutive payments
    growth_rate: float  # Dividend growth rate
    quality: DividendQuality
    events: List[DividendEvent] = field(default_factory=list)


@dataclass
class DividendCaptureConfig:
    """Configuration for dividend capture strategy"""

    # Yield thresholds
    min_yield_for_capture: float = 0.03  # Minimum 3% yield to consider
    excellent_yield_threshold: float = 0.06  # 6%+ is excellent
    good_yield_threshold: float = 0.04  # 4%+ is good

    # Timing parameters
    buy_days_before_ex: int = 5  # Buy 5 trading days before ex-date
    min_hold_days_after_ex: int = 3  # Minimum hold after ex-date
    max_hold_days_after_ex: int = 10  # Maximum hold after ex-date

    # T+2 settlement buffer
    settlement_buffer_days: int = 2  # Account for T+2

    # Quality filters
    min_consecutive_years: int = 2  # Minimum consecutive dividend years
    min_payout_ratio: float = 0.20  # Minimum 20% payout ratio
    max_payout_ratio: float = 0.80  # Maximum 80% (sustainability)

    # Position sizing
    max_dividend_positions: int = 5  # Max concurrent dividend captures
    position_size_pct: float = 0.10  # 10% per dividend capture

    # Risk management
    max_price_drop_to_hold: float = 0.10  # Max 10% drop before exit
    stop_loss_pct: float = 0.05  # 5% stop loss

    # Price behavior expectations
    # Stocks typically drop by dividend amount on ex-date
    expected_ex_date_drop_pct: float = 0.95  # Expect 95% of dividend drop


@dataclass
class DividendCaptureRecommendation:
    """Recommendation for dividend capture"""

    symbol: str
    signal: DividendSignal
    confidence: float
    ex_date: date
    days_to_ex: int
    dividend_amount: float
    dividend_yield: float
    buy_price_target: float
    expected_total_return: float
    risk_reward_ratio: float
    quality: DividendQuality
    notes: List[str]


class DividendCaptureStrategy:
    """
    Dividend capture strategy for Vietnam market

    Strategy Logic:
    1. Identify upcoming ex-dividend dates
    2. Filter by yield and quality
    3. Buy before ex-date (accounting for T+2)
    4. Hold through ex-date to receive dividend
    5. Sell after ex-date when price recovers (or at stop loss)

    Key Considerations for VN Market:
    - T+2 settlement: Must buy at least 2 days before ex-date
    - Price typically drops by dividend amount on ex-date
    - Tax: 5% withholding on cash dividends
    - Stock dividends: Dilution but no immediate tax
    """

    def __init__(self, config: Optional[DividendCaptureConfig] = None):
        self.config = config or DividendCaptureConfig()

        # State tracking
        self.upcoming_dividends: Dict[str, DividendEvent] = {}
        self.dividend_history: Dict[str, DividendHistory] = {}
        self.active_captures: Dict[str, Dict] = {}  # Current capture positions
        self.capture_history: List[Dict] = []

        # Try to import earnings calendar for dividend data
        try:
            from src.data.earnings_calendar import get_dividend_calendar

            self.dividend_calendar = get_dividend_calendar()
            self._has_calendar = True
        except ImportError:
            self._has_calendar = False
            logger.warning("Dividend calendar not available, using manual data")

    def add_dividend_event(
        self,
        symbol: str,
        ex_date: date,
        dividend_amount: float,
        current_price: float,
        dividend_type: DividendType = DividendType.CASH,
        stock_ratio: float = 0,
        announcement_date: Optional[date] = None,
        payment_date: Optional[date] = None,
    ) -> DividendEvent:
        """
        Add a dividend event to track

        Args:
            symbol: Stock symbol
            ex_date: Ex-dividend date
            dividend_amount: Cash dividend per share
            current_price: Current stock price
            dividend_type: Type of dividend
            stock_ratio: Stock dividend ratio (for stock/mixed)
            announcement_date: Announcement date
            payment_date: Payment date
        """
        # Calculate record date (T+2 before ex-date in VN)
        record_date = ex_date - timedelta(days=1)

        # Default dates if not provided
        if announcement_date is None:
            announcement_date = date.today()
        if payment_date is None:
            payment_date = ex_date + timedelta(days=30)  # Typical 1 month after

        # Calculate yield
        yield_pct = dividend_amount / current_price if current_price > 0 else 0

        event = DividendEvent(
            symbol=symbol,
            announcement_date=announcement_date,
            ex_date=ex_date,
            record_date=record_date,
            payment_date=payment_date,
            dividend_type=dividend_type,
            cash_amount=dividend_amount,
            stock_ratio=stock_ratio,
            yield_percent=yield_pct,
        )

        self.upcoming_dividends[symbol] = event

        # Update history
        if symbol not in self.dividend_history:
            self.dividend_history[symbol] = DividendHistory(
                symbol=symbol,
                total_dividends_5y=dividend_amount,
                avg_yield_5y=yield_pct,
                payout_ratio=0.5,  # Default estimate
                consecutive_years=1,
                growth_rate=0,
                quality=DividendQuality.AVERAGE,
                events=[event],
            )
        else:
            self.dividend_history[symbol].events.append(event)

        return event

    def calculate_dividend_quality(
        self, symbol: str, historical_dividends: List[float], historical_eps: List[float]
    ) -> DividendQuality:
        """
        Calculate dividend quality rating

        Factors:
        - Consistency of payments
        - Dividend growth
        - Payout ratio sustainability
        """
        if len(historical_dividends) < 2:
            return DividendQuality.POOR

        # Check consistency
        non_zero_years = sum(1 for d in historical_dividends if d > 0)
        consistency = non_zero_years / len(historical_dividends)

        # Check growth
        if len(historical_dividends) >= 3:
            growth_rates = []
            for i in range(1, len(historical_dividends)):
                if historical_dividends[i - 1] > 0:
                    growth = (
                        historical_dividends[i] - historical_dividends[i - 1]
                    ) / historical_dividends[i - 1]
                    growth_rates.append(growth)
            avg_growth = np.mean(growth_rates) if growth_rates else 0
        else:
            avg_growth = 0

        # Calculate payout ratio
        if historical_eps and len(historical_eps) == len(historical_dividends):
            payout_ratios = []
            for d, e in zip(historical_dividends, historical_eps):
                if e > 0:
                    payout_ratios.append(d / e)
            avg_payout = np.mean(payout_ratios) if payout_ratios else 0.5
        else:
            avg_payout = 0.5

        # Score calculation
        score = 0

        # Consistency score (0-40)
        score += consistency * 40

        # Growth score (0-30)
        if avg_growth > 0.05:
            score += 30
        elif avg_growth > 0:
            score += 20
        elif avg_growth > -0.05:
            score += 10

        # Payout sustainability score (0-30)
        if 0.30 <= avg_payout <= 0.60:
            score += 30  # Ideal range
        elif 0.20 <= avg_payout <= 0.70:
            score += 20
        elif avg_payout < 0.80:
            score += 10

        # Convert to quality
        if score >= 80:
            return DividendQuality.EXCELLENT
        elif score >= 60:
            return DividendQuality.GOOD
        elif score >= 40:
            return DividendQuality.AVERAGE
        else:
            return DividendQuality.POOR

    def analyze_capture_opportunity(
        self, symbol: str, current_price: float, current_date: Optional[date] = None
    ) -> Optional[DividendCaptureRecommendation]:
        """
        Analyze dividend capture opportunity for a stock

        Args:
            symbol: Stock symbol
            current_price: Current stock price
            current_date: Current date (default: today)

        Returns:
            DividendCaptureRecommendation or None
        """
        if current_date is None:
            current_date = date.today()

        # Get dividend event
        event = self.upcoming_dividends.get(symbol)
        if not event:
            return None

        # Calculate days to ex-date
        days_to_ex = (event.ex_date - current_date).days

        # Check if too late (need T+2 buffer)
        if days_to_ex < self.config.settlement_buffer_days:
            return DividendCaptureRecommendation(
                symbol=symbol,
                signal=DividendSignal.AVOID,
                confidence=0,
                ex_date=event.ex_date,
                days_to_ex=days_to_ex,
                dividend_amount=event.cash_amount,
                dividend_yield=event.yield_percent,
                buy_price_target=0,
                expected_total_return=0,
                risk_reward_ratio=0,
                quality=DividendQuality.POOR,
                notes=["Too late: T+2 settlement requires earlier entry"],
            )

        # Check yield threshold
        if event.yield_percent < self.config.min_yield_for_capture:
            return DividendCaptureRecommendation(
                symbol=symbol,
                signal=DividendSignal.AVOID,
                confidence=0.3,
                ex_date=event.ex_date,
                days_to_ex=days_to_ex,
                dividend_amount=event.cash_amount,
                dividend_yield=event.yield_percent,
                buy_price_target=current_price,
                expected_total_return=event.yield_percent,
                risk_reward_ratio=0.5,
                quality=DividendQuality.POOR,
                notes=[
                    f"Yield {event.yield_percent:.1%} below minimum {self.config.min_yield_for_capture:.1%}"
                ],
            )

        # Get history and quality
        history = self.dividend_history.get(symbol)
        quality = history.quality if history else DividendQuality.AVERAGE

        # Calculate expected returns
        notes = []

        # Gross dividend return
        gross_yield = event.yield_percent

        # Tax on cash dividend (5% in VN)
        tax_rate = 0.05 if event.dividend_type == DividendType.CASH else 0
        net_dividend = event.cash_amount * (1 - tax_rate)
        net_yield = net_dividend / current_price

        # Expected price drop on ex-date (typically ~90-100% of dividend)
        expected_drop = event.cash_amount * self.config.expected_ex_date_drop_pct
        expected_drop_pct = expected_drop / current_price

        # Expected recovery (historical average ~50-80% recovery within 10 days)
        expected_recovery_pct = expected_drop_pct * 0.65  # Assume 65% recovery

        # Total expected return
        expected_total_return = net_yield - expected_drop_pct + expected_recovery_pct

        # Transaction costs (~1.5% round trip)
        transaction_cost = 0.015
        net_total_return = expected_total_return - transaction_cost

        # Risk (max drawdown)
        max_risk = self.config.stop_loss_pct + expected_drop_pct

        # Risk/Reward ratio
        rr_ratio = net_total_return / max_risk if max_risk > 0 else 0

        # Generate signal
        signal, confidence = self._generate_signal(
            yield_pct=event.yield_percent,
            days_to_ex=days_to_ex,
            quality=quality,
            rr_ratio=rr_ratio,
            net_return=net_total_return,
        )

        # Build notes
        notes.append(f"Gross yield: {gross_yield:.2%}")
        notes.append(f"Net yield after tax: {net_yield:.2%}")
        notes.append(f"Expected ex-date drop: {expected_drop_pct:.2%}")
        notes.append(f"Expected recovery: {expected_recovery_pct:.2%}")
        notes.append(f"Net return after costs: {net_total_return:.2%}")

        if days_to_ex <= self.config.buy_days_before_ex:
            notes.append("⚡ Entry window open")
        else:
            notes.append(
                f"Entry window opens in {days_to_ex - self.config.buy_days_before_ex} days"
            )

        # Calculate buy price target (slight discount for safety)
        buy_price_target = current_price * 0.99  # Target 1% below current

        return DividendCaptureRecommendation(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            ex_date=event.ex_date,
            days_to_ex=days_to_ex,
            dividend_amount=event.cash_amount,
            dividend_yield=event.yield_percent,
            buy_price_target=buy_price_target,
            expected_total_return=net_total_return,
            risk_reward_ratio=rr_ratio,
            quality=quality,
            notes=notes,
        )

    def _generate_signal(
        self,
        yield_pct: float,
        days_to_ex: int,
        quality: DividendQuality,
        rr_ratio: float,
        net_return: float,
    ) -> Tuple[DividendSignal, float]:
        """Generate capture signal with confidence"""
        score = 0

        # Yield score (0-30)
        if yield_pct >= self.config.excellent_yield_threshold:
            score += 30
        elif yield_pct >= self.config.good_yield_threshold:
            score += 20
        elif yield_pct >= self.config.min_yield_for_capture:
            score += 10

        # Timing score (0-25)
        optimal_days = self.config.buy_days_before_ex
        if days_to_ex == optimal_days:
            score += 25
        elif optimal_days - 2 <= days_to_ex <= optimal_days + 2:
            score += 20
        elif days_to_ex > self.config.settlement_buffer_days:
            score += 10

        # Quality score (0-25)
        quality_scores = {
            DividendQuality.EXCELLENT: 25,
            DividendQuality.GOOD: 18,
            DividendQuality.AVERAGE: 10,
            DividendQuality.POOR: 0,
        }
        score += quality_scores.get(quality, 10)

        # Risk/Reward score (0-20)
        if rr_ratio >= 2.0:
            score += 20
        elif rr_ratio >= 1.5:
            score += 15
        elif rr_ratio >= 1.0:
            score += 10
        elif rr_ratio >= 0.5:
            score += 5

        # Convert to signal
        if score >= 80:
            signal = DividendSignal.STRONG_BUY
        elif score >= 60:
            signal = DividendSignal.BUY
        elif score >= 40:
            signal = DividendSignal.HOLD
        else:
            signal = DividendSignal.AVOID

        # Confidence
        confidence = min(1.0, score / 100)

        return signal, confidence

    def get_upcoming_opportunities(
        self,
        min_yield: Optional[float] = None,
        max_days: int = 30,
        current_date: Optional[date] = None,
    ) -> List[DividendCaptureRecommendation]:
        """
        Get all upcoming dividend capture opportunities

        Args:
            min_yield: Minimum yield filter
            max_days: Maximum days ahead to look
            current_date: Current date

        Returns:
            List of recommendations sorted by attractiveness
        """
        if current_date is None:
            current_date = date.today()

        min_yield = min_yield or self.config.min_yield_for_capture

        opportunities = []

        for symbol, event in self.upcoming_dividends.items():
            days_to_ex = (event.ex_date - current_date).days

            # Filter by time window
            if days_to_ex < self.config.settlement_buffer_days or days_to_ex > max_days:
                continue

            # Filter by yield
            if event.yield_percent < min_yield:
                continue

            # Get current price (would integrate with data provider)
            # For now, estimate from yield
            estimated_price = (
                event.cash_amount / event.yield_percent if event.yield_percent > 0 else 100000
            )

            rec = self.analyze_capture_opportunity(symbol, estimated_price, current_date)
            if rec and rec.signal in [DividendSignal.STRONG_BUY, DividendSignal.BUY]:
                opportunities.append(rec)

        # Sort by expected return descending
        opportunities.sort(key=lambda x: x.expected_total_return, reverse=True)

        return opportunities

    def manage_active_capture(
        self, symbol: str, current_price: float, current_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Manage an active dividend capture position

        Returns exit signal if conditions met
        """
        if current_date is None:
            current_date = date.today()

        if symbol not in self.active_captures:
            return {"action": "NO_POSITION", "symbol": symbol}

        capture = self.active_captures[symbol]
        entry_price = capture["entry_price"]
        entry_date = capture["entry_date"]
        event = capture["event"]

        days_held = (current_date - entry_date).days
        pnl_pct = (current_price - entry_price) / entry_price

        # Check if past ex-date
        days_since_ex = (current_date - event.ex_date).days

        # Exit conditions
        exit_signal = None
        exit_reason = None

        # 1. Stop loss hit
        if pnl_pct < -self.config.stop_loss_pct:
            exit_signal = DividendSignal.SELL_AFTER
            exit_reason = f"Stop loss hit: {pnl_pct:.1%}"

        # 2. Max price drop exceeded
        elif pnl_pct < -self.config.max_price_drop_to_hold:
            exit_signal = DividendSignal.SELL_AFTER
            exit_reason = f"Max drawdown exceeded: {pnl_pct:.1%}"

        # 3. Held long enough after ex-date with profit
        elif days_since_ex >= self.config.min_hold_days_after_ex and pnl_pct > 0:
            exit_signal = DividendSignal.SELL_AFTER
            exit_reason = f"Profit target after ex-date: {pnl_pct:.1%}"

        # 4. Max hold period reached
        elif days_since_ex >= self.config.max_hold_days_after_ex:
            exit_signal = DividendSignal.SELL_AFTER
            exit_reason = f"Max holding period: {days_held} days"

        # 5. Price recovered significantly
        elif days_since_ex > 0 and pnl_pct > event.yield_percent * 0.5:
            exit_signal = DividendSignal.SELL_AFTER
            exit_reason = f"Good recovery: {pnl_pct:.1%}"

        result = {
            "symbol": symbol,
            "current_price": current_price,
            "entry_price": entry_price,
            "pnl_pct": pnl_pct,
            "days_held": days_held,
            "days_since_ex": days_since_ex,
            "dividend_amount": event.cash_amount,
            "dividend_received": days_since_ex >= 0,
        }

        if exit_signal:
            result["action"] = "EXIT"
            result["signal"] = exit_signal.value
            result["reason"] = exit_reason
        else:
            result["action"] = "HOLD"
            if days_since_ex < 0:
                result["status"] = f"Waiting for ex-date ({-days_since_ex} days)"
            else:
                result["status"] = f"Post ex-date, waiting for recovery"

        return result

    def record_capture_entry(
        self, symbol: str, entry_price: float, shares: int, entry_date: Optional[date] = None
    ) -> Dict:
        """Record entry into a dividend capture position"""
        if entry_date is None:
            entry_date = date.today()

        event = self.upcoming_dividends.get(symbol)
        if not event:
            return {"error": f"No dividend event found for {symbol}"}

        self.active_captures[symbol] = {
            "entry_date": entry_date,
            "entry_price": entry_price,
            "shares": shares,
            "event": event,
            "expected_dividend": event.cash_amount * shares,
        }

        return {
            "symbol": symbol,
            "entry_recorded": True,
            "ex_date": event.ex_date,
            "expected_dividend": event.cash_amount * shares,
        }

    def record_capture_exit(
        self, symbol: str, exit_price: float, exit_date: Optional[date] = None
    ) -> Dict:
        """Record exit from a dividend capture position"""
        if exit_date is None:
            exit_date = date.today()

        if symbol not in self.active_captures:
            return {"error": f"No active capture for {symbol}"}

        capture = self.active_captures[symbol]
        entry_price = capture["entry_price"]
        shares = capture["shares"]
        event = capture["event"]

        # Calculate returns
        price_pnl = (exit_price - entry_price) * shares
        price_pnl_pct = (exit_price - entry_price) / entry_price

        # Dividend received if exit is after ex-date
        dividend_received = exit_date >= event.ex_date
        dividend_amount = event.cash_amount * shares if dividend_received else 0
        dividend_after_tax = dividend_amount * 0.95  # 5% withholding

        total_pnl = price_pnl + dividend_after_tax
        total_return_pct = total_pnl / (entry_price * shares)

        result = {
            "symbol": symbol,
            "entry_date": capture["entry_date"],
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "price_pnl": price_pnl,
            "price_pnl_pct": price_pnl_pct,
            "dividend_received": dividend_received,
            "dividend_amount": dividend_after_tax,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "holding_days": (exit_date - capture["entry_date"]).days,
        }

        # Record to history
        self.capture_history.append(result)

        # Remove from active
        del self.active_captures[symbol]

        return result

    def get_performance_summary(self) -> Dict:
        """Get summary of dividend capture performance"""
        if not self.capture_history:
            return {"message": "No completed captures yet"}

        total_captures = len(self.capture_history)
        winning_captures = sum(1 for c in self.capture_history if c["total_pnl"] > 0)

        total_pnl = sum(c["total_pnl"] for c in self.capture_history)
        total_dividends = sum(c["dividend_amount"] for c in self.capture_history)
        total_price_pnl = sum(c["price_pnl"] for c in self.capture_history)

        returns = [c["total_return_pct"] for c in self.capture_history]

        return {
            "total_captures": total_captures,
            "winning_captures": winning_captures,
            "win_rate": winning_captures / total_captures if total_captures > 0 else 0,
            "total_pnl": total_pnl,
            "total_dividends_received": total_dividends,
            "total_price_pnl": total_price_pnl,
            "avg_return_pct": np.mean(returns) if returns else 0,
            "best_return_pct": max(returns) if returns else 0,
            "worst_return_pct": min(returns) if returns else 0,
            "avg_holding_days": np.mean([c["holding_days"] for c in self.capture_history]),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_dividend_strategy: Optional[DividendCaptureStrategy] = None


def get_dividend_capture_strategy(
    config: Optional[DividendCaptureConfig] = None,
) -> DividendCaptureStrategy:
    """Get singleton dividend capture strategy instance"""
    global _dividend_strategy
    if _dividend_strategy is None:
        _dividend_strategy = DividendCaptureStrategy(config)
    return _dividend_strategy


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Dividend Capture Strategy...\n")

    # Create strategy
    strategy = DividendCaptureStrategy()

    # Add sample dividend events
    from datetime import date, timedelta

    today = date.today()

    # VNM - strong dividend payer
    strategy.add_dividend_event(
        symbol="VNM",
        ex_date=today + timedelta(days=10),
        dividend_amount=1500,  # 1,500 VND per share
        current_price=75000,
        dividend_type=DividendType.CASH,
    )

    # FPT - tech dividend
    strategy.add_dividend_event(
        symbol="FPT",
        ex_date=today + timedelta(days=15),
        dividend_amount=2000,
        current_price=110000,
        dividend_type=DividendType.CASH,
    )

    # HPG - steel company
    strategy.add_dividend_event(
        symbol="HPG",
        ex_date=today + timedelta(days=20),
        dividend_amount=500,
        current_price=27000,
        dividend_type=DividendType.CASH,
    )

    # Analyze opportunities
    print("Dividend Capture Analysis:\n")

    for symbol in ["VNM", "FPT", "HPG"]:
        event = strategy.upcoming_dividends[symbol]
        price = event.cash_amount / event.yield_percent if event.yield_percent > 0 else 100000

        rec = strategy.analyze_capture_opportunity(symbol, price)
        if rec:
            print(f"{symbol}:")
            print(f"  Ex-date: {rec.ex_date} ({rec.days_to_ex} days)")
            print(f"  Yield: {rec.dividend_yield:.2%}")
            print(f"  Signal: {rec.signal.value} ({rec.confidence:.0%} confidence)")
            print(f"  Expected return: {rec.expected_total_return:.2%}")
            print(f"  R/R ratio: {rec.risk_reward_ratio:.2f}")
            print(f"  Notes: {rec.notes[0]}")
            print()

    # Simulate a capture
    print("Simulating VNM Capture:\n")

    # Entry
    entry = strategy.record_capture_entry("VNM", 74500, 1000)
    print(f"Entry: {entry}")

    # Check after ex-date
    strategy.active_captures["VNM"]["entry_date"] = today - timedelta(days=12)
    strategy.upcoming_dividends["VNM"].ex_date = today - timedelta(days=3)

    status = strategy.manage_active_capture("VNM", 73500)
    print(f"Status: {status}")

    # Exit
    exit_result = strategy.record_capture_exit("VNM", 74000)
    print(f"Exit: Total return = {exit_result['total_return_pct']:.2%}")

    # Performance summary
    print("\nPerformance Summary:")
    print(strategy.get_performance_summary())

    print("\n✅ Dividend Capture Strategy test completed!")
