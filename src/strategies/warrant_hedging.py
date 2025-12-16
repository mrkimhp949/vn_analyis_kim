# -*- coding: utf-8 -*-
"""
Covered Warrant Hedging Strategy for Vietnam Market

This module provides hedging logic for covered warrants including:
1. Delta hedging for directional risk
2. Gamma scalping for volatility trading
3. Time decay (theta) management
4. Portfolio hedging using warrants
5. Risk metrics calculation (Greeks)

Vietnam Covered Warrant Specifics:
- Settlement: T+0 (can trade same day)
- Price limit: ±50% (vs ±7% for stocks)
- Conversion ratio: typically 2:1 to 10:1
- Issuers: SSI, VND, MBS, HSC, KIS, VCSC, ACBS
- European style (exercise only at expiry)

Hedging Strategies:
1. PROTECTIVE_PUT: Buy put warrants to hedge long stock positions
2. COVERED_CALL: Sell call exposure by shorting underlying at high prices
3. DELTA_NEUTRAL: Maintain delta-neutral position
4. COLLAR: Combine protective put with covered call

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from threading import RLock

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Risk-free rate (Vietnam government bond yield)
RISK_FREE_RATE = 0.045  # 4.5% annualized

# Trading days per year (Vietnam market)
TRADING_DAYS_PER_YEAR = 250

# Warrant Greeks thresholds
MIN_DELTA = 0.10  # Don't hedge if delta too low
MAX_DELTA = 0.90  # Don't hedge if delta too high (deep ITM)
GAMMA_SCALP_THRESHOLD = 0.05  # Gamma threshold for scalping
THETA_DECAY_WARNING = 0.02  # 2% daily decay warning

# Hedge ratios
DEFAULT_HEDGE_RATIO = 0.80  # Hedge 80% of delta by default
MIN_HEDGE_RATIO = 0.50
MAX_HEDGE_RATIO = 1.00

# Cost thresholds
MAX_HEDGE_COST_PCT = 0.03  # Max 3% of portfolio for hedging


class HedgeStrategy(Enum):
    """Hedging strategy types"""

    PROTECTIVE_PUT = "PROTECTIVE_PUT"
    COVERED_CALL = "COVERED_CALL"
    DELTA_NEUTRAL = "DELTA_NEUTRAL"
    COLLAR = "COLLAR"
    GAMMA_SCALP = "GAMMA_SCALP"
    NO_HEDGE = "NO_HEDGE"


class WarrantPosition(Enum):
    """Warrant position type"""

    LONG_CALL = "LONG_CALL"
    SHORT_CALL = "SHORT_CALL"
    LONG_PUT = "LONG_PUT"
    SHORT_PUT = "SHORT_PUT"


@dataclass
class WarrantGreeks:
    """Option Greeks for a warrant"""

    delta: float  # Price sensitivity to underlying
    gamma: float  # Delta sensitivity to underlying
    theta: float  # Time decay (daily)
    vega: float  # Sensitivity to volatility
    rho: float  # Sensitivity to interest rates

    # Additional metrics
    implied_volatility: float
    intrinsic_value: float
    time_value: float
    leverage: float
    break_even_price: float

    # Risk metrics
    max_loss: float
    max_profit: float
    probability_of_profit: float


@dataclass
class HedgeRecommendation:
    """Hedge recommendation for a position"""

    strategy: HedgeStrategy
    hedge_ratio: float

    # Actions
    actions: List[Dict[str, Any]]

    # Cost analysis
    upfront_cost: float
    ongoing_cost_per_day: float
    max_cost: float

    # Risk metrics
    unhedged_risk: float
    hedged_risk: float
    risk_reduction_pct: float

    # Timing
    rebalance_frequency_days: int
    expiry_warning: bool

    # Confidence
    confidence: int
    reasons: List[str]
    warnings: List[str]

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioHedge:
    """Portfolio-level hedge analysis"""

    portfolio_delta: float
    portfolio_gamma: float
    portfolio_theta: float
    portfolio_vega: float

    # Recommendations
    hedge_recommendations: List[HedgeRecommendation]

    # Summary
    total_hedge_cost: float
    total_risk_reduction: float
    optimal_hedges: List[str]  # Warrant symbols

    timestamp: datetime = field(default_factory=datetime.now)


class WarrantPricer:
    """
    Black-Scholes based warrant pricer for Vietnam market.

    Note: Vietnam covered warrants are European-style,
    so Black-Scholes is appropriate.
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 for Black-Scholes."""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 for Black-Scholes."""
        if T <= 0 or sigma <= 0:
            return 0
        return WarrantPricer.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / np.sqrt(2)))

    @staticmethod
    def norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    @classmethod
    def call_price(
        cls,
        S: float,  # Underlying price
        K: float,  # Strike price
        T: float,  # Time to expiry (years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
    ) -> float:
        """Calculate call warrant price."""
        if T <= 0:
            return max(0, S - K)

        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)

        return S * cls.norm_cdf(d1) - K * np.exp(-r * T) * cls.norm_cdf(d2)

    @classmethod
    def put_price(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Calculate put warrant price."""
        if T <= 0:
            return max(0, K - S)

        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)

        return K * np.exp(-r * T) * cls.norm_cdf(-d2) - S * cls.norm_cdf(-d1)

    @classmethod
    def calculate_greeks(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        is_call: bool = True,
        warrant_price: float = 0,
        conversion_ratio: float = 1,
    ) -> WarrantGreeks:
        """
        Calculate all Greeks for a warrant.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate
            sigma: Implied volatility
            is_call: True for call, False for put
            warrant_price: Current warrant price (for IV calculation)
            conversion_ratio: Warrant conversion ratio

        Returns:
            WarrantGreeks with all metrics
        """
        if T <= 0:
            T = 1 / TRADING_DAYS_PER_YEAR  # Minimum 1 day

        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        sqrt_T = np.sqrt(T)

        # Delta
        if is_call:
            delta = cls.norm_cdf(d1)
        else:
            delta = cls.norm_cdf(d1) - 1

        # Gamma (same for call and put)
        gamma = cls.norm_pdf(d1) / (S * sigma * sqrt_T) if S > 0 else 0

        # Theta (daily)
        theta_annual = -(S * sigma * cls.norm_pdf(d1)) / (2 * sqrt_T)
        if is_call:
            theta_annual -= r * K * np.exp(-r * T) * cls.norm_cdf(d2)
        else:
            theta_annual += r * K * np.exp(-r * T) * cls.norm_cdf(-d2)
        theta_daily = theta_annual / TRADING_DAYS_PER_YEAR

        # Vega
        vega = S * sqrt_T * cls.norm_pdf(d1) / 100  # Per 1% vol change

        # Rho
        if is_call:
            rho = K * T * np.exp(-r * T) * cls.norm_cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * cls.norm_cdf(-d2) / 100

        # Intrinsic and time value
        if is_call:
            intrinsic = max(0, S - K)
            theoretical = cls.call_price(S, K, T, r, sigma)
        else:
            intrinsic = max(0, K - S)
            theoretical = cls.put_price(S, K, T, r, sigma)

        time_value = max(0, theoretical - intrinsic)

        # Leverage
        warrant_value = warrant_price if warrant_price > 0 else theoretical
        leverage = (S * abs(delta)) / warrant_value if warrant_value > 0 else 0

        # Break-even
        if is_call:
            break_even = K + (warrant_value * conversion_ratio)
        else:
            break_even = K - (warrant_value * conversion_ratio)

        # Max loss/profit
        max_loss = warrant_value  # Premium paid
        if is_call:
            max_profit = float("inf")  # Unlimited upside
            prob_profit = cls.norm_cdf(d2)
        else:
            max_profit = K - warrant_value  # Max is strike - premium
            prob_profit = cls.norm_cdf(-d2)

        return WarrantGreeks(
            delta=delta / conversion_ratio,  # Adjust for conversion
            gamma=gamma / conversion_ratio,
            theta=theta_daily / conversion_ratio,
            vega=vega / conversion_ratio,
            rho=rho / conversion_ratio,
            implied_volatility=sigma,
            intrinsic_value=intrinsic / conversion_ratio,
            time_value=time_value / conversion_ratio,
            leverage=leverage,
            break_even_price=break_even,
            max_loss=max_loss,
            max_profit=max_profit if max_profit != float("inf") else 0,
            probability_of_profit=prob_profit,
        )

    @classmethod
    def implied_volatility(
        cls,
        warrant_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        is_call: bool = True,
        initial_guess: float = 0.30,
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson.

        Args:
            warrant_price: Market price of warrant
            S: Underlying price
            K: Strike price
            T: Time to expiry
            r: Risk-free rate
            is_call: True for call warrant
            initial_guess: Initial IV guess

        Returns:
            Implied volatility
        """
        sigma = initial_guess
        MAX_ITER = 100
        TOLERANCE = 0.0001

        for _ in range(MAX_ITER):
            if is_call:
                price = cls.call_price(S, K, T, r, sigma)
            else:
                price = cls.put_price(S, K, T, r, sigma)

            diff = price - warrant_price

            if abs(diff) < TOLERANCE:
                return sigma

            # Vega for Newton-Raphson
            d1 = cls.d1(S, K, T, r, sigma)
            vega = S * np.sqrt(T) * cls.norm_pdf(d1)

            if vega < TOLERANCE:
                break

            sigma = sigma - diff / vega
            sigma = max(0.01, min(3.0, sigma))  # Bound IV

        return sigma


class CoveredWarrantHedger:
    """
    Hedging strategy manager for covered warrants.

    Usage:
        hedger = CoveredWarrantHedger()

        # Calculate Greeks for a warrant
        greeks = hedger.calculate_warrant_greeks(warrant_info, underlying_price)

        # Get hedge recommendation
        recommendation = hedger.get_hedge_recommendation(
            stock_position, available_warrants
        )

        # Calculate portfolio hedge
        portfolio_hedge = hedger.calculate_portfolio_hedge(positions)
    """

    def __init__(
        self,
        risk_free_rate: float = RISK_FREE_RATE,
        default_hedge_ratio: float = DEFAULT_HEDGE_RATIO,
        max_hedge_cost_pct: float = MAX_HEDGE_COST_PCT,
    ):
        self.risk_free_rate = risk_free_rate
        self.default_hedge_ratio = default_hedge_ratio
        self.max_hedge_cost_pct = max_hedge_cost_pct

        self._pricer = WarrantPricer()
        self._cache: Dict[str, Any] = {}
        self._lock = RLock()

        logger.info("CoveredWarrantHedger initialized")

    def calculate_warrant_greeks(
        self,
        warrant_symbol: str,
        underlying_price: float,
        warrant_price: float,
        strike_price: float,
        days_to_expiry: int,
        is_call: bool = True,
        conversion_ratio: float = 1,
        historical_volatility: float = 0.30,
    ) -> WarrantGreeks:
        """
        Calculate Greeks for a specific warrant.

        Args:
            warrant_symbol: Warrant symbol
            underlying_price: Current price of underlying
            warrant_price: Current warrant price
            strike_price: Strike price
            days_to_expiry: Days until expiry
            is_call: True for call warrant
            conversion_ratio: Conversion ratio
            historical_volatility: Historical vol (fallback)

        Returns:
            WarrantGreeks
        """
        T = days_to_expiry / TRADING_DAYS_PER_YEAR

        # Calculate IV from market price
        try:
            iv = self._pricer.implied_volatility(
                warrant_price=warrant_price * conversion_ratio,  # Adjust for ratio
                S=underlying_price,
                K=strike_price,
                T=T,
                r=self.risk_free_rate,
                is_call=is_call,
                initial_guess=historical_volatility,
            )
        except Exception:
            iv = historical_volatility

        return self._pricer.calculate_greeks(
            S=underlying_price,
            K=strike_price,
            T=T,
            r=self.risk_free_rate,
            sigma=iv,
            is_call=is_call,
            warrant_price=warrant_price,
            conversion_ratio=conversion_ratio,
        )

    def get_hedge_recommendation(
        self,
        stock_symbol: str,
        stock_position_value: float,
        stock_shares: int,
        current_stock_price: float,
        available_warrants: List[Dict[str, Any]],
        market_outlook: str = "NEUTRAL",
        max_hedge_cost: Optional[float] = None,
    ) -> HedgeRecommendation:
        """
        Get hedge recommendation for a stock position.

        Args:
            stock_symbol: Stock symbol being hedged
            stock_position_value: Total position value in VND
            stock_shares: Number of shares
            current_stock_price: Current stock price
            available_warrants: List of available warrants for this stock
            market_outlook: "BULLISH", "BEARISH", "NEUTRAL"
            max_hedge_cost: Maximum cost for hedging

        Returns:
            HedgeRecommendation
        """
        if max_hedge_cost is None:
            max_hedge_cost = stock_position_value * self.max_hedge_cost_pct

        actions = []
        reasons = []
        warnings = []

        # Filter relevant warrants
        put_warrants = [w for w in available_warrants if not w.get("is_call", True)]
        call_warrants = [w for w in available_warrants if w.get("is_call", True)]

        # Determine strategy based on outlook
        if market_outlook == "BEARISH":
            strategy = HedgeStrategy.PROTECTIVE_PUT
            target_warrants = put_warrants
            hedge_ratio = 0.90  # Hedge more in bearish
        elif market_outlook == "BULLISH":
            strategy = HedgeStrategy.COVERED_CALL
            target_warrants = call_warrants
            hedge_ratio = 0.50  # Light hedge in bullish
        else:
            strategy = HedgeStrategy.COLLAR
            hedge_ratio = self.default_hedge_ratio
            target_warrants = put_warrants + call_warrants

        if not target_warrants:
            return HedgeRecommendation(
                strategy=HedgeStrategy.NO_HEDGE,
                hedge_ratio=0,
                actions=[],
                upfront_cost=0,
                ongoing_cost_per_day=0,
                max_cost=0,
                unhedged_risk=stock_position_value * 0.07,  # ±7% daily risk
                hedged_risk=stock_position_value * 0.07,
                risk_reduction_pct=0,
                rebalance_frequency_days=0,
                expiry_warning=False,
                confidence=0,
                reasons=["No suitable warrants available for hedging"],
                warnings=["Position remains unhedged"],
            )

        # Find best warrant for hedging
        best_warrant = None
        best_score = -float("inf")

        for warrant in target_warrants:
            days_to_expiry = warrant.get("days_to_expiry", 0)

            # Skip if too close to expiry
            if days_to_expiry < 7:
                warnings.append(f"Warrant {warrant.get('symbol')} too close to expiry")
                continue

            # Calculate Greeks
            greeks = self.calculate_warrant_greeks(
                warrant_symbol=warrant.get("symbol", ""),
                underlying_price=current_stock_price,
                warrant_price=warrant.get("price", 0),
                strike_price=warrant.get("strike", current_stock_price),
                days_to_expiry=days_to_expiry,
                is_call=warrant.get("is_call", True),
                conversion_ratio=warrant.get("conversion_ratio", 1),
            )

            # Score warrant: prefer high delta, low theta, reasonable cost
            delta_score = abs(greeks.delta) * 40  # 0-40 points
            theta_penalty = abs(greeks.theta) * 1000  # Lower is better
            time_score = min(30, days_to_expiry / 2)  # More time = better
            cost_penalty = (warrant.get("price", 0) / current_stock_price) * 100

            score = delta_score + time_score - theta_penalty - cost_penalty

            if score > best_score:
                best_score = score
                best_warrant = warrant
                best_greeks = greeks

        if best_warrant is None:
            return HedgeRecommendation(
                strategy=HedgeStrategy.NO_HEDGE,
                hedge_ratio=0,
                actions=[],
                upfront_cost=0,
                ongoing_cost_per_day=0,
                max_cost=0,
                unhedged_risk=stock_position_value * 0.07,
                hedged_risk=stock_position_value * 0.07,
                risk_reduction_pct=0,
                rebalance_frequency_days=0,
                expiry_warning=False,
                confidence=0,
                reasons=["No suitable warrants found after screening"],
                warnings=warnings,
            )

        # Calculate hedge position
        warrants_needed = int(
            (stock_shares * hedge_ratio) / best_warrant.get("conversion_ratio", 1)
        )
        warrant_cost = warrants_needed * best_warrant.get("price", 0)

        # Check cost constraint
        if warrant_cost > max_hedge_cost:
            warrants_needed = int(max_hedge_cost / best_warrant.get("price", 1))
            warrant_cost = warrants_needed * best_warrant.get("price", 0)
            warnings.append(f"Hedge reduced to fit cost budget: {max_hedge_cost:,.0f} VND")

        # Calculate risk reduction
        unhedged_risk = stock_position_value * 0.07  # Daily VaR at 7%
        hedged_risk = unhedged_risk * (1 - hedge_ratio * abs(best_greeks.delta))
        risk_reduction = (unhedged_risk - hedged_risk) / unhedged_risk * 100

        # Build action
        actions.append(
            {
                "action": "BUY" if strategy != HedgeStrategy.COVERED_CALL else "SELL",
                "symbol": best_warrant.get("symbol"),
                "quantity": warrants_needed,
                "price": best_warrant.get("price"),
                "total_cost": warrant_cost,
                "delta": best_greeks.delta,
                "days_to_expiry": best_warrant.get("days_to_expiry"),
            }
        )

        reasons.append(f"Hedge {stock_symbol} with {best_warrant.get('symbol')}")
        reasons.append(f"Delta: {best_greeks.delta:.2f}, Leverage: {best_greeks.leverage:.1f}x")

        # Expiry warning
        expiry_warning = best_warrant.get("days_to_expiry", 0) < 30
        if expiry_warning:
            warnings.append(f"Warrant expires in {best_warrant.get('days_to_expiry')} days")

        # Confidence based on Greeks and time
        confidence = min(
            90,
            int(
                50
                + abs(best_greeks.delta) * 20
                + min(20, best_warrant.get("days_to_expiry", 0) / 3)
            ),
        )

        return HedgeRecommendation(
            strategy=strategy,
            hedge_ratio=hedge_ratio,
            actions=actions,
            upfront_cost=warrant_cost,
            ongoing_cost_per_day=abs(best_greeks.theta) * warrants_needed,
            max_cost=warrant_cost,  # Max loss is premium paid
            unhedged_risk=unhedged_risk,
            hedged_risk=hedged_risk,
            risk_reduction_pct=risk_reduction,
            rebalance_frequency_days=7 if abs(best_greeks.gamma) > GAMMA_SCALP_THRESHOLD else 14,
            expiry_warning=expiry_warning,
            confidence=confidence,
            reasons=reasons,
            warnings=warnings,
        )

    def calculate_delta_hedge_adjustment(
        self,
        current_delta: float,
        target_delta: float,
        underlying_price: float,
        warrant_delta: float,
        warrant_price: float,
        conversion_ratio: float = 1,
    ) -> Dict[str, Any]:
        """
        Calculate adjustment needed to maintain delta-neutral hedge.

        Args:
            current_delta: Current portfolio delta
            target_delta: Target delta (usually 0 for delta-neutral)
            underlying_price: Current underlying price
            warrant_delta: Delta of warrant being used
            warrant_price: Current warrant price
            conversion_ratio: Warrant conversion ratio

        Returns:
            Dict with adjustment recommendation
        """
        delta_gap = current_delta - target_delta

        if abs(delta_gap) < 0.05:  # Small enough gap
            return {
                "needs_adjustment": False,
                "action": "HOLD",
                "warrants_to_trade": 0,
                "cost": 0,
                "reason": "Delta within tolerance",
            }

        # Calculate warrants needed
        adjusted_warrant_delta = warrant_delta / conversion_ratio
        warrants_needed = abs(int(delta_gap / adjusted_warrant_delta))

        if delta_gap > 0:
            # Need to reduce delta (buy puts or sell calls)
            action = "BUY_PUT" if warrant_delta < 0 else "SELL_CALL"
        else:
            # Need to increase delta (buy calls or sell puts)
            action = "BUY_CALL" if warrant_delta > 0 else "SELL_PUT"

        return {
            "needs_adjustment": True,
            "action": action,
            "warrants_to_trade": warrants_needed,
            "cost": warrants_needed * warrant_price,
            "new_delta": current_delta - (warrants_needed * adjusted_warrant_delta),
            "reason": f"Delta gap: {delta_gap:.3f}",
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_hedger: Optional[CoveredWarrantHedger] = None
_hedger_lock = RLock()


def get_warrant_hedger() -> CoveredWarrantHedger:
    """Get singleton instance of CoveredWarrantHedger."""
    global _hedger

    with _hedger_lock:
        if _hedger is None:
            _hedger = CoveredWarrantHedger()
        return _hedger


def calculate_warrant_greeks(
    underlying_price: float,
    strike_price: float,
    days_to_expiry: int,
    warrant_price: float,
    is_call: bool = True,
    conversion_ratio: float = 1,
    volatility: float = 0.30,
) -> WarrantGreeks:
    """
    Convenience function to calculate warrant Greeks.

    Args:
        underlying_price: Current underlying stock price
        strike_price: Warrant strike price
        days_to_expiry: Days until expiry
        warrant_price: Current warrant market price
        is_call: True for call warrant
        conversion_ratio: Conversion ratio
        volatility: Historical or implied volatility

    Returns:
        WarrantGreeks
    """
    hedger = get_warrant_hedger()
    return hedger.calculate_warrant_greeks(
        warrant_symbol="",
        underlying_price=underlying_price,
        warrant_price=warrant_price,
        strike_price=strike_price,
        days_to_expiry=days_to_expiry,
        is_call=is_call,
        conversion_ratio=conversion_ratio,
        historical_volatility=volatility,
    )
