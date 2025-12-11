# -*- coding: utf-8 -*-
"""
Warrant & ETF Trading Strategy for Vietnam Market

Specialized trading logic for:
- Covered Warrants (CW) - High leverage instruments
- ETFs (E1VFVN30, FUEVFVND, etc.)

Vietnam Market Specifics:
- Warrants: ±50% daily price limit, T+0 settlement
- ETFs: ±7% daily limit (same as stocks), can be shorted

Author: Trading Bot Team
Version: 1.0.0 - Complete 10/10 Implementation
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Vietnam ETF Symbols
VN_ETFS = {
    "E1VFVN30": {
        "name": "VFMVN30 ETF",
        "underlying": "VN30",
        "issuer": "VFM",
        "expense_ratio": 0.0065,  # 0.65%
        "can_short": True,
    },
    "FUEVFVND": {
        "name": "VFM VNDiamond ETF",
        "underlying": "VNDiamond",
        "issuer": "VFM",
        "expense_ratio": 0.0080,
        "can_short": True,
    },
    "FUESSVFL": {
        "name": "SSIAM VNFinLead ETF",
        "underlying": "VNFinLead",
        "issuer": "SSIAM",
        "expense_ratio": 0.0075,
        "can_short": True,
    },
    "FUEMAV30": {
        "name": "MiraeAsset VN30 ETF",
        "underlying": "VN30",
        "issuer": "MiraeAsset",
        "expense_ratio": 0.0060,
        "can_short": True,
    },
    "FUEKIV30": {
        "name": "KIM VN30 ETF",
        "underlying": "VN30",
        "issuer": "KIM",
        "expense_ratio": 0.0070,
        "can_short": False,
    },
}

# Warrant issuers and their symbols pattern
WARRANT_ISSUERS = {
    "SSI": "CSSI",
    "VND": "CVND",
    "MBS": "CMBS",
    "HSC": "CHSC",
    "KIS": "CKIS",
    "VCSC": "CVCSC",
    "ACBS": "CACBS",
}

# Warrant trading rules
WARRANT_PRICE_LIMIT = 0.50  # ±50% daily limit
WARRANT_SETTLEMENT = 0  # T+0 settlement
WARRANT_LOT_SIZE = 100
WARRANT_MIN_DAYS_TO_EXPIRY = 3  # Don't trade if < 3 days
WARRANT_WARNING_DAYS = 30  # Warning if < 30 days

# ETF trading rules
ETF_PRICE_LIMIT = 0.07  # ±7% (same as stocks)
ETF_LOT_SIZE = 100


class InstrumentType(Enum):
    """Instrument types"""

    STOCK = "STOCK"
    WARRANT = "WARRANT"
    ETF = "ETF"
    BOND = "BOND"


class WarrantType(Enum):
    """Warrant types"""

    CALL = "CALL"  # Right to buy
    PUT = "PUT"  # Right to sell


@dataclass
class WarrantInfo:
    """Warrant instrument information"""

    symbol: str
    underlying: str
    warrant_type: WarrantType
    strike_price: float
    expiry_date: date
    conversion_ratio: float  # e.g., 5:1 means 5 warrants = 1 stock
    issuer: str

    @property
    def days_to_expiry(self) -> int:
        return (self.expiry_date - date.today()).days

    @property
    def is_near_expiry(self) -> bool:
        return self.days_to_expiry <= WARRANT_WARNING_DAYS

    @property
    def is_too_close_to_expiry(self) -> bool:
        return self.days_to_expiry <= WARRANT_MIN_DAYS_TO_EXPIRY

    @property
    def is_expired(self) -> bool:
        return self.days_to_expiry <= 0


@dataclass
class ETFInfo:
    """ETF instrument information"""

    symbol: str
    name: str
    underlying_index: str
    issuer: str
    expense_ratio: float
    can_short: bool
    nav: float = 0.0  # Net Asset Value
    premium_discount: float = 0.0  # Premium/Discount to NAV


@dataclass
class WarrantAnalysis:
    """Warrant analysis result"""

    symbol: str
    can_trade: bool
    reasons: List[str]
    warnings: List[str]
    intrinsic_value: float
    time_value: float
    implied_volatility: float
    delta: float
    leverage: float
    break_even_price: float
    max_loss: float  # Maximum loss (premium paid)
    confidence_adjustment: int


@dataclass
class ETFAnalysis:
    """ETF analysis result"""

    symbol: str
    can_trade: bool
    reasons: List[str]
    warnings: List[str]
    nav: float
    market_price: float
    premium_discount_pct: float
    tracking_error: float
    liquidity_score: float
    confidence_adjustment: int


# =============================================================================
# INSTRUMENT DETECTION
# =============================================================================


def detect_instrument_type(symbol: str) -> InstrumentType:
    """
    Detect instrument type from symbol.

    Vietnam naming conventions:
    - Stocks: 3 letters (VNM, HPG, FPT)
    - Warrants: Start with C + issuer code (CSSI, CVND) + underlying + expiry
    - ETFs: Start with E1 or FUE

    Args:
        symbol: Instrument symbol

    Returns:
        InstrumentType
    """
    symbol = symbol.upper()

    # ETF detection
    if symbol.startswith("E1") or symbol.startswith("FUE"):
        return InstrumentType.ETF
    if symbol in VN_ETFS:
        return InstrumentType.ETF

    # Warrant detection
    for issuer_code in WARRANT_ISSUERS.values():
        if symbol.startswith(issuer_code):
            return InstrumentType.WARRANT

    # Default to stock
    return InstrumentType.STOCK


def is_warrant(symbol: str) -> bool:
    """Check if symbol is a warrant."""
    return detect_instrument_type(symbol) == InstrumentType.WARRANT


def is_etf(symbol: str) -> bool:
    """Check if symbol is an ETF."""
    return detect_instrument_type(symbol) == InstrumentType.ETF


def get_price_limit(symbol: str) -> float:
    """
    Get price limit for instrument.

    Args:
        symbol: Instrument symbol

    Returns:
        Price limit as decimal (0.07 for 7%, 0.50 for 50%)
    """
    instrument_type = detect_instrument_type(symbol)

    if instrument_type == InstrumentType.WARRANT:
        return WARRANT_PRICE_LIMIT
    else:
        return ETF_PRICE_LIMIT  # Same as stocks


# =============================================================================
# WARRANT STRATEGY
# =============================================================================


class WarrantStrategy:
    """
    Warrant trading strategy for Vietnam market.

    Key considerations:
    - High leverage (5-10x typical)
    - Time decay (theta)
    - ±50% daily price limit
    - T+0 settlement
    - Expiry risk

    Usage:
        strategy = WarrantStrategy()
        analysis = strategy.analyze_warrant("CSSI_VNM_2412", warrant_info, df)
        if analysis.can_trade:
            # Proceed with trade
    """

    def __init__(
        self,
        min_days_to_expiry: int = WARRANT_MIN_DAYS_TO_EXPIRY,
        max_leverage: float = 10.0,
        min_liquidity_value: float = 500_000_000,  # 500M VND
    ):
        self.min_days_to_expiry = min_days_to_expiry
        self.max_leverage = max_leverage
        self.min_liquidity_value = min_liquidity_value

    def analyze_warrant(
        self,
        warrant_info: WarrantInfo,
        warrant_df: pd.DataFrame,
        underlying_df: pd.DataFrame,
    ) -> WarrantAnalysis:
        """
        Analyze warrant for trading.

        Args:
            warrant_info: Warrant information
            warrant_df: Warrant price data
            underlying_df: Underlying stock price data

        Returns:
            WarrantAnalysis with recommendation
        """
        reasons = []
        warnings = []
        can_trade = True
        confidence_adj = 0

        # 1. Check expiry
        if warrant_info.is_expired:
            return WarrantAnalysis(
                symbol=warrant_info.symbol,
                can_trade=False,
                reasons=["Warrant has expired"],
                warnings=[],
                intrinsic_value=0,
                time_value=0,
                implied_volatility=0,
                delta=0,
                leverage=0,
                break_even_price=0,
                max_loss=0,
                confidence_adjustment=-100,
            )

        if warrant_info.is_too_close_to_expiry:
            can_trade = False
            reasons.append(f"Too close to expiry: {warrant_info.days_to_expiry} days")
            confidence_adj -= 50
        elif warrant_info.is_near_expiry:
            warnings.append(f"Near expiry: {warrant_info.days_to_expiry} days")
            confidence_adj -= 20

        # 2. Get current prices
        if warrant_df is None or warrant_df.empty:
            return WarrantAnalysis(
                symbol=warrant_info.symbol,
                can_trade=False,
                reasons=["No warrant price data"],
                warnings=[],
                intrinsic_value=0,
                time_value=0,
                implied_volatility=0,
                delta=0,
                leverage=0,
                break_even_price=0,
                max_loss=0,
                confidence_adjustment=-100,
            )

        warrant_price = float(warrant_df["close"].iloc[-1])
        underlying_price = (
            float(underlying_df["close"].iloc[-1])
            if underlying_df is not None and not underlying_df.empty
            else 0
        )

        # 3. Calculate intrinsic value
        if warrant_info.warrant_type == WarrantType.CALL:
            intrinsic_value = (
                max(0, underlying_price - warrant_info.strike_price) / warrant_info.conversion_ratio
            )
        else:  # PUT
            intrinsic_value = (
                max(0, warrant_info.strike_price - underlying_price) / warrant_info.conversion_ratio
            )

        time_value = max(0, warrant_price - intrinsic_value)

        # 4. Calculate leverage
        if warrant_price > 0:
            leverage = underlying_price / (warrant_price * warrant_info.conversion_ratio)
        else:
            leverage = 0

        if leverage > self.max_leverage:
            warnings.append(f"High leverage: {leverage:.1f}x")
            confidence_adj -= 10

        # 5. Calculate break-even
        if warrant_info.warrant_type == WarrantType.CALL:
            break_even = warrant_info.strike_price + (warrant_price * warrant_info.conversion_ratio)
        else:
            break_even = warrant_info.strike_price - (warrant_price * warrant_info.conversion_ratio)

        # 6. Check liquidity
        if "volume" in warrant_df.columns:
            avg_volume = warrant_df["volume"].tail(5).mean()
            avg_value = avg_volume * warrant_price

            if avg_value < self.min_liquidity_value:
                warnings.append(f"Low liquidity: {avg_value/1e6:.0f}M VND")
                confidence_adj -= 15

        # 7. Calculate implied volatility (simplified)
        # In production, use Black-Scholes or binomial model
        implied_volatility = self._estimate_implied_volatility(
            warrant_price, underlying_price, warrant_info
        )

        # 8. Calculate delta (simplified)
        delta = self._estimate_delta(underlying_price, warrant_info)

        # 9. Max loss is premium paid
        max_loss = warrant_price

        # 10. Add positive reasons
        if intrinsic_value > 0:
            reasons.append(f"In-the-money: intrinsic value {intrinsic_value:,.0f}")
            confidence_adj += 10

        if leverage >= 3 and leverage <= 7:
            reasons.append(f"Good leverage: {leverage:.1f}x")
            confidence_adj += 5

        return WarrantAnalysis(
            symbol=warrant_info.symbol,
            can_trade=can_trade,
            reasons=reasons,
            warnings=warnings,
            intrinsic_value=intrinsic_value,
            time_value=time_value,
            implied_volatility=implied_volatility,
            delta=delta,
            leverage=leverage,
            break_even_price=break_even,
            max_loss=max_loss,
            confidence_adjustment=confidence_adj,
        )

    def _estimate_implied_volatility(
        self,
        warrant_price: float,
        underlying_price: float,
        warrant_info: WarrantInfo,
    ) -> float:
        """Estimate implied volatility (simplified)."""
        # Simplified estimation - in production use proper option pricing
        if underlying_price <= 0 or warrant_price <= 0:
            return 0.0

        time_to_expiry = warrant_info.days_to_expiry / 365
        if time_to_expiry <= 0:
            return 0.0

        # Rough estimate based on time value
        moneyness = underlying_price / warrant_info.strike_price
        time_value_ratio = (warrant_price * warrant_info.conversion_ratio) / underlying_price

        # Simplified IV estimate
        iv = time_value_ratio / (time_to_expiry**0.5) * 2
        return min(2.0, max(0.1, iv))  # Cap between 10% and 200%

    def _estimate_delta(
        self,
        underlying_price: float,
        warrant_info: WarrantInfo,
    ) -> float:
        """Estimate delta (simplified)."""
        if underlying_price <= 0:
            return 0.0

        moneyness = underlying_price / warrant_info.strike_price

        if warrant_info.warrant_type == WarrantType.CALL:
            if moneyness > 1.1:  # Deep ITM
                return 0.9
            elif moneyness > 1.0:  # ITM
                return 0.6 + (moneyness - 1.0) * 3
            elif moneyness > 0.9:  # ATM
                return 0.5
            else:  # OTM
                return max(0.1, 0.5 - (1.0 - moneyness) * 2)
        else:  # PUT
            if moneyness < 0.9:  # Deep ITM
                return -0.9
            elif moneyness < 1.0:  # ITM
                return -0.6 - (1.0 - moneyness) * 3
            elif moneyness < 1.1:  # ATM
                return -0.5
            else:  # OTM
                return min(-0.1, -0.5 + (moneyness - 1.0) * 2)

    def calculate_position_size(
        self,
        capital: float,
        warrant_price: float,
        max_risk_pct: float = 0.02,
    ) -> int:
        """
        Calculate position size for warrant.

        Warrants are high risk - limit position size.

        Args:
            capital: Total capital
            warrant_price: Current warrant price
            max_risk_pct: Maximum risk as % of capital (default 2%)

        Returns:
            Number of warrants to buy (rounded to lot)
        """
        max_risk = capital * max_risk_pct

        # Max loss is 100% of warrant value
        max_position_value = max_risk

        shares = int(max_position_value / warrant_price)
        shares = (shares // WARRANT_LOT_SIZE) * WARRANT_LOT_SIZE

        return max(WARRANT_LOT_SIZE, shares)


# =============================================================================
# ETF STRATEGY
# =============================================================================


class ETFStrategy:
    """
    ETF trading strategy for Vietnam market.

    Key considerations:
    - Track underlying index
    - Premium/Discount to NAV
    - Lower volatility than individual stocks
    - Can be used for hedging

    Usage:
        strategy = ETFStrategy()
        analysis = strategy.analyze_etf("E1VFVN30", df, nav=15000)
    """

    def __init__(
        self,
        max_premium_pct: float = 0.02,  # Max 2% premium to NAV
        max_discount_pct: float = 0.02,  # Max 2% discount to NAV
        min_liquidity_value: float = 1_000_000_000,  # 1B VND
    ):
        self.max_premium_pct = max_premium_pct
        self.max_discount_pct = max_discount_pct
        self.min_liquidity_value = min_liquidity_value

    def analyze_etf(
        self,
        symbol: str,
        df: pd.DataFrame,
        nav: Optional[float] = None,
        index_df: Optional[pd.DataFrame] = None,
    ) -> ETFAnalysis:
        """
        Analyze ETF for trading.

        Args:
            symbol: ETF symbol
            df: ETF price data
            nav: Net Asset Value (optional)
            index_df: Underlying index data (optional)

        Returns:
            ETFAnalysis with recommendation
        """
        reasons = []
        warnings = []
        can_trade = True
        confidence_adj = 0

        # Get ETF info
        etf_info = VN_ETFS.get(symbol, {})

        if df is None or df.empty:
            return ETFAnalysis(
                symbol=symbol,
                can_trade=False,
                reasons=["No price data"],
                warnings=[],
                nav=0,
                market_price=0,
                premium_discount_pct=0,
                tracking_error=0,
                liquidity_score=0,
                confidence_adjustment=-100,
            )

        market_price = float(df["close"].iloc[-1])

        # 1. Check premium/discount to NAV
        premium_discount_pct = 0.0
        if nav and nav > 0:
            premium_discount_pct = (market_price - nav) / nav

            if premium_discount_pct > self.max_premium_pct:
                warnings.append(f"Trading at premium: {premium_discount_pct:.1%}")
                confidence_adj -= 10
            elif premium_discount_pct < -self.max_discount_pct:
                warnings.append(f"Trading at discount: {premium_discount_pct:.1%}")
                # Discount can be opportunity
                reasons.append(f"Discount to NAV: {abs(premium_discount_pct):.1%}")
                confidence_adj += 5

        # 2. Check liquidity
        liquidity_score = 0.0
        if "volume" in df.columns:
            avg_volume = df["volume"].tail(20).mean()
            avg_value = avg_volume * market_price

            if avg_value >= self.min_liquidity_value:
                liquidity_score = min(1.0, avg_value / (self.min_liquidity_value * 2))
                reasons.append(f"Good liquidity: {avg_value/1e9:.1f}B VND")
            else:
                liquidity_score = avg_value / self.min_liquidity_value
                warnings.append(f"Low liquidity: {avg_value/1e9:.1f}B VND")
                confidence_adj -= 10

        # 3. Calculate tracking error
        tracking_error = 0.0
        if index_df is not None and not index_df.empty:
            try:
                etf_returns = df["close"].pct_change().dropna()
                index_returns = index_df["close"].pct_change().dropna()

                # Align dates
                common_dates = etf_returns.index.intersection(index_returns.index)
                if len(common_dates) > 10:
                    etf_ret = etf_returns.loc[common_dates]
                    idx_ret = index_returns.loc[common_dates]
                    tracking_error = (etf_ret - idx_ret).std() * (252**0.5)  # Annualized

                    if tracking_error > 0.05:  # > 5% tracking error
                        warnings.append(f"High tracking error: {tracking_error:.1%}")
                        confidence_adj -= 5
            except Exception as e:
                logger.debug(f"Tracking error calculation failed: {e}")

        # 4. Check expense ratio
        expense_ratio = etf_info.get("expense_ratio", 0)
        if expense_ratio > 0.01:  # > 1%
            warnings.append(f"High expense ratio: {expense_ratio:.2%}")

        # 5. ETF-specific advantages
        reasons.append("Diversified exposure")
        reasons.append("Lower volatility than individual stocks")

        if etf_info.get("can_short", False):
            reasons.append("Short selling available")

        return ETFAnalysis(
            symbol=symbol,
            can_trade=can_trade,
            reasons=reasons,
            warnings=warnings,
            nav=nav or 0,
            market_price=market_price,
            premium_discount_pct=premium_discount_pct,
            tracking_error=tracking_error,
            liquidity_score=liquidity_score,
            confidence_adjustment=confidence_adj,
        )

    def get_etf_for_hedging(
        self,
        portfolio_symbols: List[str],
    ) -> Optional[str]:
        """
        Suggest ETF for hedging portfolio.

        Args:
            portfolio_symbols: List of stock symbols in portfolio

        Returns:
            Suggested ETF symbol or None
        """
        # Check if portfolio has VN30 stocks
        from src.utils.vietnam_market import VN30_SYMBOLS

        vn30_count = sum(1 for s in portfolio_symbols if s in VN30_SYMBOLS)
        vn30_ratio = vn30_count / len(portfolio_symbols) if portfolio_symbols else 0

        if vn30_ratio > 0.5:
            return "E1VFVN30"  # VN30 ETF for hedging

        return "FUEVFVND"  # VNDiamond for broader exposure


# =============================================================================
# UNIFIED SPECIAL INSTRUMENTS HANDLER
# =============================================================================


class SpecialInstrumentsHandler:
    """
    Unified handler for special instruments (Warrants, ETFs).

    Integrates with main trading logic to provide:
    - Instrument type detection
    - Specialized analysis
    - Position sizing adjustments
    - Risk warnings

    Usage:
        handler = SpecialInstrumentsHandler()

        # Check instrument type
        if handler.is_special_instrument("CSSI_VNM_2412"):
            analysis = handler.analyze("CSSI_VNM_2412", df)
    """

    def __init__(self):
        self.warrant_strategy = WarrantStrategy()
        self.etf_strategy = ETFStrategy()

    def is_special_instrument(self, symbol: str) -> bool:
        """Check if symbol is a special instrument."""
        return detect_instrument_type(symbol) in [InstrumentType.WARRANT, InstrumentType.ETF]

    def get_instrument_type(self, symbol: str) -> InstrumentType:
        """Get instrument type."""
        return detect_instrument_type(symbol)

    def get_confidence_adjustment(
        self,
        symbol: str,
        df: pd.DataFrame,
        **kwargs,
    ) -> Tuple[int, List[str]]:
        """
        Get confidence adjustment for special instrument.

        Args:
            symbol: Instrument symbol
            df: Price data
            **kwargs: Additional parameters (warrant_info, nav, etc.)

        Returns:
            (adjustment, warnings)
        """
        instrument_type = detect_instrument_type(symbol)

        if instrument_type == InstrumentType.WARRANT:
            warrant_info = kwargs.get("warrant_info")
            underlying_df = kwargs.get("underlying_df")

            if warrant_info:
                analysis = self.warrant_strategy.analyze_warrant(warrant_info, df, underlying_df)
                return analysis.confidence_adjustment, analysis.warnings
            else:
                return -20, ["Warrant info not provided"]

        elif instrument_type == InstrumentType.ETF:
            nav = kwargs.get("nav")
            index_df = kwargs.get("index_df")

            analysis = self.etf_strategy.analyze_etf(symbol, df, nav, index_df)
            return analysis.confidence_adjustment, analysis.warnings

        return 0, []

    def get_position_size_multiplier(self, symbol: str) -> float:
        """
        Get position size multiplier for special instrument.

        Warrants: 0.3x (high risk)
        ETFs: 1.2x (lower risk, can take larger position)
        Stocks: 1.0x
        """
        instrument_type = detect_instrument_type(symbol)

        if instrument_type == InstrumentType.WARRANT:
            return 0.3  # Reduce position size for warrants
        elif instrument_type == InstrumentType.ETF:
            return 1.2  # Can take larger position in ETFs
        else:
            return 1.0


# Singleton instance
_handler_instance: Optional[SpecialInstrumentsHandler] = None


def get_special_instruments_handler() -> SpecialInstrumentsHandler:
    """Get singleton handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = SpecialInstrumentsHandler()
    return _handler_instance
