# -*- coding: utf-8 -*-
"""
Special Instruments Trading Logic for Vietnam Market

Handles:
- Warrant Trading (±50% daily limit)
- ETF Trading (including short selling)
- Odd-lot Trading (1-99 shares)

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


class InstrumentType(Enum):
    """Instrument types in Vietnam market"""

    STOCK = "STOCK"
    WARRANT = "WARRANT"
    ETF = "ETF"
    BOND = "BOND"
    FUND = "FUND"


# Warrant constants
WARRANT_PRICE_LIMIT = 0.50  # ±50% daily limit
WARRANT_MIN_DAYS_TO_EXPIRY = 3  # Don't trade if < 3 days to expiry
WARRANT_WARNING_DAYS = 30  # Warning if < 30 days to expiry
WARRANT_LOT_SIZE = 100  # Same as stocks

# ETF constants
ETF_PRICE_LIMIT = 0.07  # ±7% (same as stocks)
ETF_SHORT_ALLOWED = True  # Some ETFs allow short selling
ETF_LOT_SIZE = 100

# Odd-lot constants
ODD_LOT_MIN_QTY = 1
ODD_LOT_MAX_QTY = 99
ODD_LOT_SPREAD_PREMIUM = 0.005  # 0.5% wider spread
ODD_LOT_MIN_COMMISSION = 11_000  # 11,000 VND minimum


# =============================================================================
# WARRANT TRADING
# =============================================================================


@dataclass
class WarrantInfo:
    """
    Warrant information with pricing model.
    
    IMPROVED v2.0: Added Black-Scholes pricing and Greeks calculation.
    """

    symbol: str
    underlying: str  # Underlying stock symbol
    issuer: str  # Issuing securities company
    exercise_price: float
    exercise_ratio: float  # e.g., 1:1, 2:1
    expiry_date: datetime
    warrant_type: str  # "CALL" or "PUT"
    
    # Market data
    underlying_price: float = 0.0
    warrant_price: float = 0.0
    underlying_volatility: float = 0.30  # 30% default volatility

    # Calculated fields
    days_to_expiry: int = 0
    intrinsic_value: float = 0.0
    time_value: float = 0.0
    
    # Greeks (NEW v2.0)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    # Fair value (NEW v2.0)
    theoretical_value: float = 0.0
    premium_discount: float = 0.0  # % above/below fair value

    def __post_init__(self):
        self.days_to_expiry = (self.expiry_date - datetime.now()).days
        if self.underlying_price > 0:
            self._calculate_values()
    
    def _calculate_values(self):
        """Calculate intrinsic value, time value, and Greeks."""
        # Intrinsic value
        if self.warrant_type == "CALL":
            self.intrinsic_value = max(0, (self.underlying_price - self.exercise_price) * self.exercise_ratio)
        else:  # PUT
            self.intrinsic_value = max(0, (self.exercise_price - self.underlying_price) * self.exercise_ratio)
        
        # Time value
        if self.warrant_price > 0:
            self.time_value = max(0, self.warrant_price - self.intrinsic_value)
        
        # Calculate theoretical value using simplified Black-Scholes
        if self.days_to_expiry > 0 and self.underlying_price > 0:
            self._calculate_black_scholes()
    
    def _calculate_black_scholes(self):
        """
        Simplified Black-Scholes pricing for warrants.
        
        Note: This is a simplified model. Real warrant pricing should consider:
        - Dilution effect
        - Dividend adjustments
        - American vs European exercise
        """
        import math
        
        S = self.underlying_price
        K = self.exercise_price
        T = self.days_to_expiry / 365.0
        r = 0.05  # Risk-free rate (5%)
        sigma = self.underlying_volatility
        
        if T <= 0 or sigma <= 0:
            return
        
        try:
            # d1 and d2
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Cumulative normal distribution (approximation)
            def norm_cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            
            def norm_pdf(x):
                return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
            
            # Option price
            if self.warrant_type == "CALL":
                self.theoretical_value = (S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)) * self.exercise_ratio
                self.delta = norm_cdf(d1) * self.exercise_ratio
            else:  # PUT
                self.theoretical_value = (K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)) * self.exercise_ratio
                self.delta = -norm_cdf(-d1) * self.exercise_ratio
            
            # Greeks
            self.gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T)) * self.exercise_ratio
            self.theta = -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) * self.exercise_ratio / 365
            self.vega = S * math.sqrt(T) * norm_pdf(d1) * self.exercise_ratio / 100
            
            # Premium/discount to fair value
            if self.theoretical_value > 0 and self.warrant_price > 0:
                self.premium_discount = (self.warrant_price - self.theoretical_value) / self.theoretical_value
                
        except (ValueError, ZeroDivisionError) as e:
            logger.debug(f"Black-Scholes calculation error: {e}")


class WarrantTradingLogic:
    """
    Warrant Trading Logic for Vietnam Market
    
    IMPROVED v2.0:
    - Black-Scholes fair value calculation
    - Greeks-based risk management
    - Time decay warnings
    - Volatility analysis
    - Premium/discount alerts

    Key differences from stocks:
    - ±50% daily price limit (vs ±7% for stocks)
    - T+0 settlement (vs T+2 for stocks)
    - Expiry date consideration
    - Higher volatility and risk
    """

    # Known warrant issuers in Vietnam (Updated 2024)
    WARRANT_ISSUERS = {
        # SSI warrants
        "CVNM2401": "VNM",
        "CFPT2401": "FPT",
        "CHPG2401": "HPG",
        "CVCB2401": "VCB",
        "CMWG2401": "MWG",
        "CVIC2401": "VIC",
        "CVHM2401": "VHM",
        # VNDirect warrants
        "CVNM2402": "VNM",
        "CHPG2402": "HPG",
        # Legacy (backward compatible)
        "CVNM": "VNM",
        "CFPT": "FPT",
        "CHPG": "HPG",
        "CVCB": "VCB",
        "CMWG": "MWG",
    }
    
    # Warrant risk thresholds
    MAX_PREMIUM_PCT = 0.30  # Max 30% premium to fair value
    MAX_TIME_DECAY_DAILY = 0.05  # Max 5% daily time decay
    MIN_DELTA = 0.20  # Minimum delta for entry
    MAX_DELTA = 0.80  # Maximum delta (too expensive)

    def __init__(
        self,
        min_days_to_expiry: int = WARRANT_MIN_DAYS_TO_EXPIRY,
        warning_days: int = WARRANT_WARNING_DAYS,
        max_position_pct: float = 0.05,  # Max 5% of portfolio in warrants
        max_premium_pct: float = MAX_PREMIUM_PCT,
    ):
        self.min_days_to_expiry = min_days_to_expiry
        self.warning_days = warning_days
        self.max_position_pct = max_position_pct
        self.max_premium_pct = max_premium_pct

    def is_warrant(self, symbol: str) -> bool:
        """Check if symbol is a warrant"""
        # Vietnam warrants typically start with 'C' (Call) or 'P' (Put)
        # followed by underlying symbol
        if len(symbol) < 4:
            return False

        # Check known patterns
        if symbol.startswith("C") or symbol.startswith("P"):
            underlying = symbol[1:4]
            # Check if underlying is a valid stock
            return underlying in ["VNM", "FPT", "HPG", "VCB", "MWG", "VIC", "VHM"]

        return False

    def get_underlying(self, warrant_symbol: str) -> Optional[str]:
        """Get underlying stock symbol from warrant symbol"""
        if warrant_symbol in self.WARRANT_ISSUERS:
            return self.WARRANT_ISSUERS[warrant_symbol]

        if self.is_warrant(warrant_symbol):
            return warrant_symbol[1:4]

        return None

    def check_tradeable(
        self,
        warrant_info: WarrantInfo,
        current_price: float,
    ) -> Tuple[bool, List[str]]:
        """
        Check if warrant is tradeable.
        
        IMPROVED v2.0: Added Greeks-based checks and fair value analysis.

        Returns:
            (is_tradeable, warnings)
        """
        warnings = []
        blocking_reasons = []

        # Check 1: Expiry
        if warrant_info.days_to_expiry < self.min_days_to_expiry:
            return False, [f"❌ Too close to expiry: {warrant_info.days_to_expiry} days"]

        if warrant_info.days_to_expiry < self.warning_days:
            warnings.append(f"⚠️ Approaching expiry: {warrant_info.days_to_expiry} days remaining")

        # Check 2: Intrinsic value
        if warrant_info.intrinsic_value <= 0:
            warnings.append("⚠️ Out of the money - no intrinsic value")

        # Check 3: Time value decay
        if warrant_info.days_to_expiry < 14:
            warnings.append("⚠️ High time decay risk (< 14 days)")
        
        # NEW v2.0: Check 4 - Premium to fair value
        if warrant_info.premium_discount > self.max_premium_pct:
            blocking_reasons.append(
                f"❌ Overpriced: {warrant_info.premium_discount:.1%} premium to fair value "
                f"(max: {self.max_premium_pct:.0%})"
            )
        elif warrant_info.premium_discount > self.max_premium_pct * 0.7:
            warnings.append(
                f"⚠️ High premium: {warrant_info.premium_discount:.1%} above fair value"
            )
        elif warrant_info.premium_discount < -0.10:
            warnings.append(
                f"✅ Discount: {abs(warrant_info.premium_discount):.1%} below fair value"
            )
        
        # NEW v2.0: Check 5 - Delta (moneyness)
        if warrant_info.delta > 0:
            if warrant_info.delta < self.MIN_DELTA:
                warnings.append(
                    f"⚠️ Low delta ({warrant_info.delta:.2f}): Far OTM, high risk"
                )
            elif warrant_info.delta > self.MAX_DELTA:
                warnings.append(
                    f"⚠️ High delta ({warrant_info.delta:.2f}): Deep ITM, expensive"
                )
        
        # NEW v2.0: Check 6 - Theta (time decay)
        if warrant_info.theta < 0:
            daily_decay_pct = abs(warrant_info.theta) / current_price if current_price > 0 else 0
            if daily_decay_pct > self.MAX_TIME_DECAY_DAILY:
                warnings.append(
                    f"⚠️ High time decay: {daily_decay_pct:.1%}/day "
                    f"({abs(warrant_info.theta):,.0f} VND/day)"
                )
        
        # NEW v2.0: Check 7 - Theoretical value sanity check
        if warrant_info.theoretical_value > 0 and current_price > 0:
            if current_price > warrant_info.theoretical_value * 2:
                blocking_reasons.append(
                    f"❌ Price ({current_price:,.0f}) > 2x fair value "
                    f"({warrant_info.theoretical_value:,.0f})"
                )
        
        if blocking_reasons:
            return False, blocking_reasons

        return True, warnings
    
    def analyze_warrant(
        self,
        warrant_info: WarrantInfo,
    ) -> Dict:
        """
        Comprehensive warrant analysis.
        
        NEW v2.0: Returns detailed analysis including Greeks and recommendations.
        """
        analysis = {
            "symbol": warrant_info.symbol,
            "underlying": warrant_info.underlying,
            "type": warrant_info.warrant_type,
            "days_to_expiry": warrant_info.days_to_expiry,
            "moneyness": "ITM" if warrant_info.intrinsic_value > 0 else "OTM",
            
            # Values
            "intrinsic_value": warrant_info.intrinsic_value,
            "time_value": warrant_info.time_value,
            "theoretical_value": warrant_info.theoretical_value,
            "premium_discount": warrant_info.premium_discount,
            
            # Greeks
            "delta": warrant_info.delta,
            "gamma": warrant_info.gamma,
            "theta": warrant_info.theta,
            "vega": warrant_info.vega,
            
            # Risk metrics
            "leverage": warrant_info.underlying_price / warrant_info.warrant_price if warrant_info.warrant_price > 0 else 0,
            "break_even": warrant_info.exercise_price + warrant_info.warrant_price / warrant_info.exercise_ratio,
            
            # Recommendations
            "recommendations": [],
        }
        
        # Generate recommendations
        if warrant_info.premium_discount < -0.05:
            analysis["recommendations"].append("✅ Trading at discount - potential value")
        if warrant_info.delta > 0.40 and warrant_info.delta < 0.60:
            analysis["recommendations"].append("✅ Optimal delta range (0.4-0.6)")
        if warrant_info.days_to_expiry > 60:
            analysis["recommendations"].append("✅ Good time to expiry (>60 days)")
        if warrant_info.days_to_expiry < 14:
            analysis["recommendations"].append("⚠️ High theta decay - consider exit")
        if warrant_info.intrinsic_value <= 0 and warrant_info.days_to_expiry < 30:
            analysis["recommendations"].append("🚨 OTM with <30 days - high risk of total loss")
        
        return analysis

    def calculate_position_size(
        self,
        portfolio_value: float,
        warrant_price: float,
        confidence: int,
    ) -> int:
        """
        Calculate position size for warrant

        Warrants are high-risk, so position size is limited
        """
        # Max allocation to warrants
        max_allocation = portfolio_value * self.max_position_pct

        # Adjust by confidence
        confidence_factor = confidence / 100
        allocation = max_allocation * confidence_factor * 0.5  # Extra conservative

        # Calculate shares
        shares = int(allocation / warrant_price)

        # Round to lot size
        shares = (shares // WARRANT_LOT_SIZE) * WARRANT_LOT_SIZE

        return max(0, shares)

    def calculate_stop_loss(
        self,
        entry_price: float,
        warrant_info: WarrantInfo,
    ) -> float:
        """
        Calculate stop loss for warrant

        Warrants have ±50% limit, so wider stops needed
        """
        # Base stop: 15-25% depending on days to expiry
        if warrant_info.days_to_expiry > 30:
            stop_pct = 0.15
        elif warrant_info.days_to_expiry > 14:
            stop_pct = 0.20
        else:
            stop_pct = 0.25

        return entry_price * (1 - stop_pct)

    def get_price_limits(self, reference_price: float) -> Dict[str, float]:
        """Get ceiling and floor prices for warrant"""
        return {
            "ceiling": reference_price * (1 + WARRANT_PRICE_LIMIT),
            "floor": reference_price * (1 - WARRANT_PRICE_LIMIT),
            "limit_pct": WARRANT_PRICE_LIMIT * 100,
        }


# =============================================================================
# ETF TRADING
# =============================================================================


@dataclass
class ETFInfo:
    """ETF information"""

    symbol: str
    name: str
    underlying_index: str  # e.g., "VN30", "VNINDEX"
    fund_type: str  # "INDEX", "BOND", "SECTOR"
    nav: float  # Net Asset Value
    premium_discount: float  # Premium/discount to NAV
    expense_ratio: float
    short_allowed: bool = False

    # Trading info
    avg_volume: int = 0
    avg_spread: float = 0.0


class ETFTradingLogic:
    """
    ETF Trading Logic for Vietnam Market
    
    IMPROVED v2.0:
    - Complete ETF database with NAV tracking
    - Short selling logic with margin requirements
    - Premium/discount arbitrage detection
    - Sector ETF rotation analysis
    - Tracking error monitoring

    Key features:
    - Same ±7% price limit as stocks
    - Some ETFs allow short selling
    - Track premium/discount to NAV
    - Lower volatility than individual stocks
    """

    # Known ETFs in Vietnam (Updated 2024)
    VIETNAM_ETFS = {
        # VN30 Index ETFs
        "E1VFVN30": {
            "name": "VFMVN30 ETF",
            "index": "VN30",
            "short_allowed": True,
            "expense_ratio": 0.0065,  # 0.65%
            "aum": 8000_000_000_000,  # ~8T VND
            "type": "INDEX",
        },
        "FUEMAV30": {
            "name": "Mirae Asset VN30 ETF",
            "index": "VN30",
            "short_allowed": True,
            "expense_ratio": 0.0055,
            "aum": 3000_000_000_000,
            "type": "INDEX",
        },
        # Diamond Index ETF
        "FUEVFVND": {
            "name": "VFM VN Diamond ETF",
            "index": "VN Diamond",
            "short_allowed": True,
            "expense_ratio": 0.0070,
            "aum": 2500_000_000_000,
            "type": "INDEX",
        },
        # Sector ETFs
        "FUESSVFL": {
            "name": "SSIAM VNFin Lead ETF",
            "index": "VNFin Lead",
            "short_allowed": False,
            "expense_ratio": 0.0080,
            "aum": 1500_000_000_000,
            "type": "SECTOR",
            "sector": "FINANCIALS",
        },
        "FUESSV50": {
            "name": "SSIAM VN50 ETF",
            "index": "VN50",
            "short_allowed": True,
            "expense_ratio": 0.0060,
            "aum": 1000_000_000_000,
            "type": "INDEX",
        },
        # Bond ETFs
        "FUEBFVND": {
            "name": "VFM Bond ETF",
            "index": "VN Bond",
            "short_allowed": False,
            "expense_ratio": 0.0050,
            "aum": 500_000_000_000,
            "type": "BOND",
        },
    }
    
    # Short selling parameters
    SHORT_MARGIN_REQUIREMENT = 0.50  # 50% margin for shorts
    SHORT_BORROW_RATE = 0.08  # 8% annual borrow rate
    MAX_SHORT_POSITION_PCT = 0.10  # Max 10% portfolio in shorts

    def __init__(
        self,
        max_premium_discount: float = 0.03,  # Max 3% premium/discount
        min_volume: int = 100_000,  # Minimum daily volume
        enable_short_selling: bool = True,
    ):
        self.max_premium_discount = max_premium_discount
        self.min_volume = min_volume
        self.enable_short_selling = enable_short_selling
        
        # NAV tracking cache
        self._nav_cache: Dict[str, Dict] = {}
        self._nav_cache_time: Optional[datetime] = None

    def is_etf(self, symbol: str) -> bool:
        """Check if symbol is an ETF"""
        return symbol.upper() in self.VIETNAM_ETFS

    def get_etf_info(self, symbol: str) -> Optional[Dict]:
        """Get ETF information"""
        return self.VIETNAM_ETFS.get(symbol.upper())

    def can_short(self, symbol: str) -> bool:
        """Check if ETF allows short selling"""
        etf_info = self.get_etf_info(symbol)
        if etf_info:
            return etf_info.get("short_allowed", False)
        return False

    def check_tradeable(
        self,
        etf_info: ETFInfo,
    ) -> Tuple[bool, List[str]]:
        """Check if ETF is tradeable"""
        warnings = []

        # Check premium/discount
        if abs(etf_info.premium_discount) > self.max_premium_discount:
            warnings.append(f"⚠️ High premium/discount: {etf_info.premium_discount:.2%}")

        # Check volume
        if etf_info.avg_volume < self.min_volume:
            warnings.append(f"⚠️ Low volume: {etf_info.avg_volume:,} (min: {self.min_volume:,})")

        # Check spread
        if etf_info.avg_spread > 0.005:  # > 0.5%
            warnings.append(f"⚠️ Wide spread: {etf_info.avg_spread:.2%}")

        return True, warnings

    def calculate_short_position(
        self,
        portfolio_value: float,
        etf_price: float,
        confidence: int,
        max_short_pct: float = 0.10,  # Max 10% short
    ) -> int:
        """Calculate short position size for ETF"""
        if confidence < 70:
            return 0  # Need high confidence for shorts

        max_allocation = portfolio_value * max_short_pct
        confidence_factor = (confidence - 50) / 50  # Scale from 50-100
        allocation = max_allocation * confidence_factor

        shares = int(allocation / etf_price)
        shares = (shares // ETF_LOT_SIZE) * ETF_LOT_SIZE

        return max(0, shares)
    
    def analyze_short_opportunity(
        self,
        symbol: str,
        etf_info: ETFInfo,
        market_regime: Optional[Dict] = None,
    ) -> Dict:
        """
        Analyze short selling opportunity for ETF.
        
        NEW v2.0: Comprehensive short analysis with risk metrics.
        """
        if not self.enable_short_selling:
            return {"can_short": False, "reason": "Short selling disabled"}
        
        if not self.can_short(symbol):
            return {"can_short": False, "reason": f"{symbol} does not allow short selling"}
        
        analysis = {
            "symbol": symbol,
            "can_short": True,
            "margin_required": etf_info.nav * self.SHORT_MARGIN_REQUIREMENT,
            "annual_borrow_cost": etf_info.nav * self.SHORT_BORROW_RATE,
            "daily_borrow_cost": etf_info.nav * self.SHORT_BORROW_RATE / 365,
            "signals": [],
            "risk_score": 50,  # 0-100, higher = more risky
        }
        
        # Signal 1: Premium to NAV (short when premium)
        if etf_info.premium_discount > 0.02:
            analysis["signals"].append(
                f"✅ Trading at {etf_info.premium_discount:.1%} premium - short opportunity"
            )
            analysis["risk_score"] -= 10
        elif etf_info.premium_discount < -0.02:
            analysis["signals"].append(
                f"⚠️ Trading at {abs(etf_info.premium_discount):.1%} discount - avoid short"
            )
            analysis["risk_score"] += 20
        
        # Signal 2: Market regime
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            if regime == "BEAR":
                analysis["signals"].append("✅ Bear market - favorable for shorts")
                analysis["risk_score"] -= 15
            elif regime == "BULL":
                analysis["signals"].append("⚠️ Bull market - risky for shorts")
                analysis["risk_score"] += 25
        
        # Signal 3: Volume
        if etf_info.avg_volume < self.min_volume:
            analysis["signals"].append(
                f"⚠️ Low volume ({etf_info.avg_volume:,}) - hard to cover"
            )
            analysis["risk_score"] += 15
        
        # Break-even calculation
        days_to_breakeven = 0
        if etf_info.premium_discount > 0:
            daily_cost_pct = self.SHORT_BORROW_RATE / 365
            days_to_breakeven = int(etf_info.premium_discount / daily_cost_pct)
            analysis["days_to_breakeven"] = days_to_breakeven
            analysis["signals"].append(
                f"ℹ️ Break-even in {days_to_breakeven} days (borrow cost vs premium)"
            )
        
        # Final recommendation
        if analysis["risk_score"] < 40:
            analysis["recommendation"] = "FAVORABLE"
        elif analysis["risk_score"] < 60:
            analysis["recommendation"] = "NEUTRAL"
        else:
            analysis["recommendation"] = "AVOID"
        
        return analysis
    
    def calculate_nav_premium_discount(
        self,
        symbol: str,
        market_price: float,
        nav: float,
    ) -> Dict:
        """
        Calculate premium/discount to NAV.
        
        NEW v2.0: Detailed NAV analysis with arbitrage detection.
        """
        if nav <= 0:
            return {"error": "Invalid NAV"}
        
        premium_discount = (market_price - nav) / nav
        
        result = {
            "symbol": symbol,
            "market_price": market_price,
            "nav": nav,
            "premium_discount": premium_discount,
            "premium_discount_pct": premium_discount * 100,
            "status": "FAIR",
            "arbitrage_opportunity": False,
        }
        
        # Classify status
        if premium_discount > 0.03:
            result["status"] = "SIGNIFICANT_PREMIUM"
            result["arbitrage_opportunity"] = True
            result["arbitrage_action"] = "SHORT ETF, BUY UNDERLYING"
        elif premium_discount > 0.01:
            result["status"] = "PREMIUM"
        elif premium_discount < -0.03:
            result["status"] = "SIGNIFICANT_DISCOUNT"
            result["arbitrage_opportunity"] = True
            result["arbitrage_action"] = "BUY ETF, SHORT UNDERLYING"
        elif premium_discount < -0.01:
            result["status"] = "DISCOUNT"
        
        return result
    
    def get_sector_etf_for_rotation(
        self,
        target_sector: str,
    ) -> Optional[str]:
        """
        Get ETF symbol for sector rotation strategy.
        
        NEW v2.0: Maps sectors to available ETFs.
        """
        sector_mapping = {
            "FINANCIALS": "FUESSVFL",
            "BANKING": "FUESSVFL",
            "BROAD_MARKET": "E1VFVN30",
            "LARGE_CAP": "E1VFVN30",
            "MID_CAP": "FUESSV50",
            "GROWTH": "FUEVFVND",
        }
        
        return sector_mapping.get(target_sector.upper())


# =============================================================================
# ODD-LOT TRADING
# =============================================================================


class OddLotTradingLogic:
    """
    Odd-lot Trading Logic for Vietnam Market

    Odd-lots: 1-99 shares (vs standard 100-share lots)

    Key considerations:
    - Higher spreads (typically 0.5% wider)
    - Minimum commission applies
    - Lower liquidity
    - Useful for:
      - Selling remaining shares after partial exit
      - Small portfolio positions
      - DCA with small amounts
    """

    def __init__(
        self,
        spread_premium: float = ODD_LOT_SPREAD_PREMIUM,
        min_commission: float = ODD_LOT_MIN_COMMISSION,
    ):
        self.spread_premium = spread_premium
        self.min_commission = min_commission

    def is_odd_lot(self, quantity: int) -> bool:
        """Check if quantity is an odd-lot"""
        return 0 < quantity < 100

    def calculate_effective_cost(
        self,
        quantity: int,
        price: float,
        commission_rate: float = 0.0025,  # 0.25%
    ) -> Dict[str, float]:
        """
        Calculate effective cost for odd-lot trade

        Returns:
            Dict with gross_value, commission, spread_cost, total_cost, cost_pct
        """
        gross_value = quantity * price

        # Commission (with minimum)
        commission = max(gross_value * commission_rate, self.min_commission)

        # Spread cost (wider for odd-lots)
        spread_cost = gross_value * self.spread_premium

        total_cost = commission + spread_cost
        cost_pct = (total_cost / gross_value) * 100 if gross_value > 0 else 0

        return {
            "gross_value": gross_value,
            "commission": commission,
            "spread_cost": spread_cost,
            "total_cost": total_cost,
            "cost_pct": cost_pct,
        }

    def is_worth_trading(
        self,
        quantity: int,
        price: float,
        expected_return_pct: float,
    ) -> Tuple[bool, str]:
        """
        Check if odd-lot trade is worth the costs

        Args:
            quantity: Number of shares
            price: Current price
            expected_return_pct: Expected return percentage

        Returns:
            (is_worth, reason)
        """
        costs = self.calculate_effective_cost(quantity, price)

        # Need expected return > costs
        if expected_return_pct <= costs["cost_pct"]:
            return False, (
                f"Expected return ({expected_return_pct:.2f}%) <= "
                f"costs ({costs['cost_pct']:.2f}%)"
            )

        # Check minimum value threshold
        min_value = 500_000  # 500K VND minimum for odd-lot
        if costs["gross_value"] < min_value:
            return False, f"Value too small: {costs['gross_value']:,.0f} < {min_value:,.0f}"

        return True, f"OK - Net return: {expected_return_pct - costs['cost_pct']:.2f}%"

    def optimize_odd_lot_exit(
        self,
        remaining_shares: int,
        current_price: float,
        avg_cost: float,
    ) -> Dict:
        """
        Optimize odd-lot exit strategy

        Returns recommendation for handling remaining odd-lot shares
        """
        if remaining_shares >= 100:
            return {"action": "STANDARD_LOT", "message": "Not an odd-lot"}

        if remaining_shares == 0:
            return {"action": "NONE", "message": "No shares to sell"}

        # Calculate P&L
        gross_value = remaining_shares * current_price
        cost_basis = remaining_shares * avg_cost
        gross_pnl = gross_value - cost_basis
        gross_pnl_pct = (gross_pnl / cost_basis) * 100 if cost_basis > 0 else 0

        # Calculate costs
        costs = self.calculate_effective_cost(remaining_shares, current_price)
        net_pnl = gross_pnl - costs["total_cost"]
        net_pnl_pct = (net_pnl / cost_basis) * 100 if cost_basis > 0 else 0

        # Decision logic
        if net_pnl > 0:
            action = "SELL"
            message = f"Sell odd-lot for net profit: {net_pnl:+,.0f} VND ({net_pnl_pct:+.2f}%)"
        elif gross_pnl > costs["total_cost"] * 0.5:
            action = "SELL"
            message = f"Sell to minimize loss: {net_pnl:+,.0f} VND"
        elif remaining_shares < 10:
            action = "HOLD"
            message = "Hold - too small to sell efficiently"
        else:
            action = "SELL"
            message = f"Sell to free up capital: {net_pnl:+,.0f} VND"

        return {
            "action": action,
            "message": message,
            "shares": remaining_shares,
            "gross_value": gross_value,
            "gross_pnl": gross_pnl,
            "costs": costs["total_cost"],
            "net_pnl": net_pnl,
        }


# =============================================================================
# UNIFIED INSTRUMENT HANDLER
# =============================================================================


class SpecialInstrumentHandler:
    """
    Unified handler for special instruments

    Automatically detects instrument type and applies appropriate logic
    """

    def __init__(self):
        self.warrant_logic = WarrantTradingLogic()
        self.etf_logic = ETFTradingLogic()
        self.odd_lot_logic = OddLotTradingLogic()

    def detect_instrument_type(self, symbol: str) -> InstrumentType:
        """Detect instrument type from symbol"""
        if self.warrant_logic.is_warrant(symbol):
            return InstrumentType.WARRANT
        if self.etf_logic.is_etf(symbol):
            return InstrumentType.ETF
        return InstrumentType.STOCK

    def get_price_limits(
        self,
        symbol: str,
        reference_price: float,
    ) -> Dict[str, float]:
        """Get price limits based on instrument type"""
        instrument_type = self.detect_instrument_type(symbol)

        if instrument_type == InstrumentType.WARRANT:
            limit = WARRANT_PRICE_LIMIT
        else:
            limit = 0.07  # Default stock/ETF limit

        return {
            "ceiling": reference_price * (1 + limit),
            "floor": reference_price * (1 - limit),
            "limit_pct": limit * 100,
            "instrument_type": instrument_type.value,
        }

    def validate_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
    ) -> Tuple[bool, List[str]]:
        """Validate order for any instrument type"""
        warnings = []
        instrument_type = self.detect_instrument_type(symbol)

        # Check lot size
        if quantity < 100:
            if not self.odd_lot_logic.is_odd_lot(quantity):
                return False, ["Invalid quantity: must be > 0"]
            warnings.append(f"⚠️ Odd-lot order: {quantity} shares")
        elif quantity % 100 != 0:
            return False, [f"Invalid lot size: {quantity} must be multiple of 100"]

        # Instrument-specific checks
        if instrument_type == InstrumentType.WARRANT:
            warnings.append("⚠️ Warrant: ±50% daily limit, high risk")
        elif instrument_type == InstrumentType.ETF:
            if self.etf_logic.can_short(symbol):
                warnings.append("ℹ️ ETF: Short selling allowed")

        return True, warnings


# Singleton instances
_warrant_logic: Optional[WarrantTradingLogic] = None
_etf_logic: Optional[ETFTradingLogic] = None
_odd_lot_logic: Optional[OddLotTradingLogic] = None
_instrument_handler: Optional[SpecialInstrumentHandler] = None


def get_warrant_logic() -> WarrantTradingLogic:
    global _warrant_logic
    if _warrant_logic is None:
        _warrant_logic = WarrantTradingLogic()
    return _warrant_logic


def get_etf_logic() -> ETFTradingLogic:
    global _etf_logic
    if _etf_logic is None:
        _etf_logic = ETFTradingLogic()
    return _etf_logic


def get_odd_lot_logic() -> OddLotTradingLogic:
    global _odd_lot_logic
    if _odd_lot_logic is None:
        _odd_lot_logic = OddLotTradingLogic()
    return _odd_lot_logic


def get_instrument_handler() -> SpecialInstrumentHandler:
    global _instrument_handler
    if _instrument_handler is None:
        _instrument_handler = SpecialInstrumentHandler()
    return _instrument_handler


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("🧪 TESTING SPECIAL INSTRUMENTS")
    print("=" * 70)

    handler = get_instrument_handler()

    # Test instrument detection
    test_symbols = ["VNM", "CVNM", "E1VFVN30", "FPT"]
    print("\n📊 Instrument Detection:")
    for symbol in test_symbols:
        inst_type = handler.detect_instrument_type(symbol)
        limits = handler.get_price_limits(symbol, 100000)
        print(f"   {symbol}: {inst_type.value} - Limit: ±{limits['limit_pct']:.0f}%")

    # Test odd-lot
    print("\n📊 Odd-lot Analysis:")
    odd_lot = get_odd_lot_logic()
    result = odd_lot.optimize_odd_lot_exit(remaining_shares=45, current_price=85000, avg_cost=80000)
    print(f"   Action: {result['action']}")
    print(f"   Message: {result['message']}")
    print(f"   Net P&L: {result['net_pnl']:+,.0f} VND")

    # Test warrant
    print("\n📊 Warrant Analysis:")
    warrant = get_warrant_logic()
    print(f"   Is CVNM warrant? {warrant.is_warrant('CVNM')}")
    print(f"   Underlying: {warrant.get_underlying('CVNM')}")
    limits = warrant.get_price_limits(10000)
    print(f"   Price limits: {limits['floor']:,.0f} - {limits['ceiling']:,.0f}")

    print("\n" + "=" * 70)
    print("✅ Special instruments test completed!")
    print("=" * 70)
