# -*- coding: utf-8 -*-
"""
Warrant and ETF Trading Rules for Vietnam Stock Market

Vietnam Covered Warrants (Chứng quyền có bảo đảm - CW):
- Introduced in 2019 on HOSE
- Symbol format: CXXX_YYYY_ZZ (e.g., CVNM2301 = VNM warrant, issuer code 23, batch 01)
- Can be used for SHORT exposure (put warrants)
- T+0 settlement for covered warrants
- Price limits: ±50% for warrants (not ±7% like stocks)
- Expiration dates matter - decay to zero

ETFs on Vietnam Market:
- E1VFVN30 (VN30 Index ETF) - Most liquid
- FUEVFVND (Diamond ETF)
- FUESSVFL (SSIAM VN30 ETF)
- ETFs can be shorted via margin (unlike individual stocks)
- Lower transaction costs than individual stocks
- T+2 settlement like stocks

This module provides:
1. Warrant classification and validation
2. Warrant pricing checks (Greeks awareness)
3. ETF-specific rules
4. Expiration and decay tracking
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InstrumentType(Enum):
    """Type of trading instrument"""
    STOCK = "STOCK"
    COVERED_WARRANT = "CW"
    ETF = "ETF"
    BOND = "BOND"
    UNKNOWN = "UNKNOWN"


class WarrantType(Enum):
    """Type of covered warrant"""
    CALL = "CALL"  # Bullish - profits when underlying rises
    PUT = "PUT"    # Bearish - profits when underlying falls


class WarrantStatus(Enum):
    """Status of covered warrant"""
    ACTIVE = "ACTIVE"           # Trading normally
    NEAR_EXPIRY = "NEAR_EXPIRY" # < 30 days to expiry
    EXPIRING_SOON = "EXPIRING"  # < 7 days to expiry
    EXPIRED = "EXPIRED"         # Past expiry date
    SUSPENDED = "SUSPENDED"     # Trading halted


@dataclass
class WarrantInfo:
    """Information about a covered warrant"""
    symbol: str
    underlying: str           # Underlying stock (e.g., "VNM")
    warrant_type: WarrantType # CALL or PUT
    issuer: str              # Issuing securities company
    strike_price: float      # Exercise price
    exercise_ratio: float    # How many warrants = 1 underlying share
    expiry_date: date
    issue_date: date
    status: WarrantStatus
    last_price: float = 0.0
    underlying_price: float = 0.0

    @property
    def days_to_expiry(self) -> int:
        """Days until expiration"""
        return (self.expiry_date - date.today()).days

    @property
    def is_in_the_money(self) -> bool:
        """Check if warrant is in the money"""
        if self.underlying_price <= 0:
            return False
        if self.warrant_type == WarrantType.CALL:
            return self.underlying_price > self.strike_price
        else:  # PUT
            return self.underlying_price < self.strike_price

    @property
    def intrinsic_value(self) -> float:
        """Calculate intrinsic value"""
        if self.underlying_price <= 0:
            return 0.0
        if self.warrant_type == WarrantType.CALL:
            return max(0, (self.underlying_price - self.strike_price) / self.exercise_ratio)
        else:  # PUT
            return max(0, (self.strike_price - self.underlying_price) / self.exercise_ratio)

    @property
    def time_value(self) -> float:
        """Calculate time value (if last_price available)"""
        if self.last_price <= 0:
            return 0.0
        return max(0, self.last_price - self.intrinsic_value)


@dataclass
class ETFInfo:
    """Information about an ETF"""
    symbol: str
    name: str
    index_tracked: str        # Index being tracked (e.g., "VN30")
    nav: float               # Net Asset Value
    market_price: float
    aum: float               # Assets Under Management
    expense_ratio: float     # Annual expense ratio
    can_short: bool          # Whether shorting is allowed
    tracking_error: float    # Tracking error percentage

    @property
    def premium_discount(self) -> float:
        """Premium/discount to NAV"""
        if self.nav <= 0:
            return 0.0
        return (self.market_price - self.nav) / self.nav


class WarrantETFHandler:
    """
    Handler for Warrant and ETF trading rules in Vietnam market.

    Key features:
    - Automatic instrument type detection
    - Warrant expiration tracking
    - Price limit validation (±50% for warrants vs ±7% for stocks)
    - ETF premium/discount alerts
    - T+0 settlement for warrants

    Usage:
        handler = WarrantETFHandler()
        instrument_type = handler.classify_instrument("CVNM2301")
        if instrument_type == InstrumentType.COVERED_WARRANT:
            can_trade, reason = handler.validate_warrant_trade("CVNM2301", "BUY")
    """

    # Warrant symbol patterns
    WARRANT_PATTERN = re.compile(r"^C([A-Z]{3})(\d{2})(\d{2})$")  # CVNM2301

    # Known ETFs on Vietnam market
    ETFS = {
        "E1VFVN30": {"name": "VN30 ETF", "index": "VN30", "can_short": True},
        "FUEVFVND": {"name": "Diamond ETF", "index": "VNDiamond", "can_short": True},
        "FUESSVFL": {"name": "SSIAM VN30", "index": "VN30", "can_short": True},
        "FUEMAV30": {"name": "MAFM VN30", "index": "VN30", "can_short": True},
        "FUEKIV30": {"name": "KIM VN30", "index": "VN30", "can_short": True},
        "FUEVN100": {"name": "VN100 ETF", "index": "VN100", "can_short": False},
    }

    # Warrant issuers by code
    WARRANT_ISSUERS = {
        "01": "SSI",
        "02": "VNDirect",
        "03": "HSC",
        "04": "TCBS",
        "05": "MBS",
        "06": "KIS",
        "07": "VCSC",
        "08": "BVSC",
        "23": "VPS",  # Example
    }

    # Price limits
    STOCK_PRICE_LIMIT = 0.07    # ±7%
    WARRANT_PRICE_LIMIT = 0.50  # ±50%
    ETF_PRICE_LIMIT = 0.07      # ±7% (same as stocks)

    # Trading rules
    WARRANT_MIN_DAYS_TO_EXPIRY = 3  # Don't trade if < 3 days to expiry
    WARRANT_WARNING_DAYS = 30       # Warning if < 30 days to expiry
    WARRANT_EXPIRING_DAYS = 7       # "Expiring soon" if < 7 days

    def __init__(self):
        """Initialize warrant/ETF handler."""
        self._warrant_cache: Dict[str, WarrantInfo] = {}
        self._etf_cache: Dict[str, ETFInfo] = {}

        logger.info("✅ WarrantETFHandler initialized")

    def classify_instrument(self, symbol: str) -> InstrumentType:
        """
        Classify a trading instrument by symbol.

        Args:
            symbol: Trading symbol

        Returns:
            InstrumentType enum
        """
        symbol = symbol.upper().strip()

        # Check if warrant (starts with 'C' + 3 letters + 4 digits)
        if self.WARRANT_PATTERN.match(symbol):
            return InstrumentType.COVERED_WARRANT

        # Check if known ETF
        if symbol in self.ETFS:
            return InstrumentType.ETF

        # Check ETF patterns
        if symbol.startswith("E1VF") or symbol.startswith("FUE"):
            return InstrumentType.ETF

        # Default to stock
        return InstrumentType.STOCK

    def parse_warrant_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Parse warrant symbol to extract information.

        Vietnam warrant symbol format: CXXX_YYZZ
        - C: Covered warrant prefix
        - XXX: Underlying stock (3 letters)
        - YY: Issuer code (2 digits)
        - ZZ: Batch number (2 digits)

        Args:
            symbol: Warrant symbol (e.g., "CVNM2301")

        Returns:
            Dict with parsed info or None if invalid
        """
        match = self.WARRANT_PATTERN.match(symbol.upper())
        if not match:
            return None

        underlying = match.group(1)
        issuer_code = match.group(2)
        batch = match.group(3)

        return {
            "symbol": symbol.upper(),
            "underlying": underlying,
            "issuer_code": issuer_code,
            "issuer_name": self.WARRANT_ISSUERS.get(issuer_code, "Unknown"),
            "batch": batch,
        }

    def get_price_limit(self, symbol: str) -> float:
        """
        Get price limit for an instrument.

        Args:
            symbol: Trading symbol

        Returns:
            Price limit as decimal (e.g., 0.07 for 7%)
        """
        instrument_type = self.classify_instrument(symbol)

        if instrument_type == InstrumentType.COVERED_WARRANT:
            return self.WARRANT_PRICE_LIMIT  # ±50%
        else:
            return self.STOCK_PRICE_LIMIT  # ±7%

    def get_settlement_days(self, symbol: str) -> int:
        """
        Get settlement days for an instrument.

        Args:
            symbol: Trading symbol

        Returns:
            Settlement days (0 for warrants, 2 for stocks/ETFs)
        """
        instrument_type = self.classify_instrument(symbol)

        if instrument_type == InstrumentType.COVERED_WARRANT:
            return 0  # T+0 for covered warrants
        else:
            return 2  # T+2 for stocks and ETFs

    def validate_warrant_trade(
        self,
        symbol: str,
        side: str,
        expiry_date: Optional[date] = None
    ) -> Tuple[bool, str, List[str]]:
        """
        Validate if a warrant trade is allowed.

        Checks:
        - Symbol format validity
        - Days to expiry (min 3 days)
        - Expiry warnings

        Args:
            symbol: Warrant symbol
            side: "BUY" or "SELL"
            expiry_date: Warrant expiry date (if known)

        Returns:
            Tuple of (is_valid, reason, warnings)
        """
        warnings = []

        # Check symbol format
        parsed = self.parse_warrant_symbol(symbol)
        if not parsed:
            return False, f"Invalid warrant symbol format: {symbol}", []

        # Check expiry if provided
        if expiry_date:
            days_to_expiry = (expiry_date - date.today()).days

            if days_to_expiry <= 0:
                return False, f"Warrant {symbol} has expired", []

            if days_to_expiry < self.WARRANT_MIN_DAYS_TO_EXPIRY:
                return (
                    False,
                    f"Warrant {symbol} expires in {days_to_expiry} days - "
                    f"minimum {self.WARRANT_MIN_DAYS_TO_EXPIRY} days required",
                    []
                )

            if days_to_expiry < self.WARRANT_EXPIRING_DAYS:
                warnings.append(
                    f"⚠️ EXPIRING SOON: {symbol} expires in {days_to_expiry} days"
                )
            elif days_to_expiry < self.WARRANT_WARNING_DAYS:
                warnings.append(
                    f"🟡 Near expiry: {symbol} expires in {days_to_expiry} days"
                )

        # Additional warnings for buying
        if side.upper() == "BUY":
            warnings.append(
                f"💡 Warrant tip: {symbol} has ±50% daily price limit (not ±7%)"
            )
            warnings.append(
                f"⏰ T+0 settlement - can sell same day"
            )

        return True, "✅ Warrant trade validated", warnings

    def validate_etf_trade(
        self,
        symbol: str,
        side: str,
        nav: Optional[float] = None,
        market_price: Optional[float] = None
    ) -> Tuple[bool, str, List[str]]:
        """
        Validate if an ETF trade is allowed.

        Checks:
        - Premium/discount to NAV
        - Short selling eligibility

        Args:
            symbol: ETF symbol
            side: "BUY", "SELL", or "SHORT"
            nav: Net Asset Value (if known)
            market_price: Current market price (if known)

        Returns:
            Tuple of (is_valid, reason, warnings)
        """
        warnings = []
        symbol = symbol.upper()

        # Check if known ETF
        etf_info = self.ETFS.get(symbol)
        if not etf_info:
            warnings.append(f"⚠️ {symbol} not in known ETF list - verify before trading")

        # Check short selling eligibility
        if side.upper() == "SHORT":
            if etf_info and not etf_info.get("can_short", False):
                return False, f"ETF {symbol} does not allow short selling", []

        # Check premium/discount
        if nav and market_price and nav > 0:
            premium_pct = ((market_price - nav) / nav) * 100

            if premium_pct > 2:
                warnings.append(
                    f"⚠️ ETF trading at {premium_pct:.1f}% PREMIUM to NAV - consider waiting"
                )
            elif premium_pct < -2:
                warnings.append(
                    f"💡 ETF trading at {abs(premium_pct):.1f}% DISCOUNT to NAV - may be opportunity"
                )

        # General ETF tips
        if side.upper() == "BUY":
            warnings.append(f"💡 ETF tip: Lower transaction costs than buying component stocks")
            if etf_info:
                warnings.append(f"📊 Tracks: {etf_info.get('index', 'Unknown index')}")

        return True, "✅ ETF trade validated", warnings

    def get_trading_rules(self, symbol: str) -> Dict:
        """
        Get comprehensive trading rules for an instrument.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with all applicable trading rules
        """
        instrument_type = self.classify_instrument(symbol)

        rules = {
            "symbol": symbol,
            "instrument_type": instrument_type.value,
            "price_limit": self.get_price_limit(symbol),
            "price_limit_pct": f"±{self.get_price_limit(symbol)*100:.0f}%",
            "settlement_days": self.get_settlement_days(symbol),
            "settlement": f"T+{self.get_settlement_days(symbol)}",
            "lot_size": 100,  # Same for all
            "can_short": False,
            "margin_available": True,
            "notes": []
        }

        if instrument_type == InstrumentType.COVERED_WARRANT:
            parsed = self.parse_warrant_symbol(symbol)
            rules.update({
                "underlying": parsed["underlying"] if parsed else None,
                "issuer": parsed["issuer_name"] if parsed else None,
                "can_short": False,  # Can't short warrants directly
                "margin_available": False,  # No margin for warrants
                "notes": [
                    "T+0 settlement - can sell same day",
                    "±50% daily price limit",
                    "Check expiry date before trading",
                    "Time decay accelerates near expiry",
                ]
            })
        elif instrument_type == InstrumentType.ETF:
            etf_info = self.ETFS.get(symbol.upper())
            rules.update({
                "index_tracked": etf_info.get("index") if etf_info else None,
                "can_short": etf_info.get("can_short", False) if etf_info else False,
                "margin_available": True,
                "notes": [
                    "T+2 settlement",
                    "±7% daily price limit",
                    "Can be used for index exposure",
                    "Lower costs than buying component stocks",
                ]
            })
            if rules["can_short"]:
                rules["notes"].append("Short selling allowed via margin")
        else:  # STOCK
            rules["notes"] = [
                "T+2 settlement",
                "±7% daily price limit (HOSE)",
                "No short selling allowed",
            ]

        return rules

    def get_warrant_decay_warning(
        self,
        days_to_expiry: int,
        is_out_of_money: bool
    ) -> Optional[str]:
        """
        Get time decay warning for warrant.

        Args:
            days_to_expiry: Days until expiration
            is_out_of_money: Whether warrant is OTM

        Returns:
            Warning message or None
        """
        if days_to_expiry <= 0:
            return "🚨 EXPIRED: Warrant has no value"

        if days_to_expiry <= 3:
            return "🚨 CRITICAL: < 3 days to expiry - extreme time decay"

        if days_to_expiry <= 7:
            if is_out_of_money:
                return "⚠️ HIGH RISK: OTM warrant with < 7 days - likely to expire worthless"
            return "⚠️ < 7 days to expiry - accelerated time decay"

        if days_to_expiry <= 14:
            return "🟡 < 2 weeks to expiry - time decay increasing"

        if days_to_expiry <= 30:
            return "📊 < 30 days to expiry - monitor time decay"

        return None

    def get_status_message(self) -> str:
        """Get handler status message."""
        lines = [
            "=" * 50,
            "📊 WARRANT/ETF TRADING RULES - VIETNAM",
            "=" * 50,
            "",
            "📜 COVERED WARRANTS (CW):",
            f"   • Price limit: ±50% (vs ±7% for stocks)",
            f"   • Settlement: T+0 (same day)",
            f"   • Symbol format: CXXX_YYZZ (e.g., CVNM2301)",
            f"   • No short selling, no margin",
            f"   • Watch expiry dates - time decay!",
            "",
            "📈 ETFs:",
            f"   • Price limit: ±7% (same as stocks)",
            f"   • Settlement: T+2",
            f"   • Some allow short selling via margin",
            "",
            "Known ETFs:",
        ]

        for symbol, info in self.ETFS.items():
            short_status = "✓ Short OK" if info.get("can_short") else "✗ No short"
            lines.append(f"   • {symbol}: {info['name']} ({short_status})")

        lines.extend([
            "",
            "=" * 50,
        ])

        return "\n".join(lines)


# Singleton instance
_warrant_etf_handler: Optional[WarrantETFHandler] = None


def get_warrant_etf_handler() -> WarrantETFHandler:
    """Get singleton warrant/ETF handler instance."""
    global _warrant_etf_handler
    if _warrant_etf_handler is None:
        _warrant_etf_handler = WarrantETFHandler()
    return _warrant_etf_handler


def classify_and_validate(
    symbol: str,
    side: str = "BUY"
) -> Dict:
    """
    Convenience function to classify instrument and get trading rules.

    Args:
        symbol: Trading symbol
        side: "BUY", "SELL", or "SHORT"

    Returns:
        Dict with instrument type, validation, and rules
    """
    handler = get_warrant_etf_handler()
    instrument_type = handler.classify_instrument(symbol)

    result = {
        "symbol": symbol,
        "instrument_type": instrument_type.value,
        "is_valid": True,
        "reason": "",
        "warnings": [],
        "rules": handler.get_trading_rules(symbol)
    }

    if instrument_type == InstrumentType.COVERED_WARRANT:
        is_valid, reason, warnings = handler.validate_warrant_trade(symbol, side)
        result.update({
            "is_valid": is_valid,
            "reason": reason,
            "warnings": warnings
        })
    elif instrument_type == InstrumentType.ETF:
        is_valid, reason, warnings = handler.validate_etf_trade(symbol, side)
        result.update({
            "is_valid": is_valid,
            "reason": reason,
            "warnings": warnings
        })
    else:
        result["reason"] = "✅ Standard stock - normal trading rules apply"

    return result


# Test
if __name__ == "__main__":
    print("Testing Warrant/ETF Handler...")

    handler = WarrantETFHandler()

    print("\n" + handler.get_status_message())

    print("\n1️⃣ Testing instrument classification:")
    test_symbols = ["VNM", "HPG", "CVNM2301", "E1VFVN30", "FUEVFVND", "ABC"]
    for symbol in test_symbols:
        itype = handler.classify_instrument(symbol)
        print(f"   {symbol}: {itype.value}")

    print("\n2️⃣ Testing warrant validation:")
    result = classify_and_validate("CVNM2301", "BUY")
    print(f"   Symbol: {result['symbol']}")
    print(f"   Type: {result['instrument_type']}")
    print(f"   Valid: {result['is_valid']}")
    print(f"   Reason: {result['reason']}")
    for warning in result['warnings']:
        print(f"   Warning: {warning}")

    print("\n3️⃣ Testing ETF validation:")
    result = classify_and_validate("E1VFVN30", "SHORT")
    print(f"   Symbol: {result['symbol']}")
    print(f"   Type: {result['instrument_type']}")
    print(f"   Valid: {result['is_valid']}")
    print(f"   Can Short: {result['rules']['can_short']}")

    print("\n4️⃣ Testing trading rules:")
    for symbol in ["VNM", "CVNM2301", "E1VFVN30"]:
        rules = handler.get_trading_rules(symbol)
        print(f"\n   {symbol}:")
        print(f"   Type: {rules['instrument_type']}")
        print(f"   Price Limit: {rules['price_limit_pct']}")
        print(f"   Settlement: {rules['settlement']}")
        print(f"   Can Short: {rules['can_short']}")

    print("\n5️⃣ Testing warrant decay warnings:")
    for days in [1, 5, 10, 20, 45]:
        warning = handler.get_warrant_decay_warning(days, is_out_of_money=True)
        print(f"   {days} days to expiry (OTM): {warning or 'No warning'}")

    print("\n✅ Test completed!")
