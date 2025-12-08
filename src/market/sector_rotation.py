# -*- coding: utf-8 -*-
"""
Sector Rotation Analysis for Vietnam Stock Market

Analyzes sector momentum to identify rotation opportunities.
Vietnam market has distinct sector cycles driven by:
- Banking: Interest rate cycles, credit growth
- Real Estate: Policy changes, interest rates
- Technology: Global tech trends, FDI
- Consumer: Domestic consumption, inflation

Usage:
    from src.market.sector_rotation import get_sector_rotation_analyzer
    
    analyzer = get_sector_rotation_analyzer()
    signal = analyzer.get_rotation_signal()
    print(f"Overweight: {signal['overweight']}")
    print(f"Underweight: {signal['underweight']}")

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Sector representative stocks (top 3-5 by market cap)
SECTOR_REPRESENTATIVES = {
    "BANKING": ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "HDB"],
    "REAL_ESTATE": ["VHM", "VIC", "NVL", "VRE", "KDH", "DXG"],
    "TECHNOLOGY": ["FPT", "CMG", "ELC"],
    "CONSUMER": ["VNM", "MSN", "MWG", "SAB", "PNJ"],
    "ENERGY": ["GAS", "PLX", "PVD", "PVS"],
    "INDUSTRIAL": ["HPG", "HSG", "NKG", "GVR"],
    "SECURITIES": ["SSI", "VCI", "HCM", "VND"],
    "UTILITIES": ["POW", "NT2", "PPC"],
    "AVIATION": ["VJC", "HVN"],
    "INSURANCE": ["BVH", "PVI", "BMI"],
}

# Sector characteristics for Vietnam market
SECTOR_CHARACTERISTICS = {
    "BANKING": {
        "beta": 1.2,  # High correlation with market
        "cycle_sensitivity": "HIGH",  # Sensitive to interest rate cycles
        "foreign_limit": 0.30,  # 30% foreign ownership limit
        "typical_pe": 8,  # Typical P/E ratio
    },
    "REAL_ESTATE": {
        "beta": 1.4,  # Very volatile
        "cycle_sensitivity": "VERY_HIGH",
        "foreign_limit": 0.49,
        "typical_pe": 12,
    },
    "TECHNOLOGY": {
        "beta": 0.9,  # Less correlated
        "cycle_sensitivity": "LOW",
        "foreign_limit": 1.00,  # No limit
        "typical_pe": 18,
    },
    "CONSUMER": {
        "beta": 0.8,  # Defensive
        "cycle_sensitivity": "LOW",
        "foreign_limit": 0.49,
        "typical_pe": 20,
    },
    "ENERGY": {
        "beta": 1.1,
        "cycle_sensitivity": "MEDIUM",
        "foreign_limit": 0.49,
        "typical_pe": 10,
    },
    "INDUSTRIAL": {
        "beta": 1.3,
        "cycle_sensitivity": "HIGH",
        "foreign_limit": 0.49,
        "typical_pe": 8,
    },
    "SECURITIES": {
        "beta": 1.5,  # Highest beta
        "cycle_sensitivity": "VERY_HIGH",
        "foreign_limit": 0.49,
        "typical_pe": 12,
    },
}


# =============================================================================
# NEW v7.0: SECTOR LEADING INDICATORS
# =============================================================================
# Leading indicators help predict sector rotation before price moves
# Each sector has specific macro/micro indicators that lead price by N days
#
# Usage:
#   from src.market.sector_rotation import SECTOR_LEADING_INDICATORS
#   banking_indicators = SECTOR_LEADING_INDICATORS["BANKING"]
#   lead_time = banking_indicators["lead_time_days"]  # 30 days
#
SECTOR_LEADING_INDICATORS = {
    "BANKING": {
        "indicators": [
            "interest_rate",  # SBV policy rate changes
            "credit_growth",  # System-wide credit growth
            "npl_ratio",  # Non-performing loan ratio
            "deposit_growth",  # Deposit mobilization
            "nim_trend",  # Net Interest Margin trend
        ],
        "lead_time_days": 30,  # Banking leads market by ~30 days
        "correlation_with_market": 0.85,
        "best_entry_regime": "EARLY_BULL",  # Best to enter early in bull cycle
        "warning_signals": [
            "rising_npl",  # NPL > 3% is warning
            "credit_tightening",  # SBV credit growth cap
            "foreign_selling",  # Consecutive foreign net sell
        ],
    },
    "REAL_ESTATE": {
        "indicators": [
            "bond_yield",  # Government bond yield (inverse correlation)
            "policy_news",  # Land law, housing policy changes
            "land_price_index",  # Land auction prices
            "mortgage_rate",  # Home loan interest rates
            "inventory_level",  # Unsold inventory
        ],
        "lead_time_days": 60,  # Real estate leads by ~60 days (slow cycle)
        "correlation_with_market": 0.75,
        "best_entry_regime": "LATE_BEAR",  # Best to accumulate late in bear
        "warning_signals": [
            "rising_bond_yield",  # Bond yield > 10% is warning
            "policy_tightening",  # Credit restrictions for RE
            "high_inventory",  # Inventory > 2 years supply
        ],
    },
    "SECURITIES": {
        "indicators": [
            "trading_volume",  # Market trading volume
            "new_accounts",  # New trading accounts opened
            "margin_debt",  # System-wide margin debt
            "ipo_activity",  # IPO/listing pipeline
            "foreign_flow",  # Foreign net buy/sell
        ],
        "lead_time_days": 0,  # Coincident indicator (moves with market)
        "correlation_with_market": 0.95,
        "best_entry_regime": "EARLY_BULL",  # High beta, enter early
        "warning_signals": [
            "margin_debt_peak",  # Margin debt > 100T VND
            "volume_divergence",  # Price up but volume down
            "excessive_ipo",  # Too many IPOs = market top
        ],
    },
    "TECHNOLOGY": {
        "indicators": [
            "global_tech_index",  # NASDAQ, global tech trends
            "fdi_inflow",  # Foreign direct investment
            "digital_adoption",  # E-commerce, fintech growth
            "it_spending",  # Corporate IT spending
            "talent_market",  # Tech salary trends
        ],
        "lead_time_days": 45,  # Tech leads by ~45 days (global correlation)
        "correlation_with_market": 0.60,
        "best_entry_regime": "ANY",  # Defensive, can enter anytime
        "warning_signals": [
            "global_tech_crash",  # NASDAQ correction > 10%
            "fdi_slowdown",  # FDI growth < 5%
            "margin_compression",  # Sector margin declining
        ],
    },
    "CONSUMER": {
        "indicators": [
            "cpi_inflation",  # Consumer price index
            "retail_sales",  # Retail sales growth
            "consumer_confidence",  # Consumer confidence index
            "disposable_income",  # Income growth
            "unemployment",  # Unemployment rate
        ],
        "lead_time_days": 30,  # Consumer leads by ~30 days
        "correlation_with_market": 0.50,
        "best_entry_regime": "LATE_BULL",  # Defensive, hold in late cycle
        "warning_signals": [
            "high_inflation",  # CPI > 5% is warning
            "falling_confidence",  # Consumer confidence declining
            "rising_unemployment",  # Unemployment > 3%
        ],
    },
    "ENERGY": {
        "indicators": [
            "oil_price",  # Brent crude price
            "gas_price",  # Natural gas price
            "electricity_demand",  # Power consumption growth
            "policy_support",  # Energy policy, subsidies
            "capacity_addition",  # New power plant capacity
        ],
        "lead_time_days": 20,  # Energy leads by ~20 days
        "correlation_with_market": 0.70,
        "best_entry_regime": "EARLY_BULL",
        "warning_signals": [
            "oil_price_crash",  # Oil < $60/barrel
            "policy_change",  # Subsidy removal
            "overcapacity",  # Power oversupply
        ],
    },
    "INDUSTRIAL": {
        "indicators": [
            "pmi_index",  # Purchasing Managers Index
            "export_growth",  # Export value growth
            "steel_price",  # Steel/commodity prices
            "construction_index",  # Construction activity
            "fdi_manufacturing",  # FDI into manufacturing
        ],
        "lead_time_days": 25,  # Industrial leads by ~25 days
        "correlation_with_market": 0.80,
        "best_entry_regime": "EARLY_BULL",
        "warning_signals": [
            "pmi_contraction",  # PMI < 50
            "export_decline",  # Export growth < 0%
            "commodity_crash",  # Steel price crash > 20%
        ],
    },
}


@dataclass
class SectorMomentum:
    """Sector momentum data"""

    sector: str
    momentum_score: float  # -1 to +1
    return_1w: float
    return_1m: float
    return_3m: float
    relative_strength: float  # vs VNINDEX
    volume_trend: str  # INCREASING, DECREASING, STABLE
    trend: str  # BULLISH, BEARISH, NEUTRAL
    rank: int  # 1 = strongest


@dataclass
class RotationSignal:
    """Sector rotation recommendation"""

    timestamp: str
    overweight: List[str]  # Sectors to increase exposure
    underweight: List[str]  # Sectors to decrease exposure
    neutral: List[str]  # Sectors to maintain
    top_picks: Dict[str, List[str]]  # Top stocks per overweight sector
    avoid_list: Dict[str, List[str]]  # Stocks to avoid per underweight sector
    confidence: float  # 0-100
    rationale: str


class SectorRotationAnalyzer:
    """
    Analyze sector momentum for rotation strategy.

    Rotation Logic:
    1. Calculate momentum for each sector (1W, 1M, 3M returns)
    2. Calculate relative strength vs VNINDEX
    3. Rank sectors by combined score
    4. Overweight top 2-3 sectors, underweight bottom 2-3

    Vietnam-specific considerations:
    - Banking sector leads in bull markets
    - Real Estate is most volatile
    - Consumer/Tech are defensive
    - Securities amplify market moves
    """

    def __init__(
        self,
        lookback_days: int = 60,
        momentum_weights: Dict[str, float] = None,
        overweight_threshold: float = 0.5,
        underweight_threshold: float = -0.3,
        cache_ttl_seconds: int = 3600,  # 1 hour
    ):
        self.lookback_days = lookback_days
        self.momentum_weights = momentum_weights or {
            "1w": 0.3,
            "1m": 0.4,
            "3m": 0.3,
        }
        self.overweight_threshold = overweight_threshold
        self.underweight_threshold = underweight_threshold
        self.cache_ttl = cache_ttl_seconds

        # Cache
        self._cache: Optional[Dict[str, SectorMomentum]] = None
        self._cache_time: Optional[datetime] = None

    def get_sector_momentum(self, force_refresh: bool = False) -> Dict[str, SectorMomentum]:
        """
        Calculate momentum score for each sector.

        Returns:
            Dict of {sector_name: SectorMomentum}
        """
        # Check cache
        if not force_refresh and self._is_cache_valid():
            return self._cache

        try:
            momentum_data = {}

            for sector, symbols in SECTOR_REPRESENTATIVES.items():
                sector_momentum = self._calculate_sector_momentum(sector, symbols)
                if sector_momentum:
                    momentum_data[sector] = sector_momentum

            # Rank sectors
            sorted_sectors = sorted(
                momentum_data.values(), key=lambda x: x.momentum_score, reverse=True
            )
            for rank, sector_data in enumerate(sorted_sectors, 1):
                sector_data.rank = rank

            # Update cache
            self._cache = momentum_data
            self._cache_time = datetime.now()

            return momentum_data

        except Exception as e:
            logger.error(f"Sector momentum calculation failed: {e}", exc_info=True)
            return {}

    def _calculate_sector_momentum(
        self, sector: str, symbols: List[str]
    ) -> Optional[SectorMomentum]:
        """Calculate momentum for a single sector"""
        try:
            returns_1w = []
            returns_1m = []
            returns_3m = []
            volume_changes = []

            for symbol in symbols[:5]:  # Top 5 stocks
                df = self._load_stock_data(symbol)
                if df is None or len(df) < 60:
                    continue

                # Calculate returns
                current_price = df["close"].iloc[-1]
                price_1w = df["close"].iloc[-5] if len(df) >= 5 else current_price
                price_1m = df["close"].iloc[-20] if len(df) >= 20 else current_price
                price_3m = df["close"].iloc[-60] if len(df) >= 60 else current_price

                returns_1w.append((current_price - price_1w) / price_1w)
                returns_1m.append((current_price - price_1m) / price_1m)
                returns_3m.append((current_price - price_3m) / price_3m)

                # Volume trend
                recent_vol = df["volume"].tail(5).mean()
                avg_vol = df["volume"].tail(20).mean()
                volume_changes.append(recent_vol / avg_vol if avg_vol > 0 else 1)

            if not returns_1w:
                return None

            # Average returns
            avg_1w = sum(returns_1w) / len(returns_1w)
            avg_1m = sum(returns_1m) / len(returns_1m)
            avg_3m = sum(returns_3m) / len(returns_3m)

            # Weighted momentum score
            momentum_score = (
                avg_1w * self.momentum_weights["1w"]
                + avg_1m * self.momentum_weights["1m"]
                + avg_3m * self.momentum_weights["3m"]
            )

            # Normalize to -1 to +1 range
            momentum_score = max(-1, min(1, momentum_score * 5))

            # Calculate relative strength vs VNINDEX
            vnindex_return = self._get_vnindex_return()
            relative_strength = avg_1m - vnindex_return if vnindex_return else avg_1m

            # Volume trend
            avg_volume_change = sum(volume_changes) / len(volume_changes)
            if avg_volume_change > 1.2:
                volume_trend = "INCREASING"
            elif avg_volume_change < 0.8:
                volume_trend = "DECREASING"
            else:
                volume_trend = "STABLE"

            # Trend determination
            if momentum_score > 0.3:
                trend = "BULLISH"
            elif momentum_score < -0.3:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            return SectorMomentum(
                sector=sector,
                momentum_score=momentum_score,
                return_1w=avg_1w,
                return_1m=avg_1m,
                return_3m=avg_3m,
                relative_strength=relative_strength,
                volume_trend=volume_trend,
                trend=trend,
                rank=0,  # Will be set later
            )

        except Exception as e:
            logger.warning(f"Failed to calculate momentum for {sector}: {e}")
            return None

    def _load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load stock data for a symbol"""
        try:
            from src.data.loader import load_data

            return load_data(symbol, lookback=self.lookback_days + 10)
        except Exception as e:
            logger.debug(f"Failed to load data for {symbol}: {e}")
            return None

    def _get_vnindex_return(self) -> float:
        """Get VNINDEX 1-month return"""
        try:
            from src.data.vnindex_cache import get_cached_vnindex

            df = get_cached_vnindex(lookback=30)
            if df is not None and len(df) >= 20:
                return (df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20]
        except Exception:
            pass
        return 0.0

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if self._cache is None or self._cache_time is None:
            return False
        age = (datetime.now() - self._cache_time).total_seconds()
        return age < self.cache_ttl

    def get_rotation_signal(self, force_refresh: bool = False) -> RotationSignal:
        """
        Get sector rotation recommendation.

        Returns:
            RotationSignal with overweight/underweight recommendations
        """
        momentum_data = self.get_sector_momentum(force_refresh)

        if not momentum_data:
            return RotationSignal(
                timestamp=datetime.now().isoformat(),
                overweight=[],
                underweight=[],
                neutral=list(SECTOR_REPRESENTATIVES.keys()),
                top_picks={},
                avoid_list={},
                confidence=0,
                rationale="Insufficient data for sector rotation analysis",
            )

        overweight = []
        underweight = []
        neutral = []
        top_picks = {}
        avoid_list = {}

        for sector, data in momentum_data.items():
            if data.momentum_score >= self.overweight_threshold:
                overweight.append(sector)
                # Get top 3 stocks in sector
                top_picks[sector] = SECTOR_REPRESENTATIVES.get(sector, [])[:3]
            elif data.momentum_score <= self.underweight_threshold:
                underweight.append(sector)
                avoid_list[sector] = SECTOR_REPRESENTATIVES.get(sector, [])[:3]
            else:
                neutral.append(sector)

        # Calculate confidence based on signal clarity
        if overweight and underweight:
            # Clear rotation signal
            avg_overweight = sum(momentum_data[s].momentum_score for s in overweight) / len(
                overweight
            )
            avg_underweight = sum(momentum_data[s].momentum_score for s in underweight) / len(
                underweight
            )
            spread = avg_overweight - avg_underweight
            confidence = min(100, spread * 100)
        else:
            confidence = 30  # Low confidence if no clear rotation

        # Generate rationale
        rationale = self._generate_rationale(momentum_data, overweight, underweight)

        return RotationSignal(
            timestamp=datetime.now().isoformat(),
            overweight=overweight,
            underweight=underweight,
            neutral=neutral,
            top_picks=top_picks,
            avoid_list=avoid_list,
            confidence=confidence,
            rationale=rationale,
        )

    def _generate_rationale(
        self,
        momentum_data: Dict[str, SectorMomentum],
        overweight: List[str],
        underweight: List[str],
    ) -> str:
        """Generate human-readable rationale"""
        parts = []

        if overweight:
            top_sector = overweight[0]
            top_data = momentum_data.get(top_sector)
            if top_data:
                parts.append(
                    f"OVERWEIGHT {top_sector}: "
                    f"Momentum {top_data.momentum_score:.2f}, "
                    f"1M return {top_data.return_1m*100:+.1f}%, "
                    f"Volume {top_data.volume_trend}"
                )

        if underweight:
            bottom_sector = underweight[-1]
            bottom_data = momentum_data.get(bottom_sector)
            if bottom_data:
                parts.append(
                    f"UNDERWEIGHT {bottom_sector}: "
                    f"Momentum {bottom_data.momentum_score:.2f}, "
                    f"1M return {bottom_data.return_1m*100:+.1f}%"
                )

        return " | ".join(parts) if parts else "No clear rotation signal"

    def get_sector_for_symbol(self, symbol: str) -> Optional[str]:
        """Get sector for a given symbol"""
        symbol = symbol.upper()
        for sector, symbols in SECTOR_REPRESENTATIVES.items():
            if symbol in symbols:
                return sector

        # Try VN30 sectors mapping
        try:
            from src.utils.vietnam_market import VN30_SECTORS

            return VN30_SECTORS.get(symbol)
        except ImportError:
            pass

        return None

    def should_trade_symbol(self, symbol: str) -> Tuple[bool, str, float]:
        """
        Check if should trade a symbol based on sector rotation.

        Returns:
            (should_trade, reason, adjustment_factor)
            adjustment_factor: 1.2 for overweight, 0.7 for underweight, 1.0 for neutral
        """
        sector = self.get_sector_for_symbol(symbol)
        if not sector:
            return True, "Unknown sector - neutral", 1.0

        signal = self.get_rotation_signal()

        if sector in signal.overweight:
            return True, f"Sector {sector} is OVERWEIGHT - favorable", 1.2
        elif sector in signal.underweight:
            return False, f"Sector {sector} is UNDERWEIGHT - avoid", 0.7
        else:
            return True, f"Sector {sector} is NEUTRAL", 1.0


# Singleton instance
_analyzer_instance: Optional[SectorRotationAnalyzer] = None


def get_sector_rotation_analyzer() -> SectorRotationAnalyzer:
    """Get singleton instance of sector rotation analyzer"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SectorRotationAnalyzer()
    return _analyzer_instance


# =============================================================================
# NEW v7.0: LEADING INDICATOR ANALYZER
# =============================================================================


@dataclass
class LeadingIndicatorSignal:
    """Signal from leading indicator analysis"""

    sector: str
    signal_type: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float  # 0-100
    lead_time_days: int
    triggered_indicators: List[str]
    warning_signals: List[str]
    recommendation: str
    timestamp: str


class SectorLeadingIndicatorAnalyzer:
    """
    Analyze sector leading indicators for early rotation signals.

    NEW v7.0: Uses macro/micro indicators to predict sector rotation
    before price moves. Each sector has specific indicators with
    different lead times.

    Usage:
        analyzer = SectorLeadingIndicatorAnalyzer()
        signal = analyzer.analyze_sector("BANKING")
        if signal.signal_type == "BULLISH":
            print(f"Banking sector bullish in {signal.lead_time_days} days")
    """

    def __init__(self, cache_ttl_seconds: int = 3600):
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, LeadingIndicatorSignal] = {}
        self._cache_time: Optional[datetime] = None

    def analyze_sector(self, sector: str) -> Optional[LeadingIndicatorSignal]:
        """
        Analyze leading indicators for a specific sector.

        Args:
            sector: Sector name (e.g., "BANKING", "REAL_ESTATE")

        Returns:
            LeadingIndicatorSignal with prediction, or None if sector not found
        """
        sector = sector.upper()
        if sector not in SECTOR_LEADING_INDICATORS:
            logger.warning(f"Unknown sector: {sector}")
            return None

        # Check cache
        if self._is_cache_valid() and sector in self._cache:
            return self._cache[sector]

        try:
            config = SECTOR_LEADING_INDICATORS[sector]

            # Analyze each indicator
            bullish_count = 0
            bearish_count = 0
            triggered_indicators = []
            warning_signals = []

            for indicator in config["indicators"]:
                result = self._check_indicator(sector, indicator)
                if result == "BULLISH":
                    bullish_count += 1
                    triggered_indicators.append(f"✅ {indicator}")
                elif result == "BEARISH":
                    bearish_count += 1
                    triggered_indicators.append(f"❌ {indicator}")

            # Check warning signals
            for warning in config.get("warning_signals", []):
                if self._check_warning_signal(sector, warning):
                    warning_signals.append(f"⚠️ {warning}")

            # Determine overall signal
            total_indicators = len(config["indicators"])
            if bullish_count > total_indicators * 0.6:
                signal_type = "BULLISH"
                confidence = min(100, (bullish_count / total_indicators) * 100)
            elif bearish_count > total_indicators * 0.6:
                signal_type = "BEARISH"
                confidence = min(100, (bearish_count / total_indicators) * 100)
            else:
                signal_type = "NEUTRAL"
                confidence = 50

            # Reduce confidence if warning signals present
            if warning_signals:
                confidence *= 0.8

            # Generate recommendation
            recommendation = self._generate_recommendation(
                sector, signal_type, config, warning_signals
            )

            signal = LeadingIndicatorSignal(
                sector=sector,
                signal_type=signal_type,
                confidence=confidence,
                lead_time_days=config["lead_time_days"],
                triggered_indicators=triggered_indicators,
                warning_signals=warning_signals,
                recommendation=recommendation,
                timestamp=datetime.now().isoformat(),
            )

            # Cache result
            self._cache[sector] = signal
            self._cache_time = datetime.now()

            return signal

        except Exception as e:
            logger.error(f"Leading indicator analysis failed for {sector}: {e}")
            return None

    def analyze_all_sectors(self) -> Dict[str, LeadingIndicatorSignal]:
        """Analyze leading indicators for all sectors."""
        results = {}
        for sector in SECTOR_LEADING_INDICATORS.keys():
            signal = self.analyze_sector(sector)
            if signal:
                results[sector] = signal
        return results

    def get_rotation_prediction(self) -> Dict:
        """
        Get sector rotation prediction based on leading indicators.

        Returns:
            Dict with:
            - rotate_into: List of sectors to increase exposure
            - rotate_out: List of sectors to decrease exposure
            - timeline_days: Expected time for rotation
            - confidence: Overall confidence
        """
        all_signals = self.analyze_all_sectors()

        rotate_into = []
        rotate_out = []

        for sector, signal in all_signals.items():
            if signal.signal_type == "BULLISH" and signal.confidence >= 60:
                rotate_into.append(
                    {
                        "sector": sector,
                        "confidence": signal.confidence,
                        "lead_time": signal.lead_time_days,
                    }
                )
            elif signal.signal_type == "BEARISH" and signal.confidence >= 60:
                rotate_out.append(
                    {
                        "sector": sector,
                        "confidence": signal.confidence,
                        "lead_time": signal.lead_time_days,
                    }
                )

        # Sort by confidence
        rotate_into.sort(key=lambda x: x["confidence"], reverse=True)
        rotate_out.sort(key=lambda x: x["confidence"], reverse=True)

        # Calculate average timeline
        if rotate_into:
            avg_lead_time = sum(s["lead_time"] for s in rotate_into) / len(rotate_into)
        else:
            avg_lead_time = 30

        # Overall confidence
        if rotate_into or rotate_out:
            all_conf = [s["confidence"] for s in rotate_into + rotate_out]
            overall_confidence = sum(all_conf) / len(all_conf)
        else:
            overall_confidence = 30

        return {
            "rotate_into": [s["sector"] for s in rotate_into[:3]],
            "rotate_out": [s["sector"] for s in rotate_out[:3]],
            "timeline_days": int(avg_lead_time),
            "confidence": overall_confidence,
            "details": {
                "bullish_sectors": rotate_into,
                "bearish_sectors": rotate_out,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _check_indicator(self, sector: str, indicator: str) -> str:
        """
        Check a specific indicator for a sector.

        Returns: "BULLISH", "BEARISH", or "NEUTRAL"

        Note: This is a simplified implementation. In production,
        you would connect to actual data sources for each indicator.
        """
        # Placeholder implementation - returns NEUTRAL by default
        # In production, implement actual indicator checks:
        # - interest_rate: Check SBV policy rate
        # - credit_growth: Check banking system credit growth
        # - pmi_index: Check Vietnam PMI
        # etc.

        try:
            # Try to get indicator data from external sources
            indicator_value = self._get_indicator_value(sector, indicator)
            if indicator_value is None:
                return "NEUTRAL"

            # Evaluate based on indicator type
            threshold = self._get_indicator_threshold(indicator)
            if indicator_value > threshold["bullish"]:
                return "BULLISH"
            elif indicator_value < threshold["bearish"]:
                return "BEARISH"
            return "NEUTRAL"

        except Exception:
            return "NEUTRAL"

    def _get_indicator_value(self, sector: str, indicator: str) -> Optional[float]:
        """Get current value for an indicator. Override in production."""
        # Placeholder - return None to indicate no data
        # In production, connect to:
        # - SBV API for interest rates
        # - GSO for economic indicators
        # - Bloomberg/Reuters for market data
        return None

    def _get_indicator_threshold(self, indicator: str) -> Dict[str, float]:
        """Get bullish/bearish thresholds for an indicator."""
        # Default thresholds - customize per indicator
        thresholds = {
            "interest_rate": {"bullish": -0.25, "bearish": 0.25},  # Rate change
            "credit_growth": {"bullish": 12, "bearish": 8},  # % growth
            "pmi_index": {"bullish": 52, "bearish": 48},  # PMI value
            "cpi_inflation": {"bullish": 3, "bearish": 5},  # % inflation
            "oil_price": {"bullish": 70, "bearish": 50},  # USD/barrel
        }
        return thresholds.get(indicator, {"bullish": 0.5, "bearish": -0.5})

    def _check_warning_signal(self, sector: str, warning: str) -> bool:
        """Check if a warning signal is triggered."""
        # Placeholder - implement actual warning checks
        return False

    def _generate_recommendation(
        self,
        sector: str,
        signal_type: str,
        config: Dict,
        warnings: List[str],
    ) -> str:
        """Generate human-readable recommendation."""
        lead_time = config["lead_time_days"]
        best_regime = config.get("best_entry_regime", "ANY")

        if signal_type == "BULLISH":
            if warnings:
                return (
                    f"🟡 {sector}: Bullish signals but with warnings. "
                    f"Consider gradual accumulation over {lead_time} days. "
                    f"Best entry in {best_regime} regime."
                )
            return (
                f"🟢 {sector}: Strong bullish signals. "
                f"Expected move in ~{lead_time} days. "
                f"Best entry in {best_regime} regime."
            )
        elif signal_type == "BEARISH":
            return (
                f"🔴 {sector}: Bearish signals detected. "
                f"Consider reducing exposure over {lead_time} days. "
                f"Warnings: {len(warnings)}"
            )
        else:
            return (
                f"⚪ {sector}: Neutral signals. "
                f"No clear direction. Monitor for {lead_time} days."
            )

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_time is None:
            return False
        age = (datetime.now() - self._cache_time).total_seconds()
        return age < self.cache_ttl


# Singleton instance for leading indicator analyzer
_leading_indicator_instance: Optional[SectorLeadingIndicatorAnalyzer] = None


def get_leading_indicator_analyzer() -> SectorLeadingIndicatorAnalyzer:
    """Get singleton instance of leading indicator analyzer."""
    global _leading_indicator_instance
    if _leading_indicator_instance is None:
        _leading_indicator_instance = SectorLeadingIndicatorAnalyzer()
    return _leading_indicator_instance
