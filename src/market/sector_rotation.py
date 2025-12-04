# -*- coding: utf-8 -*-
"""
Sector Rotation Analysis for Vietnam Stock Market

Identifies market phase based on sector leadership patterns.
Helps determine optimal sectors for investment based on economic cycle.

Sector Rotation Theory (adapted for Vietnam):
- Early Recovery: Financials (Banks), Consumer Discretionary
- Mid Expansion: Technology, Industrials, Real Estate
- Late Expansion: Materials (Steel), Energy
- Recession: Utilities, Healthcare, Consumer Staples (Defensive)

Vietnam Market Sectors:
- Banking: VCB, BID, CTG, TCB, MBB, ACB, VPB, STB
- Real Estate: VHM, VIC, NVL, DXG, KDH, PDR
- Technology: FPT, CMG
- Retail: MWG, PNJ, DGW, FRT
- Materials: HPG, HSG, NKG, TLH
- Energy: GAS, PLX, PVD, PVS
- Utilities: POW, REE, NT2, PC1
- Food & Beverage: VNM, SAB, MSN

Usage:
    from src.market.sector_rotation import get_sector_analyzer
    
    analyzer = get_sector_analyzer()
    rotation = analyzer.analyze()
    print(f"Market phase: {rotation['phase']}")
    print(f"Leading sectors: {rotation['leading']}")
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Vietnam market sector definitions - EXPANDED v4.0
VIETNAM_SECTORS = {
    "banking": {
        "name": "Ngân hàng",
        "symbols": [
            "VCB",
            "BID",
            "CTG",
            "TCB",
            "MBB",
            "ACB",
            "VPB",
            "STB",
            "HDB",
            "TPB",
            "LPB",
            "EIB",
            "SHB",
            "MSB",
            "OCB",
        ],
        "cycle_phase": "EARLY",  # Leads in early recovery
        "defensive": False,
    },
    "real_estate": {
        "name": "Bất động sản",
        "symbols": [
            "VHM",
            "VIC",
            "NVL",
            "DXG",
            "KDH",
            "PDR",
            "DIG",
            "NLG",
            "HDG",
            "CEO",
            "IJC",
            "SCR",
            "LDG",
            "NBB",
        ],
        "cycle_phase": "MID",
        "defensive": False,
    },
    "technology": {
        "name": "Công nghệ",
        "symbols": ["FPT", "CMG", "VGI", "ELC", "ITD"],
        "cycle_phase": "MID",
        "defensive": False,
    },
    "retail": {
        "name": "Bán lẻ",
        "symbols": ["MWG", "PNJ", "DGW", "FRT", "VRE"],
        "cycle_phase": "EARLY",
        "defensive": False,
    },
    "materials": {
        "name": "Vật liệu (Thép)",
        "symbols": ["HPG", "HSG", "NKG", "TLH", "SMC", "POM", "VGS", "DTL"],
        "cycle_phase": "LATE",
        "defensive": False,
    },
    "energy": {
        "name": "Năng lượng",
        "symbols": ["GAS", "PLX", "PVD", "PVS", "BSR", "PVT", "OIL", "PVC"],
        "cycle_phase": "LATE",
        "defensive": False,
    },
    "utilities": {
        "name": "Tiện ích",
        "symbols": ["POW", "REE", "NT2", "PC1", "PPC", "GEG", "BCG", "HDC"],
        "cycle_phase": "RECESSION",
        "defensive": True,
    },
    "food_beverage": {
        "name": "Thực phẩm & Đồ uống",
        "symbols": ["VNM", "SAB", "MSN", "QNS", "KDC", "MCH", "VHC", "ANV"],
        "cycle_phase": "RECESSION",
        "defensive": True,
    },
    "healthcare": {
        "name": "Y tế",
        "symbols": ["DHG", "DMC", "IMP", "DBD", "TRA", "PME", "DCL"],
        "cycle_phase": "RECESSION",
        "defensive": True,
    },
    "securities": {
        "name": "Chứng khoán",
        "symbols": ["SSI", "VND", "HCM", "VCI", "SHS", "MBS", "VIX", "BSI", "CTS", "FTS"],
        "cycle_phase": "EARLY",
        "defensive": False,
    },
    "manufacturing": {
        "name": "Sản xuất",
        "symbols": ["GEX", "HAX", "GMD", "TCM", "TNG", "VGT", "STK"],
        "cycle_phase": "MID",
        "defensive": False,
    },
    "agriculture": {
        "name": "Nông nghiệp",
        "symbols": ["HAG", "HNG", "BAF", "DBC", "PAN", "LTG"],
        "cycle_phase": "RECESSION",
        "defensive": True,
    },
}

# Reverse mapping: symbol -> sector_id for quick lookup
VN_SYMBOL_TO_SECTOR = {}
for sector_id, sector_info in VIETNAM_SECTORS.items():
    for symbol in sector_info["symbols"]:
        VN_SYMBOL_TO_SECTOR[symbol] = sector_id


@dataclass
class SectorPerformance:
    """Performance metrics for a sector"""

    sector_id: str
    name: str
    return_1w: float  # 1-week return
    return_1m: float  # 1-month return
    return_3m: float  # 3-month return
    relative_strength: float  # vs VNINDEX
    momentum_score: float  # Combined momentum
    is_leading: bool
    is_lagging: bool


@dataclass
class SectorRotationResult:
    """Result of sector rotation analysis"""

    date: str
    phase: str  # EARLY, MID, LATE, RECESSION, UNKNOWN
    confidence: float  # 0-100
    leading_sectors: List[str]
    lagging_sectors: List[str]
    sector_performances: Dict[str, SectorPerformance]
    recommendation: str
    score: float  # -1 to +1 for regime detection


class SectorRotationAnalyzer:
    """
    Analyze sector rotation to identify market phase.

    Methodology:
    1. Calculate relative strength of each sector vs VNINDEX
    2. Identify leading and lagging sectors
    3. Match pattern to economic cycle phase
    4. Generate investment recommendations
    """

    def __init__(
        self,
        lookback_weeks: int = 4,
        leading_threshold: float = 0.05,  # 5% outperformance = leading
        lagging_threshold: float = -0.05,  # 5% underperformance = lagging
        cache_ttl_seconds: int = 3600,  # 1 hour cache
    ):
        self.lookback_weeks = lookback_weeks
        self.leading_threshold = leading_threshold
        self.lagging_threshold = lagging_threshold
        self.cache_ttl = cache_ttl_seconds

        # Cache
        self._cache: Optional[SectorRotationResult] = None
        self._cache_time: Optional[datetime] = None

        # Sector data cache
        self._sector_data: Dict[str, pd.DataFrame] = {}

    def analyze(self, vnindex_df: Optional[pd.DataFrame] = None) -> SectorRotationResult:
        """
        Analyze current sector rotation.

        Args:
            vnindex_df: VNINDEX data for relative strength calculation

        Returns:
            SectorRotationResult with analysis
        """
        # Check cache
        if self._is_cache_valid():
            return self._cache

        try:
            # Calculate sector performances
            performances = self._calculate_sector_performances(vnindex_df)

            # Identify leading/lagging
            leading = [s for s, p in performances.items() if p.is_leading]
            lagging = [s for s, p in performances.items() if p.is_lagging]

            # Determine market phase
            phase, confidence = self._determine_phase(leading, lagging)

            # Generate recommendation
            recommendation = self._generate_recommendation(phase, leading)

            # Calculate score for regime detection
            score = self._calculate_rotation_score(performances, phase)

            result = SectorRotationResult(
                date=datetime.now().isoformat(),
                phase=phase,
                confidence=confidence,
                leading_sectors=leading,
                lagging_sectors=lagging,
                sector_performances=performances,
                recommendation=recommendation,
                score=score,
            )

            # Update cache
            self._cache = result
            self._cache_time = datetime.now()

            logger.info(
                f"📊 Sector Rotation: Phase={phase} ({confidence:.0f}% conf), "
                f"Leading: {leading}, Lagging: {lagging}"
            )

            return result

        except Exception as e:
            logger.error(f"Sector rotation analysis failed: {e}", exc_info=True)
            return self._default_result(f"Error: {str(e)}")

    def _calculate_sector_performances(
        self, vnindex_df: Optional[pd.DataFrame]
    ) -> Dict[str, SectorPerformance]:
        """Calculate performance metrics for each sector"""

        performances = {}

        # Get VNINDEX returns for relative strength
        vnindex_return_1m = 0.0
        if vnindex_df is not None and len(vnindex_df) >= 20:
            vnindex_return_1m = vnindex_df["close"].iloc[-1] / vnindex_df["close"].iloc[-20] - 1

        for sector_id, sector_info in VIETNAM_SECTORS.items():
            try:
                # Calculate sector return (average of constituent stocks)
                # PLACEHOLDER: Would fetch real data in production
                sector_return_1m = self._get_sector_return(sector_id, days=20)
                sector_return_1w = self._get_sector_return(sector_id, days=5)
                sector_return_3m = self._get_sector_return(sector_id, days=60)

                # Relative strength vs VNINDEX
                relative_strength = sector_return_1m - vnindex_return_1m

                # Momentum score (weighted average of returns)
                momentum = sector_return_1w * 0.4 + sector_return_1m * 0.4 + sector_return_3m * 0.2

                # Determine if leading/lagging
                is_leading = relative_strength >= self.leading_threshold
                is_lagging = relative_strength <= self.lagging_threshold

                performances[sector_id] = SectorPerformance(
                    sector_id=sector_id,
                    name=sector_info["name"],
                    return_1w=sector_return_1w,
                    return_1m=sector_return_1m,
                    return_3m=sector_return_3m,
                    relative_strength=relative_strength,
                    momentum_score=momentum,
                    is_leading=is_leading,
                    is_lagging=is_lagging,
                )

            except Exception as e:
                logger.warning(f"Failed to calculate {sector_id} performance: {e}")
                continue

        return performances

    def _get_sector_return(self, sector_id: str, days: int) -> float:
        """
        Get sector return over specified days.

        Calculates equal-weighted average return of sector constituents.
        Uses cached data when available for performance.

        Args:
            sector_id: Sector identifier (e.g., 'banking', 'real_estate')
            days: Number of trading days to calculate return

        Returns:
            Average return as decimal (e.g., 0.05 = 5%)
        """
        if sector_id not in VIETNAM_SECTORS:
            return 0.0

        symbols = VIETNAM_SECTORS[sector_id]["symbols"]
        if not symbols:
            return 0.0

        returns = []

        for symbol in symbols:
            try:
                # Try to get data from cache first
                if symbol in self._sector_data:
                    df = self._sector_data[symbol]
                else:
                    # Fetch from data loader
                    try:
                        from src.data.loader import load_data

                        df = load_data(symbol, lookback=days + 10, use_cache=True)
                        if df is not None and not df.empty:
                            self._sector_data[symbol] = df
                    except ImportError:
                        logger.debug(f"Data loader not available for {symbol}")
                        continue
                    except Exception as e:
                        logger.debug(f"Failed to load data for {symbol}: {e}")
                        continue

                if df is None or len(df) < days:
                    continue

                # Calculate return
                df_period = df.tail(days)
                if len(df_period) >= 2:
                    start_price = df_period["close"].iloc[0]
                    end_price = df_period["close"].iloc[-1]
                    if start_price > 0:
                        stock_return = (end_price - start_price) / start_price
                        returns.append(stock_return)

            except Exception as e:
                logger.debug(f"Error calculating return for {symbol}: {e}")
                continue

        if not returns:
            return 0.0

        # Return equal-weighted average
        return sum(returns) / len(returns)

    def _determine_phase(self, leading: List[str], lagging: List[str]) -> Tuple[str, float]:
        """
        Determine market phase based on sector leadership.

        Returns:
            (phase, confidence)
        """
        # Count sectors by cycle phase
        phase_scores = {
            "EARLY": 0,
            "MID": 0,
            "LATE": 0,
            "RECESSION": 0,
        }

        for sector_id in leading:
            if sector_id in VIETNAM_SECTORS:
                phase = VIETNAM_SECTORS[sector_id]["cycle_phase"]
                phase_scores[phase] += 2  # Leading = +2

        for sector_id in lagging:
            if sector_id in VIETNAM_SECTORS:
                phase = VIETNAM_SECTORS[sector_id]["cycle_phase"]
                phase_scores[phase] -= 1  # Lagging = -1

        # Find dominant phase
        if not any(phase_scores.values()):
            return "UNKNOWN", 30.0

        max_phase = max(phase_scores, key=phase_scores.get)
        max_score = phase_scores[max_phase]
        total_score = sum(abs(s) for s in phase_scores.values())

        confidence = (max_score / total_score * 100) if total_score > 0 else 30.0
        confidence = min(confidence, 90.0)  # Cap at 90%

        return max_phase, confidence

    def _generate_recommendation(self, phase: str, leading: List[str]) -> str:
        """Generate investment recommendation based on phase"""

        recommendations = {
            "EARLY": (
                "🟢 Early Recovery: Focus on Banking, Retail sectors. "
                "Consider cyclical stocks with strong balance sheets."
            ),
            "MID": (
                "🟡 Mid Expansion: Technology, Real Estate leading. "
                "Growth stocks favored, but watch for overvaluation."
            ),
            "LATE": (
                "🟠 Late Expansion: Materials, Energy leading. "
                "Consider taking profits on growth, rotate to value."
            ),
            "RECESSION": (
                "🔴 Defensive Phase: Utilities, Healthcare, F&B leading. "
                "Reduce exposure, focus on dividend stocks and cash."
            ),
            "UNKNOWN": (
                "⚪ Unclear Phase: Mixed signals. "
                "Maintain balanced portfolio, avoid aggressive positions."
            ),
        }

        base_rec = recommendations.get(phase, recommendations["UNKNOWN"])

        if leading:
            leading_names = [VIETNAM_SECTORS.get(s, {}).get("name", s) for s in leading[:3]]
            base_rec += f"\n   Leading: {', '.join(leading_names)}"

        return base_rec

    def _calculate_rotation_score(
        self, performances: Dict[str, SectorPerformance], phase: str
    ) -> float:
        """
        Calculate rotation score for regime detection.

        Score interpretation:
        - +1.0: Strong risk-on (cyclicals leading)
        - 0.0: Neutral
        - -1.0: Strong risk-off (defensives leading)
        """
        if not performances:
            return 0.0

        # Calculate weighted score based on sector type
        cyclical_score = 0.0
        defensive_score = 0.0

        for sector_id, perf in performances.items():
            if sector_id not in VIETNAM_SECTORS:
                continue

            is_defensive = VIETNAM_SECTORS[sector_id]["defensive"]

            if perf.is_leading:
                if is_defensive:
                    defensive_score += 1
                else:
                    cyclical_score += 1
            elif perf.is_lagging:
                if is_defensive:
                    cyclical_score += 0.5  # Defensive lagging = bullish
                else:
                    defensive_score += 0.5  # Cyclical lagging = bearish

        total = cyclical_score + defensive_score
        if total == 0:
            return 0.0

        # Score: positive = risk-on, negative = risk-off
        score = (cyclical_score - defensive_score) / total
        return np.clip(score, -1.0, 1.0)

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if self._cache is None or self._cache_time is None:
            return False

        age = (datetime.now() - self._cache_time).total_seconds()
        return age < self.cache_ttl

    def _default_result(self, reason: str) -> SectorRotationResult:
        """Return default neutral result"""
        return SectorRotationResult(
            date=datetime.now().isoformat(),
            phase="UNKNOWN",
            confidence=0.0,
            leading_sectors=[],
            lagging_sectors=[],
            sector_performances={},
            recommendation=f"Analysis unavailable: {reason}",
            score=0.0,
        )

    def get_sector_symbols(self, sector_id: str) -> List[str]:
        """Get list of symbols for a sector"""
        return VIETNAM_SECTORS.get(sector_id, {}).get("symbols", [])

    def get_all_sectors(self) -> Dict:
        """Get all sector definitions"""
        return VIETNAM_SECTORS

    def get_symbol_sector(self, symbol: str) -> str:
        """
        Get sector ID for a given symbol.

        Args:
            symbol: Stock symbol (e.g., 'VCB', 'FPT')

        Returns:
            Sector ID (e.g., 'banking', 'technology') or 'unknown'
        """
        return VN_SYMBOL_TO_SECTOR.get(symbol, "unknown")

    def get_symbol_sector_info(self, symbol: str) -> Dict:
        """
        Get full sector information for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with sector_id, name, cycle_phase, defensive, is_leading, is_lagging
        """
        sector_id = self.get_symbol_sector(symbol)
        if sector_id == "unknown":
            return {
                "sector_id": "unknown",
                "name": "Unknown",
                "cycle_phase": "UNKNOWN",
                "defensive": False,
                "is_leading": False,
                "is_lagging": False,
            }

        sector_info = VIETNAM_SECTORS.get(sector_id, {})

        # Get current rotation status
        rotation = self._cache if self._cache else self.analyze()
        is_leading = sector_id in rotation.leading_sectors
        is_lagging = sector_id in rotation.lagging_sectors

        return {
            "sector_id": sector_id,
            "name": sector_info.get("name", "Unknown"),
            "cycle_phase": sector_info.get("cycle_phase", "UNKNOWN"),
            "defensive": sector_info.get("defensive", False),
            "is_leading": is_leading,
            "is_lagging": is_lagging,
        }

    def is_sector_leading(self, sector_id: str) -> bool:
        """Check if a sector is currently leading"""
        rotation = self._cache if self._cache else self.analyze()
        return sector_id in rotation.leading_sectors

    def is_sector_lagging(self, sector_id: str) -> bool:
        """Check if a sector is currently lagging"""
        rotation = self._cache if self._cache else self.analyze()
        return sector_id in rotation.lagging_sectors

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._cache = None
        self._cache_time = None
        self._sector_data.clear()
        logger.info("Sector rotation cache cleared")


# Singleton instance
_analyzer_instance: Optional[SectorRotationAnalyzer] = None


def get_sector_analyzer() -> SectorRotationAnalyzer:
    """Get singleton instance of sector rotation analyzer"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SectorRotationAnalyzer()
    return _analyzer_instance
