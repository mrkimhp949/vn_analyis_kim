# -*- coding: utf-8 -*-
"""
Sector Momentum Strategy - Advanced Rotation Logic for Vietnam Market

This module implements a comprehensive sector rotation strategy based on:
1. Sector momentum scoring and ranking
2. Economic cycle phase detection
3. Leading indicator analysis
4. Foreign flow by sector
5. Relative strength comparison

Key Features:
- Automatic sector rotation signals (overweight/underweight)
- Economic cycle mapping to optimal sectors
- Position sizing adjustment by sector strength
- Integration with entry/exit logic

Vietnam Market Sector Cycles:
- EARLY_RECOVERY: Banking, Securities, Industrial
- MID_CYCLE: Technology, Consumer, Real Estate
- LATE_CYCLE: Energy, Utilities, Consumer (defensive)
- RECESSION: Utilities, Consumer staples

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import RLock
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Sector allocation by economic cycle
CYCLE_SECTOR_ALLOCATION = {
    "EARLY_RECOVERY": {
        "overweight": ["BANKING", "SECURITIES", "INDUSTRIAL"],
        "neutral": ["TECHNOLOGY", "ENERGY"],
        "underweight": ["UTILITIES", "CONSUMER"],
        "description": "Recovery phase - favor cyclicals and financials",
    },
    "MID_CYCLE": {
        "overweight": ["TECHNOLOGY", "CONSUMER", "REAL_ESTATE"],
        "neutral": ["BANKING", "INDUSTRIAL"],
        "underweight": ["SECURITIES", "UTILITIES"],
        "description": "Expansion phase - favor growth sectors",
    },
    "LATE_CYCLE": {
        "overweight": ["ENERGY", "UTILITIES", "CONSUMER"],
        "neutral": ["TECHNOLOGY"],
        "underweight": ["BANKING", "REAL_ESTATE", "SECURITIES"],
        "description": "Late cycle - favor defensives and commodities",
    },
    "RECESSION": {
        "overweight": ["UTILITIES", "CONSUMER"],
        "neutral": ["TECHNOLOGY"],
        "underweight": ["BANKING", "REAL_ESTATE", "SECURITIES", "INDUSTRIAL"],
        "description": "Recession - favor defensives, avoid cyclicals",
    },
}

# Momentum scoring weights
MOMENTUM_WEIGHTS = {
    "return_1w": 0.25,  # 1-week return
    "return_1m": 0.35,  # 1-month return
    "return_3m": 0.25,  # 3-month return
    "volume_trend": 0.15,  # Volume momentum
}

# Thresholds
STRONG_MOMENTUM_THRESHOLD = 0.6
WEAK_MOMENTUM_THRESHOLD = -0.3
MIN_STOCKS_FOR_SECTOR = 2


class SectorSignal(Enum):
    """Sector trading signal"""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    SELL = "SELL"


@dataclass
class SectorMomentumScore:
    """Detailed sector momentum analysis"""

    sector: str
    momentum_score: float  # -1 to +1
    rank: int  # 1 = best performing sector

    # Component scores
    return_1w: float
    return_1m: float
    return_3m: float
    volume_momentum: float

    # Relative analysis
    relative_strength_vs_vnindex: float
    relative_strength_vs_vn30: float

    # Foreign flow
    foreign_net_buy_value: float  # VND
    foreign_flow_trend: str  # "BUYING", "SELLING", "NEUTRAL"

    # Signal
    signal: SectorSignal
    confidence: int  # 0-100

    # Metadata
    num_stocks_analyzed: int
    top_performers: List[str]
    worst_performers: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RotationRecommendation:
    """Sector rotation recommendation"""

    overweight_sectors: List[str]
    underweight_sectors: List[str]
    neutral_sectors: List[str]

    # Position adjustments
    sector_multipliers: Dict[str, float]  # Sector -> position size multiplier

    # Top picks per sector
    top_picks: Dict[str, List[str]]  # Sector -> [symbols]

    # Cycle info
    detected_cycle: str
    cycle_confidence: int

    # Overall recommendation
    overall_sentiment: str  # "RISK_ON", "RISK_OFF", "NEUTRAL"

    timestamp: datetime = field(default_factory=datetime.now)


class SectorMomentumStrategy:
    """
    Advanced sector momentum and rotation strategy.

    Usage:
        strategy = SectorMomentumStrategy()

        # Get sector rankings
        rankings = strategy.get_sector_rankings(sector_data)

        # Get rotation recommendation
        recommendation = strategy.get_rotation_recommendation(
            sector_data, economic_indicators
        )

        # Apply to position sizing
        multiplier = strategy.get_position_multiplier("VNM", recommendation)
    """

    # Sector representatives for VN market
    SECTOR_STOCKS = {
        "BANKING": ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "HDB", "STB", "TPB"],
        "REAL_ESTATE": ["VHM", "VIC", "NVL", "VRE", "KDH", "DXG", "PDR", "NLG"],
        "TECHNOLOGY": ["FPT", "CMG", "ELC", "SAM"],
        "CONSUMER": ["VNM", "MSN", "MWG", "SAB", "PNJ", "VRE"],
        "ENERGY": ["GAS", "PLX", "PVD", "PVS", "POW"],
        "INDUSTRIAL": ["HPG", "HSG", "NKG", "GVR", "HT1", "HBC"],
        "SECURITIES": ["SSI", "VCI", "HCM", "VND", "SHS", "MBS"],
        "UTILITIES": ["POW", "NT2", "PPC", "REE"],
        "AVIATION": ["VJC", "HVN", "ACV"],
        "INSURANCE": ["BVH", "PVI", "BMI"],
    }

    def __init__(
        self,
        momentum_lookback_days: int = 60,
        rebalance_frequency_days: int = 7,
        min_sector_stocks: int = 2,
    ):
        self.momentum_lookback_days = momentum_lookback_days
        self.rebalance_frequency_days = rebalance_frequency_days
        self.min_sector_stocks = min_sector_stocks

        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        self._lock = RLock()

        logger.info("SectorMomentumStrategy initialized")

    def calculate_sector_momentum(
        self,
        sector: str,
        stock_data: Dict[str, pd.DataFrame],
        vnindex_data: Optional[pd.DataFrame] = None,
    ) -> SectorMomentumScore:
        """
        Calculate momentum score for a sector.

        Args:
            sector: Sector name
            stock_data: Dict of symbol -> OHLCV DataFrame
            vnindex_data: VNINDEX data for relative strength

        Returns:
            SectorMomentumScore with detailed analysis
        """
        sector_stocks = self.SECTOR_STOCKS.get(sector, [])
        available_stocks = [s for s in sector_stocks if s in stock_data]

        if len(available_stocks) < self.min_sector_stocks:
            return SectorMomentumScore(
                sector=sector,
                momentum_score=0.0,
                rank=99,
                return_1w=0.0,
                return_1m=0.0,
                return_3m=0.0,
                volume_momentum=0.0,
                relative_strength_vs_vnindex=0.0,
                relative_strength_vs_vn30=0.0,
                foreign_net_buy_value=0.0,
                foreign_flow_trend="NEUTRAL",
                signal=SectorSignal.HOLD,
                confidence=0,
                num_stocks_analyzed=0,
                top_performers=[],
                worst_performers=[],
            )

        # Calculate returns for each stock
        stock_returns = {}
        stock_volumes = {}

        for symbol in available_stocks:
            df = stock_data[symbol]
            if len(df) < 20:
                continue

            try:
                # Calculate returns
                close = df["close"].values
                volume = df["volume"].values if "volume" in df.columns else None

                ret_1w = (close[-1] / close[-5] - 1) if len(close) >= 5 else 0
                ret_1m = (close[-1] / close[-20] - 1) if len(close) >= 20 else 0
                ret_3m = (close[-1] / close[-60] - 1) if len(close) >= 60 else ret_1m

                stock_returns[symbol] = {
                    "1w": ret_1w,
                    "1m": ret_1m,
                    "3m": ret_3m,
                }

                # Volume momentum
                if volume is not None and len(volume) >= 20:
                    recent_vol = np.mean(volume[-5:])
                    avg_vol = np.mean(volume[-20:])
                    stock_volumes[symbol] = recent_vol / avg_vol if avg_vol > 0 else 1.0

            except Exception as e:
                logger.debug(f"Error calculating returns for {symbol}: {e}")
                continue

        if not stock_returns:
            return SectorMomentumScore(
                sector=sector,
                momentum_score=0.0,
                rank=99,
                return_1w=0.0,
                return_1m=0.0,
                return_3m=0.0,
                volume_momentum=0.0,
                relative_strength_vs_vnindex=0.0,
                relative_strength_vs_vn30=0.0,
                foreign_net_buy_value=0.0,
                foreign_flow_trend="NEUTRAL",
                signal=SectorSignal.HOLD,
                confidence=0,
                num_stocks_analyzed=0,
                top_performers=[],
                worst_performers=[],
            )

        # Calculate sector averages
        avg_1w = np.mean([r["1w"] for r in stock_returns.values()])
        avg_1m = np.mean([r["1m"] for r in stock_returns.values()])
        avg_3m = np.mean([r["3m"] for r in stock_returns.values()])
        avg_volume = np.mean(list(stock_volumes.values())) if stock_volumes else 1.0

        # Calculate relative strength vs VNINDEX
        rs_vnindex = 0.0
        if vnindex_data is not None and len(vnindex_data) >= 20:
            vnindex_ret_1m = vnindex_data["close"].iloc[-1] / vnindex_data["close"].iloc[-20] - 1
            rs_vnindex = avg_1m - vnindex_ret_1m

        # Calculate momentum score
        momentum_score = (
            avg_1w * MOMENTUM_WEIGHTS["return_1w"]
            + avg_1m * MOMENTUM_WEIGHTS["return_1m"]
            + avg_3m * MOMENTUM_WEIGHTS["return_3m"]
            + (avg_volume - 1) * MOMENTUM_WEIGHTS["volume_trend"]
        )

        # Normalize to -1 to +1 range
        momentum_score = np.clip(momentum_score * 5, -1, 1)

        # Determine signal
        if momentum_score >= STRONG_MOMENTUM_THRESHOLD:
            signal = SectorSignal.STRONG_BUY
            confidence = min(90, int(50 + momentum_score * 40))
        elif momentum_score >= 0.3:
            signal = SectorSignal.BUY
            confidence = min(75, int(40 + momentum_score * 35))
        elif momentum_score >= WEAK_MOMENTUM_THRESHOLD:
            signal = SectorSignal.HOLD
            confidence = 50
        elif momentum_score >= -0.6:
            signal = SectorSignal.REDUCE
            confidence = min(60, int(40 + abs(momentum_score) * 20))
        else:
            signal = SectorSignal.SELL
            confidence = min(80, int(50 + abs(momentum_score) * 30))

        # Sort stocks by 1-month return
        sorted_stocks = sorted(stock_returns.items(), key=lambda x: x[1]["1m"], reverse=True)
        top_performers = [s[0] for s in sorted_stocks[:3]]
        worst_performers = [s[0] for s in sorted_stocks[-3:]]

        # Volume trend
        if avg_volume > 1.2:
            volume_trend = "INCREASING"
        elif avg_volume < 0.8:
            volume_trend = "DECREASING"
        else:
            volume_trend = "STABLE"

        return SectorMomentumScore(
            sector=sector,
            momentum_score=momentum_score,
            rank=0,  # Will be set in ranking
            return_1w=avg_1w,
            return_1m=avg_1m,
            return_3m=avg_3m,
            volume_momentum=avg_volume,
            relative_strength_vs_vnindex=rs_vnindex,
            relative_strength_vs_vn30=rs_vnindex,  # Use same for now
            foreign_net_buy_value=0.0,  # TODO: integrate with foreign flow
            foreign_flow_trend="NEUTRAL",
            signal=signal,
            confidence=confidence,
            num_stocks_analyzed=len(stock_returns),
            top_performers=top_performers,
            worst_performers=worst_performers,
        )

    def get_sector_rankings(
        self,
        stock_data: Dict[str, pd.DataFrame],
        vnindex_data: Optional[pd.DataFrame] = None,
    ) -> List[SectorMomentumScore]:
        """
        Get ranked list of sectors by momentum.

        Args:
            stock_data: Dict of symbol -> OHLCV DataFrame
            vnindex_data: VNINDEX data for relative strength

        Returns:
            List of SectorMomentumScore, sorted by momentum (best first)
        """
        scores = []

        for sector in self.SECTOR_STOCKS.keys():
            score = self.calculate_sector_momentum(sector, stock_data, vnindex_data)
            if score.num_stocks_analyzed > 0:
                scores.append(score)

        # Sort by momentum score
        scores.sort(key=lambda x: x.momentum_score, reverse=True)

        # Assign ranks
        for i, score in enumerate(scores):
            score.rank = i + 1

        return scores

    def detect_economic_cycle(
        self,
        market_data: Dict[str, Any],
    ) -> Tuple[str, int]:
        """
        Detect current economic cycle phase.

        Args:
            market_data: Dict with economic indicators

        Returns:
            (cycle_phase, confidence)
        """
        # Simple heuristic based on available data
        # In production, this would use proper economic indicators

        vnindex_change_3m = market_data.get("vnindex_change_3m", 0)
        volume_trend = market_data.get("volume_trend", "STABLE")
        credit_growth = market_data.get("credit_growth", 0.10)

        # Early recovery: market bottoming, volume increasing
        if vnindex_change_3m < -0.10 and volume_trend == "INCREASING":
            return "EARLY_RECOVERY", 60

        # Mid cycle: market rising, moderate volume
        if 0 < vnindex_change_3m < 0.20 and volume_trend in ["STABLE", "INCREASING"]:
            return "MID_CYCLE", 65

        # Late cycle: market extended, volume decreasing
        if vnindex_change_3m > 0.15 and volume_trend == "DECREASING":
            return "LATE_CYCLE", 55

        # Recession: market falling, volume spiking
        if vnindex_change_3m < -0.15:
            return "RECESSION", 60

        return "MID_CYCLE", 50  # Default

    def get_rotation_recommendation(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: Dict[str, Any],
        vnindex_data: Optional[pd.DataFrame] = None,
    ) -> RotationRecommendation:
        """
        Get comprehensive rotation recommendation.

        Args:
            stock_data: Dict of symbol -> OHLCV DataFrame
            market_data: Economic indicators and market stats
            vnindex_data: VNINDEX data

        Returns:
            RotationRecommendation with actionable signals
        """
        # Get sector rankings
        rankings = self.get_sector_rankings(stock_data, vnindex_data)

        # Detect economic cycle
        cycle, cycle_confidence = self.detect_economic_cycle(market_data)
        cycle_allocation = CYCLE_SECTOR_ALLOCATION.get(cycle, CYCLE_SECTOR_ALLOCATION["MID_CYCLE"])

        # Combine momentum with cycle allocation
        overweight = []
        underweight = []
        neutral = []
        sector_multipliers = {}
        top_picks = {}

        for score in rankings:
            sector = score.sector

            # Get cycle recommendation
            if sector in cycle_allocation["overweight"]:
                cycle_bias = 0.2
            elif sector in cycle_allocation["underweight"]:
                cycle_bias = -0.2
            else:
                cycle_bias = 0

            # Combine momentum + cycle
            combined_score = score.momentum_score + cycle_bias

            if combined_score >= 0.3:
                overweight.append(sector)
                sector_multipliers[sector] = 1.2 + min(0.3, combined_score * 0.3)
            elif combined_score <= -0.2:
                underweight.append(sector)
                sector_multipliers[sector] = 0.5 + max(0, (combined_score + 1) * 0.3)
            else:
                neutral.append(sector)
                sector_multipliers[sector] = 1.0

            top_picks[sector] = score.top_performers

        # Determine overall sentiment
        if len(overweight) > len(underweight) + 2:
            overall_sentiment = "RISK_ON"
        elif len(underweight) > len(overweight) + 2:
            overall_sentiment = "RISK_OFF"
        else:
            overall_sentiment = "NEUTRAL"

        return RotationRecommendation(
            overweight_sectors=overweight,
            underweight_sectors=underweight,
            neutral_sectors=neutral,
            sector_multipliers=sector_multipliers,
            top_picks=top_picks,
            detected_cycle=cycle,
            cycle_confidence=cycle_confidence,
            overall_sentiment=overall_sentiment,
        )

    def get_position_multiplier(
        self,
        symbol: str,
        recommendation: RotationRecommendation,
    ) -> float:
        """
        Get position size multiplier for a symbol based on sector rotation.

        Args:
            symbol: Stock symbol
            recommendation: Current rotation recommendation

        Returns:
            Position size multiplier (0.5 to 1.5)
        """
        # Find symbol's sector
        symbol_sector = None
        for sector, stocks in self.SECTOR_STOCKS.items():
            if symbol.upper() in stocks:
                symbol_sector = sector
                break

        if symbol_sector is None:
            return 1.0  # Neutral if sector unknown

        return recommendation.sector_multipliers.get(symbol_sector, 1.0)

    def get_top_sector_picks(
        self,
        recommendation: RotationRecommendation,
        max_picks: int = 5,
    ) -> List[str]:
        """
        Get top stock picks from overweight sectors.

        Args:
            recommendation: Current rotation recommendation
            max_picks: Maximum number of picks

        Returns:
            List of stock symbols
        """
        picks = []
        for sector in recommendation.overweight_sectors:
            sector_picks = recommendation.top_picks.get(sector, [])
            picks.extend(sector_picks[:2])  # Top 2 from each sector

            if len(picks) >= max_picks:
                break

        return picks[:max_picks]


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_sector_momentum_strategy: Optional[SectorMomentumStrategy] = None
_strategy_lock = RLock()


def get_sector_momentum_strategy() -> SectorMomentumStrategy:
    """Get singleton instance of SectorMomentumStrategy."""
    global _sector_momentum_strategy

    with _strategy_lock:
        if _sector_momentum_strategy is None:
            _sector_momentum_strategy = SectorMomentumStrategy()
        return _sector_momentum_strategy


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================


def get_sector_for_symbol(symbol: str) -> Optional[str]:
    """Get sector for a given symbol."""
    strategy = get_sector_momentum_strategy()
    for sector, stocks in strategy.SECTOR_STOCKS.items():
        if symbol.upper() in stocks:
            return sector
    return None


def is_in_overweight_sector(
    symbol: str,
    recommendation: RotationRecommendation,
) -> bool:
    """Check if symbol is in an overweight sector."""
    sector = get_sector_for_symbol(symbol)
    return sector in recommendation.overweight_sectors if sector else False


def is_in_underweight_sector(
    symbol: str,
    recommendation: RotationRecommendation,
) -> bool:
    """Check if symbol is in an underweight sector."""
    sector = get_sector_for_symbol(symbol)
    return sector in recommendation.underweight_sectors if sector else False
