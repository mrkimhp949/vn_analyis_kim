# -*- coding: utf-8 -*-
"""
Multi-Timeframe Confirmation Module
Phân tích xu hướng trên nhiều khung thời gian để giảm false negatives
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TimeframeTrend(Enum):
    """Xu hướng trên từng khung thời gian"""

    STRONG_UP = 3
    MODERATE_UP = 2
    WEAK_UP = 1
    NEUTRAL = 0
    WEAK_DOWN = -1
    MODERATE_DOWN = -2
    STRONG_DOWN = -3


@dataclass
class MultiTimeframeAnalysis:
    """Kết quả phân tích multi-timeframe"""

    daily_trend: TimeframeTrend
    daily_score: float
    weekly_trend: TimeframeTrend
    weekly_score: float
    four_hour_trend: Optional[TimeframeTrend] = None
    four_hour_score: Optional[float] = None

    alignment_score: float = 0.0  # -100 to +100
    is_aligned: bool = False
    confidence_adjustment: int = 0  # -20 to +20
    reasons: list = None
    warnings: list = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.warnings is None:
            self.warnings = []


class MultiTimeframeAnalyzer:
    """
    Phân tích xu hướng trên nhiều khung thời gian

    IMPROVEMENT v2.0:
    - Adds 4-hour timeframe (intraday momentum)
    - Weighted scoring system (daily > 4H > weekly for Vietnam market)
    - Adaptive alignment thresholds based on market regime
    - Confidence adjustments instead of hard blocks

    Vietnam market characteristics:
    - Shorter price cycles than US → Daily matters more than weekly
    - Mean reversion faster → 4H can catch early reversals
    - Lower volumes → Need flexible thresholds
    """

    def __init__(
        self,
        daily_lookback: int = 20,  # Daily bars for trend calc
        weekly_lookback: int = 10,  # Weekly bars (= 50 trading days)
        four_hour_lookback: int = 30,  # 4H bars (= 5 trading days)
        alignment_threshold: float = 50.0,  # Min score for "aligned"
        use_four_hour: bool = False,  # Enable when intraday data available
        # Weights for composite score (must sum to 1.0)
        daily_weight: float = 0.50,  # Daily most important in VN
        weekly_weight: float = 0.30,  # Weekly secondary
        four_hour_weight: float = 0.20,  # 4H tertiary (if enabled)
    ):
        """
        Args:
            daily_lookback: Number of daily bars for trend analysis
            weekly_lookback: Number of weekly bars
            four_hour_lookback: Number of 4-hour bars (if available)
            alignment_threshold: Minimum alignment score to consider "aligned"
            use_four_hour: Enable 4H analysis (requires intraday data)
            daily_weight: Weight for daily trend (0-1)
            weekly_weight: Weight for weekly trend (0-1)
            four_hour_weight: Weight for 4H trend (0-1)
        """
        self.daily_lookback = daily_lookback
        self.weekly_lookback = weekly_lookback
        self.four_hour_lookback = four_hour_lookback
        self.alignment_threshold = alignment_threshold
        self.use_four_hour = use_four_hour

        # Normalize weights
        total_weight = daily_weight + weekly_weight + (four_hour_weight if use_four_hour else 0)
        self.daily_weight = daily_weight / total_weight
        self.weekly_weight = weekly_weight / total_weight
        self.four_hour_weight = four_hour_weight / total_weight if use_four_hour else 0.0

        logger.info(
            f"MultiTimeframeAnalyzer initialized: "
            f"weights=[daily:{self.daily_weight:.2f}, weekly:{self.weekly_weight:.2f}, "
            f"4H:{self.four_hour_weight:.2f}], alignment_threshold={alignment_threshold}"
        )

    def analyze(
        self,
        df_daily: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None,
        market_regime: Optional[Dict] = None,
    ) -> MultiTimeframeAnalysis:
        """
        Phân tích xu hướng trên nhiều timeframes

        Args:
            df_daily: Daily OHLCV data with indicators
            df_4h: 4-hour OHLCV data (optional)
            market_regime: Market regime info for adaptive thresholds

        Returns:
            MultiTimeframeAnalysis object
        """
        reasons = []
        warnings = []

        # 1. DAILY TIMEFRAME ANALYSIS
        daily_trend, daily_score = self._analyze_timeframe(
            df_daily, lookback=self.daily_lookback, timeframe_name="Daily"
        )

        # 2. WEEKLY TIMEFRAME ANALYSIS (resample from daily)
        df_weekly = self._resample_to_weekly(df_daily)
        weekly_trend, weekly_score = self._analyze_timeframe(
            df_weekly, lookback=self.weekly_lookback, timeframe_name="Weekly"
        )

        # 3. 4-HOUR TIMEFRAME ANALYSIS (if available)
        four_hour_trend = None
        four_hour_score = None
        if self.use_four_hour and df_4h is not None and not df_4h.empty:
            four_hour_trend, four_hour_score = self._analyze_timeframe(
                df_4h, lookback=self.four_hour_lookback, timeframe_name="4H"
            )

        # 4. CALCULATE ALIGNMENT SCORE
        alignment_score = self._calculate_alignment_score(
            daily_score, weekly_score, four_hour_score
        )

        # 5. ADAPTIVE THRESHOLD BASED ON MARKET REGIME
        adjusted_threshold = self._get_adaptive_threshold(market_regime)
        is_aligned = alignment_score >= adjusted_threshold

        # 6. CONFIDENCE ADJUSTMENT
        confidence_adjustment = self._calculate_confidence_adjustment(
            alignment_score, is_aligned, daily_trend, weekly_trend, four_hour_trend
        )

        # 7. BUILD REASONS AND WARNINGS
        self._build_messages(
            reasons,
            warnings,
            daily_trend,
            weekly_trend,
            four_hour_trend,
            daily_score,
            weekly_score,
            four_hour_score,
            alignment_score,
        )

        return MultiTimeframeAnalysis(
            daily_trend=daily_trend,
            daily_score=daily_score,
            weekly_trend=weekly_trend,
            weekly_score=weekly_score,
            four_hour_trend=four_hour_trend,
            four_hour_score=four_hour_score,
            alignment_score=alignment_score,
            is_aligned=is_aligned,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons,
            warnings=warnings,
        )

    def _analyze_timeframe(
        self, df: pd.DataFrame, lookback: int, timeframe_name: str
    ) -> tuple[TimeframeTrend, float]:
        """
        Phân tích xu hướng cho một timeframe cụ thể

        Uses multiple indicators:
        1. EMA alignment (20/50/200)
        2. Price momentum (ROC)
        3. MACD histogram
        4. Linear regression slope

        Returns:
            (TimeframeTrend, score)
            score: -100 (strong down) to +100 (strong up)
        """
        if df is None or df.empty or len(df) < lookback:
            logger.warning(f"{timeframe_name}: Insufficient data for analysis")
            return TimeframeTrend.NEUTRAL, 0.0

        try:
            # Get latest data
            recent_df = df.tail(lookback)
            latest = df.iloc[-1]

            score_components = []

            # 1. EMA ALIGNMENT (weight: 40%)
            if all(col in latest.index for col in ["ema_20", "ema_50", "ema_200"]):
                ema_score = self._calculate_ema_score(latest)
                score_components.append(("ema", ema_score, 0.40))

            # 2. PRICE MOMENTUM - ROC (weight: 25%)
            if "close" in recent_df.columns and len(recent_df) >= 10:
                roc_10 = (
                    (latest["close"] - recent_df["close"].iloc[-10])
                    / recent_df["close"].iloc[-10]
                    * 100
                )
                # Normalize: -5% to +5% maps to -100 to +100
                roc_score = np.clip(roc_10 / 0.05 * 100, -100, 100)
                score_components.append(("roc", roc_score, 0.25))

            # 3. MACD HISTOGRAM (weight: 20%)
            if "macd_histogram" in latest.index:
                # MACD histogram: positive = bullish, negative = bearish
                macd_hist = latest["macd_histogram"]
                # Normalize: -1.0 to +1.0 maps to -100 to +100
                macd_score = np.clip(macd_hist / 1000 * 100, -100, 100)
                score_components.append(("macd", macd_score, 0.20))

            # 4. LINEAR REGRESSION SLOPE (weight: 15%)
            if "close" in recent_df.columns:
                slope_score = self._calculate_slope_score(recent_df["close"])
                score_components.append(("slope", slope_score, 0.15))

            # Calculate weighted composite score
            if not score_components:
                logger.warning(f"{timeframe_name}: No indicators available for scoring")
                return TimeframeTrend.NEUTRAL, 0.0

            total_score = sum(score * weight for _, score, weight in score_components)

            # Determine trend from score
            trend = self._score_to_trend(total_score)

            logger.debug(
                f"{timeframe_name} Analysis: "
                f"score={total_score:.1f}, trend={trend.name}, "
                f"components={[(name, f'{s:.1f}') for name, s, _ in score_components]}"
            )

            return trend, total_score

        except Exception as e:
            logger.error(f"Error analyzing {timeframe_name}: {e}", exc_info=True)
            return TimeframeTrend.NEUTRAL, 0.0

    def _calculate_ema_score(self, latest: pd.Series) -> float:
        """
        Calculate EMA alignment score

        Perfect alignment: EMA20 > EMA50 > EMA200 = +100
        Perfect bearish: EMA20 < EMA50 < EMA200 = -100
        """
        ema_20 = latest["ema_20"]
        ema_50 = latest["ema_50"]
        ema_200 = latest["ema_200"]

        # Calculate separation percentages
        sep_20_50 = ((ema_20 - ema_50) / ema_50) * 100
        sep_50_200 = ((ema_50 - ema_200) / ema_200) * 100

        # Score based on alignment and separation
        if ema_20 > ema_50 > ema_200:
            # Bullish alignment
            # More separation = stronger trend
            score = 50 + min(sep_20_50 * 10, 25) + min(sep_50_200 * 10, 25)
        elif ema_20 < ema_50 < ema_200:
            # Bearish alignment
            score = -50 + max(sep_20_50 * 10, -25) + max(sep_50_200 * 10, -25)
        elif ema_20 > ema_50:
            # Partial bullish (20 > 50 but 50 < 200)
            score = 25 + min(sep_20_50 * 10, 25)
        elif ema_20 < ema_50:
            # Partial bearish
            score = -25 + max(sep_20_50 * 10, -25)
        else:
            # Neutral/choppy
            score = 0

        return np.clip(score, -100, 100)

    def _calculate_slope_score(self, close_series: pd.Series) -> float:
        """
        Calculate linear regression slope score

        Positive slope = bullish, negative = bearish
        """
        try:
            # Linear regression on close prices
            x = np.arange(len(close_series))
            y = close_series.values

            # Calculate slope
            slope, intercept = np.polyfit(x, y, 1)

            # Normalize slope: ±2% per bar = ±100 score
            avg_price = close_series.mean()
            slope_pct = (slope / avg_price) * 100

            score = np.clip(slope_pct / 0.02 * 100, -100, 100)

            return score

        except Exception:
            return 0.0

    def _score_to_trend(self, score: float) -> TimeframeTrend:
        """Convert numeric score to TimeframeTrend enum"""
        if score >= 60:
            return TimeframeTrend.STRONG_UP
        elif score >= 30:
            return TimeframeTrend.MODERATE_UP
        elif score >= 10:
            return TimeframeTrend.WEAK_UP
        elif score <= -60:
            return TimeframeTrend.STRONG_DOWN
        elif score <= -30:
            return TimeframeTrend.MODERATE_DOWN
        elif score <= -10:
            return TimeframeTrend.WEAK_DOWN
        else:
            return TimeframeTrend.NEUTRAL

    def _resample_to_weekly(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Resample daily data to weekly

        OHLC: standard resampling
        Indicators: take last value of week
        """
        try:
            # Set date as index if not already
            df = df_daily.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

            # Resample rules
            ohlc_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }

            # Add indicators (take last value of week)
            indicator_cols = [col for col in df.columns if col not in ohlc_dict.keys()]
            for col in indicator_cols:
                ohlc_dict[col] = "last"

            df_weekly = df.resample("W").agg(ohlc_dict)
            df_weekly.dropna(inplace=True)

            return df_weekly

        except Exception as e:
            logger.error(f"Error resampling to weekly: {e}")
            return pd.DataFrame()

    def _calculate_alignment_score(
        self, daily_score: float, weekly_score: float, four_hour_score: Optional[float]
    ) -> float:
        """
        Calculate composite alignment score using weighted average

        Returns: -100 (fully bearish) to +100 (fully bullish)
        """
        # Weighted composite
        alignment = daily_score * self.daily_weight + weekly_score * self.weekly_weight

        if four_hour_score is not None and self.use_four_hour:
            alignment += four_hour_score * self.four_hour_weight

        return np.clip(alignment, -100, 100)

    def _get_adaptive_threshold(self, market_regime: Optional[Dict]) -> float:
        """
        Adjust alignment threshold based on market regime

        BULL market: Lower threshold (easier to get aligned signal)
        BEAR market: Higher threshold (stricter)
        """
        base_threshold = self.alignment_threshold

        if not market_regime:
            return base_threshold

        regime = market_regime.get("regime", "SIDEWAYS")
        confidence = market_regime.get("confidence", 50)

        if regime == "BULL" and confidence >= 70:
            # In strong bull, relax threshold by 20%
            return base_threshold * 0.80
        elif regime == "BEAR":
            # In bear, tighten threshold by 20%
            return base_threshold * 1.20
        elif regime == "HIGH_VOLATILITY":
            # In high vol, tighten threshold by 30%
            return base_threshold * 1.30

        return base_threshold

    def _calculate_confidence_adjustment(
        self,
        alignment_score: float,
        is_aligned: bool,
        daily_trend: TimeframeTrend,
        weekly_trend: TimeframeTrend,
        four_hour_trend: Optional[TimeframeTrend],
    ) -> int:
        """
        Calculate confidence adjustment based on timeframe alignment

        Returns: -20 to +20
        """
        # Base adjustment from alignment score
        # Perfect alignment (100) = +20, perfect misalignment (-100) = -20
        base_adj = int(alignment_score / 5)  # -20 to +20

        # Bonus for strong trends
        if daily_trend == TimeframeTrend.STRONG_UP and weekly_trend in [
            TimeframeTrend.STRONG_UP,
            TimeframeTrend.MODERATE_UP,
        ]:
            base_adj += 5  # Extra bonus for strong multi-timeframe alignment

        # Penalty for conflicting trends
        if daily_trend.value > 0 and weekly_trend.value < 0:
            base_adj -= 10  # Daily bullish but weekly bearish = warning

        return np.clip(base_adj, -20, 20)

    def _build_messages(
        self,
        reasons: list,
        warnings: list,
        daily_trend: TimeframeTrend,
        weekly_trend: TimeframeTrend,
        four_hour_trend: Optional[TimeframeTrend],
        daily_score: float,
        weekly_score: float,
        four_hour_score: Optional[float],
        alignment_score: float,
    ):
        """Build reason and warning messages"""

        # Daily
        if daily_trend.value >= 2:
            reasons.append(f"✅ Daily: {daily_trend.name} ({daily_score:+.0f})")
        elif daily_trend.value <= -2:
            warnings.append(f"⚠️ Daily: {daily_trend.name} ({daily_score:+.0f})")

        # Weekly
        if weekly_trend.value >= 2:
            reasons.append(f"✅ Weekly: {weekly_trend.name} ({weekly_score:+.0f})")
        elif weekly_trend.value <= -2:
            warnings.append(f"⚠️ Weekly: {weekly_trend.name} ({weekly_score:+.0f})")

        # 4H (if available)
        if four_hour_trend is not None:
            if four_hour_trend.value >= 2:
                reasons.append(f"✅ 4H: {four_hour_trend.name} ({four_hour_score:+.0f})")
            elif four_hour_trend.value <= -2:
                warnings.append(f"⚠️ 4H: {four_hour_trend.name} ({four_hour_score:+.0f})")

        # Alignment
        if alignment_score >= 60:
            reasons.append(f"✅ Strong multi-timeframe alignment ({alignment_score:+.0f})")
        elif alignment_score <= -60:
            warnings.append(f"⚠️ Multi-timeframe conflict ({alignment_score:+.0f})")


# Singleton instance
_mtf_analyzer = None


def get_mtf_analyzer() -> MultiTimeframeAnalyzer:
    """Get singleton instance"""
    global _mtf_analyzer
    if _mtf_analyzer is None:
        _mtf_analyzer = MultiTimeframeAnalyzer()
    return _mtf_analyzer
