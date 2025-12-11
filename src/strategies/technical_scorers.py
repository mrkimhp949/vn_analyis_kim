"""
Technical Scorers Module

Contains all _score_* methods for technical indicator scoring.
Each scorer returns a 0-1 score for individual indicators.
"""

import logging
from typing import Optional

import pandas as pd

from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

logger = logging.getLogger(__name__)

# Scoring constants
TECH_SCORE_HIGH = 1.0
TECH_SCORE_GOOD = 0.8
TECH_SCORE_MODERATE = 0.6
TECH_SCORE_LOW = 0.4
TECH_SCORE_POOR = 0.2

# RSI thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


class TechnicalScorer:
    """
    Scores technical indicators for confidence calculation.

    All scoring methods return 0-1 scale where:
    - 1.0 = Very bullish / strong buy signal
    - 0.8 = Bullish / good
    - 0.6 = Neutral / moderate
    - 0.4 = Caution
    - 0.2 = Bearish / avoid
    """

    # Weight configuration for confidence calculation
    DEFAULT_WEIGHTS = {
        "rsi": 0.15,
        "macd": 0.20,
        "bollinger": 0.15,
        "stochastic": 0.15,
        "ema_alignment": 0.20,
        "volume": 0.10,
        "price_action": 0.05,
    }

    def __init__(self, weights: Optional[dict] = None):
        """
        Initialize TechnicalScorer.

        Args:
            weights: Optional custom weights for each indicator
        """
        self.weights = weights or self.DEFAULT_WEIGHTS

    def calculate_technical_confidence(self, df: pd.DataFrame) -> float:
        """
        Calculate confidence from technical indicators with weighted scoring.

        Uses 7 weighted technical factors:
        1. RSI position (15% weight) - Oversold/overbought detection
        2. MACD signal (20% weight) - Momentum and trend
        3. Bollinger Bands (15% weight) - Volatility and extremes
        4. Stochastic (15% weight) - Additional momentum
        5. Moving average alignment (20% weight) - Trend confirmation
        6. Volume confirmation (10% weight) - Interest validation
        7. Price action (5% weight) - Recent momentum

        Args:
            df: DataFrame with OHLCV and indicator data

        Returns:
            Confidence score 0-100 (weighted average)
        """
        if len(df) < 50:
            return 0.0

        scores = {}

        # Calculate all scores
        scores["rsi"] = self.score_rsi(df)
        scores["macd"] = self.score_macd(df)
        scores["bollinger"] = self.score_bollinger_bands(df)
        scores["stochastic"] = self.score_stochastic(df)
        scores["ema_alignment"] = self.score_ema_alignment(df)
        scores["volume"] = self.score_volume(df)
        scores["price_action"] = self.score_price_action(df)

        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0

        for factor, score in scores.items():
            if score is not None:
                weight = self.weights.get(factor, 0.0)
                total_score += score * weight
                total_weight += weight

        # Normalize to 0-100
        if total_weight > 0:
            confidence = (total_score / total_weight) * 100
        else:
            confidence = 50.0

        return min(max(confidence, 0.0), 100.0)

    def score_rsi(self, df: pd.DataFrame) -> Optional[float]:
        """
        Score RSI indicator on 0-1 scale.

        Scoring:
            RSI 30-40: 1.0 (oversold, strong buy)
            RSI 40-60: 0.8 (neutral, good)
            RSI 60-70: 0.5 (overbought warning)
            RSI >70: 0.2 (overbought, avoid)
            RSI <30: 0.6 (very oversold, risky but opportunity)

        Args:
            df: DataFrame with RSI data

        Returns:
            Score 0-1 or None if RSI unavailable
        """
        if "rsi" not in df.columns:
            return None

        rsi = safe_get_latest(df, "rsi", 50)
        if pd.isna(rsi):
            return None

        if rsi < RSI_OVERSOLD:
            return TECH_SCORE_MODERATE  # 0.6 - risky but opportunity
        elif RSI_OVERSOLD <= rsi <= 40:
            return TECH_SCORE_HIGH  # 1.0 - strong buy
        elif 40 < rsi <= 60:
            return TECH_SCORE_GOOD  # 0.8 - good
        elif 60 < rsi <= RSI_OVERBOUGHT:
            return 0.5  # Warning
        else:  # RSI > 70
            return TECH_SCORE_POOR  # 0.2 - avoid

    def score_macd(self, df: pd.DataFrame) -> Optional[float]:
        """
        Score MACD indicator on 0-1 scale.

        Scoring:
            MACD > Signal and both positive: 1.0 (strong bullish)
            MACD > Signal and MACD positive: 0.9 (bullish)
            MACD > Signal: 0.7 (turning bullish)
            MACD < Signal: 0.3 (bearish)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Score 0-1 or None if calculation fails
        """
        try:
            # Calculate MACD if not present
            if "macd" not in df.columns or "macd_signal" not in df.columns:
                ema12 = df["close"].ewm(span=12).mean()
                ema26 = df["close"].ewm(span=26).mean()
                macd_series = ema12 - ema26
                signal_series = macd_series.ewm(span=9).mean()
                macd_latest = macd_series.iloc[-1]
                signal_latest = signal_series.iloc[-1]
            else:
                macd_latest = safe_get_latest(df, "macd", 0)
                signal_latest = safe_get_latest(df, "macd_signal", 0)

            if pd.isna(macd_latest) or pd.isna(signal_latest):
                return None

            if macd_latest > signal_latest:
                if macd_latest > 0 and signal_latest > 0:
                    return TECH_SCORE_HIGH  # 1.0
                elif macd_latest > 0:
                    return 0.9
                else:
                    return 0.7
            return 0.3

        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"MACD scoring failed: {e}")
            return None

    def score_bollinger_bands(self, df: pd.DataFrame) -> Optional[float]:
        """
        Score Bollinger Bands indicator on 0-1 scale.

        Scoring based on position within bands:
            Near lower band (0-0.2): 1.0 (oversold)
            Lower quarter (0.2-0.4): 0.8 (good entry)
            Middle (0.4-0.6): 0.6 (neutral)
            Upper quarter (0.6-0.8): 0.4 (caution)
            Near upper band (0.8-1.0): 0.2 (overbought)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Score 0-1 or None if calculation fails
        """
        try:
            if len(df) < 20:
                return None

            sma20 = df["close"].rolling(window=20).mean()
            std20 = df["close"].rolling(window=20).std()
            upper_band = sma20 + (2 * std20)
            lower_band = sma20 - (2 * std20)

            current_price = safe_get_latest(df, "close", 0)
            upper = upper_band.iloc[-1]
            lower = lower_band.iloc[-1]

            if pd.isna(upper) or pd.isna(lower):
                return None

            band_width = upper - lower
            if band_width <= 0:
                return 0.5

            position = (current_price - lower) / band_width

            if position <= 0.2:
                return TECH_SCORE_HIGH
            elif position <= 0.4:
                return TECH_SCORE_GOOD
            elif position <= 0.6:
                return TECH_SCORE_MODERATE
            elif position <= 0.8:
                return TECH_SCORE_LOW
            return TECH_SCORE_POOR

        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"Bollinger Bands scoring failed: {e}")
            return None

    def score_stochastic(self, df: pd.DataFrame) -> Optional[float]:
        """
        Score Stochastic indicator on 0-1 scale.

        Scoring:
            K < 20 and K > D: 1.0 (oversold turning up)
            K < 30: 0.9 (oversold)
            K 30-70: 0.7 (neutral)
            K > 70 and K < D: 0.4 (overbought turning down)
            K > 80: 0.2 (overbought)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Score 0-1 or None if calculation fails
        """
        try:
            if len(df) < 14:
                return None

            low_14 = df["low"].rolling(window=14).min()
            high_14 = df["high"].rolling(window=14).max()
            range_14 = high_14 - low_14

            # Avoid division by zero
            if (range_14 == 0).any():
                return None

            k_percent = 100 * ((df["close"] - low_14) / range_14)
            d_percent = k_percent.rolling(window=3).mean()

            k = k_percent.iloc[-1]
            d = d_percent.iloc[-1]

            if pd.isna(k) or pd.isna(d):
                return None

            if k < 20 and k > d:
                return TECH_SCORE_HIGH
            elif k < 30:
                return 0.9
            elif 30 <= k <= 70:
                return 0.7
            elif k > 70 and k < d:
                return TECH_SCORE_LOW
            return TECH_SCORE_POOR

        except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
            logger.debug(f"Stochastic scoring failed: {e}")
            return None

    def score_ema_alignment(self, df: pd.DataFrame) -> Optional[float]:
        """
        Score EMA alignment on 0-1 scale.

        Scoring:
            Price > EMA20 > EMA50: 1.0 (strong uptrend)
            Price > EMA20: 0.5 (above short-term MA)
            EMA20 > EMA50: 0.3 (bullish alignment)
            Otherwise: 0.3 minimum (weak/bearish)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Score 0-1 or None if calculation fails
        """
        try:
            if len(df) < 50:
                return None

            ema20 = df["close"].ewm(span=20).mean()
            ema50 = df["close"].ewm(span=50).mean()
            current_price = safe_get_latest(df, "close", 0)
            latest_ema20 = ema20.iloc[-1]
            latest_ema50 = ema50.iloc[-1]

            if pd.isna(latest_ema20) or pd.isna(latest_ema50):
                return None

            score = 0.0
            if current_price > latest_ema20:
                score += 0.50
            if latest_ema20 > latest_ema50:
                score += 0.30
            if current_price > latest_ema20 and latest_ema20 > latest_ema50:
                score += 0.20  # Perfect alignment bonus

            return max(score, 0.30)

        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"EMA alignment scoring failed: {e}")
            return None

    def score_volume(self, df: pd.DataFrame) -> Optional[float]:
        """
        Score volume confirmation on 0-1 scale.

        Scoring:
            Volume > 2x avg: 1.0 (very strong interest)
            Volume > 1.5x avg: 0.9 (strong interest)
            Volume > 1.2x avg: 0.8 (good interest)
            Volume 0.8-1.2x avg: 0.6 (normal)
            Volume < 0.8x avg: 0.4 (low interest)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Score 0-1 or None if calculation fails
        """
        try:
            if len(df) < 20:
                return None

            current_volume = safe_get_latest(df, "volume", 0)
            avg_volume = safe_rolling_operation(df, "volume", 20, "mean", 0)

            if avg_volume <= 0:
                return None

            volume_ratio = current_volume / avg_volume

            if volume_ratio >= 2.0:
                return TECH_SCORE_HIGH
            elif volume_ratio >= 1.5:
                return 0.9
            elif volume_ratio >= 1.2:
                return TECH_SCORE_GOOD
            elif volume_ratio >= 0.8:
                return TECH_SCORE_MODERATE
            return TECH_SCORE_LOW

        except (KeyError, ZeroDivisionError) as e:
            logger.debug(f"Volume scoring failed: {e}")
            return None

    def score_price_action(self, df: pd.DataFrame) -> Optional[float]:
        """
        Score recent price action on 0-1 scale.

        Scoring based on last 3 candles:
            3 up days: 1.0 (strong momentum)
            2 up days: 0.7 (moderate momentum)
            1 up day: 0.5 (weak momentum)
            0 up days: 0.3 (no momentum)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Score 0-1 or None if calculation fails
        """
        try:
            if len(df) < 5:
                return None

            recent_closes = df["close"].tail(4).tolist()
            if len(recent_closes) < 4:
                return None

            up_days = sum(
                1 for i in range(1, len(recent_closes)) if recent_closes[i] > recent_closes[i - 1]
            )

            score_map = {3: TECH_SCORE_HIGH, 2: 0.7, 1: 0.5, 0: 0.3}
            return score_map.get(up_days, 0.3)

        except (KeyError, IndexError) as e:
            logger.debug(f"Price action scoring failed: {e}")
            return None

    def get_technical_signal(self, df: pd.DataFrame) -> str:
        """
        Determine signal type from technical analysis.
        Multi-indicator check instead of simple 2-candle comparison.

        Scoring System (need >= 3 points for BUY, <= -3 for SELL):
        - EMA20 > EMA50: +2 (trend alignment)
        - RSI < 40: +1 (not overbought)
        - Price > EMA20: +1 (above short-term MA)
        - Volume > 20-day avg: +1 (volume confirmation)
        - MACD > 0 or MACD crossover: +1 (momentum)

        Args:
            df: DataFrame with OHLCV and indicator data

        Returns:
            "BUY", "SELL", or "HOLD"
        """
        if len(df) < 50:
            return "HOLD"

        score = 0

        # Get indicators
        current_price = safe_get_latest(df, "close", 0)
        ema20 = safe_get_latest(df, "ema20", 0)
        ema50 = safe_get_latest(df, "ema50", 0)
        rsi = safe_get_latest(df, "rsi", 50)
        current_volume = safe_get_latest(df, "volume", 0)
        avg_volume = df["volume"].tail(20).mean() if "volume" in df.columns else 0

        # 1. EMA alignment (most important - 2 points)
        if ema20 > 0 and ema50 > 0:
            if ema20 > ema50:
                score += 2  # Bullish trend
            elif ema20 < ema50 * 0.98:  # 2% below
                score -= 2  # Bearish trend

        # 2. RSI check (1 point)
        if 30 <= rsi <= 65:
            score += 1  # Healthy RSI zone for buying
        elif rsi > 70:
            score -= 1  # Overbought

        # 3. Price vs EMA20 (1 point)
        if ema20 > 0:
            if current_price > ema20:
                score += 1
            elif current_price < ema20 * 0.98:
                score -= 1

        # 4. Volume confirmation (1 point)
        if avg_volume > 0:
            if current_volume > avg_volume * 1.2:
                score += 1
            elif current_volume < avg_volume * 0.5:
                score -= 1

        # 5. MACD check
        try:
            if "macd" in df.columns:
                macd = safe_get_latest(df, "macd", 0)
                macd_signal = safe_get_latest(df, "macd_signal", 0)
                if macd > macd_signal and macd > 0:
                    score += 1
                elif macd < macd_signal and macd < 0:
                    score -= 1
        except Exception:
            pass

        # Decision
        if score >= 3:
            return "BUY"
        elif score <= -3:
            return "SELL"
        return "HOLD"

    def get_all_scores(self, df: pd.DataFrame) -> dict:
        """
        Get all individual scores for debugging/analysis.

        Args:
            df: DataFrame with OHLCV and indicator data

        Returns:
            Dict with all scores and their weights
        """
        scores = {
            "rsi": self.score_rsi(df),
            "macd": self.score_macd(df),
            "bollinger": self.score_bollinger_bands(df),
            "stochastic": self.score_stochastic(df),
            "ema_alignment": self.score_ema_alignment(df),
            "volume": self.score_volume(df),
            "price_action": self.score_price_action(df),
        }

        return {
            "scores": scores,
            "weights": self.weights,
            "confidence": self.calculate_technical_confidence(df),
            "signal": self.get_technical_signal(df),
        }
