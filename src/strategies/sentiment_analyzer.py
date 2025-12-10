"""
Sentiment Analyzer Module

Contains sentiment analysis logic including NLP-based and price-based sentiment.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes market sentiment using multiple sources:
    1. NLP-based sentiment from news articles
    2. Price-based sentiment proxy (fallback)
    3. Volume profile analysis
    """

    # Sentiment adjustment values
    SENTIMENT_ADJUSTMENTS = {
        "VERY_POSITIVE": 8,
        "POSITIVE": 5,
        "SLIGHTLY_POSITIVE": 2,
        "NEUTRAL": 0,
        "SLIGHTLY_NEGATIVE": -3,
        "NEGATIVE": -5,
        "VERY_NEGATIVE": -10,
    }

    def __init__(self, use_nlp: bool = True):
        """
        Initialize SentimentAnalyzer.

        Args:
            use_nlp: Whether to attempt NLP-based sentiment analysis
        """
        self.use_nlp = use_nlp
        self._nlp_available = None  # Cached availability check

    def analyze_sentiment(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Analyze sentiment for a symbol using available methods.

        Priority:
        1. NLP-based sentiment (if available)
        2. Price-based sentiment proxy (fallback)

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data

        Returns:
            Dict with sentiment, adjustment, score, source
        """
        if not symbol:
            return self._neutral_result("No symbol provided")

        # Try NLP-based sentiment first
        if self.use_nlp:
            nlp_result = self._get_nlp_sentiment(symbol, df)
            if nlp_result:
                return nlp_result

        # Fallback to price-based sentiment
        price_result = self.calculate_price_based_sentiment(df)
        if price_result:
            price_result["source"] = "price_proxy"
            return price_result

        return self._neutral_result("Unable to calculate sentiment")

    def _get_nlp_sentiment(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Get NLP-based sentiment from news analysis.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data

        Returns:
            Dict with sentiment data or None if unavailable
        """
        try:
            from src.nlp.multimodal_fusion import get_sentiment_adjustment

            sentiment_data = get_sentiment_adjustment(symbol, df)

            if sentiment_data:
                return {
                    "sentiment": sentiment_data.get("sentiment", "NEUTRAL"),
                    "adjustment": sentiment_data.get("adjustment", 0),
                    "score": sentiment_data.get("score", 0),
                    "news_count": sentiment_data.get("news_count", 0),
                    "source": "nlp",
                }
            return None

        except ImportError:
            logger.debug("NLP module not available")
            self._nlp_available = False
            return None
        except Exception as e:
            logger.warning(f"NLP sentiment error: {e}")
            return None

    def calculate_price_based_sentiment(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate sentiment proxy from price action when NLP is unavailable.

        Uses recent price momentum, volume patterns, and volume profile:

        Scoring System:
        - Price momentum (5d, 20d returns)
        - Volume confirmation (current vs average)
        - Volume profile: Accumulation/Distribution pattern
        - OBV trend (if available)

        Final sentiment:
        - VERY_POSITIVE: score >= 4 (+8)
        - POSITIVE: score >= 2 (+5)
        - SLIGHTLY_POSITIVE: score >= 1 (+2)
        - NEUTRAL: score around 0 (0)
        - SLIGHTLY_NEGATIVE: score <= -1 (-3)
        - NEGATIVE: score <= -2 (-5)
        - VERY_NEGATIVE: score <= -4 (-10)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict with sentiment, adjustment, reason
        """
        if len(df) < 20:
            return None

        try:
            close = df["close"]
            volume = df["volume"] if "volume" in df.columns else None

            score = 0
            components = []

            # 1. Price momentum (5-day and 20-day returns)
            ret_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(df) >= 5 else 0
            ret_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(df) >= 20 else 0

            if ret_5d > 3:
                score += 1
                components.append(f"+{ret_5d:.1f}% 5d")
            elif ret_5d > 1:
                score += 0.5
                components.append(f"+{ret_5d:.1f}% 5d")
            elif ret_5d < -3:
                score -= 1
                components.append(f"{ret_5d:.1f}% 5d")
            elif ret_5d < -1:
                score -= 0.5
                components.append(f"{ret_5d:.1f}% 5d")

            if ret_20d > 5:
                score += 1
                components.append(f"+{ret_20d:.1f}% 20d")
            elif ret_20d > 2:
                score += 0.5
            elif ret_20d < -5:
                score -= 1
                components.append(f"{ret_20d:.1f}% 20d")
            elif ret_20d < -2:
                score -= 0.5

            # 2. Volume profile analysis
            if volume is not None and len(volume) >= 20:
                avg_vol = volume.tail(20).mean()
                current_vol = volume.iloc[-1]
                vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

                # Volume surge with price movement
                if vol_ratio > 1.5:
                    if ret_5d > 0:
                        score += 1
                        components.append("vol surge↑")
                    elif ret_5d < 0:
                        score -= 1
                        components.append("vol surge↓")
                elif vol_ratio > 1.2:
                    if ret_5d > 0:
                        score += 0.5
                        components.append("vol confirm↑")
                    elif ret_5d < 0:
                        score -= 0.5
                        components.append("vol confirm↓")

                # Volume trend analysis (accumulation/distribution)
                vol_5d_avg = volume.tail(5).mean()
                vol_10d_avg = volume.tail(10).mean()

                if vol_5d_avg > vol_10d_avg * 1.2:
                    if ret_5d > 0:
                        score += 0.5  # Accumulation
                        components.append("accumulation")
                    elif ret_5d < 0:
                        score -= 0.5  # Distribution
                        components.append("distribution")

                # Price-volume divergence check
                price_up = close.iloc[-1] > close.iloc[-5]
                vol_declining = vol_5d_avg < vol_10d_avg * 0.8

                if price_up and vol_declining:
                    score -= 0.5  # Weak rally
                    components.append("weak rally")
                elif not price_up and vol_declining:
                    score += 0.5  # Selling exhaustion
                    components.append("exhaustion")

            # 3. OBV trend (if available)
            if "obv" in df.columns:
                obv = df["obv"]
                obv_5d = obv.tail(5)
                obv_trend = (
                    (obv_5d.iloc[-1] - obv_5d.iloc[0]) / abs(obv_5d.iloc[0]) * 100
                    if obv_5d.iloc[0] != 0
                    else 0
                )

                if obv_trend > 5:
                    score += 0.5
                    components.append("OBV↑")
                elif obv_trend < -5:
                    score -= 0.5
                    components.append("OBV↓")

            # Determine sentiment from final score
            reason = " | ".join(components) if components else f"Price: {ret_5d:+.1f}% 5d"

            return self._score_to_sentiment(score, reason)

        except Exception as e:
            logger.debug(f"Price-based sentiment calculation failed: {e}")
            return None

    def _score_to_sentiment(self, score: float, reason: str) -> Dict:
        """
        Convert numeric score to sentiment classification.

        Args:
            score: Numeric sentiment score
            reason: Explanation of score components

        Returns:
            Dict with sentiment, adjustment, reason
        """
        if score >= 4:
            sentiment = "VERY_POSITIVE"
        elif score >= 2:
            sentiment = "POSITIVE"
        elif score >= 1:
            sentiment = "SLIGHTLY_POSITIVE"
        elif score <= -4:
            sentiment = "VERY_NEGATIVE"
        elif score <= -2:
            sentiment = "NEGATIVE"
        elif score <= -1:
            sentiment = "SLIGHTLY_NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        return {
            "sentiment": sentiment,
            "adjustment": self.SENTIMENT_ADJUSTMENTS.get(sentiment, 0),
            "score": score,
            "reason": reason,
        }

    def _neutral_result(self, reason: str) -> Dict:
        """
        Return a neutral sentiment result.

        Args:
            reason: Explanation for neutral result

        Returns:
            Dict with neutral sentiment
        """
        return {
            "sentiment": "NEUTRAL",
            "adjustment": 0,
            "score": 0,
            "reason": reason,
            "source": "default",
        }


class VolumeAnalyzer:
    """
    Analyzes volume patterns for sentiment and manipulation detection.
    """

    def __init__(self):
        """Initialize VolumeAnalyzer."""
        pass

    def calculate_obv(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Calculate On-Balance Volume (OBV) with error handling.

        OBV measures buying/selling pressure by adding volume on up days
        and subtracting on down days.

        Args:
            df: DataFrame with close and volume data

        Returns:
            Series with OBV values, or None if calculation fails
        """
        try:
            if "close" not in df.columns or "volume" not in df.columns:
                logger.warning("OBV calculation failed: missing close or volume columns")
                return None

            if df["close"].isna().any() or df["volume"].isna().any():
                logger.warning("OBV calculation: NaN values detected, filling forward")
                df = df.ffill()

            if len(df) < 2:
                logger.warning("OBV calculation failed: insufficient data")
                return None

            obv = [0]
            for i in range(1, len(df)):
                try:
                    close_curr = df["close"].iloc[i]
                    close_prev = df["close"].iloc[i - 1]
                    volume_curr = df["volume"].iloc[i]

                    if pd.isna(close_curr) or pd.isna(close_prev) or pd.isna(volume_curr):
                        obv.append(obv[-1])
                        continue

                    if close_curr > close_prev:
                        obv.append(obv[-1] + volume_curr)
                    elif close_curr < close_prev:
                        obv.append(obv[-1] - volume_curr)
                    else:
                        obv.append(obv[-1])

                except Exception as e:
                    logger.warning(f"OBV calculation error at index {i}: {e}")
                    obv.append(obv[-1] if obv else 0)

            return pd.Series(obv, index=df.index)

        except Exception as e:
            logger.error(f"OBV calculation failed: {e}")
            return None

    def check_volume_confirmation(
        self, df: pd.DataFrame, market_regime: Optional[Dict] = None
    ) -> Dict:
        """
        Check volume confirmation với multiple indicators và dynamic threshold.

        Checks:
        1. Volume ratio (current vs average)
        2. Volume trend (5-day vs 20-day MA)
        3. OBV (On-Balance Volume) - accumulation/distribution

        Dynamic threshold based on market regime:
        - BULL market: Lower threshold (0.4) - more opportunities
        - BEAR/HIGH_VOL: Higher threshold (0.6) - more selective
        - SIDEWAYS: Normal threshold (0.5)

        Args:
            df: DataFrame with OHLCV data
            market_regime: Optional market regime data

        Returns:
            Dict with detailed volume analysis
        """
        if len(df) < 20:
            return {
                "confirmed": True,
                "reason": "Chưa đủ data",
                "surge": False,
                "obv_bullish": True,
                "volume_trending": True,
                "confidence": 0.5,
            }

        current_volume = safe_get_latest(df, "volume", 0)
        avg_volume_20 = safe_rolling_operation(df, "volume", 20, "mean", 0)

        if avg_volume_20 == 0:
            return {
                "confirmed": True,
                "reason": "Volume data invalid",
                "surge": False,
                "obv_bullish": True,
                "volume_trending": True,
                "confidence": 0.5,
            }

        # Dynamic threshold calculation
        base_threshold = 0.5

        # Adjust for liquidity tier
        avg_value = avg_volume_20 * safe_get_latest(df, "close", 100_000)
        liquidity_adjustment = 0.0

        if avg_value < 1_000_000_000:  # Small cap
            liquidity_adjustment = -0.15
        elif avg_value < 5_000_000_000:  # Mid cap
            liquidity_adjustment = -0.10

        # Adjust for market regime
        regime_adjustment = 0.0
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            regime_confidence = market_regime.get("confidence", 50)

            if regime == "BULL" and regime_confidence >= 70:
                regime_adjustment = -0.10
            elif regime in ["BEAR", "HIGH_VOLATILITY"]:
                regime_adjustment = +0.10

        base_threshold = max(
            0.25, min(0.75, base_threshold + liquidity_adjustment + regime_adjustment)
        )

        # Volume ratio
        volume_ratio = current_volume / avg_volume_20

        # Volume trend
        avg_volume_5 = safe_rolling_operation(df, "volume", 5, "mean", 0)
        volume_trending_up = avg_volume_5 > avg_volume_20

        # OBV analysis
        obv = self.calculate_obv(df)
        obv_bullish = True
        obv_available = False

        if obv is not None and len(obv) >= 5:
            try:
                obv_recent = obv.iloc[-5:]
                obv_slope = (obv_recent.iloc[-1] - obv_recent.iloc[0]) / 5

                if len(obv) >= 20:
                    obv_ma_5 = obv.rolling(5).mean().iloc[-1]
                    obv_ma_20 = obv.rolling(20).mean().iloc[-1]

                    if not pd.isna(obv_ma_5) and not pd.isna(obv_ma_20):
                        obv_bullish = (obv_slope > 0) and (obv_ma_5 > obv_ma_20)
                        obv_available = True
                    else:
                        obv_bullish = obv_slope > 0
                        obv_available = True
                else:
                    obv_bullish = obv_slope > 0
                    obv_available = True

            except Exception as e:
                logger.warning(f"OBV analysis failed: {e}")
                obv_bullish = True

        # Calculate confidence score
        confidence_score = 0.0

        if volume_ratio >= 1.5:
            confidence_score += 0.4
        elif volume_ratio >= 1.2:
            confidence_score += 0.3
        elif volume_ratio >= 1.0:
            confidence_score += 0.2

        if volume_trending_up:
            confidence_score += 0.3

        if obv_bullish:
            confidence_score += 0.3

        confirmed = confidence_score >= base_threshold

        # Generate reason
        reasons = []
        if volume_ratio >= 1.5:
            reasons.append(f"Volume surge {volume_ratio:.1f}x")
        elif volume_ratio >= 1.2:
            reasons.append(f"Volume tăng {volume_ratio:.1f}x")
        else:
            reasons.append(f"Volume {volume_ratio:.1f}x")

        if volume_trending_up:
            reasons.append("Volume trending up")
        else:
            reasons.append("Volume trending down")

        if obv_bullish:
            reasons.append("OBV bullish (accumulation)")
        else:
            reasons.append("OBV bearish (distribution)")

        return {
            "confirmed": confirmed,
            "reason": " | ".join(reasons),
            "surge": volume_ratio >= 1.5,
            "volume_ratio": volume_ratio,
            "volume_trending": volume_trending_up,
            "obv_bullish": obv_bullish,
            "confidence": confidence_score,
        }
