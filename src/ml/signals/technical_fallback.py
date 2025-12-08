"""
Advanced Technical Analysis Fallback
Used when ML model is unavailable or fails
Provides sophisticated signal generation using multiple technical indicators
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from utils.dataframe_utils import safe_get_latest, safe_rolling_operation

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignal:
    """Container for technical analysis signal"""

    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: int  # 0-100
    score: float  # Combined score from all indicators
    components: Dict[str, float]  # Individual component scores
    reason: str  # Human-readable reason
    ml_score: float = 0.5  # Neutral ML score when ML unavailable


class AdvancedTechnicalAnalysis:
    """
    Advanced technical analysis with multiple indicators
    Much more sophisticated than the basic fallback
    """

    def __init__(self):
        # Weights for each component
        self.weights = {
            "trend": 0.25,  # 25%
            "momentum": 0.25,  # 25%
            "volume": 0.20,  # 20%
            "volatility": 0.15,  # 15%
            "support_resistance": 0.15,  # 15%
        }

    def analyze(self, df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None) -> TechnicalSignal:
        """
        Comprehensive technical analysis

        Args:
            df: OHLCV dataframe with indicators
            index_df: Optional market index for relative strength

        Returns:
            TechnicalSignal with detailed analysis
        """
        try:
            if df is None or df.empty or len(df) < 50:
                return self._neutral_signal("Insufficient data")

            # Calculate component scores
            trend_score = self._analyze_trend(df)
            momentum_score = self._analyze_momentum(df)
            volume_score = self._analyze_volume(df)
            volatility_score = self._analyze_volatility(df)
            sr_score = self._analyze_support_resistance(df)

            # Relative strength if index available
            if index_df is not None and not index_df.empty:
                rs_score = self._analyze_relative_strength(df, index_df)
                # Add RS as bonus/penalty to trend
                trend_score = trend_score * 0.8 + rs_score * 0.2

            # Combine scores
            components = {
                "trend": trend_score,
                "momentum": momentum_score,
                "volume": volume_score,
                "volatility": volatility_score,
                "support_resistance": sr_score,
            }

            combined_score = sum(components[k] * self.weights[k] for k in components.keys())

            # Generate signal and confidence
            signal, confidence, reason = self._make_decision(combined_score, components)

            return TechnicalSignal(
                signal=signal,
                confidence=confidence,
                score=combined_score,
                components=components,
                reason=reason,
                ml_score=0.5,  # Neutral when using technical fallback
            )

        except Exception as e:
            logger.error(f"❌ Error in advanced technical analysis: {e}")
            return self._neutral_signal(f"Analysis error: {str(e)}")

    def _analyze_trend(self, df: pd.DataFrame) -> float:
        """
        Analyze trend using multiple moving averages
        Returns: -1.0 to +1.0
        """
        try:
            # Get EMAs
            ema20 = df["close"].ewm(span=20).mean()
            ema50 = df["close"].ewm(span=50).mean() if len(df) >= 50 else ema20
            ema200 = df["close"].ewm(span=200).mean() if len(df) >= 200 else ema50

            current_price = safe_get_latest(df, "close", 0)
            latest_ema20 = ema20.iloc[-1]
            latest_ema50 = ema50.iloc[-1]
            latest_ema200 = ema200.iloc[-1]

            score = 0.0

            # 1. Price position vs EMAs
            if current_price > latest_ema20:
                score += 0.3
            if current_price > latest_ema50:
                score += 0.2
            if current_price > latest_ema200:
                score += 0.1

            # 2. EMA alignment (20 > 50 > 200 = perfect uptrend)
            if latest_ema20 > latest_ema50:
                score += 0.2
            if latest_ema50 > latest_ema200:
                score += 0.1

            # 3. EMA slope (trending up/down)
            if len(ema20) >= 5:
                ema20_slope = (ema20.iloc[-1] - ema20.iloc[-5]) / ema20.iloc[-5]
                score += ema20_slope * 2  # Amplify slope contribution

            # 4. Price crossovers (early signals)
            if len(df) >= 2:
                prev_price = df["close"].iloc[-2]
                prev_ema20 = ema20.iloc[-2]

                # Bullish crossover
                if prev_price <= prev_ema20 and current_price > latest_ema20:
                    score += 0.15
                # Bearish crossover
                elif prev_price >= prev_ema20 and current_price < latest_ema20:
                    score -= 0.15

            # Normalize to -1 to +1
            return max(-1.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"⚠️ Trend analysis error: {e}")
            return 0.0

    def _analyze_momentum(self, df: pd.DataFrame) -> float:
        """
        Analyze momentum using RSI, MACD, Stochastic
        Returns: -1.0 to +1.0
        """
        try:
            score = 0.0

            # 1. RSI Analysis
            if "rsi" in df.columns:
                rsi = safe_get_latest(df, "rsi", 50)
                if not pd.isna(rsi):
                    # RSI scoring: oversold = very bullish, overbought = very bearish
                    if rsi < 30:
                        score += 0.4  # Strong buy
                    elif 30 <= rsi <= 45:
                        score += 0.2  # Moderate buy
                    elif 45 < rsi < 55:
                        score += 0.0  # Neutral
                    elif 55 <= rsi <= 70:
                        score -= 0.1  # Slight bearish
                    else:  # > 70
                        score -= 0.3  # Overbought

            # 2. MACD Analysis
            if "macd" in df.columns and "macd_signal" in df.columns:
                macd = safe_get_latest(df, "macd", 0)
                macd_signal = safe_get_latest(df, "macd_signal", 0)
                macd_diff = macd - macd_signal

                if macd_diff > 0:
                    score += 0.3  # Bullish MACD
                else:
                    score -= 0.2  # Bearish MACD

                # MACD crossover
                if len(df) >= 2:
                    prev_macd = df["macd"].iloc[-2] if not pd.isna(df["macd"].iloc[-2]) else macd
                    prev_signal = (
                        df["macd_signal"].iloc[-2]
                        if not pd.isna(df["macd_signal"].iloc[-2])
                        else macd_signal
                    )

                    # Bullish crossover
                    if prev_macd <= prev_signal and macd > macd_signal:
                        score += 0.2
                    # Bearish crossover
                    elif prev_macd >= prev_signal and macd < macd_signal:
                        score -= 0.2

            # 3. Stochastic (if available)
            if "stoch_k" in df.columns and "stoch_d" in df.columns:
                stoch_k = safe_get_latest(df, "stoch_k", 50)
                stoch_d = safe_get_latest(df, "stoch_d", 50)

                if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                    # Oversold/overbought
                    if stoch_k < 20:
                        score += 0.2
                    elif stoch_k > 80:
                        score -= 0.2

                    # K/D crossover
                    if stoch_k > stoch_d:
                        score += 0.1
                    else:
                        score -= 0.1

            return max(-1.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"⚠️ Momentum analysis error: {e}")
            return 0.0

    def _analyze_volume(self, df: pd.DataFrame) -> float:
        """
        Analyze volume patterns and OBV
        Returns: -1.0 to +1.0
        """
        try:
            if "volume" not in df.columns or len(df) < 20:
                return 0.0

            score = 0.0

            # 1. Volume ratio
            current_volume = safe_get_latest(df, "volume", 0)
            avg_volume_20 = safe_rolling_operation(df, "volume", 20, "mean", 0)

            if avg_volume_20 > 0:
                volume_ratio = current_volume / avg_volume_20

                if volume_ratio >= 1.5:
                    score += 0.4  # Strong volume surge
                elif volume_ratio >= 1.2:
                    score += 0.2  # Good volume
                elif volume_ratio < 0.8:
                    score -= 0.2  # Low volume

            # 2. Volume trend (5-day vs 20-day)
            avg_volume_5 = safe_rolling_operation(df, "volume", 5, "mean", 0)
            if avg_volume_20 > 0:
                if avg_volume_5 > avg_volume_20:
                    score += 0.2  # Volume trending up
                else:
                    score -= 0.1  # Volume declining

            # 3. OBV Analysis
            obv = self._calculate_obv(df)
            if len(obv) >= 20:
                obv_ma_5 = obv.rolling(5).mean().iloc[-1]
                obv_ma_20 = obv.rolling(20).mean().iloc[-1]

                if obv_ma_5 > obv_ma_20:
                    score += 0.3  # OBV bullish (accumulation)
                else:
                    score -= 0.2  # OBV bearish (distribution)

            # 4. Price-Volume correlation
            if len(df) >= 10:
                recent_price = df["close"].tail(10)
                recent_volume = df["volume"].tail(10)
                correlation = recent_price.corr(recent_volume)

                if not pd.isna(correlation):
                    if correlation > 0.5:
                        score += 0.1  # Price and volume moving together
                    elif correlation < -0.5:
                        score -= 0.1  # Divergence

            return max(-1.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"⚠️ Volume analysis error: {e}")
            return 0.0

    def _analyze_volatility(self, df: pd.DataFrame) -> float:
        """
        Analyze volatility using ATR and Bollinger Bands
        Returns: -1.0 to +1.0 (optimal volatility = positive score)
        """
        try:
            score = 0.0

            # 1. ATR Analysis
            if "atr" in df.columns:
                atr = safe_get_latest(df, "atr", 0)
                price = safe_get_latest(df, "close", 0)

                if price > 0:
                    volatility_pct = (atr / price) * 100

                    # Optimal volatility: 2-3%
                    if 2.0 <= volatility_pct <= 3.0:
                        score += 0.5  # Optimal
                    elif volatility_pct < 2.0:
                        score += 0.2  # Low but acceptable
                    elif volatility_pct > 4.0:
                        score -= 0.3  # Too high (risky)

            # 2. Bollinger Bands
            if all(col in df.columns for col in ["bb_upper", "bb_lower", "bb_middle"]):
                price = safe_get_latest(df, "close", 0)
                bb_upper = safe_get_latest(df, "bb_upper", 0)
                bb_lower = safe_get_latest(df, "bb_lower", 0)
                bb_middle = safe_get_latest(df, "bb_middle", 0)

                if bb_upper > bb_lower:
                    # Position in Bollinger Bands
                    bb_position = (price - bb_lower) / (bb_upper - bb_lower)

                    if bb_position < 0.2:
                        score += 0.3  # Near lower band (oversold)
                    elif bb_position > 0.8:
                        score -= 0.2  # Near upper band (overbought)

                    # Bollinger Squeeze (low volatility → potential breakout)
                    bb_width = (bb_upper - bb_lower) / bb_middle
                    if len(df) >= 20:
                        avg_width = (
                            (df["bb_upper"] - df["bb_lower"]).tail(20) / df["bb_middle"].tail(20)
                        ).mean()
                        if bb_width < avg_width * 0.7:
                            score += 0.2  # Squeeze → potential breakout

            return max(-1.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"⚠️ Volatility analysis error: {e}")
            return 0.0

    def _analyze_support_resistance(self, df: pd.DataFrame) -> float:
        """
        Analyze support/resistance levels
        Returns: -1.0 to +1.0
        """
        try:
            if len(df) < 20:
                return 0.0

            score = 0.0
            current_price = safe_get_latest(df, "close", 0)

            # 1. Support/Resistance from recent highs/lows
            support = safe_rolling_operation(df, "low", 20, "min", 0)
            resistance = safe_rolling_operation(df, "high", 20, "max", 0)

            distance_to_support = (
                ((current_price - support) / support) * 100 if support > 0 else 100
            )
            distance_to_resistance = (
                ((resistance - current_price) / current_price) * 100 if current_price > 0 else 100
            )

            # Near support = bullish
            if distance_to_support <= 3.0:
                score += 0.5

            # Too close to resistance = bearish
            if distance_to_resistance <= 2.0:
                score -= 0.4

            # 2. Support bounce detection
            if len(df) >= 3:
                recent_low = safe_rolling_operation(df, "low", 3, "min", 0)
                if abs(recent_low - support) / support < 0.02:  # Within 2% of support
                    prev_close = df["close"].iloc[-2] if len(df) >= 2 else current_price
                    if current_price > prev_close:  # Bouncing up
                        score += 0.4  # Strong reversal signal

            # 3. Breakout detection
            if len(df) >= 5:
                recent_high = safe_rolling_operation(df, "high", 5, "max", 0)
                if current_price > recent_high * 1.01:  # Broke above recent high
                    score += 0.3  # Breakout

            return max(-1.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"⚠️ Support/Resistance analysis error: {e}")
            return 0.0

    def _analyze_relative_strength(self, df: pd.DataFrame, index_df: pd.DataFrame) -> float:
        """
        Analyze relative strength vs market index
        Returns: -1.0 to +1.0
        """
        try:
            if "rs" in df.columns:
                rs = safe_get_latest(df, "rs", 1.0)
                if not pd.isna(rs):
                    # RS > 1.0 = outperforming market
                    if rs > 1.1:
                        return 0.5  # Strong outperformance
                    elif rs > 1.0:
                        return 0.2  # Moderate outperformance
                    elif rs < 0.9:
                        return -0.3  # Underperformance
                    else:
                        return 0.0  # Neutral

            return 0.0

        except Exception as e:
            logger.warning(f"⚠️ Relative strength analysis error: {e}")
            return 0.0

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])

        return pd.Series(obv, index=df.index)

    def _make_decision(
        self, combined_score: float, components: Dict[str, float]
    ) -> tuple[str, int, str]:
        """
        Generate signal, confidence, and reason from scores

        Args:
            combined_score: Combined weighted score (-1 to +1)
            components: Individual component scores

        Returns:
            (signal, confidence, reason)
        """
        # Generate signal
        if combined_score >= 0.3:
            signal = "BUY"
        elif combined_score <= -0.3:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Calculate confidence (0-100)
        # Higher absolute score = higher confidence
        confidence = min(int(abs(combined_score) * 100), 100)

        # Ensure minimum confidence for BUY/SELL signals
        if signal in ["BUY", "SELL"]:
            confidence = max(confidence, 40)

        # Generate reason
        reasons = []

        # Highlight strongest components
        sorted_components = sorted(components.items(), key=lambda x: abs(x[1]), reverse=True)

        for component, score in sorted_components[:3]:  # Top 3 components
            if abs(score) > 0.2:  # Only mention significant components
                direction = "bullish" if score > 0 else "bearish"
                reasons.append(f"{component.capitalize()} {direction} ({score:+.2f})")

        reason = " | ".join(reasons) if reasons else f"Combined score: {combined_score:.2f}"

        return signal, confidence, f"Technical: {reason}"

    def _neutral_signal(self, reason: str) -> TechnicalSignal:
        """Return neutral signal"""
        return TechnicalSignal(
            signal="HOLD",
            confidence=0,
            score=0.0,
            components={},
            reason=reason,
            ml_score=0.5,
        )


# Singleton instance
_technical_analyzer = None


def get_technical_analyzer() -> AdvancedTechnicalAnalysis:
    """Get technical analyzer singleton"""
    global _technical_analyzer
    if _technical_analyzer is None:
        _technical_analyzer = AdvancedTechnicalAnalysis()
    return _technical_analyzer


# Convenience function
def analyze_technical(df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None) -> TechnicalSignal:
    """Analyze using advanced technical analysis"""
    analyzer = get_technical_analyzer()
    return analyzer.analyze(df, index_df)


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("\n" + "=" * 70)
    print("🧪 TESTING ADVANCED TECHNICAL ANALYSIS")
    print("=" * 70 + "\n")

    # Test with sample data
    from src.data.loader import load_data
    from src.ml.features.enhanced_v2 import add_ml_features

    symbol = "VNM"
    print(f"📊 Analyzing {symbol}...")

    df = load_data(symbol, lookback=200)
    index_df = load_data("VNINDEX", lookback=200, is_index=True)

    df = add_ml_features(df, index_df=index_df)

    signal = analyze_technical(df, index_df)

    print(f"\n✅ Technical Signal for {symbol}:")
    print(f"  Signal: {signal.signal}")
    print(f"  Confidence: {signal.confidence}%")
    print(f"  Combined Score: {signal.score:.2f}")
    print(f"  Components:")
    for component, score in signal.components.items():
        print(f"    {component.capitalize()}: {score:+.2f}")
    print(f"  Reason: {signal.reason}")

    print("\n" + "=" * 70)
