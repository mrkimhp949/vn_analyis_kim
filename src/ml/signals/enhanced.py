# -*- coding: utf-8 -*-
"""
Enhanced ML Signal Generator
Sử dụng enhanced models và features
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from src.ml.features.enhanced import add_enhanced_features, get_feature_columns
from src.ml.models.ensemble import EnhancedMLPredictor

logger = logging.getLogger(__name__)


class EnhancedMLSignalGenerator:
    """
    Enhanced ML Signal Generator với:
    - Multiple models (RF, XGBoost, LightGBM)
    - Ensemble predictions
    - Feature importance
    - Model explainability
    """

    def __init__(self):
        self.predictor = EnhancedMLPredictor()
        self.model_loaded = False

        # Try to load models
        try:
            self.model_loaded = self.predictor.load_models()
            if self.model_loaded:
                logger.info("✅ Enhanced ML models loaded successfully")
            else:
                logger.warning("⚠️ Enhanced models not found, will use fallback")
        except Exception:
            logger.error("❌ Error loading enhanced models")
            self.model_loaded = False

    def analyze(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        explain: bool = False,
        symbol: Optional[str] = None,  # Added for compatibility with orchestrator
    ) -> Dict:
        """
        Phân tích và tạo tín hiệu từ Enhanced ML + Technical Analysis

        Args:
            df: DataFrame với OHLCV data
            index_df: DataFrame của VNINDEX
            explain: Có explain prediction không
            symbol: Stock symbol (optional, for logging)

        Returns:
            Dict với signal, confidence, reasons, etc.
        """
        try:
            # Add enhanced features
            df_enhanced = add_enhanced_features(df, index_df)

            # Check data
            if len(df_enhanced) < 50:
                return self._fallback_technical_analysis(df_enhanced)

            # Get latest data
            latest = df_enhanced.iloc[-1]

            # Prepare features for ML
            feature_cols = get_feature_columns()

            # Check if all features exist
            missing_features = [col for col in feature_cols if col not in df_enhanced.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                return self._fallback_technical_analysis(df_enhanced)

            # Extract features - ensure numeric types
            X = df_enhanced[feature_cols].values.astype(np.float64)

            # Check for NaN
            try:
                if np.isnan(X[-1]).any():
                    logger.warning("NaN in features, using fallback")
                    return self._fallback_technical_analysis(df_enhanced)
            except (TypeError, ValueError) as e:
                logger.warning(f"Feature type error: {e}, using fallback")
                return self._fallback_technical_analysis(df_enhanced)

            # ML Prediction
            if self.model_loaded:
                try:
                    ml_scores = self.predictor.predict(X, use_ensemble=True)
                    ml_score = ml_scores[-1]
                except Exception:
                    logger.error("ML prediction error")
                    ml_score = 0.5
            else:
                ml_score = 0.5

            # Technical Analysis Score
            tech_score = self._calculate_technical_score(latest)

            # Ensemble Decision
            signal, confidence, reason = self._make_decision(ml_score, tech_score, latest)

            # Build result
            result = {
                "signal": signal,
                "confidence": int(confidence),
                "raw_confidence": confidence,
                "ml_score": float(ml_score),
                "technical_score": tech_score,
                "reason": reason,
                "price": float(latest["close"]),
                "rsi": float(latest.get("rsi", 50)),
                "ema_trend": ("UP" if latest.get("ema20", 0) > latest.get("ema50", 0) else "DOWN"),
            }

            # Add explanation if requested
            if explain and self.model_loaded:
                try:
                    explanation = self.predictor.explain_prediction(X, sample_idx=-1)
                    if explanation:
                        result["explanation"] = explanation
                        result["top_features"] = explanation["top_features"]
                except Exception:
                    logger.warning("Could not explain prediction")

            return result

        except Exception:
            logger.error("Error in enhanced ML analysis", exc_info=True)
            return self._fallback_technical_analysis(df)

    def _calculate_technical_score(self, latest: pd.Series) -> Dict:
        """Tính technical score từ indicators"""
        score = {"trend": 0, "momentum": 0, "volatility": 0, "volume": 0}

        try:
            # Trend (EMA alignment)
            ema20 = latest.get("ema20", 0)
            ema50 = latest.get("ema50", 0)
            if ema20 > ema50:
                score["trend"] = (ema20 - ema50) / ema50 if ema50 > 0 else 0
            else:
                score["trend"] = -(ema50 - ema20) / ema50 if ema50 > 0 else 0

            # Momentum (RSI + Stochastic)
            rsi = latest.get("rsi", 50)
            stoch_k = latest.get("stoch_k", 50)
            score["momentum"] = ((rsi - 50) / 50 + (stoch_k - 50) / 50) / 2

            # Volatility (ATR percentile)
            atr_pct = latest.get("atr_percentile", 0.5)
            score["volatility"] = atr_pct

            # Volume (OBV signal)
            obv_signal = latest.get("obv_signal", 0)
            volume_ratio = latest.get("volume_ratio", 1.0)
            score["volume"] = obv_signal * min(volume_ratio / 1.5, 1.0)

        except Exception:
            logger.error("Error calculating technical score")

        return score

    def _make_decision(self, ml_score: float, tech_score: Dict, latest: pd.Series) -> tuple:
        """
        Decision engine: Kết hợp ML + Technical

        UPDATED: Allow technical signals to override weak ML signals
        When ML is uncertain (0.35-0.65), give more weight to technical analysis

        Returns:
            (signal, confidence, reason)
        """
        reasons = []

        # Technical Signal - Calculate first
        tech_signal = 0

        # Trend (EMA alignment)
        if tech_score["trend"] > 0.02:
            tech_signal += 0.5
            reasons.append(f"Trend Up ({tech_score['trend']:.2f})")
        elif tech_score["trend"] < -0.02:
            tech_signal -= 0.5
            reasons.append(f"Trend Down ({tech_score['trend']:.2f})")

        # Momentum (RSI + Stochastic)
        if tech_score["momentum"] > 0.1:
            tech_signal += 0.5
            reasons.append(f"Momentum Up ({tech_score['momentum']:.2f})")
        elif tech_score["momentum"] < -0.1:
            tech_signal -= 0.5
            reasons.append(f"Momentum Down ({tech_score['momentum']:.2f})")

        # Volume confirmation
        if tech_score["volume"] > 0.5:
            tech_signal += 0.3
            reasons.append("Volume Confirm")

        # ADX (trend strength) - bonus for strong trends
        adx = latest.get("adx", 0)
        if adx > 25:
            tech_signal += 0.2
            reasons.append(f"Strong Trend (ADX {adx:.0f})")

        # RSI oversold/overbought - strong reversal signals
        rsi = latest.get("rsi", 50)
        if rsi < 30:
            tech_signal += 0.4
            reasons.append(f"RSI Oversold ({rsi:.0f})")
        elif rsi > 70:
            tech_signal -= 0.4
            reasons.append(f"RSI Overbought ({rsi:.0f})")

        # DYNAMIC ML WEIGHT based on ML confidence
        # When ML is uncertain (score near 0.5), trust technical more
        ml_uncertainty = 1 - abs(ml_score - 0.5) * 2  # 0 at extremes, 1 at 0.5

        # Adjust weights: ML gets less weight when uncertain
        ml_weight = 0.6 - (ml_uncertainty * 0.3)  # 0.3 to 0.6
        tech_weight = 1 - ml_weight  # 0.4 to 0.7

        # ML Signal with VERY relaxed thresholds
        # Only consider ML bearish if score < 0.40 (was 0.48)
        # Only consider ML bullish if score > 0.60 (was 0.52)
        # This creates a wider "neutral" zone where technical can dominate
        if ml_score > 0.60:
            ml_signal = 1
        elif ml_score < 0.40:
            ml_signal = -1
        else:
            ml_signal = 0  # Neutral - let technical decide

        # Combined Signal with dynamic weights
        combined_signal = (ml_signal * ml_weight) + (tech_signal * tech_weight)

        # Confidence calculation
        ml_confidence = abs(ml_score - 0.5) * 200
        ml_confidence = min(ml_confidence * 1.5, 80)
        tech_confidence = min(abs(tech_signal) * 50, 70)  # Increased tech confidence weight

        # Base confidence
        confidence = (ml_confidence * ml_weight) + (tech_confidence * tech_weight)

        # Bonus for alignment
        if ml_score > 0.55 and tech_signal > 0.3:
            confidence += 15
        elif ml_score < 0.45 and tech_signal < -0.3:
            confidence += 15
        # NEW: Bonus for strong technical when ML is neutral
        elif 0.40 <= ml_score <= 0.60 and abs(tech_signal) > 0.5:
            confidence += 10
            reasons.append("Tech Override")

        confidence = min(confidence, 100)

        # Decision - VERY RELAXED for current market
        # Allow BUY when combined >= 0.2 (was 0.3)
        # Allow technical to generate BUY even with neutral/weak ML
        if combined_signal >= 0.2:
            signal = "BUY"
            reasons.insert(0, f"ML({ml_score:.2f})")
        elif combined_signal <= -0.2:
            signal = "SELL"
            reasons.insert(0, f"ML({ml_score:.2f})")
        else:
            signal = "HOLD"
            reasons = [f"ML({ml_score:.2f})", "Neutral"]

        return signal, int(confidence), " | ".join(reasons)

    def _fallback_technical_analysis(self, df: pd.DataFrame) -> Dict:
        """Fallback to pure technical analysis"""
        try:
            if df.empty or len(df) < 20:
                return self._default_signal()

            latest = df.iloc[-1]  # Get latest row for technical analysis

            # Simple technical signals
            signal = "HOLD"
            confidence = 0
            reasons = []

            # EMA
            ema20 = latest.get("ema20", 0)
            ema50 = latest.get("ema50", 0)
            if ema20 > ema50:
                signal = "BUY"
                confidence += 30
                reasons.append("EMA20 > EMA50")
            else:
                signal = "SELL"
                confidence += 20
                reasons.append("EMA20 < EMA50")

            # RSI
            rsi = latest.get("rsi", 50)
            if rsi < 35:
                signal = "BUY"
                confidence += 40
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 65:
                signal = "SELL"
                confidence += 40
                reasons.append(f"RSI overbought ({rsi:.1f})")

            # MACD
            macd_diff = latest.get("macd_dif", 0)
            if macd_diff > 0:
                confidence += 10
                reasons.append("MACD bullish")
            else:
                confidence -= 10
                reasons.append("MACD bearish")

            return {
                "signal": signal,
                "confidence": min(confidence, 100),
                "ml_score": 0.5,
                "technical_score": {
                    "trend": 0,
                    "momentum": 0,
                    "volatility": 0,
                    "volume": 0,
                },
                "reason": "Fallback: " + " | ".join(reasons),
                "price": float(latest["close"]),
                "rsi": float(rsi),
                "ema_trend": "UP" if ema20 > ema50 else "DOWN",
            }

        except Exception:
            logger.error("Error in fallback analysis")
            return self._default_signal()

    def _default_signal(self) -> Dict:
        """Default signal when everything fails"""
        return {
            "signal": "HOLD",
            "confidence": 0,
            "ml_score": 0.5,
            "technical_score": {
                "trend": 0,
                "momentum": 0,
                "volatility": 0,
                "volume": 0,
            },
            "reason": "Insufficient data",
            "price": 0,
            "rsi": 50,
            "ema_trend": "UNKNOWN",
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from src.data.loader import load_data
    from utils.dataframe_utils import safe_get_latest

    print("\n" + "=" * 70)
    print("🧪 TESTING ENHANCED ML SIGNAL GENERATOR")
    print("=" * 70 + "\n")

    # Load data
    symbol = "VNM"
    df = load_data(symbol, lookback=200)
    index_df = load_data("VNINDEX", lookback=200, is_index=True)

    # Generate signal
    generator = EnhancedMLSignalGenerator()
    signal = generator.analyze(df, index_df, explain=True)

    # Print result
    print(f"📊 Symbol: {symbol}")
    print(f"📊 Signal: {signal['signal']}")
    print("📊 Confidence: {signal['confidence']}%")
    print(f"📊 ML Score: {signal['ml_score']:.4f}")
    print(f"📊 Reason: {signal['reason']}")

    if "explanation" in signal:
        print("\n🔍 Top contributing features:")
        for feature, shap_value in signal["top_features"]:
            print(f"   {feature:25s}: {shap_value:+.4f}")

    print("\n✅ Testing complete!")
