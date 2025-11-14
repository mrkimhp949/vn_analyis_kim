# -*- coding: utf-8 -*-
"""
Enhanced ML Signal Generator
Sử dụng enhanced models và features
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

from features_enhanced import add_enhanced_features, get_feature_columns
from ml_models_enhanced import EnhancedMLPredictor

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
        except Exception as e:
            logger.error(f"❌ Error loading enhanced models: {e}")
            self.model_loaded = False
    
    def analyze(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        explain: bool = False
    ) -> Dict:
        """
        Phân tích và tạo tín hiệu từ Enhanced ML + Technical Analysis
        
        Args:
            df: DataFrame với OHLCV data
            index_df: DataFrame của VNINDEX
            explain: Có explain prediction không
        
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
            
            # Extract features
            X = df_enhanced[feature_cols].values
            
            # Check for NaN
            if np.isnan(X[-1]).any():
                logger.warning("NaN in features, using fallback")
                return self._fallback_technical_analysis(df_enhanced)
            
            # ML Prediction
            if self.model_loaded:
                try:
                    ml_scores = self.predictor.predict(X, use_ensemble=True)
                    ml_score = ml_scores[-1]
                except Exception as e:
                    logger.error(f"ML prediction error: {e}")
                    ml_score = 0.5
            else:
                ml_score = 0.5
            
            # Technical Analysis Score
            tech_score = self._calculate_technical_score(latest)
            
            # Ensemble Decision
            signal, confidence, reason = self._make_decision(
                ml_score, tech_score, latest
            )
            
            # Build result
            result = {
                'signal': signal,
                'confidence': int(confidence),
                'raw_confidence': confidence,
                'ml_score': float(ml_score),
                'technical_score': tech_score,
                'reason': reason,
                'price': float(latest['close']),
                'rsi': float(latest.get('rsi', 50)),
                'ema_trend': 'UP' if latest.get('ema20', 0) > latest.get('ema50', 0) else 'DOWN',
            }
            
            # Add explanation if requested
            if explain and self.model_loaded:
                try:
                    explanation = self.predictor.explain_prediction(X, sample_idx=-1)
                    if explanation:
                        result['explanation'] = explanation
                        result['top_features'] = explanation['top_features']
                except Exception as e:
                    logger.warning(f"Could not explain prediction: {e}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error in enhanced ML analysis: {e}", exc_info=True)
            return self._fallback_technical_analysis(df)
    
    def _calculate_technical_score(self, latest: pd.Series) -> Dict:
        """Tính technical score từ indicators"""
        score = {
            'trend': 0,
            'momentum': 0,
            'volatility': 0,
            'volume': 0
        }
        
        try:
            # Trend (EMA alignment)
            ema20 = latest.get('ema20', 0)
            ema50 = latest.get('ema50', 0)
            if ema20 > ema50:
                score['trend'] = (ema20 - ema50) / ema50 if ema50 > 0 else 0
            else:
                score['trend'] = -(ema50 - ema20) / ema50 if ema50 > 0 else 0
            
            # Momentum (RSI + Stochastic)
            rsi = latest.get('rsi', 50)
            stoch_k = latest.get('stoch_k', 50)
            score['momentum'] = ((rsi - 50) / 50 + (stoch_k - 50) / 50) / 2
            
            # Volatility (ATR percentile)
            atr_pct = latest.get('atr_percentile', 0.5)
            score['volatility'] = atr_pct
            
            # Volume (OBV signal)
            obv_signal = latest.get('obv_signal', 0)
            volume_ratio = latest.get('volume_ratio', 1.0)
            score['volume'] = obv_signal * min(volume_ratio / 1.5, 1.0)
        
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
        
        return score
    
    def _make_decision(
        self,
        ml_score: float,
        tech_score: Dict,
        latest: pd.Series
    ) -> tuple:
        """
        Decision engine: Kết hợp ML + Technical
        
        Returns:
            (signal, confidence, reason)
        """
        reasons = []
        
        # ML Signal
        ml_signal = 1 if ml_score > 0.55 else (-1 if ml_score < 0.45 else 0)
        ml_confidence = abs(ml_score - 0.5) * 200  # 0-100
        
        # Technical Signal
        tech_signal = 0
        
        # Trend
        if tech_score['trend'] > 0.02:
            tech_signal += 0.5
            reasons.append(f"Trend Up ({tech_score['trend']:.2f})")
        elif tech_score['trend'] < -0.02:
            tech_signal -= 0.5
            reasons.append(f"Trend Down ({tech_score['trend']:.2f})")
        
        # Momentum
        if tech_score['momentum'] > 0.1:
            tech_signal += 0.5
            reasons.append(f"Momentum Up ({tech_score['momentum']:.2f})")
        elif tech_score['momentum'] < -0.1:
            tech_signal -= 0.5
            reasons.append(f"Momentum Down ({tech_score['momentum']:.2f})")
        
        # Volume
        if tech_score['volume'] > 0.5:
            tech_signal += 0.3
            reasons.append("Volume Confirm")
        
        # ADX (trend strength)
        adx = latest.get('adx', 0)
        if adx > 25:
            reasons.append(f"Strong Trend (ADX {adx:.0f})")
        
        # Combined Signal (ML weight = 60%, Technical = 40%)
        combined_signal = (ml_signal * 0.6) + (tech_signal * 0.4)
        
        # Confidence
        tech_confidence = min(abs(tech_signal) * 30, 50)
        confidence = (ml_confidence * 0.6) + (tech_confidence * 0.4)
        confidence = min(confidence, 100)
        
        # Decision
        if combined_signal >= 0.8:
            signal = 'BUY'
            reasons.insert(0, f"ML({ml_score:.2f})")
        elif combined_signal <= -0.8:
            signal = 'SELL'
            reasons.insert(0, f"ML({ml_score:.2f})")
        else:
            signal = 'HOLD'
            reasons = [f"ML({ml_score:.2f})", "Neutral"]
        
        return signal, int(confidence), " | ".join(reasons)
    
    def _fallback_technical_analysis(self, df: pd.DataFrame) -> Dict:
        """Fallback to pure technical analysis"""
        try:
            if df.empty or len(df) < 20:
                return self._default_signal()
            
            latest = df.iloc[-1]
            
            # Simple technical signals
            signal = 'HOLD'
            confidence = 0
            reasons = []
            
            # EMA
            ema20 = latest.get('ema20', 0)
            ema50 = latest.get('ema50', 0)
            if ema20 > ema50:
                signal = 'BUY'
                confidence += 30
                reasons.append("EMA20 > EMA50")
            else:
                signal = 'SELL'
                confidence += 20
                reasons.append("EMA20 < EMA50")
            
            # RSI
            rsi = latest.get('rsi', 50)
            if rsi < 35:
                signal = 'BUY'
                confidence += 40
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 65:
                signal = 'SELL'
                confidence += 40
                reasons.append(f"RSI overbought ({rsi:.1f})")
            
            # MACD
            macd_diff = latest.get('macd_diff', 0)
            if macd_diff > 0:
                confidence += 10
                reasons.append("MACD bullish")
            else:
                confidence -= 10
                reasons.append("MACD bearish")
            
            return {
                'signal': signal,
                'confidence': min(confidence, 100),
                'ml_score': 0.5,
                'technical_score': {'trend': 0, 'momentum': 0, 'volatility': 0, 'volume': 0},
                'reason': "Fallback: " + " | ".join(reasons),
                'price': float(latest['close']),
                'rsi': float(rsi),
                'ema_trend': 'UP' if ema20 > ema50 else 'DOWN',
            }
        
        except Exception as e:
            logger.error(f"Error in fallback analysis: {e}")
            return self._default_signal()
    
    def _default_signal(self) -> Dict:
        """Default signal when everything fails"""
        return {
            'signal': 'HOLD',
            'confidence': 0,
            'ml_score': 0.5,
            'technical_score': {'trend': 0, 'momentum': 0, 'volatility': 0, 'volume': 0},
            'reason': 'Insufficient data',
            'price': 0,
            'rsi': 50,
            'ema_trend': 'UNKNOWN',
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from data_loader import load_data
    
    print("\n" + "="*70)
    print("🧪 TESTING ENHANCED ML SIGNAL GENERATOR")
    print("="*70 + "\n")
    
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
    print(f"📊 Confidence: {signal['confidence']}%")
    print(f"📊 ML Score: {signal['ml_score']:.4f}")
    print(f"📊 Reason: {signal['reason']}")
    
    if 'explanation' in signal:
        print(f"\n🔍 Top contributing features:")
        for feature, shap_value in signal['top_features']:
            print(f"   {feature:25s}: {shap_value:+.4f}")
    
    print("\n✅ Testing complete!")
