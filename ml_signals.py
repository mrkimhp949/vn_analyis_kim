from features import add_ml_features, get_feature_columns
from ml_models import MLPredictor
import numpy as np

# ML Model Monitor
try:
    from ml_model_monitor import get_ml_model_monitor
    ml_monitor = get_ml_model_monitor()
    use_monitoring = True
except ImportError:
    ml_monitor = None
    use_monitoring = False

class MLSignalGenerator:
    def __init__(self):
        self.predictor = MLPredictor()
        self.predictor.load_models()
        self.model_version = "default"  # Can be updated when models are retrained
        
    def analyze(self, df):
        """Phân tích và tạo tín hiệu từ ML + Technical Analysis"""
        try:
            # Thêm ML features
            df = add_ml_features(df)
            
            # Kiểm tra xem có đủ data không
            if len(df) < 20:
                return self._fallback_technical_analysis(df)
                    
            # Lấy data gần nhất
            latest = df.iloc[-1]
            
            # Chuẩn bị features cho ML - GIẢM NGƯỠNG
            feature_cols = get_feature_columns()
            available_features = [col for col in feature_cols if col in df.columns]
            
            print(f"🔍 Features check: {len(available_features)} available of {len(feature_cols)}")
            
            # CHỈ CẦN 12/18 FEATURES LÀ CHẠY ĐƯỢC
            if len(available_features) >= 12:  
                X = df[available_features].values
                
                # ML Prediction
                ml_scores = self.predictor.predict(X)
                ml_score = ml_scores[-1] if len(ml_scores) > 0 else 0.5
                
                # Technical Analysis Score
                tech_score = self._calculate_technical_score(latest)
                
                # Ensemble Decision
                signal, confidence, reason = self._make_decision(ml_score, tech_score, latest)
                
                # Calibrate confidence nếu có monitoring
                calibrated_confidence = confidence
                if use_monitoring and ml_monitor:
                    try:
                        calibrated_confidence = ml_monitor.calibrate_confidence(
                            confidence,
                            model_version=self.model_version
                        )
                        if abs(calibrated_confidence - confidence) > 5:  # Significant difference
                            reason += f" | Calibrated: {calibrated_confidence:.0f}%"
                    except Exception as e:
                        print(f"⚠️ Lỗi calibrate confidence: {e}")
                
                # Note: Prediction recording will be done in bot_runner with symbol context
                
                return {
                    'signal': signal,
                    'confidence': int(calibrated_confidence),
                    'raw_confidence': confidence,
                    'ml_score': ml_score,
                    'technical_score': tech_score,
                    'reason': reason + f" | ML: {len(available_features)}/{len(feature_cols)} features",
                    'price': latest['close'],
                    'rsi': latest.get('rsi', 50),
                    'ema_trend': 'UP' if latest.get('ema20', 0) > latest.get('ema50', 0) else 'DOWN'
                }
            else:
                print(f"⚠️ Không đủ features cho ML ({len(available_features)}/{len(feature_cols)}), dùng technical analysis")
                return self._fallback_technical_analysis(df)
                
        except Exception as e:
            print(f"⚠️ Lỗi ML analysis: {e}")
            return self._fallback_technical_analysis(df)
        
    def _fallback_technical_analysis(self, df):
        """Phân tích kỹ thuật khi ML không khả dụng"""
        try:
            df = add_ml_features(df)
            latest = df.iloc[-1]
            
            # Simple technical signals
            signal = 'HOLD'
            confidence = 0
            reasons = []
            
            # EMA crossover
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
            if rsi < 30:
                signal = 'BUY'
                confidence += 40
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
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
                'technical_score': {'trend': 0, 'momentum': 0, 'volatility': 0},
                'reason': " | ".join(reasons),
                'price': latest['close'],
                'rsi': rsi,
                'ema_trend': 'UP' if ema20 > ema50 else 'DOWN'
            }
        except Exception as e:
            print(f"⚠️ Lỗi fallback analysis: {e}")
            # Return default values
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'ml_score': 0.5,
                'technical_score': {'trend': 0, 'momentum': 0, 'volatility': 0},
                'reason': "Lỗi phân tích",
                'price': 0,
                'rsi': 50,
                'ema_trend': 'UNKNOWN'
            }
    
    def _calculate_technical_score(self, latest):
        """Tính điểm Technical Analysis"""
        score = {
            'trend': 0,      # -1 to 1
            'momentum': 0,   # -1 to 1
            'volatility': 0  # 0 to 1
        }
        
        try:
            # Trend Score (EMA)
            ema20 = latest.get('ema20', 0)
            ema50 = latest.get('ema50', 0)
            if ema20 > ema50:
                score['trend'] = (ema20 - ema50) / ema50 if ema50 > 0 else 0
            else:
                score['trend'] = -(ema50 - ema20) / ema50 if ema50 > 0 else 0
            
            # Momentum Score (RSI)
            rsi = latest.get('rsi', 50)
            if rsi < 30:
                score['momentum'] = 1  # Oversold - bullish
            elif rsi > 70:
                score['momentum'] = -1  # Overbought - bearish
            else:
                score['momentum'] = (50 - rsi) / 50  # Normalize
            
            # Volatility Score (ATR)
            volatility = latest.get('volatility', 0)
            score['volatility'] = min(volatility * 10, 1)  # Normalize
        except Exception as e:
            print(f"⚠️ Lỗi tính technical score: {e}")
            
        return score
    
    def _make_decision(self, ml_score, tech_score, latest):
        """
        Decision Engine: Kết hợp ML + Technical
        
        ML Score: 0-1 (xác suất giá tăng)
        Tech Score: dict với trend, momentum, volatility
        """
        reasons = []
        
        # ML Signal
        ml_signal = 1 if ml_score > 0.6 else (-1 if ml_score < 0.4 else 0)
        
        # Technical Signal
        tech_signal = 0
        
        try:
            # Trend
            if tech_score['trend'] > 0.02:
                tech_signal += 1
                reasons.append(f"EMA20 > EMA50")
            elif tech_score['trend'] < -0.02:
                tech_signal -= 1
                reasons.append(f"EMA20 < EMA50")
            
            # Momentum
            rsi = latest.get('rsi', 50)
            if rsi < 30:
                tech_signal += 1
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                tech_signal -= 1
                reasons.append(f"RSI overbought ({rsi:.1f})")
            
            # MACD
            macd_diff = latest.get('macd_diff', 0)
            if macd_diff > 0:
                tech_signal += 0.5
                reasons.append("MACD bullish")
            else:
                tech_signal -= 0.5
                reasons.append("MACD bearish")
        except Exception as e:
            print(f"⚠️ Lỗi tính tech signal: {e}")
        
        # Combined Signal
        combined_signal = ml_signal + tech_signal
        
        # Confidence (0-100)
        confidence = min(abs(combined_signal) * 30 + abs(ml_score - 0.5) * 100, 100)
        
        # Decision
        if combined_signal >= 1.5:
            signal = 'BUY'
            reasons.insert(0, f"ML Score: {ml_score:.2f}")
        elif combined_signal <= -1.5:
            signal = 'SELL'
            reasons.insert(0, f"ML Score: {ml_score:.2f}")
        else:
            signal = 'HOLD'
            reasons = [f"ML Score: {ml_score:.2f}", "Không có tín hiệu rõ ràng"]
        
        return signal, int(confidence), " | ".join(reasons)
    
    def train_models(self, df):
        """Train models với historical data"""
        print("🎓 Bắt đầu training models...")
        
        try:
            # Add features
            df = add_ml_features(df)
            
            # Prepare data
            feature_cols = get_feature_columns()
            available_features = [col for col in feature_cols if col in df.columns]
            X = df[available_features].values
            y = df['target'].values
            
            # Split train/test
            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            
            # Scale features
            X_train = self.predictor.scaler.fit_transform(X_train)
            self.predictor.save_scaler()
            
            # Train models
            self.predictor.train_random_forest(X_train, y_train)
            self.predictor.train_lstm(X_train, y_train)
            
            print("✅ Training hoàn tất!")
        except Exception as e:
            print(f"❌ Lỗi training models: {e}")