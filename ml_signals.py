from features import add_ml_features, get_feature_columns
from ml_models import MLPredictor
import numpy as np

class MLSignalGenerator:
    def __init__(self):
        self.predictor = MLPredictor()
        self.predictor.load_models()
        
    def analyze(self, df):
        """
        Phân tích và tạo tín hiệu từ ML + Technical Analysis
        
        Returns:
            dict: {
                'signal': 'BUY'|'SELL'|'HOLD',
                'confidence': 0-100,
                'ml_score': 0-1,
                'technical_score': dict,
                'reason': str
            }
        """
        # Thêm ML features
        df = add_ml_features(df)
        
        # Lấy data gần nhất
        latest = df.iloc[-1]
        
        # Chuẩn bị features cho ML
        feature_cols = get_feature_columns()
        X = df[feature_cols].values
        
        # ML Prediction
        ml_scores = self.predictor.predict(X)
        ml_score = ml_scores[-1]  # Score cho ngày cuối
        
        # Technical Analysis Score
        tech_score = self._calculate_technical_score(latest)
        
        # Ensemble Decision
        signal, confidence, reason = self._make_decision(ml_score, tech_score, latest)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'ml_score': ml_score,
            'technical_score': tech_score,
            'reason': reason,
            'price': latest['close'],
            'rsi': latest['rsi'],
            'ema_trend': 'UP' if latest['ema20'] > latest['ema50'] else 'DOWN'
        }
    
    def _calculate_technical_score(self, latest):
        """Tính điểm Technical Analysis"""
        score = {
            'trend': 0,      # -1 to 1
            'momentum': 0,   # -1 to 1
            'volatility': 0  # 0 to 1
        }
        
        # Trend Score (EMA)
        if latest['ema20'] > latest['ema50']:
            score['trend'] = (latest['ema20'] - latest['ema50']) / latest['ema50']
        else:
            score['trend'] = -(latest['ema50'] - latest['ema20']) / latest['ema50']
        
        # Momentum Score (RSI)
        if latest['rsi'] < 30:
            score['momentum'] = 1  # Oversold - bullish
        elif latest['rsi'] > 70:
            score['momentum'] = -1  # Overbought - bearish
        else:
            score['momentum'] = (50 - latest['rsi']) / 50  # Normalize
        
        # Volatility Score (ATR)
        score['volatility'] = min(latest['volatility'] * 10, 1)  # Normalize
        
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
        
        # Trend
        if tech_score['trend'] > 0.02:
            tech_signal += 1
            reasons.append(f"EMA20 > EMA50")
        elif tech_score['trend'] < -0.02:
            tech_signal -= 1
            reasons.append(f"EMA20 < EMA50")
        
        # Momentum
        if latest['rsi'] < 30:
            tech_signal += 1
            reasons.append(f"RSI oversold ({latest['rsi']:.1f})")
        elif latest['rsi'] > 70:
            tech_signal -= 1
            reasons.append(f"RSI overbought ({latest['rsi']:.1f})")
        
        # MACD
        if latest['macd_diff'] > 0:
            tech_signal += 0.5
            reasons.append("MACD bullish")
        else:
            tech_signal -= 0.5
            reasons.append("MACD bearish")
        
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
        
        # Add features
        df = add_ml_features(df)
        
        # Prepare data
        feature_cols = get_feature_columns()
        X = df[feature_cols].values
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