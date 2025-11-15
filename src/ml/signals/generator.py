from features import add_ml_features, get_feature_columns
from ml_models import MLPredictor
import numpy as np
from data_loader import load_data

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
        # Load models safely - fallback to technical-only if model loading fails
        try:
            self.predictor.load_models()
            self.model_version = "default"
            self.model_loaded = True
        except Exception as e:
            print(f"⚠️ ML model load failed: {e}")
            # Keep predictor but mark as not loaded — we'll fallback to technical analysis
            self.model_loaded = False

    def analyze(self, df, index_df=None):
        """Phân tích và tạo tín hiệu từ ML + Technical Analysis"""
        try:
            # Thêm ML features, yêu cầu có index_df
            if index_df is None:
                print("⚠️ Missing index_df for ML analysis, falling back to technical.")
                return self._fallback_technical_analysis(df)
            df = add_ml_features(df, index_df=index_df)

            # Kiểm tra xem có đủ data không
            if len(df) < 20:
                return self._fallback_technical_analysis(df)

            # Lấy data gần nhất
            latest = df.iloc[-1]

            # Chuẩn bị features cho ML
            feature_cols = get_feature_columns()
            available_features = [col for col in feature_cols if col in df.columns]

            # Yêu cầu tất cả các feature phải có mặt
            if len(available_features) == len(feature_cols):
                X = df[available_features].values

                # ML Prediction (only if models loaded)
                if getattr(self, "model_loaded", False):
                    try:
                        ml_scores = self.predictor.predict(X)
                        ml_score = ml_scores[-1] if len(ml_scores) > 0 else 0.5
                    except Exception as e:
                        print(f"⚠️ ML prediction failed: {e}")
                        ml_score = 0.5
                else:
                    # Model not available — use neutral ML score and rely on technical ensemble
                    print("⚠️ ML model not available, using neutral ML score")
                    ml_score = 0.5

                # Technical Analysis Score
                tech_score = self._calculate_technical_score(latest)

                # Ensemble Decision
                signal, confidence, reason = self._make_decision(
                    ml_score, tech_score, latest
                )

                # Calibrate confidence nếu có monitoring
                calibrated_confidence = confidence
                if use_monitoring and ml_monitor:
                    try:
                        calibrated_confidence = ml_monitor.calibrate_confidence(
                            confidence, model_version=self.model_version
                        )
                        if (
                            abs(calibrated_confidence - confidence) > 5
                        ):  # Significant difference
                            reason += f" | Calibrated: {calibrated_confidence:.0f}%"
                    except Exception as e:
                        print(f"⚠️ Lỗi calibrate confidence: {e}")

                return {
                    "signal": signal,
                    "confidence": int(calibrated_confidence),
                    "raw_confidence": confidence,
                    "ml_score": ml_score,
                    "technical_score": tech_score,
                    "reason": reason,
                    "price": latest["close"],
                    "rsi": latest.get("rsi", 50),
                    "ema_trend": (
                        "UP"
                        if latest.get("ema20", 0) > latest.get("ema50", 0)
                        else "DOWN"
                    ),
                }
            else:
                missing_features = set(feature_cols) - set(available_features)
                print(
                    f"⚠️ Không đủ features cho ML ({len(available_features)}/{len(feature_cols)}), thiếu: {missing_features}. Dùng technical analysis."
                )
                return self._fallback_technical_analysis(df)

        except Exception as e:
            print(f"⚠️ Lỗi ML analysis: {e}")
            return self._fallback_technical_analysis(df)

    def _fallback_technical_analysis(self, df):
        """Phân tích kỹ thuật khi ML không khả dụng"""
        try:
            # Cố gắng thêm các feature cơ bản mà không cần index_df
            if "rsi" not in df.columns:
                df = add_ml_features(df.copy(), index_df=None)
            latest = df.iloc[-1]

            # Simple technical signals
            signal = "HOLD"
            confidence = 0
            reasons = []

            # EMA crossover
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
            if rsi < 35:  # Nới lỏng điều kiện mua
                signal = "BUY"
                confidence += 40
                reasons.append(f"RSI potential reversal ({rsi:.1f})")
            elif rsi > 65:  # Nới lỏng điều kiện bán
                signal = "SELL"
                confidence += 40
                reasons.append(f"RSI potential peak ({rsi:.1f})")

            # MACD
            macd_diff = latest.get("macd_diff", 0)
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
                "technical_score": {"trend": 0, "momentum": 0, "volatility": 0},
                "reason": "Fallback: " + " | ".join(reasons),
                "price": latest["close"],
                "rsi": rsi,
                "ema_trend": "UP" if ema20 > ema50 else "DOWN",
            }
        except Exception as e:
            print(f"⚠️ Lỗi fallback analysis: {e}")
            # Return default values
            return {
                "signal": "HOLD",
                "confidence": 0,
                "ml_score": 0.5,
                "technical_score": {"trend": 0, "momentum": 0, "volatility": 0},
                "reason": "Lỗi phân tích",
                "price": 0,
                "rsi": 50,
                "ema_trend": "UNKNOWN",
            }

    def _calculate_technical_score(self, latest):
        """Tính điểm Technical Analysis"""
        score = {
            "trend": 0,  # -1 to 1
            "momentum": 0,  # -1 to 1
            "volatility": 0,  # 0 to 1
        }

        try:
            # Trend Score (EMA)
            ema20 = latest.get("ema20", 0)
            ema50 = latest.get("ema50", 0)
            if ema20 > ema50:
                score["trend"] = (ema20 - ema50) / ema50 if ema50 > 0 else 0
            else:
                score["trend"] = -(ema50 - ema20) / ema50 if ema50 > 0 else 0

            # Momentum Score (RSI)
            rsi = latest.get("rsi", 50)
            score["momentum"] = (rsi - 50) / 50  # Normalize from -1 to 1

            # Volatility Score (ATR)
            volatility = latest.get("volatility", 0)
            score["volatility"] = min(volatility * 10, 1)  # Normalize
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
        ml_signal = 1 if ml_score > 0.55 else (-1 if ml_score < 0.45 else 0)

        # Technical Signal
        tech_signal = 0

        try:
            # Trend
            if tech_score["trend"] > 0.01:  # Tăng nhẹ là đủ
                tech_signal += 0.5
                reasons.append(f"Trend Up ({tech_score['trend']:.2f})")
            elif tech_score["trend"] < -0.01:
                tech_signal -= 0.5
                reasons.append(f"Trend Down ({tech_score['trend']:.2f})")

            # Momentum
            if tech_score["momentum"] > 0.1:  # RSI > 55
                tech_signal += 0.5
                reasons.append(f"Momentum Up ({tech_score['momentum']:.2f})")
            elif tech_score["momentum"] < -0.1:  # RSI < 45
                tech_signal -= 0.5
                reasons.append(f"Momentum Down ({tech_score['momentum']:.2f})")

            # MACD
            macd_diff = latest.get("macd_diff", 0)
            if macd_diff > 0:
                tech_signal += 0.5
                reasons.append("MACD Bullish")
            else:
                tech_signal -= 0.5
                reasons.append("MACD Bearish")
        except Exception as e:
            print(f"⚠️ Lỗi tính tech signal: {e}")

        # Combined Signal
        # Trọng số ML cao hơn
        combined_signal = (ml_signal * 1.5) + (tech_signal * 0.5)

        # Confidence (0-100)
        confidence = min(abs(combined_signal) * 25 + abs(ml_score - 0.5) * 100, 100)

        # Decision
        if combined_signal >= 1.0:
            signal = "BUY"
            reasons.insert(0, f"ML({ml_score:.2f})")
        elif combined_signal <= -1.0:
            signal = "SELL"
            reasons.insert(0, f"ML({ml_score:.2f})")
        else:
            signal = "HOLD"
            reasons = [f"ML({ml_score:.2f})", "Neutral"]

        return signal, int(confidence), " | ".join(reasons)

    def train_models(self, df):
        """Train models với historical data, sử dụng các feature mới và class_weight."""
        print("🎓 Bắt đầu training models với pipeline nâng cao...")

        try:
            # 1. Load VN-Index data
            print("   - Đang tải dữ liệu VNINDEX cho việc tính toán features...")
            index_df = load_data(
                "VNINDEX", lookback="5y", use_cache=True, is_index=True
            )
            if index_df.empty:
                print("❌ Không thể tải dữ liệu VNINDEX. Dừng training.")
                return

            # 2. Add features
            print("   - Thêm các features mới (RS, Lag Features)...")
            df = add_ml_features(df, index_df=index_df)

            # 3. Prepare data
            print("   - Chuẩn bị dữ liệu cho training...")
            feature_cols = get_feature_columns()

            # Loại bỏ các hàng không có đủ dữ liệu
            df.dropna(subset=feature_cols + ["target"], inplace=True)

            if df.empty:
                print("❌ Không còn dữ liệu sau khi loại bỏ NaN. Dừng training.")
                return

            X = df[feature_cols].values
            y = df["target"].values

            # 4. Split train/test
            split_ratio = 0.8
            split_index = int(len(X) * split_ratio)
            X_train, y_train = X[:split_index], y[:split_index]
            X_test, y_test = X[split_index:], y[split_index:]
            print(f"   - Chia dữ liệu: {len(X_train)} train, {len(X_test)} test.")

            # 5. Scale features
            print("   - Scaling features...")
            X_train = self.predictor.scaler.fit_transform(X_train)
            X_test = self.predictor.scaler.transform(X_test)
            self.predictor.save_scaler()

            # 6. Train models with improved config
            print(
                "   - Huấn luyện RandomForest với class_weight='balanced' và params mới..."
            )
            self.predictor.train_random_forest(X_train, y_train)

            # (Optional) Train LSTM - hiện tại đang tắt để tập trung vào RandomForest
            # print("   - Huấn luyện LSTM...")
            # self.predictor.train_lstm(X_train, y_train)

            # 7. Evaluate model
            print("   - Đánh giá mô hình trên tập test...")
            self.predictor.evaluate(X_test, y_test)

            print("✅ Training và đánh giá hoàn tất!")
            self.model_loaded = True  # Đảm bảo model được đánh dấu là đã load

        except Exception as e:
            print(f"❌ Lỗi training models: {e}")
            import traceback

            traceback.print_exc()
