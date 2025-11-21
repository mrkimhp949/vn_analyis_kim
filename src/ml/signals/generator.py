from src.data.loader import load_data
import logging
import traceback
from src.ml.features.technical import add_ml_features, get_feature_columns
from src.ml.models.predictor import MLPredictor
from utils.dataframe_utils import safe_get_latest

# ML Model Monitor
try:
    from src.ml.monitor import get_ml_model_monitor

    ml_monitor = get_ml_model_monitor()
    use_monitoring = True
except ImportError:
    ml_monitor = None
    use_monitoring = False

# Timing Analyzer
try:
    from src.ml.signals.timing import add_timing_to_signal

    use_timing = True
except ImportError:
    use_timing = False
    logging.getLogger(__name__).warning("Timing module not available")

# Quality Scorer
try:
    from src.ml.signals.quality_scorer import add_quality_score

    use_quality_scorer = True
except ImportError:
    use_quality_scorer = False
    logging.getLogger(__name__).warning("Quality scorer not available")


class MLSignalGenerator:
    def __init__(self):
        self.predictor = MLPredictor()
        # Load models safely - fallback to technical-only if model loading fails
        try:
            self.predictor.load_models()
            self.model_version = "default"
            self.model_loaded = True
        except Exception:
            print("⚠️ ML model load failed")
            # Keep predictor but mark as not loaded — we'll fallback to technical analysis
            self.model_loaded = False

        # ENHANCEMENT: Confidence calibration based on historical accuracy
        self._confidence_history = []  # Track (predicted_conf, actual_result) pairs
        self._max_history = 100  # Keep last 100 predictions

    def analyze(self, df, index_df=None):
        """Phân tích và tạo tín hiệu từ ML + Technical Analysis"""
        try:
            # Validate input data
            if df is None or df.empty:
                raise ValueError("Empty or None dataframe")

            if len(df) < 50:
                raise ValueError(f"Insufficient data: {len(df)} rows, need at least 50")

            # Thêm ML features, cố gắng tự nạp VNINDEX nếu thiếu index_df
            if index_df is None or getattr(index_df, "empty", True):
                try:
                    index_df = load_data("VNINDEX", lookback=200, is_index=True)
                except Exception:
                    index_df = None

            # Require OHLCV basics before attempting ML features
            required_cols = {"open", "high", "low", "close", "volume"}
            if not required_cols.issubset(set(df.columns)):
                # Data incomplete -> fallback to technical
                return self._fallback_technical_analysis(df)

            if index_df is None or getattr(index_df, "empty", True):
                # Không đủ điều kiện cho ML features, fallback sang technical
                return self._fallback_technical_analysis(df)

            df = add_ml_features(df, index_df=index_df)

            # Kiểm tra xem có đủ data không
            if len(df) < 20:
                return self._fallback_technical_analysis(df)

            # Lấy data gần nhất
            # Use safe access instead of df.iloc[-1]
            from utils.dataframe_utils import safe_get_latest

            latest = df.iloc[-1]  # Get latest row for analysis

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
                    except Exception:
                        print("⚠️ ML prediction failed")
                        ml_score = 0.5
                else:
                    # Model not available — use neutral ML score and rely on technical ensemble
                    print("⚠️ ML model not available, using neutral ML score")
                    ml_score = 0.5

                # Technical Analysis Score
                tech_score = self._calculate_technical_score(latest)

                # Ensemble Decision
                signal, confidence, reason = self._make_decision(ml_score, tech_score, latest)

                # ENHANCEMENT: Calibrate confidence based on historical accuracy
                calibrated_confidence = self._calibrate_confidence(confidence, signal, ml_score)

                # Also use ml_monitor if available
                if use_monitoring and ml_monitor:
                    try:
                        monitor_calibrated = ml_monitor.calibrate_confidence(
                            calibrated_confidence, model_version=self.model_version
                        )
                        if abs(monitor_calibrated - calibrated_confidence) > 5:
                            calibrated_confidence = monitor_calibrated
                            reason += f" | Monitor calibrated"
                    except Exception:
                        print("⚠️ Lỗi calibrate confidence from monitor")

                result = {
                    "signal": signal,
                    "action": signal,  # For timing analyzer
                    "confidence": int(calibrated_confidence),
                    "raw_confidence": confidence,
                    "ml_score": ml_score,
                    "technical_score": tech_score,
                    "reason": reason,
                    "price": latest["close"],
                    "rsi": latest.get("rsi", 50),
                    "ema_trend": (
                        "UP" if latest.get("ema20", 0) > latest.get("ema50", 0) else "DOWN"
                    ),
                }

                # Add timing analysis
                if use_timing:
                    try:
                        result = add_timing_to_signal(result)
                    except Exception as e:
                        logging.getLogger(__name__).debug(f"Timing analysis failed: {e}")

                # Add quality score
                if use_quality_scorer:
                    try:
                        result = add_quality_score(result)
                    except Exception as e:
                        logging.getLogger(__name__).debug(f"Quality scoring failed: {e}")

                return result
            else:
                missing_features = set(feature_cols) - set(available_features)
                print(
                    f"⚠️ Không đủ features cho ML ({len(available_features)}/{len(feature_cols)}), "
                    f"thiếu: {missing_features}. Dùng technical analysis."
                )
                return self._fallback_technical_analysis(df)

        except Exception as e:
            # Surface the real error to help diagnose recurring "Lỗi ML analysis" in logs
            logging.getLogger(__name__).warning(f"⚠️ Lỗi ML analysis: {str(e)}")
            try:
                traceback.print_exc()
            except Exception:
                pass
            return self._fallback_technical_analysis(df)

    def _fallback_technical_analysis(self, df):
        """
        IMPROVED: Use advanced technical analysis fallback
        Much more sophisticated than basic EMA/RSI checks
        """
        try:
            # Use advanced technical fallback
            from src.ml.signals.technical_fallback import analyze_technical

            # Try to load index for better analysis
            index_df = None
            try:
                index_df = load_data("VNINDEX", lookback=200, is_index=True)
            except Exception:
                pass

            # Run advanced technical analysis
            tech_signal = analyze_technical(df, index_df)

            return {
                "signal": tech_signal.signal,
                "confidence": tech_signal.confidence,
                "ml_score": tech_signal.ml_score,  # 0.5 for technical-only
                "technical_score": tech_signal.components,
                "reason": tech_signal.reason,
                "price": safe_get_latest(df, "close", 0),
                "rsi": safe_get_latest(df, "rsi", 50),
                "ema_trend": "UP" if tech_signal.components.get("trend", 0) > 0 else "DOWN",
            }

        except Exception as e:
            print(f"⚠️ Advanced fallback failed: {e}, using basic fallback")

            # BASIC FALLBACK (last resort)
            try:
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
                if rsi < 35:
                    signal = "BUY"
                    confidence += 40
                    reasons.append(f"RSI reversal ({rsi:.1f})")
                elif rsi > 65:
                    signal = "SELL"
                    confidence += 40
                    reasons.append(f"RSI peak ({rsi:.1f})")

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
                    "technical_score": {"trend": 0, "momentum": 0, "volatility": 0},
                    "reason": "Basic Fallback: " + " | ".join(reasons),
                    "price": latest["close"],
                    "rsi": rsi,
                    "ema_trend": "UP" if ema20 > ema50 else "DOWN",
                }
            except Exception:
                print("⚠️ Basic fallback failed")
                return {
                    "signal": "HOLD",
                    "confidence": 0,
                    "ml_score": 0.5,
                    "technical_score": {"trend": 0, "momentum": 0, "volatility": 0},
                    "reason": "Fallback error",
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
        except Exception:
            print("⚠️ Lỗi tính technical score")

        return score

    def _make_decision(self, ml_score, tech_score, latest):
        """
        Decision Engine: Kết hợp ML + Technical + Volume

        ML Score: 0-1 (xác suất giá tăng)
        Tech Score: dict với trend, momentum, volatility
        Latest: Latest bar data with volume
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
            macd_diff = latest.get("macd_dif", 0)
            if macd_diff > 0:
                tech_signal += 0.5
                reasons.append("MACD Bullish")
            else:
                tech_signal -= 0.5
                reasons.append("MACD Bearish")
        except Exception:
            print("⚠️ Lỗi tính tech signal")

        # VOLUME CONFIRMATION (NEW!)
        volume_signal = 0
        try:
            volume_ratio = latest.get("volume_ratio", 1.0)
            current_volume = latest.get("volume", 0)
            volume_sma20 = latest.get("volume_sma20", current_volume)

            # Calculate actual volume ratio if not available
            if volume_ratio == 1.0 and volume_sma20 > 0:
                volume_ratio = current_volume / volume_sma20

            # Volume confirmation logic
            if volume_ratio >= 1.5:
                volume_signal = 0.6  # Strong volume surge
                reasons.append(f"Volume Surge ({volume_ratio:.1f}x)")
            elif volume_ratio >= 1.2:
                volume_signal = 0.3  # Good volume
                reasons.append(f"High Volume ({volume_ratio:.1f}x)")
            elif volume_ratio < 0.8:
                volume_signal = -0.2  # Low volume warning
                reasons.append(f"Low Volume ({volume_ratio:.1f}x)")
            else:
                volume_signal = 0.0  # Normal volume
                reasons.append(f"Normal Volume ({volume_ratio:.1f}x)")

            # OBV signal if available
            obv_signal = latest.get("obv_signal", 0)
            if obv_signal == 1:
                volume_signal += 0.2
                reasons.append("OBV+")
            elif obv_signal == 0:
                volume_signal -= 0.1

        except Exception as e:
            print(f"⚠️ Lỗi tính volume signal: {e}")
            volume_signal = 0

        # Combined Signal
        # ML (45%) + Technical (30%) + Volume (25%)
        combined_signal = (ml_signal * 1.35) + (tech_signal * 0.45) + (volume_signal * 0.75)

        # Confidence (0-100)
        # Higher weight for ML and volume confirmation
        base_confidence = abs(combined_signal) * 25 + abs(ml_score - 0.5) * 100
        volume_boost = max(0, (volume_ratio - 1.0) * 10) if "volume_ratio" in locals() else 0
        confidence = min(base_confidence + volume_boost, 100)

        # =================================================================
        # DYNAMIC THRESHOLD BASED ON MARKET REGIME
        # =================================================================
        buy_threshold = 0.85  # Default (BALANCED)
        sell_threshold = -0.85  # Default

        # Try to detect market regime for dynamic threshold adjustment
        try:
            from src.market.regime_detector import detect_regime

            # Get VN-Index data for regime detection (need 200+ bars)
            vnindex_df = None
            try:
                vnindex_df = load_data("VNINDEX", lookback=250, is_index=True)
            except Exception as e:
                logging.getLogger(__name__).debug(f"Could not load VNINDEX: {e}")

            # Check if we have enough data for regime detection
            if vnindex_df is not None and not vnindex_df.empty and len(vnindex_df) >= 200:
                regime_obj = detect_regime(vnindex_df)
                regime = regime_obj.regime
                conf = regime_obj.confidence

                # Only adjust thresholds if confidence is sufficient (≥40%)
                if conf >= 40:
                    # Adjust thresholds based on regime
                    if regime == "BULL":
                        # Bull market - lower threshold for more opportunities
                        buy_threshold = 0.75
                        sell_threshold = -0.95  # Harder to sell
                        reasons.append(f"🐂 Bull ({conf:.0f}%)")
                    elif regime == "BEAR":
                        # Bear market - higher threshold for safety
                        buy_threshold = 0.95  # Very selective
                        sell_threshold = -0.75  # Easier to sell
                        reasons.append(f"🐻 Bear ({conf:.0f}%)")
                    elif regime == "SIDEWAYS":
                        # Sideways - moderate threshold (default)
                        buy_threshold = 0.85
                        sell_threshold = -0.85
                        reasons.append(f"📊 Sideways ({conf:.0f}%)")
                    elif regime == "HIGH_VOLATILITY":
                        # High volatility - much higher threshold
                        buy_threshold = 1.0  # Very conservative
                        sell_threshold = -0.70
                        reasons.append(f"⚡ High Vol ({conf:.0f}%)")

                    logging.getLogger(__name__).debug(
                        f"Dynamic threshold: BUY={buy_threshold:.2f}, "
                        f"SELL={sell_threshold:.2f} (regime={regime}, conf={conf:.0f}%)"
                    )
                else:
                    # Low confidence - use default thresholds
                    logging.getLogger(__name__).debug(
                        f"Regime confidence too low ({conf:.0f}%), using default thresholds"
                    )
            else:
                # Not enough data - use default thresholds
                data_len = len(vnindex_df) if vnindex_df is not None else 0
                logging.getLogger(__name__).debug(
                    f"Insufficient VNINDEX data ({data_len} bars, need 200+), using default thresholds"
                )
        except Exception as e:
            logging.getLogger(__name__).debug(f"Regime detection error: {e}")
            # Use default thresholds (no action needed)

        # Decision with dynamic thresholds
        if combined_signal >= buy_threshold:
            signal = "BUY"
            reasons.insert(0, f"ML({ml_score:.2f})")
        elif combined_signal <= sell_threshold:
            signal = "SELL"
            reasons.insert(0, f"ML({ml_score:.2f})")
        else:
            signal = "HOLD"
            reasons = [f"ML({ml_score:.2f})", "Neutral"]

        return signal, int(confidence), " | ".join(reasons)

    def _calibrate_confidence(self, raw_confidence: float, signal: str, ml_score: float) -> float:
        """
        ENHANCEMENT: Calibrate confidence based on historical performance

        Logic:
        - If historical accuracy at this confidence level is lower than stated,
          adjust down
        - If historical accuracy is higher, adjust up (but conservatively)

        Args:
            raw_confidence: Raw confidence from model (0-100)
            signal: Signal type (BUY/SELL/HOLD)
            ml_score: ML model score (0-1)

        Returns:
            Calibrated confidence (0-100)
        """
        if len(self._confidence_history) < 20:
            # Not enough history - return raw confidence with conservative adjustment
            return raw_confidence * 0.95  # Slightly conservative

        # Calculate historical accuracy at similar confidence levels
        similar_predictions = [
            (conf, result)
            for conf, result in self._confidence_history
            if abs(conf - raw_confidence) < 15  # Within 15% range
        ]

        if len(similar_predictions) < 5:
            # Not enough similar predictions
            return raw_confidence * 0.95

        # Calculate actual accuracy
        correct_predictions = sum(1 for _, result in similar_predictions if result)
        historical_accuracy = correct_predictions / len(similar_predictions)

        # Expected accuracy based on confidence (e.g., 70% confidence should be 70% accurate)
        expected_accuracy = raw_confidence / 100.0

        # Calibration factor
        if historical_accuracy < expected_accuracy:
            # Model is overconfident - reduce confidence
            calibration_factor = historical_accuracy / expected_accuracy
            calibrated = raw_confidence * calibration_factor

            print(
                f"📉 Confidence calibrated DOWN: {raw_confidence:.0f}% → {calibrated:.0f}% "
                f"(historical accuracy: {historical_accuracy:.1%} vs expected: {expected_accuracy:.1%})"
            )
        elif historical_accuracy > expected_accuracy * 1.1:
            # Model is underconfident - increase slightly (conservatively)
            calibration_factor = min(1.1, historical_accuracy / expected_accuracy)
            calibrated = raw_confidence * calibration_factor

            print(
                f"📈 Confidence calibrated UP: {raw_confidence:.0f}% → {calibrated:.0f}% "
                f"(historical accuracy: {historical_accuracy:.1%} vs expected: {expected_accuracy:.1%})"
            )
        else:
            # Good calibration
            calibrated = raw_confidence

        # Clamp to valid range
        return max(0, min(100, calibrated))

    def record_prediction_outcome(self, confidence: float, was_correct: bool):
        """
        ENHANCEMENT: Record prediction outcome for calibration

        Args:
            confidence: The confidence that was predicted
            was_correct: Whether the prediction was correct
        """
        self._confidence_history.append((confidence, was_correct))

        # Limit history size
        if len(self._confidence_history) > self._max_history:
            self._confidence_history = self._confidence_history[-self._max_history :]

        # Log statistics periodically
        if len(self._confidence_history) % 20 == 0:
            overall_accuracy = sum(1 for _, result in self._confidence_history if result) / len(
                self._confidence_history
            )
            print(
                f"📊 ML Signal accuracy (last {len(self._confidence_history)} predictions): "
                f"{overall_accuracy:.1%}"
            )

    def train_models(self, df):
        """Train models với historical data, sử dụng các feature mới và class_weight."""
        print("🎓 Bắt đầu training models với pipeline nâng cao...")

        try:
            # 1. Load VN-Index data
            print("   - Đang tải dữ liệu VNINDEX cho việc tính toán features...")
            index_df = load_data("VNINDEX", lookback="5y", use_cache=True, is_index=True)
            if index_df.empty:
                print("❌ Không thể tải dữ liệu VNINDEX. Dừng training.")
                return

            # 2. Add features
            print("   - Thêm các features mới (RS, Lag Features)...")
            df = add_ml_features(df, index_df=index_df)

            # 3. Prepare data
            print("   - Chuẩn bị dữ liệu cho training...")
            feature_cols = get_feature_columns()

            # Loại bỏ các hàng không có đủ dữ liệu (pandas 3.0 compatible)
            df = df.dropna(subset=feature_cols + ["target"])

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
            print("   - Huấn luyện RandomForest với class_weight='balanced' và params mới...")
            self.predictor.train_random_forest(X_train, y_train)

            # (Optional) Train LSTM - hiện tại đang tắt để tập trung vào RandomForest
            # print("   - Huấn luyện LSTM...")
            # self.predictor.train_lstm(X_train, y_train)

            # 7. Evaluate model
            print("   - Đánh giá mô hình trên tập test...")
            self.predictor.evaluate(X_test, y_test)

            print("✅ Training và đánh giá hoàn tất!")
            self.model_loaded = True  # Đảm bảo model được đánh dấu là đã load

        except Exception:
            print("❌ Lỗi training models")
            import traceback

            traceback.print_exc()
