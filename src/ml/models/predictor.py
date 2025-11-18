import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MLPredictor:
    def __init__(self):
        self.rf_model = None
        self.scaler = StandardScaler()
        self.models_dir = "models"
        self.ml_enabled = True  # NEW: Flag to track if ML is usable
        self.using_dummy_models = False  # NEW: Flag to track dummy models
        self.ensure_models_dir()
        # Đồng bộ số features mong đợi với features.get_feature_columns()
        try:
            from src.ml.features.technical import (
                get_feature_columns,
            )  # tránh import vòng bằng cách import khi cần

            self.expected_features = len(get_feature_columns())
        except Exception:
            # Fallback an toàn nếu không import được
            self.expected_features = 18

    def ensure_models_dir(self):
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            logger.info(f"✅ Models directory: {os.path.abspath(self.models_dir)}")
        except Exception:
            logger.error("⚠️ Không thể tạo thư mục models")

    def _get_feature_count(self, obj):
        """Helper: lấy số lượng features của model hoặc scaler."""
        if obj is None:
            return None
        if hasattr(obj, "n_features_in_"):
            return getattr(obj, "n_features_in_")
        if hasattr(obj, "mean_"):
            mean = getattr(obj, "mean_", None)
            return len(mean) if mean is not None else None
        return None

    def _validate_feature_count(self, rf_model, scaler, model_label):
        """Đảm bảo models tải lên khớp với số lượng features hiện tại."""
        expected = self.expected_features
        rf_features = self._get_feature_count(rf_model)
        scaler_features = self._get_feature_count(scaler)

        mismatches = []
        if rf_features and rf_features != expected:
            mismatches.append(f"RF={rf_features}")
        if scaler_features and scaler_features != expected:
            mismatches.append(f"Scaler={scaler_features}")

        if mismatches:
            logger.warning(
                "⚠️ %s models feature mismatch (%s). Pipeline expects %s features. "
                "Bỏ qua %s models. Hãy train lại models (python -m ml_pipeline.train_pipeline) "
                "hoặc xóa thư mục models/ nếu cần.",
                model_label.capitalize(),
                ", ".join(mismatches),
                expected,
                model_label,
            )
            return False

        return True

    def create_dummy_models(self):
        """Tạo models mẫu phù hợp với số lượng features hiện tại."""
        logger.info(
            "🔄 Creating dummy models with %s features...",
            self.expected_features,
        )

        # Tạo scaler mẫu với 18 features
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array([0] * self.expected_features)
        self.scaler.scale_ = np.array([1] * self.expected_features)

        # Tạo RF model mẫu với 18 features
        self.rf_model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Train với data giả 18 features
        X_dummy = np.random.randn(100, self.expected_features)
        y_dummy = np.random.randint(0, 2, 100)
        self.rf_model.fit(X_dummy, y_dummy)

        # Đánh dấu đang sử dụng dummy models
        self.using_dummy_models = True
        self.ml_enabled = True

        # Lưu models
        self.save_models()
        logger.info("✅ Dummy models created and saved")


    def save_models(self):
        """Lưu models"""
        self.ensure_models_dir()
        try:
            if self.rf_model:
                joblib.dump(self.rf_model, os.path.join(self.models_dir, "random_forest.pkl"))
            joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.pkl"))

            # Lưu model metadata với feature list
            metadata = {
                "expected_features": self.expected_features,
                "saved_at": pd.Timestamp.now().isoformat(),
            }

            # Save feature names if available
            try:
                from src.ml.features.technical import get_feature_columns

                metadata["feature_names"] = get_feature_columns()
            except Exception:
                pass

            with open(os.path.join(self.models_dir, "model_info.json"), "w") as f:
                import json

                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Models saved successfully with {self.expected_features} features")
        except Exception:
            logger.error("❌ Lỗi khi lưu models")

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest với class_weight và params tối ưu."""
        logger.info("🌲 Training Random Forest with optimized parameters...")

        if X_train.shape[1] != self.expected_features:
            from src.config.exceptions import ModelPredictionError

            raise ModelPredictionError(
                "Feature count mismatch during training",
                context={
                    "got": X_train.shape[1],
                    "expected": self.expected_features,
                    "message": "Ensure all features are generated before training.",
                },
            )

        self.rf_model = RandomForestClassifier(
            n_estimators=200,  # Tăng số lượng cây
            max_depth=15,  # Tăng độ sâu
            min_samples_split=10,  # Yêu cầu ít nhất 10 mẫu để split
            min_samples_leaf=5,  # Yêu cầu ít nhất 5 mẫu ở mỗi leaf
            class_weight="balanced",  # QUAN TRỌNG: Xử lý mất cân bằng dữ liệu
            random_state=42,
            n_jobs=-1,
        )

        self.rf_model.fit(X_train, y_train)
        self.save_models()
        logger.info("✅ Random Forest trained & saved!")

    def evaluate(self, X_test, y_test):
        """Đánh giá model trên test set."""
        if self.rf_model is None:
            logger.warning("Model not trained yet. Cannot evaluate.")
            return

        logger.info("📊 Evaluating model performance...")
        try:
            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                f1_score,
                precision_score,
                recall_score,
            )

            y_pred = self.rf_model.predict(X_test)

            _accuracy = accuracy_score(y_test, y_pred)  # noqa: F841
            _precision = precision_score(y_test, y_pred, average="weighted")  # noqa: F841
            _recall = recall_score(y_test, y_pred, average="weighted")  # noqa: F841
            _f1 = f1_score(y_test, y_pred, average="weighted")  # noqa: F841

            logger.info(f"   - Accuracy:  {_accuracy:.4f}")
            logger.info(f"   - Precision: {_precision:.4f}")
            logger.info(f"   - Recall:    {_recall:.4f}")
            logger.info(f"   - F1-Score:  {_f1:.4f}")

            logger.info("   - Classification Report:")
            # Dùng print để format đẹp hơn trong log
            print(classification_report(y_test, y_pred, target_names=["Down/Hold", "Up"]))

        except Exception:
            logger.error("❌ Error during model evaluation")

    def predict(self, X):
        """
        Prediction với feature validation

        NEW: Checks if ML is enabled before predicting
        """
        # NEW: Check if ML is enabled
        if not self.ml_enabled:
            raise ValueError(
                "ML predictions disabled: Models not loaded. "
                "Train models first: python scripts/train_models.py"
            )

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        n = len(X_arr)
        if n == 0:
            return np.array([])

        # Kiểm tra số features
        if X_arr.shape[1] != self.expected_features:
            logger.warning(
                f"⚠️ Feature mismatch: got {X_arr.shape[1]}, expected {self.expected_features}. "
                f"Models need retraining. Falling back to technical analysis."
            )
            # Ném lỗi để tầng trên (ml_signals.analyze) fallback sang Technical Analysis
            raise ValueError(
                f"Feature mismatch: got {X_arr.shape[1]}, expected {self.expected_features}"
            )

        # Scale features
        try:
            if hasattr(self.scaler, "mean_"):
                # Check for NaN or infinite values
                if np.any(np.isnan(X_arr)) or np.any(np.isinf(X_arr)):
                    logger.warning(f"⚠️ Found NaN/inf values in features. NaN count: {np.sum(np.isnan(X_arr))}, Inf count: {np.sum(np.isinf(X_arr))}")
                    # Replace NaN/inf with 0
                    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
                
                X_scaled = self.scaler.transform(X_arr)
            else:
                X_scaled = X_arr
        except Exception as e:
            logger.error(f"⚠️ Lỗi scaling: {str(e)}")
            logger.error(f"⚠️ X_arr shape: {X_arr.shape}, dtype: {X_arr.dtype}")
            logger.error(f"⚠️ X_arr sample: {X_arr[0] if len(X_arr) > 0 else 'empty'}")
            X_scaled = X_arr

        # RF prediction
        if self.rf_model is not None:
            try:
                rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
                return rf_pred
            except Exception as e:
                logger.error(f"⚠️ RF predict error: {str(e)}")
                logger.error(f"⚠️ X_scaled shape: {X_scaled.shape}, dtype: {X_scaled.dtype}")
                raise ValueError(f"Model prediction failed: {str(e)}")
        else:
            raise ValueError("RF model not initialized")

    def load_models(self):
        """Load pre-trained models và scaler
        
        Ưu tiên tìm models theo thứ tự:
        1. Ensemble models (ensemble_rf.pkl, ensemble_scaler.pkl) - nếu có
        2. Standard models (random_forest.pkl, scaler.pkl) - fallback
        """
        self.ensure_models_dir()

        models_loaded = False

        try:
            # Ưu tiên 1: Tìm ensemble models (từ ml_pipeline)
            ensemble_rf_path = os.path.join(self.models_dir, "ensemble_rf.pkl")
            ensemble_scaler_path = os.path.join(self.models_dir, "ensemble_scaler.pkl")
            
            # Ưu tiên 2: Tìm standard models
            rf_path = os.path.join(self.models_dir, "random_forest.pkl")
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            
            info_path = os.path.join(self.models_dir, "model_info.json")

            # Thử load ensemble models trước
            if os.path.exists(ensemble_rf_path) and os.path.exists(ensemble_scaler_path):
                logger.info("📦 Tìm thấy ensemble models, đang load...")
                try:
                    candidate_rf = joblib.load(ensemble_rf_path)
                    candidate_scaler = joblib.load(ensemble_scaler_path)

                    if self._validate_feature_count(candidate_rf, candidate_scaler, "ensemble"):
                        self.rf_model = candidate_rf
                        self.scaler = candidate_scaler
                        logger.info(
                            "✅ Loaded ensemble models (expecting %s features)",
                            self.expected_features,
                        )
                        models_loaded = True
                        self.ml_enabled = True
                        self.using_dummy_models = False
                    else:
                        logger.info("🔄 Bỏ qua ensemble models, thử standard models...")
                except Exception as e:
                    logger.warning(f"⚠️ Lỗi khi load ensemble models: {e}")
                    logger.info("🔄 Thử load standard models...")

            # Nếu chưa load được, thử standard models
            if not models_loaded and os.path.exists(rf_path) and os.path.exists(scaler_path):
                # Load metadata trước để xác thực
                if os.path.exists(info_path):
                    with open(info_path, "r") as f:
                        import json

                        info = json.load(f)
                        saved_features = info.get("expected_features", 18)

                        # Check if saved model matches current feature definition
                        try:
                            from src.ml.features.technical import get_feature_columns

                            current_features = len(get_feature_columns())

                            if saved_features != current_features:
                                logger.warning(
                                    f"⚠️ Model feature mismatch: saved={saved_features}, current={current_features}"
                                )
                                logger.warning("🔄 Recreating models with current feature set...")
                                self.create_dummy_models()
                                self.ml_enabled = True
                                return True
                        except Exception:
                            logger.warning("Could not verify features")

                candidate_rf = joblib.load(rf_path)
                candidate_scaler = joblib.load(scaler_path)

                if not self._validate_feature_count(candidate_rf, candidate_scaler, "standard"):
                    logger.info("🔄 Recreating dummy models để khớp số features hiện tại...")
                    self.create_dummy_models()
                    self.ml_enabled = True
                    return True

                self.rf_model = candidate_rf
                self.scaler = candidate_scaler

                logger.info(
                    f"✅ Loaded trained models (expecting {self.expected_features} features)"
                )
                models_loaded = True
                self.ml_enabled = True
                self.using_dummy_models = False

            if not models_loaded:
                # Kiểm tra xem có bất kỳ model nào khác không
                has_any_model = (
                    os.path.exists(ensemble_rf_path)
                    or os.path.exists(rf_path)
                    or os.path.exists(os.path.join(self.models_dir, "ensemble_gb.pkl"))
                    or os.path.exists(os.path.join(self.models_dir, "ensemble_xgb.pkl"))
                )

                if has_any_model:
                    # Có models nhưng không đầy đủ - cảnh báo nhẹ
                    logger.warning(
                        f"⚠️ Không tìm thấy đầy đủ model files. "
                        f"Đã tìm thấy một số models nhưng thiếu RF model hoặc scaler.\n"
                        f"Thư mục: {os.path.abspath(self.models_dir)}\n"
                        f"Bot sẽ tiếp tục nhưng ML có thể không hoạt động đầy đủ."
                    )
                else:
                    # CRITICAL: No models found at all
                    logger.critical(
                        "\n"
                        + "=" * 70
                        + "\n"
                        "⚠️⚠️⚠️ CẢNH BÁO NGHIÊM TRỌNG: ML MODELS KHÔNG TỒN TẠI ⚠️⚠️⚠️\n"
                        + "=" * 70
                        + "\n"
                        f"Model files không tìm thấy tại: {os.path.abspath(self.models_dir)}\n"
                        "\n"
                        "❌ BOT SẼ KHÔNG SỬ DỤNG ML PREDICTIONS!\n"
                        "\n"
                        "🔧 ĐỂ SỬA LỖI NÀY:\n"
                        "1. Chạy lệnh: python scripts/init_models.py (tạo dummy models)\n"
                        "2. Hoặc: python -m ml_pipeline.train_pipeline (train models thật)\n"
                        "3. Sau khi train xong, khởi động lại bot\n"
                        "\n"
                        "⚠️  Trading sẽ tiếp tục KHÔNG CÓ ML SIGNALS\n"
                        + "=" * 70
                    )

                # DISABLE ML instead of creating dummy models
                self.ml_enabled = False
                self.using_dummy_models = False
                models_loaded = False

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.critical(
                "\n" + "=" * 70 + "\n"
                "⚠️⚠️⚠️ LỖI KHI LOAD ML MODELS ⚠️⚠️⚠️\n" + "=" * 70 + "\n"
                f"Lỗi: {type(e).__name__}: {str(e)}\n"
                f"\nChi tiết:\n{error_details}\n"
                "❌ BOT SẼ KHÔNG SỬ DỤNG ML PREDICTIONS!\n"
                "\n"
                "🔧 ĐỂ SỬA LỖI NÀY:\n"
                "1. Kiểm tra log chi tiết ở trên\n"
                "2. Xóa models cũ (nếu bị corrupt): rm -rf models/\n"
                "3. Chạy: python scripts/init_models.py (tạo dummy models)\n"
                "4. Hoặc train lại: python -m ml_pipeline.train_pipeline\n"
                "5. Khởi động lại bot\n"
                "\n"
                "⚠️  Trading sẽ tiếp tục KHÔNG CÓ ML SIGNALS\n" + "=" * 70
            )

            # DISABLE ML instead of creating dummy models
            self.ml_enabled = False
            self.using_dummy_models = False
            models_loaded = False

        return models_loaded


# [file content end]
