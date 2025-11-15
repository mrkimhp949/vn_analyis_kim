import logging
import os
from typing import Optional

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
            from features import \
                get_feature_columns  # tránh import vòng bằng cách import khi cần

            self.expected_features = len(get_feature_columns())
        except Exception:
            # Fallback an toàn nếu không import được
            self.expected_features = 18

    def ensure_models_dir(self):
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            logger.info(f"✅ Models directory: {os.path.abspath(self.models_dir)}")
        except Exception as e:
            logger.error(f"⚠️ Không thể tạo thư mục models: {e}")

    def create_dummy_models(self):
        """Tạo models mẫu với ĐÚNG 18 features"""
        logger.info("🔄 Creating dummy models with 18 features...")

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

        # Lưu models
        self.save_models()
        logger.info("✅ Dummy models created and saved")

    def save_models(self):
        """Lưu models"""
        self.ensure_models_dir()
        try:
            if self.rf_model:
                joblib.dump(
                    self.rf_model, os.path.join(self.models_dir, "random_forest.pkl")
                )
            joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.pkl"))

            # Lưu model metadata với feature list
            metadata = {
                "expected_features": self.expected_features,
                "saved_at": pd.Timestamp.now().isoformat(),
            }

            # Save feature names if available
            try:
                from features import get_feature_columns

                metadata["feature_names"] = get_feature_columns()
            except Exception:
                pass

            with open(os.path.join(self.models_dir, "model_info.json"), "w") as f:
                import json

                json.dump(metadata, f, indent=2)

            logger.info(
                f"✅ Models saved successfully with {self.expected_features} features"
            )
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu models: {e}")

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest với class_weight và params tối ưu."""
        logger.info("🌲 Training Random Forest with optimized parameters...")

        if X_train.shape[1] != self.expected_features:
            from exceptions import ModelPredictionError

            raise ModelPredictionError(
                f"Feature count mismatch during training",
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
            from sklearn.metrics import (accuracy_score, classification_report,
                                         f1_score, precision_score,
                                         recall_score)

            y_pred = self.rf_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            logger.info(f"   - Accuracy:  {accuracy:.4f}")
            logger.info(f"   - Precision: {precision:.4f}")
            logger.info(f"   - Recall:    {recall:.4f}")
            logger.info(f"   - F1-Score:  {f1:.4f}")

            logger.info("   - Classification Report:")
            # Dùng print để format đẹp hơn trong log
            print(
                classification_report(y_test, y_pred, target_names=["Down/Hold", "Up"])
            )

        except Exception as e:
            logger.error(f"❌ Error during model evaluation: {e}")

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
            # Ném lỗi để tầng trên (ml_signals.analyze) fallback sang Technical Analysis
            raise ValueError(
                f"Feature mismatch: got {X_arr.shape[1]}, expected {self.expected_features}"
            )

        # Scale features
        try:
            if hasattr(self.scaler, "mean_"):
                X_scaled = self.scaler.transform(X_arr)
            else:
                X_scaled = X_arr
        except Exception as e:
            logger.error(f"⚠️ Lỗi scaling: {e}")
            X_scaled = X_arr

        # RF prediction
        if self.rf_model is not None:
            try:
                rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
                return rf_pred
            except Exception as e:
                logger.error(f"⚠️ RF predict error: {e}")
                raise ValueError(f"Model prediction failed: {e}")
        else:
            raise ValueError("RF model not initialized")

    def load_models(self):
        """Load pre-trained models và scaler"""
        self.ensure_models_dir()

        models_loaded = False

        try:
            rf_path = os.path.join(self.models_dir, "random_forest.pkl")
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            info_path = os.path.join(self.models_dir, "model_info.json")

            if os.path.exists(rf_path) and os.path.exists(scaler_path):
                # Load metadata first to validate
                if os.path.exists(info_path):
                    with open(info_path, "r") as f:
                        import json

                        info = json.load(f)
                        saved_features = info.get("expected_features", 18)

                        # Check if saved model matches current feature definition
                        try:
                            from features import get_feature_columns

                            current_features = len(get_feature_columns())

                            if saved_features != current_features:
                                logger.warning(
                                    f"⚠️ Model feature mismatch: saved={saved_features}, current={current_features}"
                                )
                                logger.warning(
                                    "🔄 Recreating models with current feature set..."
                                )
                                self.create_dummy_models()
                                return True
                        except Exception as e:
                            logger.warning(f"Could not verify features: {e}")

                # Load models
                self.rf_model = joblib.load(rf_path)
                self.scaler = joblib.load(scaler_path)

                logger.info(
                    f"✅ Loaded trained models (expecting {self.expected_features} features)"
                )
                models_loaded = True
                self.ml_enabled = True
                self.using_dummy_models = False
            else:
                # CRITICAL: No models found
                logger.critical(
                    "\n" + "=" * 70 + "\n"
                    "⚠️⚠️⚠️ CẢNH BÁO NGHIÊM TRỌNG: ML MODELS KHÔNG TỒN TẠI ⚠️⚠️⚠️\n"
                    + "=" * 70
                    + "\n"
                    f"Model files không tìm thấy tại: {os.path.abspath(self.models_dir)}\n"
                    "\n"
                    "❌ BOT SẼ KHÔNG SỬ DỤNG ML PREDICTIONS!\n"
                    "\n"
                    "🔧 ĐỂ SỬA LỖI NÀY:\n"
                    "1. Chạy lệnh: python scripts/train_models.py\n"
                    "2. Hoặc: python -m src.ml.training.pipeline\n"
                    "3. Sau khi train xong, khởi động lại bot\n"
                    "\n"
                    "⚠️  Trading sẽ tiếp tục KHÔNG CÓ ML SIGNALS\n" + "=" * 70
                )

                # DISABLE ML instead of creating dummy models
                self.ml_enabled = False
                self.using_dummy_models = False
                models_loaded = False

        except Exception as e:
            logger.critical(
                "\n" + "=" * 70 + "\n"
                "⚠️⚠️⚠️ LỖI KHI LOAD ML MODELS ⚠️⚠️⚠️\n" + "=" * 70 + "\n"
                f"Lỗi: {e}\n"
                "\n"
                "❌ BOT SẼ KHÔNG SỬ DỤNG ML PREDICTIONS!\n"
                "\n"
                "🔧 ĐỂ SỬA LỖI NÀY:\n"
                "1. Kiểm tra log chi tiết ở trên\n"
                "2. Xóa models cũ (nếu bị corrupt): rm -rf models/\n"
                "3. Train lại: python scripts/train_models.py\n"
                "4. Khởi động lại bot\n"
                "\n"
                "⚠️  Trading sẽ tiếp tục KHÔNG CÓ ML SIGNALS\n" + "=" * 70
            )

            # DISABLE ML instead of creating dummy models
            self.ml_enabled = False
            self.using_dummy_models = False
            models_loaded = False

        return models_loaded


# [file content end]
