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
        except Exception:
            logger.error("⚠️ Lỗi scaling")
            X_scaled = X_arr

        # RF prediction
        if self.rf_model is not None:
            try:
                rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
                return rf_pred
            except Exception:
                logger.error("⚠️ RF predict error")
                raise ValueError("Model prediction failed")
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
                    self.rf_model = joblib.load(ensemble_rf_path)
                    self.scaler = joblib.load(ensemble_scaler_path)
                    logger.info(f"✅ Loaded ensemble models (expecting {self.expected_features} features)")
                    models_loaded = True
                    self.ml_enabled = True
                    self.using_dummy_models = False
                except Exception as e:
                    logger.warning(f"⚠️ Lỗi khi load ensemble models: {e}")
                    logger.info("🔄 Thử load standard models...")
                    # Fallback to standard models
                    if os.path.exists(rf_path) and os.path.exists(scaler_path):
                        self.rf_model = joblib.load(rf_path)
                        self.scaler = joblib.load(scaler_path)
                        logger.info(f"✅ Loaded standard models (expecting {self.expected_features} features)")
                        models_loaded = True
                        self.ml_enabled = True
                        self.using_dummy_models = False
                    else:
                        raise  # Re-raise if standard models also don't exist
            elif os.path.exists(rf_path) and os.path.exists(scaler_path):
                # Load metadata first to validate
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
                                return True
                        except Exception:
                            logger.warning("Could not verify features")

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
                # Kiểm tra xem có bất kỳ model nào khác không
                has_any_model = (
                    os.path.exists(ensemble_rf_path) or 
                    os.path.exists(rf_path) or
                    os.path.exists(os.path.join(self.models_dir, "ensemble_gb.pkl")) or
                    os.path.exists(os.path.join(self.models_dir, "ensemble_xgb.pkl"))
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
                        "\n" + "=" * 70 + "\n"
                        "⚠️⚠️⚠️ CẢNH BÁO NGHIÊM TRỌNG: ML MODELS KHÔNG TỒN TẠI ⚠️⚠️⚠️\n" + "=" * 70 + "\n"
                        f"Model files không tìm thấy tại: {os.path.abspath(self.models_dir)}\n"
                        "\n"
                        "❌ BOT SẼ KHÔNG SỬ DỤNG ML PREDICTIONS!\n"
                        "\n"
                        "🔧 ĐỂ SỬA LỖI NÀY:\n"
                        "1. Chạy lệnh: python scripts/init_models.py (tạo dummy models)\n"
                        "2. Hoặc: python -m ml_pipeline.train_pipeline (train models thật)\n"
                        "3. Sau khi train xong, khởi động lại bot\n"
                        "\n"
                        "⚠️  Trading sẽ tiếp tục KHÔNG CÓ ML SIGNALS\n" + "=" * 70
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
