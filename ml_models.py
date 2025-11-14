import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        self.rf_model = None
        self.scaler = StandardScaler()
        self.models_dir = 'models'
        self.ensure_models_dir()
        # Đồng bộ số features mong đợi với features.get_feature_columns()
        try:
            from features import get_feature_columns  # tránh import vòng bằng cách import khi cần
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
                joblib.dump(self.rf_model, os.path.join(self.models_dir, 'random_forest.pkl'))
            joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))

            # Lưu model metadata với feature list
            metadata = {
                'expected_features': self.expected_features,
                'saved_at': pd.Timestamp.now().isoformat()
            }

            # Save feature names if available
            try:
                from features import get_feature_columns
                metadata['feature_names'] = get_feature_columns()
            except Exception:
                pass

            with open(os.path.join(self.models_dir, 'model_info.json'), 'w') as f:
                import json
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Models saved successfully with {self.expected_features} features")
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu models: {e}")

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest với feature validation"""
        logger.info("🌲 Training Random Forest...")

        # Validate số features - STRICT CHECK
        if X_train.shape[1] != self.expected_features:
            from exceptions import ModelPredictionError
            raise ModelPredictionError(
                f"Feature count mismatch during training",
                context={
                    'got': X_train.shape[1],
                    'expected': self.expected_features,
                    'message': 'Please check features.get_feature_columns() and ensure all features are generated'
                }
            )

        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )

        self.rf_model.fit(X_train, y_train)
        self.save_models()
        logger.info("✅ Random Forest trained & saved!")

    def predict(self, X):
        """Prediction với feature validation"""
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
            raise ValueError(f"Feature mismatch: got {X_arr.shape[1]}, expected {self.expected_features}")

        # Scale features
        try:
            if hasattr(self.scaler, 'mean_'):
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
                return np.random.uniform(0.3, 0.7, n)
        else:
            return np.random.uniform(0.3, 0.7, n)

    def load_models(self):
        """Load pre-trained models và scaler"""
        self.ensure_models_dir()

        models_loaded = False

        try:
            rf_path = os.path.join(self.models_dir, 'random_forest.pkl')
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            info_path = os.path.join(self.models_dir, 'model_info.json')

            if os.path.exists(rf_path) and os.path.exists(scaler_path):
                # Load metadata first to validate
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        import json
                        info = json.load(f)
                        saved_features = info.get('expected_features', 18)

                        # Check if saved model matches current feature definition
                        try:
                            from features import get_feature_columns
                            current_features = len(get_feature_columns())

                            if saved_features != current_features:
                                logger.warning(
                                    f"⚠️ Model feature mismatch: saved={saved_features}, current={current_features}"
                                )
                                logger.warning("🔄 Recreating models with current feature set...")
                                self.create_dummy_models()
                                return True
                        except Exception as e:
                            logger.warning(f"Could not verify features: {e}")

                # Load models
                self.rf_model = joblib.load(rf_path)
                self.scaler = joblib.load(scaler_path)

                logger.info(f"✅ Loaded trained models (expecting {self.expected_features} features)")
                models_loaded = True
            else:
                logger.warning("ℹ️ Models not found, creating dummy models...")
                self.create_dummy_models()
                models_loaded = True

        except Exception as e:
            logger.error(f"⚠️ Lỗi khi load models: {e}")
            logger.info("🔄 Creating dummy models as fallback...")
            self.create_dummy_models()
            models_loaded = True

        return models_loaded
# [file content end]