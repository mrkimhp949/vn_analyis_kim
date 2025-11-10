import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        self.rf_model = None
        self.scaler = StandardScaler()
        self.models_dir = 'models'
        self.ensure_models_dir()

    def ensure_models_dir(self):
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            logger.info(f"✅ Models directory: {os.path.abspath(self.models_dir)}")
        except Exception as e:
            logger.error(f"⚠️ Không thể tạo thư mục models: {e}")

    def create_dummy_models(self):
        """Tạo models mẫu nếu không có models thật"""
        logger.info("🔄 Creating dummy models for testing...")
        
        # Tạo scaler mẫu
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array([0] * 18)  # 18 features
        self.scaler.scale_ = np.array([1] * 18)
        
        # Tạo RF model mẫu
        self.rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Train với data giả
        X_dummy = np.random.randn(100, 18)
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
            logger.info("✅ Models saved successfully")
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu models: {e}")

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest"""
        logger.info("🌲 Training Random Forest...")

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

    def train_lstm(self, X_train, y_train, epochs=50, batch_size=32):
        """Skip LSTM trên Render (quá nặng)"""
        logger.info("⚠️ LSTM skipped - using Random Forest only")
        return None

    def predict(self, X):
        """Prediction chỉ dùng Random Forest"""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        n = len(X_arr)
        if n == 0:
            return np.array([])

        # Scale features
        try:
            if hasattr(self.scaler, 'mean_'):
                X_scaled = self.scaler.transform(X_arr)
            else:
                X_scaled = X_arr
        except Exception:
            X_scaled = X_arr

        # RF prediction only
        if self.rf_model is not None:
            try:
                rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
                return rf_pred
            except Exception as e:
                logger.error(f"⚠️ RF predict error: {e}")
                return np.zeros(n)
        else:
            return np.zeros(n)

    def load_models(self):
        """Load pre-trained models và scaler nếu có"""
        self.ensure_models_dir()
        
        models_loaded = False
        
        try:
            rf_path = os.path.join(self.models_dir, 'random_forest.pkl')
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            
            if os.path.exists(rf_path) and os.path.exists(scaler_path):
                self.rf_model = joblib.load(rf_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Loaded trained models")
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

    def save_scaler(self):
        """Lưu scaler"""
        self.ensure_models_dir()
        try:
            joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
            logger.info("✅ Scaler saved.")
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu scaler: {e}")