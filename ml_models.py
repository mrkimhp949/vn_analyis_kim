import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

class MLPredictor:
    def __init__(self):
        self.rf_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.lookback = 20  # Số ngày lookback cho LSTM
        self.models_dir = 'models'
        self.ensure_models_dir()

    def ensure_models_dir(self):
        """Tạo thư mục models nếu chưa tồn tại"""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Không thể tạo thư mục models: {e}")

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest (ổn định) và lưu model."""
        print("🌲 Training Random Forest...")

        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )

        self.rf_model.fit(X_train, y_train)

        # Lưu model (đảm bảo thư mục tồn tại)
        self.ensure_models_dir()
        try:
            joblib.dump(self.rf_model, os.path.join(self.models_dir, 'random_forest.pkl'))
            print("✅ Random Forest trained & saved!")
        except Exception as e:
            print(f"❌ Lỗi khi lưu Random Forest: {e}")

    def train_lstm(self, X_train, y_train, epochs=50, batch_size=32):
        """Train LSTM và lưu model."""
        print("🧠 Training LSTM...")

        # Reshape data cho LSTM [samples, timesteps, features]
        X_train_lstm = self._prepare_lstm_data(X_train)

        # Build LSTM model
        self.lstm_model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(self.lookback, X_train.shape[1])),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        self.lstm_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train
        self.lstm_model.fit(
            X_train_lstm,
            y_train[self.lookback:],  # Skip first lookback samples
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )

        # Save model (đảm bảo thư mục tồn tại)
        self.ensure_models_dir()
        try:
            self.lstm_model.save(os.path.join(self.models_dir, 'lstm_model.h5'))
            print("✅ LSTM trained & saved!")
        except Exception as e:
            print(f"❌ Lỗi khi lưu LSTM: {e}")

    def _prepare_lstm_data(self, X):
        """Chuẩn bị data cho LSTM với sliding window"""
        X_lstm = []
        for i in range(self.lookback, len(X)):
            X_lstm.append(X[i-self.lookback:i])
        return np.array(X_lstm)

    def predict(self, X):
        """Ensemble prediction: RF + LSTM
        X: numpy array hoặc pandas DataFrame (2D) with shape (n_samples, n_features)
        Trả về: 1D numpy array ensemble scores (0..1)
        """
        # Chuyển DataFrame -> numpy
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        n = len(X_arr)
        if n == 0:
            return np.array([])

        # Nếu scaler đã fit (có attribute mean_), dùng transform
        try:
            if hasattr(self.scaler, 'mean_'):
                X_scaled = self.scaler.transform(X_arr)
            else:
                X_scaled = X_arr
        except Exception:
            X_scaled = X_arr

        # RF prediction
        if self.rf_model is not None:
            try:
                rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
            except Exception as e:
                print(f"⚠️ RF predict error: {e}")
                rf_pred = np.zeros(n)
        else:
            rf_pred = np.zeros(n)

        # LSTM prediction
        if self.lstm_model is not None and n >= self.lookback:
            try:
                X_lstm = self._prepare_lstm_data(X_scaled)
                lstm_pred = self.lstm_model.predict(X_lstm, verbose=0).flatten()
                # Pad zeros at start so lengths match
                lstm_pred = np.concatenate([np.zeros(self.lookback), lstm_pred])
                # If length mismatch due to rounding, adjust
                if len(lstm_pred) < n:
                    lstm_pred = np.pad(lstm_pred, (0, n - len(lstm_pred)), 'constant', constant_values=0)
                elif len(lstm_pred) > n:
                    lstm_pred = lstm_pred[:n]
            except Exception as e:
                print(f"⚠️ LSTM predict error: {e}")
                lstm_pred = np.zeros(n)
        else:
            lstm_pred = np.zeros(n)

        # Ensemble: RF (60%) + LSTM (40%)
        ensemble_pred = 0.6 * rf_pred + 0.4 * lstm_pred
        return ensemble_pred

    def load_models(self):
        """Load pre-trained models và scaler nếu có"""
        self.ensure_models_dir()
        try:
            rf_path = os.path.join(self.models_dir, 'random_forest.pkl')
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                print("✅ Loaded Random Forest")
            else:
                print("ℹ️ Random Forest model not found.")

            lstm_path = os.path.join(self.models_dir, 'lstm_model.h5')
            if os.path.exists(lstm_path):
                try:
                    self.lstm_model = keras.models.load_model(lstm_path)
                    print("✅ Loaded LSTM")
                except Exception as e:
                    print(f"⚠️ Không thể load LSTM: {e}")
            else:
                print("ℹ️ LSTM model not found.")

            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    print("✅ Loaded Scaler")
                except Exception as e:
                    print(f"⚠️ Không thể load scaler: {e}")
            else:
                print("ℹ️ Scaler not found.")
        except Exception as e:
            print(f"⚠️ Lỗi khi load models: {e}")

    def save_scaler(self):
        """Lưu scaler nếu tồn tại"""
        self.ensure_models_dir()
        try:
            joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
            print("✅ Scaler saved.")
        except Exception as e:
            print(f"❌ Lỗi khi lưu scaler: {e}")