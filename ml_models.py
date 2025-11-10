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
        
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest (Vũ Khúc - ổn định)"""
        print("🌲 Training Random Forest...")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Lưu model
        joblib.dump(self.rf_model, 'models/random_forest.pkl')
        print("✅ Random Forest trained!")
        
    def train_lstm(self, X_train, y_train):
        """Train LSTM (Phá Quân - biến động)"""
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
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Save model
        self.lstm_model.save('models/lstm_model.h5')
        print("✅ LSTM trained!")
        
    def _prepare_lstm_data(self, X):
        """Chuẩn bị data cho LSTM với sliding window"""
        X_lstm = []
        for i in range(self.lookback, len(X)):
            X_lstm.append(X[i-self.lookback:i])
        return np.array(X_lstm)
    
    def predict(self, X):
        """Ensemble prediction: RF + LSTM"""
        
        # Random Forest prediction
        rf_pred = self.rf_model.predict_proba(X)[:, 1] if self.rf_model else np.zeros(len(X))
        
        # LSTM prediction
        if self.lstm_model and len(X) >= self.lookback:
            X_lstm = self._prepare_lstm_data(X)
            lstm_pred = self.lstm_model.predict(X_lstm, verbose=0).flatten()
            
            # Pad beginning (LSTM cần lookback)
            lstm_pred = np.concatenate([np.zeros(self.lookback), lstm_pred])
        else:
            lstm_pred = np.zeros(len(X))
        
        # Ensemble: Trung bình có trọng số
        # RF (60%) + LSTM (40%) - RF ổn định hơn
        ensemble_pred = 0.6 * rf_pred + 0.4 * lstm_pred
        
        return ensemble_pred
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            if os.path.exists('models/random_forest.pkl'):
                self.rf_model = joblib.load('models/random_forest.pkl')
                print("✅ Loaded Random Forest")
            
            if os.path.exists('models/lstm_model.h5'):
                self.lstm_model = keras.models.load_model('models/lstm_model.h5')
                print("✅ Loaded LSTM")
                
            if os.path.exists('models/scaler.pkl'):
                self.scaler = joblib.load('models/scaler.pkl')
                print("✅ Loaded Scaler")
        except Exception as e:
            print(f"⚠️ Không load được models: {e}")
    
    def save_scaler(self):
        """Lưu scaler"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')