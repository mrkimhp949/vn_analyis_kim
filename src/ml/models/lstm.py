# -*- coding: utf-8 -*-
"""
LSTM Model for Time-Series Prediction
Advanced ML for better predictions
"""
import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import TensorFlow
try:
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adam

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not installed. Install: pip install tensorflow")


class LSTMPredictor:
    """
    LSTM-based predictor for stock price movements

    Features:
    - Sequence-based learning
    - Captures temporal dependencies
    - Better for time-series than traditional ML
    """

    def __init__(
        self,
        sequence_length: int = 20,
        features: int = 28,
        model_path: str = "models/lstm_model.h5",
    ):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM model")

        self.sequence_length = sequence_length
        self.features = features
        self.model_path = model_path
        self.model = None
        self.scaler = None

    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture

        Returns:
            Compiled Keras model
        """
        model = Sequential(
            [
                # First LSTM layer
                LSTM(
                    units=50,
                    return_sequences=True,
                    input_shape=(self.sequence_length, self.features),
                ),
                Dropout(0.2),
                BatchNormalization(),
                # Second LSTM layer
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                BatchNormalization(),
                # Third LSTM layer
                LSTM(units=50),
                Dropout(0.2),
                BatchNormalization(),
                # Dense layers
                Dense(units=25, activation="relu"),
                Dropout(0.2),
                # Output layer
                Dense(units=1, activation="sigmoid"),
            ]
        )

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", "AUC"],
        )

        return model

    def prepare_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequences for LSTM

        Args:
            X: Features array (n_samples, n_features)
            y: Labels array (n_samples,)

        Returns:
            X_seq: Sequences (n_sequences, sequence_length, n_features)
            y_seq: Labels for sequences (n_sequences,)
        """
        n_samples = len(X)
        n_sequences = n_samples - self.sequence_length + 1

        if n_sequences <= 0:
            raise ValueError(
                f"Not enough samples for sequence_length={self.sequence_length}"
            )

        # Create sequences
        X_seq = np.array([X[i:i + self.sequence_length] for i in range(n_sequences)])

        if y is not None:
            # Take label from last timestep of each sequence
            y_seq = y[self.sequence_length - 1:]
            return X_seq, y_seq

        return X_seq, None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        """
        Train LSTM model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of epochs
            batch_size: Batch size
        """
        logger.info("🎓 Training LSTM model...")

        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)

        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None

        # Build model
        self.model = self.build_model()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=10,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                self.model_path,
                monitor="val_loss" if validation_data else "loss",
                save_best_only=True,
            ),
        ]

        # Train
        history = self.model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("✅ LSTM training complete")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using LSTM model

        Args:
            X: Features array

        Returns:
            Predictions (probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first")

        # Prepare sequences
        X_seq, _ = self.prepare_sequences(X)

        # Predict
        predictions = self.model.predict(X_seq, verbose=0)

        # Pad predictions to match original length
        # (first sequence_length-1 samples don't have predictions)
        padding = np.full(self.sequence_length - 1, 0.5)
        predictions_padded = np.concatenate([padding, predictions.flatten()])

        return predictions_padded

    def save_model(self):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        logger.info(f"✅ Model saved to {self.model_path}")

    def load_model(self) -> bool:
        """
        Load model from disk

        Returns:
            True if successful
        """
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found: {self.model_path}")
            return False

        try:
            self.model = load_model(self.model_path)
            logger.info(f"✅ Model loaded from {self.model_path}")
            return True
        except Exception:
            logger.error("Error loading model")
            return False

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dict with metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Prepare sequences
        X_test_seq, y_test_seq = self.prepare_sequences(X_test, y_test)

        # Evaluate
        results = self.model.evaluate(X_test_seq, y_test_seq, verbose=0)

        metrics = {
            "loss": results[0],
            "accuracy": results[1],
            "auc": results[2] if len(results) > 2 else None,
        }

        logger.info(f"📊 LSTM Evaluation: {metrics}")

        return metrics


# ============================================================================
# ENSEMBLE WITH LSTM
# ============================================================================


class EnsembleWithLSTM:
    """
    Ensemble traditional ML models with LSTM

    Combines:
    - Random Forest
    - XGBoost
    - LightGBM
    - LSTM
    """

    def __init__(self):
        self.lstm = None
        self.traditional_models = None

        # Try to load LSTM
        if TENSORFLOW_AVAILABLE:
            try:
                self.lstm = LSTMPredictor()
                self.lstm.load_model()
            except Exception:
                logger.warning("Could not load LSTM")

        # Load traditional models
        try:
            from src.ml.models.ensemble import EnhancedMLPredictor

            self.traditional_models = EnhancedMLPredictor()
            self.traditional_models.load_models()
        except Exception:
            logger.warning("Could not load traditional models")

    def predict(self, X: np.ndarray, lstm_weight: float = 0.3) -> np.ndarray:
        """
        Ensemble prediction

        Args:
            X: Features
            lstm_weight: Weight for LSTM (0-1)

        Returns:
            Ensemble predictions
        """
        predictions = []
        weights = []

        # LSTM prediction
        if self.lstm and self.lstm.model:
            try:
                lstm_pred = self.lstm.predict(X)
                predictions.append(lstm_pred)
                weights.append(lstm_weight)
            except Exception:
                logger.warning("LSTM prediction failed")

        # Traditional models prediction
        if self.traditional_models:
            try:
                trad_pred = self.traditional_models.predict(X, use_ensemble=True)
                predictions.append(trad_pred)
                weights.append(1 - lstm_weight)
            except Exception:
                logger.warning("Traditional prediction failed")

        if not predictions:
            # Fallback to random
            return np.random.uniform(0.4, 0.6, len(X))

        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_pred = sum(p * w for p, w in zip(predictions, weights))

        return ensemble_pred


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available")
        exit(1)

    print("\n" + "=" * 70)
    print("🧪 TESTING LSTM PREDICTOR")
    print("=" * 70 + "\n")

    # Create dummy data
    n_samples = 1000
    n_features = 28
    sequence_length = 20

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    # Split
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Initialize
    lstm = LSTMPredictor(sequence_length=sequence_length, features=n_features)

    # Train
    print("Training LSTM...")
    history = lstm.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)

    # Evaluate
    print("\nEvaluating...")
    metrics = lstm.evaluate(X_test, y_test)
    print(f"Metrics: {metrics}")

    # Predict
    print("\nPredicting...")
    predictions = lstm.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")

    # Save
    lstm.save_model()

    print("\n✅ Testing complete!")
