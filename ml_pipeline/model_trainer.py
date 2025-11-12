import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Optional imports for LSTM and XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except ImportError:
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        LSTM_AVAILABLE = True
    except ImportError:
        LSTM_AVAILABLE = False
        logger.warning("LSTM not available. Install with: pip install tensorflow")


@dataclass
class TrainingConfig:
    feature_columns: List[str]
    target_column: str = "target"
    save_dir: str = "models"
    n_splits: int = 5


class EnsembleTrainer:
    """
    Huấn luyện ensemble model (RandomForest + GradientBoosting + LSTM + XGBoost) 
    với Cross validation và logging metrics.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        os.makedirs(self.config.save_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.lstm_model = None
        self.xgb_model = None

    def _split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        X = df[self.config.feature_columns].values
        y = df[self.config.target_column].values
        splits = []
        for train_index, test_index in tscv.split(X):
            splits.append((X[train_index], X[test_index], y[train_index], y[test_index]))
        return splits

    def _build_lstm(self, input_shape: Tuple[int, int]) -> Optional:
        """Build LSTM model for time series prediction"""
        if not LSTM_AVAILABLE:
            return None
        
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            logger.warning(f"Failed to build LSTM: {e}")
            return None

    def _prepare_lstm_data(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM (time series sequences)"""
        if len(X) < sequence_length:
            return None, None
        
        X_seq = []
        y_seq = []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])  # Use actual target, not feature values
        
        return np.array(X_seq), np.array(y_seq)

    def train(self, df: pd.DataFrame, market_regime: Optional[str] = None) -> Dict:
        """
        Train ensemble with RF + GB + LSTM + XGBoost
        
        Args:
            df: Training dataframe
            market_regime: 'BULL', 'BEAR', 'SIDEWAYS' for regime-specific analysis
        """
        splits = self._split(df)
        metrics = []
        regime_metrics = [] if market_regime else None

        # Initialize models
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        gb_model = GradientBoostingClassifier(random_state=42)
        
        if XGBOOST_AVAILABLE:
            xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )
        else:
            xgb_model = None

        logger.info(f"Training ensemble with RF + GB + {'XGBoost' if XGBOOST_AVAILABLE else 'No XGBoost'} + {'LSTM' if LSTM_AVAILABLE else 'No LSTM'}")

        for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(splits):
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train RF and GB
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            # Train XGBoost if available
            if xgb_model:
                xgb_model.fit(X_train_scaled, y_train)

            # Predictions
            rf_pred = rf_model.predict_proba(X_val_scaled)[:, 1]
            gb_pred = gb_model.predict_proba(X_val_scaled)[:, 1]
            
            predictions = [rf_pred, gb_pred]
            weights = [0.3, 0.3]  # Base weights for RF and GB
            
            # XGBoost prediction
            if xgb_model:
                xgb_pred = xgb_model.predict_proba(X_val_scaled)[:, 1]
                predictions.append(xgb_pred)
                weights.append(0.2)
            
            # LSTM prediction (if available)
            lstm_pred = None
            if LSTM_AVAILABLE and len(X_train) > 20:
                try:
                    sequence_length = min(10, len(X_train) // 2)
                    X_train_seq, y_train_seq = self._prepare_lstm_data(X_train_scaled, y_train, sequence_length)
                    if X_train_seq is not None and len(X_train_seq) > 0:
                        lstm_model = self._build_lstm((sequence_length, X_train_scaled.shape[1]))
                        if lstm_model:
                            # y_train_seq is already binary (0 or 1) from target column
                            if len(y_train_seq) > 0:
                                lstm_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=0)
                                
                                # Prepare validation sequences
                                X_val_seq, y_val_seq = self._prepare_lstm_data(X_val_scaled, y_val, sequence_length)
                                if X_val_seq is not None and len(X_val_seq) > 0:
                                    lstm_pred = lstm_model.predict(X_val_seq, verbose=0).flatten()
                                    # Pad or trim to match validation size
                                    if len(lstm_pred) < len(y_val):
                                        lstm_pred = np.pad(lstm_pred, (0, len(y_val) - len(lstm_pred)), mode='edge')
                                    elif len(lstm_pred) > len(y_val):
                                        lstm_pred = lstm_pred[:len(y_val)]
                                    predictions.append(lstm_pred)
                                    weights.append(0.2)
                except Exception as e:
                    logger.warning(f"LSTM training failed in fold {fold_idx}: {e}")

            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Ensemble prediction
            ensemble_pred = sum(pred * w for pred, w in zip(predictions, weights))
            ensemble_cls = (ensemble_pred >= 0.5).astype(int)

            fold_metrics = {
                "accuracy": accuracy_score(y_val, ensemble_cls),
                "precision": precision_score(y_val, ensemble_cls, zero_division=0),
                "recall": recall_score(y_val, ensemble_cls, zero_division=0),
                "f1": f1_score(y_val, ensemble_cls, zero_division=0),
            }
            metrics.append(fold_metrics)
            
            # Log metrics
            logger.info(f"Fold {fold_idx + 1}/{len(splits)} - Accuracy: {fold_metrics['accuracy']:.4f}, "
                       f"F1: {fold_metrics['f1']:.4f}, Precision: {fold_metrics['precision']:.4f}, "
                       f"Recall: {fold_metrics['recall']:.4f}")
            
            if market_regime:
                regime_metrics.append(fold_metrics)

        metrics_summary = {
            metric: {
                "mean": float(np.mean([m[metric] for m in metrics])),
                "std": float(np.std([m[metric] for m in metrics])),
            }
            for metric in metrics[0].keys()
        }
        
        if market_regime and regime_metrics:
            metrics_summary[f"{market_regime}_regime"] = {
                metric: {
                    "mean": float(np.mean([m[metric] for m in regime_metrics])),
                    "std": float(np.std([m[metric] for m in regime_metrics])),
                }
                for metric in regime_metrics[0].keys()
            }

        # Refit on full data
        X_full = df[self.config.feature_columns].values
        y_full = df[self.config.target_column].values
        X_full_scaled = self.scaler.fit_transform(X_full)
        
        rf_model.fit(X_full_scaled, y_full)
        gb_model.fit(X_full_scaled, y_full)
        
        if xgb_model:
            xgb_model.fit(X_full_scaled, y_full)
            self.xgb_model = xgb_model

        # Save models
        joblib.dump(rf_model, os.path.join(self.config.save_dir, "ensemble_rf.pkl"))
        joblib.dump(gb_model, os.path.join(self.config.save_dir, "ensemble_gb.pkl"))
        joblib.dump(self.scaler, os.path.join(self.config.save_dir, "ensemble_scaler.pkl"))
        
        if xgb_model:
            joblib.dump(xgb_model, os.path.join(self.config.save_dir, "ensemble_xgb.pkl"))

        # Save LSTM if trained
        if LSTM_AVAILABLE and len(X_full) > 20:
            try:
                sequence_length = min(10, len(X_full) // 2)
                X_full_seq, y_full_seq = self._prepare_lstm_data(X_full_scaled, y_full, sequence_length)
                if X_full_seq is not None and len(X_full_seq) > 0:
                    lstm_model = self._build_lstm((sequence_length, X_full_scaled.shape[1]))
                    if lstm_model:
                        # y_full_seq is already binary (0 or 1) from target column
                        if len(y_full_seq) > 0:
                            lstm_model.fit(X_full_seq, y_full_seq, epochs=20, batch_size=32, verbose=0)
                            lstm_model.save(os.path.join(self.config.save_dir, "ensemble_lstm.h5"))
                            self.lstm_model = lstm_model
            except Exception as e:
                logger.warning(f"Failed to save LSTM: {e}")

        # Save metrics with logging
        metrics_file = os.path.join(self.config.save_dir, "ensemble_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Training complete. Metrics saved to {metrics_file}")
        logger.info(f"Final metrics - Accuracy: {metrics_summary['accuracy']['mean']:.4f} ± {metrics_summary['accuracy']['std']:.4f}, "
                   f"F1: {metrics_summary['f1']['mean']:.4f} ± {metrics_summary['f1']['std']:.4f}")

        return metrics_summary

