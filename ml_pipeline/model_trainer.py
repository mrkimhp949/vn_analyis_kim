import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
    pass

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    LSTM_AVAILABLE = True
except ImportError:
    try:
        from keras.layers import LSTM, Dense, Dropout
        from keras.models import Sequential

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
        self.lstm_sequence_length = None
        self.xgb_model = None

    def _split(
        self, df: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        X = df[self.config.feature_columns].values
        y = df[self.config.target_column].values
        splits = []
        for train_index, test_index in tscv.split(X):
            splits.append(
                (X[train_index], X[test_index], y[train_index], y[test_index])
            )
        return splits

    def _build_lstm(self, input_shape: Tuple[int, int]) -> Optional:
        """Build LSTM model for time series prediction"""
        if not LSTM_AVAILABLE:
            return None

        try:
            model = Sequential(
                [
                    LSTM(50, return_sequences=True, input_shape=input_shape),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25, activation="relu"),
                    Dense(1, activation="sigmoid"),
                ]
            )
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
            return model
        except Exception:
            logger.warning("Failed to build LSTM")
            return None

    def _prepare_lstm_data(
        self, X: np.ndarray, y: np.ndarray, sequence_length: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM (time series sequences)"""
        if len(X) < sequence_length:
            return None, None

        X_seq = []
        y_seq = []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i - sequence_length : i])
            y_seq.append(y[i])  # Use actual target, not feature values

        return np.array(X_seq), np.array(y_seq)

    def train(self, df: pd.DataFrame, market_regime: Optional[str] = None) -> Dict:
        """
        Train ensemble with RF + GB + (optional) XGBoost + LSTM + stacking meta-model.
        """
        if df.empty:
            raise ValueError("Training dataframe is empty")

        missing_features = [
            col for col in self.config.feature_columns if col not in df.columns
        ]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        if self.config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' is missing from dataframe"
            )

        X = df[self.config.feature_columns].values
        y = df[self.config.target_column].values.astype(int)

        len(df)
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)

        base_weights = {
            "r": 0.35,
            "gb": 0.25,
            "xgb": 0.25,
            "lstm": 0.15,
        }

        base_models_order = ["r", "gb"]
        if XGBOOST_AVAILABLE:
            base_models_order.append("xgb")
        if LSTM_AVAILABLE:
            base_models_order.append("lstm")

        logger.info(
            "Training ensemble with base models: %s",
            " + ".join(base_models_order) if base_models_order else "None",
        )

        metrics: List[Dict[str, float]] = []
        regime_metrics: Optional[List[Dict[str, float]]] = [] if market_regime else None
        meta_features_parts: List[np.ndarray] = []
        meta_targets_parts: List[np.ndarray] = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(
                "Fold %d/%d: train=%d samples, val=%d samples",
                fold_idx + 1,
                self.config.n_splits,
                len(train_idx),
                len(val_idx),
            )

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_val_scaled = fold_scaler.transform(X_val)

            # Train base models for this fold
            rf_fold = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            )
            rf_fold.fit(X_train_scaled, y_train)

            gb_fold = GradientBoostingClassifier(random_state=42)
            gb_fold.fit(X_train_scaled, y_train)

            xgb_fold = None
            if "xgb" in base_models_order:
                try:
                    xgb_fold = XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        eval_metric="logloss",
                    )
                    xgb_fold.fit(X_train_scaled, y_train)
                except Exception:
                    logger.warning(f"XGBoost training failed in fold {fold_idx + 1}")
                    xgb_fold = None

            # Prepare meta features for this fold
            fold_meta = np.full(
                (len(val_idx), len(base_models_order)), 0.5, dtype=np.float32
            )

            fold_predictions = {}
            # RF predictions
            rf_pred = rf_fold.predict_proba(X_val_scaled)[:, 1]
            fold_meta[:, base_models_order.index("r")] = rf_pred
            fold_predictions["r"] = rf_pred

            # GB predictions
            gb_pred = gb_fold.predict_proba(X_val_scaled)[:, 1]
            fold_meta[:, base_models_order.index("gb")] = gb_pred
            fold_predictions["gb"] = gb_pred

            # XGB predictions
            if "xgb" in base_models_order and xgb_fold is not None:
                xgb_pred = xgb_fold.predict_proba(X_val_scaled)[:, 1]
                fold_meta[:, base_models_order.index("xgb")] = xgb_pred
                fold_predictions["xgb"] = xgb_pred

            # LSTM predictions
            lstm_pred = None
            if (
                "lstm" in base_models_order
                and len(X_train_scaled) > 20
                and LSTM_AVAILABLE
            ):
                try:
                    sequence_length = min(10, len(X_train_scaled) // 2)
                    X_train_seq, y_train_seq = self._prepare_lstm_data(
                        X_train_scaled, y_train, sequence_length
                    )
                    if X_train_seq is not None and len(X_train_seq) > 0:
                        lstm_model = self._build_lstm(
                            (sequence_length, X_train_scaled.shape[1])
                        )
                        if lstm_model is not None and len(y_train_seq) > 0:
                            lstm_model.fit(
                                X_train_seq,
                                y_train_seq,
                                epochs=10,
                                batch_size=32,
                                verbose=0,
                            )
                            X_val_seq, _ = self._prepare_lstm_data(
                                X_val_scaled, y_val, sequence_length
                            )
                            if X_val_seq is not None and len(X_val_seq) > 0:
                                lstm_pred = lstm_model.predict(
                                    X_val_seq, verbose=0
                                ).flatten()
                                if len(lstm_pred) < len(y_val):
                                    lstm_pred = np.pad(
                                        lstm_pred,
                                        (0, len(y_val) - len(lstm_pred)),
                                        mode="edge",
                                    )
                                elif len(lstm_pred) > len(y_val):
                                    lstm_pred = lstm_pred[: len(y_val)]
                                fold_meta[
                                    :, base_models_order.index("lstm")
                                ] = lstm_pred
                                fold_predictions["lstm"] = lstm_pred
                except Exception:
                    logger.warning(f"LSTM training failed in fold {fold_idx + 1}")

            meta_features_parts.append(fold_meta)
            meta_targets_parts.append(y_val)

            # Ensemble metrics with weighted average (baseline before stacking)
            weights_vector = np.array(
                [base_weights[key] for key in base_models_order], dtype=np.float32
            )
            available_mask = np.array(
                [1.0 if key in fold_predictions else 0.0 for key in base_models_order],
                dtype=np.float32,
            )

            if available_mask.sum() > 0:
                adjusted_weights = weights_vector * available_mask
                adjusted_weights = adjusted_weights / adjusted_weights.sum()
                ensemble_pred = fold_meta @ adjusted_weights
            else:
                ensemble_pred = np.full(len(y_val), 0.5, dtype=np.float32)

            ensemble_cls = (ensemble_pred >= 0.5).astype(int)

            fold_metrics = {
                "accuracy": accuracy_score(y_val, ensemble_cls),
                "precision": precision_score(y_val, ensemble_cls, zero_division=0),
                "recall": recall_score(y_val, ensemble_cls, zero_division=0),
                "f1": f1_score(y_val, ensemble_cls, zero_division=0),
            }
            metrics.append(fold_metrics)

            logger.info(
                "Fold %d/%d - Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4",
                fold_idx + 1,
                self.config.n_splits,
                fold_metrics["accuracy"],
                fold_metrics["f1"],
                fold_metrics["precision"],
                fold_metrics["recall"],
            )

            if regime_metrics is not None:
                regime_metrics.append(fold_metrics)

        meta_features = (
            np.vstack(meta_features_parts)
            if meta_features_parts
            else np.empty((0, len(base_models_order)))
        )
        meta_targets = (
            np.concatenate(meta_targets_parts) if meta_targets_parts else np.array([])
        )

        stacking_metrics = {}
        stacking_model = None
        if meta_features.size > 0:
            from ml_pipeline.stacking_ensemble import StackingMetaModel

            stacking_model = StackingMetaModel(
                model_type="lightgbm" if LIGHTGBM_AVAILABLE else "logistic"
            )
            stacking_model.fit(
                meta_features,
                meta_targets,
                feature_names=[f"{name}_prob" for name in base_models_order],
            )

            stacking_pred = stacking_model.predict(meta_features)
            stacking_metrics = {
                "accuracy": float(accuracy_score(meta_targets, stacking_pred)),
                "precision": float(
                    precision_score(meta_targets, stacking_pred, zero_division=0)
                ),
                "recall": float(
                    recall_score(meta_targets, stacking_pred, zero_division=0)
                ),
                "f1": float(f1_score(meta_targets, stacking_pred, zero_division=0)),
            }

            stacking_proba = stacking_model.predict_proba(meta_features)
            stacking_metrics["auc_like"] = float(
                np.mean(np.abs(stacking_proba - meta_targets))
            )

            logger.info(
                "Stacking meta-model metrics - Acc: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4",
                stacking_metrics["accuracy"],
                stacking_metrics["f1"],
                stacking_metrics["precision"],
                stacking_metrics["recall"],
            )
        else:
            logger.warning("Không đủ dữ liệu để train stacking meta-model.")

        metrics_summary = {}
        if metrics:
            metrics_summary = {
                metric: {
                    "mean": float(np.mean([m[metric] for m in metrics])),
                    "std": float(np.std([m[metric] for m in metrics])),
                }
                for metric in metrics[0].keys()
            }
        if stacking_metrics:
            metrics_summary["stacking_meta"] = stacking_metrics

        if market_regime and regime_metrics:
            metrics_summary[f"{market_regime}_regime"] = {
                metric: {
                    "mean": float(np.mean([m[metric] for m in regime_metrics])),
                    "std": float(np.std([m[metric] for m in regime_metrics])),
                }
                for metric in regime_metrics[0].keys()
            }

        # Train base models on full dataset for persistence
        X_full_scaled = self.scaler.fit_transform(X)

        rf_model_final = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1,
        )
        rf_model_final.fit(X_full_scaled, y)

        gb_model_final = GradientBoostingClassifier(random_state=42)
        gb_model_final.fit(X_full_scaled, y)

        xgb_model_final = None
        if "xgb" in base_models_order:
            try:
                xgb_model_final = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="logloss",
                )
                xgb_model_final.fit(X_full_scaled, y)
                self.xgb_model = xgb_model_final
            except Exception:
                logger.warning("Final XGBoost training failed")
                xgb_model_final = None

        self.lstm_model = None
        self.lstm_sequence_length = None
        if "lstm" in base_models_order and len(X_full_scaled) > 20 and LSTM_AVAILABLE:
            try:
                sequence_length = min(10, len(X_full_scaled) // 2)
                X_full_seq, y_full_seq = self._prepare_lstm_data(
                    X_full_scaled, y, sequence_length
                )
                if (
                    X_full_seq is not None
                    and len(X_full_seq) > 0
                    and len(y_full_seq) > 0
                ):
                    lstm_model_final = self._build_lstm(
                        (sequence_length, X_full_scaled.shape[1])
                    )
                    if lstm_model_final:
                        lstm_model_final.fit(
                            X_full_seq, y_full_seq, epochs=25, batch_size=32, verbose=0
                        )
                        lstm_model_final.save(
                            os.path.join(self.config.save_dir, "ensemble_lstm.h5")
                        )
                        self.lstm_model = lstm_model_final
                        self.lstm_sequence_length = sequence_length
                        logger.info("Đã huấn luyện & lưu LSTM cho ensemble.")
            except Exception:
                logger.warning("Final LSTM training failed")

        # Persist models & auxiliary artefacts
        joblib.dump(
            rf_model_final, os.path.join(self.config.save_dir, "ensemble_rf.pkl")
        )
        joblib.dump(
            gb_model_final, os.path.join(self.config.save_dir, "ensemble_gb.pkl")
        )
        joblib.dump(
            self.scaler, os.path.join(self.config.save_dir, "ensemble_scaler.pkl")
        )

        if xgb_model_final:
            joblib.dump(
                xgb_model_final, os.path.join(self.config.save_dir, "ensemble_xgb.pkl")
            )

        if stacking_model:
            stacking_model.save(self.config.save_dir)

        metrics_file = os.path.join(self.config.save_dir, "ensemble_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

        # Save ensemble configuration (base models order, weights, etc.)
        ensemble_config = {
            "base_models_order": base_models_order,
            "base_weights": {name: base_weights[name] for name in base_models_order},
            "lstm_sequence_length": self.lstm_sequence_length,
            "feature_columns": self.config.feature_columns,
            "target_column": self.config.target_column,
            "n_splits": self.config.n_splits,
        }
        with open(
            os.path.join(self.config.save_dir, "ensemble_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(ensemble_config, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Training complete. Metrics saved to {metrics_file}")
        if "accuracy" in metrics_summary:
            logger.info(
                "Final CV metrics - Accuracy: %.4f ± %.4f | F1: %.4f ± %.4",
                metrics_summary["accuracy"]["mean"],
                metrics_summary["accuracy"]["std"],
                metrics_summary["f1"]["mean"],
                metrics_summary["f1"]["std"],
            )

        return metrics_summary
