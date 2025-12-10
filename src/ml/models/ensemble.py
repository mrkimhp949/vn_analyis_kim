# -*- coding: utf-8 -*-
"""
Enhanced ML Models với:
- XGBoost & LightGBM
- Hyperparameter tuning
- Feature importance
- Model explainability (SHAP)
- Ensemble voting với weighted averaging
- Confidence calibration cho trading decisions
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS - Trading-specific thresholds
# =============================================================================

# Prediction thresholds cho trading decisions
CONFIDENCE_HIGH = 0.7  # High confidence threshold
CONFIDENCE_MEDIUM = 0.6  # Medium confidence threshold
CONFIDENCE_LOW = 0.5  # Neutral/uncertain threshold

# Model weights cho ensemble (dựa trên historical performance)
DEFAULT_MODEL_WEIGHTS = {
    "rf": 0.3,  # Random Forest - stable but less adaptive
    "xgb": 0.4,  # XGBoost - best for structured data
    "lgb": 0.3,  # LightGBM - fast and efficient
}

# Minimum samples required for reliable training
MIN_TRAINING_SAMPLES = 500
MIN_POSITIVE_RATIO = 0.1  # At least 10% positive samples
MAX_POSITIVE_RATIO = 0.9  # At most 90% positive samples


@dataclass
class PredictionResult:
    """Structured prediction result với confidence và metadata"""

    probability: float
    confidence_level: str  # 'high', 'medium', 'low'
    model_agreement: float  # 0-1, how much models agree
    individual_predictions: Dict[str, float]
    recommendation: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probability": self.probability,
            "confidence_level": self.confidence_level,
            "model_agreement": self.model_agreement,
            "individual_predictions": self.individual_predictions,
            "recommendation": self.recommendation,
        }


# Optional imports
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. Install: pip install xgboost")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Install: pip install lightgbm")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install: pip install shap")


class EnhancedMLPredictor:
    """
    Enhanced ML Predictor với multiple models và advanced features.

    Designed for Vietnam stock market prediction với:
    - Ensemble của RF, XGBoost, LightGBM
    - Weighted voting dựa trên model performance
    - Confidence calibration cho trading decisions
    - SHAP explainability

    Usage:
        predictor = EnhancedMLPredictor()
        predictor.load_models()  # Load pre-trained models
        result = predictor.predict_with_confidence(features)
        if result.confidence_level == 'high' and result.probability > 0.7:
            # Execute trade
    """

    def __init__(
        self, models_dir: str = "models", model_weights: Optional[Dict[str, float]] = None
    ):
        self.models_dir = models_dir
        self.ensure_models_dir()

        # Models
        self.rf_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.ensemble_model = None

        # Model weights cho weighted ensemble
        self.model_weights = model_weights or DEFAULT_MODEL_WEIGHTS.copy()

        # Scaler
        self.scaler = StandardScaler()
        self._scaler_fitted = False

        # Feature importance
        self.feature_importance: Dict[str, Dict[str, float]] = {}

        # SHAP explainer
        self.shap_explainer = None

        # Training metrics để track model quality
        self.training_metrics: Dict[str, Dict[str, float]] = {}

        # Expected features
        try:
            from src.ml.features.enhanced_v2 import get_feature_columns

            self.expected_features = len(get_feature_columns())
            self.feature_names = get_feature_columns()
        except ImportError:
            logger.warning("Could not import feature columns, using default")
            self.expected_features = 28  # Base 18 + 10 new features
            self.feature_names = [f"feature_{i}" for i in range(28)]

    def ensure_models_dir(self):
        """Tạo thư mục models nếu chưa có"""
        os.makedirs(self.models_dir, exist_ok=True)
        logger.info(f"✅ Models directory: {os.path.abspath(self.models_dir)}")

    # ========================================================================
    # TRAINING
    # ========================================================================

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        tune_hyperparameters: bool = False,
    ):
        """
        Train tất cả models

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            tune_hyperparameters: Có chạy hyperparameter tuning không
        """
        logger.info("🎓 Training all models...")

        # Validate input data
        self._validate_training_data(X_train, y_train)

        # Auto-adjust expected features if needed (first training)
        if self.expected_features != X_train.shape[1]:
            logger.warning(
                f"Adjusting expected features: {self.expected_features} -> {X_train.shape[1]}"
            )
            self.expected_features = X_train.shape[1]
            if not self.feature_names or len(self.feature_names) != self.expected_features:
                self.feature_names = [f"feature_{i}" for i in range(self.expected_features)]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        self._scaler_fitted = True
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_val_scaled = None

        # Train individual models
        self.train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val, tune_hyperparameters)

        if XGBOOST_AVAILABLE:
            self.train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val, tune_hyperparameters)

        if LIGHTGBM_AVAILABLE:
            self.train_lightgbm(X_train_scaled, y_train, X_val_scaled, y_val, tune_hyperparameters)

        # Create ensemble
        self.create_ensemble()

        # Calculate feature importance
        self.calculate_feature_importance(X_train_scaled, y_train)

        # Save models
        self.save_models()

        # Log training summary
        self._log_training_summary()
        logger.info("✅ All models trained successfully!")

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate training data quality cho trading model"""
        n_samples = len(y)

        # Check minimum samples
        if n_samples < MIN_TRAINING_SAMPLES:
            logger.warning(
                f"⚠️ Low sample count: {n_samples} < {MIN_TRAINING_SAMPLES}. " "Model may overfit!"
            )

        # Check class balance
        positive_ratio = y.sum() / len(y)
        if positive_ratio < MIN_POSITIVE_RATIO or positive_ratio > MAX_POSITIVE_RATIO:
            logger.warning(
                f"⚠️ Imbalanced classes: {positive_ratio:.1%} positive. "
                "Consider resampling or adjusting thresholds."
            )

        # Check for NaN/Inf
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("Training data contains NaN or Inf values!")

        # Check feature variance
        low_var_features = np.where(np.var(X, axis=0) < 1e-10)[0]
        if len(low_var_features) > 0:
            logger.warning(f"⚠️ {len(low_var_features)} features have near-zero variance")

    def _log_training_summary(self) -> None:
        """Log summary of trained models"""
        models_trained = []
        if self.rf_model is not None:
            models_trained.append("RandomForest")
        if self.xgb_model is not None:
            models_trained.append("XGBoost")
        if self.lgb_model is not None:
            models_trained.append("LightGBM")

        logger.info(f"📊 Training Summary:")
        logger.info(f"   Models trained: {', '.join(models_trained)}")
        logger.info(f"   Features: {self.expected_features}")
        logger.info(f"   Model weights: {self.model_weights}")

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        tune: bool = False,
    ):
        """Train Random Forest với optional hyperparameter tuning"""
        logger.info("🌲 Training Random Forest...")

        if tune:
            logger.info("   Tuning hyperparameters...")
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20],
                "min_samples_split": [5, 10, 15],
                "min_samples_leaf": [2, 5, 10],
            }

            rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)

            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring="f1", n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            self.rf_model = grid_search.best_estimator_
            logger.info(f"   Best params: {grid_search.best_params_}")
        else:
            self.rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            self.rf_model.fit(X_train, y_train)

        # Evaluate
        if X_val is not None and y_val is not None:
            self._evaluate_model(self.rf_model, X_val, y_val, "Random Forest")

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        tune: bool = False,
    ):
        """Train XGBoost"""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping...")
            return

        logger.info("🚀 Training XGBoost...")

        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        if tune:
            logger.info("   Tuning hyperparameters...")
            param_grid = {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200, 300],
                "subsample": [0.8, 0.9, 1.0],
            }

            xgb_clf = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1
            )

            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                xgb_clf, param_grid, cv=tscv, scoring="f1", n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)

            self.xgb_model = grid_search.best_estimator_
            logger.info(f"   Best params: {grid_search.best_params_}")
        else:
            self.xgb_model = xgb.XGBClassifier(
                max_depth=5,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            )

            # Early stopping nếu có validation set
            if X_val is not None and y_val is not None:
                self.xgb_model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=False,
                )
            else:
                self.xgb_model.fit(X_train, y_train)

        # Evaluate
        if X_val is not None and y_val is not None:
            self._evaluate_model(self.xgb_model, X_val, y_val, "XGBoost")

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        tune: bool = False,
    ):
        """Train LightGBM"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, skipping...")
            return

        logger.info("⚡ Training LightGBM...")

        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        if tune:
            logger.info("   Tuning hyperparameters...")
            param_grid = {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200, 300],
                "num_leaves": [31, 63, 127],
            }

            lgb_clf = lgb.LGBMClassifier(
                scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1
            )

            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                lgb_clf, param_grid, cv=tscv, scoring="f1", n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)

            self.lgb_model = grid_search.best_estimator_
            logger.info(f"   Best params: {grid_search.best_params_}")
        else:
            self.lgb_model = lgb.LGBMClassifier(
                max_depth=5,
                learning_rate=0.05,
                n_estimators=200,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            )

            # Early stopping
            if X_val is not None and y_val is not None:
                self.lgb_model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
                )
            else:
                self.lgb_model.fit(X_train, y_train)

        # Evaluate
        if X_val is not None and y_val is not None:
            self._evaluate_model(self.lgb_model, X_val, y_val, "LightGBM")

    def create_ensemble(self):
        """Tạo ensemble model từ các models đã train"""
        logger.info("🎭 Creating ensemble model...")

        estimators = []

        if self.rf_model is not None:
            estimators.append(("r", self.rf_model))

        if self.xgb_model is not None:
            estimators.append(("xgb", self.xgb_model))

        if self.lgb_model is not None:
            estimators.append(("lgb", self.lgb_model))

        if len(estimators) > 1:
            self.ensemble_model = VotingClassifier(
                estimators=estimators, voting="soft"  # Use probability voting
            )
            logger.info(f"   Ensemble created with {len(estimators)} models")
        else:
            logger.warning("   Not enough models for ensemble, using single model")
            self.ensemble_model = None

    # ========================================================================
    # PREDICTION
    # ========================================================================

    def predict(self, X: np.ndarray, use_ensemble: bool = True) -> np.ndarray:
        """
        Predict probabilities

        Args:
            X: Features
            use_ensemble: Use ensemble if available

        Returns:
            Probabilities of class 1 (price up)
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values

        X = np.asarray(X)

        if len(X) == 0:
            return np.array([])

        # Validate features
        if X.shape[1] != self.expected_features:
            raise ValueError(
                f"Feature mismatch: got {X.shape[1]}, " f"expected {self.expected_features}"
            )

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        if use_ensemble and self.ensemble_model is not None:
            # Ensemble prediction (already fitted during create_ensemble)
            # We need to manually predict since VotingClassifier needs to be fitted
            predictions = []

            if self.rf_model is not None:
                try:
                    predictions.append(self.rf_model.predict_proba(X_scaled)[:, 1])
                except ValueError as e:
                    logger.warning(f"RF prediction skipped: {e}")

            if self.xgb_model is not None:
                try:
                    predictions.append(self.xgb_model.predict_proba(X_scaled)[:, 1])
                except ValueError as e:
                    logger.warning(f"XGB prediction skipped: {e}")

            if self.lgb_model is not None:
                try:
                    predictions.append(self.lgb_model.predict_proba(X_scaled)[:, 1])
                except ValueError as e:
                    logger.warning(f"LGB/GB prediction skipped (feature mismatch): {e}")

            # Average predictions if any succeeded
            if predictions:
                return np.mean(predictions, axis=0)
            else:
                logger.warning("All models failed, returning neutral")
                return np.full(len(X), 0.5)

        elif self.rf_model is not None:
            return self.rf_model.predict_proba(X_scaled)[:, 1]

        else:
            # Fallback to neutral - KHÔNG random vì ảnh hưởng trading decision
            logger.warning("No models available, returning neutral probability")
            return np.full(len(X), 0.5)

    def predict_with_confidence(self, X: np.ndarray) -> List[PredictionResult]:
        """
        Predict với confidence level và recommendation cho trading.

        Args:
            X: Features array

        Returns:
            List of PredictionResult với detailed info cho mỗi sample
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        X = np.asarray(X)

        if len(X) == 0:
            return []

        # Get individual predictions
        individual_preds = self._get_individual_predictions(X)

        results = []
        for i in range(len(X)):
            sample_preds = {k: v[i] for k, v in individual_preds.items()}

            # Weighted average
            weighted_sum = 0.0
            total_weight = 0.0
            for model_name, pred in sample_preds.items():
                weight = self.model_weights.get(model_name, 0.33)
                weighted_sum += pred * weight
                total_weight += weight

            probability = weighted_sum / total_weight if total_weight > 0 else 0.5

            # Calculate model agreement (standard deviation based)
            if len(sample_preds) > 1:
                preds_array = np.array(list(sample_preds.values()))
                std_dev = np.std(preds_array)
                # Agreement: 1 = perfect agreement, 0 = max disagreement
                model_agreement = max(0, 1 - std_dev * 4)  # Scale: std=0.25 -> agreement=0
            else:
                model_agreement = 0.5  # Single model = medium confidence

            # Determine confidence level
            if model_agreement > 0.8 and (
                probability > CONFIDENCE_HIGH or probability < (1 - CONFIDENCE_HIGH)
            ):
                confidence_level = "high"
            elif model_agreement > 0.5 and (
                probability > CONFIDENCE_MEDIUM or probability < (1 - CONFIDENCE_MEDIUM)
            ):
                confidence_level = "medium"
            else:
                confidence_level = "low"

            # Trading recommendation
            recommendation = self._get_recommendation(probability, confidence_level)

            results.append(
                PredictionResult(
                    probability=probability,
                    confidence_level=confidence_level,
                    model_agreement=model_agreement,
                    individual_predictions=sample_preds,
                    recommendation=recommendation,
                )
            )

        return results

    def _get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions từ từng model riêng lẻ"""
        if not self._scaler_fitted:
            raise RuntimeError("Scaler not fitted. Load models or train first.")

        X_scaled = self.scaler.transform(X)
        predictions = {}

        if self.rf_model is not None:
            try:
                predictions["rf"] = self.rf_model.predict_proba(X_scaled)[:, 1]
            except Exception as e:
                logger.warning(f"RF prediction failed: {e}")

        if self.xgb_model is not None:
            try:
                predictions["xgb"] = self.xgb_model.predict_proba(X_scaled)[:, 1]
            except Exception as e:
                logger.warning(f"XGB prediction failed: {e}")

        if self.lgb_model is not None:
            try:
                predictions["lgb"] = self.lgb_model.predict_proba(X_scaled)[:, 1]
            except Exception as e:
                logger.warning(f"LGB prediction failed: {e}")

        return predictions

    def _get_recommendation(self, probability: float, confidence: str) -> str:
        """Convert probability và confidence thành trading recommendation"""
        if confidence == "high":
            if probability >= 0.75:
                return "strong_buy"
            elif probability >= 0.6:
                return "buy"
            elif probability <= 0.25:
                return "strong_sell"
            elif probability <= 0.4:
                return "sell"
        elif confidence == "medium":
            if probability >= 0.65:
                return "buy"
            elif probability <= 0.35:
                return "sell"

        return "hold"

    # ========================================================================
    # FEATURE IMPORTANCE & EXPLAINABILITY
    # ========================================================================

    def calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importance từ các models"""
        logger.info("📊 Calculating feature importance...")

        self.feature_importance = {}

        # Random Forest importance
        if self.rf_model is not None:
            self.feature_importance["r"] = dict(
                zip(self.feature_names, self.rf_model.feature_importances_)
            )

        # XGBoost importance
        if self.xgb_model is not None:
            self.feature_importance["xgb"] = dict(
                zip(self.feature_names, self.xgb_model.feature_importances_)
            )

        # LightGBM importance
        if self.lgb_model is not None:
            self.feature_importance["lgb"] = dict(
                zip(self.feature_names, self.lgb_model.feature_importances_)
            )

        # Average importance
        if self.feature_importance:
            avg_importance = {}
            for feature in self.feature_names:
                importances = [
                    imp_dict[feature]
                    for imp_dict in self.feature_importance.values()
                    if feature in imp_dict
                ]
                avg_importance[feature] = np.mean(importances) if importances else 0

            self.feature_importance["average"] = avg_importance

            # Sort by importance
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

            logger.info("   Top 10 features:")
            for feature, importance in sorted_features[:10]:
                logger.info(f"      {feature}: {importance:.4f}")

    def explain_prediction(self, X: np.ndarray, sample_idx: int = -1) -> Optional[Dict]:
        """
        Explain prediction using SHAP

        Args:
            X: Features
            sample_idx: Index of sample to explain (-1 for last)

        Returns:
            Dict with SHAP values and explanation
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return None

        if self.rf_model is None:
            logger.warning("No model available for explanation")
            return None

        try:
            # Create SHAP explainer if not exists
            if self.shap_explainer is None:
                logger.info("Creating SHAP explainer...")
                self.shap_explainer = shap.TreeExplainer(self.rf_model)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_scaled)

            # Get values for specific sample
            if isinstance(shap_values, list):
                # Binary classification returns list
                sample_shap = shap_values[1][sample_idx]
            else:
                sample_shap = shap_values[sample_idx]

            # Create explanation dict
            explanation = {
                "shap_values": dict(zip(self.feature_names, sample_shap)),
                "base_value": self.shap_explainer.expected_value,
                "prediction": self.predict(X[sample_idx : sample_idx + 1])[0],
            }

            # Sort by absolute SHAP value
            sorted_shap = sorted(
                explanation["shap_values"].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            explanation["top_features"] = sorted_shap[:5]

            return explanation

        except Exception:
            logger.error("Error explaining prediction")
            return None

    # ========================================================================
    # EVALUATION
    # ========================================================================

    def _evaluate_model(self, model, X_val: np.ndarray, y_val: np.ndarray, model_name: str):
        """Evaluate model performance"""
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_val, y_pred_proba)
        except ValueError:
            auc = 0.0

        # Store metrics for model weight adjustment
        self.training_metrics[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

        logger.info(f"   {model_name} Performance:")
        logger.info(f"      Accuracy:  {accuracy:.4f}")
        logger.info(f"      Precision: {precision:.4f}")
        logger.info(f"      Recall:    {recall:.4f}")
        logger.info(f"      F1-Score:  {f1:.4f}")
        logger.info(f"      AUC:       {auc:.4f}")

        # Trading-specific metrics
        # Precision quan trọng hơn recall cho trading (tránh false positives)
        if precision < 0.5:
            logger.warning(f"   ⚠️ Low precision for {model_name} - may generate false signals!")
        if auc < 0.55:
            logger.warning(f"   ⚠️ Low AUC for {model_name} - barely better than random!")

    def update_model_weights_from_performance(self) -> None:
        """Cập nhật model weights dựa trên training performance"""
        if not self.training_metrics:
            logger.warning("No training metrics available")
            return

        # Use F1 score weighted by AUC as performance metric
        performances = {}
        for model_key, metrics in self.training_metrics.items():
            # Map model names
            if "Random Forest" in model_key:
                key = "rf"
            elif "XGBoost" in model_key:
                key = "xgb"
            elif "LightGBM" in model_key:
                key = "lgb"
            else:
                continue
            performances[key] = metrics["f1"] * metrics["auc"]

        if not performances:
            return

        # Normalize to sum to 1
        total = sum(performances.values())
        if total > 0:
            self.model_weights = {k: v / total for k, v in performances.items()}
            logger.info(f"📊 Updated model weights based on performance: {self.model_weights}")

    # ========================================================================
    # SAVE/LOAD
    # ========================================================================

    def save_models(self):
        """Save all models"""
        logger.info("💾 Saving models...")

        try:
            # Save scaler
            joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler_enhanced.pkl"))

            # Save individual models
            if self.rf_model is not None:
                joblib.dump(self.rf_model, os.path.join(self.models_dir, "rf_enhanced.pkl"))

            if self.xgb_model is not None:
                joblib.dump(self.xgb_model, os.path.join(self.models_dir, "xgb_enhanced.pkl"))

            if self.lgb_model is not None:
                joblib.dump(self.lgb_model, os.path.join(self.models_dir, "lgb_enhanced.pkl"))

            # Save feature importance
            if self.feature_importance:
                joblib.dump(
                    self.feature_importance,
                    os.path.join(self.models_dir, "feature_importance.pkl"),
                )

            # Save metadata với training metrics
            metadata = {
                "expected_features": self.expected_features,
                "feature_names": self.feature_names,
                "models_available": {
                    "rf": self.rf_model is not None,
                    "xgb": self.xgb_model is not None,
                    "lgb": self.lgb_model is not None,
                },
                "model_weights": self.model_weights,
                "training_metrics": self.training_metrics,
                "saved_at": pd.Timestamp.now().isoformat(),
                "version": "2.0",  # Track model version
            }

            with open(os.path.join(self.models_dir, "model_info_enhanced.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info("✅ Models saved successfully!")

        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")

    def load_models(self) -> bool:
        """Load all models"""
        logger.info("📂 Loading models...")

        try:
            # Load scaler - try multiple possible names
            scaler_paths = [
                os.path.join(self.models_dir, "scaler_enhanced.pkl"),
                os.path.join(self.models_dir, "scaler.pkl"),
            ]
            for scaler_path in scaler_paths:
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"   ✅ Scaler loaded from {os.path.basename(scaler_path)}")
                    break

            # Load RF - try multiple possible names
            rf_paths = [
                os.path.join(self.models_dir, "rf_enhanced.pkl"),
                os.path.join(self.models_dir, "random_forest.pkl"),
                os.path.join(self.models_dir, "ensemble_rf.pkl"),
            ]
            for rf_path in rf_paths:
                if os.path.exists(rf_path):
                    self.rf_model = joblib.load(rf_path)
                    logger.info(f"   ✅ Random Forest loaded from {os.path.basename(rf_path)}")
                    break

            # Load XGBoost - try multiple possible names
            xgb_paths = [
                os.path.join(self.models_dir, "xgb_enhanced.pkl"),
                os.path.join(self.models_dir, "xgboost.pkl"),
                os.path.join(self.models_dir, "ensemble_xgb.pkl"),
            ]
            for xgb_path in xgb_paths:
                if os.path.exists(xgb_path):
                    self.xgb_model = joblib.load(xgb_path)
                    logger.info(f"   ✅ XGBoost loaded from {os.path.basename(xgb_path)}")
                    break

            # Load LightGBM/GradientBoosting - try multiple possible names
            lgb_paths = [
                os.path.join(self.models_dir, "lgb_enhanced.pkl"),
                os.path.join(self.models_dir, "lightgbm.pkl"),
                os.path.join(self.models_dir, "ensemble_gb.pkl"),
            ]
            for lgb_path in lgb_paths:
                if os.path.exists(lgb_path):
                    self.lgb_model = joblib.load(lgb_path)
                    logger.info(f"   ✅ LightGBM/GB loaded from {os.path.basename(lgb_path)}")

            # Load feature importance
            fi_path = os.path.join(self.models_dir, "feature_importance.pkl")
            if os.path.exists(fi_path):
                self.feature_importance = joblib.load(fi_path)

            # Load metadata và model weights
            metadata_path = os.path.join(self.models_dir, "model_info_enhanced.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    if "model_weights" in metadata:
                        self.model_weights = metadata["model_weights"]
                    if "training_metrics" in metadata:
                        self.training_metrics = metadata["training_metrics"]
                    if "expected_features" in metadata:
                        self.expected_features = metadata["expected_features"]
                    if "feature_names" in metadata:
                        self.feature_names = metadata["feature_names"]

            # Mark scaler as fitted
            self._scaler_fitted = True

            # Recreate ensemble
            self.create_ensemble()

            # Log loaded models
            loaded_models = []
            if self.rf_model:
                loaded_models.append("RF")
            if self.xgb_model:
                loaded_models.append("XGB")
            if self.lgb_model:
                loaded_models.append("LGB")

            logger.info(f"✅ Models loaded successfully: {', '.join(loaded_models)}")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if predictor is ready for predictions"""
        return self._scaler_fitted and (
            self.rf_model is not None or self.xgb_model is not None or self.lgb_model is not None
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive info about loaded models"""
        return {
            "ready": self.is_ready(),
            "models": {
                "rf": self.rf_model is not None,
                "xgb": self.xgb_model is not None,
                "lgb": self.lgb_model is not None,
            },
            "expected_features": self.expected_features,
            "model_weights": self.model_weights,
            "training_metrics": self.training_metrics,
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("🧪 TESTING ENHANCED ML PREDICTOR")
    print("=" * 70 + "\n")

    # Create dummy data with realistic distribution
    n_samples = 1000
    n_features = 28

    np.random.seed(42)  # For reproducibility
    X = np.random.randn(n_samples, n_features)
    # Imbalanced like real trading (30% positive signals)
    y = (np.random.rand(n_samples) < 0.3).astype(int)

    # Split chronologically (important for time series!)
    split = int(n_samples * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"📊 Data: {n_samples} samples, {n_features} features")
    print(f"   Train: {len(y_train)} ({y_train.sum()/len(y_train):.1%} positive)")
    print(f"   Val:   {len(y_val)} ({y_val.sum()/len(y_val):.1%} positive)\n")

    # Train
    predictor = EnhancedMLPredictor()
    predictor.train_all_models(X_train, y_train, X_val, y_val, tune_hyperparameters=False)

    # Update weights based on performance
    predictor.update_model_weights_from_performance()

    # Check if ready
    print(f"\n🔍 Model ready: {predictor.is_ready()}")
    print(f"   Model info: {predictor.get_model_info()}")

    # Basic predict
    predictions = predictor.predict(X_val)
    print(f"\n📊 Basic Predictions (first 5): {predictions[:5]}")

    # Predict with confidence - THE NEW WAY!
    results = predictor.predict_with_confidence(X_val[:10])
    print("\n📊 Predictions with Confidence:")
    for i, result in enumerate(results):
        print(
            f"   Sample {i}: prob={result.probability:.3f}, "
            f"confidence={result.confidence_level}, "
            f"agreement={result.model_agreement:.2f}, "
            f"recommendation={result.recommendation}"
        )

    # Feature importance
    if predictor.feature_importance and "average" in predictor.feature_importance:
        print("\n📊 Top 5 features:")
        sorted_features = sorted(
            predictor.feature_importance["average"].items(), key=lambda x: x[1], reverse=True
        )
        for feature, importance in sorted_features[:5]:
            print(f"   {feature}: {importance:.4f}")

    print("\n✅ Testing complete!")
