"""
Stacking Ensemble with Meta-Model
Dùng Logistic Regression hoặc LightGBM làm meta-layer để kết hợp RF, XGB, LSTM
"""
import logging
import os
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

# Optional import for LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm. Will use LogisticRegression instead.")


class StackingEnsemble:
    """
    Stacking ensemble với meta-model
    
    Base models: RF, GB, XGBoost, LSTM
    Meta model: LogisticRegression hoặc LightGBM
    """
    
    def __init__(
        self,
        base_models: List,
        meta_model_type: str = "lightgbm",  # "lightgbm" or "logistic"
        use_cv: bool = True,
        n_splits: int = 5
    ):
        self.base_models = base_models
        self.meta_model_type = meta_model_type
        self.use_cv = use_cv
        self.n_splits = n_splits
        self.meta_model = None
        self.base_model_names = []
    
    def _build_meta_model(self, n_base_models: int):
        """Build meta-model"""
        if self.meta_model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            self.meta_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            logger.info("Using LightGBM as meta-model")
        else:
            # Use LogisticRegression as fallback
            self.meta_model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            )
            logger.info("Using LogisticRegression as meta-model")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """
        Train stacking ensemble
        
        Returns:
            Dict với metrics
        """
        n_base_models = len(self.base_models)
        self._build_meta_model(n_base_models)
        
        # Step 1: Get base model predictions on validation set
        base_predictions = []
        
        for i, model in enumerate(self.base_models):
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)[:, 1]  # Probability of class 1
                else:
                    # For LSTM or other models
                    pred = model.predict(X_val).flatten()
                    # Convert to probabilities if needed
                    if pred.max() > 1 or pred.min() < 0:
                        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                
                base_predictions.append(pred)
                model_name = getattr(model, '__class__', type(model)).__name__
                self.base_model_names.append(model_name)
                logger.info(f"Base model {i+1} ({model_name}): predictions shape {pred.shape}")
            except Exception as e:
                logger.warning(f"Failed to get predictions from base model {i+1}: {e}")
                # Use zeros as fallback
                base_predictions.append(np.zeros(len(y_val)))
                self.base_model_names.append(f"Model_{i+1}")
        
        # Stack base predictions
        meta_features = np.column_stack(base_predictions)
        logger.info(f"Meta features shape: {meta_features.shape}")
        
        # Step 2: Train meta-model on base predictions
        self.meta_model.fit(meta_features, y_val)
        
        # Step 3: Evaluate
        meta_pred = self.meta_model.predict_proba(meta_features)[:, 1]
        meta_cls = (meta_pred >= 0.5).astype(int)
        
        metrics = {
            "accuracy": float(accuracy_score(y_val, meta_cls)),
            "precision": float(precision_score(y_val, meta_cls, zero_division=0)),
            "recall": float(recall_score(y_val, meta_cls, zero_division=0)),
            "f1": float(f1_score(y_val, meta_cls, zero_division=0)),
        }
        
        logger.info(f"Stacking ensemble - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict với stacking ensemble
        
        Returns:
            (probabilities, classes)
        """
        if self.meta_model is None:
            raise ValueError("Meta-model chưa được train. Gọi fit() trước.")
        
        # Get base model predictions
        base_predictions = []
        
        for model in self.base_models:
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X).flatten()
                    if pred.max() > 1 or pred.min() < 0:
                        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                
                base_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to get predictions from base model: {e}")
                base_predictions.append(np.zeros(len(X)))
        
        # Stack and predict with meta-model
        meta_features = np.column_stack(base_predictions)
        probabilities = self.meta_model.predict_proba(meta_features)[:, 1]
        classes = (probabilities >= 0.5).astype(int)
        
        return probabilities, classes
    
    def save(self, save_dir: str):
        """Save meta-model"""
        os.makedirs(save_dir, exist_ok=True)
        meta_model_path = os.path.join(save_dir, "stacking_meta_model.pkl")
        joblib.dump(self.meta_model, meta_model_path)
        logger.info(f"Saved stacking meta-model to {meta_model_path}")
    
    def load(self, save_dir: str):
        """Load meta-model"""
        meta_model_path = os.path.join(save_dir, "stacking_meta_model.pkl")
        if os.path.exists(meta_model_path):
            self.meta_model = joblib.load(meta_model_path)
            logger.info(f"Loaded stacking meta-model from {meta_model_path}")
        else:
            raise FileNotFoundError(f"Meta-model not found at {meta_model_path}")

