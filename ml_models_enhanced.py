# -*- coding: utf-8 -*-
"""
Enhanced ML Models với:
- XGBoost & LightGBM
- Hyperparameter tuning
- Feature importance
- Model explainability (SHAP)
- Ensemble voting
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

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
    Enhanced ML Predictor với multiple models và advanced features
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.ensure_models_dir()
        
        # Models
        self.rf_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.ensemble_model = None
        
        # Scaler
        self.scaler = StandardScaler()
        
        # Feature importance
        self.feature_importance = {}
        
        # SHAP explainer
        self.shap_explainer = None
        
        # Expected features
        try:
            from features_enhanced import get_feature_columns
            self.expected_features = len(get_feature_columns())
            self.feature_names = get_feature_columns()
        except Exception:
            self.expected_features = 28  # Base 18 + 10 new features
            self.feature_names = []
    
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
        tune_hyperparameters: bool = False
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
        
        # Validate features
        if X_train.shape[1] != self.expected_features:
            raise ValueError(
                f"Feature mismatch: got {X_train.shape[1]}, "
                f"expected {self.expected_features}"
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_val_scaled = None
        
        # Train individual models
        self.train_random_forest(
            X_train_scaled, y_train, X_val_scaled, y_val, tune_hyperparameters
        )
        
        if XGBOOST_AVAILABLE:
            self.train_xgboost(
                X_train_scaled, y_train, X_val_scaled, y_val, tune_hyperparameters
            )
        
        if LIGHTGBM_AVAILABLE:
            self.train_lightgbm(
                X_train_scaled, y_train, X_val_scaled, y_val, tune_hyperparameters
            )
        
        # Create ensemble
        self.create_ensemble()
        
        # Calculate feature importance
        self.calculate_feature_importance(X_train_scaled, y_train)
        
        # Save models
        self.save_models()
        
        logger.info("✅ All models trained successfully!")
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        tune: bool = False
    ):
        """Train Random Forest với optional hyperparameter tuning"""
        logger.info("🌲 Training Random Forest...")
        
        if tune:
            logger.info("   Tuning hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10],
            }
            
            rf = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                rf, param_grid, cv=tscv, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.rf_model = grid_search.best_estimator_
            logger.info(f"   Best params: {grid_search.best_params_}")
        else:
            self.rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
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
        tune: bool = False
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
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 0.9, 1.0],
            }
            
            xgb_clf = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1
            )
            
            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                xgb_clf, param_grid, cv=tscv, scoring='f1', n_jobs=-1, verbose=1
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
                n_jobs=-1
            )
            
            # Early stopping nếu có validation set
            if X_val is not None and y_val is not None:
                self.xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=False
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
        tune: bool = False
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
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'num_leaves': [31, 63, 127],
            }
            
            lgb_clf = lgb.LGBMClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1
            )
            
            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                lgb_clf, param_grid, cv=tscv, scoring='f1', n_jobs=-1, verbose=1
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
                n_jobs=-1
            )
            
            # Early stopping
            if X_val is not None and y_val is not None:
                self.lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
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
            estimators.append(('rf', self.rf_model))
        
        if self.xgb_model is not None:
            estimators.append(('xgb', self.xgb_model))
        
        if self.lgb_model is not None:
            estimators.append(('lgb', self.lgb_model))
        
        if len(estimators) > 1:
            self.ensemble_model = VotingClassifier(
                estimators=estimators,
                voting='soft'  # Use probability voting
            )
            logger.info(f"   Ensemble created with {len(estimators)} models")
        else:
            logger.warning("   Not enough models for ensemble, using single model")
            self.ensemble_model = None
    
    # ========================================================================
    # PREDICTION
    # ========================================================================
    
    def predict(
        self,
        X: np.ndarray,
        use_ensemble: bool = True
    ) -> np.ndarray:
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
                f"Feature mismatch: got {X.shape[1]}, "
                f"expected {self.expected_features}"
            )
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        if use_ensemble and self.ensemble_model is not None:
            # Ensemble prediction (already fitted during create_ensemble)
            # We need to manually predict since VotingClassifier needs to be fitted
            predictions = []
            
            if self.rf_model is not None:
                predictions.append(self.rf_model.predict_proba(X_scaled)[:, 1])
            
            if self.xgb_model is not None:
                predictions.append(self.xgb_model.predict_proba(X_scaled)[:, 1])
            
            if self.lgb_model is not None:
                predictions.append(self.lgb_model.predict_proba(X_scaled)[:, 1])
            
            # Average predictions
            return np.mean(predictions, axis=0)
        
        elif self.rf_model is not None:
            return self.rf_model.predict_proba(X_scaled)[:, 1]
        
        else:
            # Fallback to random
            return np.random.uniform(0.3, 0.7, len(X))
    
    # ========================================================================
    # FEATURE IMPORTANCE & EXPLAINABILITY
    # ========================================================================
    
    def calculate_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ):
        """Calculate feature importance từ các models"""
        logger.info("📊 Calculating feature importance...")
        
        self.feature_importance = {}
        
        # Random Forest importance
        if self.rf_model is not None:
            self.feature_importance['rf'] = dict(
                zip(self.feature_names, self.rf_model.feature_importances_)
            )
        
        # XGBoost importance
        if self.xgb_model is not None:
            self.feature_importance['xgb'] = dict(
                zip(self.feature_names, self.xgb_model.feature_importances_)
            )
        
        # LightGBM importance
        if self.lgb_model is not None:
            self.feature_importance['lgb'] = dict(
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
            
            self.feature_importance['average'] = avg_importance
            
            # Sort by importance
            sorted_features = sorted(
                avg_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            logger.info("   Top 10 features:")
            for feature, importance in sorted_features[:10]:
                logger.info(f"      {feature}: {importance:.4f}")
    
    def explain_prediction(
        self,
        X: np.ndarray,
        sample_idx: int = -1
    ) -> Optional[Dict]:
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
                'shap_values': dict(zip(self.feature_names, sample_shap)),
                'base_value': self.shap_explainer.expected_value,
                'prediction': self.predict(X[sample_idx:sample_idx+1])[0]
            }
            
            # Sort by absolute SHAP value
            sorted_shap = sorted(
                explanation['shap_values'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            explanation['top_features'] = sorted_shap[:5]
            
            return explanation
        
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return None
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    def _evaluate_model(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str
    ):
        """Evaluate model performance"""
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
        except:
            auc = 0.0
        
        logger.info(f"   {model_name} Performance:")
        logger.info(f"      Accuracy:  {accuracy:.4f}")
        logger.info(f"      Precision: {precision:.4f}")
        logger.info(f"      Recall:    {recall:.4f}")
        logger.info(f"      F1-Score:  {f1:.4f}")
        logger.info(f"      AUC:       {auc:.4f}")
    
    # ========================================================================
    # SAVE/LOAD
    # ========================================================================
    
    def save_models(self):
        """Save all models"""
        logger.info("💾 Saving models...")
        
        try:
            # Save scaler
            joblib.dump(
                self.scaler,
                os.path.join(self.models_dir, 'scaler_enhanced.pkl')
            )
            
            # Save individual models
            if self.rf_model is not None:
                joblib.dump(
                    self.rf_model,
                    os.path.join(self.models_dir, 'rf_enhanced.pkl')
                )
            
            if self.xgb_model is not None:
                joblib.dump(
                    self.xgb_model,
                    os.path.join(self.models_dir, 'xgb_enhanced.pkl')
                )
            
            if self.lgb_model is not None:
                joblib.dump(
                    self.lgb_model,
                    os.path.join(self.models_dir, 'lgb_enhanced.pkl')
                )
            
            # Save feature importance
            if self.feature_importance:
                joblib.dump(
                    self.feature_importance,
                    os.path.join(self.models_dir, 'feature_importance.pkl')
                )
            
            # Save metadata
            metadata = {
                'expected_features': self.expected_features,
                'feature_names': self.feature_names,
                'models_available': {
                    'rf': self.rf_model is not None,
                    'xgb': self.xgb_model is not None,
                    'lgb': self.lgb_model is not None,
                },
                'saved_at': pd.Timestamp.now().isoformat()
            }
            
            import json
            with open(os.path.join(self.models_dir, 'model_info_enhanced.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✅ Models saved successfully!")
        
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
    
    def load_models(self) -> bool:
        """Load all models"""
        logger.info("📂 Loading models...")
        
        try:
            # Load scaler
            scaler_path = os.path.join(self.models_dir, 'scaler_enhanced.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load RF
            rf_path = os.path.join(self.models_dir, 'rf_enhanced.pkl')
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                logger.info("   ✅ Random Forest loaded")
            
            # Load XGBoost
            xgb_path = os.path.join(self.models_dir, 'xgb_enhanced.pkl')
            if os.path.exists(xgb_path):
                self.xgb_model = joblib.load(xgb_path)
                logger.info("   ✅ XGBoost loaded")
            
            # Load LightGBM
            lgb_path = os.path.join(self.models_dir, 'lgb_enhanced.pkl')
            if os.path.exists(lgb_path):
                self.lgb_model = joblib.load(lgb_path)
                logger.info("   ✅ LightGBM loaded")
            
            # Load feature importance
            fi_path = os.path.join(self.models_dir, 'feature_importance.pkl')
            if os.path.exists(fi_path):
                self.feature_importance = joblib.load(fi_path)
            
            # Recreate ensemble
            self.create_ensemble()
            
            logger.info("✅ Models loaded successfully!")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING ENHANCED ML PREDICTOR")
    print("="*70 + "\n")
    
    # Create dummy data
    n_samples = 1000
    n_features = 28
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Split
    split = int(n_samples * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Train
    predictor = EnhancedMLPredictor()
    predictor.train_all_models(X_train, y_train, X_val, y_val, tune_hyperparameters=False)
    
    # Predict
    predictions = predictor.predict(X_val)
    print(f"\n📊 Predictions: {predictions[:5]}")
    
    # Feature importance
    if predictor.feature_importance:
        print("\n📊 Top 5 features:")
        for feature, importance in list(predictor.feature_importance['average'].items())[:5]:
            print(f"   {feature}: {importance:.4f}")
    
    print("\n✅ Testing complete!")
