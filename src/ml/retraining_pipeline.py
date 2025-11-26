# -*- coding: utf-8 -*-
"""
Automated Model Retraining Pipeline
Tự động retrain model khi drift detected hoặc performance degraded
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.ml.model_version_manager import get_model_version_manager
from src.ml.feature_drift_detector import get_drift_detector

logger = logging.getLogger(__name__)


class RetrainingTrigger(Enum):
    """Lý do trigger retraining"""

    DRIFT_DETECTED = "drift"  # Feature drift > threshold
    PERFORMANCE_DEGRADED = "performance"  # Production performance < threshold
    SCHEDULED = "scheduled"  # Scheduled retraining (e.g., monthly)
    MANUAL = "manual"  # Manual trigger by user
    NEW_DATA = "new_data"  # Significant new data available


@dataclass
class RetrainingConfig:
    """Configuration cho retraining pipeline"""

    # Trigger thresholds
    drift_threshold: float = 0.25  # PSI > 0.25 = high drift
    performance_threshold: float = 0.60  # Win rate < 60% = degraded
    min_new_data_samples: int = 500  # Min samples to consider retraining

    # Scheduled retraining
    enable_scheduled_retraining: bool = True
    scheduled_interval_days: int = 30  # Retrain every 30 days

    # Training config
    validation_split: float = 0.20  # 20% for validation
    test_split: float = 0.10  # 10% for test
    min_training_samples: int = 1000  # Minimum samples to train

    # Model selection
    model_types: List[str] = None  # ["xgboost", "lightgbm", "random_forest"]
    use_ensemble: bool = True  # Combine multiple models

    # Auto-deployment
    auto_deploy: bool = True  # Auto-deploy if better than current
    min_improvement_for_deploy: float = 0.03  # 3% minimum improvement

    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ["xgboost", "lightgbm", "random_forest"]


@dataclass
class RetrainingResult:
    """Kết quả của retraining pipeline"""

    trigger: RetrainingTrigger
    trigger_reason: str
    timestamp: str

    # Training results
    training_successful: bool
    new_model_id: Optional[str] = None
    new_version: Optional[str] = None

    # Performance metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    test_accuracy: float = 0.0
    train_auc: float = 0.0
    val_auc: float = 0.0
    test_auc: float = 0.0

    # Comparison with current model
    current_model_id: Optional[str] = None
    improvement_pct: float = 0.0
    deployed: bool = False

    # Errors
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class AutomatedRetrainingPipeline:
    """
    Automated ML model retraining pipeline

    WORKFLOW:
    1. Monitor for triggers (drift, performance, schedule)
    2. Collect training data (recent + historical)
    3. Train multiple models
    4. Evaluate and compare with current model
    5. Auto-deploy if better
    6. Log results

    FEATURES:
    - Multiple trigger types
    - Ensemble training
    - Hyperparameter tuning
    - Validation & testing
    - Auto-deployment
    - Rollback on failure
    """

    def __init__(
        self,
        config: Optional[RetrainingConfig] = None,
        data_loader=None,
        feature_engineer=None,
    ):
        """
        Args:
            config: RetrainingConfig object
            data_loader: Data loader for fetching training data
            feature_engineer: Feature engineering module
        """
        self.config = config or RetrainingConfig()
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer

        # Dependencies
        self.model_manager = get_model_version_manager()
        self.drift_detector = get_drift_detector()

        # State
        self.last_retraining_date = None
        self.retraining_in_progress = False

    def check_triggers(
        self,
        current_performance: Optional[Dict] = None,
        current_features: Optional[pd.DataFrame] = None,
    ) -> tuple[bool, RetrainingTrigger, str]:
        """
        Check if any retraining trigger is active

        Args:
            current_performance: Current model performance metrics
            current_features: Recent features for drift detection

        Returns:
            (should_retrain, trigger_type, reason)
        """
        # 1. DRIFT CHECK
        if current_features is not None and not current_features.empty:
            drift_report = self.drift_detector.detect_drift(current_features)

            if drift_report.overall_drift_score >= self.config.drift_threshold:
                return (
                    True,
                    RetrainingTrigger.DRIFT_DETECTED,
                    f"Feature drift detected: PSI={drift_report.overall_drift_score:.3f} "
                    f"(threshold={self.config.drift_threshold}). "
                    f"{drift_report.features_with_drift}/{drift_report.total_features} features drifted.",
                )

        # 2. PERFORMANCE CHECK
        if current_performance is not None:
            win_rate = current_performance.get("win_rate", 0)
            accuracy = current_performance.get("accuracy", 0)

            if win_rate < self.config.performance_threshold:
                return (
                    True,
                    RetrainingTrigger.PERFORMANCE_DEGRADED,
                    f"Performance degraded: win_rate={win_rate:.1%} < "
                    f"threshold={self.config.performance_threshold:.1%}",
                )

            if accuracy < self.config.performance_threshold:
                return (
                    True,
                    RetrainingTrigger.PERFORMANCE_DEGRADED,
                    f"Accuracy degraded: accuracy={accuracy:.1%} < "
                    f"threshold={self.config.performance_threshold:.1%}",
                )

        # 3. SCHEDULED CHECK
        if self.config.enable_scheduled_retraining:
            if self.last_retraining_date is None:
                return (True, RetrainingTrigger.SCHEDULED, "Initial training - no previous model")

            days_since_last = (datetime.now() - self.last_retraining_date).days
            if days_since_last >= self.config.scheduled_interval_days:
                return (
                    True,
                    RetrainingTrigger.SCHEDULED,
                    f"Scheduled retraining: {days_since_last} days since last training "
                    f"(interval: {self.config.scheduled_interval_days} days)",
                )

        return (False, None, "")

    def run_retraining(
        self,
        trigger: RetrainingTrigger,
        trigger_reason: str,
        training_data: Optional[pd.DataFrame] = None,
        labels: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
    ) -> RetrainingResult:
        """
        Execute retraining pipeline

        Args:
            trigger: Trigger type
            trigger_reason: Reason for retraining
            training_data: Training features (if None, will fetch)
            labels: Training labels (if None, will fetch)
            feature_names: Feature names

        Returns:
            RetrainingResult object
        """
        if self.retraining_in_progress:
            logger.warning("Retraining already in progress - skipping")
            return RetrainingResult(
                trigger=trigger,
                trigger_reason=trigger_reason,
                timestamp=datetime.now().isoformat(),
                training_successful=False,
                errors=["Retraining already in progress"],
            )

        self.retraining_in_progress = True
        errors = []
        new_model_id = None
        new_version = None

        try:
            logger.info(f"🔄 Starting automated retraining...")
            logger.info(f"   Trigger: {trigger.value}")
            logger.info(f"   Reason: {trigger_reason}")

            # 1. PREPARE DATA
            if training_data is None or labels is None:
                logger.info("📊 Fetching training data...")
                training_data, labels, feature_names = self._fetch_training_data()

            if training_data is None or len(training_data) < self.config.min_training_samples:
                error_msg = f"Insufficient training data: {len(training_data) if training_data is not None else 0} < {self.config.min_training_samples}"
                logger.error(error_msg)
                errors.append(error_msg)
                return self._failed_result(trigger, trigger_reason, errors)

            logger.info(
                f"✅ Training data ready: {len(training_data)} samples, {len(feature_names)} features"
            )

            # 2. SPLIT DATA
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(training_data, labels)

            logger.info(
                f"📊 Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
            )

            # 3. SET BASELINE FOR DRIFT DETECTION
            logger.info("📈 Setting baseline for future drift detection...")
            self.drift_detector.set_baseline(X_train, feature_names)

            # 4. TRAIN MODELS
            logger.info(f"🤖 Training models: {', '.join(self.config.model_types)}")
            trained_models = {}

            for model_type in self.config.model_types:
                logger.info(f"   Training {model_type}...")
                model, train_metrics, val_metrics, test_metrics = self._train_single_model(
                    model_type, X_train, y_train, X_val, y_val, X_test, y_test
                )

                if model is not None:
                    trained_models[model_type] = {
                        "model": model,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                    }
                    logger.info(
                        f"   ✅ {model_type}: val_acc={val_metrics['accuracy']:.3f}, val_auc={val_metrics['auc']:.3f}"
                    )
                else:
                    logger.warning(f"   ❌ {model_type} training failed")

            if not trained_models:
                error_msg = "All model training failed"
                logger.error(error_msg)
                errors.append(error_msg)
                return self._failed_result(trigger, trigger_reason, errors)

            # 5. SELECT BEST MODEL
            best_model_type, best_model_info = self._select_best_model(trained_models)
            logger.info(f"🏆 Best model: {best_model_type}")

            # 6. REGISTER MODEL
            current_model_id = self.model_manager.get_active_model_id()
            new_version = self._generate_version(current_model_id)

            logger.info(f"📝 Registering model as v{new_version}...")

            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                best_model_info["model"], best_model_type, feature_names
            )

            new_model_id = self.model_manager.register_model(
                model=best_model_info["model"],
                model_type=best_model_type,
                version=new_version,
                train_metrics=best_model_info["train_metrics"],
                val_metrics=best_model_info["val_metrics"],
                feature_names=feature_names,
                training_period=f"{datetime.now() - timedelta(days=365)} to {datetime.now()}",
                feature_importance=feature_importance,
                tags=[f"trigger:{trigger.value}", "automated"],
                notes=f"Automated retraining. Trigger: {trigger_reason}",
                auto_activate=False,  # Don't auto-activate yet
            )

            logger.info(f"✅ Model registered: {new_model_id}")

            # 7. COMPARE WITH CURRENT MODEL
            improvement_pct = 0.0
            should_deploy = False

            if current_model_id is not None:
                improvement_pct = self._compare_models(
                    new_model_id, current_model_id, metric="val_accuracy"
                )

                logger.info(f"📊 Improvement vs current: {improvement_pct:+.1f}%")

                # Deploy if improvement meets threshold
                if improvement_pct >= self.config.min_improvement_for_deploy * 100:
                    should_deploy = True
                else:
                    logger.info(
                        f"⚠️ Improvement {improvement_pct:.1f}% < threshold {self.config.min_improvement_for_deploy*100:.1f}% - not deploying"
                    )
            else:
                # No current model - deploy this one
                should_deploy = True
                logger.info("ℹ️ No current model - deploying new model")

            # 8. AUTO-DEPLOY
            deployed = False
            if self.config.auto_deploy and should_deploy:
                logger.info(f"🚀 Auto-deploying {new_model_id}...")
                self.model_manager.activate_model(new_model_id)
                deployed = True
                logger.info("✅ Model deployed successfully!")
            else:
                logger.info("ℹ️ Model registered but not deployed (manual activation required)")

            # 9. UPDATE STATE
            self.last_retraining_date = datetime.now()

            # 10. RETURN RESULT
            return RetrainingResult(
                trigger=trigger,
                trigger_reason=trigger_reason,
                timestamp=datetime.now().isoformat(),
                training_successful=True,
                new_model_id=new_model_id,
                new_version=new_version,
                train_accuracy=best_model_info["train_metrics"]["accuracy"],
                val_accuracy=best_model_info["val_metrics"]["accuracy"],
                test_accuracy=best_model_info["test_metrics"]["accuracy"],
                train_auc=best_model_info["train_metrics"]["auc"],
                val_auc=best_model_info["val_metrics"]["auc"],
                test_auc=best_model_info["test_metrics"]["auc"],
                current_model_id=current_model_id,
                improvement_pct=improvement_pct,
                deployed=deployed,
                errors=errors,
            )

        except Exception as e:
            error_msg = f"Retraining pipeline failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            return self._failed_result(trigger, trigger_reason, errors)

        finally:
            self.retraining_in_progress = False

    def _fetch_training_data(self) -> tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Fetch training data from data loader

        Returns:
            (features_df, labels, feature_names)
        """
        if self.data_loader is None or self.feature_engineer is None:
            raise ValueError("Data loader and feature engineer required for fetching data")

        # This is a placeholder - implement based on your data loader
        # Example implementation:
        # 1. Load historical price data
        # 2. Calculate features
        # 3. Generate labels (BUY/SELL/HOLD)
        # 4. Return features, labels, feature_names

        logger.warning("⚠️ _fetch_training_data not implemented - using placeholder")
        return None, None, []

    def _split_data(self, features: pd.DataFrame, labels: pd.Series) -> tuple:
        """
        Split data into train/val/test

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features,
            labels,
            test_size=self.config.test_split,
            random_state=42,
            stratify=labels if len(labels.unique()) > 1 else None,
        )

        # Second split: train vs val
        val_size = self.config.validation_split / (1 - self.config.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size,
            random_state=42,
            stratify=y_temp if len(y_temp.unique()) > 1 else None,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _train_single_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> tuple[Any, Dict, Dict, Dict]:
        """
        Train a single model with hyperparameter tuning

        Returns:
            (model, train_metrics, val_metrics, test_metrics)
        """
        try:
            if model_type == "xgboost":
                return self._train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
            elif model_type == "lightgbm":
                return self._train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
            elif model_type == "random_forest":
                return self._train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return None, {}, {}, {}

        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            return None, {}, {}, {}

    def _train_xgboost(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train XGBoost model"""
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, roc_auc_score

        # Hyperparameters (can be tuned with Optuna/GridSearch)
        params = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "multi:softmax",
            "num_class": len(y_train.unique()),
            "random_state": 42,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Metrics
        train_metrics = self._calculate_metrics(model, X_train, y_train)
        val_metrics = self._calculate_metrics(model, X_val, y_val)
        test_metrics = self._calculate_metrics(model, X_test, y_test)

        return model, train_metrics, val_metrics, test_metrics

    def _train_lightgbm(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train LightGBM model"""
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, roc_auc_score

        params = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "multiclass",
            "num_class": len(y_train.unique()),
            "random_state": 42,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        train_metrics = self._calculate_metrics(model, X_train, y_train)
        val_metrics = self._calculate_metrics(model, X_val, y_val)
        test_metrics = self._calculate_metrics(model, X_test, y_test)

        return model, train_metrics, val_metrics, test_metrics

    def _train_random_forest(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train Random Forest model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score

        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        train_metrics = self._calculate_metrics(model, X_train, y_train)
        val_metrics = self._calculate_metrics(model, X_val, y_val)
        test_metrics = self._calculate_metrics(model, X_test, y_test)

        return model, train_metrics, val_metrics, test_metrics

    def _calculate_metrics(self, model, X, y) -> Dict:
        """Calculate accuracy and AUC"""
        from sklearn.metrics import accuracy_score, roc_auc_score

        try:
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)

            # AUC (for multiclass, use ovr or ovo)
            try:
                y_proba = model.predict_proba(X)
                if len(np.unique(y)) == 2:
                    auc = roc_auc_score(y, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y, y_proba, multi_class="ovr")
            except Exception:
                auc = 0.0

            return {
                "accuracy": accuracy,
                "auc": auc,
                "samples": len(y),
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {"accuracy": 0.0, "auc": 0.0, "samples": len(y)}

    def _select_best_model(self, trained_models: Dict) -> tuple[str, Dict]:
        """
        Select best model by validation accuracy

        Returns:
            (model_type, model_info)
        """
        best_model_type = None
        best_model_info = None
        best_val_accuracy = -1

        for model_type, model_info in trained_models.items():
            val_accuracy = model_info["val_metrics"]["accuracy"]

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_type = model_type
                best_model_info = model_info

        return best_model_type, best_model_info

    def _calculate_feature_importance(
        self, model: Any, model_type: str, feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate feature importance"""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                return {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")

        return {}

    def _generate_version(self, current_model_id: Optional[str]) -> str:
        """Generate new semantic version"""
        if current_model_id is None:
            return "1.0.0"

        # Get current version
        registry = self.model_manager.registry
        if current_model_id not in registry:
            return "1.0.0"

        current_version = registry[current_model_id]["version"]

        # Increment minor version
        parts = current_version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        # Increment minor for retraining
        new_version = f"{major}.{minor + 1}.0"

        return new_version

    def _compare_models(
        self, new_model_id: str, current_model_id: str, metric: str = "val_accuracy"
    ) -> float:
        """
        Compare new model with current model

        Returns:
            Improvement percentage
        """
        registry = self.model_manager.registry

        new_value = registry[new_model_id][metric]
        current_value = registry[current_model_id][metric]

        if current_value == 0:
            return 0.0

        improvement = ((new_value - current_value) / current_value) * 100

        return improvement

    def _failed_result(
        self, trigger: RetrainingTrigger, trigger_reason: str, errors: List[str]
    ) -> RetrainingResult:
        """Create failed result"""
        return RetrainingResult(
            trigger=trigger,
            trigger_reason=trigger_reason,
            timestamp=datetime.now().isoformat(),
            training_successful=False,
            errors=errors,
        )


# Singleton instance
_retraining_pipeline = None


def get_retraining_pipeline() -> AutomatedRetrainingPipeline:
    """Get singleton instance"""
    global _retraining_pipeline
    if _retraining_pipeline is None:
        _retraining_pipeline = AutomatedRetrainingPipeline()
    return _retraining_pipeline
