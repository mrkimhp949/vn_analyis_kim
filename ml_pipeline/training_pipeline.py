"""
ML Training Pipeline - Quản lý training và retraining models.

Hỗ trợ:
- Automated training pipeline
- Data preprocessing và validation
- Cross-validation với time series split
- Hyperparameter tuning
- Model evaluation và comparison
- Scheduled retraining
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from ml_pipeline.model_registry import ModelMetrics, ModelRegistry, ModelStage, get_registry

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Các mức độ chất lượng data."""

    EXCELLENT = "excellent"  # < 1% missing, đủ samples
    GOOD = "good"  # 1-5% missing
    ACCEPTABLE = "acceptable"  # 5-10% missing
    POOR = "poor"  # > 10% missing hoặc không đủ samples


@dataclass
class DataValidationResult:
    """Kết quả validate data."""

    is_valid: bool
    quality: DataQuality
    total_samples: int
    missing_ratio: float
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Cấu hình training."""

    model_type: str
    feature_columns: List[str]
    target_column: str = "target"
    test_size: float = 0.2
    n_cv_splits: int = 5
    tune_hyperparameters: bool = False
    random_state: int = 42
    min_samples: int = 1000
    min_positive_ratio: float = 0.2
    max_positive_ratio: float = 0.8


@dataclass
class TrainingResult:
    """Kết quả training."""

    success: bool
    model_version: Optional[str] = None
    metrics: Optional[ModelMetrics] = None
    cv_scores: List[float] = field(default_factory=list)
    training_time_seconds: float = 0.0
    error_message: str = ""
    data_validation: Optional[DataValidationResult] = None


class MLTrainingPipeline:
    """
    Pipeline training ML models.

    Features:
    - Data validation và preprocessing
    - Model training với cross-validation
    - Automatic model registration
    - Hyperparameter tuning
    - Model comparison
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        output_dir: str = "models",
    ):
        """
        Khởi tạo training pipeline.

        Args:
            registry: Model registry instance
            output_dir: Thư mục output
        """
        self.registry = registry or get_registry(output_dir)
        self.output_dir = Path(output_dir)
        self.training_history: List[TrainingResult] = []

        # Model factories
        self._model_factories: Dict[str, Callable] = {}
        self._register_default_factories()

    def _register_default_factories(self):
        """Đăng ký các model factory mặc định."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression

            self._model_factories["random_forest"] = lambda: RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

            self._model_factories["gradient_boosting"] = lambda: GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )

            self._model_factories["logistic_regression"] = lambda: LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )
        except ImportError:
            logger.warning("sklearn không khả dụng")

        try:
            import xgboost as xgb

            self._model_factories["xgboost"] = lambda: xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=-1,
            )
        except ImportError:
            logger.info("XGBoost không khả dụng")

        try:
            import lightgbm as lgb

            self._model_factories["lightgbm"] = lambda: lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError:
            logger.info("LightGBM không khả dụng")

    def register_model_factory(self, model_type: str, factory: Callable):
        """Đăng ký model factory custom."""
        self._model_factories[model_type] = factory

    def validate_data(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
    ) -> DataValidationResult:
        """
        Validate data trước khi training.

        Args:
            df: DataFrame training data
            config: Training config

        Returns:
            DataValidationResult
        """
        issues = []
        recommendations = []

        # Check số lượng samples
        if len(df) < config.min_samples:
            issues.append(f"Không đủ samples: {len(df)} < {config.min_samples}")
            recommendations.append("Thu thập thêm dữ liệu hoặc giảm min_samples")

        # Check features có tồn tại
        missing_features = [col for col in config.feature_columns if col not in df.columns]
        if missing_features:
            issues.append(f"Thiếu features: {missing_features}")
            recommendations.append("Kiểm tra lại feature_columns trong config")

        # Check target column
        if config.target_column not in df.columns:
            issues.append(f"Không có target column: {config.target_column}")
            return DataValidationResult(
                is_valid=False,
                quality=DataQuality.POOR,
                total_samples=len(df),
                missing_ratio=1.0,
                issues=issues,
                recommendations=recommendations,
            )

        # Check missing values
        available_features = [col for col in config.feature_columns if col in df.columns]
        missing_ratio = df[available_features].isna().sum().sum() / (
            len(df) * len(available_features)
        )

        if missing_ratio > 0.1:
            issues.append(f"Tỷ lệ missing cao: {missing_ratio:.1%}")
            recommendations.append("Xử lý missing values trước khi training")

        # Check class balance
        target = df[config.target_column]
        positive_ratio = target.mean()

        if positive_ratio < config.min_positive_ratio:
            issues.append(f"Class imbalance - positive ratio quá thấp: {positive_ratio:.1%}")
            recommendations.append("Sử dụng oversampling hoặc class weights")
        elif positive_ratio > config.max_positive_ratio:
            issues.append(f"Class imbalance - positive ratio quá cao: {positive_ratio:.1%}")
            recommendations.append("Kiểm tra lại cách tạo labels")

        # Feature statistics
        feature_stats = {}
        for col in available_features[:20]:  # Top 20 features
            feature_stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "missing": float(df[col].isna().mean()),
            }

        # Xác định quality
        if len(issues) == 0:
            quality = DataQuality.EXCELLENT
        elif missing_ratio < 0.05 and len(df) >= config.min_samples:
            quality = DataQuality.GOOD
        elif missing_ratio < 0.1:
            quality = DataQuality.ACCEPTABLE
        else:
            quality = DataQuality.POOR

        is_valid = quality != DataQuality.POOR and not missing_features

        return DataValidationResult(
            is_valid=is_valid,
            quality=quality,
            total_samples=len(df),
            missing_ratio=missing_ratio,
            feature_stats=feature_stats,
            issues=issues,
            recommendations=recommendations,
        )

    def preprocess_data(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        fit_scaler: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Preprocess data cho training.

        Args:
            df: DataFrame
            config: Training config
            fit_scaler: Có fit scaler mới không

        Returns:
            (X, y, scaler)
        """
        from sklearn.preprocessing import StandardScaler

        # Lọc features có trong data
        available_features = [col for col in config.feature_columns if col in df.columns]

        X = df[available_features].copy()
        y = df[config.target_column].values

        # Fill missing values
        X = X.fillna(X.median())

        # Replace inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        # Scale features
        scaler = StandardScaler()
        if fit_scaler:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)

        return X_scaled, y, scaler

    def train_model(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        auto_register: bool = True,
        version_bump: str = "patch",
        description: str = "",
    ) -> TrainingResult:
        """
        Train model với data.

        Args:
            df: Training data
            config: Training config
            auto_register: Tự động đăng ký vào registry
            version_bump: "major", "minor", "patch"
            description: Mô tả

        Returns:
            TrainingResult
        """
        import time

        start_time = time.time()

        # Validate data
        validation = self.validate_data(df, config)
        if not validation.is_valid:
            return TrainingResult(
                success=False,
                error_message=f"Data validation failed: {validation.issues}",
                data_validation=validation,
            )

        # Check model factory
        if config.model_type not in self._model_factories:
            return TrainingResult(
                success=False,
                error_message=f"Unknown model type: {config.model_type}",
            )

        try:
            # Preprocess
            X, y, scaler = self.preprocess_data(df, config)

            # Create model
            model = self._model_factories[config.model_type]()

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=config.n_cv_splits)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="roc_auc")

            # Train on full data
            model.fit(X, y)

            # Predictions
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]

            # Calculate metrics
            metrics = ModelMetrics(
                accuracy=float(accuracy_score(y, y_pred)),
                precision=float(precision_score(y, y_pred, zero_division=0)),
                recall=float(recall_score(y, y_pred, zero_division=0)),
                f1_score=float(f1_score(y, y_pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, y_proba)),
            )

            # Training time
            training_time = time.time() - start_time

            # Register model
            version = None
            if auto_register:
                # Register scaler first
                scaler_version = self.registry.register_model(
                    model=scaler,
                    model_type=f"{config.model_type}_scaler",
                    version_bump=version_bump,
                    description=f"Scaler cho {config.model_type}",
                )

                # Register model
                version = self.registry.register_model(
                    model=model,
                    model_type=config.model_type,
                    metrics=metrics,
                    version_bump=version_bump,
                    description=description,
                    feature_columns=config.feature_columns,
                    hyperparameters={
                        "n_cv_splits": config.n_cv_splits,
                        "test_size": config.test_size,
                    },
                    training_data_info={
                        "samples": len(df),
                        "features": len(config.feature_columns),
                        "positive_ratio": float(y.mean()),
                        "scaler_version": scaler_version,
                    },
                )

            result = TrainingResult(
                success=True,
                model_version=version,
                metrics=metrics,
                cv_scores=cv_scores.tolist(),
                training_time_seconds=training_time,
                data_validation=validation,
            )

            self.training_history.append(result)
            logger.info(
                f"Training {config.model_type} thành công! "
                f"ROC-AUC: {metrics.roc_auc:.4f}, CV mean: {cv_scores.mean():.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time_seconds=time.time() - start_time,
            )

    def train_ensemble(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        model_types: Optional[List[str]] = None,
        description: str = "",
    ) -> Dict[str, TrainingResult]:
        """
        Train ensemble của nhiều models.

        Args:
            df: Training data
            config: Base training config
            model_types: Danh sách model types
            description: Mô tả

        Returns:
            Dict {model_type: TrainingResult}
        """
        if model_types is None:
            model_types = ["random_forest", "xgboost", "lightgbm"]

        results = {}
        for model_type in model_types:
            if model_type not in self._model_factories:
                logger.warning(f"Model type {model_type} không khả dụng, skip")
                continue

            model_config = TrainingConfig(
                model_type=model_type,
                feature_columns=config.feature_columns,
                target_column=config.target_column,
                test_size=config.test_size,
                n_cv_splits=config.n_cv_splits,
            )

            logger.info(f"Training {model_type}...")
            results[model_type] = self.train_model(
                df=df,
                config=model_config,
                description=f"{description} - {model_type}",
            )

        return results

    def retrain_production_model(
        self,
        df: pd.DataFrame,
        model_type: str,
        promote_if_better: bool = True,
        min_improvement: float = 0.01,
    ) -> TrainingResult:
        """
        Retrain model đang production.

        Args:
            df: Training data mới
            model_type: Loại model
            promote_if_better: Tự động promote nếu model mới tốt hơn
            min_improvement: Mức cải thiện tối thiểu để promote

        Returns:
            TrainingResult
        """
        # Lấy model production hiện tại
        current_production = self.registry.get_production_models()
        current_metrics = None

        if model_type in current_production:
            current_mv = current_production[model_type]
            current_metrics = current_mv.metrics
            feature_columns = current_mv.feature_columns
        else:
            logger.warning(f"Không có production model cho {model_type}")
            return TrainingResult(
                success=False,
                error_message="No production model found",
            )

        # Train model mới
        config = TrainingConfig(
            model_type=model_type,
            feature_columns=feature_columns,
        )

        result = self.train_model(
            df=df,
            config=config,
            auto_register=True,
            version_bump="minor",
            description=f"Retrained {model_type}",
        )

        if not result.success:
            return result

        # So sánh với model cũ
        if promote_if_better and current_metrics and result.metrics:
            improvement = result.metrics.roc_auc - current_metrics.roc_auc

            if improvement >= min_improvement:
                self.registry.promote_model(
                    model_type=model_type,
                    version=result.model_version,
                    to_stage=ModelStage.PRODUCTION,
                )
                logger.info(
                    f"Đã promote {model_type} {result.model_version} "
                    f"(improvement: +{improvement:.4f})"
                )
            else:
                logger.info(
                    f"Model mới không tốt hơn đủ "
                    f"(improvement: {improvement:.4f} < {min_improvement})"
                )

        return result

    def evaluate_model_on_new_data(
        self,
        df: pd.DataFrame,
        model_type: str,
        version: Optional[str] = None,
    ) -> Optional[ModelMetrics]:
        """
        Evaluate model trên data mới (out-of-sample).

        Args:
            df: New data
            model_type: Loại model
            version: Version cụ thể (None = production)

        Returns:
            ModelMetrics hoặc None nếu lỗi
        """
        try:
            # Load model
            model, model_version = self.registry.load_model(model_type, version)

            # Load scaler
            scaler_type = f"{model_type}_scaler"
            scaler, _ = self.registry.load_model(scaler_type, version)

            # Prepare data
            available_features = [col for col in model_version.feature_columns if col in df.columns]
            X = df[available_features].fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            X_scaled = scaler.transform(X)

            y = df["target"].values if "target" in df.columns else None

            if y is None:
                logger.warning("Không có target column để evaluate")
                return None

            # Predict
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)[:, 1]

            # Calculate metrics
            metrics = ModelMetrics(
                accuracy=float(accuracy_score(y, y_pred)),
                precision=float(precision_score(y, y_pred, zero_division=0)),
                recall=float(recall_score(y, y_pred, zero_division=0)),
                f1_score=float(f1_score(y, y_pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, y_proba)),
            )

            logger.info(
                f"Evaluation {model_type} {version or 'production'}: "
                f"ROC-AUC={metrics.roc_auc:.4f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None

    def save_training_report(self, output_path: str = "training_report.json"):
        """Lưu báo cáo training."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_trainings": len(self.training_history),
            "successful": sum(1 for r in self.training_history if r.success),
            "failed": sum(1 for r in self.training_history if not r.success),
            "trainings": [
                {
                    "success": r.success,
                    "version": r.model_version,
                    "metrics": r.metrics.to_dict() if r.metrics else None,
                    "cv_scores": r.cv_scores,
                    "training_time": r.training_time_seconds,
                    "error": r.error_message,
                }
                for r in self.training_history
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Đã lưu training report vào {output_path}")


class RetrainingScheduler:
    """
    Scheduler để tự động retrain models.
    """

    def __init__(self, pipeline: MLTrainingPipeline):
        self.pipeline = pipeline
        self.schedule: Dict[str, Dict] = {}

    def add_schedule(
        self,
        model_type: str,
        interval_days: int = 7,
        min_samples_for_retrain: int = 1000,
    ):
        """Thêm schedule retrain cho model."""
        self.schedule[model_type] = {
            "interval_days": interval_days,
            "min_samples": min_samples_for_retrain,
            "last_retrain": None,
        }

    def check_and_retrain(
        self,
        model_type: str,
        df: pd.DataFrame,
    ) -> Optional[TrainingResult]:
        """Check và retrain nếu cần."""
        if model_type not in self.schedule:
            return None

        schedule = self.schedule[model_type]

        # Check if enough time has passed
        if schedule["last_retrain"]:
            days_since = (datetime.now() - schedule["last_retrain"]).days
            if days_since < schedule["interval_days"]:
                logger.info(
                    f"Skip retrain {model_type}: "
                    f"chỉ mới {days_since}/{schedule['interval_days']} ngày"
                )
                return None

        # Check if enough samples
        if len(df) < schedule["min_samples"]:
            logger.info(
                f"Skip retrain {model_type}: "
                f"không đủ samples ({len(df)}/{schedule['min_samples']})"
            )
            return None

        # Retrain
        result = self.pipeline.retrain_production_model(df, model_type)

        if result.success:
            schedule["last_retrain"] = datetime.now()

        return result
