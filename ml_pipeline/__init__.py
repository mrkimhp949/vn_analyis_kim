# ML pipeline package initialization
"""
ML Pipeline for Vietnam Stock Trading Bot.

Modules:
- model_registry: Model versioning và management
- training_pipeline: Automated training và retraining
- feature_selection: SHAP-based feature selection
- stacking_ensemble: Meta-model stacking
- volatility_forecaster: Volatility prediction
- sentiment_model: News sentiment analysis
- data_manager: Data loading và preprocessing
"""

from ml_pipeline.model_registry import (
    ModelRegistry,
    ModelMetrics,
    ModelStage,
    ModelType,
    ModelVersion,
    get_registry,
)

from ml_pipeline.training_pipeline import (
    MLTrainingPipeline,
    TrainingConfig,
    TrainingResult,
    DataValidationResult,
    RetrainingScheduler,
)

from ml_pipeline.feature_selection import select_features_with_shap

from ml_pipeline.stacking_ensemble import StackingMetaModel

__all__ = [
    # Model Registry
    "ModelRegistry",
    "ModelMetrics",
    "ModelStage",
    "ModelType",
    "ModelVersion",
    "get_registry",
    # Training Pipeline
    "MLTrainingPipeline",
    "TrainingConfig",
    "TrainingResult",
    "DataValidationResult",
    "RetrainingScheduler",
    # Feature Selection
    "select_features_with_shap",
    # Ensemble
    "StackingMetaModel",
]
