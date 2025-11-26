# -*- coding: utf-8 -*-
"""
ML Model Version Manager
Quản lý versioning, tracking, và deployment cho ML models
"""

import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata cho ML model version"""

    model_id: str  # Unique ID (hash của model + config)
    version: str  # Semantic version (e.g., "1.2.3")
    created_at: str  # ISO timestamp
    model_type: str  # "xgboost", "lightgbm", "random_forest", "ensemble"

    # Performance metrics
    train_accuracy: float
    val_accuracy: float
    train_auc: float
    val_auc: float

    # Feature info
    num_features: int
    feature_names: List[str]
    feature_importance: Optional[Dict[str, float]] = None

    # Training info
    training_samples: int
    training_period: str  # "2023-01-01 to 2024-01-01"
    hyperparameters: Optional[Dict] = None

    # Deployment info
    is_active: bool = False  # Currently deployed
    deployment_date: Optional[str] = None

    # Tags and notes
    tags: List[str] = None
    notes: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ModelPerformanceLog:
    """Log hiệu suất của model trong production"""

    model_id: str
    timestamp: str

    # Signal performance
    total_signals: int
    buy_signals: int
    accuracy: float  # Actual win rate
    avg_confidence: float

    # Trade performance (if available)
    total_trades: int = 0
    win_rate: float = 0.0
    avg_profit_pct: float = 0.0
    sharpe_ratio: float = 0.0

    # Drift metrics
    prediction_drift: float = 0.0  # Distribution shift in predictions
    feature_drift: float = 0.0  # Distribution shift in features


class ModelVersionManager:
    """
    Quản lý versioning và deployment cho ML models

    FEATURES:
    - Model versioning with metadata
    - Performance tracking
    - Model comparison
    - A/B testing support
    - Automated model selection
    - Rollback capability
    """

    def __init__(
        self,
        models_dir: str = "models",
        metadata_file: str = "models/model_registry.json",
        performance_log_file: str = "models/performance_log.jsonl",
        auto_promote_threshold: float = 0.05,  # Auto-promote if 5% better
    ):
        """
        Args:
            models_dir: Directory to store model files
            metadata_file: JSON file for model metadata registry
            performance_log_file: JSONL file for performance logs
            auto_promote_threshold: Threshold for auto-promotion (e.g., 0.05 = 5% better)
        """
        self.models_dir = Path(models_dir)
        self.metadata_file = Path(metadata_file)
        self.performance_log_file = Path(performance_log_file)
        self.auto_promote_threshold = auto_promote_threshold

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # Load registry
        self.registry = self._load_registry()

    def register_model(
        self,
        model: Any,
        model_type: str,
        version: str,
        train_metrics: Dict,
        val_metrics: Dict,
        feature_names: List[str],
        training_period: str,
        hyperparameters: Optional[Dict] = None,
        feature_importance: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        auto_activate: bool = False,
    ) -> str:
        """
        Register một model mới vào registry

        Args:
            model: Trained model object (XGBoost, LightGBM, etc.)
            model_type: Type of model
            version: Semantic version
            train_metrics: Training metrics (accuracy, AUC, etc.)
            val_metrics: Validation metrics
            feature_names: List of feature names
            training_period: Period used for training
            hyperparameters: Model hyperparameters
            feature_importance: Feature importance dict
            tags: Tags for categorization
            notes: Additional notes
            auto_activate: Automatically activate if better than current

        Returns:
            model_id: Unique ID for the registered model
        """
        # Generate unique model ID
        model_id = self._generate_model_id(model, version, hyperparameters)

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            created_at=datetime.now().isoformat(),
            model_type=model_type,
            train_accuracy=train_metrics.get('accuracy', 0.0),
            val_accuracy=val_metrics.get('accuracy', 0.0),
            train_auc=train_metrics.get('auc', 0.0),
            val_auc=val_metrics.get('auc', 0.0),
            num_features=len(feature_names),
            feature_names=feature_names,
            feature_importance=feature_importance,
            training_samples=train_metrics.get('samples', 0),
            training_period=training_period,
            hyperparameters=hyperparameters,
            is_active=False,
            tags=tags or [],
            notes=notes,
        )

        # Save model file
        model_path = self.models_dir / f"{model_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"✅ Model saved: {model_path}")

        # Add to registry
        self.registry[model_id] = asdict(metadata)
        self._save_registry()

        logger.info(
            f"📝 Model registered: {model_id} (v{version})\n"
            f"   Train: acc={metadata.train_accuracy:.3f}, auc={metadata.train_auc:.3f}\n"
            f"   Val: acc={metadata.val_accuracy:.3f}, auc={metadata.val_auc:.3f}"
        )

        # Auto-activate if requested
        if auto_activate:
            current_active = self.get_active_model_id()
            if current_active is None:
                # No active model - activate this one
                self.activate_model(model_id)
            else:
                # Check if better than current
                if self._should_auto_promote(model_id, current_active):
                    logger.info(f"🚀 Auto-promoting {model_id} (better than {current_active})")
                    self.activate_model(model_id)

        return model_id

    def activate_model(self, model_id: str):
        """
        Activate một model (set as production model)

        Deactivates current active model and activates new one
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")

        # Deactivate all models
        for mid in self.registry:
            self.registry[mid]['is_active'] = False
            self.registry[mid]['deployment_date'] = None

        # Activate target model
        self.registry[model_id]['is_active'] = True
        self.registry[model_id]['deployment_date'] = datetime.now().isoformat()

        self._save_registry()

        logger.info(f"✅ Model activated: {model_id} (v{self.registry[model_id]['version']})")

    def load_model(self, model_id: Optional[str] = None) -> Any:
        """
        Load model from disk

        Args:
            model_id: Model ID to load (if None, load active model)

        Returns:
            Loaded model object
        """
        if model_id is None:
            model_id = self.get_active_model_id()
            if model_id is None:
                raise ValueError("No active model found")

        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")

        model_path = self.models_dir / f"{model_id}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"📦 Model loaded: {model_id}")
        return model

    def get_active_model_id(self) -> Optional[str]:
        """Get currently active model ID"""
        for model_id, metadata in self.registry.items():
            if metadata.get('is_active', False):
                return model_id
        return None

    def log_performance(
        self,
        model_id: str,
        total_signals: int,
        buy_signals: int,
        accuracy: float,
        avg_confidence: float,
        total_trades: int = 0,
        win_rate: float = 0.0,
        avg_profit_pct: float = 0.0,
        sharpe_ratio: float = 0.0,
        prediction_drift: float = 0.0,
        feature_drift: float = 0.0,
    ):
        """
        Log production performance của model

        Args:
            model_id: Model ID
            total_signals: Total signals generated
            buy_signals: Number of BUY signals
            accuracy: Signal accuracy (win rate if trades available)
            avg_confidence: Average confidence score
            total_trades: Number of actual trades
            win_rate: Trade win rate
            avg_profit_pct: Average profit %
            sharpe_ratio: Sharpe ratio
            prediction_drift: Prediction distribution drift
            feature_drift: Feature distribution drift
        """
        log_entry = ModelPerformanceLog(
            model_id=model_id,
            timestamp=datetime.now().isoformat(),
            total_signals=total_signals,
            buy_signals=buy_signals,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_profit_pct=avg_profit_pct,
            sharpe_ratio=sharpe_ratio,
            prediction_drift=prediction_drift,
            feature_drift=feature_drift,
        )

        # Append to log file (JSONL format)
        with open(self.performance_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(log_entry)) + '\n')

        logger.debug(f"📊 Performance logged for {model_id}")

    def get_performance_history(
        self,
        model_id: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get performance history for model(s)

        Args:
            model_id: Specific model ID (if None, get all models)
            days: Number of days to look back

        Returns:
            DataFrame with performance history
        """
        if not self.performance_log_file.exists():
            return pd.DataFrame()

        # Read JSONL file
        logs = []
        with open(self.performance_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                log = json.loads(line.strip())

                # Filter by model_id if specified
                if model_id is not None and log.get('model_id') != model_id:
                    continue

                logs.append(log)

        if not logs:
            return pd.DataFrame()

        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter by days
        cutoff = datetime.now() - pd.Timedelta(days=days)
        df = df[df['timestamp'] >= cutoff]

        return df.sort_values('timestamp')

    def compare_models(
        self,
        model_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models

        Args:
            model_ids: List of model IDs to compare (if None, compare all)

        Returns:
            DataFrame with comparison
        """
        if model_ids is None:
            model_ids = list(self.registry.keys())

        comparisons = []
        for model_id in model_ids:
            if model_id not in self.registry:
                continue

            metadata = self.registry[model_id]

            # Get recent performance
            perf_df = self.get_performance_history(model_id, days=7)
            recent_accuracy = perf_df['accuracy'].mean() if not perf_df.empty else 0.0
            recent_win_rate = perf_df['win_rate'].mean() if not perf_df.empty else 0.0

            comparisons.append({
                'model_id': model_id,
                'version': metadata['version'],
                'model_type': metadata['model_type'],
                'is_active': metadata['is_active'],
                'val_accuracy': metadata['val_accuracy'],
                'val_auc': metadata['val_auc'],
                'recent_accuracy_7d': recent_accuracy,
                'recent_win_rate_7d': recent_win_rate,
                'num_features': metadata['num_features'],
                'created_at': metadata['created_at'],
            })

        return pd.DataFrame(comparisons).sort_values('val_accuracy', ascending=False)

    def _generate_model_id(
        self,
        model: Any,
        version: str,
        hyperparameters: Optional[Dict]
    ) -> str:
        """Generate unique model ID from model + config"""
        # Create hash from version + hyperparameters
        hash_input = f"{version}_{json.dumps(hyperparameters or {}, sort_keys=True)}"
        hash_digest = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        # Model ID format: model_type_version_hash
        model_id = f"model_v{version.replace('.', '_')}_{hash_digest}"

        return model_id

    def _should_auto_promote(
        self,
        new_model_id: str,
        current_model_id: str
    ) -> bool:
        """
        Check if new model should be auto-promoted

        Logic:
        - Val accuracy improvement >= threshold
        - No degradation in AUC
        """
        new_meta = self.registry[new_model_id]
        current_meta = self.registry[current_model_id]

        # Compare validation metrics
        acc_improvement = new_meta['val_accuracy'] - current_meta['val_accuracy']
        auc_improvement = new_meta['val_auc'] - current_meta['val_auc']

        # Auto-promote if accuracy improved by threshold AND AUC not degraded
        should_promote = (
            acc_improvement >= self.auto_promote_threshold and
            auc_improvement >= -0.01  # Allow tiny AUC degradation
        )

        if should_promote:
            logger.info(
                f"📈 New model is better: "
                f"acc_improvement={acc_improvement:+.3f}, "
                f"auc_improvement={auc_improvement:+.3f}"
            )

        return should_promote

    def _load_registry(self) -> Dict:
        """Load model registry from file"""
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            return {}

    def _save_registry(self):
        """Save model registry to file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")


# Singleton instance
_model_manager = None


def get_model_version_manager() -> ModelVersionManager:
    """Get singleton instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelVersionManager()
    return _model_manager
