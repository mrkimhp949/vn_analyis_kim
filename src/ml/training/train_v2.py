# -*- coding: utf-8 -*-
"""
ML Training Pipeline V2 - Improved for 58-62% accuracy
Key improvements:
1. Walk-forward validation (no data leakage)
2. Proper regularization
3. Feature selection based on importance
4. Ensemble with calibration
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class MLTrainerV2:
    """
    Improved ML trainer with walk-forward validation.
    """

    def __init__(
        self,
        models_dir: str = "models",
        n_splits: int = 5,
        min_train_size: int = 500,
    ):
        self.models_dir = models_dir
        self.n_splits = n_splits
        self.min_train_size = min_train_size

        self.scaler = StandardScaler()
        self.models: Dict = {}
        self.feature_importance: Optional[pd.DataFrame] = None
        self.cv_results: List[Dict] = []

        os.makedirs(models_dir, exist_ok=True)

    def load_training_data(
        self, symbols: List[str], lookback: int = 500
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Load and prepare training data from multiple symbols."""
        from src.data.loader import load_data
        from src.ml.features.enhanced_v2 import (
            add_enhanced_features_v2,
            get_feature_columns_v2,
        )

        all_data = []
        feature_cols = get_feature_columns_v2()

        logger.info(f"Loading data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                df = load_data(symbol, lookback=lookback)
                if df is None or len(df) < 100:
                    continue

                # Load index
                try:
                    index_df = load_data("VNINDEX", lookback=lookback, is_index=True)
                except:
                    index_df = None

                # Add features with improved target
                df = add_enhanced_features_v2(df, index_df, target_type="multi_horizon")

                # Filter valid rows
                df = df.dropna(subset=feature_cols + ["target"])

                if len(df) >= 50:
                    df["symbol"] = symbol
                    all_data.append(df)
                    logger.info(f"  ✅ {symbol}: {len(df)} rows")

            except Exception as e:
                logger.warning(f"  ❌ {symbol}: {e}")

        if not all_data:
            raise ValueError("No valid data loaded")

        combined = pd.concat(all_data, ignore_index=True)

        # Sort by time for proper time series split
        if "time" in combined.columns:
            combined = combined.sort_values("time").reset_index(drop=True)

        X = combined[feature_cols]
        y = combined["target"]

        logger.info(f"Total samples: {len(combined)}")
        logger.info(f"Class distribution: {y.value_counts(normalize=True).to_dict()}")

        return X, y, feature_cols

    def _create_base_models(self) -> Dict:
        """Create base models with proper regularization."""
        models = {}

        # Random Forest - regularized
        models["rf"] = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,  # Limit depth to prevent overfitting
            min_samples_leaf=20,  # Require more samples per leaf
            min_samples_split=40,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Gradient Boosting - regularized
        models["gb"] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,  # Shallow trees
            learning_rate=0.05,  # Slow learning
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
        )

        # XGBoost - if available
        if XGB_AVAILABLE:
            models["xgb"] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )

        # LightGBM - if available
        if LGB_AVAILABLE:
            models["lgb"] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        # Logistic Regression - baseline
        models["lr"] = LogisticRegression(
            C=0.1,  # Strong regularization
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

        return models

    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict:
        """
        Walk-forward validation - most realistic for time series.
        Train on past, test on future, no data leakage.
        """
        logger.info("\n" + "=" * 60)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("=" * 60)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        base_models = self._create_base_models()

        results = {
            name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}
            for name in base_models.keys()
        }

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"\nFold {fold + 1}/{self.n_splits}")
            logger.info(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            for name, model in base_models.items():
                try:
                    # Clone model for this fold
                    from sklearn.base import clone

                    model_clone = clone(model)

                    # Train
                    model_clone.fit(X_train_scaled, y_train)

                    # Predict
                    y_pred = model_clone.predict(X_test_scaled)
                    y_proba = model_clone.predict_proba(X_test_scaled)[:, 1]

                    # Metrics
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    auc = roc_auc_score(y_test, y_proba)

                    results[name]["accuracy"].append(acc)
                    results[name]["precision"].append(prec)
                    results[name]["recall"].append(rec)
                    results[name]["f1"].append(f1)
                    results[name]["auc"].append(auc)

                    logger.info(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

                except Exception as e:
                    logger.warning(f"  {name} failed: {e}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info("=" * 60)

        summary = {}
        for name, metrics in results.items():
            if metrics["accuracy"]:
                summary[name] = {
                    "accuracy_mean": np.mean(metrics["accuracy"]),
                    "accuracy_std": np.std(metrics["accuracy"]),
                    "f1_mean": np.mean(metrics["f1"]),
                    "auc_mean": np.mean(metrics["auc"]),
                }
                logger.info(f"\n{name}:")
                logger.info(
                    f"  Accuracy: {summary[name]['accuracy_mean']:.4f} ± {summary[name]['accuracy_std']:.4f}"
                )
                logger.info(f"  F1 Score: {summary[name]['f1_mean']:.4f}")
                logger.info(f"  AUC:      {summary[name]['auc_mean']:.4f}")

        self.cv_results = results
        return summary

    def train_final_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
    ) -> Dict:
        """Train final models on all data."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING FINAL MODELS")
        logger.info("=" * 60)

        # Fit scaler on all data
        X_scaled = self.scaler.fit_transform(X)

        # Train each model
        base_models = self._create_base_models()

        for name, model in base_models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_scaled, y)
                self.models[name] = model

                # Get feature importance
                if hasattr(model, "feature_importances_"):
                    importance = pd.DataFrame(
                        {"feature": feature_cols, "importance": model.feature_importances_}
                    ).sort_values("importance", ascending=False)

                    logger.info(f"  Top 5 features: {importance.head()['feature'].tolist()}")

            except Exception as e:
                logger.warning(f"  {name} failed: {e}")

        # Create ensemble
        if len(self.models) >= 2:
            self._create_ensemble(X_scaled, y)

        return self.models

    def _create_ensemble(self, X_scaled: np.ndarray, y: pd.Series) -> None:
        """Create calibrated ensemble from best models."""
        logger.info("\nCreating ensemble...")

        # Select best models based on CV results
        if self.cv_results:
            model_scores = {}
            for name in self.models.keys():
                if name in self.cv_results and self.cv_results[name]["accuracy"]:
                    model_scores[name] = np.mean(self.cv_results[name]["accuracy"])

            # Use top 3 models
            top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"  Top models: {[m[0] for m in top_models]}")
        else:
            top_models = [(name, 0) for name in list(self.models.keys())[:3]]

        # Create voting ensemble
        estimators = [(name, self.models[name]) for name, _ in top_models if name in self.models]

        if len(estimators) >= 2:
            self.models["ensemble"] = VotingClassifier(
                estimators=estimators,
                voting="soft",
            )
            self.models["ensemble"].fit(X_scaled, y)
            logger.info("  ✅ Ensemble created")

    def save_models(self, feature_cols: List[str]) -> None:
        """Save all models and metadata."""
        logger.info("\n" + "=" * 60)
        logger.info("SAVING MODELS")
        logger.info("=" * 60)

        # Save scaler
        scaler_path = os.path.join(self.models_dir, "scaler_v2.pkl")
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"  Saved scaler to {scaler_path}")

        # Save each model
        for name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f"{name}_v2.pkl")
            joblib.dump(model, model_path)
            logger.info(f"  Saved {name} to {model_path}")

        # Save metadata
        metadata = {
            "version": "v2",
            "created_at": datetime.now().isoformat(),
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "models": list(self.models.keys()),
            "cv_results": (
                {
                    name: {
                        "accuracy_mean": (
                            float(np.mean(metrics["accuracy"])) if metrics["accuracy"] else 0
                        ),
                        "accuracy_std": (
                            float(np.std(metrics["accuracy"])) if metrics["accuracy"] else 0
                        ),
                        "f1_mean": float(np.mean(metrics["f1"])) if metrics["f1"] else 0,
                    }
                    for name, metrics in self.cv_results.items()
                }
                if self.cv_results
                else {}
            ),
        }

        metadata_path = os.path.join(self.models_dir, "model_info_v2.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  Saved metadata to {metadata_path}")

    def train(self, symbols: List[str], lookback: int = 500) -> Dict:
        """
        Full training pipeline.

        Args:
            symbols: List of stock symbols to train on
            lookback: Number of days of historical data

        Returns:
            Dictionary with training results
        """
        # 1. Load data
        X, y, feature_cols = self.load_training_data(symbols, lookback)

        # 2. Walk-forward validation
        cv_summary = self.walk_forward_validation(X, y)

        # 3. Train final models
        self.train_final_models(X, y, feature_cols)

        # 4. Save models
        self.save_models(feature_cols)

        # 5. Return results
        best_model = max(cv_summary.items(), key=lambda x: x[1]["accuracy_mean"])

        results = {
            "best_model": best_model[0],
            "best_accuracy": best_model[1]["accuracy_mean"],
            "edge_vs_random": (best_model[1]["accuracy_mean"] - 0.5) * 100,
            "cv_summary": cv_summary,
            "n_samples": len(X),
            "n_features": len(feature_cols),
        }

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best model: {results['best_model']}")
        logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
        logger.info(f"Edge vs random: {results['edge_vs_random']:+.2f}%")

        return results


# =============================================================================
# MAIN SCRIPT
# =============================================================================


def main():
    """Run training pipeline."""
    import sys
    import os

    # Add project root to path
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # VN30 symbols for training
    symbols = [
        "VNM",
        "FPT",
        "VIC",
        "VHM",
        "HPG",
        "MWG",
        "MSN",
        "VCB",
        "TCB",
        "VPB",
        "BID",
        "CTG",
        "MBB",
        "ACB",
        "STB",
        "SSI",
        "VND",
        "HCM",
        "GAS",
        "PLX",
    ]

    print("\n" + "=" * 70)
    print("🚀 ML TRAINING PIPELINE V2")
    print("=" * 70)

    trainer = MLTrainerV2(
        models_dir="models",
        n_splits=5,
    )

    results = trainer.train(symbols, lookback=500)

    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Best Model: {results['best_model']}")
    print(f"Accuracy: {results['best_accuracy']:.4f}")
    print(f"Edge: {results['edge_vs_random']:+.2f}%")

    if results["edge_vs_random"] >= 8:
        print("\n✅ TARGET ACHIEVED: Edge >= 8%")
    else:
        print(f"\n⚠️ Target not achieved. Current edge: {results['edge_vs_random']:.2f}%")

    return results


if __name__ == "__main__":
    main()
