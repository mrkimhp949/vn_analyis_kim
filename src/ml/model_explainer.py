# -*- coding: utf-8 -*-
"""
ML Model Explainability using SHAP
Giải thích predictions của ML model cho transparency và debugging
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# SHAP is optional dependency
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed - explainability features disabled. "
                   "Install with: pip install shap")


@dataclass
class PredictionExplanation:
    """Giải thích cho một prediction"""

    prediction: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-100
    base_value: float  # Model baseline prediction

    # Feature contributions (SHAP values)
    feature_contributions: Dict[str, float]  # {feature_name: shap_value}
    top_positive_features: List[tuple]  # [(feature, value), ...]
    top_negative_features: List[tuple]  # [(feature, value), ...]

    # Summary
    total_positive_contribution: float
    total_negative_contribution: float
    net_contribution: float  # Should equal (prediction - base_value)


class ModelExplainer:
    """
    Giải thích ML predictions using SHAP (SHapley Additive exPlanations)

    FEATURES:
    - Feature importance analysis
    - Individual prediction explanations
    - Waterfall plots (programmatic)
    - Force plots (programmatic)

    NOTE: SHAP requires model-specific explainers:
    - TreeExplainer: XGBoost, LightGBM, CatBoost, RandomForest
    - LinearExplainer: Linear models
    - DeepExplainer: Neural networks
    """

    def __init__(
        self,
        model: Any,
        model_type: str,  # "xgboost", "lightgbm", "random_forest"
        feature_names: List[str],
        background_samples: Optional[pd.DataFrame] = None,
        background_sample_size: int = 100,
    ):
        """
        Args:
            model: Trained model object
            model_type: Type of model for selecting SHAP explainer
            feature_names: List of feature names
            background_samples: Background dataset for SHAP (optional)
            background_sample_size: Number of background samples to use
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - explainer will not work")
            self.explainer = None
            return

        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names

        # Initialize SHAP explainer
        try:
            if model_type in ["xgboost", "lightgbm", "random_forest", "catboost"]:
                # Tree-based models
                self.explainer = shap.TreeExplainer(model)
                logger.info(f"✅ TreeExplainer initialized for {model_type}")

            elif model_type == "linear":
                # Linear models
                self.explainer = shap.LinearExplainer(model, background_samples)
                logger.info("✅ LinearExplainer initialized")

            else:
                # Fallback to KernelExplainer (model-agnostic but slow)
                if background_samples is None or background_samples.empty:
                    logger.warning("Background samples needed for KernelExplainer")
                    self.explainer = None
                else:
                    # Sample background data
                    bg_data = shap.sample(background_samples, background_sample_size)
                    self.explainer = shap.KernelExplainer(model.predict, bg_data)
                    logger.info("✅ KernelExplainer initialized (slow but model-agnostic)")

        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            self.explainer = None

    def explain_prediction(
        self,
        features: pd.Series,
        prediction: str,
        confidence: float,
        top_n: int = 5
    ) -> PredictionExplanation:
        """
        Explain a single prediction

        Args:
            features: Feature values for this prediction
            prediction: Model prediction ("BUY", "SELL", "HOLD")
            confidence: Model confidence (0-100)
            top_n: Number of top features to show

        Returns:
            PredictionExplanation object
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._fallback_explanation(prediction, confidence)

        try:
            # Convert to numpy array
            feature_values = features[self.feature_names].values.reshape(1, -1)

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(feature_values)

            # Handle different output formats
            # Some models return list [class0_shap, class1_shap, ...]
            if isinstance(shap_values, list):
                # For multiclass, take the predicted class
                # Assume BUY=1, SELL=2, HOLD=0 (adjust based on your encoding)
                class_idx = {"HOLD": 0, "BUY": 1, "SELL": 2}.get(prediction, 0)
                shap_values = shap_values[class_idx]

            # Get SHAP values for this prediction
            shap_values_single = shap_values[0] if shap_values.ndim > 1 else shap_values

            # Base value (expected value of model output)
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                class_idx = {"HOLD": 0, "BUY": 1, "SELL": 2}.get(prediction, 0)
                base_value = base_value[class_idx]

            # Feature contributions
            contributions = {
                self.feature_names[i]: float(shap_values_single[i])
                for i in range(len(self.feature_names))
            }

            # Sort by absolute value
            sorted_contributions = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Top positive and negative
            positive = [(f, v) for f, v in sorted_contributions if v > 0][:top_n]
            negative = [(f, v) for f, v in sorted_contributions if v < 0][:top_n]

            # Sums
            total_positive = sum(v for _, v in positive)
            total_negative = sum(v for _, v in negative)
            net = total_positive + total_negative

            return PredictionExplanation(
                prediction=prediction,
                confidence=confidence,
                base_value=float(base_value),
                feature_contributions=contributions,
                top_positive_features=positive,
                top_negative_features=negative,
                total_positive_contribution=total_positive,
                total_negative_contribution=total_negative,
                net_contribution=net,
            )

        except Exception as e:
            logger.error(f"Error explaining prediction: {e}", exc_info=True)
            return self._fallback_explanation(prediction, confidence)

    def get_feature_importance(
        self,
        test_features: pd.DataFrame,
        sample_size: int = 1000
    ) -> pd.DataFrame:
        """
        Calculate global feature importance using SHAP

        Args:
            test_features: Test dataset
            sample_size: Number of samples to use (SHAP can be slow)

        Returns:
            DataFrame with feature importance
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            logger.warning("SHAP not available - cannot calculate feature importance")
            return pd.DataFrame()

        try:
            # Sample data
            if len(test_features) > sample_size:
                test_sample = test_features.sample(n=sample_size, random_state=42)
            else:
                test_sample = test_features

            # Extract features
            X = test_sample[self.feature_names].values

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)

            # Handle multiclass output
            if isinstance(shap_values, list):
                # Average across classes
                shap_values = np.mean(np.abs(shap_values), axis=0)

            # Mean absolute SHAP value per feature
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_abs_shap
            })

            importance_df = importance_df.sort_values('importance', ascending=False)

            return importance_df

        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}", exc_info=True)
            return pd.DataFrame()

    def format_explanation(self, explanation: PredictionExplanation) -> str:
        """
        Format explanation as human-readable string

        Returns:
            Formatted explanation text
        """
        lines = []

        lines.append(f"📊 **PREDICTION EXPLANATION**")
        lines.append(f"   Prediction: {explanation.prediction}")
        lines.append(f"   Confidence: {explanation.confidence:.1f}%")
        lines.append(f"   Base value: {explanation.base_value:.3f}")
        lines.append(f"   Net contribution: {explanation.net_contribution:+.3f}\n")

        # Top positive features
        if explanation.top_positive_features:
            lines.append(f"✅ **TOP POSITIVE CONTRIBUTORS:**")
            for feature, value in explanation.top_positive_features:
                lines.append(f"   • {feature}: {value:+.3f}")
            lines.append("")

        # Top negative features
        if explanation.top_negative_features:
            lines.append(f"⛔ **TOP NEGATIVE CONTRIBUTORS:**")
            for feature, value in explanation.top_negative_features:
                lines.append(f"   • {feature}: {value:+.3f}")
            lines.append("")

        lines.append(f"📈 Total positive: {explanation.total_positive_contribution:+.3f}")
        lines.append(f"📉 Total negative: {explanation.total_negative_contribution:+.3f}")

        return "\n".join(lines)

    def _fallback_explanation(
        self,
        prediction: str,
        confidence: float
    ) -> PredictionExplanation:
        """Return minimal explanation when SHAP unavailable"""
        return PredictionExplanation(
            prediction=prediction,
            confidence=confidence,
            base_value=0.0,
            feature_contributions={},
            top_positive_features=[],
            top_negative_features=[],
            total_positive_contribution=0.0,
            total_negative_contribution=0.0,
            net_contribution=0.0,
        )
