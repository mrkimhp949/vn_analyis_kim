# -*- coding: utf-8 -*-
"""
Feature Drift Detector
Phát hiện drift trong feature distributions để cảnh báo model degradation
"""

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Metrics for feature drift detection"""

    feature_name: str
    timestamp: str

    # Distribution metrics
    mean_baseline: float
    mean_current: float
    std_baseline: float
    std_current: float

    # Drift scores
    psi_score: float  # Population Stability Index
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    ks_pvalue: float  # KS p-value

    # Drift status
    has_drift: bool
    drift_severity: str  # "none", "low", "medium", "high", "critical"


@dataclass
class DriftReport:
    """Comprehensive drift report"""

    timestamp: str
    total_features: int
    features_with_drift: int
    overall_drift_score: float  # 0-1 (average PSI across features)

    # Individual feature drifts
    feature_drifts: List[DriftMetrics]

    # Summary
    drift_severity: str  # Overall severity
    requires_retraining: bool
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class FeatureDriftDetector:
    """
    Detect feature distribution drift using statistical tests

    METHODS:
    1. PSI (Population Stability Index) - Standard in credit risk
    2. KS Test (Kolmogorov-Smirnov) - Distribution comparison
    3. Mean/Std shift detection

    THRESHOLDS (industry standard):
    - PSI < 0.1: No drift
    - PSI 0.1-0.2: Low drift (monitor)
    - PSI 0.2-0.25: Medium drift (investigate)
    - PSI > 0.25: High drift (retrain recommended)
    - PSI > 0.5: Critical drift (model may be broken)
    """

    def __init__(
        self,
        baseline_file: str = "models/feature_baseline.json",
        drift_log_file: str = "models/drift_log.jsonl",
        window_size: int = 1000,  # Number of recent samples for current distribution
        psi_threshold_low: float = 0.10,
        psi_threshold_medium: float = 0.20,
        psi_threshold_high: float = 0.25,
        psi_threshold_critical: float = 0.50,
        ks_pvalue_threshold: float = 0.05,  # KS test significance level
    ):
        """
        Args:
            baseline_file: JSON file to store baseline distributions
            drift_log_file: JSONL file to log drift detections
            window_size: Number of recent samples for drift detection
            psi_threshold_*: PSI thresholds for drift severity
            ks_pvalue_threshold: Significance level for KS test
        """
        self.baseline_file = Path(baseline_file)
        self.drift_log_file = Path(drift_log_file)
        self.window_size = window_size
        self.psi_threshold_low = psi_threshold_low
        self.psi_threshold_medium = psi_threshold_medium
        self.psi_threshold_high = psi_threshold_high
        self.psi_threshold_critical = psi_threshold_critical
        self.ks_pvalue_threshold = ks_pvalue_threshold

        # Create directories
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)

        # Load baseline
        self.baseline = self._load_baseline()

        # Recent samples buffer (for online drift detection)
        self.recent_samples = deque(maxlen=window_size)

    def set_baseline(self, features_df: pd.DataFrame, feature_names: List[str]):
        """
        Set baseline distribution từ training data

        Args:
            features_df: DataFrame with features
            feature_names: List of feature names to track
        """
        baseline = {}

        for feature in feature_names:
            if feature not in features_df.columns:
                logger.warning(f"Feature {feature} not found in training data")
                continue

            values = features_df[feature].dropna().values

            if len(values) == 0:
                logger.warning(f"Feature {feature} has no valid values")
                continue

            # Calculate baseline statistics
            baseline[feature] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q25": float(np.percentile(values, 25)),
                "q50": float(np.percentile(values, 50)),
                "q75": float(np.percentile(values, 75)),
                # Store histogram for PSI calculation
                "hist_bins": np.percentile(values, np.linspace(0, 100, 11)).tolist(),
                "hist_counts": np.histogram(
                    values, bins=np.percentile(values, np.linspace(0, 100, 11))
                )[0].tolist(),
                "sample_size": len(values),
                "created_at": datetime.now().isoformat(),
            }

        self.baseline = baseline
        self._save_baseline()

        logger.info(f"✅ Baseline set for {len(baseline)} features")

    def detect_drift(
        self, current_features_df: pd.DataFrame, feature_names: Optional[List[str]] = None
    ) -> DriftReport:
        """
        Detect drift in current features vs baseline

        Args:
            current_features_df: Current feature DataFrame
            feature_names: Features to check (if None, check all baseline features)

        Returns:
            DriftReport with drift analysis
        """
        if not self.baseline:
            logger.warning("No baseline set - cannot detect drift")
            return self._no_baseline_report()

        if feature_names is None:
            feature_names = list(self.baseline.keys())

        drift_metrics = []
        warnings = []

        for feature in feature_names:
            if feature not in self.baseline:
                warnings.append(f"Feature {feature} not in baseline - skipping")
                continue

            if feature not in current_features_df.columns:
                warnings.append(f"Feature {feature} not in current data - skipping")
                continue

            # Calculate drift for this feature
            drift = self._calculate_feature_drift(
                feature, current_features_df[feature].dropna().values
            )

            if drift is not None:
                drift_metrics.append(drift)

        # Calculate overall drift score
        if drift_metrics:
            overall_psi = np.mean([m.psi_score for m in drift_metrics])
            features_with_drift = sum(1 for m in drift_metrics if m.has_drift)
        else:
            overall_psi = 0.0
            features_with_drift = 0

        # Determine overall severity
        overall_severity = self._classify_psi(overall_psi)

        # Determine if retraining needed
        requires_retraining = (
            overall_psi >= self.psi_threshold_high
            or features_with_drift >= len(feature_names) * 0.30  # 30% features drifted
        )

        # Create report
        report = DriftReport(
            timestamp=datetime.now().isoformat(),
            total_features=len(feature_names),
            features_with_drift=features_with_drift,
            overall_drift_score=overall_psi,
            feature_drifts=drift_metrics,
            drift_severity=overall_severity,
            requires_retraining=requires_retraining,
            warnings=warnings,
        )

        # Log report
        self._log_drift_report(report)

        # Log warning if drift detected
        if requires_retraining:
            logger.warning(
                f"🚨 SIGNIFICANT DRIFT DETECTED! "
                f"Overall PSI: {overall_psi:.3f}, "
                f"{features_with_drift}/{len(feature_names)} features drifted. "
                f"MODEL RETRAINING RECOMMENDED!"
            )
        elif features_with_drift > 0:
            logger.info(
                f"ℹ️ Minor drift detected: "
                f"{features_with_drift}/{len(feature_names)} features, "
                f"overall PSI: {overall_psi:.3f}"
            )

        return report

    def _calculate_feature_drift(
        self, feature_name: str, current_values: np.ndarray
    ) -> Optional[DriftMetrics]:
        """
        Calculate drift metrics for a single feature

        Returns:
            DriftMetrics or None if calculation failed
        """
        try:
            baseline = self.baseline[feature_name]

            if len(current_values) == 0:
                logger.warning(f"No current values for {feature_name}")
                return None

            # Current statistics
            mean_current = np.mean(current_values)
            std_current = np.std(current_values)

            # Baseline statistics
            mean_baseline = baseline["mean"]
            std_baseline = baseline["std"]

            # 1. PSI CALCULATION
            psi = self._calculate_psi(
                current_values,
                baseline["hist_bins"],
                baseline["hist_counts"],
                baseline["sample_size"],
            )

            # 2. KS TEST
            # Reconstruct baseline distribution from histogram
            baseline_bins = np.array(baseline["hist_bins"])
            baseline_counts = np.array(baseline["hist_counts"])

            # Create samples from baseline histogram (approximate)
            baseline_samples = []
            for i in range(len(baseline_counts)):
                if i < len(baseline_bins) - 1:
                    bin_center = (baseline_bins[i] + baseline_bins[i + 1]) / 2
                    baseline_samples.extend([bin_center] * int(baseline_counts[i]))

            baseline_samples = np.array(baseline_samples)

            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_samples, current_values)

            # 3. DETERMINE DRIFT
            # Drift if: PSI > threshold OR KS test significant
            has_drift = psi >= self.psi_threshold_low or ks_pvalue < self.ks_pvalue_threshold

            # Classify severity
            severity = self._classify_psi(psi)

            return DriftMetrics(
                feature_name=feature_name,
                timestamp=datetime.now().isoformat(),
                mean_baseline=mean_baseline,
                mean_current=mean_current,
                std_baseline=std_baseline,
                std_current=std_current,
                psi_score=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pvalue,
                has_drift=has_drift,
                drift_severity=severity,
            )

        except Exception as e:
            logger.error(f"Error calculating drift for {feature_name}: {e}")
            return None

    def _calculate_psi(
        self,
        current_values: np.ndarray,
        baseline_bins: List[float],
        baseline_counts: List[int],
        baseline_total: int,
    ) -> float:
        """
        Calculate Population Stability Index (PSI)

        PSI = Σ (current_pct - baseline_pct) * ln(current_pct / baseline_pct)

        Returns:
            PSI score (0 to ∞, typically 0-1 for most applications)
        """
        try:
            # Current distribution
            current_hist, _ = np.histogram(current_values, bins=baseline_bins)
            current_total = len(current_values)

            # Percentages (add small epsilon to avoid division by zero)
            epsilon = 1e-6
            current_pcts = (current_hist + epsilon) / (current_total + epsilon)
            baseline_pcts = (np.array(baseline_counts) + epsilon) / (baseline_total + epsilon)

            # PSI formula
            psi = np.sum((current_pcts - baseline_pcts) * np.log(current_pcts / baseline_pcts))

            return float(psi)

        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0

    def _classify_psi(self, psi: float) -> str:
        """Classify PSI score to severity level"""
        if psi >= self.psi_threshold_critical:
            return "critical"
        elif psi >= self.psi_threshold_high:
            return "high"
        elif psi >= self.psi_threshold_medium:
            return "medium"
        elif psi >= self.psi_threshold_low:
            return "low"
        else:
            return "none"

    def _load_baseline(self) -> Dict:
        """Load baseline from file"""
        if not self.baseline_file.exists():
            return {}

        try:
            with open(self.baseline_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading baseline: {e}")
            return {}

    def _save_baseline(self):
        """Save baseline to file"""
        try:
            with open(self.baseline_file, "w", encoding="utf-8") as f:
                json.dump(self.baseline, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving baseline: {e}")

    def _log_drift_report(self, report: DriftReport):
        """Log drift report to file"""
        try:
            # Convert to dict (simplified for logging)
            log_entry = {
                "timestamp": report.timestamp,
                "total_features": report.total_features,
                "features_with_drift": report.features_with_drift,
                "overall_drift_score": report.overall_drift_score,
                "drift_severity": report.drift_severity,
                "requires_retraining": report.requires_retraining,
                "drifted_features": [
                    {"feature": m.feature_name, "psi": m.psi_score, "severity": m.drift_severity}
                    for m in report.feature_drifts
                    if m.has_drift
                ],
            }

            with open(self.drift_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.error(f"Error logging drift report: {e}")

    def _no_baseline_report(self) -> DriftReport:
        """Return empty report when no baseline"""
        return DriftReport(
            timestamp=datetime.now().isoformat(),
            total_features=0,
            features_with_drift=0,
            overall_drift_score=0.0,
            feature_drifts=[],
            drift_severity="none",
            requires_retraining=False,
            warnings=["No baseline set - cannot detect drift"],
        )


# Singleton instance
_drift_detector = None


def get_drift_detector() -> FeatureDriftDetector:
    """Get singleton instance"""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = FeatureDriftDetector()
    return _drift_detector
