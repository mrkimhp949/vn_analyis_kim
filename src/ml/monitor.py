# -*- coding: utf-8 -*-
"""
ML Model Performance Monitoring and Drift Detection
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
from src.data.database import get_db
from src.monitoring.enhanced import get_enhanced_monitor

logger = logging.getLogger(__name__)


class MLModelMonitor:
    """
    Monitor ML model performance and detect drift

    Features:
    - Track predictions vs actual outcomes
    - Calculate rolling accuracy
    - Detect model drift
    - Trigger retraining when needed
    """

    def __init__(
        self,
        baseline_accuracy: float = 0.60,
        drift_threshold: float = 0.10,  # 10% drop triggers alert
        window_days: int = 30,
    ):
        self.db = get_db()
        self.monitor = get_enhanced_monitor()
        self.baseline_accuracy = baseline_accuracy
        self.drift_threshold = drift_threshold
        self.window_days = window_days

        # Create table if not exists
        self._create_table()

    def _create_table(self):
        """Create predictions table"""
        self.db.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                prediction REAL NOT NULL,
                predicted_class INTEGER NOT NULL,
                actual_outcome INTEGER,
                outcome_date TEXT,
                confidence REAL,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.db.conn.commit()

    def log_prediction(
        self,
        symbol: str,
        prediction: float,
        predicted_class: int,
        confidence: float,
        model_version: str = "1.0",
    ):
        """
        Log ML prediction for future validation

        Args:
            symbol: Stock symbol
            prediction: Raw prediction score (0-1)
            predicted_class: Predicted class (0 or 1)
            confidence: Confidence score
            model_version: Model version identifier
        """
        try:
            self.db.conn.execute(
                """
                INSERT INTO ml_predictions
                (symbol, prediction_date, prediction, predicted_class, confidence, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    symbol,
                    datetime.now().isoformat(),
                    prediction,
                    predicted_class,
                    confidence,
                    model_version,
                ),
            )
            self.db.conn.commit()

        except Exception:
            logger.error("Error logging prediction")

    def update_outcome(self, symbol: str, prediction_date: str, actual_outcome: int):
        """
        Update actual outcome for a prediction

        Args:
            symbol: Stock symbol
            prediction_date: Date of prediction
            actual_outcome: Actual outcome (0 or 1)
        """
        try:
            self.db.conn.execute(
                """
                UPDATE ml_predictions
                SET actual_outcome = ?, outcome_date = ?
                WHERE symbol = ? AND prediction_date = ?
            """,
                (actual_outcome, datetime.now().isoformat(), symbol, prediction_date),
            )
            self.db.conn.commit()

        except Exception:
            logger.error("Error updating outcome")

    def calculate_accuracy(self, window_days: Optional[int] = None) -> Dict:
        """
        Calculate model accuracy over time window

        Args:
            window_days: Number of days to look back (default: self.window_days)

        Returns:
            Dict with accuracy metrics
        """
        if window_days is None:
            window_days = self.window_days

        cutoff_date = (datetime.now() - timedelta(days=window_days)).isoformat()
        cursor = self.db.conn.execute(
            """
            SELECT
                predicted_class,
                actual_outcome,
                confidence
            FROM ml_predictions
            WHERE prediction_date >= ?
            AND actual_outcome IS NOT NULL
        """,
            (cutoff_date,),
        )

        results = cursor.fetchall()

        if not results:
            return {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "total_predictions": 0,
                "window_days": window_days,
            }

        # Calculate metrics
        predictions = np.array([r[0] for r in results])
        actuals = np.array([r[1] for r in results])

        accuracy = (predictions == actuals).mean()

        # Precision and recall for class 1 (BUY signals)
        tp = ((predictions == 1) & (actuals == 1)).sum()
        fp = ((predictions == 1) & (actuals == 0)).sum()
        fn = ((predictions == 0) & (actuals == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "total_predictions": len(results),
            "window_days": window_days,
            "calculated_at": datetime.now().isoformat(),
        }

    def check_drift(self) -> Dict:
        """
        Check for model drift

        Returns:
            Dict with drift status and metrics
        """
        # Calculate recent accuracy
        recent_metrics = self.calculate_accuracy(window_days=self.window_days)

        if recent_metrics["accuracy"] is None:
            return {
                "drift_detected": False,
                "reason": "Insufficient data",
                "recent_accuracy": None,
                "baseline_accuracy": self.baseline_accuracy,
            }

        recent_accuracy = recent_metrics["accuracy"]

        # Check if accuracy dropped significantly
        accuracy_drop = self.baseline_accuracy - recent_accuracy
        drift_detected = accuracy_drop > self.drift_threshold

        result = {
            "drift_detected": drift_detected,
            "recent_accuracy": recent_accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "accuracy_drop": accuracy_drop,
            "threshold": self.drift_threshold,
            "metrics": recent_metrics,
        }

        if drift_detected:
            result["reason"] = f"Accuracy dropped by {accuracy_drop:.2%}"
            logger.warning("⚠️ MODEL DRIFT DETECTED: {result['reason']}")

            # Track in monitoring
            self.monitor.track_error("model_drift")

        return result

    def should_retrain(self) -> bool:
        """
        Determine if model should be retrained

        Returns:
            True if retraining is recommended
        """
        drift_status = self.check_drift()

        # Retrain if drift detected
        if drift_status["drift_detected"]:
            return True

        # Retrain if accuracy is low (even without drift)
        if drift_status["recent_accuracy"] and drift_status["recent_accuracy"] < 0.55:
            logger.warning("⚠️ Low accuracy: {drift_status['recent_accuracy']:.2%}")
            return True

        # Check last retraining date
        last_retrain = self._get_last_retrain_date()
        if last_retrain:
            days_since_retrain = (datetime.now() - last_retrain).days
            if days_since_retrain > 30:  # Retrain monthly
                logger.info("ℹ️ {days_since_retrain} days since last retrain")
                return True

        return False

    def _get_last_retrain_date(self) -> Optional[datetime]:
        """Get date of last model retraining"""
        try:
            # Check if retraining log exists
            cursor = self.db.conn.execute(
                """
                SELECT MAX(created_at) FROM ml_retraining_log
            """
            )
            result = cursor.fetchone()

            if result and result[0]:
                return datetime.fromisoformat(result[0])

        except Exception:
            # Table doesn't exist yet
            pass

        return None

    def log_retraining(self, model_version: str, metrics: Dict, training_samples: int):
        """
        Log model retraining event

        Args:
            model_version: New model version
            metrics: Training metrics
            training_samples: Number of training samples
        """
        try:
            # Create table if not exists
            self.db.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_retraining_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    training_samples INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            self.db.conn.execute(
                """
                INSERT INTO ml_retraining_log
                (model_version, accuracy, precision, recall, f1_score, training_samples)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    model_version,
                    metrics.get("accuracy"),
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("f1_score"),
                    training_samples,
                ),
            )
            self.db.conn.commit()

            logger.info("✅ Logged retraining: {model_version}")

        except Exception:
            logger.error("Error logging retraining")

    def get_performance_report(self) -> str:
        """
        Generate performance report

        Returns:
            Formatted report string
        """
        # Recent metrics
        recent = self.calculate_accuracy(window_days=30)

        # Drift status
        drift = self.check_drift()

        # Build report
        lines = []
        lines.append("📊 ML MODEL PERFORMANCE REPORT")
        lines.append("=" * 50)

        if recent["accuracy"] is not None:
            lines.append("\n📈 Last 30 Days:")
            lines.append(f"  Accuracy:  {recent['accuracy']:.2%}")
            lines.append(f"  Precision: {recent['precision']:.2%}")
            lines.append(f"  Recall:    {recent['recall']:.2%}")
            lines.append(f"  Predictions: {recent['total_predictions']}")
        else:
            lines.append("\n⚠️ Insufficient data for metrics")

        lines.append("\n🎯 Drift Detection:")
        lines.append(f"  Baseline: {drift['baseline_accuracy']:.2%}")
        lines.append(f"  Current:  {drift.get('recent_accuracy', 'N/A')}")

        if drift["drift_detected"]:
            lines.append(f"  ⚠️ DRIFT DETECTED: {drift['reason']}")
        else:
            lines.append("  ✅ No drift detected")

        lines.append("\n🔄 Retraining:")
        if self.should_retrain():
            lines.append("  ⚠️ RETRAINING RECOMMENDED")
        else:
            lines.append("  ✅ Model performing well")

        return "\n".join(lines)


# Singleton
_ml_monitor = None


def get_ml_model_monitor() -> MLModelMonitor:
    """Get ML model monitor singleton"""
    global _ml_monitor
    if _ml_monitor is None:
        _ml_monitor = MLModelMonitor()
    return _ml_monitor


if __name__ == "__main__":
    # Test model monitor
    monitor = get_ml_model_monitor()

    # Log some predictions
    monitor.log_prediction("VNM", 0.75, 1, 75, "1.0")
    monitor.log_prediction("VCB", 0.45, 0, 55, "1.0")

    # Check drift
    drift = monitor.check_drift()
    print("Drift detected: {drift['drift_detected']}")

    # Get report
    report = monitor.get_performance_report()
    print("\n{report}")
