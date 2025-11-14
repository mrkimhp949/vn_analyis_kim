"""
ML Model Monitoring - Track performance, calibrate confidence, auto-retrain triggers
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from exceptions import ModelPredictionError


@dataclass
class PredictionRecord:
    """Record một prediction"""

    symbol: str
    date: str
    predicted_signal: str  # 'BUY', 'SELL', 'HOLD'
    predicted_confidence: float
    actual_price_change: Optional[float] = None  # % change after N days
    actual_signal: Optional[str] = None  # 'BUY' if price went up, 'SELL' if down
    correct: Optional[bool] = None
    model_version: str = "default"


@dataclass
class ModelMetrics:
    """Metrics cho một model"""

    model_name: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    avg_confidence: float
    avg_confidence_correct: float
    avg_confidence_incorrect: float
    calibration_error: float  # Difference between confidence and accuracy
    last_updated: str


class MLModelMonitor:
    """
    Monitor ML model performance:
    1. Track predictions vs actuals
    2. Calculate accuracy metrics
    3. Calibrate confidence scores
    4. Trigger auto-retrain when performance degrades
    """

    def __init__(
        self,
        predictions_file: str = "ml_predictions.json",
        metrics_file: str = "ml_model_metrics.json",
        min_predictions_for_metrics: int = 50,
        accuracy_threshold: float = 0.55,  # 55% minimum accuracy
        calibration_error_threshold: float = 0.15,  # 15% max calibration error
        retrain_trigger_days: int = 7,
    ):  # Retrain if accuracy drops for 7 days
        self.predictions_file = predictions_file
        self.metrics_file = metrics_file
        self.min_predictions = min_predictions_for_metrics
        self.accuracy_threshold = accuracy_threshold
        self.calibration_error_threshold = calibration_error_threshold
        self.retrain_trigger_days = retrain_trigger_days

        self.predictions = self._load_predictions()
        self.metrics = self._load_metrics()

    def _load_predictions(self) -> List[Dict]:
        """Load predictions từ file"""
        if os.path.exists(self.predictions_file):
            try:
                with open(self.predictions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("predictions", [])
            except Exception as e:
                print(f"⚠️ Error loading predictions: {e}")
        return []

    def _save_predictions(self):
        """Save predictions to file"""
        try:
            with open(self.predictions_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"predictions": self.predictions}, f, indent=2, ensure_ascii=False
                )
        except Exception as e:
            print(f"⚠️ Error saving predictions: {e}")

    def _load_metrics(self) -> Dict:
        """Load metrics từ file"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading metrics: {e}")
        return {}

    def _save_metrics(self):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error saving metrics: {e}")

    def record_prediction(
        self,
        symbol: str,
        predicted_signal: str,
        predicted_confidence: float,
        model_version: str = "default",
    ):
        """
        Record một prediction

        Args:
            symbol: Stock symbol
            predicted_signal: 'BUY', 'SELL', 'HOLD'
            predicted_confidence: 0-100
            model_version: Model version identifier
        """
        record = PredictionRecord(
            symbol=symbol,
            date=datetime.now().isoformat(),
            predicted_signal=predicted_signal,
            predicted_confidence=predicted_confidence,
            model_version=model_version,
        )

        self.predictions.append(asdict(record))
        self._save_predictions()

    def update_prediction_result(
        self, symbol: str, date: str, actual_price_change: float, days_forward: int = 5
    ):
        """
        Update prediction với actual result

        Args:
            symbol: Stock symbol
            date: Date of prediction (ISO format)
            actual_price_change: % price change after N days
            days_forward: Number of days to evaluate
        """
        # Find prediction
        for pred in self.predictions:
            if pred["symbol"] == symbol and pred["date"] == date:
                pred["actual_price_change"] = actual_price_change

                # Determine actual signal
                if actual_price_change > 0.02:  # > 2% increase
                    pred["actual_signal"] = "BUY"
                elif actual_price_change < -0.02:  # < -2% decrease
                    pred["actual_signal"] = "SELL"
                else:
                    pred["actual_signal"] = "HOLD"

                # Check if correct
                if pred["predicted_signal"] == "HOLD":
                    pred["correct"] = pred["actual_signal"] == "HOLD"
                else:
                    pred["correct"] = pred["predicted_signal"] == pred["actual_signal"]

                self._save_predictions()
                return

        print(f"⚠️ Prediction not found: {symbol} @ {date}")

    def calculate_metrics(self, model_version: str = "default") -> ModelMetrics:
        """
        Calculate metrics cho model

        Returns:
            ModelMetrics
        """
        # Filter predictions for this model
        model_predictions = [
            p
            for p in self.predictions
            if p.get("model_version") == model_version and p.get("correct") is not None
        ]

        if len(model_predictions) < self.min_predictions:
            return ModelMetrics(
                model_name=model_version,
                total_predictions=len(model_predictions),
                correct_predictions=0,
                accuracy=0.0,
                avg_confidence=0.0,
                avg_confidence_correct=0.0,
                avg_confidence_incorrect=0.0,
                calibration_error=0.0,
                last_updated=datetime.now().isoformat(),
            )

        # Calculate accuracy
        correct = sum(1 for p in model_predictions if p.get("correct", False))
        accuracy = correct / len(model_predictions)

        # Calculate average confidence
        avg_confidence = np.mean([p["predicted_confidence"] for p in model_predictions])

        # Confidence for correct vs incorrect
        correct_predictions = [p for p in model_predictions if p.get("correct", False)]
        incorrect_predictions = [
            p for p in model_predictions if not p.get("correct", False)
        ]

        avg_confidence_correct = (
            np.mean([p["predicted_confidence"] for p in correct_predictions])
            if correct_predictions
            else 0
        )
        avg_confidence_incorrect = (
            np.mean([p["predicted_confidence"] for p in incorrect_predictions])
            if incorrect_predictions
            else 0
        )

        # Calibration error: difference between confidence and accuracy
        # If confidence is 70% but accuracy is 55%, error is 15%
        calibration_error = abs(avg_confidence / 100.0 - accuracy)

        metrics = ModelMetrics(
            model_name=model_version,
            total_predictions=len(model_predictions),
            correct_predictions=correct,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            avg_confidence_correct=avg_confidence_correct,
            avg_confidence_incorrect=avg_confidence_incorrect,
            calibration_error=calibration_error,
            last_updated=datetime.now().isoformat(),
        )

        # Save metrics
        self.metrics[model_version] = asdict(metrics)
        self._save_metrics()

        return metrics

    def calibrate_confidence(
        self, raw_confidence: float, model_version: str = "default"
    ) -> float:
        """
        Calibrate confidence score based on historical performance

        Args:
            raw_confidence: Raw confidence from model (0-100)
            model_version: Model version

        Returns:
            Calibrated confidence (0-100)
        """
        metrics = self.calculate_metrics(model_version)

        if metrics.total_predictions < self.min_predictions:
            # Not enough data, return raw confidence
            return raw_confidence

        # Calibration: adjust based on calibration error
        # If model is overconfident (confidence > accuracy), reduce confidence
        # If model is underconfident (confidence < accuracy), increase confidence

        calibration_factor = (
            metrics.accuracy / (metrics.avg_confidence / 100.0)
            if metrics.avg_confidence > 0
            else 1.0
        )
        calibration_factor = max(
            0.5, min(calibration_factor, 1.5)
        )  # Clamp between 0.5x and 1.5x

        calibrated = raw_confidence * calibration_factor

        return max(0, min(calibrated, 100))  # Clamp to 0-100

    def should_retrain(self, model_version: str = "default") -> Tuple[bool, str]:
        """
        Check if model should be retrained

        Returns:
            (should_retrain, reason)
        """
        metrics = self.calculate_metrics(model_version)

        if metrics.total_predictions < self.min_predictions:
            return False, "Not enough predictions yet"

        reasons = []

        # Check 1: Accuracy below threshold
        if metrics.accuracy < self.accuracy_threshold:
            reasons.append(
                f"Accuracy {metrics.accuracy:.1%} < threshold {self.accuracy_threshold:.1%}"
            )

        # Check 2: Calibration error too high
        if metrics.calibration_error > self.calibration_error_threshold:
            reasons.append(
                f"Calibration error {metrics.calibration_error:.1%} > threshold {self.calibration_error_threshold:.1%}"
            )

        # Check 3: Recent performance degradation
        recent_predictions = [
            p
            for p in self.predictions
            if p.get("model_version") == model_version
            and p.get("correct") is not None
            and (datetime.now() - datetime.fromisoformat(p["date"])).days
            <= self.retrain_trigger_days
        ]

        if len(recent_predictions) >= 20:  # Need at least 20 recent predictions
            recent_correct = sum(
                1 for p in recent_predictions if p.get("correct", False)
            )
            recent_accuracy = recent_correct / len(recent_predictions)

            if recent_accuracy < metrics.accuracy * 0.8:  # 20% drop in accuracy
                reasons.append(
                    f"Recent accuracy {recent_accuracy:.1%} dropped significantly from {metrics.accuracy:.1%}"
                )

        if reasons:
            return True, "; ".join(reasons)

        return False, "Performance OK"

    def get_performance_summary(self, model_version: str = "default") -> str:
        """Get formatted performance summary"""
        metrics = self.calculate_metrics(model_version)

        lines = []
        lines.append(f"📊 **ML MODEL PERFORMANCE - {model_version}**")
        lines.append("=" * 50)
        lines.append(f"📈 Total Predictions: {metrics.total_predictions}")
        lines.append(f"✅ Correct: {metrics.correct_predictions}")
        lines.append(f"🎯 Accuracy: {metrics.accuracy:.1%}")
        lines.append(f"💪 Avg Confidence: {metrics.avg_confidence:.1f}%")
        lines.append(
            f"✅ Avg Confidence (Correct): {metrics.avg_confidence_correct:.1f}%"
        )
        lines.append(
            f"❌ Avg Confidence (Incorrect): {metrics.avg_confidence_incorrect:.1f}%"
        )
        lines.append(f"⚖️ Calibration Error: {metrics.calibration_error:.1%}")

        should_retrain, reason = self.should_retrain(model_version)
        if should_retrain:
            lines.append(f"\n⚠️ **RETRAIN RECOMMENDED**")
            lines.append(f"Reason: {reason}")
        else:
            lines.append(f"\n✅ Performance OK")

        return "\n".join(lines)


# Singleton
_monitor = None


def get_ml_model_monitor() -> MLModelMonitor:
    """Get ML model monitor singleton"""
    global _monitor
    if _monitor is None:
        _monitor = MLModelMonitor()
    return _monitor
