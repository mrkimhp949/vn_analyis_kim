# -*- coding: utf-8 -*-
"""
ML Performance Validator - Validate ML model accuracy in production

Tracks actual ML predictions vs outcomes to:
1. Validate accuracy assumptions (55-60% target)
2. Detect model degradation
3. Calculate actual expected value
4. Trigger retraining when needed

Author: Trading Bot Team
Version: 1.0.0
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import RLock
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ML_STATS_FILE = "ml_performance_stats.json"


class PredictionOutcome(Enum):
    """Outcome of an ML prediction."""

    PENDING = "PENDING"  # Trade still open
    WIN = "WIN"  # Profitable trade
    LOSS = "LOSS"  # Losing trade
    BREAKEVEN = "BREAKEVEN"  # Within transaction costs


@dataclass
class MLPrediction:
    """Single ML prediction record."""

    prediction_id: str
    symbol: str
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-100
    predicted_direction: str  # "UP", "DOWN"
    predicted_magnitude: float  # Expected % move
    entry_price: float
    timestamp: str

    # Outcome fields (filled when trade closes)
    outcome: str = PredictionOutcome.PENDING.value
    exit_price: float = 0.0
    actual_return_pct: float = 0.0
    holding_days: int = 0
    exit_timestamp: str = ""
    exit_reason: str = ""


@dataclass
class MLModelStats:
    """Aggregated ML model statistics."""

    model_version: str
    total_predictions: int = 0
    completed_predictions: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0

    # Accuracy metrics
    accuracy_pct: float = 0.0  # Win rate
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0  # Total wins / Total losses
    expected_value_pct: float = 0.0  # Expected return per trade

    # Confidence calibration
    high_conf_accuracy: float = 0.0  # Accuracy when confidence > 70
    med_conf_accuracy: float = 0.0  # Accuracy when confidence 50-70
    low_conf_accuracy: float = 0.0  # Accuracy when confidence < 50

    # Time-based metrics
    avg_holding_days: float = 0.0
    last_updated: str = ""

    # Rolling metrics (last 30 days)
    rolling_accuracy_30d: float = 0.0
    rolling_ev_30d: float = 0.0


@dataclass
class ModelHealthStatus:
    """Health status of ML model."""

    is_healthy: bool
    accuracy_status: str  # "GOOD", "WARNING", "CRITICAL"
    ev_status: str  # "POSITIVE", "NEGATIVE", "MARGINAL"
    calibration_status: str  # "CALIBRATED", "OVERCONFIDENT", "UNDERCONFIDENT"
    recommendation: str
    details: Dict = field(default_factory=dict)


class MLPerformanceValidator:
    """
    Validate ML model performance in production.

    Key metrics tracked:
    - Accuracy (win rate) - Target: 55-60%
    - Expected Value - Must be positive after costs
    - Confidence calibration - High confidence should = higher accuracy
    - Model degradation - Detect when model needs retraining
    """

    # Target thresholds
    TARGET_ACCURACY_MIN = 0.52  # Minimum 52% accuracy
    TARGET_ACCURACY_GOOD = 0.55  # Good accuracy
    TARGET_ACCURACY_EXCELLENT = 0.60  # Excellent accuracy

    # Expected value thresholds (after ~1% transaction costs)
    TARGET_EV_MIN = 0.001  # Minimum 0.1% EV per trade
    TARGET_EV_GOOD = 0.005  # Good 0.5% EV per trade

    # Degradation detection
    DEGRADATION_LOOKBACK_DAYS = 14
    DEGRADATION_ACCURACY_DROP = 0.05  # 5% drop from baseline

    # Minimum samples for statistical significance
    MIN_SAMPLES_FOR_STATS = 30
    MIN_SAMPLES_FOR_CONFIDENCE_TIER = 10

    def __init__(
        self,
        stats_file: str = DEFAULT_ML_STATS_FILE,
        model_version: str = "v2.0",
        transaction_cost_pct: float = 0.01,  # 1% round trip
    ):
        self.stats_file = stats_file
        self.model_version = model_version
        self.transaction_cost_pct = transaction_cost_pct

        self._lock = RLock()
        self._predictions: Dict[str, MLPrediction] = {}  # prediction_id -> prediction
        self._model_stats: Optional[MLModelStats] = None
        self._historical_accuracy: List[Tuple[str, float]] = []  # (date, accuracy)

        self._load_stats()
        logger.info(f"✅ MLPerformanceValidator initialized for model {model_version}")

    def record_prediction(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        entry_price: float,
        predicted_magnitude: float = 0.05,  # Default 5% expected move
    ) -> str:
        """
        Record a new ML prediction.

        Args:
            symbol: Stock symbol
            signal: "BUY", "SELL", or "HOLD"
            confidence: Model confidence 0-100
            entry_price: Entry price
            predicted_magnitude: Expected % move

        Returns:
            prediction_id for tracking
        """
        with self._lock:
            prediction_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            predicted_direction = (
                "UP" if signal == "BUY" else "DOWN" if signal == "SELL" else "NEUTRAL"
            )

            prediction = MLPrediction(
                prediction_id=prediction_id,
                symbol=symbol.upper(),
                signal=signal.upper(),
                confidence=confidence,
                predicted_direction=predicted_direction,
                predicted_magnitude=predicted_magnitude,
                entry_price=entry_price,
                timestamp=datetime.now().isoformat(),
            )

            self._predictions[prediction_id] = prediction
            self._update_stats()
            self._save_stats()

            logger.debug(
                f"📊 ML Prediction recorded: {symbol} {signal} "
                f"conf={confidence:.0f}% entry={entry_price:,.0f}"
            )

            return prediction_id

    def record_outcome(
        self,
        prediction_id: str,
        exit_price: float,
        exit_reason: str = "NORMAL",
    ) -> Optional[MLPrediction]:
        """
        Record the outcome of a prediction.

        Args:
            prediction_id: ID from record_prediction
            exit_price: Exit price
            exit_reason: Reason for exit

        Returns:
            Updated MLPrediction or None if not found
        """
        with self._lock:
            if prediction_id not in self._predictions:
                logger.warning(f"Prediction {prediction_id} not found")
                return None

            prediction = self._predictions[prediction_id]

            # Calculate actual return
            if prediction.signal == "BUY":
                actual_return_pct = (exit_price - prediction.entry_price) / prediction.entry_price
            else:
                actual_return_pct = (prediction.entry_price - exit_price) / prediction.entry_price

            # Determine outcome (accounting for transaction costs)
            net_return = actual_return_pct - self.transaction_cost_pct

            if net_return > 0.001:  # > 0.1% profit
                outcome = PredictionOutcome.WIN
            elif net_return < -0.001:  # > 0.1% loss
                outcome = PredictionOutcome.LOSS
            else:
                outcome = PredictionOutcome.BREAKEVEN

            # Calculate holding days
            entry_dt = datetime.fromisoformat(prediction.timestamp)
            holding_days = (datetime.now() - entry_dt).days

            # Update prediction
            prediction.outcome = outcome.value
            prediction.exit_price = exit_price
            prediction.actual_return_pct = actual_return_pct
            prediction.holding_days = holding_days
            prediction.exit_timestamp = datetime.now().isoformat()
            prediction.exit_reason = exit_reason

            self._update_stats()
            self._save_stats()

            logger.info(
                f"📊 ML Outcome: {prediction.symbol} {outcome.value} "
                f"return={actual_return_pct:+.2%} (net={net_return:+.2%})"
            )

            return prediction

    def _update_stats(self) -> None:
        """Update aggregated model statistics."""
        completed = [
            p for p in self._predictions.values() if p.outcome != PredictionOutcome.PENDING.value
        ]

        if not completed:
            self._model_stats = MLModelStats(model_version=self.model_version)
            return

        wins = [p for p in completed if p.outcome == PredictionOutcome.WIN.value]
        losses = [p for p in completed if p.outcome == PredictionOutcome.LOSS.value]
        breakevens = [p for p in completed if p.outcome == PredictionOutcome.BREAKEVEN.value]

        # Basic metrics
        accuracy = len(wins) / len(completed) if completed else 0

        avg_win = sum(p.actual_return_pct for p in wins) / len(wins) if wins else 0
        avg_loss = sum(p.actual_return_pct for p in losses) / len(losses) if losses else 0

        total_wins = sum(p.actual_return_pct for p in wins)
        total_losses = abs(sum(p.actual_return_pct for p in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        # Expected value
        ev = (accuracy * avg_win) + ((1 - accuracy) * avg_loss) - self.transaction_cost_pct

        # Confidence tier accuracy
        high_conf = [p for p in completed if p.confidence >= 70]
        med_conf = [p for p in completed if 50 <= p.confidence < 70]
        low_conf = [p for p in completed if p.confidence < 50]

        high_conf_acc = (
            len([p for p in high_conf if p.outcome == PredictionOutcome.WIN.value]) / len(high_conf)
            if len(high_conf) >= self.MIN_SAMPLES_FOR_CONFIDENCE_TIER
            else 0
        )
        med_conf_acc = (
            len([p for p in med_conf if p.outcome == PredictionOutcome.WIN.value]) / len(med_conf)
            if len(med_conf) >= self.MIN_SAMPLES_FOR_CONFIDENCE_TIER
            else 0
        )
        low_conf_acc = (
            len([p for p in low_conf if p.outcome == PredictionOutcome.WIN.value]) / len(low_conf)
            if len(low_conf) >= self.MIN_SAMPLES_FOR_CONFIDENCE_TIER
            else 0
        )

        # Rolling 30-day metrics
        cutoff_30d = datetime.now() - timedelta(days=30)
        recent = [p for p in completed if datetime.fromisoformat(p.exit_timestamp) > cutoff_30d]

        rolling_accuracy = (
            len([p for p in recent if p.outcome == PredictionOutcome.WIN.value]) / len(recent)
            if recent
            else 0
        )
        rolling_ev = (
            sum(p.actual_return_pct - self.transaction_cost_pct for p in recent) / len(recent)
            if recent
            else 0
        )

        # Average holding days
        avg_holding = sum(p.holding_days for p in completed) / len(completed) if completed else 0

        self._model_stats = MLModelStats(
            model_version=self.model_version,
            total_predictions=len(self._predictions),
            completed_predictions=len(completed),
            wins=len(wins),
            losses=len(losses),
            breakevens=len(breakevens),
            accuracy_pct=accuracy * 100,
            avg_win_pct=avg_win * 100,
            avg_loss_pct=avg_loss * 100,
            profit_factor=profit_factor,
            expected_value_pct=ev * 100,
            high_conf_accuracy=high_conf_acc * 100,
            med_conf_accuracy=med_conf_acc * 100,
            low_conf_accuracy=low_conf_acc * 100,
            avg_holding_days=avg_holding,
            rolling_accuracy_30d=rolling_accuracy * 100,
            rolling_ev_30d=rolling_ev * 100,
            last_updated=datetime.now().isoformat(),
        )

    def get_model_health(self) -> ModelHealthStatus:
        """
        Get current model health status.

        Returns:
            ModelHealthStatus with recommendations
        """
        if (
            not self._model_stats
            or self._model_stats.completed_predictions < self.MIN_SAMPLES_FOR_STATS
        ):
            return ModelHealthStatus(
                is_healthy=True,  # Assume healthy until proven otherwise
                accuracy_status="INSUFFICIENT_DATA",
                ev_status="INSUFFICIENT_DATA",
                calibration_status="INSUFFICIENT_DATA",
                recommendation="Need more trades for statistical significance",
                details={
                    "completed": self._model_stats.completed_predictions if self._model_stats else 0
                },
            )

        stats = self._model_stats

        # Accuracy status
        accuracy = stats.accuracy_pct / 100
        if accuracy >= self.TARGET_ACCURACY_GOOD:
            accuracy_status = "GOOD"
        elif accuracy >= self.TARGET_ACCURACY_MIN:
            accuracy_status = "WARNING"
        else:
            accuracy_status = "CRITICAL"

        # EV status
        ev = stats.expected_value_pct / 100
        if ev >= self.TARGET_EV_GOOD:
            ev_status = "POSITIVE"
        elif ev >= self.TARGET_EV_MIN:
            ev_status = "MARGINAL"
        else:
            ev_status = "NEGATIVE"

        # Calibration status (high confidence should have higher accuracy)
        if stats.high_conf_accuracy > 0 and stats.low_conf_accuracy > 0:
            if stats.high_conf_accuracy > stats.low_conf_accuracy + 5:
                calibration_status = "CALIBRATED"
            elif stats.high_conf_accuracy < stats.low_conf_accuracy:
                calibration_status = "OVERCONFIDENT"
            else:
                calibration_status = "UNDERCONFIDENT"
        else:
            calibration_status = "INSUFFICIENT_DATA"

        # Overall health
        is_healthy = (
            accuracy_status != "CRITICAL"
            and ev_status != "NEGATIVE"
            and calibration_status != "OVERCONFIDENT"
        )

        # Generate recommendation
        recommendations = []
        if accuracy_status == "CRITICAL":
            recommendations.append(
                "URGENT: Model accuracy below minimum threshold. Consider retraining."
            )
        if ev_status == "NEGATIVE":
            recommendations.append("WARNING: Expected value is negative. Review entry criteria.")
        if calibration_status == "OVERCONFIDENT":
            recommendations.append(
                "Model is overconfident. Reduce position sizes for high-confidence signals."
            )
        if stats.rolling_accuracy_30d < stats.accuracy_pct - 5:
            recommendations.append("Recent performance degrading. Monitor closely.")

        if not recommendations:
            recommendations.append("Model performing within acceptable parameters.")

        return ModelHealthStatus(
            is_healthy=is_healthy,
            accuracy_status=accuracy_status,
            ev_status=ev_status,
            calibration_status=calibration_status,
            recommendation=" | ".join(recommendations),
            details={
                "accuracy": stats.accuracy_pct,
                "expected_value": stats.expected_value_pct,
                "profit_factor": stats.profit_factor,
                "rolling_accuracy_30d": stats.rolling_accuracy_30d,
                "sample_size": stats.completed_predictions,
            },
        )

    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if model should be retrained.

        Returns:
            Tuple of (should_retrain, reason)
        """
        health = self.get_model_health()

        if health.accuracy_status == "CRITICAL":
            return True, "Accuracy below minimum threshold"

        if health.ev_status == "NEGATIVE":
            return True, "Expected value is negative"

        if self._model_stats:
            # Check for degradation
            if self._model_stats.rolling_accuracy_30d < self._model_stats.accuracy_pct - 5:
                return True, "Recent performance degradation detected"

        return False, "Model performing adequately"

    def get_confidence_adjustment(self, raw_confidence: float) -> float:
        """
        Adjust ML confidence based on historical calibration.

        If model is overconfident, reduce confidence.
        If model is underconfident, increase confidence.

        Args:
            raw_confidence: Raw model confidence 0-100

        Returns:
            Adjusted confidence 0-100
        """
        if (
            not self._model_stats
            or self._model_stats.completed_predictions < self.MIN_SAMPLES_FOR_STATS
        ):
            return raw_confidence

        stats = self._model_stats

        # Calculate calibration factor
        if raw_confidence >= 70 and stats.high_conf_accuracy > 0:
            # High confidence tier
            expected_accuracy = raw_confidence
            actual_accuracy = stats.high_conf_accuracy
            calibration_factor = (
                actual_accuracy / expected_accuracy if expected_accuracy > 0 else 1.0
            )
        elif raw_confidence >= 50 and stats.med_conf_accuracy > 0:
            # Medium confidence tier
            expected_accuracy = raw_confidence
            actual_accuracy = stats.med_conf_accuracy
            calibration_factor = (
                actual_accuracy / expected_accuracy if expected_accuracy > 0 else 1.0
            )
        else:
            calibration_factor = 1.0

        # Apply calibration (capped)
        calibration_factor = max(0.7, min(1.3, calibration_factor))
        adjusted = raw_confidence * calibration_factor

        return max(0, min(100, adjusted))

    def get_stats_summary(self) -> Dict:
        """Get summary of ML performance stats."""
        if not self._model_stats:
            return {"message": "No stats available"}

        stats = self._model_stats
        return {
            "model_version": stats.model_version,
            "total_predictions": stats.total_predictions,
            "completed_predictions": stats.completed_predictions,
            "accuracy_pct": round(stats.accuracy_pct, 1),
            "expected_value_pct": round(stats.expected_value_pct, 2),
            "profit_factor": round(stats.profit_factor, 2),
            "avg_win_pct": round(stats.avg_win_pct, 2),
            "avg_loss_pct": round(stats.avg_loss_pct, 2),
            "confidence_calibration": {
                "high_conf_accuracy": round(stats.high_conf_accuracy, 1),
                "med_conf_accuracy": round(stats.med_conf_accuracy, 1),
                "low_conf_accuracy": round(stats.low_conf_accuracy, 1),
            },
            "rolling_30d": {
                "accuracy": round(stats.rolling_accuracy_30d, 1),
                "expected_value": round(stats.rolling_ev_30d, 2),
            },
            "health": asdict(self.get_model_health()),
        }

    def _load_stats(self) -> None:
        """Load stats from file."""
        if not os.path.exists(self.stats_file):
            return

        try:
            with open(self.stats_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load predictions
            for pred_id, pred_dict in data.get("predictions", {}).items():
                self._predictions[pred_id] = MLPrediction(**pred_dict)

            # Load model stats
            if "model_stats" in data:
                self._model_stats = MLModelStats(**data["model_stats"])

            logger.info(f"📊 Loaded ML stats: {len(self._predictions)} predictions")

        except Exception as e:
            logger.warning(f"Failed to load ML stats: {e}")

    def _save_stats(self) -> None:
        """Save stats to file."""
        try:
            data = {
                "predictions": {pid: asdict(pred) for pid, pred in self._predictions.items()},
                "model_stats": asdict(self._model_stats) if self._model_stats else None,
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Failed to save ML stats: {e}")


# Singleton instance
_validator_instance: Optional[MLPerformanceValidator] = None


def get_ml_validator() -> MLPerformanceValidator:
    """Get singleton MLPerformanceValidator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = MLPerformanceValidator()
    return _validator_instance
