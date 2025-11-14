"""Model versioning and drift monitoring utilities."""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelMonitor:
    def __init__(
        self,
        history_file: str = "models/model_history.json",
        drift_threshold: float = 0.05,
    ):
        self.history_file = history_file
        self.drift_threshold = drift_threshold
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        self._history: Dict[str, List[Dict]] = self._load_history()

    def _load_history(self) -> Dict[str, List[Dict]]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except (
                Exception
            ) as exc:  # pragma: no cover - corrupted file handled gracefully
                logger.warning(f"Không thể đọc lịch sử model: {exc}")
        return {}

    def _save_history(self) -> None:
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Không thể lưu lịch sử model: {exc}")

    def record_training_run(
        self,
        model_name: str,
        metrics: Dict,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "metadata": metadata or {},
        }
        self._history.setdefault(model_name, []).append(entry)
        entry["version"] = len(self._history[model_name])
        self._save_history()
        logger.info(
            "Model '%s' training recorded (version %s)",
            model_name,
            entry["version"],
        )
        return entry

    def check_drift(
        self,
        model_name: str,
        metric_key: str = "accuracy",
        window: int = 3,
    ) -> Dict:
        history = self._history.get(model_name, [])
        if len(history) < 2:
            return {"is_drifting": False, "reason": "Not enough history"}

        current_entry = history[-1]
        current_metrics = current_entry.get("metrics", {})
        current_value = self._extract_metric(current_metrics, metric_key)
        if current_value is None:
            return {"is_drifting": False, "reason": "Metric missing"}

        recent = history[-(window + 1) : -1]
        recent_values = [
            self._extract_metric(item.get("metrics", {}), metric_key) for item in recent
        ]
        recent_values = [val for val in recent_values if val is not None]
        if not recent_values:
            return {"is_drifting": False, "reason": "No baseline metric"}

        baseline = sum(recent_values) / len(recent_values)
        drop = baseline - current_value
        is_drifting = drop >= self.drift_threshold

        info = {
            "is_drifting": is_drifting,
            "baseline": baseline,
            "current": current_value,
            "delta": drop,
            "threshold": self.drift_threshold,
            "version": current_entry.get("version"),
        }
        if is_drifting:
            logger.warning(
                "Drift detected for model '%s': baseline=%.4f, current=%.4f",
                model_name,
                baseline,
                current_value,
            )
        return info

    @staticmethod
    def _extract_metric(metrics: Dict, metric_key: str) -> Optional[float]:
        value = metrics.get(metric_key)
        if isinstance(value, dict):
            return float(value.get("mean")) if value.get("mean") is not None else None
        if isinstance(value, (int, float)):
            return float(value)
        return None


_monitor_instance: Optional[ModelMonitor] = None


def get_model_monitor() -> ModelMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor()
    return _monitor_instance
