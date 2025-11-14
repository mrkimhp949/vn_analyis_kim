"""
Stacking meta-model cho ensemble
"""

import json
import logging
import os
from typing import List, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    LIGHTGBM_AVAILABLE = False
    logger.info(
        "LightGBM not installed. Stacking meta-model sẽ dùng LogisticRegression."
    )


class StackingMetaModel:
    """Meta-model dùng để kết hợp dự báo từ các base model."""

    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.model = None
        self.feature_names: List[str] = []

    def _build_model(self):
        if self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            logger.info("Stacking meta-model sử dụng LightGBM.")
        else:
            self.model = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )
            if self.model_type == "lightgbm":
                logger.info(
                    "LightGBM không khả dụng. Fallback sang LogisticRegression cho stacking meta-model."
                )

    def fit(
        self,
        meta_features: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        if meta_features.ndim != 2:
            raise ValueError("meta_features phải là mảng 2 chiều [n_samples, n_models]")
        if len(meta_features) != len(y):
            raise ValueError("meta_features và y phải có cùng số lượng mẫu")

        self._build_model()
        self.model.fit(meta_features, y)
        self.feature_names = feature_names or [
            f"model_{i}" for i in range(meta_features.shape[1])
        ]
        logger.info(
            "Đã huấn luyện stacking meta-model với %d mẫu và %d base models.",
            meta_features.shape[0],
            meta_features.shape[1],
        )

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Meta-model chưa được huấn luyện")
        return self.model.predict_proba(meta_features)[:, 1]

    def predict(self, meta_features: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(meta_features)
        return (proba >= 0.5).astype(int)

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(save_dir, "stacking_meta_model.pkl"))
        with open(
            os.path.join(save_dir, "stacking_meta_info.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {"model_type": self.model_type, "feature_names": self.feature_names},
                f,
                indent=2,
            )
        logger.info("Đã lưu stacking meta-model vào %s", save_dir)

    def load(self, save_dir: str):
        model_path = os.path.join(save_dir, "stacking_meta_model.pkl")
        info_path = os.path.join(save_dir, "stacking_meta_info.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy meta-model tại {model_path}")
        self.model = joblib.load(model_path)
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
                self.model_type = info.get("model_type", self.model_type)
                self.feature_names = info.get("feature_names", [])
        logger.info("Đã load stacking meta-model từ %s", model_path)
