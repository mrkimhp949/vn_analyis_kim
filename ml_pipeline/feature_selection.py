"""SHAP-based feature selection utilities."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _compute_shap_importance(
    model, X: np.ndarray, feature_names: List[str]
) -> Optional[pd.DataFrame]:
    try:
        import shap  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("SHAP chưa được cài đặt. Bỏ qua bước feature importance.")
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Với mô hình phân loại nhị phân, shap_values có dạng [class0, class1]
        if isinstance(shap_values, list):
            shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_array = shap_values

        importance = np.mean(np.abs(shap_array), axis=0)
        importance = importance / (importance.sum() + 1e-8)

        df_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_importance": importance,
            }
        ).sort_values("shap_importance", ascending=False)

        return df_importance.reset_index(drop=True)
    except Exception:  # pragma: no cover - shap fallback
        logger.warning("Không thể tính SHAP values")
        return None


def select_features_with_shap(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    max_samples: int = 2000,
    top_k: Optional[int] = None,
    correlation_threshold: float = 0.92,
) -> Tuple[List[str], Optional[pd.DataFrame]]:
    """
    Chọn features dựa trên SHAP value và lọc các chỉ báo trùng lặp.

    Returns:
        (selected_features, shap_importance_dataframe)
    """
    if df.empty:
        logger.warning("Dataset rỗng. Bỏ qua feature selection.")
        return feature_columns, None

    available_features = [col for col in feature_columns if col in df.columns]
    if len(available_features) < 5:
        logger.warning(
            "Không đủ features để thực hiện SHAP selection (%d).",
            len(available_features),
        )
        return available_features, None

    X = df[available_features].copy()
    y = df[target_column].astype(int).copy()

    # Sample to speed up SHAP
    if len(X) > max_samples:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=max_samples, stratify=y, random_state=42
        )
    else:
        X_sample, y_sample = X, y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_scaled, y_sample)

    shap_importance = _compute_shap_importance(rf, X_scaled, list(X_sample.columns))
    if shap_importance is None:
        return available_features, None

    # Optional top-k truncation
    if top_k is not None:
        shap_importance = shap_importance.head(top_k)

    selected_features: List[str] = []
    for feature in shap_importance["feature"].tolist():
        if not selected_features:
            selected_features.append(feature)
            continue

        correlated = False
        for kept in selected_features:
            corr = X[feature].corr(X[kept])
            if pd.notna(corr) and abs(corr) >= correlation_threshold:
                correlated = True
                logger.debug(
                    "Loại bỏ feature %s do tương quan cao (%.2f) với %s",
                    feature,
                    corr,
                    kept,
                )
                break
        if not correlated:
            selected_features.append(feature)

    logger.info(
        "SHAP feature selection: từ %d còn %d features (corr threshold=%.2f)",
        len(available_features),
        len(selected_features),
        correlation_threshold,
    )

    return selected_features, shap_importance
