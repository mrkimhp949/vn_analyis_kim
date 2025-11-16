"""
Volatility Forecasting Model
Dự báo biến động để điều chỉnh position size
"""

import logging
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available for volatility forecasting")


class VolatilityForecaster:
    """
    Dự báo volatility (biến động) dựa trên:
    - Historical volatility (ATR, rolling std)
    - Market regime
    - Volume patterns
    - Technical indicators
    """

    def __init__(self, lookback_window: int = 20, forecast_horizon: int = 5):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán features cho volatility forecasting"""
        features_df = df.copy()

        # Historical volatility features
        if "close" in df.columns:
            returns = df["close"].pct_change()
            features_df["volatility_5d"] = returns.rolling(5).std()
            features_df["volatility_10d"] = returns.rolling(10).std()
            features_df["volatility_20d"] = returns.rolling(20).std()
            features_df["volatility_ratio"] = features_df["volatility_5d"] / (
                features_df["volatility_20d"] + 1e-6
            )

        # ATR-based features
        if "atr" in df.columns:
            features_df["atr_pct"] = df["atr"] / (df["close"] + 1e-6)
            features_df["atr_ma_ratio"] = df["atr"] / (df["atr"].rolling(20).mean() + 1e-6)

        # Volume features
        if "volume" in df.columns:
            features_df["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-6)
            features_df["volume_volatility"] = df["volume"].pct_change().rolling(10).std()

        # Price range features
        if all(col in df.columns for col in ["high", "low", "close"]):
            features_df["daily_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-6)
            features_df["range_ma_ratio"] = features_df["daily_range"] / (
                features_df["daily_range"].rolling(20).mean() + 1e-6
            )

        # RSI volatility
        if "rsi" in df.columns:
            features_df["rsi_volatility"] = df["rsi"].rolling(10).std()

        # MACD volatility
        if "macd" in df.columns:
            features_df["macd_volatility"] = df["macd"].rolling(10).std()

        return features_df

    def _prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Chuẩn bị target: volatility trong tương lai"""
        if "close" not in df.columns:
            return pd.Series(dtype=float)

        returns = df["close"].pct_change()
        # Forward-looking volatility (next N days)
        future_volatility = (
            returns.shift(-self.forecast_horizon).rolling(self.forecast_horizon).std()
        )
        return future_volatility

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train volatility forecasting model

        Returns:
            Dict with training metrics
        """
        logger.info("Training volatility forecasting model...")

        # Calculate features
        features_df = self._calculate_features(df)

        # Prepare target
        target = self._prepare_target(df)

        # Feature columns
        feature_cols = [
            "volatility_5d",
            "volatility_10d",
            "volatility_20d",
            "volatility_ratio",
            "atr_pct",
            "atr_ma_ratio",
            "volume_ratio",
            "volume_volatility",
            "daily_range",
            "range_ma_ratio",
            "rsi_volatility",
            "macd_volatility",
        ]

        # Filter available features
        available_features = [col for col in feature_cols if col in features_df.columns]

        if len(available_features) < 3:
            logger.warning("Not enough features for volatility forecasting")
            return {"error": "Insufficient features"}

        # Prepare data
        valid_idx = ~(features_df[available_features].isna().any(axis=1) | target.isna())
        X = features_df.loc[valid_idx, available_features].values
        y = target.loc[valid_idx].values

        if len(X) < 50:
            logger.warning("Not enough data for training")
            return {"error": "Insufficient data"}

        # Split train/test (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model (use XGBoost if available, else RandomForest)
        if XGBOOST_AVAILABLE:
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        metrics = {
            "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            "features_used": available_features,
        }

        logger.info(
            f"Volatility model trained - Test MAE: {metrics['test_mae']:.6f}, RMSE: {metrics['test_rmse']:.6f}"
        )

        # Save model
        self.save()

        return metrics

    def forecast(self, df: pd.DataFrame) -> float:
        """
        Dự báo volatility cho period tiếp theo

        Returns:
            Predicted volatility (as percentage)
        """
        if self.model is None:
            logger.warning("Model not trained, loading from disk...")
            if not self.load():
                return 0.02  # Default 2% volatility

        # Calculate features for latest data
        features_df = self._calculate_features(df)

        # Get feature columns
        feature_cols = [
            "volatility_5d",
            "volatility_10d",
            "volatility_20d",
            "volatility_ratio",
            "atr_pct",
            "atr_ma_ratio",
            "volume_ratio",
            "volume_volatility",
            "daily_range",
            "range_ma_ratio",
            "rsi_volatility",
            "macd_volatility",
        ]

        available_features = [col for col in feature_cols if col in features_df.columns]

        if len(available_features) < 3:
            # Fallback: use recent ATR or historical volatility
            if "atr" in df.columns and "close" in df.columns:
                return (
                    (df["atr"].iloc[-1] / df["close"].iloc[-1])
                    if df["close"].iloc[-1] > 0
                    else 0.02
                )
            return 0.02

        # Get latest features
        latest_features = features_df[available_features].iloc[-1:].values

        # Check for NaN
        if np.isnan(latest_features).any():
            # Use fallback
            if "atr" in df.columns and "close" in df.columns:
                return (
                    (df["atr"].iloc[-1] / df["close"].iloc[-1])
                    if df["close"].iloc[-1] > 0
                    else 0.02
                )
            return 0.02

        # Scale and predict
        latest_scaled = self.scaler.transform(latest_features)
        predicted_vol = self.model.predict(latest_scaled)[0]

        # Ensure reasonable bounds (0.1% to 10%)
        predicted_vol = max(0.001, min(0.10, predicted_vol))

        return float(predicted_vol)

    def save(self):
        """Save model and scaler"""
        if self.model is not None:
            joblib.dump(self.model, os.path.join(self.models_dir, "volatility_model.pkl"))
            joblib.dump(self.scaler, os.path.join(self.models_dir, "volatility_scaler.pkl"))
            logger.info("Volatility model saved")

    def load(self) -> bool:
        """Load model and scaler"""
        model_path = os.path.join(self.models_dir, "volatility_model.pkl")
        scaler_path = os.path.join(self.models_dir, "volatility_scaler.pkl")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("Volatility model loaded")
                return True
            except Exception:
                logger.warning("Failed to load volatility model")
                return False
        return False
