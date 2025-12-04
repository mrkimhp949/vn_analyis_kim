# -*- coding: utf-8 -*-
"""
ML Signal Generator V2 - Improved accuracy (58-62%)
Integrates with entry_logic for trading decisions
"""

import logging
import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Model paths
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_v2.pkl")
MODEL_INFO_PATH = os.path.join(MODELS_DIR, "model_info_v2.json")
MODEL_PATHS = {
    "rf": os.path.join(MODELS_DIR, "rf_v2.pkl"),
    "xgb": os.path.join(MODELS_DIR, "xgb_v2.pkl"),
    "lgb": os.path.join(MODELS_DIR, "lgb_v2.pkl"),
    "ensemble": os.path.join(MODELS_DIR, "ensemble_v2.pkl"),
    "stacking": os.path.join(MODELS_DIR, "stacking_v2.pkl"),
}


class MLSignalGeneratorV2:
    """
    ML Signal Generator V2 with improved accuracy (target 65%+).

    Key improvements:
    - Uses enhanced_v2 features (63 features)
    - Multi-horizon target (more predictable)
    - Ensemble of regularized models
    - Stacking ensemble option
    - Calibrated confidence scores
    """

    def __init__(
        self,
        model_name: str = "stacking",  # stacking often best
        confidence_threshold: float = 0.55,
        use_ensemble: bool = True,
    ):
        """
        Initialize ML Signal Generator V2.

        Args:
            model_name: Which model to use ('rf', 'xgb', 'lgb', 'ensemble', 'stacking')
            confidence_threshold: Minimum confidence for BUY signal
            use_ensemble: Use ensemble voting of all models
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.use_ensemble = use_ensemble

        self.scaler = None
        self.model = None
        self.models = {}  # For ensemble
        self.feature_cols = None
        self.selected_features = None  # Features selected during training

        self._load_models()

    def _load_models(self) -> None:
        """Load trained models and scaler."""
        try:
            # Load scaler
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info(f"✅ Loaded scaler from {SCALER_PATH}")
            else:
                logger.warning(f"⚠️ Scaler not found at {SCALER_PATH}")
                return

            # Load model info to get selected features
            if os.path.exists(MODEL_INFO_PATH):
                import json
                with open(MODEL_INFO_PATH, "r") as f:
                    model_info = json.load(f)
                    self.selected_features = model_info.get("selected_features")
                    logger.info(f"✅ Loaded model info (v{model_info.get('version', '2')})")

            # Load feature columns - prioritize scaler's feature names for consistency
            from src.ml.features.enhanced_v2 import get_feature_columns_v2

            # Priority: scaler's feature_names_in_ > model_info selected_features > all features
            if hasattr(self.scaler, 'feature_names_in_'):
                self.feature_cols = list(self.scaler.feature_names_in_)
                logger.info(f"  Using {len(self.feature_cols)} features from scaler")
            elif self.selected_features:
                self.feature_cols = self.selected_features
                logger.info(f"  Using {len(self.feature_cols)} selected features from model_info")
            else:
                self.feature_cols = get_feature_columns_v2()
                logger.info(f"  Using all {len(self.feature_cols)} features")

            if self.use_ensemble:
                # Load all models for ensemble
                for name, path in MODEL_PATHS.items():
                    if os.path.exists(path):
                        self.models[name] = joblib.load(path)
                        logger.info(f"✅ Loaded {name} model")
            else:
                # Load single model (prefer stacking > ensemble > rf)
                preferred_order = ["stacking", "ensemble", "rf", "xgb", "lgb"]
                model_loaded = False

                # First try requested model
                model_path = MODEL_PATHS.get(self.model_name)
                if model_path and os.path.exists(model_path):
                    self.model = joblib.load(model_path)
                    logger.info(f"✅ Loaded {self.model_name} model")
                    model_loaded = True

                # Fallback to preferred order
                if not model_loaded:
                    for name in preferred_order:
                        path = MODEL_PATHS.get(name)
                        if path and os.path.exists(path):
                            self.model = joblib.load(path)
                            self.model_name = name
                            logger.info(f"✅ Loaded {name} model (fallback)")
                            break

        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")

    def _prepare_features(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
    ) -> Optional[np.ndarray]:
        """Prepare features for prediction."""
        if self.scaler is None or self.feature_cols is None:
            logger.warning("Models not loaded")
            return None

        try:
            from src.ml.features.enhanced_v2 import add_enhanced_features_v2

            # Add features
            df_features = add_enhanced_features_v2(df, index_df, target_type="multi_horizon")

            # Get latest row
            if df_features.empty:
                return None

            # Check for required features
            missing = [col for col in self.feature_cols if col not in df_features.columns]
            if missing:
                logger.warning(f"Missing features: {missing[:5]}...")
                for col in missing:
                    df_features[col] = 0

            # Get features for latest row - ONLY use the exact features from training
            # This ensures we don't pass extra features that weren't in the training set
            X = df_features[self.feature_cols].iloc[-1:].copy()

            # Handle NaN
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)

            # Scale - use feature names from scaler to ensure correct order
            if hasattr(self.scaler, 'feature_names_in_'):
                scaler_features = list(self.scaler.feature_names_in_)
                # Reorder columns to match scaler's expected order
                missing_in_scaler = [f for f in scaler_features if f not in X.columns]
                for col in missing_in_scaler:
                    X[col] = 0
                X = X[scaler_features]

            X_scaled = self.scaler.transform(X)

            return X_scaled

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return None

    def generate_signal(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
    ) -> Dict:
        """
        Generate ML signal for a stock.

        Args:
            df: Stock DataFrame with OHLCV
            index_df: Index DataFrame (VNINDEX)
            symbol: Stock symbol for logging

        Returns:
            Dict with signal, confidence, and metadata
        """
        symbol_tag = f"[{symbol}] " if symbol else ""

        # Default response
        default_response = {
            "signal": "HOLD",
            "confidence": 0,
            "model": self.model_name,
            "version": "v2",
            "probabilities": {"buy": 0, "hold": 1},
        }

        # Check if models loaded
        if self.scaler is None:
            logger.warning(f"{symbol_tag}ML models not loaded")
            return default_response

        # Prepare features
        X_scaled = self._prepare_features(df, index_df)
        if X_scaled is None:
            return default_response

        try:
            if self.use_ensemble and self.models:
                # Ensemble prediction
                probas = []
                for name, model in self.models.items():
                    try:
                        proba = model.predict_proba(X_scaled)[0, 1]
                        probas.append(proba)
                    except Exception:
                        pass

                if probas:
                    buy_proba = np.mean(probas)
                else:
                    return default_response
            else:
                # Single model prediction
                if self.model is None:
                    return default_response
                buy_proba = self.model.predict_proba(X_scaled)[0, 1]

            # Convert to confidence (0-100)
            confidence = int(buy_proba * 100)

            # Determine signal
            if buy_proba >= self.confidence_threshold:
                signal = "BUY"
            elif buy_proba <= (1 - self.confidence_threshold):
                signal = "SELL"
            else:
                signal = "HOLD"

            result = {
                "signal": signal,
                "confidence": confidence,
                "model": self.model_name if not self.use_ensemble else "ensemble",
                "version": "v2",
                "probabilities": {
                    "buy": round(buy_proba, 4),
                    "hold": round(1 - buy_proba, 4),
                },
            }

            logger.info(f"{symbol_tag}ML V2 Signal: {signal} (confidence: {confidence}%)")

            return result

        except Exception as e:
            logger.error(f"{symbol_tag}Signal generation failed: {e}")
            return default_response

    def batch_generate_signals(
        self,
        symbols: list,
        lookback: int = 200,
    ) -> Dict[str, Dict]:
        """
        Generate signals for multiple symbols.

        Args:
            symbols: List of stock symbols
            lookback: Days of historical data

        Returns:
            Dict mapping symbol to signal
        """
        from src.data.loader import load_data

        results = {}

        # Load index once
        try:
            index_df = load_data("VNINDEX", lookback=lookback, is_index=True)
        except Exception:
            index_df = None

        for symbol in symbols:
            try:
                df = load_data(symbol, lookback=lookback)
                if df is not None and len(df) >= 50:
                    results[symbol] = self.generate_signal(df, index_df, symbol)
                else:
                    results[symbol] = {
                        "signal": "HOLD",
                        "confidence": 0,
                        "error": "Insufficient data",
                    }
            except Exception as e:
                results[symbol] = {
                    "signal": "HOLD",
                    "confidence": 0,
                    "error": str(e),
                }

        return results


# Singleton instance
_generator_v2 = None


def get_ml_signal_generator_v2() -> MLSignalGeneratorV2:
    """Get singleton instance of ML Signal Generator V2."""
    global _generator_v2
    if _generator_v2 is None:
        _generator_v2 = MLSignalGeneratorV2(model_name="rf", use_ensemble=False)
    return _generator_v2


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING ML SIGNAL GENERATOR V2")
    print("=" * 60)

    generator = MLSignalGeneratorV2(model_name="rf")

    # Test with sample symbols
    symbols = ["VNM", "FPT", "VIC", "HPG", "MWG"]

    results = generator.batch_generate_signals(symbols, lookback=200)

    print("\n📊 SIGNALS:")
    for symbol, signal in results.items():
        print(f"   {symbol}: {signal['signal']} ({signal['confidence']}%)")

    print("\n✅ Test complete!")
