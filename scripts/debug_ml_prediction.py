"""Debug ML prediction in detail"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

from src.data.loader import load_data
from src.ml.features.enhanced_v2 import add_enhanced_features, get_feature_columns
from src.ml.models.ensemble import EnhancedMLPredictor

# Load data for one ticker
ticker = "FPT"
print(f"Loading data for {ticker}...")
df = load_data(ticker, lookback=250, use_cache=True, required_bars=50)
print(f"Loaded {len(df)} bars")

# Add features
print("\nAdding enhanced features...")
df_enhanced = add_enhanced_features(df)
print(f"Enhanced df shape: {df_enhanced.shape}")

# Get feature columns
feature_cols = get_feature_columns()
print(f"\nExpected features: {len(feature_cols)}")
print(f"Feature columns: {feature_cols[:10]}...")

# Check missing features
missing = [col for col in feature_cols if col not in df_enhanced.columns]
if missing:
    print(f"\n❌ Missing features: {missing}")
else:
    print("\n✅ All features present")

# Extract features
X = df_enhanced[feature_cols].values
print(f"\nX shape: {X.shape}")
print(f"X dtype: {X.dtype}")

# Check for non-numeric
for i, col in enumerate(feature_cols):
    val = X[-1, i]
    if not isinstance(val, (int, float, np.integer, np.floating)):
        print(f"❌ Non-numeric in {col}: {type(val)} = {val}")

# Convert to float
try:
    X_float = X.astype(np.float64)
    print(f"\n✅ Converted to float64")
except Exception as e:
    print(f"\n❌ Cannot convert to float: {e}")

# Check for NaN
nan_count = np.isnan(X_float[-1]).sum()
print(f"NaN count in last row: {nan_count}")

# Load predictor
print("\nLoading ML predictor...")
predictor = EnhancedMLPredictor()
predictor.load_models()

print(f"Expected features by predictor: {predictor.expected_features}")
print(f"Scaler fitted: {hasattr(predictor.scaler, 'mean_')}")

# Try prediction
print("\nTrying prediction...")
try:
    # Scale first
    X_scaled = predictor.scaler.transform(X_float)
    print(f"Scaled shape: {X_scaled.shape}")

    # Predict with RF
    if predictor.rf_model is not None:
        proba = predictor.rf_model.predict_proba(X_scaled[-1:])
        print(f"RF prediction: {proba}")

    # Full predict
    result = predictor.predict(X_float, use_ensemble=True)
    print(f"Final prediction: {result[-1]:.4f}")

except Exception as e:
    print(f"❌ Prediction error: {e}")
    import traceback

    traceback.print_exc()
