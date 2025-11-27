"""Check ML model status"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.models.ensemble import EnhancedMLPredictor

predictor = EnhancedMLPredictor()
loaded = predictor.load_models()
print(f"Models loaded: {loaded}")
print(f"RF model: {predictor.rf_model is not None}")
print(f"XGB model: {predictor.xgb_model is not None}")
print(f"LGB model: {predictor.lgb_model is not None}")
print(f"Ensemble model: {predictor.ensemble_model is not None}")

# Check models directory
models_dir = "models"
if os.path.exists(models_dir):
    files = os.listdir(models_dir)
    print(f"\nFiles in models/: {files}")
else:
    print(f"\nmodels/ directory does not exist!")
