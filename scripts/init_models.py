"""
Script khởi tạo models tự động
Chạy: python scripts/init_models.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from data_loader import load_data
from features import add_ml_features
from ml_models import MLPredictor


def init_models():
    print("🔄 Khởi tạo models...")

    # Khởi tạo ML predictor
    predictor = MLPredictor()

    # Kiểm tra xem models đã tồn tại chưa
    models_dir = "models"
    rf_path = os.path.join(models_dir, "random_forest.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if os.path.exists(rf_path) and os.path.exists(scaler_path):
        print("✅ Models đã tồn tại, loading...")
        predictor.load_models()
    else:
        print("📦 Tạo dummy models...")
        predictor.create_dummy_models()

    print("✅ Models initialization completed!")


if __name__ == "__main__":
    init_models()
