"""
Script khởi tạo models tự động
Chạy: python scripts/init_models.py
"""

import os
import sys
from pathlib import Path

# Fix encoding for Windows
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ml.models.predictor import MLPredictor


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
