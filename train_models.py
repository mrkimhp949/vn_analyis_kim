"""
Script để train ML models
Chạy lệnh: python train_models.py
"""

from data_loader import load_data
from ml_signals import MLSignalGenerator
from config import TICKERS
import pandas as pd


def train_models():
    print("=" * 60)
    print("🎓 TRAINING ML MODELS")
    print("=" * 60)

    # Load data từ nhiều cổ phiếu để train tốt hơn
    all_data = []

    for symbol in TICKERS:
        try:
            print(f"\n📥 Tải dữ liệu {symbol}...")
            df = load_data(symbol, lookback=500)  # Load nhiều data hơn
            all_data.append(df)
            print(f"✅ Đã tải {len(df)} nến")
        except Exception as e:
            print(f"❌ Lỗi tải {symbol}: {e}")

    # Combine tất cả data
    print(f"\n📊 Tổng hợp dữ liệu từ {len(all_data)} cổ phiếu...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Tổng số nến: {len(combined_df)}")

    # Train models
    ml_generator = MLSignalGenerator()
    ml_generator.train_models(combined_df)

    print("\n" + "=" * 60)
    print("✅ TRAINING HOÀN TẤT!")
    print("=" * 60)
    print("\nCác file models đã được lưu tại thư mục 'models/':")
    print("  - random_forest.pkl")
    print("  - lstm_model.h5")
    print("  - scaler.pkl")
    print("\nBây giờ bạn có thể chạy bot với: python main.py")


if __name__ == "__main__":
    train_models()
