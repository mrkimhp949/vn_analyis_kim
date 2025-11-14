# [file name]: retrain_models.py
# [file content begin]
"""
Script để train lại ML models với đúng 18 features
Chạy: python retrain_models.py
"""

from data_loader import load_data
from features import add_ml_features, get_feature_columns
from ml_models import MLPredictor
from config import TICKERS
import pandas as pd
import numpy as np


def retrain_models():
    print("=" * 60)
    print("🔄 TRAIN LẠI ML MODELS VỚI 18 FEATURES")
    print("=" * 60)

    # Load data từ nhiều cổ phiếu
    all_data = []

    for symbol in TICKERS[:10]:  # Chỉ dùng 10 mã để test nhanh
        try:
            print(f"📥 Tải dữ liệu {symbol}...")
            df = load_data(symbol, lookback=300)
            df = add_ml_features(df)

            # Kiểm tra features
            feature_cols = get_feature_columns()
            available_features = [col for col in feature_cols if col in df.columns]
            print(
                f"   ✅ {symbol}: {len(available_features)}/{len(feature_cols)} features"
            )

            all_data.append(df)
        except Exception as e:
            print(f"❌ Lỗi tải {symbol}: {e}")

    if not all_data:
        print("❌ Không có dữ liệu để train")
        return

    # Combine tất cả data
    print(f"\n📊 Tổng hợp dữ liệu từ {len(all_data)} cổ phiếu...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Tổng số nến: {len(combined_df)}")

    # Chuẩn bị features
    feature_cols = get_feature_columns()
    available_features = [col for col in feature_cols if col in combined_df.columns]

    print(f"\n🎯 Features để train: {len(available_features)}")
    print(f"   {available_features}")

    if len(available_features) < 16:
        print("❌ Không đủ features để train")
        return

    # Prepare data for training
    X = combined_df[available_features].values
    y = combined_df["target"].values

    # Remove rows with NaN
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    print(f"✅ Data sau khi clean: {len(X)} samples")

    if len(X) < 100:
        print("❌ Không đủ data để train")
        return

    # Train models
    ml_predictor = MLPredictor()

    # Update expected features
    ml_predictor.expected_features = len(available_features)

    # Split train/test
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]

    print(f"📚 Training với {len(X_train)} samples...")

    # Scale features
    X_train_scaled = ml_predictor.scaler.fit_transform(X_train)

    # Train Random Forest
    ml_predictor.train_random_forest(X_train_scaled, y_train)

    print("\n" + "=" * 60)
    print("✅ TRAINING HOÀN TẤT!")
    print(f"🎯 Model được train với {len(available_features)} features")
    print("=" * 60)


if __name__ == "__main__":
    retrain_models()
# [file content end]
