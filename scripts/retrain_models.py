#!/usr/bin/env python3
"""
Retrain ML Models V2 - Improved accuracy (58-62%)

Script này sẽ:
1. Load data từ các tickers phổ biến
2. Tạo features với enhanced_v2 feature set (41 features)
3. Train RF, XGBoost, LightGBM models với walk-forward validation
4. Save models vào thư mục models/

V2 Improvements:
- Multi-horizon target (3d, 5d, 10d) thay vì next-day
- 41 predictive features (momentum, regime, mean reversion)
- Walk-forward validation (no data leakage)
- Proper regularization
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    """Main training pipeline - uses V2 trainer"""
    print("=" * 70)
    print("🤖 ML MODEL RETRAINING V2")
    print("   Training with Enhanced V2 Features (41 features)")
    print("   Target: 58-62% accuracy (8-12% edge)")
    print("=" * 70)

    # Import V2 trainer
    from src.ml.training.train_v2 import MLTrainerV2

    # Popular tickers for training
    tickers = [
        # Blue chips (VN30)
        "VNM", "VCB", "VIC", "VHM", "HPG", "FPT", "MWG", "MSN", "GAS", "SAB",
        "TCB", "MBB", "ACB", "VPB", "CTG", "BID", "STB", "TPB", "HDB", "VIB",
        # Mid caps
        "REE", "DGC", "PNJ", "MPC", "DPM", "DCM", "GVR", "PHR", "HSG", "NKG",
        "VRE", "KDH", "NLG", "DXG", "PDR", "KBC", "IJC", "SZC", "BCM", "IDC",
        # Others
        "PLX", "POW", "PPC", "NT2", "GEX", "PC1", "VGC", "HCM", "SSI", "VND",
    ]

    # Initialize trainer
    trainer = MLTrainerV2(
        models_dir="models",
        n_splits=5,  # 5-fold time series CV
    )

    # Run training
    results = trainer.train(tickers, lookback=500)

    # Summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n🏆 Best Model: {results['best_model']}")
    print(f"📊 Accuracy: {results['best_accuracy']:.4f}")
    print(f"📈 Edge vs Random: {results['edge_vs_random']:+.2f}%")
    print(f"📋 Features: {results['n_features']}")
    print(f"📋 Samples: {results['n_samples']}")

    if results['edge_vs_random'] >= 8:
        print("\n✅ TARGET ACHIEVED: Edge >= 8%")
    else:
        print(f"\n⚠️ Target not achieved. Current edge: {results['edge_vs_random']:.2f}%")

    print("\n🔄 Restart the bot to use new models.")

    return results


def main_legacy():
    """Legacy training pipeline (V1) - kept for backward compatibility"""
    print("=" * 70)
    print("🤖 ML MODEL RETRAINING (Legacy V1)")
    print("   Training with Enhanced Features (28 features)")
    print("=" * 70)

    import json
    from datetime import datetime

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Try import optional libraries
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
        logger.warning("XGBoost not available")

    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False
        logger.warning("LightGBM not available")

    def load_training_data(tickers, lookback=500):
        from src.data.loader import load_data
        from src.ml.features.enhanced import add_enhanced_features, get_feature_columns

        all_data = []
        feature_cols = get_feature_columns()

        logger.info(f"Loading data for {len(tickers)} tickers...")

        for ticker in tickers:
            try:
                df = load_data(ticker, lookback=lookback, use_cache=True, required_bars=100)
                if df is None or df.empty or len(df) < 100:
                    continue

                df_enhanced = add_enhanced_features(df)
                df_enhanced["target"] = (df_enhanced["close"].shift(-5) > df_enhanced["close"]).astype(int)
                df_enhanced = df_enhanced.dropna()

                if len(df_enhanced) < 50:
                    continue

                cols_to_keep = [c for c in feature_cols if c in df_enhanced.columns] + ["target"]
                df_subset = df_enhanced[cols_to_keep].copy()
                df_subset["symbol"] = ticker

                all_data.append(df_subset)
                logger.info(f"  {ticker}: {len(df_subset)} samples")

            except Exception as e:
                logger.error(f"  {ticker}: Error - {e}")

        if not all_data:
            raise ValueError("No data loaded!")

        return pd.concat(all_data, ignore_index=True)

    def prepare_features(df):
        from src.ml.features.enhanced import get_feature_columns

        feature_cols = get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols].values.astype(np.float64)
        y = df["target"].values
        X = np.nan_to_num(X, nan=0.0)

        return X, y, available_cols

    # Tickers
    tickers = [
        "VNM", "VCB", "VIC", "VHM", "HPG", "FPT", "MWG", "MSN", "GAS", "SAB",
        "TCB", "MBB", "ACB", "VPB", "CTG", "BID", "STB", "TPB", "HDB", "VIB",
        "REE", "DGC", "PNJ", "MPC", "DPM", "DCM", "GVR", "PHR", "HSG", "NKG",
    ]

    # Load and prepare
    df = load_training_data(tickers, lookback=500)
    X, y, feature_names = prepare_features(df)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train
    models = {}

    # RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    models["rf"] = rf
    logger.info(f"RF Accuracy: {accuracy_score(y_val, rf.predict(X_val_scaled)):.4f}")

    # XGB
    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
        xgb_model.fit(X_train_scaled, y_train)
        models["xgb"] = xgb_model
        logger.info(f"XGB Accuracy: {accuracy_score(y_val, xgb_model.predict(X_val_scaled)):.4f}")

    # LGB
    if LIGHTGBM_AVAILABLE:
        lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
        lgb_model.fit(X_train_scaled, y_train)
        models["lgb"] = lgb_model
        logger.info(f"LGB Accuracy: {accuracy_score(y_val, lgb_model.predict(X_val_scaled)):.4f}")

    # Save to backup folder (V1)
    models_dir = "models/backup_v1_latest"
    os.makedirs(models_dir, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, os.path.join(models_dir, f"{name}_v1.pkl"))

    joblib.dump(scaler, os.path.join(models_dir, "scaler_v1.pkl"))

    logger.info(f"Legacy V1 models saved to {models_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrain ML Models")
    parser.add_argument("--legacy", action="store_true", help="Use legacy V1 training")
    args = parser.parse_args()

    if args.legacy:
        main_legacy()
    else:
        main()
