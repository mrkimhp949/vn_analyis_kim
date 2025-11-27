#!/usr/bin/env python3
"""
Retrain ML Models với Enhanced Features (28 features)

Script này sẽ:
1. Load data từ các tickers phổ biến
2. Tạo features với enhanced feature set (28 features)
3. Train RF, XGBoost, LightGBM models
4. Save models vào thư mục models/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

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


def load_training_data(tickers: list, lookback: int = 500) -> pd.DataFrame:
    """Load và combine data từ nhiều tickers"""
    from src.data.loader import load_data
    from src.ml.features.enhanced import add_enhanced_features, get_feature_columns

    all_data = []
    feature_cols = get_feature_columns()

    logger.info(f"Loading data for {len(tickers)} tickers...")

    for ticker in tickers:
        try:
            df = load_data(ticker, lookback=lookback, use_cache=True, required_bars=100)
            if df is None or df.empty or len(df) < 100:
                logger.warning(f"  {ticker}: Insufficient data")
                continue

            # Add enhanced features
            df_enhanced = add_enhanced_features(df)

            # Create target: price up after 5 days?
            df_enhanced["target"] = (df_enhanced["close"].shift(-5) > df_enhanced["close"]).astype(
                int
            )

            # Drop rows with NaN
            df_enhanced = df_enhanced.dropna()

            if len(df_enhanced) < 50:
                logger.warning(f"  {ticker}: Not enough data after feature engineering")
                continue

            # Select only feature columns + target
            cols_to_keep = [c for c in feature_cols if c in df_enhanced.columns] + ["target"]
            df_subset = df_enhanced[cols_to_keep].copy()
            df_subset["symbol"] = ticker

            all_data.append(df_subset)
            logger.info(f"  {ticker}: {len(df_subset)} samples")

        except Exception as e:
            logger.error(f"  {ticker}: Error - {e}")

    if not all_data:
        raise ValueError("No data loaded!")

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"\nTotal samples: {len(combined)}")

    return combined


def prepare_features(df: pd.DataFrame):
    """Prepare X and y for training"""
    from src.ml.features.enhanced import get_feature_columns

    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]

    X = df[available_cols].values.astype(np.float64)
    y = df["target"].values

    # Handle any remaining NaN
    X = np.nan_to_num(X, nan=0.0)

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {np.bincount(y)}")

    return X, y, available_cols


def train_models(X_train, y_train, X_val, y_val, feature_names):
    """Train all models"""
    models = {}

    # Calculate class weight for imbalanced data
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # 1. Random Forest
    logger.info("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    logger.info(f"   RF Accuracy: {accuracy_score(y_val, rf_pred):.4f}")
    logger.info(f"   RF F1: {f1_score(y_val, rf_pred):.4f}")
    models["rf"] = rf

    # 2. XGBoost
    if XGBOOST_AVAILABLE:
        logger.info("\n🚀 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_val)
        logger.info(f"   XGB Accuracy: {accuracy_score(y_val, xgb_pred):.4f}")
        logger.info(f"   XGB F1: {f1_score(y_val, xgb_pred):.4f}")
        models["xgb"] = xgb_model

    # 3. LightGBM
    if LIGHTGBM_AVAILABLE:
        logger.info("\n💡 Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_val)
        logger.info(f"   LGB Accuracy: {accuracy_score(y_val, lgb_pred):.4f}")
        logger.info(f"   LGB F1: {f1_score(y_val, lgb_pred):.4f}")
        models["lgb"] = lgb_model

    # 4. Gradient Boosting (sklearn) - as backup
    logger.info("\n📈 Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_val)
    logger.info(f"   GB Accuracy: {accuracy_score(y_val, gb_pred):.4f}")
    logger.info(f"   GB F1: {f1_score(y_val, gb_pred):.4f}")
    models["gb"] = gb

    return models


def save_models(models, scaler, feature_names, models_dir="models"):
    """Save all models"""
    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup old models
    backup_dir = os.path.join(models_dir, f"backup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    old_files = [
        "random_forest.pkl",
        "xgboost.pkl",
        "ensemble_gb.pkl",
        "scaler.pkl",
        "rf_enhanced.pkl",
        "xgb_enhanced.pkl",
        "lgb_enhanced.pkl",
    ]
    for f in old_files:
        old_path = os.path.join(models_dir, f)
        if os.path.exists(old_path):
            import shutil

            shutil.move(old_path, os.path.join(backup_dir, f))
            logger.info(f"   Backed up {f}")

    # Save new models
    logger.info("\n💾 Saving models...")

    # Save with both naming conventions for compatibility
    if "rf" in models:
        joblib.dump(models["rf"], os.path.join(models_dir, "random_forest.pkl"))
        joblib.dump(models["rf"], os.path.join(models_dir, "rf_enhanced.pkl"))
        logger.info("   ✅ Random Forest saved")

    if "xgb" in models:
        joblib.dump(models["xgb"], os.path.join(models_dir, "xgboost.pkl"))
        joblib.dump(models["xgb"], os.path.join(models_dir, "xgb_enhanced.pkl"))
        logger.info("   ✅ XGBoost saved")

    if "lgb" in models:
        joblib.dump(models["lgb"], os.path.join(models_dir, "lgb_enhanced.pkl"))
        logger.info("   ✅ LightGBM saved")

    if "gb" in models:
        joblib.dump(models["gb"], os.path.join(models_dir, "ensemble_gb.pkl"))
        logger.info("   ✅ Gradient Boosting saved")

    # Save scaler
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler_enhanced.pkl"))
    logger.info("   ✅ Scaler saved")

    # Save model info
    model_info = {
        "timestamp": timestamp,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "models": list(models.keys()),
    }
    import json

    with open(os.path.join(models_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info("   ✅ Model info saved")

    logger.info(f"\n✅ All models saved to {models_dir}/")
    logger.info(f"   Old models backed up to {backup_dir}/")


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("🤖 ML MODEL RETRAINING")
    print("   Training with Enhanced Features (28 features)")
    print("=" * 70)

    # Popular tickers for training
    tickers = [
        # Blue chips
        "VNM",
        "VCB",
        "VIC",
        "VHM",
        "HPG",
        "FPT",
        "MWG",
        "MSN",
        "GAS",
        "SAB",
        "TCB",
        "MBB",
        "ACB",
        "VPB",
        "CTG",
        "BID",
        "STB",
        "TPB",
        "HDB",
        "VIB",
        # Mid caps
        "REE",
        "DGC",
        "PNJ",
        "MPC",
        "DPM",
        "DCM",
        "GVR",
        "PHR",
        "HSG",
        "NKG",
        "VRE",
        "KDH",
        "NLG",
        "DXG",
        "PDR",
        "KBC",
        "IJC",
        "SZC",
        "BCM",
        "IDC",
        # Others
        "PLX",
        "POW",
        "PPC",
        "NT2",
        "GEX",
        "PC1",
        "VGC",
        "HCM",
        "SSI",
        "VND",
    ]

    # Load data
    logger.info("\n📊 Step 1: Loading training data...")
    df = load_training_data(tickers, lookback=500)

    # Prepare features
    logger.info("\n🔧 Step 2: Preparing features...")
    X, y, feature_names = prepare_features(df)

    # Split data
    logger.info("\n✂️ Step 3: Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"   Train: {len(X_train)}, Val: {len(X_val)}")

    # Scale features
    logger.info("\n📏 Step 4: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train models
    logger.info("\n🏋️ Step 5: Training models...")
    models = train_models(X_train_scaled, y_train, X_val_scaled, y_val, feature_names)

    # Save models
    logger.info("\n💾 Step 6: Saving models...")
    save_models(models, scaler, feature_names)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModels trained with {len(feature_names)} features:")
    for i, name in enumerate(feature_names[:10]):
        print(f"   {i+1}. {name}")
    if len(feature_names) > 10:
        print(f"   ... and {len(feature_names) - 10} more")

    print("\n🔄 Restart the bot to use new models.")


if __name__ == "__main__":
    main()
