#!/usr/bin/env python3
"""
Train ML Models V3 for Signal Generator V3

This script trains ensemble models (RF, GB, XGBoost) and saves them
with v3 naming convention for use with EnhancedMLSignalGeneratorV3.

Usage:
    python scripts/train_models_v3.py
    python scripts/train_models_v3.py --max-symbols 50 --lookback 300
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not available, will skip")

from src.data.loader import load_data
from src.ml.features.enhanced_v2 import add_enhanced_features, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"


def load_symbols(ticker_file: str = "quality_tickers.txt") -> list:
    """Load symbols from file or use VN30 defaults"""
    vn30 = [
        "VNM",
        "VCB",
        "FPT",
        "HPG",
        "MWG",
        "TCB",
        "VHM",
        "VIC",
        "MBB",
        "ACB",
        "CTG",
        "BID",
        "VPB",
        "HDB",
        "STB",
        "SSI",
        "VND",
        "TPB",
        "MSN",
        "POW",
        "GAS",
        "PLX",
        "VJC",
        "SAB",
        "REE",
        "PNJ",
        "KDH",
        "DGC",
        "GVR",
        "VRE",
    ]

    if os.path.exists(ticker_file):
        try:
            with open(ticker_file, "r") as f:
                symbols = [
                    line.strip().upper() for line in f if line.strip() and not line.startswith("#")
                ]
            if symbols:
                logger.info(f"✅ Loaded {len(symbols)} symbols from {ticker_file}")
                return symbols
        except Exception as e:
            logger.warning(f"⚠️ Could not load {ticker_file}: {e}")

    logger.info(f"📊 Using VN30 symbols: {len(vn30)} symbols")
    return vn30


def load_training_data(symbols: list, lookback: int = 500, max_symbols: int = 100):
    """Load and prepare training data"""
    logger.info(f"\n📥 Loading data from {min(len(symbols), max_symbols)} symbols...")

    all_data = []
    index_df = None

    # Load VNINDEX first
    try:
        index_df = load_data("VNINDEX", lookback=lookback, is_index=True)
        if index_df is not None and not index_df.empty:
            logger.info(f"   ✅ VNINDEX: {len(index_df)} bars")
    except Exception as e:
        logger.warning(f"   ⚠️ Could not load VNINDEX: {e}")

    # Load symbols
    success_count = 0
    for i, symbol in enumerate(symbols[:max_symbols], 1):
        try:
            if i % 10 == 0 or i == 1:
                logger.info(f"   [{i}/{min(len(symbols), max_symbols)}] Loading {symbol}...")

            df = load_data(symbol, lookback=lookback)

            if df is None or df.empty or len(df) < 100:
                continue

            # Add enhanced features
            df_feat = add_enhanced_features(df, index_df=index_df)

            if df_feat is not None and not df_feat.empty:
                df_feat["symbol"] = symbol
                all_data.append(df_feat)
                success_count += 1

        except Exception as e:
            continue

    if not all_data:
        raise ValueError("No data loaded! Check data source.")

    logger.info(f"\n📊 Loaded data from {success_count} symbols")

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)

    # Get feature columns and drop NaN
    feature_cols = get_feature_columns()
    combined = combined.dropna(subset=feature_cols + ["target"])

    logger.info(f"✅ Total samples: {len(combined):,}")
    logger.info(f"✅ Features: {len(feature_cols)}")
    logger.info(f"✅ Class distribution:")
    logger.info(
        f"   - BUY (1):  {(combined['target'] == 1).sum():,} ({(combined['target'] == 1).mean()*100:.1f}%)"
    )
    logger.info(
        f"   - SELL (0): {(combined['target'] == 0).sum():,} ({(combined['target'] == 0).mean()*100:.1f}%)"
    )

    return combined, feature_cols


def train_and_save_models(df: pd.DataFrame, feature_cols: list, test_size: float = 0.2):
    """Train RF, GB, XGBoost and save with v3 naming"""

    logger.info("\n" + "=" * 70)
    logger.info("🎓 TRAINING ML MODELS V3")
    logger.info("=" * 70)

    # Prepare data
    X = df[feature_cols].values
    y = df["target"].values

    # Time-based split (not random)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"\n📊 Data split (time-based):")
    logger.info(f"   Training: {len(X_train):,} samples")
    logger.info(f"   Testing:  {len(X_test):,} samples")

    # Scale features
    logger.info("\n📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler_v3.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"   ✅ Saved scaler to {scaler_path}")

    results = {}

    # 1. Train Random Forest
    logger.info("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    rf_pred = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    logger.info(f"   Accuracy: {rf_acc:.2%}")
    logger.info(f"   F1 Score: {rf_f1:.2%}")

    rf_path = os.path.join(MODELS_DIR, "random_forest_v3.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    logger.info(f"   ✅ Saved to {rf_path}")
    results["random_forest"] = {"accuracy": rf_acc, "f1": rf_f1}

    # 2. Train Gradient Boosting
    logger.info("\n🚀 Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    gb.fit(X_train_scaled, y_train)

    gb_pred = gb.predict(X_test_scaled)
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_f1 = f1_score(y_test, gb_pred)
    logger.info(f"   Accuracy: {gb_acc:.2%}")
    logger.info(f"   F1 Score: {gb_f1:.2%}")

    gb_path = os.path.join(MODELS_DIR, "gradient_boosting_v3.pkl")
    with open(gb_path, "wb") as f:
        pickle.dump(gb, f)
    logger.info(f"   ✅ Saved to {gb_path}")
    results["gradient_boosting"] = {"accuracy": gb_acc, "f1": gb_f1}

    # 3. Train XGBoost (if available)
    if HAS_XGBOOST:
        logger.info("\n⚡ Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        xgb.fit(X_train_scaled, y_train)

        xgb_pred = xgb.predict(X_test_scaled)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_f1 = f1_score(y_test, xgb_pred)
        logger.info(f"   Accuracy: {xgb_acc:.2%}")
        logger.info(f"   F1 Score: {xgb_f1:.2%}")

        xgb_path = os.path.join(MODELS_DIR, "xgboost_v3.pkl")
        with open(xgb_path, "wb") as f:
            pickle.dump(xgb, f)
        logger.info(f"   ✅ Saved to {xgb_path}")
        results["xgboost"] = {"accuracy": xgb_acc, "f1": xgb_f1}

    # Save model info
    model_info = {
        "version": "v3",
        "created_at": datetime.now().isoformat(),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "models": results,
    }

    info_path = os.path.join(MODELS_DIR, "model_info_v3.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"\n📄 Saved model info to {info_path}")

    # Save feature importance
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "rf_importance": rf.feature_importances_,
            "gb_importance": gb.feature_importances_,
        }
    )
    if HAS_XGBOOST:
        importance_df["xgb_importance"] = xgb.feature_importances_

    importance_df["avg_importance"] = importance_df[
        [c for c in importance_df.columns if "importance" in c]
    ].mean(axis=1)
    importance_df = importance_df.sort_values("avg_importance", ascending=False)

    importance_path = os.path.join(MODELS_DIR, "feature_importance_v3.csv")
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"📊 Saved feature importance to {importance_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETED!")
    logger.info("=" * 70)
    logger.info("\n📊 Model Performance Summary:")
    for name, metrics in results.items():
        logger.info(f"   {name}: Accuracy={metrics['accuracy']:.2%}, F1={metrics['f1']:.2%}")

    logger.info("\n📁 Files saved:")
    logger.info("   - random_forest_v3.pkl")
    logger.info("   - gradient_boosting_v3.pkl")
    if HAS_XGBOOST:
        logger.info("   - xgboost_v3.pkl")
    logger.info("   - scaler_v3.pkl")
    logger.info("   - model_info_v3.json")
    logger.info("   - feature_importance_v3.csv")

    # Print top features
    logger.info("\n🔝 Top 10 Important Features:")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"   {row['feature']}: {row['avg_importance']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train ML Models V3")
    parser.add_argument("--lookback", type=int, default=500, help="Days of data (default: 500)")
    parser.add_argument("--max-symbols", type=int, default=100, help="Max symbols (default: 100)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size (default: 0.2)")
    parser.add_argument(
        "--ticker-file", type=str, default="quality_tickers.txt", help="Ticker file"
    )
    args = parser.parse_args()

    logger.info("\n" + "=" * 70)
    logger.info("🚀 ML MODEL TRAINING PIPELINE V3")
    logger.info("=" * 70)

    # Load symbols
    symbols = load_symbols(args.ticker_file)

    logger.info(f"\n⚙️ Configuration:")
    logger.info(f"   Symbols: {len(symbols)} (using up to {args.max_symbols})")
    logger.info(f"   Lookback: {args.lookback} days")
    logger.info(f"   Test size: {args.test_size * 100:.0f}%")

    try:
        # Load data
        df, feature_cols = load_training_data(
            symbols=symbols, lookback=args.lookback, max_symbols=args.max_symbols
        )

        # Train models
        results = train_and_save_models(df, feature_cols, test_size=args.test_size)

        logger.info("\n🎉 Training complete! Models are ready to use.")
        return 0

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
