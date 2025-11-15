# -*- coding: utf-8 -*-
"""
Script để train Enhanced ML Models
Chạy: python train_enhanced_models.py
"""

import numpy as np
import pandas as pd
import logging
from typing import List
from sklearn.model_selection import train_test_split

from data_loader import load_data
from features_enhanced import add_enhanced_features, get_feature_columns
from ml_models_enhanced import EnhancedMLPredictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_training_data(
    symbols: List[str], lookback: int = 500, max_symbols: int = 50
) -> pd.DataFrame:
    """
    Load và combine data từ nhiều symbols

    Args:
        symbols: List symbols để train
        lookback: Số ngày data
        max_symbols: Max số symbols để train

    Returns:
        Combined DataFrame
    """
    logger.info(f"📥 Loading data from {min(len(symbols), max_symbols)} symbols...")

    all_data = []
    index_df = None

    # Load VNINDEX first
    try:
        logger.info("   Loading VNINDEX...")
        index_df = load_data("VNINDEX", lookback=lookback, is_index=True)
        logger.info(f"   ✅ VNINDEX: {len(index_df)} candles")
    except Exception as e:
        logger.warning(f"   ⚠️ Could not load VNINDEX: {e}")

    # Load symbols
    for i, symbol in enumerate(symbols[:max_symbols], 1):
        try:
            logger.info(
                f"   [{i}/{min(len(symbols), max_symbols)}] Loading {symbol}..."
            )
            df = load_data(symbol, lookback=lookback)

            if df.empty or len(df) < 100:
                logger.warning(f"      ⚠️ {symbol}: Not enough data")
                continue

            # Add features
            df_with_features = add_enhanced_features(df, index_df)

            # Add symbol column
            df_with_features["symbol"] = symbol

            all_data.append(df_with_features)
            logger.info(f"      ✅ {symbol}: {len(df_with_features)} candles")

        except Exception as e:
            logger.error(f"      ❌ {symbol}: {e}")
            continue

    if not all_data:
        raise ValueError("No data loaded!")

    # Combine
    logger.info(f"\n📊 Combining data from {len(all_data)} symbols...")
    combined_df = pd.concat(all_data, ignore_index=True)

    logger.info(f"✅ Total: {len(combined_df):,} samples")

    return combined_df


def prepare_training_data(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1
):
    """
    Prepare data for training

    Args:
        df: DataFrame with features
        test_size: Test set size
        val_size: Validation set size

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("\n📊 Preparing training data...")

    # Get feature columns
    feature_cols = get_feature_columns()

    # Check if all features exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Extract features and target
    X = df[feature_cols].values
    y = df["target"].values

    # Remove NaN rows
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    logger.info(f"   Features: {X.shape[1]}")
    logger.info(f"   Samples: {len(X):,}")
    logger.info(f"   Class distribution: {np.bincount(y.astype(int))}")

    # Split train/temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(test_size + val_size),
        shuffle=False,  # Time series - no shuffle
    )

    # Split temp into val/test
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), shuffle=False
    )

    logger.info(f"   Train: {len(X_train):,} samples")
    logger.info(f"   Val:   {len(X_val):,} samples")
    logger.info(f"   Test:  {len(X_test):,} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_models(
    symbols: List[str],
    lookback: int = 500,
    max_symbols: int = 50,
    tune_hyperparameters: bool = False,
):
    """
    Main training function

    Args:
        symbols: List symbols to train on
        lookback: Days of data
        max_symbols: Max symbols to use
        tune_hyperparameters: Run hyperparameter tuning
    """
    print("\n" + "=" * 70)
    print("🎓 TRAINING ENHANCED ML MODELS")
    print("=" * 70 + "\n")

    # 1. Load data
    df = load_training_data(symbols, lookback, max_symbols)

    # 2. Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data(df)

    # 3. Train models
    logger.info("\n🚀 Training models...")
    predictor = EnhancedMLPredictor()

    predictor.train_all_models(
        X_train, y_train, X_val, y_val, tune_hyperparameters=tune_hyperparameters
    )

    # 4. Final evaluation on test set
    logger.info("\n📊 Final evaluation on test set...")
    predictions = predictor.predict(X_test, use_ensemble=True)
    y_pred = (predictions > 0.5).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix

    print("\n" + "=" * 70)
    print("📊 TEST SET PERFORMANCE")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=["Down/Hold", "Up"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 5. Feature importance
    if predictor.feature_importance:
        print("\n" + "=" * 70)
        print("📊 TOP 15 FEATURES")
        print("=" * 70)

        avg_importance = predictor.feature_importance.get("average", {})
        sorted_features = sorted(
            avg_importance.items(), key=lambda x: x[1], reverse=True
        )

        for i, (feature, importance) in enumerate(sorted_features[:15], 1):
            print(f"{i:2d}. {feature:25s}: {importance:.4f}")

    # 6. Explain sample prediction
    if len(X_test) > 0:
        logger.info("\n🔍 Explaining sample prediction...")
        explanation = predictor.explain_prediction(X_test, sample_idx=-1)

        if explanation:
            print("\n" + "=" * 70)
            print("🔍 SAMPLE PREDICTION EXPLANATION")
            print("=" * 70)
            print(f"Prediction: {explanation['prediction']:.4f}")
            print(f"Base value: {explanation['base_value']:.4f}")
            print("\nTop 5 contributing features:")
            for feature, shap_value in explanation["top_features"]:
                print(f"   {feature:25s}: {shap_value:+.4f}")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModels saved to 'models/' directory:")
    print("   - rf_enhanced.pkl")
    print("   - xgb_enhanced.pkl (if XGBoost installed)")
    print("   - lgb_enhanced.pkl (if LightGBM installed)")
    print("   - scaler_enhanced.pkl")
    print("   - feature_importance.pkl")
    print("\nYou can now use these models in your trading bot!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Enhanced ML Models")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: load from config)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=500,
        help="Days of historical data (default: 500)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=50,
        help="Maximum number of symbols to train on (default: 50)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning (slower but better results)",
    )

    args = parser.parse_args()

    # Get symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        try:
            from config import TICKERS

            symbols = TICKERS
        except ImportError:
            symbols = ["VNM", "VCB", "FPT", "HPG", "VHM", "GAS", "MSN", "MWG"]

    # Train
    train_models(
        symbols=symbols,
        lookback=args.lookback,
        max_symbols=args.max_symbols,
        tune_hyperparameters=args.tune,
    )
