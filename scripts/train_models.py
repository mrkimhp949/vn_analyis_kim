#!/usr/bin/env python3
"""
Unified ML Model Training Script
Train RF + XGBoost ensemble with enhanced features (28+)

Usage:
    python scripts/train_models.py [OPTIONS]

Examples:
    # Train with quality tickers (default, 200 mã chất lượng cao)
    python scripts/train_models.py

    # Train with specific symbols
    python scripts/train_models.py --symbols VNM,FPT,VCB,HPG

    # Train with custom ticker file
    python scripts/train_models.py --ticker-file my_tickers.txt

    # Train with more historical data
    python scripts/train_models.py --lookback 1000

    # Train with fewer symbols (faster testing)
    python scripts/train_models.py --max-symbols 50
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from src.config.legacy_config import TICKERS
from src.data.loader import load_data
from src.ml.features.enhanced import add_enhanced_features, get_feature_columns
from src.ml.models.predictor import MLPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_training_data(symbols: list, lookback: int = 500, max_symbols: int = 50):
    """
    Load and combine data from multiple symbols

    Args:
        symbols: List of symbols to train
        lookback: Number of days of data
        max_symbols: Maximum number of symbols to use

    Returns:
        Combined DataFrame with features
    """
    logger.info(f"📥 Loading data from {min(len(symbols), max_symbols)} symbols...")

    all_data = []
    index_df = None

    # Load VNINDEX first (needed for enhanced features)
    try:
        logger.info("   Loading VNINDEX for feature engineering...")
        index_df = load_data("VNINDEX", lookback=lookback, is_index=True)
        logger.info(f"   ✅ VNINDEX: {len(index_df)} bars")
    except Exception as e:
        logger.warning(f"   ⚠️ Could not load VNINDEX: {e}")

    # Load symbols
    success_count = 0
    for i, symbol in enumerate(symbols[:max_symbols], 1):
        try:
            logger.info(f"   [{i}/{min(len(symbols), max_symbols)}] Loading {symbol}...")
            df = load_data(symbol, lookback=lookback)

            if df is None or df.empty or len(df) < 100:
                logger.warning(f"      ⚠️ Insufficient data for {symbol}, skipping")
                continue

            # Add enhanced features
            df_with_features = add_enhanced_features(df, index_df=index_df)

            if df_with_features is not None and not df_with_features.empty:
                all_data.append(df_with_features)
                success_count += 1
                logger.info(f"      ✅ {symbol}: {len(df_with_features)} bars")
            else:
                logger.warning(f"      ⚠️ Feature engineering failed for {symbol}")

        except Exception as e:
            logger.warning(f"      ❌ Error loading {symbol}: {e}")
            continue

    if not all_data:
        raise ValueError("No data loaded! Check symbols or data source.")

    # Combine all data
    logger.info(f"\n📊 Combining data from {success_count} symbols...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # Drop NaN in target and features
    feature_cols = get_feature_columns()
    combined_df = combined_df.dropna(subset=feature_cols + ["target"])

    logger.info(f"✅ Total samples: {len(combined_df):,}")
    logger.info(f"✅ Features: {len(feature_cols)}")
    logger.info(f"✅ Positive samples: {(combined_df['target'] == 1).sum():,}")
    logger.info(f"✅ Negative samples: {(combined_df['target'] == 0).sum():,}")

    return combined_df


def train_models(df: pd.DataFrame, test_size: float = 0.2):
    """
    Train RF + XGBoost ensemble models

    Args:
        df: Training data with features and target
        test_size: Test set size (default: 20%)
    """
    logger.info("\n" + "=" * 70)
    logger.info("🎓 TRAINING ML MODELS")
    logger.info("=" * 70 + "\n")

    # Get feature columns
    feature_cols = get_feature_columns()

    # Prepare data
    X = df[feature_cols].values
    y = df["target"].values

    # Split train/test (time-based)
    split_idx = int(len(X) * (1 - test_size))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    logger.info(f"📊 Data split:")
    logger.info(f"   Training: {len(X_train):,} samples")
    logger.info(f"   Testing:  {len(X_test):,} samples")

    # Initialize predictor
    predictor = MLPredictor()

    # Scale features
    logger.info("\n📏 Scaling features...")
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)

    # Train Random Forest
    logger.info("\n🌲 Training Random Forest...")
    predictor.train_random_forest(X_train, y_train)

    # Train XGBoost (if available)
    logger.info("\n🚀 Training XGBoost...")
    try:
        predictor.train_xgboost(X_train, y_train)
    except Exception as e:
        logger.warning(f"⚠️ XGBoost training failed: {e}")
        logger.info("💡 Install XGBoost with: pip install xgboost")

    # Evaluate
    logger.info("\n📊 Evaluating models on test set...")
    predictor.evaluate(X_test, y_test)

    # Save models
    logger.info("\n💾 Saving models...")
    predictor.save_models()

    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info("\n📁 Models saved to: models/")
    logger.info("   - random_forest.pkl")
    logger.info("   - xgboost.pkl (if XGBoost available)")
    logger.info("   - scaler.pkl")
    logger.info("   - model_info.json")
    logger.info("   - feature_importance.csv")

    return predictor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train ML models (RF + XGBoost ensemble)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with quality tickers (default, ~200 mã chất lượng cao)
  python scripts/train_models.py

  # Train with specific symbols
  python scripts/train_models.py --symbols VNM,FPT,VCB,HPG

  # Train with custom ticker file
  python scripts/train_models.py --ticker-file my_tickers.txt

  # Train with more historical data
  python scripts/train_models.py --lookback 1000

  # Train with fewer symbols (faster testing)
  python scripts/train_models.py --max-symbols 50

  # Generate quality tickers first (run once)
  python scripts/filter_quality_tickers.py
        """
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: quality_tickers.txt or VN30)"
    )

    parser.add_argument(
        "--ticker-file",
        type=str,
        default="quality_tickers.txt",
        help="File containing ticker symbols (one per line, default: quality_tickers.txt)"
    )

    parser.add_argument(
        "--lookback", type=int, default=500, help="Number of days of historical data (default: 500)"
    )

    parser.add_argument(
        "--max-symbols",
        type=int,
        default=200,
        help="Maximum number of symbols to use (default: 200)"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size as fraction (default: 0.2 = 20%%)",
    )

    return parser.parse_args()


def load_symbols_from_file(filepath: str) -> list:
    """
    Load ticker symbols from file (one per line)

    Args:
        filepath: Path to ticker file

    Returns:
        List of ticker symbols
    """
    symbols = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                ticker = line.strip().upper()
                if ticker and not ticker.startswith('#'):  # Skip comments
                    symbols.append(ticker)

        logger.info(f"✅ Loaded {len(symbols)} symbols from {filepath}")
        return symbols

    except FileNotFoundError:
        logger.warning(f"⚠️ File {filepath} not found")
        return []
    except Exception as e:
        logger.error(f"❌ Error reading {filepath}: {e}")
        return []


def main():
    """Main training pipeline"""
    args = parse_args()

    # Determine symbols (priority: --symbols > --ticker-file > VN30)
    if args.symbols:
        # Option 1: Command line symbols
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        logger.info(f"📊 Using symbols from command line: {len(symbols)} symbols")
    elif args.ticker_file and Path(args.ticker_file).exists():
        # Option 2: Load from ticker file
        symbols = load_symbols_from_file(args.ticker_file)
        if not symbols:
            logger.warning("⚠️ No symbols loaded from file, falling back to VN30")
            symbols = TICKERS[:30]
        else:
            logger.info(f"📊 Using symbols from {args.ticker_file}: {len(symbols)} symbols")
    else:
        # Option 3: Default VN30
        symbols = TICKERS[:30]
        logger.info(f"📊 Using default VN30 symbols: {len(symbols)} symbols")

    logger.info("\n" + "=" * 70)
    logger.info("🚀 ML MODEL TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"\n⚙️ Configuration:")
    logger.info(f"   Symbols: {len(symbols)} ({', '.join(symbols[:5])}...)")
    logger.info(f"   Lookback: {args.lookback} days")
    logger.info(f"   Max symbols: {args.max_symbols}")
    logger.info(f"   Test size: {args.test_size * 100:.0f}%")

    try:
        # Load data
        df = load_training_data(
            symbols=symbols, lookback=args.lookback, max_symbols=args.max_symbols
        )

        # Train models
        predictor = train_models(df, test_size=args.test_size)

        logger.info("\n🎉 SUCCESS! Models are ready to use.")
        logger.info("\n📝 Next steps:")
        logger.info("   1. Run backtests: python scripts/run_backtest.py")
        logger.info("   2. Start live trading: python main.py")

        return 0

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
