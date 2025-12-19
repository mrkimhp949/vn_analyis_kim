# -*- coding: utf-8 -*-
"""
Train Multimodal Fusion Model

Script để train model MultimodalFusionModel cho dự đoán cổ phiếu Việt Nam.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.nlp.multimodal_fusion import MultimodalPredictor, MultimodalConstants

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_historical_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Load historical data for a symbol.

    Args:
        symbol: Stock symbol (e.g., "VNM", "FPT")
        days: Number of days of historical data

    Returns:
        DataFrame with OHLCV data
    """
    try:
        from src.data.loader import load_data

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        df = load_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True,
        )

        if df is not None and not df.empty:
            logger.info(f"✅ Loaded {len(df)} bars for {symbol}")
            return df
        else:
            logger.warning(f"❌ No data for {symbol}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error loading {symbol}: {e}")
        return pd.DataFrame()


def prepare_training_data(
    symbols: List[str],
    lookback: int = 20,
    lookahead: int = 5,
) -> Tuple[List[Tuple[pd.DataFrame, int]], List[Tuple[pd.DataFrame, int]]]:
    """
    Prepare training and validation data.

    Args:
        symbols: List of stock symbols
        lookback: Number of days to look back for features
        lookahead: Number of days ahead for label

    Returns:
        Tuple of (train_data, val_data)
    """
    all_data = []

    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        df = load_historical_data(symbol, days=365)

        if df.empty or len(df) < lookback + lookahead + 10:
            continue

        # Create samples
        for i in range(lookback, len(df) - lookahead):
            # Get lookback window
            window_df = df.iloc[i - lookback : i].copy()
            window_df["symbol"] = symbol

            # Label: 1 if price goes up, 0 otherwise
            current_price = df.iloc[i]["close"]
            future_price = df.iloc[i + lookahead]["close"]
            label = 1 if future_price > current_price else 0

            all_data.append((window_df, label))

    if not all_data:
        logger.error("No training data available!")
        return [], []

    # Shuffle data
    np.random.shuffle(all_data)

    # Split 80/20 train/val
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    logger.info(f"📊 Training samples: {len(train_data)}")
    logger.info(f"📊 Validation samples: {len(val_data)}")

    return train_data, val_data


def train_model(
    symbols: List[str] = None,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict:
    """
    Train the multimodal fusion model.

    Args:
        symbols: List of stock symbols to train on
        epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Training history
    """
    # Default symbols (VN30)
    if symbols is None:
        symbols = [
            "VNM",
            "FPT",
            "VCB",
            "VIC",
            "VHM",
            "HPG",
            "MWG",
            "MSN",
            "TCB",
            "VPB",
            "ACB",
            "MBB",
            "CTG",
            "BID",
            "STB",
        ]

    logger.info("=" * 60)
    logger.info("🚀 Starting Multimodal Fusion Model Training")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch Size: {batch_size}")

    # Create models directory if not exists
    os.makedirs(MultimodalConstants.MODEL_DIR, exist_ok=True)

    # Prepare data
    logger.info("\n📦 Preparing training data...")
    train_data, val_data = prepare_training_data(symbols)

    if not train_data:
        logger.error("❌ No training data available. Exiting.")
        return {"error": "No training data"}

    # Initialize predictor
    logger.info("\n🔧 Initializing MultimodalPredictor...")
    predictor = MultimodalPredictor()

    # Train
    logger.info("\n🏋️ Training model...")
    history = predictor.train(
        train_data=train_data,
        val_data=val_data if val_data else None,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Save model
    if "error" not in history:
        predictor.save_model()
        logger.info(f"\n✅ Model saved to {MultimodalConstants.FUSION_MODEL_PATH}")

    logger.info("\n" + "=" * 60)
    logger.info("🎉 Training Complete!")
    logger.info("=" * 60)

    return history


def quick_train():
    """Quick training with fewer symbols and epochs for testing."""
    logger.info("🏃 Quick training mode (for testing)")

    symbols = ["VNM", "FPT", "VCB"]  # Just 3 symbols

    return train_model(
        symbols=symbols,
        epochs=10,
        batch_size=16,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Multimodal Fusion Model")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training with fewer data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Stock symbols to train on",
    )

    args = parser.parse_args()

    if args.quick:
        history = quick_train()
    else:
        history = train_model(
            symbols=args.symbols,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    print("\n📈 Training History:")
    print(history)
