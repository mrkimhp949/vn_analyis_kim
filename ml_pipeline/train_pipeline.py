import argparse
import json
import logging
import os

from config import TICKERS
from features import get_feature_columns
from ml_pipeline.data_manager import DataIngestionConfig, DataManager
from ml_pipeline.model_trainer import EnsembleTrainer, TrainingConfig
from ml_pipeline.volatility_forecaster import VolatilityForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(tickers, lookback, refresh, market_regime=None):
    """
    Run full training pipeline:
    1. Data ingestion
    2. Ensemble model training (RF + GB + LSTM + XGBoost)
    3. Volatility forecasting model training
    4. Regime-specific analysis (if market_regime provided)
    """
    # Validate inputs
    if not tickers or len(tickers) == 0:
        error_msg = "No tickers provided for training!"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Filter out empty/invalid tickers
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        error_msg = "All tickers are invalid!"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Starting training pipeline for {len(tickers)} tickers: {tickers}, lookback={lookback} days")
    
    # Data ingestion
    try:
        ingestion_config = DataIngestionConfig(
            tickers=tickers,
            lookback=lookback,
            feature_store_path="feature_store",
            refresh=refresh,
        )
        manager = DataManager(ingestion_config)
        dataset = manager.ingest_all()
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}", exc_info=True)
        raise
    
    if dataset.empty:
        logger.error("No data ingested!")
        return {"error": "No data"}

    # Ensemble model training
    feature_cols = [col for col in get_feature_columns() if col in dataset.columns]
    logger.info(f"Training ensemble with {len(feature_cols)} features")
    
    trainer = EnsembleTrainer(TrainingConfig(feature_columns=feature_cols))
    metrics = trainer.train(dataset, market_regime=market_regime)
    
    # Volatility forecasting model training
    logger.info("Training volatility forecasting model...")
    vol_forecaster = VolatilityForecaster()
    vol_metrics = vol_forecaster.train(dataset)
    
    # Combine metrics
    all_metrics = {
        "ensemble": metrics,
        "volatility": vol_metrics,
        "market_regime": market_regime,
        "tickers": tickers,
        "lookback_days": lookback,
    }
    
    # Save reports
    os.makedirs("reports", exist_ok=True)
    report_file = "reports/training_report.json"
    if market_regime:
        report_file = f"reports/training_report_{market_regime.lower()}.json"
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training complete! Report saved to {report_file}")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Train ensemble ML pipeline with volatility forecasting.")
    parser.add_argument("--tickers", type=str, default=",".join(TICKERS))
    parser.add_argument("--lookback", type=int, default=600)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--regime", type=str, choices=["BULL", "BEAR", "SIDEWAYS"], 
                       help="Market regime for regime-specific training")
    args = parser.parse_args()
    tickers = [sym.strip().upper() for sym in args.tickers.split(",") if sym.strip()]
    metrics = run_pipeline(tickers, args.lookback, args.refresh, market_regime=args.regime)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

