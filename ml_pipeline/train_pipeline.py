import argparse
import json
import logging
import os

from features import get_feature_columns

from config import TICKERS
from ml_pipeline.data_manager import DataIngestionConfig, DataManager
from ml_pipeline.feature_selection import select_features_with_shap
from ml_pipeline.model_trainer import EnsembleTrainer, TrainingConfig
from ml_pipeline.volatility_forecaster import VolatilityForecaster

try:
    from model_monitor import get_model_monitor
except ImportError:  # pragma: no cover - monitoring optional
    get_model_monitor = None

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

    logger.info(
        f"Starting training pipeline for {len(tickers)} tickers: {tickers}, lookback={lookback} days"
    )

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
    except Exception:
        logger.error("Data ingestion failed", exc_info=True)
        raise

    if dataset.empty:
        logger.error("No data ingested!")
        return {"error": "No data"}

    # Ensemble model training
    feature_cols = [col for col in get_feature_columns() if col in dataset.columns]
    logger.info(f"Initial feature count: {len(feature_cols)}")

    selected_features, shap_importance = select_features_with_shap(
        dataset,
        feature_columns=feature_cols,
        target_column="target",
        max_samples=min(3000, len(dataset)),
        correlation_threshold=0.92,
    )
    if selected_features and len(selected_features) >= 5:
        logger.info(
            "Using SHAP-selected features (%d -> %d)",
            len(feature_cols),
            len(selected_features),
        )
        feature_cols = selected_features
    else:
        logger.info("SHAP selection skipped; using original feature set.")

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
        "feature_columns": feature_cols,
    }
    if shap_importance is not None:
        all_metrics["shap_feature_importance"] = shap_importance.to_dict(orient="records")

    # Record model version & drift monitoring
    if get_model_monitor:
        try:
            monitor = get_model_monitor()
            record = monitor.record_training_run(
                model_name="ensemble_classifier",
                metrics=metrics,
                metadata={
                    "tickers": tickers,
                    "lookback": lookback,
                    "market_regime": market_regime,
                },
            )
            drift_info = monitor.check_drift("ensemble_classifier", metric_key="accuracy")
            all_metrics["model_monitor"] = {
                "version": record.get("version"),
                "drift": drift_info,
            }
        except Exception:
            logger.warning("Model monitor failed")

    # Save reports
    os.makedirs("reports", exist_ok=True)
    report_file = "reports/training_report.json"
    if market_regime:
        report_file = f"reports/training_report_{market_regime.lower()}.json"

    if shap_importance is not None:
        shap_report = "reports/shap_feature_importance.json"
        shap_importance.to_json(shap_report, orient="records", force_ascii=False, indent=2)
        logger.info(f"Saved SHAP feature importance to {shap_report}")

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Training complete! Report saved to {report_file}")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train ensemble ML pipeline with volatility forecasting."
    )
    parser.add_argument("--tickers", type=str, default=",".join(TICKERS))
    parser.add_argument("--lookback", type=int, default=600)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--regime",
        type=str,
        choices=["BULL", "BEAR", "SIDEWAYS"],
        help="Market regime for regime-specific training",
    )
    args = parser.parse_args()
    tickers = [sym.strip().upper() for sym in args.tickers.split(",") if sym.strip()]
    metrics = run_pipeline(tickers, args.lookback, args.refresh, market_regime=args.regime)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
