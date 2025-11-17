import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd
from src.data.loader import load_data
from src.ml.features.technical import add_ml_features

logger = logging.getLogger(__name__)


@dataclass
class DataIngestionConfig:
    tickers: List[str]
    lookback: int = 600
    feature_store_path: str = "feature_store"
    refresh: bool = False


class DataManager:
    """
    Quản lý việc lấy dữ liệu, tạo features và lưu vào feature store (parquet).
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        os.makedirs(self.config.feature_store_path, exist_ok=True)

    def _feature_file(self, symbol: str) -> str:
        return os.path.join(self.config.feature_store_path, f"{symbol}.parquet")

    def ingest_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        feature_path = self._feature_file(symbol)
        if os.path.exists(feature_path) and not self.config.refresh:
            try:
                logger.info(f"Loading cached features for {symbol}")
                return pd.read_parquet(feature_path)
            except Exception:
                logger.warning(f"Failed to load cached features for {symbol}, will reload")

        try:
            logger.info(f"Loading data for {symbol} (lookback={self.config.lookback} days)")
            raw = load_data(symbol, lookback=self.config.lookback, use_cache=True)
            if raw.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            logger.info(f"Adding ML features for {symbol} ({len(raw)} rows)")
            feat = add_ml_features(raw)
            if feat.empty:
                logger.warning(f"Features are empty for {symbol} after processing")
                return None

            # Remove last row where target is NaN (future price unknown)
            if "target" in feat.columns:
                feat = feat.dropna(subset=["target"])
                if feat.empty:
                    logger.warning(f"All rows have NaN target for {symbol}")
                    return None

            feat["symbol"] = symbol
            feat["ingested_at"] = datetime.utcnow()

            logger.info(f"Saving features for {symbol} to {feature_path}")
            feat.to_parquet(feature_path, index=False)
            logger.info(f"✅ Successfully ingested {symbol} ({len(feat)} rows)")
            return feat
        except Exception:
            logger.error(f"❌ Failed to ingest {symbol}", exc_info=True)
            return None

    def ingest_all(self) -> pd.DataFrame:
        frames = []
        failed_tickers = []

        logger.info(
            f"Starting ingestion for {len(self.config.tickers)} tickers: {self.config.tickers}"
        )

        for ticker in self.config.tickers:
            df = self.ingest_symbol(ticker)
            if df is not None and not df.empty:
                frames.append(df)
                logger.info(f"✅ {ticker}: {len(df)} rows ingested")
            else:
                failed_tickers.append(ticker)
                logger.warning(f"⚠️ {ticker}: Failed to ingest")

        if not frames:
            error_msg = f"Không thể ingest dữ liệu cho bất kỳ mã nào trong {self.config.tickers}"
            if failed_tickers:
                error_msg += f"\nFailed tickers: {failed_tickers}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Combining {len(frames)} dataframes...")
        combined = pd.concat(frames, ignore_index=True)

        combined_path = os.path.join(self.config.feature_store_path, "combined.parquet")
        combined.to_parquet(combined_path, index=False)
        logger.info(
            f"✅ Combined dataset saved: {len(combined)} rows, {len(combined.columns)} columns"
        )

        if failed_tickers:
            logger.warning(f"⚠️ Some tickers failed: {failed_tickers}")

        return combined


def load_feature_store(path: str = "feature_store/combined.parquet") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy feature store tại {path}")
    return pd.read_parquet(path)
