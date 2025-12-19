"""
Data loader sử dụng vnstock library (VCI source)
Hỗ trợ tốt cổ phiếu Việt Nam
"""

import hashlib
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from src.config.exceptions import DataLoadError
from src.utils.rate_limiter import tcbs_limiter

# Import vnstock
try:
    from vnstock import Vnstock

    VNSTOCK_AVAILABLE = True
except ImportError:
    VNSTOCK_AVAILABLE = False

DATA_CACHE_DIR = "data_cache"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ✅ FIX: Ensure UTF-8 output trên mọi platform
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    resolution: str = "1D",
    data_type: str = "stock",
    use_cache: bool = True,
    required_bars: int = 20,  # Giảm từ 50 xuống 20 để tránh warning với lookback ngắn
    lookback: Optional[int] = None,
    is_index: bool = False,
) -> pd.DataFrame:
    """
    Load data from TCBS API with caching and validation.

    Args:
        symbol: Ticker symbol (e.g., 'FPT', 'VNINDEX').
        start_date: Start date in 'YYYY-MM-DD' format (optional if lookback is provided).
        end_date: End date in 'YYYY-MM-DD' format (optional, defaults to today).
        resolution: Data resolution ('1D', '1H', etc.).
        data_type: Type of data ('stock', 'index').
        use_cache: Whether to use cached data.
        required_bars: Minimum number of bars required.
        lookback: Number of days to look back (alternative to start_date).
        is_index: Whether this is an index (alternative to data_type='index').

    Returns:
        A pandas DataFrame with the requested data, or an empty DataFrame if an error occurs.
    """
    # Handle is_index parameter
    if is_index:
        data_type = "index"

    # Handle lookback parameter
    if lookback is not None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")

    # Validate that we have start_date and end_date
    if start_date is None or end_date is None:
        raise ValueError("Either provide start_date/end_date or lookback parameter")
    # Indexes are volatile, don't cache them to always get the latest data
    if data_type == "index":
        use_cache = False

    # Generate a unique cache key based on all parameters
    cache_key = f"{symbol}_{start_date}_{end_date}_{resolution}_{data_type}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = os.path.join(DATA_CACHE_DIR, f"{cache_hash}.pkl")

    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                # logger.info("📁 Loading {symbol} from cache.")
                return pickle.load(f)
        except Exception:
            logger.warning(f"⚠️ Cache load failed for {symbol}. Refetching...")

    # logger.info("📥 Downloading {symbol} from vnstock...")
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        df = _download_from_vnstock(symbol, start_dt, end_dt, resolution, data_type)
    except Exception:
        logger.error(f"Failed to download data for {symbol}", exc_info=False)
        return pd.DataFrame()  # Return empty DataFrame on download failure

    # --- Data Validation and Cleaning ---
    if df.empty:
        # Don't log a warning if the API simply returned no data for the period
        if "No data available" not in getattr(df, "source_error", ""):
            logger.warning(
                f"Insufficient data for {symbol}: got 0 bars, need at least {required_bars}"
            )
        return pd.DataFrame()

    df = df.sort_values("time").reset_index(drop=True)

    # Quality check - validate and clean data
    try:
        from src.data.quality import get_quality_checker

        quality_checker = get_quality_checker()
        quality_report = quality_checker.validate(df, symbol)

        if not quality_report.valid:
            # Log critical issues
            critical_issues = [i for i in quality_report.issues if i.severity == "critical"]
            if critical_issues:
                for issue in critical_issues:
                    logger.warning(f"⚠️ Data quality issue for {symbol}: {issue.description}")
                # Return empty for critical issues
                return pd.DataFrame()

            # Clean data for non-critical issues
            error_issues = [i for i in quality_report.issues if i.severity == "error"]
            if error_issues:
                logger.info(f"🔧 Cleaning {len(error_issues)} data issues for {symbol}")
                df = quality_checker.clean_data(df, method="forward_fill")

        # Log warnings (non-blocking)
        if quality_report.warnings:
            logger.debug(f"Data quality warnings for {symbol}: {quality_report.warnings[:3]}")

        # Log quality score for monitoring
        if quality_report.score < 70:
            logger.info(f"📊 {symbol} data quality score: {quality_report.score:.0f}/100")

    except ImportError:
        logger.debug("Quality module not available, skipping validation")
    except Exception as e:
        logger.debug(f"Data quality check skipped for {symbol}: {e}")

    if len(df) < required_bars:
        logger.warning(
            f"Insufficient data for {symbol} after cleaning: got {len(df)} bars, need {required_bars}"
        )
        return pd.DataFrame()

    if use_cache:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
        except Exception:
            logger.warning(f"Failed to save cache for {symbol}")

    # logger.info("✅ Successfully loaded {len(df)} bars for {symbol}")
    return df


def _download_from_vnstock(
    symbol: str,
    start: datetime,
    end: datetime,
    resolution: str,
    data_type: str,
) -> pd.DataFrame:
    """
    Tải dữ liệu từ vnstock library với VCI source
    """
    import time
    import warnings

    # Suppress vnstock advertising messages and warnings
    warnings.filterwarnings("ignore")

    # Suppress vnstock INFO logs (e.g., "Not a stock" messages)
    logging.getLogger("vnstock").setLevel(logging.WARNING)
    logging.getLogger("vnstock.common.data").setLevel(logging.WARNING)

    if not VNSTOCK_AVAILABLE:
        raise DataLoadError(
            "vnstock library not installed. Run: pip install vnstock",
            context={"symbol": symbol},
        )

    max_retries = 3
    retry_delays = [2, 5, 10]

    for attempt in range(max_retries):
        try:
            # Apply rate limiting
            tcbs_limiter.wait()

            # Format dates for vnstock
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")

            # Determine source based on data type
            if data_type == "index":
                # For index, use VCI source
                stock = Vnstock().stock(symbol=symbol, source="VCI")
            else:
                # For regular stocks, use VCI source
                stock = Vnstock().stock(symbol=symbol, source="VCI")

            # Map resolution to vnstock interval
            interval_map = {
                "D": "1D",
                "1D": "1D",
                "W": "1W",
                "1W": "1W",
                "M": "1M",
                "1M": "1M",
            }
            interval = interval_map.get(resolution.upper(), "1D")

            # Download data
            df = stock.quote.history(start=start_str, end=end_str, interval=interval)

            if df is None or df.empty:
                empty_df = pd.DataFrame()
                empty_df.source_error = "No data available"
                return empty_df

            # Ensure correct column names
            if "time" not in df.columns and "date" in df.columns:
                df = df.rename(columns={"date": "time"})

            # Standardize columns
            required_cols = ["time", "open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    # Try to find similar column
                    for existing_col in df.columns:
                        if col.lower() in existing_col.lower():
                            df = df.rename(columns={existing_col: col})
                            break

            # Convert time column to datetime
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])

            # Select and clean columns
            available_cols = [c for c in required_cols if c in df.columns]
            df = df[available_cols]

            # Clean data
            df = df.dropna(subset=["time", "close"] if "time" in df.columns else ["close"])
            df = df[df["close"] > 0]
            if "time" in df.columns:
                df = df.drop_duplicates(subset=["time"], keep="last")

            return df

        except Exception as e:
            error_msg = str(e).lower()
            # Check if it's a rate limit or network error
            if any(x in error_msg for x in ["rate", "limit", "timeout", "connection", "network"]):
                if attempt < max_retries - 1:
                    logger.warning(
                        f"⚠️ vnstock error for {symbol}, retrying in {retry_delays[attempt]}s..."
                    )
                    time.sleep(retry_delays[attempt])
                    continue

            raise DataLoadError(
                f"vnstock download error: {str(e)[:100]}", context={"symbol": symbol}
            ) from e

    return pd.DataFrame()  # Return empty df if all retries fail


# Test function
if __name__ == "__main__":
    print("Testing vnstock Data Loader...")

    test_symbols = ["VCB", "FPT", "VNM", "HPG"]
    start_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    for symbol in test_symbols:
        try:
            df = load_data(
                symbol,
                start_date=start_date,
                end_date=end_date,
                use_cache=False,
            )
            if not df.empty:
                print(f"✅ {symbol}: {len(df)} rows")
                print(f"   Date range: {df['time'].min().date()} to {df['time'].max().date()}")
                print(f"   Latest close: {df['close'].iloc[-1]:,.0f}")
            else:
                print(f"⚠️ {symbol}: No data returned.")
        except Exception as e:
            print(f"❌ {symbol}: {e}")
