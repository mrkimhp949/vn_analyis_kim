"""
Data loader sử dụng TCBS API thay vì Yahoo Finance
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
import requests
from data_quality import get_quality_checker
from exceptions import DataLoadError, DataQualityError
from rate_limiter import tcbs_limiter

DATA_CACHE_DIR = "data_cache"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

TCBS_API_BASE = "https://apipubaws.tcbs.com.vn"

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
    required_bars: int = 50,
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
                # logger.info(f"📁 Loading {symbol} from cache.")
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Cache load failed for {symbol}: {e}. Refetching...")

    # logger.info(f"📥 Downloading {symbol} from TCBS API...")
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        df = _download_from_tcbs(symbol, start_dt, end_dt, resolution, data_type)
    except Exception as e:
        logger.error(f"Failed to download data for {symbol}: {e}", exc_info=False)
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

    try:
        quality_checker = get_quality_checker()
        quality_results = quality_checker.validate(df, symbol)

        if not quality_results["valid"]:
            if quality_results["issues"]:
                logger.warning(
                    f"Data quality issues for {symbol}: {quality_results['issues']}"
                )
                df = quality_checker.clean_data(df, method="forward_fill")
            if quality_results["warnings"]:
                logger.info(
                    f"Data quality warnings for {symbol}: {quality_results['warnings']}"
                )
    except Exception as e:
        logger.warning(f"Data quality check failed for {symbol}: {e}")

    if len(df) < required_bars:
        logger.warning(
            f"Insufficient data for {symbol} after cleaning: got {len(df)} bars, need {required_bars}"
        )
        return pd.DataFrame()

    if use_cache:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol}: {e}")

    # logger.info(f"✅ Successfully loaded {len(df)} bars for {symbol}")
    return df


def _download_from_tcbs(
    symbol: str,
    start: datetime,
    end: datetime,
    resolution: str,
    data_type: str,
) -> pd.DataFrame:
    """
    Tải dữ liệu từ TCBS API với rate limiting và retry logic
    """
    import time

    max_retries = 3
    retry_delays = [2, 4, 8]  # Exponential backoff: 2s, 4s, 8s

    for attempt in range(max_retries):
        try:
            # Apply rate limiting
            tcbs_limiter.wait()

            url = f"{TCBS_API_BASE}/stock-insight/v1/stock/bars-long-term"

            params = {
                "ticker": symbol,
                "type": data_type,
                "resolution": resolution.upper(),  # API expects uppercase
                "from": int(start.timestamp()),
                "to": int(end.timestamp()),
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                if (
                    response.status_code in [429, 500, 502, 503, 504]
                    and attempt < max_retries - 1
                ):
                    logger.warning(
                        f"⚠️ TCBS API error {response.status_code} for {symbol}, retrying in {retry_delays[attempt]}s..."
                    )
                    time.sleep(retry_delays[attempt])
                    continue
                raise DataLoadError(
                    f"TCBS API returned status {response.status_code}",
                    context={"symbol": symbol, "status_code": response.status_code},
                )

            data = response.json()

            if not isinstance(data, dict) or "data" not in data:
                raise DataLoadError(
                    f"TCBS API returned invalid format",
                    context={
                        "symbol": symbol,
                        "response_keys": (
                            list(data.keys()) if isinstance(data, dict) else None
                        ),
                    },
                )

            bars = data.get("data")

            if not bars:
                df = pd.DataFrame()
                df.source_error = "No data available"  # Attach info for caller
                return df

            df = pd.DataFrame(bars)
            df = df.rename(columns={"tradingDate": "time"})
            df["time"] = pd.to_datetime(df["time"])
            df = df[["time", "open", "high", "low", "close", "volume"]]
            df = df.dropna(subset=["time", "close"])
            df = df[df["close"] > 0]
            df = df.drop_duplicates(subset=["time"], keep="last")

            return df

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"⚠️ Network error for {symbol}, retrying in {retry_delays[attempt]}s..."
                )
                time.sleep(retry_delays[attempt])
                continue
            raise DataLoadError(
                f"Network error after {max_retries} retries", context={"symbol": symbol}
            ) from e

        except Exception as e:
            raise DataLoadError(
                f"Unexpected error during download", context={"symbol": symbol}
            ) from e

    return pd.DataFrame()  # Return empty df if all retries fail


# Test function
if __name__ == "__main__":
    print("Testing TCBS Data Loader...")

    test_symbols = ["VCB", "FPT", "VNM", "HPG", "VNINDEX"]
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    for symbol in test_symbols:
        try:
            data_type = "index" if symbol == "VNINDEX" else "stock"
            df = load_data(
                symbol,
                start_date=start_date,
                end_date=end_date,
                data_type=data_type,
                use_cache=False,
            )
            if not df.empty:
                print(f"✅ {symbol}: {len(df)} rows")
                print(
                    f"   Date range: {df['time'].min().date()} to {df['time'].max().date()}"
                )
                print(f"   Latest close: {df['close'].iloc[-1]:,.0f}")
            else:
                print(f"⚠️ {symbol}: No data returned.")
        except Exception as e:
            print(f"❌ {symbol}: {e}")
