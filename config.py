"""
DEPRECATED: Use trading_config.py instead
This file is kept for backward compatibility only
"""

import os
import sys
from typing import List
import warnings

warnings.warn(
    "config.py is deprecated. Use 'from trading_config import get_config' instead",
    DeprecationWarning,
    stacklevel=2,
)

# Load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Install: pip install python-dotenv")

# Fix encoding cho Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

# Import from new config
from trading_config import get_config

_config = get_config()

# Backward compatibility exports
TELEGRAM_TOKEN = _config.telegram.token
CHAT_ID = _config.telegram.chat_id
RESOLUTION = "1D"
LOOKBACK = _config.data.lookback
USE_CSV_TICKERS = _config.data.use_csv_tickers
MIN_VOLUME = _config.data.min_volume


def get_tickers() -> List[str]:
    """
    Lấy TẤT CẢ mã cổ phiếu từ List.csv
    """
    # Check env override
    env_tickers = os.getenv("TICKERS")
    if env_tickers:
        tickers = [x.strip().upper() for x in env_tickers.split(",") if x.strip()]
        print(f"📊 Sử dụng {len(tickers)} mã từ env variable")
        return tickers

    # Load TẤT CẢ từ List.csv
    try:
        from ticker_loader import get_ticker_loader

        loader = get_ticker_loader()
        tickers = loader.all_tickers

        if not tickers:
            raise ValueError("No tickers loaded from List.csv")

        print(f"📊 Loaded {len(tickers)} mã từ List.csv")
        return tickers

    except Exception as e:
        print(f"❌ Lỗi load từ List.csv: {e}")
        # Fallback to empty list - user must fix List.csv
        print("⚠️ Không có tickers! Vui lòng kiểm tra List.csv")
        return []


# Lấy danh sách tickers
TICKERS = get_tickers()
