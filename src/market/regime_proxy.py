# market_regime_proxy.py
import logging
import os
import sys
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Optional

import pandas as pd
from data_loader import load_data
from ml_signals import MLSignalGenerator

# Fix encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        os.environ["PYTHONIOENCODING"] = "utf-8"
    except:
        pass


def safe_print(message):
    """Print an toàn"""
    try:
        print(message)
    except UnicodeEncodeError:
        clean_message = "".join(char for char in message if ord(char) < 128)
        print(clean_message)


class ProxyMarketRegimeAnalyzer:
    """
    Lớp Proxy để quản lý và cache kết quả từ MarketRegimeAnalyzer.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        cls.cache = SimpleCache()

        # Tạo instance nếu chưa tồn tại
        if cls._instance is None:
            cls._instance = instance

        return cls._instance

    def __init__(self, analyzer_class=None, **kwargs):
        if not hasattr(self, "initialized"):
            self.initialized = True  # Set this first to prevent re-entry

            # Initialize the actual analyzer
            if analyzer_class:
                self.analyzer = analyzer_class(**kwargs)
            else:
                # Fallback to default if not provided
                from market_regime import MarketRegimeAnalyzer

                # MarketRegimeAnalyzer hiện tại không cần ticker_list, nó tự load VNINDEX
                self.analyzer = MarketRegimeAnalyzer()
                logging.info(f"📊 Khởi tạo Market Regime Analyzer.")

            # Initialize cache
            self.cache = SimpleCache()
            logging.info("✅ Market analyzer initialized")

    def analyze_market_regime(self, vnindex_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Phân tích trạng thái thị trường.
        Sử dụng cache nếu có.
        Có thể nhận vnindex_df đã được tải sẵn để tránh tải lại.
        """
        cache_key = f"market_regime_{datetime.now().strftime('%Y-%m-%d')}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            logging.info("✅ Lấy trạng thái thị trường từ cache.")
            return cached_data

        logging.info("🔧 Phân tích trạng thái thị trường (không có cache)...")
        try:
            if self.analyzer:
                # Nếu analyzer cần vnindex_df, truyền nó vào
                if (
                    "vnindex_df"
                    in self.analyzer.analyze_market_regime.__code__.co_varnames
                ):
                    regime = self.analyzer.analyze_market_regime(vnindex_df=vnindex_df)
                else:
                    # Giữ tương thích với analyzer cũ không cần df
                    regime = self.analyzer.analyze_market_regime()

                self.cache.set(cache_key, regime, timeout=3600)  # Cache 1 giờ
                return regime
            else:
                # Fallback nếu không có analyzer
                return {"regime": "UNKNOWN", "confidence": 0, "tradeable": False}
        except Exception as e:
            logging.error(
                f"Lỗi khi phân tích trạng thái thị trường: {e}", exc_info=True
            )
            return {"regime": "ERROR", "confidence": 0, "tradeable": False}


class SimpleCache:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value, timeout=3600):
        self.cache[key] = value


class MLSignalGenerator:
    def __init__(self):
        pass

    def analyze(self, df):
        # Placeholder for actual ML logic
        return {"signal": "BUY", "confidence": 0.5}


class Config:
    TICKERS = ["VND", "VNDX", "VNDY", "VNDZ", "VNDW"]


def main():
    analyzer = ProxyMarketRegimeAnalyzer()
    result = analyzer.analyze_market_regime()
    print(result)


if __name__ == "__main__":
    main()
