"""
Incremental Data Cache System
Cache dữ liệu price intraday và cập nhật incremental thay vì full fetch
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import hashlib
import pickle

INTRADAY_CACHE_DIR = "intraday_cache"
os.makedirs(INTRADAY_CACHE_DIR, exist_ok=True)

CACHE_METADATA_FILE = os.path.join(INTRADAY_CACHE_DIR, "metadata.json")


class IncrementalCache:
    """
    Cache system với incremental updates

    - Lưu cache theo symbol và date
    - Chỉ fetch dữ liệu mới (từ last_update đến hiện tại)
    - Merge với cache cũ
    """

    def __init__(self):
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load metadata về cache state"""
        if os.path.exists(CACHE_METADATA_FILE):
            try:
                with open(CACHE_METADATA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_metadata(self):
        """Lưu metadata"""
        with open(CACHE_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def _get_cache_file(self, symbol: str, date: str) -> str:
        """Lấy đường dẫn cache file cho symbol và date"""
        cache_key = f"{symbol}_{date}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(INTRADAY_CACHE_DIR, f"{cache_hash}.pkl")

    def _get_metadata_key(self, symbol: str) -> str:
        """Lấy key trong metadata"""
        return symbol.upper()

    def get_cached_data(
        self, symbol: str, lookback: int = 200
    ) -> Optional[pd.DataFrame]:
        """
        Lấy dữ liệu từ cache

        Returns:
            DataFrame hoặc None nếu không có cache
        """
        symbol_key = self._get_metadata_key(symbol)

        if symbol_key not in self.metadata:
            return None

        cache_info = self.metadata[symbol_key]
        last_update = datetime.fromisoformat(
            cache_info.get("last_update", "2000-01-01")
        )

        # Kiểm tra cache có còn fresh không (trong vòng 1 ngày)
        if (datetime.now() - last_update).days > 1:
            return None

        # Load từ cache files
        cached_dates = cache_info.get("cached_dates", [])
        if not cached_dates:
            return None

        # Load và merge các cache files
        dfs = []
        for date in sorted(cached_dates)[
            -lookback // 30 :
        ]:  # Load khoảng 30 ngày gần nhất
            cache_file = self._get_cache_file(symbol, date)
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        df = pickle.load(f)
                        if not df.empty:
                            dfs.append(df)
                except Exception:
                    continue

        if not dfs:
            return None

        # Merge và sort
        result = pd.concat(dfs, ignore_index=True)
        result = result.drop_duplicates(subset=["time"], keep="last")
        result = result.sort_values("time")

        # Limit to lookback
        if len(result) > lookback:
            result = result.tail(lookback).reset_index(drop=True)

        return result

    def update_cache(
        self, symbol: str, new_data: pd.DataFrame, incremental: bool = True
    ) -> bool:
        """
        Cập nhật cache với dữ liệu mới

        Args:
            symbol: Mã cổ phiếu
            new_data: DataFrame mới
            incremental: Nếu True, merge với cache cũ; nếu False, replace hoàn toàn

        Returns:
            True nếu thành công
        """
        if new_data.empty:
            return False

        symbol_key = self._get_metadata_key(symbol)
        today = datetime.now().date().isoformat()

        # Lấy dữ liệu cũ nếu incremental
        if incremental:
            cached_df = self.get_cached_data(symbol, lookback=500)
            if cached_df is not None and not cached_df.empty:
                # Merge: loại bỏ duplicates, giữ dữ liệu mới nhất
                combined = pd.concat([cached_df, new_data], ignore_index=True)
                combined = combined.drop_duplicates(subset=["time"], keep="last")
                combined = combined.sort_values("time")
                new_data = combined

        # Lưu theo date
        if "time" in new_data.columns:
            new_data["date"] = pd.to_datetime(new_data["time"]).dt.date.astype(str)
            dates = new_data["date"].unique().tolist()
        else:
            dates = [today]

        # Lưu từng date vào cache file riêng
        for date in dates:
            date_df = (
                new_data[new_data.get("date", "") == date].copy()
                if "date" in new_data.columns
                else new_data
            )
            if not date_df.empty:
                cache_file = self._get_cache_file(symbol, date)
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(date_df, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"⚠️ Lỗi lưu cache {symbol} {date}: {e}")
                    continue

        # Update metadata
        if symbol_key not in self.metadata:
            self.metadata[symbol_key] = {
                "cached_dates": [],
                "last_update": None,
                "total_rows": 0,
            }

        self.metadata[symbol_key]["cached_dates"] = sorted(
            list(set(self.metadata[symbol_key].get("cached_dates", []) + dates))
        )
        self.metadata[symbol_key]["last_update"] = datetime.now().isoformat()
        self.metadata[symbol_key]["total_rows"] = len(new_data)

        self._save_metadata()
        return True

    def get_incremental_range(self, symbol: str) -> Optional[tuple]:
        """
        Lấy khoảng thời gian cần fetch incremental

        Returns:
            (start_timestamp, end_timestamp) hoặc None nếu cần fetch full
        """
        symbol_key = self._get_metadata_key(symbol)

        if symbol_key not in self.metadata:
            return None  # Chưa có cache, cần fetch full

        last_update_str = self.metadata[symbol_key].get("last_update")
        if not last_update_str:
            return None

        last_update = datetime.fromisoformat(last_update_str)

        # Chỉ fetch từ last_update đến hiện tại
        start = last_update
        end = datetime.now()

        # Nếu quá lâu (hơn 7 ngày), fetch full
        if (end - start).days > 7:
            return None

        return (int(start.timestamp()), int(end.timestamp()))

    def clear_old_cache(self, days: int = 30):
        """Xóa cache cũ hơn N ngày"""
        cutoff_date = (datetime.now() - timedelta(days=days)).date()
        cleared = 0

        for symbol_key, cache_info in list(self.metadata.items()):
            cached_dates = cache_info.get("cached_dates", [])
            new_dates = []

            for date_str in cached_dates:
                try:
                    date = datetime.fromisoformat(date_str).date()
                    if date >= cutoff_date:
                        new_dates.append(date_str)
                    else:
                        # Xóa cache file
                        cache_file = self._get_cache_file(symbol_key, date_str)
                        if os.path.exists(cache_file):
                            try:
                                os.remove(cache_file)
                                cleared += 1
                            except Exception:
                                pass
                except Exception:
                    continue

            if new_dates:
                self.metadata[symbol_key]["cached_dates"] = new_dates
            else:
                # Xóa metadata nếu không còn cache
                del self.metadata[symbol_key]

        self._save_metadata()
        if cleared > 0:
            print(f"🧹 Đã xóa {cleared} cache files cũ")


# Global instance
_incremental_cache = None


def get_incremental_cache() -> IncrementalCache:
    """Get or create incremental cache instance"""
    global _incremental_cache
    if _incremental_cache is None:
        _incremental_cache = IncrementalCache()
    return _incremental_cache
