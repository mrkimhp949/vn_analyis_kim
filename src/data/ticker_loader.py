"""
Ticker Loader - Load và validate tickers từ List.csv
Validates tickers trước khi scan để tránh mã đã hủy niêm yết
"""

import json
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd


class TickerLoader:
    """Load và validate tất cả tickers từ List.csv"""

    def __init__(self, csv_file="List.csv", cache_file="ticker_validation_cache.json"):
        self.csv_file = csv_file
        self.cache_file = cache_file
        self.all_tickers = []
        self.validated_tickers = []
        self.invalid_tickers = []
        self.validation_cache = self._load_cache()

        self.load_from_csv()

    def _load_cache(self) -> Dict:
        """Load validation cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"validated": {}, "invalid": {}, "last_updated": None}

    def _save_cache(self):
        """Save validation cache"""
        self.validation_cache["last_updated"] = datetime.now().isoformat()
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.validation_cache, f, indent=2, ensure_ascii=False)

    def _is_cache_valid(self, symbol: str, max_age_days: int = 7) -> bool:
        """Check xem cache còn valid không"""
        if symbol in self.validation_cache["validated"]:
            cached_date = self.validation_cache["validated"][symbol].get("date")
            if cached_date:
                try:
                    cache_time = datetime.fromisoformat(cached_date)
                    age = (datetime.now() - cache_time).days
                    return age < max_age_days
                except (ValueError, TypeError):
                    pass
        return False

    def load_from_csv(self):
        """Load tất cả tickers từ CSV"""
        if not os.path.exists(self.csv_file):
            print(f"⚠️ File {self.csv_file} không tồn tại")
            return

        try:
            # Read CSV with error handling
            df = pd.read_csv(
                self.csv_file,
                encoding="utf-8",
                on_bad_lines="skip",  # Skip bad lines
                engine="python",  # More flexible parser
            )

            # Get ticker column (first column)
            self.all_tickers = df.iloc[:, 0].tolist()

            # Clean tickers
            self.all_tickers = [
                str(t).strip().upper() for t in self.all_tickers if pd.notna(t)
            ]

            # Remove empty strings
            self.all_tickers = [t for t in self.all_tickers if t]

            print(f"Loaded {len(self.all_tickers)} tickers from {self.csv_file}")

        except Exception:
            print(f"Error loading {self.csv_file}")
            self.all_tickers = []

    def validate_ticker(self, symbol: str, min_volume: int = 100_000) -> bool:
        """
        Validate một ticker

        Returns:
            True nếu valid, False nếu invalid
        """
        # Check cache first
        if self._is_cache_valid(symbol):
            return True

        if symbol in self.validation_cache["invalid"]:
            return False

        try:
            from src.data.loader import load_data

            # Try load data, requiring only 2 bars for a basic validity check
            # (Some newly listed stocks may have very limited data)
            df = load_data(symbol, lookback=30, use_cache=False, required_bars=2)

            if df.empty or len(df) < 2:
                self.validation_cache["invalid"][symbol] = {
                    "reason": "No data",
                    "date": datetime.now().isoformat(),
                }
                self._save_cache()
                return False

            # Check volume
            avg_volume = df["volume"].mean()
            if avg_volume < min_volume:
                self.validation_cache["invalid"][symbol] = {
                    "reason": f"Low volume ({avg_volume:,.0f})",
                    "date": datetime.now().isoformat(),
                }
                self._save_cache()
                return False

            # Valid - cache it
            self.validation_cache["validated"][symbol] = {
                "date": datetime.now().isoformat(),
                "avg_volume": float(avg_volume),
            }
            self._save_cache()
            return True

        except ValueError as e:
            error_msg = str(e)
            if (
                "hủy niêm yết" in error_msg
                or "không tồn tại" in error_msg
                or "không trả dữ liệu" in error_msg
            ):
                self.validation_cache["invalid"][symbol] = {
                    "reason": "Delisted or not found",
                    "date": datetime.now().isoformat(),
                }
                self._save_cache()
                return False
            return False
        except Exception:
            return False

    def get_validated_tickers(
        self,
        force_validate: bool = False,
        min_volume: int = 100_000,
        max_tickers: int = None,
    ) -> List[str]:
        """
        Lấy danh sách tickers đã validate

        Args:
            force_validate: Force validate lại tất cả
            min_volume: Volume tối thiểu
            max_tickers: Giới hạn số lượng tickers

        Returns:
            List validated tickers
        """
        # Check if we need to re-validate due to changed parameters
        cache_key = f"min_volume_{min_volume}_max_tickers_{max_tickers}"
        last_cache_key = getattr(self, "_last_cache_key", None)

        if (
            not force_validate
            and self.validated_tickers
            and cache_key == last_cache_key
        ):
            return (
                self.validated_tickers[:max_tickers]
                if max_tickers
                else self.validated_tickers
            )

        # Store current cache key for next time
        self._last_cache_key = cache_key

        print(f"🔍 Validating {len(self.all_tickers)} tickers...")

        validated = []
        invalid = []

        total_tickers = len(self.all_tickers)
        for i, symbol in enumerate(self.all_tickers):
            # Show progress
            progress = (i + 1) / total_tickers
            print(
                f"\r  Validating... [{i+1}/{total_tickers}, {progress:.1%}]",
                end="",
                flush=True,
            )

            if self.validate_ticker(symbol, min_volume):
                validated.append(symbol)
            else:
                invalid.append(symbol)

            # Limit if needed
            if max_tickers and len(validated) >= max_tickers:
                break

        print()  # Newline after progress bar finishes

        self.validated_tickers = validated
        self.invalid_tickers = invalid

        print(f"✅ Validated: {len(validated)} tickers")
        print(f"❌ Invalid: {len(invalid)} tickers")

        return validated

    def clear_validation_cache(self):
        """Clear validation cache để force validate lại"""
        self.validated_tickers = []
        self.invalid_tickers = []
        self._last_cache_key = None
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("🗑️ Cleared validation cache")


# Singleton instance
_loader = None


def get_ticker_loader() -> TickerLoader:
    """Get singleton instance"""
    global _loader
    if _loader is None:
        _loader = TickerLoader()
    return _loader


def run_sector_analysis():
    """
    Hàm giả lập cho `run_sector_analysis` để tương thích ngược với `main.py`.
    Thực chất chỉ lấy các mã đã được validate từ TickerLoader.
    """
    print("📊 [Compatibility] Running sector analysis (loading validated tickers)...")
    try:
        loader = get_ticker_loader()
        # Lấy các mã đã validate với các tiêu chí cơ bản
        tickers = loader.get_validated_tickers(
            force_validate=False,  # Dùng cache nếu có
            min_volume=100_000,
            max_tickers=500,  # Lấy nhiều hơn một chút để có lựa chọn
        )
        print(
            f"✅ [Compatibility] Loaded {len(tickers)} validated tickers for scanning."
        )
        return tickers
    except Exception:
        print("❌ [Compatibility] Error in fake run_sector_analysis")
        # Fallback về danh sách tickers đầy đủ nếu có lỗi
        return get_ticker_loader().all_tickers


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING TICKER LOADER")
    print("=" * 60)

    loader = TickerLoader()

    print(f"\n📊 Total tickers: {len(loader.all_tickers)}")
    print(f"📋 Sample (first 20): {loader.all_tickers[:20]}")
    print(f"📋 Sample (last 20): {loader.all_tickers[-20:]}")

    print("\n✅ Test completed!")
