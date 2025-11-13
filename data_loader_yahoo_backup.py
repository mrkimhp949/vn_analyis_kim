import os
import sys
import pickle
import hashlib
import importlib
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Suppress yfinance ERROR logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

DATA_CACHE_DIR = 'data_cache'
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ✅ FIX: Ensure UTF-8 output trên mọi platform
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def load_data(symbol, lookback=200, use_cache=True, use_incremental=True):
    """
    Load dữ liệu từ TCBS API với cache và incremental updates
    
    IMPROVEMENTS:
    - ✅ Better column handling
    - ✅ Validation for required data
    - ✅ Clear error messages
    - ✅ Type checking
    - ✅ Incremental cache updates
    - ✅ API monitoring
    """
    # ===== STEP 1: Kiểm tra incremental cache =====
    if use_incremental:
        try:
            from incremental_cache import get_incremental_cache
            inc_cache = get_incremental_cache()
            cached_df = inc_cache.get_cached_data(symbol, lookback)
            
            if cached_df is not None and not cached_df.empty:
                # Kiểm tra xem có cần update không (trong vòng 1 giờ)
                symbol_key = symbol.upper()
                cache_metadata = inc_cache.metadata.get(symbol_key, {})
                last_update_str = cache_metadata.get('last_update')
                
                if last_update_str:
                    last_update = datetime.fromisoformat(last_update_str)
                    if (datetime.now() - last_update).total_seconds() < 3600:  # < 1 giờ
                        print(f"📁 Load {symbol} từ incremental cache ({len(cached_df)} rows)")
                        return cached_df
        except Exception as e:
            print(f"⚠️ Lỗi incremental cache: {e}, fallback to normal cache...")
    
    # ===== STEP 2: Kiểm tra normal cache =====
    cache_key = f"{symbol}_{lookback}_{datetime.today().strftime('%Y%m%d')}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = os.path.join(DATA_CACHE_DIR, f"{cache_hash}.pkl")
    
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                print(f"📁 Load {symbol} từ cache")
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Lỗi load cache: {e}, tải lại từ API...")
    
    # ===== STEP 3: Load từ Yahoo Finance =====
    incremental_range: Optional[Tuple[int, int]] = None
    if use_incremental:
        try:
            from incremental_cache import get_incremental_cache
            inc_cache = get_incremental_cache()
            incremental_range = inc_cache.get_incremental_range(symbol)
        except Exception:
            incremental_range = None

    end_dt = datetime.utcnow()
    if incremental_range:
        start_dt = datetime.fromtimestamp(incremental_range[0]) - timedelta(days=3)
        # print(f"📥 Tải {symbol} (incremental) từ Yahoo Finance...")
    else:
        start_dt = end_dt - timedelta(days=lookback * 3)
        # print(f"📥 Tải {symbol} (full) từ Yahoo Finance...")

    df = _download_from_yahoo(symbol, start_dt, end_dt)

    # ✅ IMPROVED: Normalize column names với validation
    df = _normalize_columns(df, symbol)

    # ✅ VALIDATION: Check required columns exist
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột quan trọng: {missing_cols}")

    # ✅ DATA QUALITY: Remove invalid rows
    df = _clean_data(df, symbol)

    # ✅ VALIDATION: Check if we have enough data
    if len(df) < min(50, lookback * 0.3):
        raise ValueError(
            f"Dữ liệu không đủ cho {symbol}: có {len(df)} nến, "
            f"cần ít nhất {min(50, lookback * 0.3)}"
        )

    # Sort và limit
    df = df.sort_values('time')
    df = df.tail(lookback)
    df = df.reset_index(drop=True)

    print(f"✅ Tải thành công {len(df)} nến cho {symbol}")

    # ===== STEP 4: Lưu incremental cache =====
    if use_incremental:
        try:
            from incremental_cache import get_incremental_cache
            inc_cache = get_incremental_cache()
            inc_cache.update_cache(symbol, df, incremental=True)
            print(f"💾 Đã cập nhật incremental cache cho {symbol}")
        except Exception as e:
            print(f"⚠️ Lỗi lưu incremental cache: {e}")

    # Lưu normal cache
    _save_cache(df, cache_file, symbol)

    return df



def _normalize_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    ✅ IMPROVED: Chuẩn hóa tên cột với validation
    
    Xử lý các format khác nhau:
    - tradingDate (string date)
    - t (unix timestamp)
    - time (có sẵn)
    """
    # TIME COLUMN
    if 'tradingDate' in df.columns:
        print(f"  🔄 Chuyển đổi 'tradingDate' → 'time'")
        df = df.rename(columns={'tradingDate': 'time'})
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
    elif 't' in df.columns:
        print(f"  🔄 Chuyển đổi 't' → 'time'")
        df = df.rename(columns={'t': 'time'})
        
        # ✅ CHECK: Is it Unix timestamp or string?
        sample_value = df['time'].iloc[0]
        
        if isinstance(sample_value, (int, float)):
            # Unix timestamp
            print(f"  📊 Phát hiện Unix timestamp: {sample_value}")
            df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
        else:
            # String date
            print(f"  📊 Phát hiện string date: {sample_value}")
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    elif 'time' in df.columns:
        # Already has time column, just ensure it's datetime
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    else:
        raise ValueError(
            f"Không tìm thấy cột thời gian trong dữ liệu {symbol}. "
            f"Các cột có: {df.columns.tolist()}"
        )
    
    # PRICE & VOLUME COLUMNS
    # Map common variations to standard names
    column_mapping = {
        'o': 'open',
        'h': 'high', 
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'vol': 'volume'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            print(f"  🔄 Chuyển đổi '{old_name}' → '{new_name}'")
            df = df.rename(columns={old_name: new_name})
    
    # ✅ ENSURE all required columns exist
    required_price_cols = ['open', 'high', 'low', 'close']
    for col in required_price_cols:
        if col not in df.columns:
            print(f"  ⚠️ Thiếu cột '{col}' - điền giá trị 0")
            df[col] = 0
    
    if 'volume' not in df.columns:
        print(f"  ⚠️ Thiếu cột 'volume' - điền giá trị 0")
        df['volume'] = 0
    
    # Select only required columns
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    return df


def _download_from_yahoo(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Tải dữ liệu daily từ Yahoo Finance.
    """
    try:
        yf = importlib.import_module("yfinance")
    except ImportError as exc:
        raise ImportError("yfinance chưa được cài đặt. Vui lòng chạy: pip install yfinance") from exc
    # Yahoo cần thêm buffer để tránh missing data
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end_buffered = end + timedelta(days=1)

    try:
        df = yf.download(
            symbol,
            start=start,
            end=end_buffered,
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:
        error_msg = str(e)
        if "No timezone found" in error_msg or "delisted" in error_msg.lower():
            raise ValueError(f"Mã {symbol} có thể đã bị hủy niêm yết hoặc không tồn tại trên Yahoo Finance")
        raise ValueError(f"Lỗi tải dữ liệu {symbol} từ Yahoo Finance: {error_msg}")

    if df.empty:
        raise ValueError(f"Yahoo Finance không trả dữ liệu cho {symbol}")

    df = df.reset_index()
    # Thống nhất cột tên
    rename_map = {
        'Date': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume',
    }
    df = df.rename(columns=rename_map)

    # Giữ lại các cột cần thiết; lưu adj_close nếu có
    keep_cols = [col for col in ['time', 'open', 'high', 'low', 'close', 'adj_close', 'volume'] if col in df.columns]
    df = df[keep_cols]

    # Điền volume thiếu bằng 0
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0)

    return df


def _clean_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    ✅ NEW: Làm sạch dữ liệu
    
    - Remove NaN trong time, close (critical columns)
    - Remove zero/negative prices
    - Remove duplicate timestamps
    """
    original_len = len(df)
    
    # Remove NaN trong critical columns
    df = df.dropna(subset=['time', 'close'])
    
    # Remove invalid prices
    df = df[df['close'] > 0]
    df = df[df['high'] >= df['low']]
    df = df[df['high'] >= df['close']]
    df = df[df['low'] <= df['close']]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['time'], keep='last')
    
    removed = original_len - len(df)
    if removed > 0:
        print(f"  🧹 Đã xóa {removed} dòng dữ liệu không hợp lệ")
    
    return df


def _save_cache(df: pd.DataFrame, cache_file: str, symbol: str):
    """
    ✅ IMPROVED: Lưu cache với error handling
    """
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"💾 Đã lưu cache {symbol}")
    except Exception as e:
        print(f"⚠️ Không thể lưu cache cho {symbol}: {e}")
        # Continue anyway - caching is optional


# ✅ NEW: Clear old cache files
def clear_old_cache(days=7):
    """
    Xóa cache cũ hơn N ngày để tiết kiệm disk space
    """
    if not os.path.exists(DATA_CACHE_DIR):
        return
    
    cutoff_time = datetime.now().timestamp() - (days * 86400)
    cleared = 0
    
    for filename in os.listdir(DATA_CACHE_DIR):
        filepath = os.path.join(DATA_CACHE_DIR, filename)
        if os.path.isfile(filepath):
            if os.path.getmtime(filepath) < cutoff_time:
                try:
                    os.remove(filepath)
                    cleared += 1
                except Exception as e:
                    print(f"⚠️ Không thể xóa {filename}: {e}")
    
    if cleared > 0:
        print(f"🧹 Đã xóa {cleared} file cache cũ")


# ✅ NEW: Test function
def test_data_loader():
    """
    Test data loader với các trường hợp khác nhau
    """
    test_symbols = ['AAPL', 'MSFT', 'SPY']
    
    print("\n" + "="*60)
    print("🧪 TESTING DATA LOADER")
    print("="*60 + "\n")
    
    for symbol in test_symbols:
        try:
            df = load_data(symbol, lookback=100, use_cache=False)
            print(f"✅ {symbol}: {len(df)} rows")
            print(f"   Columns: {df.columns.tolist()}")
            print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
            print()
        except Exception as e:
            print(f"❌ {symbol}: {e}\n")


if __name__ == "__main__":
    # Run tests
    test_data_loader()
    
    # Clear old cache
    clear_old_cache(days=7)