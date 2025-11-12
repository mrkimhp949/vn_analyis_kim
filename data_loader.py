import pandas as pd
from datetime import datetime, timedelta
import requests
import hashlib
import pickle
import os
import sys

DATA_CACHE_DIR = 'data_cache'
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ✅ FIX: Ensure UTF-8 output trên mọi platform
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def load_data(symbol, lookback=200, use_cache=True):
    """
    Load dữ liệu từ TCBS API với cache
    
    IMPROVEMENTS:
    - ✅ Better column handling
    - ✅ Validation for required data
    - ✅ Clear error messages
    - ✅ Type checking
    """
    # Tạo cache key
    cache_key = f"{symbol}_{lookback}_{datetime.today().strftime('%Y%m%d')}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = os.path.join(DATA_CACHE_DIR, f"{cache_hash}.pkl")
    
    # Kiểm tra cache
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                print(f"📁 Load {symbol} từ cache")
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Lỗi load cache: {e}, tải lại từ API...")
    
    # Load từ API
    end = datetime.today()
    start = end - timedelta(days=lookback*2)
    
    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    params = {
        "ticker": symbol,
        "type": "stock",
        "resolution": "D",
        "from": int(start.timestamp()),
        "to": int(end.timestamp())
    }
    
    try:
        print(f"📥 Tải {symbol} từ TCBS...")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # ✅ VALIDATION: Check response structure
        if 'data' not in data:
            raise ValueError(f"API response thiếu field 'data' cho {symbol}")
        
        if not data['data'] or len(data['data']) == 0:
            raise ValueError(f"Không có dữ liệu trong response cho {symbol}")
        
        # Tạo DataFrame từ response
        df = pd.DataFrame(data['data'])
        
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
        
        # Lưu cache
        _save_cache(df, cache_file, symbol)
        
        return df
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Lỗi kết nối API cho {symbol}: {e}")
    except ValueError as e:
        raise Exception(f"Lỗi dữ liệu cho {symbol}: {e}")
    except Exception as e:
        raise Exception(f"Lỗi không xác định khi tải {symbol}: {e}")


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
        df[col] = 0
    
    # Select only required columns
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
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
    test_symbols = ['VNM', 'VCB', 'HPG']
    
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