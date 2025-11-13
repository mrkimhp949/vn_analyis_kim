"""
Data loader sử dụng TCBS API thay vì Yahoo Finance
Hỗ trợ tốt cổ phiếu Việt Nam
"""
import os
import sys
import pickle
import hashlib
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

# Suppress yfinance ERROR logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

DATA_CACHE_DIR = 'data_cache'
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

TCBS_API_BASE = 'https://apipubaws.tcbs.com.vn'

# ✅ FIX: Ensure UTF-8 output trên mọi platform
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def load_data(symbol, lookback=200, use_cache=True):
    """
    Load dữ liệu từ TCBS API với cache
    
    Args:
        symbol: Mã cổ phiếu (VD: VCB, FPT, VNM)
        lookback: Số nến cần lấy
        use_cache: Có dùng cache không
        
    Returns:
        DataFrame với columns: time, open, high, low, close, volume
    """
    # Check cache
    cache_key = f"{symbol}_{lookback}_{datetime.today().strftime('%Y%m%d')}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = os.path.join(DATA_CACHE_DIR, f"{cache_hash}.pkl")
    
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                # print(f"📁 Load {symbol} từ cache")
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Lỗi load cache: {e}, tải lại từ API...")
    
    # Load từ TCBS API
    # print(f"📥 Tải {symbol} từ TCBS API...")
    
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=lookback * 2)  # Buffer
    
    df = _download_from_tcbs(symbol, start_dt, end_dt)
    
    # Validate
    if df.empty or len(df) < min(50, lookback * 0.3):
        raise ValueError(
            f"Dữ liệu không đủ cho {symbol}: có {len(df)} nến, "
            f"cần ít nhất {min(50, lookback * 0.3)}"
        )
    
    # Sort và limit
    df = df.sort_values('time')
    df = df.tail(lookback)
    df = df.reset_index(drop=True)
    
    print(f"✅ Tải thành công {len(df)} nến cho {symbol}")
    
    # Save cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"⚠️ Không thể lưu cache: {e}")
    
    return df


def _download_from_tcbs(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Tải dữ liệu từ TCBS API
    """
    try:
        url = f"{TCBS_API_BASE}/stock-insight/v1/stock/bars-long-term"
        
        params = {
            'ticker': symbol,
            'type': 'stock',
            'resolution': 'D',
            'from': int(start.timestamp()),
            'to': int(end.timestamp())
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise ValueError(f"TCBS API trả về status {response.status_code}")
        
        data = response.json()
        
        if not isinstance(data, dict) or 'data' not in data:
            raise ValueError(f"TCBS API trả về format không đúng")
        
        bars = data['data']
        
        if not bars:
            raise ValueError(f"TCBS không có dữ liệu cho {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(bars)
        
        # Rename columns
        df = df.rename(columns={
            'tradingDate': 'time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Select required columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        
        # Clean data
        df = df.dropna(subset=['time', 'close'])
        df = df[df['close'] > 0]
        df = df.drop_duplicates(subset=['time'], keep='last')
        
        return df
        
    except requests.exceptions.Timeout:
        raise ValueError(f"Timeout khi tải dữ liệu {symbol} từ TCBS")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Lỗi kết nối TCBS API: {str(e)}")
    except Exception as e:
        raise ValueError(f"Lỗi tải dữ liệu {symbol}: {str(e)}")


# Test function
if __name__ == "__main__":
    print("Testing TCBS Data Loader...")
    
    test_symbols = ['VCB', 'FPT', 'VNM', 'HPG']
    
    for symbol in test_symbols:
        try:
            df = load_data(symbol, lookback=100, use_cache=False)
            print(f"✅ {symbol}: {len(df)} rows")
            print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
            print(f"   Latest close: {df['close'].iloc[-1]:,.0f}")
        except Exception as e:
            print(f"❌ {symbol}: {e}")
