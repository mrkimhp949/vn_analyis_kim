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
from rate_limiter import tcbs_limiter
from exceptions import DataLoadError, DataQualityError
from data_quality import get_quality_checker

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
    
    try:
        df = _download_from_tcbs(symbol, start_dt, end_dt)
    except Exception as e:
        raise DataLoadError(
            f"Failed to download data for {symbol}",
            context={'symbol': symbol, 'lookback': lookback, 'error': str(e)}
        ) from e
    
    # Validate basic requirements
    if df.empty or len(df) < min(50, lookback * 0.3):
        raise DataLoadError(
            f"Insufficient data for {symbol}: got {len(df)} bars, need at least {min(50, lookback * 0.3)}",
            context={'symbol': symbol, 'got': len(df), 'required': min(50, lookback * 0.3)}
        )
    
    # Sort và limit
    df = df.sort_values('time')
    df = df.tail(lookback)
    df = df.reset_index(drop=True)
    
    # Data quality check
    try:
        quality_checker = get_quality_checker()
        quality_results = quality_checker.validate(df, symbol)
        
        if not quality_results['valid']:
            # Log issues but don't fail - clean data instead
            if quality_results['issues']:
                logger.warning(f"Data quality issues for {symbol}: {quality_results['issues']}")
                # Auto-clean data
                df = quality_checker.clean_data(df, method='forward_fill')
            
            if quality_results['warnings']:
                logger.info(f"Data quality warnings for {symbol}: {quality_results['warnings']}")
    except Exception as e:
        logger.warning(f"Data quality check failed for {symbol}: {e}")
        # Continue anyway - don't fail on quality check errors
    
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
    Tải dữ liệu từ TCBS API với rate limiting
    """
    try:
        # Apply rate limiting
        tcbs_limiter.wait()
        
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
            raise DataLoadError(
                f"TCBS API returned status {response.status_code} for {symbol}",
                context={'symbol': symbol, 'status_code': response.status_code}
            )
        
        data = response.json()
        
        if not isinstance(data, dict) or 'data' not in data:
            raise DataLoadError(
                f"TCBS API returned invalid format for {symbol}",
                context={'symbol': symbol, 'response_keys': list(data.keys()) if isinstance(data, dict) else None}
            )
        
        bars = data['data']
        
        if not bars:
            raise DataLoadError(
                f"No data available for {symbol} from TCBS",
                context={'symbol': symbol}
            )
        
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
        df['time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce')
        
        # Select required columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        
        # Clean data
        df = df.dropna(subset=['time', 'close'])
        df = df[df['close'] > 0]
        df = df.drop_duplicates(subset=['time'], keep='last')
        
        return df
        
    except requests.exceptions.Timeout:
        raise DataLoadError(
            f"Timeout loading data for {symbol} from TCBS",
            context={'symbol': symbol, 'error_type': 'timeout'}
        )
    except requests.exceptions.RequestException as e:
        raise DataLoadError(
            f"TCBS API connection error for {symbol}",
            context={'symbol': symbol, 'error_type': 'connection', 'error': str(e)}
        )
    except Exception as e:
        raise DataLoadError(
            f"Error loading data for {symbol}",
            context={'symbol': symbol, 'error_type': 'unknown', 'error': str(e)}
        ) from e


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
