"""
Script nhanh để kiểm tra các mã không hợp lệ trong config
Sử dụng TCBS API thay vì Yahoo Finance
"""
import suppress_warnings  # noqa: F401

import requests
from datetime import datetime, timedelta
from config import KIM_SECTOR, THUY_SECTOR

TCBS_API_BASE = 'https://apipubaws.tcbs.com.vn'

def check_ticker(symbol):
    """Kiểm tra nhanh mã qua TCBS API"""
    try:
        url = f"{TCBS_API_BASE}/stock-insight/v1/stock/bars-long-term"
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=5)
        
        params = {
            'ticker': symbol,
            'type': 'stock',
            'resolution': 'D',
            'from': int(from_date.timestamp()),
            'to': int(to_date.timestamp())
        }
        
        response = requests.get(url, params=params, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and 'data' in data:
                bars = data['data']
                return len(bars) > 0
        
        return False
    except Exception:
        return False

def main():
    print("="*60)
    print("KIỂM TRA CÁC MÃ KHÔNG HỢP LỆ")
    print("="*60)
    
    all_sectors = {**KIM_SECTOR, **THUY_SECTOR}
    invalid_tickers = []
    
    for sector_name, tickers in all_sectors.items():
        print(f"\nKiểm tra {sector_name}...")
        for symbol in tickers:
            is_valid = check_ticker(symbol)
            if not is_valid:
                print(f"  ❌ {symbol}")
                invalid_tickers.append((sector_name, symbol))
            else:
                print(f"  ✓ {symbol}", end='\r')
    
    print("\n" + "="*60)
    if invalid_tickers:
        print(f"\n⚠️ Tìm thấy {len(invalid_tickers)} mã không hợp lệ:")
        for sector, symbol in invalid_tickers:
            print(f"  - {symbol} ({sector})")
        
        print("\n💡 Hãy xóa các mã này khỏi config.py")
    else:
        print("\n✅ Tất cả các mã đều hợp lệ!")

if __name__ == "__main__":
    main()
