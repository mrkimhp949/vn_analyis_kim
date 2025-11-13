import sys
import requests
from datetime import datetime, timedelta
from config import KIM_SECTOR, THUY_SECTOR

TCBS_API_BASE = 'https://apipubaws.tcbs.com.vn'

def validate_ticker(symbol):
    """Kiểm tra xem mã cổ phiếu có hợp lệ không qua TCBS API"""
    try:
        url = f"{TCBS_API_BASE}/stock-insight/v1/stock/bars-long-term"
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=10)
        
        params = {
            'ticker': symbol,
            'type': 'stock',
            'resolution': 'D',
            'from': int(from_date.timestamp()),
            'to': int(to_date.timestamp())
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, dict) and 'data' in data:
                bars = data['data']
                
                if bars and len(bars) >= 1:
                    # Tính volume trung bình
                    volumes = [bar.get('volume', 0) for bar in bars]
                    avg_volume = sum(volumes) / len(volumes) if volumes else 0
                    return True, f"OK (Volume TB: {avg_volume:,.0f})"
        
        return False, "Không có dữ liệu từ TCBS"
        
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, f"Lỗi: {str(e)[:50]}"

def validate_all_tickers():
    """Kiểm tra tất cả các mã trong config"""
    print("="*60)
    print("🔍 KIỂM TRA CÁC MÃ CỔ PHIẾU TRONG CONFIG")
    print("="*60)
    
    invalid_tickers = []
    valid_tickers = []
    
    all_sectors = {**KIM_SECTOR, **THUY_SECTOR}
    
    for sector_name, tickers in all_sectors.items():
        print(f"\n📊 Kiểm tra ngành: {sector_name}")
        for symbol in tickers:
            is_valid, message = validate_ticker(symbol)
            status = "✅" if is_valid else "❌"
            print(f"  {status} {symbol}: {message}")
            
            if is_valid:
                valid_tickers.append(symbol)
            else:
                invalid_tickers.append((symbol, message))
    
    print("\n" + "="*60)
    print("📊 KẾT QUẢ KIỂM TRA")
    print("="*60)
    print(f"✅ Hợp lệ: {len(valid_tickers)} mã")
    print(f"❌ Không hợp lệ: {len(invalid_tickers)} mã")
    
    if invalid_tickers:
        print("\n⚠️ CÁC MÃ CẦN LOẠI BỎ:")
        for symbol, reason in invalid_tickers:
            print(f"  - {symbol}: {reason}")
        
        print("\n💡 Hãy cập nhật file config.py để loại bỏ các mã này")
    else:
        print("\n🎉 Tất cả các mã đều hợp lệ!")
    
    return valid_tickers, invalid_tickers

if __name__ == "__main__":
    validate_all_tickers()
