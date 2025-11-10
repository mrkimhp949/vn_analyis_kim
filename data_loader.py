import pandas as pd
from datetime import datetime, timedelta
import requests

def load_data(symbol, lookback=200):
    """Load dữ liệu từ TCBS API (Ổn định và nhanh)"""
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
        
        if 'data' not in data or not data['data']:
            raise ValueError(f"Không có dữ liệu cho {symbol}")
        
        # Tạo DataFrame từ response
        df = pd.DataFrame(data['data'])
        
        # ✅ Đổi tên cột (TCBS dùng 'tradingDate' thay vì 't')
        df = df.rename(columns={
            'tradingDate': 'time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        # Chọn cột cần thiết
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        
        # Chuyển đổi time
        df['time'] = pd.to_datetime(df['time'])
        
        # Loại bỏ NaN và sắp xếp
        df = df.dropna()
        df = df.sort_values('time')
        df = df.tail(lookback)
        
        print(f"✅ Tải thành công {len(df)} nến")
        
        return df
        
    except Exception as e:
        raise Exception(f"Lỗi khi tải dữ liệu {symbol}: {e}")