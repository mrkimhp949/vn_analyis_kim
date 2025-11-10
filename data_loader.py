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

        # Normalize tên cột: API có thể trả 'tradingDate' (string) hoặc 't' (unix ts)
        if 'tradingDate' in df.columns:
            df = df.rename(columns={'tradingDate': 'time'})
            df['time'] = pd.to_datetime(df['time'])
        elif 't' in df.columns:
            df = df.rename(columns={'t': 'time'})
            # Nếu time là unix timestamp int/float
            try:
                df['time'] = pd.to_datetime(df['time'], unit='s')
            except Exception:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
        else:
            # fallback nếu đã có 'time' hoặc khác
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')

        # Chọn cột cần thiết (bảo đảm tồn tại)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = None

        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

        # Loại bỏ NaN và sắp xếp
        df = df.dropna()
        df = df.sort_values('time')
        df = df.tail(lookback)
        
        print(f"✅ Tải thành công {len(df)} nến")
        
        return df
        
    except Exception as e:
        raise Exception(f"Lỗi khi tải dữ liệu {symbol}: {e}")