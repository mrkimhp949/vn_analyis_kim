from vnstock import stock_historical_data
import pandas as pd
from datetime import datetime, timedelta

def load_data(symbol, lookback=200):
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback*2)).strftime("%Y-%m-%d")

    # ✅ Kiểm tra API mới của vnstock 3.2.6
    df = stock_historical_data(
        symbol=symbol,
        start_date=start,
        end_date=end,
        resolution="1D",
        type="stock"  # ✅ Có thể cần thêm parameter này
    )
    
    # ✅ Xử lý tên cột (có thể đã thay đổi)
    df = df.rename(columns={
        'Time': 'time',
        'Open': 'open', 
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    df['time'] = pd.to_datetime(df['time'])
    df = df.dropna()
    df = df.tail(lookback)
    
    return df