from vnstock import stock_historical_data
import pandas as pd
from datetime import datetime, timedelta

def load_data(symbol, lookback=200):
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback*2)).strftime("%Y-%m-%d")

    df = stock_historical_data(
        symbol,
        start,
        end,
        resolution="1D"
    )
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    df['time'] = pd.to_datetime(df['time'])
    df = df.dropna()
    df = df.tail(lookback)
    return df
