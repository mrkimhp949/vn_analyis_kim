from vnstock2 import Vnstock
import pandas as pd
from datetime import datetime, timedelta

def load_data(symbol, lookback=200):
    import pandas as pd
    from datetime import datetime, timedelta

    api = Vnstock()
    stock = api.stock(symbol, 'stock')

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback*2)).strftime("%Y-%m-%d")

    df = stock.quote.history(start, end, interval='1D')

    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    df['time'] = pd.to_datetime(df['time'])
    df = df.dropna()
    return df.tail(lookback)

