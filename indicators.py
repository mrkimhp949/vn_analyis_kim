import ta

def add_indicators(df):
    df['ema20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    return df
