def buy_signal(df):
    """Tín hiệu mua mạnh - EMA20 cắt lên EMA50"""
    c = df.iloc[-1]
    p = df.iloc[-2]

    cond1 = p['ema20'] < p['ema50'] and c['ema20'] > c['ema50']  # Golden Cross
    cond2 = c['rsi'] < 70
    cond3 = c['close'] > c['ema20']

    return cond1 and cond2 and cond3


def sell_signal(df):
    """Tín hiệu bán mạnh - EMA20 cắt xuống EMA50"""
    c = df.iloc[-1]
    p = df.iloc[-2]

    cond1 = p['ema20'] > p['ema50'] and c['ema20'] < c['ema50']  # Death Cross
    cond2 = c['rsi'] > 30
    cond3 = c['close'] < c['ema20']

    return cond1 and cond2 and cond3


def uptrend_signal(df):
    """Tín hiệu xu hướng tăng - EMA20 trên EMA50 + RSI tốt"""
    c = df.iloc[-1]
    
    cond1 = c['ema20'] > c['ema50']  # Uptrend
    cond2 = 40 < c['rsi'] < 70  # RSI trong vùng tốt
    cond3 = c['close'] > c['ema20']  # Giá trên EMA20
    
    return cond1 and cond2 and cond3


def downtrend_signal(df):
    """Tín hiệu xu hướng giảm - EMA20 dưới EMA50 + RSI yếu"""
    c = df.iloc[-1]
    
    cond1 = c['ema20'] < c['ema50']  # Downtrend
    cond2 = 30 < c['rsi'] < 60  # RSI trong vùng yếu
    cond3 = c['close'] < c['ema20']  # Giá dưới EMA20
    
    return cond1 and cond2 and cond3


def oversold_signal(df):
    """Tín hiệu quá bán - RSI < 30"""
    c = df.iloc[-1]
    return c['rsi'] < 30


def overbought_signal(df):
    """Tín hiệu quá mua - RSI > 70"""
    c = df.iloc[-1]
    return c['rsi'] > 70