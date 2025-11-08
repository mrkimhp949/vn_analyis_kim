def buy_signal(df):
    c = df.iloc[-1]
    p = df.iloc[-2]

    cond1 = p['ema20'] < p['ema50'] and c['ema20'] > c['ema50']
    cond2 = c['rsi'] < 70
    cond3 = c['close'] > c['ema20']

    return cond1 and cond2 and cond3


def sell_signal(df):
    c = df.iloc[-1]
    p = df.iloc[-2]

    cond1 = p['ema20'] > p['ema50'] and c['ema20'] < c['ema50']
    cond2 = c['rsi'] > 30
    cond3 = c['close'] < c['ema20']

    return cond1 and cond2 and cond3
