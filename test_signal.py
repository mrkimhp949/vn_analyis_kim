# test_signal.py
from data_loader import load_data
from indicators import add_indicators
from signals import buy_signal, sell_signal

symbol = "VNM"
df = load_data(symbol, 200)
df = add_indicators(df)

c = df.iloc[-1]
p = df.iloc[-2]

print(f"Symbol: {symbol}")
print(f"EMA20 hôm qua: {p['ema20']:.2f} | EMA50: {p['ema50']:.2f}")
print(f"EMA20 hôm nay: {c['ema20']:.2f} | EMA50: {c['ema50']:.2f}")
print(f"RSI: {c['rsi']:.2f}")
print(f"Close: {c['close']:.2f}")
print(f"\nBUY Signal: {buy_signal(df)}")
print(f"SELL Signal: {sell_signal(df)}")