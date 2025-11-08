from config import TICKERS, LOOKBACK
from data_loader import load_data
from indicators import add_indicators
from signals import buy_signal, sell_signal
from notifier import send_message
import schedule
import time

def run_bot():
    print("Đang chạy bot phân tích...")

    for symbol in TICKERS:
        try:
            df = load_data(symbol, LOOKBACK)
            df = add_indicators(df)

            if buy_signal(df):
                send_message(f"[{symbol}] BUY ✅ — EMA20 cắt lên EMA50 + RSI OK")

            if sell_signal(df):
                send_message(f"[{symbol}] SELL ✅ — EMA20 cắt xuống EMA50 + RSI OK")

        except Exception as e:
            print(f"Lỗi ở mã {symbol}: {e}")

# chạy 1 lần mỗi ngày sau giờ thị trường đóng
schedule.every().day.at("16:05").do(run_bot)

print("Bot đã khởi động!")
while True:
    schedule.run_pending()
    time.sleep(1)
