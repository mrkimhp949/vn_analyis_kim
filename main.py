from config import TICKERS, LOOKBACK
from data_loader import load_data
from indicators import add_indicators
from signals import buy_signal, sell_signal
from notifier import send_message
import time
import threading
from datetime import datetime
import pytz

from fastapi import FastAPI
import uvicorn
import os

# Múi giờ Việt Nam
tz = pytz.timezone("Asia/Ho_Chi_Minh")

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

def schedule_job():
    print("Bot đã khởi động!")
    while True:
        now = datetime.now(tz)
        if now.hour == 16 and now.minute == 5:
            run_bot()
            time.sleep(60)  # tránh chạy lại trong cùng phút
        time.sleep(1)

# Chạy bot trong thread nền
threading.Thread(target=schedule_job, daemon=True).start()

# Khởi tạo FastAPI để mở cổng cho Render
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bot đang chạy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
