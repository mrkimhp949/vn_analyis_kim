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
from fastapi.responses import JSONResponse
import uvicorn
import os
import requests
from notifier import start_bot_listener

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

def self_monitor():
    print("Bắt đầu giám sát Web Service...")
    while True:
        try:
            response = requests.get("http://localhost:8000", timeout=5)
            if response.status_code != 200:
                send_message(f"⚠️ Bot cảnh báo: Web Service trả về mã {response.status_code}")
        except Exception as e:
            send_message(f"❌ Bot KHÔNG HOẠT ĐỘNG: {e}")
        time.sleep(300)  # kiểm tra mỗi 5 phút

# Chạy bot trong thread nền
threading.Thread(target=schedule_job, daemon=True).start()
threading.Thread(target=self_monitor, daemon=True).start()
threading.Thread(target=start_bot_listener, daemon=True).start()

# Khởi tạo FastAPI để mở cổng cho Render
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bot đang chạy"}

@app.head("/")
def head_root():
    return JSONResponse(content=None, status_code=200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

