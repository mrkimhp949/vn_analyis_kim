import threading
from fastapi import FastAPI
import uvicorn

# phần bot như cũ
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

schedule.every().day.at("16:05").do(run_bot)

def scheduler_loop():
    while True:
        schedule.run_pending()
        time.sleep(1)

# khởi động scheduler trong thread riêng
threading.Thread(target=scheduler_loop, daemon=True).start()

# khởi tạo FastAPI để mở cổng
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bot đang chạy"}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
