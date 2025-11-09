from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import os
import threading
import time
from datetime import datetime
import pytz
import requests
from bot_runner import run_bot_sync  # ✅ Import sync version
from tg_listener import start_bot_listener

tz = pytz.timezone("Asia/Ho_Chi_Minh")

# Scheduler chạy mỗi ngày lúc 16:05 từ thứ 2 đến thứ 6
def schedule_job():
    print("Bot đã khởi động!")
    while True:
        now = datetime.now(tz)
        if now.weekday() < 5 and now.hour == 16 and now.minute == 5:
            run_bot_sync()  # ✅ Dùng sync version
            time.sleep(60)
        time.sleep(1)

# Giám sát Web Service
def self_monitor():
    print("Bắt đầu giám sát Web Service...")
    while True:
        try:
            response = requests.get("http://localhost:8000", timeout=5)
            if response.status_code != 200:
                print(f"⚠️ Web Service trả về mã {response.status_code}")
        except Exception as e:
            print(f"❌ Web Service KHÔNG HOẠT ĐỘNG: {e}")
        time.sleep(300)

# FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bot đang chạy 24/7 trên Render Web Service"}

@app.head("/")
def head_root():
    return JSONResponse(content=None, status_code=200)

@app.get("/status")
def status():
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return {"status": "OK", "time": now}

# Khởi chạy Web Service
if __name__ == "__main__":
    # ✅ Khởi động các thread nền TRƯỚC khi start FastAPI
    threading.Thread(target=schedule_job, daemon=True).start()
    threading.Thread(target=self_monitor, daemon=True).start()
    threading.Thread(target=start_bot_listener, daemon=True).start()  # ✅ Chạy trong thread
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)