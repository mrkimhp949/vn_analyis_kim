from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import os
import threading
import time
from datetime import datetime
import pytz
import requests
from bot_runner import run_bot_sync, run_sector_analysis
from tg_listener import start_bot_listener

tz = pytz.timezone("Asia/Ho_Chi_Minh")

# ═══════════════════════════════════════════════════════════
# 📅 SCHEDULER VỚI AUTO SECTOR SELECTION
# ═══════════════════════════════════════════════════════════

def schedule_job():
    print("🤖 Bot đã khởi động với Smart Sector Selection!")
    print("📅 Lịch hoạt động:")
    print("  • Thứ 7 20:00: Phân tích toàn bộ thị trường")
    print("  • Thứ 2-6 8:30: Quét các mã đã chọn")
    
    while True:
        now = datetime.now(tz)
        
        # ═══════════════════════════════════════════════════════════
        # 📊 THỨ 7 - PHÂN TÍCH TOÀN THỊ TRƯỜNG
        # ═══════════════════════════════════════════════════════════
        if now.weekday() == 5 and now.hour == 20 and now.minute == 0:
            print(f"\n{'='*70}")
            print(f"📊 [THỨ 7 20:00] PHÂN TÍCH TOÀN BỘ THỊ TRƯỜNG")
            print(f"{'='*70}\n")
            
            try:
                # Chạy sector analysis và tự động chọn top sectors
                run_sector_analysis()
                print("\n✅ Đã phân tích xong! Đã chọn ngành tốt cho tuần tới.")
            except Exception as e:
                print(f"❌ Lỗi phân tích: {e}")
            
            time.sleep(60)
        
        # ═══════════════════════════════════════════════════════════
        # 🎯 THỨ 2-6 - QUÉT CÁC MÃ ĐÃ CHỌN
        # ═══════════════════════════════════════════════════════════
        elif now.weekday() < 5 and now.hour == 8 and now.minute == 30:
            print(f"\n{'='*70}")
            print(f"📊 [{now.strftime('%A').upper()} 8:30] QUÉT TÍN HIỆU")
            print(f"{'='*70}\n")
            
            run_bot_sync()
            
            print(f"\n✅ Hoàn thành. Hẹn gặp lại vào 8:30 ngày mai!\n")
            time.sleep(60)
        
        time.sleep(1)


# Giám sát Web Service
def self_monitor():
    print("⏳ Chờ Web Service khởi động...")
    startup_wait = True
    
    while True:
        try:
            response = requests.get("http://localhost:8000", timeout=5)
            if startup_wait:
                print("✅ Web Service đã sẵn sàng!")
                startup_wait = False
            
            if response.status_code != 200:
                print(f"⚠️ Web Service trả về mã {response.status_code}")
        except Exception as e:
            if not startup_wait:
                print(f"❌ Web Service KHÔNG HOẠT ĐỘNG: {e}")
        time.sleep(300)


# FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    next_run = get_next_run_time()
    return {
        "message": "🤖 Bot Trading với Smart Sector Selection",
        "schedule": {
            "saturday_analysis": "Thứ 7 20:00 - Phân tích toàn thị trường",
            "daily_scan": "Thứ 2-6 8:30 - Quét mã đã chọn"
        },
        "next_run": next_run,
        "timezone": "Asia/Ho_Chi_Minh"
    }

@app.head("/")
def head_root():
    return JSONResponse(content=None, status_code=200)

@app.get("/status")
def status():
    now = datetime.now(tz)
    next_run = get_next_run_time()
    
    # Đọc selected tickers
    try:
        from config import TICKERS
        ticker_count = len(TICKERS)
    except:
        ticker_count = "N/A"
    
    return {
        "status": "OK",
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": now.strftime("%A"),
        "selected_tickers": ticker_count,
        "next_run": next_run
    }

@app.post("/run-now")
def run_now():
    """Manual trigger - chạy bot ngay"""
    try:
        run_bot_sync()
        return {"status": "success", "message": "Bot đã chạy xong!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze-sectors")
def analyze_sectors():
    """Manual trigger - phân tích ngành"""
    try:
        run_sector_analysis()
        return {"status": "success", "message": "Đã phân tích xong!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_next_run_time():
    """Tính thời gian chạy tiếp theo"""
    now = datetime.now(tz)
    
    # Nếu là thứ 7 và chưa đến 20:00
    if now.weekday() == 5 and now.hour < 20:
        return f"Thứ 7 này lúc 20:00 (phân tích thị trường)"
    
    # Nếu chưa đến 8:30 hôm nay
    if now.hour < 8 or (now.hour == 8 and now.minute < 30):
        if now.weekday() < 5:  # Thứ 2-6
            return f"Hôm nay {now.strftime('%d/%m/%Y')} lúc 8:30"
        elif now.weekday() == 5:  # Thứ 7
            return f"Thứ 7 lúc 20:00 (phân tích)"
        else:  # CN
            return f"Thứ Hai tới lúc 8:30"
    
    # Đã qua 8:30 hôm nay
    if now.weekday() < 4:  # Thứ 2-5
        return f"Ngày mai lúc 8:30"
    elif now.weekday() == 4:  # Thứ 6
        return f"Thứ 7 lúc 20:00 (phân tích)"
    elif now.weekday() == 5:  # Thứ 7
        return f"Thứ Hai tới lúc 8:30"
    else:  # CN
        return f"Thứ Hai tới lúc 8:30"


# Khởi chạy Web Service
if __name__ == "__main__":
    # Khởi động các thread nền
    threading.Thread(target=schedule_job, daemon=True).start()
    threading.Thread(target=self_monitor, daemon=True).start()
    threading.Thread(target=start_bot_listener, daemon=True).start()
    
    port = int(os.environ.get("PORT", 8000))
    
    print(f"\n{'='*70}")
    print(f"🚀 KHỞI ĐỘNG BOT TRADING - SMART SECTOR SELECTION")
    print(f"{'='*70}")
    print(f"📅 Lịch tự động:")
    print(f"  • Thứ 7 20:00: Phân tích 77 mã → Chọn top sectors")
    print(f"  • Thứ 2-6 8:30: Quét chỉ top sectors đã chọn")
    print(f"🌍 Timezone: Asia/Ho_Chi_Minh")
    print(f"🔗 API: http://localhost:{port}")
    print(f"💬 Telegram: /run để test ngay")
    print(f"{'='*70}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import os
import threading
import time
from datetime import datetime
import pytz
import requests
from bot_runner import run_bot_sync
from tg_listener import start_bot_listener

tz = pytz.timezone("Asia/Ho_Chi_Minh")

# ✅ Scheduler đơn giản - CHỈ 8:30 SÁNG
def schedule_job():
    print("🤖 Bot đã khởi động - Sẽ phân tích lúc 8:30 sáng mỗi ngày")
    
    while True:
        now = datetime.now(tz)
        
        # Thứ 2-6, lúc 8:30
        if now.weekday() < 5 and now.hour == 8 and now.minute == 30:
            print(f"\n{'='*60}")
            print(f"📊 [8:30] BẮT ĐẦU PHÂN TÍCH - {now.strftime('%Y-%m-%d')}")
            print(f"{'='*60}\n")
            
            run_bot_sync()
            
            print(f"\n✅ Hoàn thành phân tích. Hẹn gặp lại vào 8:30 ngày mai!\n")
            
            # Sleep 60s để không chạy lại trong cùng phút
            time.sleep(60)
        
        time.sleep(1)

# Giám sát Web Service
def self_monitor():
    print("⏳ Chờ Web Service khởi động...")
    startup_wait = True
    
    while True:
        try:
            response = requests.get("http://localhost:8000", timeout=5)
            if startup_wait:
                print("✅ Web Service đã sẵn sàng!")
                startup_wait = False
            
            if response.status_code != 200:
                print(f"⚠️ Web Service trả về mã {response.status_code}")
        except Exception as e:
            if not startup_wait:
                print(f"❌ Web Service KHÔNG HOẠT ĐỘNG: {e}")
        time.sleep(300)

# FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    next_run = get_next_run_time()
    return {
        "message": "🤖 Bot đang chạy 24/7 trên Render",
        "schedule": "Mỗi sáng 8:30 (Thứ 2-6)",
        "next_run": next_run,
        "timezone": "Asia/Ho_Chi_Minh"
    }

@app.head("/")
def head_root():
    return JSONResponse(content=None, status_code=200)

@app.get("/status")
def status():
    now = datetime.now(tz)
    next_run = get_next_run_time()
    return {
        "status": "OK",
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "next_run": next_run,
        "weekday": now.strftime("%A")
    }

@app.post("/run-now")
def run_now():
    """Manual trigger - gọi API này để chạy bot ngay"""
    try:
        run_bot_sync()
        return {"status": "success", "message": "Bot đã chạy xong!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_next_run_time():
    """Tính thời gian chạy tiếp theo"""
    now = datetime.now(tz)
    
    # Nếu chưa đến 8:30 hôm nay
    if now.hour < 8 or (now.hour == 8 and now.minute < 30):
        if now.weekday() < 5:  # Thứ 2-6
            return f"Hôm nay {now.strftime('%d/%m/%Y')} lúc 8:30"
        else:  # Cuối tuần
            days_until_monday = 7 - now.weekday()
            return f"Thứ Hai tới lúc 8:30 (sau {days_until_monday} ngày)"
    
    # Đã qua 8:30 hôm nay
    if now.weekday() < 4:  # Thứ 2-5
        tomorrow = now.day + 1
        return f"Ngày mai {tomorrow:02d}/{now.month:02d}/{now.year} lúc 8:30"
    else:  # Thứ 6
        return f"Thứ Hai tới lúc 8:30"

# Khởi chạy Web Service
if __name__ == "__main__":
    # Khởi động các thread nền
    threading.Thread(target=schedule_job, daemon=True).start()
    threading.Thread(target=self_monitor, daemon=True).start()
    threading.Thread(target=start_bot_listener, daemon=True).start()
    
    port = int(os.environ.get("PORT", 8000))
    
    print(f"\n{'='*60}")
    print(f"🚀 KHỞI ĐỘNG BOT TRADING")
    print(f"{'='*60}")
    print(f"📅 Lịch chạy: 8:30 sáng (Thứ 2-6)")
    print(f"🌍 Timezone: Asia/Ho_Chi_Minh")
    print(f"🔗 API: http://localhost:{port}")
    print(f"💬 Telegram: /run để test ngay")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)