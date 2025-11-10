from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import threading
import time
from datetime import datetime
import pytz
import asyncio

# Import các module của bot
try:
    from bot_runner import run_bot_sync, run_sector_analysis
    from tg_listener import start_bot_listener
    from logging_config import setup_logging
except ImportError as e:
    print(f"⚠️ Import error: {e}")

tz = pytz.timezone("Asia/Ho_Chi_Minh")

# ═══════════════════════════════════════════════════════════
# 📅 SCHEDULER VỚI AUTO SECTOR SELECTION
# ═══════════════════════════════════════════════════════════

def schedule_job():
    """Scheduler chạy trong background"""
    print("🤖 Bot đã khởi động với Smart Sector Selection!")
    print("📅 Lịch hoạt động:")
    print("  • Thứ 7 20:00: Phân tích toàn bộ thị trường")
    print("  • Thứ 2-6 8:30: Quét các mã đã chọn")
    
    while True:
        try:
            now = datetime.now(tz)
            current_hour = now.hour
            current_minute = now.minute
            current_weekday = now.weekday()  # 0=Monday, 6=Sunday
            
            # ═══════════════════════════════════════════════════════════
            # 📊 THỨ 7 - PHÂN TÍCH TOÀN THỊ TRƯỜNG
            # ═══════════════════════════════════════════════════════════
            if current_weekday == 5 and current_hour == 20 and current_minute == 0:
                print(f"\n{'='*70}")
                print(f"📊 [THỨ 7 20:00] PHÂN TÍCH TOÀN BỘ THỊ TRƯỜNG")
                print(f"{'='*70}\n")
                
                try:
                    run_sector_analysis()
                    print("\n✅ Đã phân tích xong! Đã chọn ngành tốt cho tuần tới.")
                except Exception as e:
                    print(f"❌ Lỗi phân tích: {e}")
                
                time.sleep(60)
            
            # ═══════════════════════════════════════════════════════════
            # 🎯 THỨ 2-6 - QUÉT CÁC MÃ ĐÃ CHỌN
            # ═══════════════════════════════════════════════════════════
            elif current_weekday < 5 and current_hour == 8 and current_minute == 30:
                print(f"\n{'='*70}")
                print(f"📊 [{now.strftime('%A').upper()} 8:30] QUÉT TÍN HIỆU")
                print(f"{'='*70}\n")
                
                try:
                    run_bot_sync()
                    print(f"\n✅ Hoàn thành. Hẹn gặp lại vào 8:30 ngày mai!\n")
                except Exception as e:
                    print(f"❌ Lỗi chạy bot: {e}")
                
                time.sleep(60)
            
            time.sleep(30)  # Giảm CPU usage
            
        except Exception as e:
            print(f"❌ Lỗi scheduler: {e}")
            time.sleep(60)

# Lifespan manager thay thế @app.on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Khởi động Trading Bot...")
    print(f"🐍 Python version")
    print(f"⏰ Timezone: {tz}")
    print(f"🔧 Starting background services...")
    
    # Khởi động scheduler trong thread riêng
    scheduler_thread = threading.Thread(target=schedule_job, daemon=True)
    scheduler_thread.start()
    print("✅ Scheduler started")
    
    # Khởi động Telegram bot trong thread riêng
    try:
        telegram_thread = threading.Thread(target=start_bot_listener, daemon=True)
        telegram_thread.start()
        print("✅ Telegram bot started")
    except Exception as e:
        print(f"⚠️ Telegram bot error: {e}")
    
    print("🎉 Tất cả services đã khởi động!")
    
    yield  # App running
    
    # Shutdown (cleanup)
    print("🛑 Shutting down Trading Bot...")

# FastAPI app với lifespan
app = FastAPI(
    title="Trading Bot API",
    description="🤖 Auto Trading Bot với Smart Sector Selection", 
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # Sử dụng lifespan thay vì on_event
)

@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {
        "message": "🤖 Trading Bot API đang hoạt động",
        "status": "online", 
        "timestamp": datetime.now(tz).isoformat(),
        "endpoints": {
            "health": "/health",
            "run_bot": "/run-bot (POST)", 
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check chi tiết"""
    from datetime import datetime
    import sys
    
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now(tz).isoformat(),
        "python_version": sys.version,
        "platform": sys.platform
    }
    
    # Kiểm tra các components
    try:
        from data_loader import load_data
        test_data = load_data("VNM", lookback=5, use_cache=False)
        health_info["data_loader"] = "OK"
        health_info["data_points"] = len(test_data)
    except Exception as e:
        health_info["data_loader"] = f"ERROR: {str(e)}"
    
    try:
        from ml_models import MLPredictor
        predictor = MLPredictor()
        models_loaded = predictor.load_models()
        health_info["ml_models"] = "OK" if models_loaded else "DUMMY_MODELS"
    except Exception as e:
        health_info["ml_models"] = f"ERROR: {str(e)}"
    
    return health_info

@app.post("/run-bot")
async def run_bot():
    """Manual trigger - chạy bot ngay"""
    try:
        # Chạy trong thread riêng để không block request
        thread = threading.Thread(target=run_bot_sync, daemon=True)
        thread.start()
        
        return {
            "status": "success", 
            "message": "Bot đang chạy trong background...",
            "started_at": datetime.now(tz).isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/analyze-sectors") 
async def analyze_sectors():
    """Manual trigger - phân tích ngành"""
    try:
        thread = threading.Thread(target=run_sector_analysis, daemon=True)
        thread.start()
        
        return {
            "status": "success",
            "message": "Đang phân tích sectors...", 
            "started_at": datetime.now(tz).isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Khởi chạy Web Service
if __name__ == "__main__":
    # Setup logging
    try:
        from logging_config import setup_logging
        setup_logging()
    except ImportError:
        print("⚠️ Logging config not available")
    
    port = int(os.environ.get("PORT", 8000))
    
    print(f"\n{'='*70}")
    print(f"🚀 KHỞI ĐỘNG BOT TRADING - PYTHON 3.11.0")
    print(f"{'='*70}")
    print(f"🔗 API: http://0.0.0.0:{port}")
    print(f"📚 Docs: http://0.0.0.0:{port}/docs") 
    print(f"📊 Health: http://0.0.0.0:{port}/health")
    print(f"{'='*70}\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )