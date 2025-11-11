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
    from bot_runner_improved import run_bot_sync, run_sector_analysis
    from tg_listener import start_bot_listener
    from logging_config import setup_logging
except ImportError as e:
    print(f"⚠️ Import error: {e}")

tz = pytz.timezone("Asia/Ho_Chi_Minh")

# ═══════════════════════════════════════════════════════════
# 📅 SCHEDULER VỚI AUTO SECTOR SELECTION
# ═══════════════════════════════════════════════════════════

# main.py - UPDATE

def schedule_job():
    """Scheduler với improved logic"""
    print("🤖 Bot khởi động với Improved Trading Logic!")
    
    while True:
        try:
            now = datetime.now(tz)
            current_hour = now.hour
            current_minute = now.minute
            current_weekday = now.weekday()
            
            # THỨ 7 20:00 - Phân tích ngành
            if current_weekday == 5 and current_hour == 20 and current_minute == 0:
                print("\n📊 [THỨ 7] PHÂN TÍCH THỊ TRƯỜNG")
                run_sector_analysis()  # Uses enhanced analyzer
                time.sleep(60)
            
            # THỨ 2-6 8:30 - Quét tín hiệu
            elif current_weekday < 5 and current_hour == 8 and current_minute == 30:
                print(f"\n🎯 [{now.strftime('%A').upper()}] QUÉT TÍN HIỆU")
                run_bot_sync()  # Uses improved logic
                time.sleep(60)
            
            time.sleep(30)
            
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