# [file name]: main.py
# [file content begin]

# Suppress warnings first
import suppress_warnings  # noqa: F401

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

# Import các module của bot - FIXED IMPORT
try:
    from bot_runner_improved import run_bot_sync, run_sector_analysis

    print("✅ Import bot_runner_improved thành công")
except ImportError as e:
    print(f"❌ Lỗi import bot_runner_improved: {e}")

    # Fallback functions
    def run_bot_sync():
        print("🤖 Bot runner không khả dụng")

    def run_sector_analysis():
        print("📊 Sector analyzer không khả dụng")
        return []


try:
    from tg_listener import start_bot_listener

    print("✅ Import tg_listener thành công")
except ImportError as e:
    print(f"❌ Lỗi import tg_listener: {e}")

    def start_bot_listener():
        print("📱 Telegram bot không khả dụng")


try:
    from logging_config import setup_logging

    setup_logging()
    print("✅ Import logging_config thành công")
except ImportError as e:
    print(f"❌ Lỗi import logging_config: {e}")

try:
    from news_analyzer import update_news_cache
except ImportError:
    update_news_cache = None

try:
    from config import TICKERS
except ImportError:
    TICKERS = []

tz = pytz.timezone("Asia/Ho_Chi_Minh")


def _run_weekly_model_retraining():
    """Trigger model retraining pipeline."""
    try:
        from ml_pipeline.train_pipeline import run_pipeline
    except ImportError as exc:
        print(f"❌ Không thể import train_pipeline: {exc}")
        return

    try:
        tickers = TICKERS[:40] if TICKERS else []
        if not tickers:
            print("⚠️ Không có danh sách tickers để retrain, bỏ qua.")
            return
        print("🔄 Đang retrain ensemble & volatility models...")
        report = run_pipeline(tickers=tickers, lookback=250, refresh=True)
        print(
            "✅ Retrain hoàn tất. Accuracy:",
            report.get("ensemble", {}).get("accuracy", {}).get("mean"),
        )
    except Exception as exc:
        print(f"❌ Lỗi retrain models: {exc}")


# ═══════════════════════════════════════════════════════════
# 📅 SCHEDULER VỚI AUTO SECTOR SELECTION - IMPROVED
# ═══════════════════════════════════════════════════════════


def schedule_job():
    """Scheduler với improved logic, error handling và VN trading schedule"""
    print("🤖 Bot khởi động với Portfolio Management!")

    from vn_trading_schedule import (
        should_run_scheduled_task,
        is_trading_hour,
        is_trading_day,
    )

    last_sector_analysis = None
    last_signal_scan = None
    last_portfolio_check = None
    last_news_refresh = None
    last_daily_summary = None
    last_pnl_record = None
    last_model_retrain = None

    while True:
        try:
            now = datetime.now(tz)
            current_hour = now.hour
            current_minute = now.minute
            current_weekday = now.weekday()
            current_date = now.date()

            # ===== WEEKLY MODEL RETRAINING (Chủ nhật 20:00) =====
            if (
                current_weekday == 6
                and current_hour == 20
                and current_minute == 0
                and last_model_retrain != current_date
            ):
                print("\n🤖 [CHỦ NHẬT] TỰ ĐỘNG HUẤN LUYỆN LẠI MÔ HÌNH")
                _run_weekly_model_retraining()
                last_model_retrain = current_date
                time.sleep(61)
                continue

            # ===== CHECK: Chỉ chạy trong giờ giao dịch VN =====
            if not should_run_scheduled_task(now):
                # Nếu không phải giờ giao dịch, sleep lâu hơn
                if not is_trading_day(now):
                    # Ngày nghỉ (T3/T7), sleep 1 giờ
                    time.sleep(3600)
                elif not is_trading_hour(now):
                    # Ngoài giờ giao dịch, sleep 30 phút
                    time.sleep(1800)
                else:
                    time.sleep(60)
                continue

            # THỨ 7 20:00 - Phân tích ngành (chỉ chạy 1 lần mỗi tuần)
            if (
                current_weekday == 5
                and current_hour == 20
                and current_minute == 0
                and last_sector_analysis != current_date
            ):
                print("\n📊 [THỨ 7] PHÂN TÍCH THỊ TRƯỜNG")
                try:
                    run_sector_analysis()
                    last_sector_analysis = current_date
                    print("✅ Đã hoàn thành phân tích ngành")
                except Exception as e:
                    print(f"❌ Lỗi phân tích ngành: {e}")
                time.sleep(61)

            # THỨ 2-6 9:15 - Quét tín hiệu (sau khi mở cửa 15 phút, chỉ chạy 1 lần mỗi ngày)
            elif (
                is_trading_day(now)
                and current_hour == 9
                and current_minute == 15
                and last_signal_scan != current_date
            ):
                print(f"\n🎯 [{now.strftime('%A').upper()}] QUÉT TÍN HIỆU")
                try:
                    run_bot_sync()
                    last_signal_scan = current_date
                    print("✅ Đã hoàn thành quét tín hiệu")
                except Exception as e:
                    print(f"❌ Lỗi quét tín hiệu: {e}")
                time.sleep(61)

            # THỨ 6 14:45 - Kiểm tra portfolio (trước khi đóng cửa, 1 lần/tuần)
            elif (
                is_trading_day(now)
                and current_weekday == 4
                and current_hour == 14
                and current_minute == 45
                and last_portfolio_check != current_date
            ):
                print("\n💼 [THỨ 6] KIỂM TRA PORTFOLIO")
                try:
                    from portfolio_manager import send_portfolio_update_to_telegram

                    send_portfolio_update_to_telegram()
                    last_portfolio_check = current_date
                    print("✅ Đã hoàn thành kiểm tra portfolio")
                except Exception as e:
                    print(f"❌ Lỗi kiểm tra portfolio: {e}")
                time.sleep(61)

            # Refresh tin tức đa nguồn trong giờ giao dịch (9:30, 13:30, 14:30)
            elif (
                update_news_cache
                and is_trading_hour(now)
                and current_hour in (9, 13, 14)
                and current_minute == 30
                and last_news_refresh != (current_date, current_hour)
            ):
                try:
                    print("\n📰 Đang cập nhật tin tức thị trường...")
                    slice_symbols = TICKERS[:40] if TICKERS else None
                    update_news_cache(symbols=slice_symbols)
                    last_news_refresh = (current_date, current_hour)
                    print("✅ Tin tức đã được cập nhật")
                except Exception as e:
                    print(f"⚠️ Lỗi cập nhật tin tức: {e}")
                time.sleep(61)

            # 15:15 - Gửi daily summary cho tất cả users (sau khi đóng cửa)
            elif (
                is_trading_day(now)
                and current_hour == 15
                and current_minute == 15
                and last_daily_summary != current_date
            ):
                print("\n📊 [17:00] GỬI DAILY SUMMARY")
                try:
                    from telegram_notifications import send_daily_summary_to_all

                    # Run async function properly in thread-safe manner
                    asyncio.run(send_daily_summary_to_all())
                    last_daily_summary = current_date
                    print("✅ Đã gửi daily summary")
                except Exception as e:
                    print(f"❌ Lỗi gửi daily summary: {e}")
                time.sleep(61)

            # 15:10 - Record daily PnL cho portfolio và paper trading (trước khi đóng cửa)
            elif (
                is_trading_hour(now)
                and current_hour == 15
                and current_minute == 10
                and last_pnl_record != current_date
            ):
                print("\n📊 [15:30] GHI LẠI PNL HÀNG NGÀY")
                try:
                    # Record portfolio history
                    from portfolio_manager import PortfolioManager

                    pm = PortfolioManager()
                    pm._record_daily_snapshot()

                    # Record paper trading PnL
                    from paper_trading import get_paper_account

                    paper_account = get_paper_account()
                    paper_account.record_daily_pnl()

                    last_pnl_record = current_date
                    print("✅ Đã ghi lại PnL hàng ngày")
                except Exception as e:
                    print(f"❌ Lỗi ghi PnL: {e}")
                time.sleep(61)

            # Sleep ngắn hơn trong giờ giao dịch để responsive hơn
            time.sleep(30)

        except Exception as e:
            print(f"❌ Lỗi scheduler: {e}")
            time.sleep(60)


# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Khởi động Trading Bot...")
    print(f"🐍 Python version")
    print(f"⏰ Timezone: {tz}")

    # Validate configuration on startup - MANDATORY
    try:
        from trading_config import get_config
        from exceptions import ConfigurationError

        config = get_config(validate=True)
        print("✅ Configuration validated successfully")
        print(config.summary())
    except ConfigurationError as e:
        print(f"❌ FATAL: Configuration validation failed: {e}")
        print("🛑 Bot cannot start without valid configuration")
        print("📝 Please set required environment variables:")
        print("   - TELEGRAM_TOKEN")
        print("   - CHAT_ID")
        raise
    except Exception as e:
        print(f"❌ FATAL: Error loading configuration: {e}")
        raise

    print(f"🔧 Starting background services...")

    # Khởi động scheduler trong thread riêng
    try:
        scheduler_thread = threading.Thread(target=schedule_job, daemon=True)
        scheduler_thread.start()
        print("✅ Scheduler started")
    except Exception as e:
        print(f"❌ Lỗi khởi động scheduler: {e}")

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
    lifespan=lifespan,
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
            "docs": "/docs",
        },
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
        "platform": sys.platform,
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
            "started_at": datetime.now(tz).isoformat(),
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
            "started_at": datetime.now(tz).isoformat(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/portfolio")
async def get_portfolio():
    """Lấy thông tin portfolio hiện tại"""
    try:
        from portfolio_manager import PortfolioManager

        manager = PortfolioManager()

        portfolio_data = {
            "current_holdings": manager.get_current_holdings(),
            "portfolio_summary": manager.get_portfolio_summary(),
            "analyzed_at": datetime.now(tz).isoformat(),
        }

        return JSONResponse(portfolio_data)

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/portfolio/analysis")
async def get_portfolio_analysis():
    """Lấy phân tích portfolio chi tiết"""
    try:
        from portfolio_manager import PortfolioManager

        manager = PortfolioManager()

        analysis = manager.analyze_portfolio()
        return JSONResponse(analysis)

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/portfolio/add")
async def add_to_portfolio(symbol: str, shares: int, price: float):
    """Thêm cổ phiếu vào portfolio"""
    try:
        from portfolio_manager import PortfolioManager

        manager = PortfolioManager()

        manager.add_stock(symbol, shares, price)

        return {
            "status": "success",
            "message": f"Đã thêm {shares} CP {symbol} vào portfolio",
            "symbol": symbol,
            "shares": shares,
            "price": price,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/portfolio/remove")
async def remove_from_portfolio(symbol: str, shares: int = None):
    """Bán cổ phiếu khỏi portfolio"""
    try:
        from portfolio_manager import PortfolioManager

        manager = PortfolioManager()

        success, message = manager.remove_stock(symbol, shares)

        if success:
            return {"status": "success", "message": message, "symbol": symbol}
        else:
            return {"status": "error", "message": message}

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

    port = int(os.environ.get("PORT", 8080))  # Changed from 8000 to 8080

    print(f"\n{'='*70}")
    print(f"🚀 KHỞI ĐỘNG BOT TRADING - PYTHON 3.11.0")
    print(f"{'='*70}")
    print(f"🔗 API: http://0.0.0.0:{port}")
    print(f"📚 Docs: http://0.0.0.0:{port}/docs")
    print(f"📊 Health: http://0.0.0.0:{port}/health")
    print(f"{'='*70}\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
# [file content end]
