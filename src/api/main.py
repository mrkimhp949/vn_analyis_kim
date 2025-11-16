# [file name]: main.py
# [file content begin]

import asyncio
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

# Add project root to Python path for direct execution
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytz
import uvicorn
from fastapi import FastAPI, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# from slowapi import _rate_limit_exceeded_handler
# from slowapi.errors import RateLimitExceeded

# Suppress warnings first
from src.utils import suppress_warnings  # noqa: F401

# Import for Prometheus metrics
try:
    from prometheus_client import CONTENT_TYPE_LATEST
except ImportError:
    CONTENT_TYPE_LATEST = "text/plain"  # Fallback if prometheus_client not installed

# Import authentication
from src.api.auth import (
    add_security_headers,
    limiter,
    rate_limit_relaxed,
    rate_limit_strict,
    verify_api_key,
    verify_ip_whitelist,
)

# Import các module của bot - FIXED IMPORT
try:
    from src.core.bot_runner import run_bot_sync

    print("✅ Import run_bot_sync từ src.core.bot_runner thành công")
except ImportError:
    print("❌ Lỗi import run_bot_sync từ src.core.bot_runner")

    # Fallback function
    def run_bot_sync():
        print("🤖 Bot runner không khả dụng")


try:
    from src.data.ticker_loader import run_sector_analysis

    print("✅ Import run_sector_analysis từ ticker_loader thành công")
except ImportError:
    print("❌ Lỗi import run_sector_analysis")

    def run_sector_analysis():
        print("📊 Sector analyzer không khả dụng")
        return []


try:
    from src.notifications.listener import start_bot_listener

    print("✅ Import tg_listener thành công")
except ImportError:
    print("❌ Lỗi import tg_listener")

    def start_bot_listener():
        print("📱 Telegram bot không khả dụng")


try:
    from src.utils.logging_config import setup_logging

    setup_logging()
    print("✅ Import logging_config thành công")
except ImportError:
    print("❌ Lỗi import logging_config")

try:
    from news_analyzer import update_news_cache
except ImportError:
    update_news_cache = None

try:
    from src.config.legacy_config import TICKERS
except ImportError:
    TICKERS = []

tz = pytz.timezone("Asia/Ho_Chi_Minh")


def _run_weekly_model_retraining():
    """Trigger model retraining pipeline."""
    try:
        from ml_pipeline.train_pipeline import run_pipeline
    except ImportError:
        print("❌ Không thể import train_pipeline")
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
    except Exception:
        print("❌ Lỗi retrain models")


# ═══════════════════════════════════════════════════════════
# 📅 SCHEDULER VỚI AUTO SECTOR SELECTION - IMPROVED
# ═══════════════════════════════════════════════════════════


def _check_weekly_retrain(
    now, current_weekday, current_hour, current_minute, last_model_retrain
):
    """Check and run weekly model retraining on Sunday at 20:00"""
    if (
        current_weekday == 6
        and current_hour == 20
        and current_minute == 0
        and last_model_retrain != now.date()
    ):
        try:
            print("\n🤖 [CHỦ NHẬT] TỰ ĐỘNG HUẤN LUYỆN LẠI MÔ HÌNH")
            _run_weekly_model_retraining()
            return True, 61, now.date()
        except Exception:
            print("❌ Lỗi retrain models")
            return True, 61, last_model_retrain
    return False, None, last_model_retrain


def _check_sector_analysis(
    now, current_weekday, current_hour, current_minute, last_sector_analysis
):
    """Check and run sector analysis on Saturday at 20:00"""
    if (
        current_weekday == 5
        and current_hour == 20
        and current_minute == 0
        and last_sector_analysis != now.date()
    ):
        try:
            print("\n📊 [THỨ 7] PHÂN TÍCH THỊ TRƯỜNG")
            run_sector_analysis()
            print("✅ Đã hoàn thành phân tích ngành")
            return True, 61, now.date()
        except Exception:
            print("❌ Lỗi phân tích ngành")
            return True, 61, last_sector_analysis
    return False, None, last_sector_analysis


def _check_signal_scan(now, current_hour, current_minute, last_signal_scan):
    """Check and run signal scan on trading days at 9:15"""
    from src.market.schedule import is_trading_day

    if (
        is_trading_day(now)
        and current_hour == 9
        and current_minute == 15
        and last_signal_scan != now.date()
    ):
        try:
            print(f"\n🎯 [{now.strftime('%A').upper()}] QUÉT TÍN HIỆU")
            run_bot_sync()
            print("✅ Đã hoàn thành quét tín hiệu")
            return True, 61, now.date()
        except Exception:
            print("❌ Lỗi quét tín hiệu")
            return True, 61, last_signal_scan
    return False, None, last_signal_scan


def _check_portfolio(
    now, current_weekday, current_hour, current_minute, last_portfolio_check
):
    """Check and send portfolio update on Friday at 14:45"""
    from src.market.schedule import is_trading_day

    if (
        is_trading_day(now)
        and current_weekday == 4
        and current_hour == 14
        and current_minute == 45
        and last_portfolio_check != now.date()
    ):
        try:
            print("\n💼 [THỨ 6] KIỂM TRA PORTFOLIO")
            from src.portfolio.manager import send_portfolio_update_to_telegram

            send_portfolio_update_to_telegram()
            print("✅ Đã hoàn thành kiểm tra portfolio")
            return True, 61, now.date()
        except Exception:
            print("❌ Lỗi kiểm tra portfolio")
            return True, 61, last_portfolio_check
    return False, None, last_portfolio_check


def _check_news_refresh(now, current_hour, current_minute, last_news_refresh):
    """Check and refresh news during trading hours"""
    from src.market.schedule import is_trading_hour

    if (
        update_news_cache
        and is_trading_hour(now)
        and current_hour in (9, 13, 14)
        and current_minute == 30
        and last_news_refresh != (now.date(), current_hour)
    ):
        try:
            print("\n📰 Đang cập nhật tin tức thị trường...")
            slice_symbols = TICKERS[:40] if TICKERS else None
            update_news_cache(symbols=slice_symbols)
            print("✅ Tin tức đã được cập nhật")
            return True, 61, (now.date(), current_hour)
        except Exception:
            print("⚠️ Lỗi cập nhật tin tức")
            return True, 61, last_news_refresh
    return False, None, last_news_refresh


def _check_daily_summary(now, current_hour, current_minute, last_daily_summary):
    """Check and send daily summary on trading days at 15:15"""
    from src.market.schedule import is_trading_day

    if (
        is_trading_day(now)
        and current_hour == 15
        and current_minute == 15
        and last_daily_summary != now.date()
    ):
        try:
            print("\n📊 [17:00] GỬI DAILY SUMMARY")
            from src.notifications.telegram import send_daily_summary_to_all

            asyncio.run(send_daily_summary_to_all())
            print("✅ Đã gửi daily summary")
            return True, 61, now.date()
        except Exception:
            print("❌ Lỗi gửi daily summary")
            return True, 61, last_daily_summary
    return False, None, last_daily_summary


def _check_pnl_record(now, current_hour, current_minute, last_pnl_record):
    """Check and record PnL + backup on trading hours at 15:10"""
    from src.market.schedule import is_trading_hour

    if (
        is_trading_hour(now)
        and current_hour == 15
        and current_minute == 10
        and last_pnl_record != now.date()
    ):
        try:
            print("\n📊 [15:10] GHI LẠI PNL VÀ BACKUP DATABASE")
            from src.portfolio.manager import PortfolioManager

            pm = PortfolioManager()
            pm._record_daily_snapshot()

            from src.portfolio.paper_trading import get_paper_account

            paper_account = get_paper_account()
            paper_account.record_daily_pnl()

            from src.utils.backup_manager import scheduled_backup

            scheduled_backup()

            print("✅ Đã ghi lại PnL và backup database")
            return True, 61, now.date()
        except Exception:
            print("❌ Lỗi ghi PnL/backup")
            return True, 61, last_pnl_record
    return False, None, last_pnl_record


def _update_last_trackers(new_val, trackers):
    """Update all last_* tracker variables based on new_val type"""
    if isinstance(new_val, tuple):
        trackers["last_news_refresh"] = new_val
    elif isinstance(new_val, datetime.date) or (
        getattr(new_val, "__class__", None).__name__ == "date"
    ):
        # Update all date trackers
        for key in [
            "last_sector_analysis",
            "last_signal_scan",
            "last_portfolio_check",
            "last_daily_summary",
            "last_pnl_record",
            "last_model_retrain",
        ]:
            if trackers[key] != new_val:
                trackers[key] = new_val
    return trackers


def schedule_job():
    """Scheduler với improved logic, error handling và VN trading schedule"""
    print("🤖 Bot khởi động với Portfolio Management!")

    from src.market.schedule import (
        is_trading_day,
        is_trading_hour,
        should_run_scheduled_task,
    )

    # Track last execution times
    trackers = {
        "last_sector_analysis": None,
        "last_signal_scan": None,
        "last_portfolio_check": None,
        "last_news_refresh": None,
        "last_daily_summary": None,
        "last_pnl_record": None,
        "last_model_retrain": None,
    }

    while True:
        try:
            now = datetime.now(tz)
            current_hour = now.hour
            current_minute = now.minute
            current_weekday = now.weekday()

            # Quick check for weekly retrain first
            ran, sleep_sec, new_retrain = _check_weekly_retrain(
                now,
                current_weekday,
                current_hour,
                current_minute,
                trackers["last_model_retrain"],
            )
            if ran:
                trackers["last_model_retrain"] = new_retrain
                time.sleep(sleep_sec)
                continue

            # If not trading time, sleep appropriately
            if not should_run_scheduled_task(now):
                if not is_trading_day(now):
                    time.sleep(3600)
                elif not is_trading_hour(now):
                    time.sleep(1800)
                else:
                    time.sleep(60)
                continue

            # Run scheduled checks in order
            checks = [
                (
                    lambda: _check_sector_analysis(
                        now,
                        current_weekday,
                        current_hour,
                        current_minute,
                        trackers["last_sector_analysis"],
                    )
                ),
                (
                    lambda: _check_signal_scan(
                        now, current_hour, current_minute, trackers["last_signal_scan"]
                    )
                ),
                (
                    lambda: _check_portfolio(
                        now,
                        current_weekday,
                        current_hour,
                        current_minute,
                        trackers["last_portfolio_check"],
                    )
                ),
                (
                    lambda: _check_news_refresh(
                        now, current_hour, current_minute, trackers["last_news_refresh"]
                    )
                ),
                (
                    lambda: _check_daily_summary(
                        now,
                        current_hour,
                        current_minute,
                        trackers["last_daily_summary"],
                    )
                ),
                (
                    lambda: _check_pnl_record(
                        now, current_hour, current_minute, trackers["last_pnl_record"]
                    )
                ),
            ]

            for check in checks:
                ran, sleep_sec, new_val = check()
                if ran:
                    trackers = _update_last_trackers(new_val, trackers)
                    time.sleep(sleep_sec)
                    break

            # Short sleep to be responsive during trading hours
            time.sleep(30)

        except Exception:
            print("❌ Lỗi scheduler")
            time.sleep(60)


# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Khởi động Trading Bot...")
    print("🐍 Python version")
    print(f"⏰ Timezone: {tz}")

    # Validate configuration on startup - MANDATORY
    try:
        from src.config.exceptions import ConfigurationError
        from src.config.trading_config import get_config

        config = get_config(validate=True)
        print("✅ Configuration validated successfully")
        print(config.summary())
    except ConfigurationError:
        print("❌ FATAL: Configuration validation failed")
        print("🛑 Bot cannot start without valid configuration")
        print("📝 Please set required environment variables:")
        print("   - TELEGRAM_TOKEN")
        print("   - CHAT_ID")
        raise
    except Exception:
        print("❌ FATAL: Error loading configuration")
        raise

    print("🔧 Starting background services...")

    # Khởi động scheduler trong thread riêng
    try:
        scheduler_thread = threading.Thread(target=schedule_job, daemon=True)
        scheduler_thread.start()
        print("✅ Scheduler started")
    except Exception:
        print("❌ Lỗi khởi động scheduler")

    # Khởi động Telegram bot trong thread riêng
    try:
        # Check if telegram is configured before starting
        from src.config.trading_config import get_config

        config = get_config(validate=False)

        if config.telegram.token and config.telegram.token.strip():
            telegram_thread = threading.Thread(target=start_bot_listener, daemon=True)
            telegram_thread.start()
            print("✅ Telegram bot started")
        else:
            print("⚠️ Telegram bot not configured (TELEGRAM_TOKEN not set)")
            print("💡 Set TELEGRAM_TOKEN in .env file to enable Telegram notifications")
    except Exception as e:
        print(f"⚠️ Telegram bot error: {str(e)}")
        print("💡 Bot will run without Telegram notifications")

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

# Add rate limiter
app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    return add_security_headers(response)


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
@rate_limit_relaxed
async def health_check(request: Request):
    """Health check chi tiết với Prometheus metrics"""
    import sys
    from datetime import datetime

    from src.monitoring.enhanced import get_enhanced_monitor

    monitor = get_enhanced_monitor()

    # Get comprehensive health status
    health_status = monitor.get_health_status()

    health_info = {
        "status": health_status["status"],
        "timestamp": datetime.now(tz).isoformat(),
        "uptime_seconds": health_status["uptime_seconds"],
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "last_scan": health_status["last_scan"],
        "checks": health_status["checks"],
    }

    # Kiểm tra các components
    try:
        from src.data.loader import load_data

        test_data = load_data("VNM", lookback=5, use_cache=True)
        health_info["data_loader"] = "OK"
        health_info["data_points"] = len(test_data)
    except Exception as e:
        health_info["data_loader"] = f"ERROR: {str(e)}"
        health_info["status"] = "degraded"

    try:
        from src.ml.models.predictor import MLPredictor

        predictor = MLPredictor()
        models_loaded = predictor.load_models()
        health_info["ml_models"] = "OK" if models_loaded else "DUMMY_MODELS"
    except Exception as e:
        health_info["ml_models"] = f"ERROR: {str(e)}"
        health_info["status"] = "degraded"

    # Update system metrics
    monitor.update_system_metrics()

    return health_info


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from src.monitoring.enhanced import get_enhanced_monitor

    monitor = get_enhanced_monitor()
    metrics_data = monitor.export_metrics()

    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)


@app.get("/health/portfolio")
@rate_limit_relaxed
async def health_portfolio(request: Request):
    """Portfolio health: DB path, positions count, and quick totals."""
    from datetime import datetime

    try:
        from src.portfolio.manager import get_portfolio_manager
        from src.data.database import get_db

        pm = get_portfolio_manager()
        db = get_db()

        positions = pm.get_positions()
        portfolio = pm.get_portfolio_value()

        resp = {
            "status": "ok",
            "timestamp": datetime.now(tz).isoformat(),
            "db_path": str(db.db_path),
            "num_positions": len(positions),
            "symbols": sorted(list(positions.keys()))[:20],
            "totals": {
                "total_value": portfolio.get("total_value", 0),
                "total_cost": portfolio.get("total_cost", 0),
                "pnl": portfolio.get("pnl", 0),
                "pnl_percent": portfolio.get("pnl_percent", 0),
            },
        }
        return JSONResponse(resp)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/run-bot")
@rate_limit_strict
async def run_bot(request: Request, api_key: str = Security(verify_api_key)):
    """Manual trigger - chạy bot ngay (Protected endpoint)"""
    try:
        # Verify IP whitelist
        await verify_ip_whitelist(request)

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
        from src.portfolio.manager import PortfolioManager

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
        from src.portfolio.manager import PortfolioManager

        manager = PortfolioManager()

        analysis = manager.analyze_portfolio()
        return JSONResponse(analysis)

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/portfolio/add")
async def add_to_portfolio(symbol: str, shares: int, price: float):
    """Thêm cổ phiếu vào portfolio"""
    try:
        from src.portfolio.manager import PortfolioManager

        from src.utils.validation import InputValidator

        # VALIDATE INPUT
        try:
            symbol = InputValidator.validate_symbol(symbol)
            shares = InputValidator.validate_shares(shares)
            price = InputValidator.validate_price(price)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        manager = PortfolioManager()

        # Check if symbol already has position (for averaging up)
        existing_positions = manager.get_positions()
        symbol_has_position = symbol in existing_positions

        manager.add_position(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            metadata={"source": "api_add_to_portfolio"},
        )

        # Get updated position info
        updated_positions = manager.get_positions()
        if symbol in updated_positions:
            pos = updated_positions[symbol]
            message = (
                f"Đã {'mua thêm' if symbol_has_position else 'thêm'} {shares} CP {symbol} "
                f"@ {price:,.0f} VNĐ. "
                f"Tổng: {pos['shares']} CP @ {pos['avg_price']:,.0f} VNĐ TB"
                if symbol_has_position
                else f"Đã thêm {shares} CP {symbol} vào portfolio"
            )
        else:
            message = f"Đã thêm {shares} CP {symbol} vào portfolio"

        return {
            "status": "success",
            "message": message,
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
        from src.portfolio.manager import PortfolioManager

        from src.utils.validation import InputValidator

        # VALIDATE INPUT
        try:
            symbol = InputValidator.validate_symbol(symbol)
            if shares is not None:
                shares = InputValidator.validate_shares(shares)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

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
        from src.utils.logging_config import setup_logging

        setup_logging()
    except ImportError:
        print("⚠️ Logging config not available")

    port = int(os.environ.get("PORT", 8080))  # Changed from 8000 to 8080

    print("\n" + "=" * 70)
    print("🚀 KHỞI ĐỘNG BOT TRADING - PYTHON 3.11.0")
    print("=" * 70)
    print(f"🔗 API: http://0.0.0.0:{port}")
    print(f"📚 Docs: http://0.0.0.0:{port}/docs")
    print(f"📊 Health: http://0.0.0.0:{port}/health")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
# [file content end]
