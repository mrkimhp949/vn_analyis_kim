import asyncio
import sys
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
from config import TELEGRAM_TOKEN
from bot_runner import run_bot_with_context

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Bot sẵn sàng! Gõ /run để lấy tín hiệu ngay.")

async def run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Đang lấy dữ liệu và phân tích...")
    
    await run_bot_with_context(context.bot, update.effective_chat.id)
    
    await update.message.reply_text("✅ Đã hoàn thành phân tích!")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📈 Bot đang hoạt động!")

async def run_bot_async():
    """Hàm async chạy bot"""
    print("✅ Telegram Bot đang khởi động...")
    print(f"🐍 Python: {sys.version}")
    
    # ✅ Tạo application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # ✅ Thêm handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("run", run))
    app.add_handler(CommandHandler("status", status))
    
    try:
        # ✅ Khởi tạo tuần tự
        print("  ↳ Initializing...")
        await app.initialize()
        
        print("  ↳ Starting...")
        await app.start()
        
        print("  ↳ Starting polling...")
        # ✅ FIX: Không dùng drop_pending_updates nếu gây lỗi
        if app.updater:
            await app.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
        
        print("✅ Telegram Bot đã sẵn sàng!")
        
        # Giữ bot chạy
        await asyncio.Event().wait()
        
    except Exception as e:
        print(f"❌ Lỗi khởi động bot: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print("🛑 Đang dừng bot...")
        if app.updater:
            await app.updater.stop()
        await app.stop()
        await app.shutdown()

def start_bot_listener():
    """Chạy Telegram bot trong thread riêng"""
    # ✅ Kiểm tra nếu đã có event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Nếu loop đang chạy, tạo loop mới
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        # Không có loop, tạo mới
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(run_bot_async())
    except KeyboardInterrupt:
        print("🛑 Bot đã dừng")
    except Exception as e:
        print(f"❌ Lỗi Telegram Bot: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            loop.close()
        except Exception:
            pass

def start_bot_listener():
    """Chạy Telegram bot trong thread riêng với event loop mới"""
    # ✅ Tạo event loop mới cho thread này
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        print("✅ Telegram Bot đang khởi động...")
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("run", run))
        app.add_handler(CommandHandler("status", status))
        
        # ✅ Chạy polling trong event loop mới
        app.run_polling()
    except Exception as e:
        print(f"❌ Lỗi Telegram Bot: {e}")
    finally:
        loop.close()