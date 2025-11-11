import asyncio
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
from config import TELEGRAM_TOKEN
from bot_runner_improved import run_bot_with_context

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Bot sẵn sàng! Gõ /run để lấy tín hiệu ngay.")

async def run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Đang lấy dữ liệu và phân tích...")
    
    # ✅ Truyền context.bot để gửi tin nhắn
    await run_bot_with_context(context.bot, update.effective_chat.id)
    
    await update.message.reply_text("✅ Đã hoàn thành phân tích!")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📈 Bot đang hoạt động!")

async def run_bot_async():
    """Hàm async chạy bot"""
    print("✅ Telegram Bot đang khởi động...")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("run", run))
    app.add_handler(CommandHandler("status", status))
    
    # ✅ Khởi tạo và chạy bot thủ công
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    
    print("✅ Telegram Bot đã sẵn sàng!")
    
    # Giữ bot chạy mãi mãi
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        print("🛑 Đang dừng bot...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

def start_bot_listener():
    """Chạy Telegram bot trong thread riêng với event loop mới"""
    # ✅ Tạo event loop mới cho thread này
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(run_bot_async())
    except Exception as e:
        print(f"❌ Lỗi Telegram Bot: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loop.close()