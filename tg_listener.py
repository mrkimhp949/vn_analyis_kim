import asyncio
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram import Update
from config import TELEGRAM_TOKEN
from bot_runner import run_bot_with_context

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Bot sẵn sàng! Gõ /run để lấy tín hiệu ngay.")

async def run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Đang lấy dữ liệu và phân tích...")
    
    # ✅ Truyền context.bot để gửi tin nhắn
    await run_bot_with_context(context.bot, update.effective_chat.id)
    
    await update.message.reply_text("✅ Đã hoàn thành phân tích!")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📈 Bot đang hoạt động!")

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