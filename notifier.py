from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from config import TELEGRAM_TOKEN, CHAT_ID, TICKERS, LOOKBACK
from main import run_bot
from datetime import datetime
import pytz

# Khởi tạo bot để gửi tin nhắn
bot = Bot(token=TELEGRAM_TOKEN)

def send_message(text):
    bot.send_message(chat_id=CHAT_ID, text=text)

# Lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Bot sẵn sàng! Gõ /run để lấy tín hiệu ngay.")

# Lệnh /run
async def run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Đang lấy dữ liệu và phân tích...")
    run_bot()
    await update.message.reply_text("✅ Đã gửi tín hiệu!")

# Lệnh /status
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    tickers = ", ".join(TICKERS)
    await update.message.reply_text(
        f"📈 Trạng thái bot:\n• Thời gian: {now}\n• Mã theo dõi: {tickers}\n• Lookback: {LOOKBACK} phiên"
    )

# Khởi động listener Telegram
def start_bot_listener():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("run", run))
    app.add_handler(CommandHandler("status", status))
    app.run_polling()
