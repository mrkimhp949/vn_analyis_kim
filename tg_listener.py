import asyncio
import time
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
from telegram.error import NetworkError as TgNetworkError
from config import TELEGRAM_TOKEN
from bot_runner_improved import run_bot_with_context

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Bot sẵn sàng!\n\n"
        "Các lệnh:\n"
        "/run - Lấy tín hiệu giao dịch\n"
        "/portfolio - Xem portfolio\n" 
        "/addstock SYMBOL SHARES PRICE - Thêm cổ phiếu\n"
        "/sellstock SYMBOL [SHARES] - Bán cổ phiếu\n"
        "/status - Trạng thái bot"
    )

async def run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Đang lấy dữ liệu và phân tích...")
    
    await run_bot_with_context(context.bot, update.effective_chat.id)
    
    await update.message.reply_text("✅ Đã hoàn thành phân tích!")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📈 Bot đang hoạt động!")

# ===== PORTFOLIO COMMANDS =====
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /portfolio - Xem portfolio hiện tại"""
    try:
        from portfolio_manager import PortfolioManager
        manager = PortfolioManager()
        analysis_report = manager.get_detailed_analysis()
        
        # Chia nhỏ message nếu cần
        if len(analysis_report) > 4000:
            parts = [analysis_report[i:i+4000] for i in range(0, len(analysis_report), 4000)]
            for part in parts:
                await update.message.reply_text(part, parse_mode='Markdown')
        else:
            await update.message.reply_text(analysis_report, parse_mode='Markdown')
            
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {e}")

async def add_stock_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /addstock symbol shares price - Thêm cổ phiếu"""
    try:
        if not context.args or len(context.args) < 3:
            await update.message.reply_text("⚠️ Usage: /addstock SYMBOL SHARES PRICE\nExample: /addstock VNM 500 80000")
            return
        
        symbol = context.args[0].upper()
        shares = int(context.args[1])
        price = float(context.args[2])
        
        from portfolio_manager import PortfolioManager
        manager = PortfolioManager()
        manager.add_stock(symbol, shares, price)
        
        await update.message.reply_text(f"✅ Đã thêm {shares} CP {symbol} với giá {price:,.0f} VNĐ")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {e}")

async def sell_stock_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /sellstock symbol [shares] - Bán cổ phiếu"""
    try:
        if not context.args or len(context.args) < 1:
            await update.message.reply_text("⚠️ Usage: /sellstock SYMBOL [SHARES]\nExample: /sellstock VNM 500")
            return
        
        symbol = context.args[0].upper()
        shares = int(context.args[1]) if len(context.args) > 1 else None
        
        from portfolio_manager import PortfolioManager
        manager = PortfolioManager()
        success = manager.remove_stock(symbol, shares)
        
        if success:
            shares_msg = f"{shares} CP" if shares else "tất cả"
            await update.message.reply_text(f"✅ Đã bán {shares_msg} {symbol}")
        else:
            await update.message.reply_text(f"❌ Không thể bán {symbol}")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {e}")

async def run_bot_async():
    """Hàm async chạy bot"""
    print("✅ Telegram Bot đang khởi động...")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Đăng ký tất cả command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("run", run))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(CommandHandler("addstock", add_stock_command))
    app.add_handler(CommandHandler("sellstock", sell_stock_command))
    
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
    backoff = 5
    while True:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_bot_async())
            break  # Thoát nếu shutdown bình thường
        except TgNetworkError as e:
            print(f"⚠️ Mất kết nối Telegram: {e}. Thử lại sau {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
        except Exception as e:
            print(f"❌ Lỗi Telegram Bot: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
        finally:
            loop.close()