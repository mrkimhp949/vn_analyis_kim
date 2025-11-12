import asyncio
import time
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import NetworkError as TgNetworkError
from config import TELEGRAM_TOKEN, CHAT_ID
from bot_runner_improved import run_bot_with_context
try:
    from news_analyzer import format_news_brief, get_hot_news
except ImportError:
    format_news_brief = None
    get_hot_news = None

try:
    from telegram_subscriptions import SubscriptionManager
    subscription_manager = SubscriptionManager()
except ImportError:
    subscription_manager = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("📊 Quét tín hiệu", callback_data="action:run"),
            InlineKeyboardButton("💼 Portfolio", callback_data="action:portfolio"),
        ],
        [
            InlineKeyboardButton("📰 Tin nóng", callback_data="action:hotnews"),
            InlineKeyboardButton("📈 Summary hôm nay", callback_data="action:dailysummary"),
        ],
        [
            InlineKeyboardButton("⚙️ Quản lý đăng ký", callback_data="action:subscriptions"),
            InlineKeyboardButton("📝 Paper Trading", callback_data="action:paperaccount"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🤖 Bot sẵn sàng!\n\n"
        "📋 Các lệnh:\n"
        "/run - Lấy tín hiệu giao dịch\n"
        "/portfolio - Xem portfolio\n" 
        "/addstock SYMBOL SHARES PRICE - Thêm cổ phiếu\n"
        "/sellstock SYMBOL [SHARES] - Bán cổ phiếu\n"
        "/news SYMBOL - Tin tức & sentiment\n"
        "/subscribe SYMBOL - Đăng ký nhận tin\n"
        "/unsubscribe SYMBOL - Hủy đăng ký\n"
        "/mysubs - Xem đăng ký của tôi\n"
        "/summary - Summary cuối ngày\n"
        "/paper - Paper trading account\n"
        "/papertrades - Lịch sử paper trading\n"
        "/status - Trạng thái bot\n\n"
        "💡 Hoặc dùng nút bên dưới để thao tác nhanh!",
        reply_markup=reply_markup
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
        success, message = manager.remove_stock(symbol, shares)
        
        if success:
            await update.message.reply_text(f"✅ {message}")
        else:
            await update.message.reply_text(f"❌ {message}")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {e}")

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /news SYMBOL - Tin tức & sentiment"""
    if format_news_brief is None:
        await update.message.reply_text("⚠️ Module tin tức chưa sẵn sàng.")
        return

    if not context.args:
        await update.message.reply_text("⚠️ Usage: /news SYMBOL\nVí dụ: /news VNM")
        return

    symbol = context.args[0].upper()
    try:
        message = format_news_brief(symbol)
        
        # Thêm inline buttons
        keyboard = [
            [
                InlineKeyboardButton("📈 Chart", url=f"https://www.tradingview.com/chart/?symbol=HOSE:{symbol}"),
                InlineKeyboardButton("🔔 Đăng ký", callback_data=f"subscribe:{symbol}"),
            ],
            [
                InlineKeyboardButton("💼 Thêm vào portfolio", callback_data=f"add:{symbol}"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup)
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi lấy tin tức cho {symbol}: {e}")

# ===== SUBSCRIPTION COMMANDS =====
async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /subscribe SYMBOL - Đăng ký nhận tin"""
    if subscription_manager is None:
        await update.message.reply_text("⚠️ Subscription manager chưa sẵn sàng.")
        return
    
    if not context.args:
        await update.message.reply_text("⚠️ Usage: /subscribe SYMBOL\nVí dụ: /subscribe VNM")
        return
    
    symbol = context.args[0].upper()
    user_id = update.effective_user.id
    
    try:
        subscription_manager.subscribe_symbol(user_id, symbol)
        await update.message.reply_text(f"✅ Đã đăng ký nhận tin cho {symbol}!")
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {e}")

async def unsubscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /unsubscribe SYMBOL - Hủy đăng ký"""
    if subscription_manager is None:
        await update.message.reply_text("⚠️ Subscription manager chưa sẵn sàng.")
        return
    
    if not context.args:
        await update.message.reply_text("⚠️ Usage: /unsubscribe SYMBOL\nVí dụ: /unsubscribe VNM")
        return
    
    symbol = context.args[0].upper()
    user_id = update.effective_user.id
    
    try:
        subscription_manager.unsubscribe_symbol(user_id, symbol)
        await update.message.reply_text(f"✅ Đã hủy đăng ký nhận tin cho {symbol}!")
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {e}")

async def mysubs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /mysubs - Xem đăng ký của tôi"""
    if subscription_manager is None:
        await update.message.reply_text("⚠️ Subscription manager chưa sẵn sàng.")
        return
    
    user_id = update.effective_user.id
    subs = subscription_manager.get_user_subscriptions(user_id)
    
    if not subs["symbols"] and not subs["sectors"]:
        await update.message.reply_text("📭 Bạn chưa đăng ký nhận tin cho mã nào.")
        return
    
    msg = "📋 Đăng ký của bạn:\n\n"
    if subs["symbols"]:
        msg += f"📈 Mã cổ phiếu ({len(subs['symbols'])}):\n"
        msg += ", ".join(subs["symbols"]) + "\n\n"
    if subs["sectors"]:
        msg += f"🏢 Ngành ({len(subs['sectors'])}):\n"
        msg += ", ".join(subs["sectors"])
    
    await update.message.reply_text(msg)

# ===== INLINE BUTTON HANDLERS =====
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý callback từ inline buttons"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user_id = query.from_user.id
    
    if data.startswith("action:"):
        action = data.split(":")[1]
        
        if action == "run":
            await query.edit_message_text("📊 Đang lấy dữ liệu và phân tích...")
            await run_bot_with_context(context.bot, query.message.chat_id)
            await query.edit_message_text("✅ Đã hoàn thành phân tích!")
        
        elif action == "portfolio":
            try:
                from portfolio_manager import PortfolioManager
                manager = PortfolioManager()
                analysis_report = manager.get_detailed_analysis()
                
                if len(analysis_report) > 4000:
                    await query.edit_message_text(analysis_report[:4000], parse_mode='Markdown')
                    await context.bot.send_message(query.message.chat_id, analysis_report[4000:], parse_mode='Markdown')
                else:
                    await query.edit_message_text(analysis_report, parse_mode='Markdown')
            except Exception as e:
                await query.edit_message_text(f"❌ Lỗi: {e}")
        
        elif action == "hotnews":
            if get_hot_news:
                try:
                    hot_news = get_hot_news(limit=5)
                    if hot_news:
                        msg = "🔥 Tin nóng hôm nay:\n\n"
                        for news in hot_news:
                            msg += f"• {news.get('title', 'N/A')}\n"
                            msg += f"  {news.get('symbol', '_market')} | {news.get('sentiment', 0):+.2f}\n\n"
                        await query.edit_message_text(msg)
                    else:
                        await query.edit_message_text("📭 Không có tin nóng hôm nay.")
                except Exception as e:
                    await query.edit_message_text(f"❌ Lỗi: {e}")
            else:
                await query.edit_message_text("⚠️ Module tin tức chưa sẵn sàng.")
        
        elif action == "dailysummary":
            await query.edit_message_text("📊 Đang tạo summary...")
            try:
                summary = await generate_daily_summary()
                await query.edit_message_text(summary, parse_mode='Markdown')
            except Exception as e:
                await query.edit_message_text(f"❌ Lỗi: {e}")
        
        elif action == "subscriptions":
            if subscription_manager:
                subs = subscription_manager.get_user_subscriptions(user_id)
                msg = "⚙️ Quản lý đăng ký:\n\n"
                msg += f"📈 Mã: {len(subs['symbols'])}\n"
                msg += f"🏢 Ngành: {len(subs['sectors'])}\n\n"
                msg += "Dùng /subscribe SYMBOL để đăng ký"
                await query.edit_message_text(msg)
            else:
                await query.edit_message_text("⚠️ Subscription manager chưa sẵn sàng.")
        
        elif action == "papertrades":
            try:
                from paper_trading import get_paper_account
                account = get_paper_account()
                trades = account.get_trade_history()
                
                if not trades:
                    await query.edit_message_text("📭 Chưa có giao dịch nào.")
                    return
                
                recent_trades = trades[:5]
                msg = "📊 *Paper Trading (5 gần nhất):*\n\n"
                for trade in recent_trades:
                    action = trade.get("action", "")
                    symbol = trade.get("symbol", "")
                    shares = trade.get("shares", 0)
                    price = trade.get("price", 0)
                    emoji = "🟢" if action == "BUY" else "🔴"
                    msg += f"{emoji} {action} {shares} CP {symbol} @ {price:,.0f}\n"
                
                await query.edit_message_text(msg, parse_mode='Markdown')
            except Exception as e:
                await query.edit_message_text(f"❌ Lỗi: {e}")
        
        elif action == "paperperf":
            try:
                from paper_trading import get_paper_account
                account = get_paper_account()
                stats = account.get_statistics()
                
                msg = "📈 *Paper Trading Performance:*\n\n"
                msg += f"💰 P&L: {stats['current_pnl']:+,.0f} VNĐ\n"
                msg += f"📊 Return: {stats['current_return_pct']:+.2f}%\n"
                msg += f"🔄 Trades: {stats['total_trades']}\n"
                msg += f"💸 Phí: {stats['total_commission']:,.0f} VNĐ"
                
                await query.edit_message_text(msg, parse_mode='Markdown')
            except Exception as e:
                await query.edit_message_text(f"❌ Lỗi: {e}")
        
        elif action == "paperaccount":
            try:
                from paper_trading import get_paper_account
                account = get_paper_account()
                summary = account.format_account_summary()
                
                keyboard = [[
                    InlineKeyboardButton("📊 Trade History", callback_data="action:papertrades"),
                    InlineKeyboardButton("📈 Performance", callback_data="action:paperperf"),
                ]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(summary, parse_mode='Markdown', reply_markup=reply_markup)
            except Exception as e:
                await query.edit_message_text(f"❌ Lỗi: {e}")
    
    elif data.startswith("subscribe:"):
        symbol = data.split(":")[1]
        if subscription_manager:
            subscription_manager.subscribe_symbol(user_id, symbol)
            await query.edit_message_text(f"✅ Đã đăng ký nhận tin cho {symbol}!")
        else:
            await query.answer("⚠️ Subscription manager chưa sẵn sàng.", show_alert=True)
    
    elif data.startswith("add:"):
        symbol = data.split(":")[1]
        await query.edit_message_text(f"💼 Thêm {symbol} vào portfolio:\n\nDùng lệnh: /addstock {symbol} SHARES PRICE")

async def send_daily_summary_to_all():
    """Gửi daily summary cho tất cả users (called from scheduler)"""
    try:
        from telegram_notifications import send_daily_summary_to_all as send_all
        await send_all()
    except ImportError:
        pass

async def generate_daily_summary() -> str:
    """Tạo daily summary: hiệu suất, tin tức, watchlist"""
    msg_parts = []
    
    # 1. Portfolio Performance
    try:
        from portfolio_manager import PortfolioManager
        manager = PortfolioManager()
        portfolio = manager.portfolio
        
        if portfolio:
            total_value = sum(pos.get("total_value", 0) for pos in portfolio.values())
            total_cost = sum(pos.get("total_cost", 0) for pos in portfolio.values())
            pnl = total_value - total_cost
            pnl_pct = (pnl / total_cost * 100) if total_cost > 0 else 0
            
            msg_parts.append(f"💼 *Portfolio hôm nay:*\n")
            msg_parts.append(f"💰 Tổng giá trị: {total_value:,.0f} VNĐ\n")
            msg_parts.append(f"{'📈' if pnl >= 0 else '📉'} P&L: {pnl:+,.0f} VNĐ ({pnl_pct:+.2f}%)\n")
            msg_parts.append(f"📊 Số mã: {len(portfolio)}\n")
    except Exception as e:
        msg_parts.append(f"⚠️ Lỗi portfolio: {e}\n")
    
    # 2. Hot News
    if get_hot_news:
        try:
            hot_news = get_hot_news(limit=3)
            if hot_news:
                msg_parts.append(f"\n🔥 *Tin nóng:*\n")
                for news in hot_news:
                    symbol = news.get("symbol", "_market")
                    title = news.get("title", "N/A")[:50]
                    sentiment = news.get("sentiment", 0)
                    msg_parts.append(f"• {symbol}: {title}... ({sentiment:+.2f})\n")
        except Exception:
            pass
    
    # 3. Watchlist (top signals từ lần scan gần nhất)
    msg_parts.append(f"\n👀 *Watchlist:*\n")
    msg_parts.append(f"Dùng /run để xem tín hiệu mới nhất")
    
    return "".join(msg_parts) if msg_parts else "📊 Summary sẽ có sau khi có dữ liệu."

async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /summary - Summary cuối ngày"""
    try:
        summary = await generate_daily_summary()
        await update.message.reply_text(summary, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {e}")

# ===== PAPER TRADING COMMANDS =====
async def paper_account_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /paper - Xem paper trading account"""
    try:
        from paper_trading import get_paper_account
        account = get_paper_account()
        summary = account.format_account_summary()
        
        keyboard = [[
            InlineKeyboardButton("📊 Trade History", callback_data="action:papertrades"),
            InlineKeyboardButton("📈 Performance", callback_data="action:paperperf"),
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(summary, parse_mode='Markdown', reply_markup=reply_markup)
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {e}")

async def paper_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /papertrades - Xem lịch sử giao dịch paper trading"""
    try:
        from paper_trading import get_paper_account
        account = get_paper_account()
        trades = account.get_trade_history()
        
        if not trades:
            await update.message.reply_text("📭 Chưa có giao dịch nào trong paper trading account.")
            return
        
        # Show last 10 trades
        recent_trades = trades[:10]
        msg = "📊 *Paper Trading History (10 gần nhất):*\n\n"
        
        for trade in recent_trades:
            action = trade.get("action", "")
            symbol = trade.get("symbol", "")
            shares = trade.get("shares", 0)
            price = trade.get("price", 0)
            timestamp = trade.get("timestamp", "")[:16]
            
            emoji = "🟢" if action == "BUY" else "🔴"
            msg += f"{emoji} {action} {shares} CP {symbol} @ {price:,.0f} VNĐ\n"
            msg += f"   {timestamp}\n\n"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {e}")

async def run_bot_async():
    """Hàm async chạy bot"""
    print("✅ Telegram Bot đang khởi động...")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Set bot instance cho notifications
    try:
        from telegram_notifications import set_bot_instance
        set_bot_instance(app.bot)
    except ImportError:
        pass
    
    # Đăng ký tất cả command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("run", run))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("portfolio", portfolio_command))
    app.add_handler(CommandHandler("addstock", add_stock_command))
    app.add_handler(CommandHandler("sellstock", sell_stock_command))
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("subscribe", subscribe_command))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe_command))
    app.add_handler(CommandHandler("mysubs", mysubs_command))
    app.add_handler(CommandHandler("summary", summary_command))
    app.add_handler(CommandHandler("paper", paper_account_command))
    app.add_handler(CommandHandler("papertrades", paper_trades_command))
    
    # Inline button handlers
    app.add_handler(CallbackQueryHandler(button_callback))
    
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