import asyncio
import logging
import time

from src.core.bot_runner import run_bot_with_context
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import NetworkError as TgNetworkError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from src.config.legacy_config import TELEGRAM_TOKEN

logger = logging.getLogger(__name__)

try:
    from news_analyzer import format_news_brief, get_hot_news
except ImportError:
    format_news_brief = None
    get_hot_news = None

try:
    from src.notifications.subscriptions import SubscriptionManager

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
            InlineKeyboardButton(
                "📈 Summary hôm nay", callback_data="action:dailysummary"
            ),
        ],
        [
            InlineKeyboardButton(
                "⚙️ Quản lý đăng ký", callback_data="action:subscriptions"
            ),
            InlineKeyboardButton(
                "📝 Paper Trading", callback_data="action:paperaccount"
            ),
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
        reply_markup=reply_markup,
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
        from src.portfolio.manager import PortfolioManager

        manager = PortfolioManager()
        analysis_report = manager.get_detailed_analysis()

        # Chia nhỏ message nếu cần
        if len(analysis_report) > 4000:
            parts = [
                analysis_report[i : i + 4000]
                for i in range(0, len(analysis_report), 4000)
            ]
            for part in parts:
                await update.message.reply_text(part, parse_mode="Markdown")
        else:
            await update.message.reply_text(analysis_report, parse_mode="Markdown")

    except Exception:
        await update.message.reply_text("❌ Lỗi")


async def add_stock_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /addstock symbol shares price - Thêm cổ phiếu hoặc mua thêm nếu đã có"""
    try:
        if not context.args or len(context.args) < 3:
            await update.message.reply_text(
                "⚠️ Usage: /addstock SYMBOL SHARES PRICE\n"
                "Example: /addstock VNM 500 80000\n\n"
                "💡 Nếu mã đã có position, sẽ mua thêm và trung bình giá (DCA)"
            )
            return

        symbol = context.args[0].upper()
        shares = int(context.args[1])
        price = float(context.args[2])

        # Validate inputs
        if shares <= 0:
            await update.message.reply_text("❌ Số lượng CP phải > 0")
            return

        if price <= 0:
            await update.message.reply_text("❌ Giá phải > 0")
            return

        from src.portfolio.manager import PortfolioManager
        from src.portfolio.paper_trading import get_paper_account

        manager = PortfolioManager()

        # Check if symbol already has position
        existing_positions = manager.get_positions()
        symbol_has_position = symbol in existing_positions

        if symbol_has_position:
            # Mua thêm (average up/DCA)
            existing_pos = existing_positions[symbol]
            old_shares = existing_pos.get("shares", 0)
            old_avg_price = existing_pos.get("avg_price", 0)

            manager.add_position(
                symbol=symbol,
                shares=shares,
                entry_price=price,
                metadata={
                    "source": "manual_addstock_command",
                    "old_shares": old_shares,
                },
            )

            # Update paper trading account to reflect the additional buy
            paper_account = get_paper_account()
            paper_account.execute_buy(
                symbol=symbol,
                shares=shares,
                price=price,
                signal_reason=f"Manual add via /addstock (DCA: {old_shares} @ {old_avg_price:,.0f})",
            )

            # Get updated position info
            updated_positions = manager.get_positions()
            if symbol in updated_positions:
                new_pos = updated_positions[symbol]
                new_shares = new_pos.get("shares", 0)
                new_avg_price = new_pos.get("avg_price", 0)

                await update.message.reply_text(
                    f"✅ Đã mua thêm {shares} CP {symbol} @ {price:,.0f} VNĐ\n\n"
                    f"📊 Position cập nhật:\n"
                    f"• Tổng CP: {old_shares} → {new_shares}\n"
                    f"• Giá TB: {old_avg_price:,.0f} → {new_avg_price:,.0f} VNĐ\n"
                    f"• Giá trị: {new_shares * new_avg_price:,.0f} VNĐ"
                )
            else:
                await update.message.reply_text(
                    f"✅ Đã mua thêm {shares} CP {symbol} @ {price:,.0f} VNĐ"
                )
        else:
            # Thêm position mới
            manager.add_position(
                symbol=symbol,
                shares=shares,
                entry_price=price,
                metadata={"source": "manual_addstock_command"},
            )

            # Update paper trading account
            paper_account = get_paper_account()
            paper_account.execute_buy(
                symbol=symbol,
                shares=shares,
                price=price,
                signal_reason="Manual add via /addstock",
            )

            await update.message.reply_text(
                f"✅ Đã thêm {shares} CP {symbol} với giá {price:,.0f} VNĐ\n"
                f"💰 Giá trị: {shares * price:,.0f} VNĐ"
            )

    except ValueError as e:
        await update.message.reply_text(f"❌ Lỗi nhập liệu: {str(e)}")
    except Exception as e:
        logger.error(f"Error in add_stock_command: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")


async def sell_stock_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /sellstock symbol [shares] [price] - Bán cổ phiếu

    - Nếu không nhập shares: bán toàn bộ
    - Nếu không nhập price: dùng giá hiện tại (gần nhất)
    - Nếu nhập shares < tổng đang có: bán một phần đúng số lượng chỉ định
    """
    try:
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "⚠️ Usage: /sellstock SYMBOL [SHARES] [PRICE]\n"
                "Ví dụ: /sellstock VNM 500 82000"
            )
            return

        symbol = context.args[0].upper()
        shares = None
        price = None

        if len(context.args) >= 2:
            try:
                shares = int(context.args[1])
            except ValueError:
                shares = None

        if len(context.args) >= 3:
            try:
                price = float(context.args[2])
            except ValueError:
                price = None

        from src.portfolio.manager import PortfolioManager
        from src.portfolio.paper_trading import get_paper_account

        manager = PortfolioManager()
        positions = manager.get_positions()
        if symbol not in positions:
            await update.message.reply_text(f"⚠️ Không có {symbol} trong portfolio")
            return

        pos = positions[symbol]
        available_shares = pos.get("shares", 0)

        if shares is not None and (shares <= 0 or shares > available_shares):
            await update.message.reply_text(
                f"⚠️ Số lượng hợp lệ: 1..{available_shares} (đang có {available_shares})"
            )
            return

        paper_account = get_paper_account()

        # Execute sell via paper account (handles DB updates and cash)
        if shares is None or shares == available_shares:
            success, message, _ = paper_account.execute_sell(
                symbol=symbol,
                price=price,
                exit_type="FULL",
                reason="Manual sell via /sellstock",
            )
        else:
            success, message, _ = paper_account.execute_sell(
                symbol=symbol,
                shares=shares,
                price=price,
                exit_type="PARTIAL",  # ignored for direct partial, used in message
                reason="Manual partial sell via /sellstock",
            )

        if success:
            await update.message.reply_text(f"✅ {message}")
        else:
            await update.message.reply_text(f"❌ {message}")

    except Exception as e:
        logger.error(f"Error in sell_stock_command: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")


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
                InlineKeyboardButton(
                    "📈 Chart",
                    url=f"https://www.tradingview.com/chart/?symbol=HOSE:{symbol}",
                ),
                InlineKeyboardButton("🔔 Đăng ký", callback_data=f"subscribe:{symbol}"),
            ],
            [
                InlineKeyboardButton(
                    "💼 Thêm vào portfolio", callback_data=f"add:{symbol}"
                ),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(message, reply_markup=reply_markup)
    except Exception:
        await update.message.reply_text(f"❌ Lỗi lấy tin tức cho {symbol}")


# ===== SUBSCRIPTION COMMANDS =====
async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /subscribe SYMBOL - Đăng ký nhận tin"""
    if subscription_manager is None:
        await update.message.reply_text("⚠️ Subscription manager chưa sẵn sàng.")
        return

    if not context.args:
        await update.message.reply_text(
            "⚠️ Usage: /subscribe SYMBOL\nVí dụ: /subscribe VNM"
        )
        return

    symbol = context.args[0].upper()
    user_id = update.effective_user.id

    try:
        subscription_manager.subscribe_symbol(user_id, symbol)
        await update.message.reply_text(f"✅ Đã đăng ký nhận tin cho {symbol}!")
    except Exception:
        await update.message.reply_text("❌ Lỗi")


async def unsubscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /unsubscribe SYMBOL - Hủy đăng ký"""
    if subscription_manager is None:
        await update.message.reply_text("⚠️ Subscription manager chưa sẵn sàng.")
        return

    if not context.args:
        await update.message.reply_text(
            "⚠️ Usage: /unsubscribe SYMBOL\nVí dụ: /unsubscribe VNM"
        )
        return

    symbol = context.args[0].upper()
    user_id = update.effective_user.id

    try:
        subscription_manager.unsubscribe_symbol(user_id, symbol)
        await update.message.reply_text(f"✅ Đã hủy đăng ký nhận tin cho {symbol}!")
    except Exception:
        await update.message.reply_text("❌ Lỗi")


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
async def _handle_run_action(query, context):
    """Handle run bot action"""
    await query.edit_message_text("📊 Đang lấy dữ liệu và phân tích...")
    await run_bot_with_context(context.bot, query.message.chat_id)
    await query.edit_message_text("✅ Đã hoàn thành phân tích!")


async def _handle_portfolio_action(query, context):
    """Handle portfolio display action"""
    try:
        from src.portfolio.manager import PortfolioManager

        manager = PortfolioManager()
        analysis_report = manager.get_detailed_analysis()

        if len(analysis_report) > 4000:
            await query.edit_message_text(analysis_report[:4000], parse_mode="Markdown")
            await context.bot.send_message(
                query.message.chat_id,
                analysis_report[4000:],
                parse_mode="Markdown",
            )
        else:
            await query.edit_message_text(analysis_report, parse_mode="Markdown")
    except Exception:
        await query.edit_message_text("❌ Lỗi")


async def _handle_hotnews_action(query):
    """Handle hot news display action"""
    if get_hot_news:
        try:
            hot_news = get_hot_news(limit=5)
            if hot_news:
                msg = "🔥 Tin nóng hôm nay:\n\n"
                for news in hot_news:
                    msg += f"• {news.get('title', 'N/A')}\n"
                    msg += f"  {news.get('symbol', '_market')} | "
                    msg += f"{news.get('sentiment', 0):+.2f}\n\n"
                await query.edit_message_text(msg)
            else:
                await query.edit_message_text("📭 Không có tin nóng hôm nay.")
        except Exception:
            await query.edit_message_text("❌ Lỗi")
    else:
        await query.edit_message_text("⚠️ Module tin tức chưa sẵn sàng.")


async def _handle_dailysummary_action(query):
    """Handle daily summary generation action"""
    await query.edit_message_text("📊 Đang tạo summary...")
    try:
        summary = await generate_daily_summary()
        await query.edit_message_text(summary, parse_mode="Markdown")
    except Exception:
        await query.edit_message_text("❌ Lỗi")


async def _handle_subscriptions_action(query, user_id):
    """Handle subscriptions display action"""
    if subscription_manager:
        subs = subscription_manager.get_user_subscriptions(user_id)
        msg = "⚙️ Quản lý đăng ký:\n\n"
        msg += f"📈 Mã: {len(subs['symbols'])}\n"
        msg += f"🏢 Ngành: {len(subs['sectors'])}\n\n"
        msg += "Dùng /subscribe SYMBOL để đăng ký"
        await query.edit_message_text(msg)
    else:
        await query.edit_message_text("⚠️ Subscription manager chưa sẵn sàng.")


async def _handle_papertrades_action(query):
    """Handle paper trades history action"""
    try:
        from src.portfolio.paper_trading import get_paper_account

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

        await query.edit_message_text(msg, parse_mode="Markdown")
    except Exception:
        await query.edit_message_text("❌ Lỗi")


async def _handle_paperper_action(query):
    """Handle paper performance display action"""
    try:
        from src.portfolio.paper_trading import get_paper_account

        account = get_paper_account()
        stats = account.get_statistics()

        msg = "📈 *Paper Trading Performance:*\n\n"
        msg += f"💰 P&L: {stats['current_pnl']:+,.0f} VNĐ\n"
        msg += f"📊 Return: {stats['current_return_pct']:+.2f}%\n"
        msg += f"🔄 Trades: {stats['total_trades']}\n"
        msg += f"💸 Phí: {stats['total_commission']:,.0f} VNĐ"

        await query.edit_message_text(msg, parse_mode="Markdown")
    except Exception:
        await query.edit_message_text("❌ Lỗi")


async def _handle_paperaccount_action(query):
    """Handle paper account summary action"""
    try:
        from src.portfolio.paper_trading import get_paper_account

        account = get_paper_account()
        summary = account.format_account_summary()

        keyboard = [
            [
                InlineKeyboardButton(
                    "📊 Trade History", callback_data="action:papertrades"
                ),
                InlineKeyboardButton("📈 Performance", callback_data="action:paperper"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            summary, parse_mode="Markdown", reply_markup=reply_markup
        )
    except Exception:
        await query.edit_message_text("❌ Lỗi")


async def _handle_subscribe_callback(query, user_id, symbol):
    """Handle subscribe button callback"""
    if subscription_manager:
        subscription_manager.subscribe_symbol(user_id, symbol)
        await query.edit_message_text(f"✅ Đã đăng ký nhận tin cho {symbol}!")
    else:
        await query.answer("⚠️ Subscription manager chưa sẵn sàng.", show_alert=True)


async def _handle_add_callback(query, symbol):
    """Handle add to portfolio button callback"""
    await query.edit_message_text(
        f"💼 Thêm {symbol} vào portfolio:\n\n"
        f"Dùng lệnh: /addstock {symbol} SHARES PRICE"
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý callback từ inline buttons"""
    query = update.callback_query
    await query.answer()

    data = query.data
    user_id = query.from_user.id

    # Action handlers mapping
    action_handlers = {
        "run": lambda: _handle_run_action(query, context),
        "portfolio": lambda: _handle_portfolio_action(query, context),
        "hotnews": lambda: _handle_hotnews_action(query),
        "dailysummary": lambda: _handle_dailysummary_action(query),
        "subscriptions": lambda: _handle_subscriptions_action(query, user_id),
        "papertrades": lambda: _handle_papertrades_action(query),
        "paperper": lambda: _handle_paperper_action(query),
        "paperaccount": lambda: _handle_paperaccount_action(query),
    }

    if data.startswith("action:"):
        action = data.split(":")[1]
        handler = action_handlers.get(action)
        if handler:
            await handler()

    elif data.startswith("subscribe:"):
        symbol = data.split(":")[1]
        await _handle_subscribe_callback(query, user_id, symbol)

    elif data.startswith("add:"):
        symbol = data.split(":")[1]
        await _handle_add_callback(query, symbol)


async def send_daily_summary_to_all():
    """Gửi daily summary cho tất cả users (called from scheduler)"""
    try:
        from src.notifications.telegram import send_daily_summary_to_all as send_all

        await send_all()
    except ImportError:
        pass


async def generate_daily_summary() -> str:
    """Tạo daily summary: hiệu suất, tin tức, watchlist"""
    from datetime import datetime

    import pytz

    msg_parts = []
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(tz)

    # Header
    msg_parts.append("📊 *DAILY SUMMARY*\n")
    msg_parts.append(f"📅 {now.strftime('%d/%m/%Y %H:%M')}\n")
    msg_parts.append(f"{'='*30}\n\n")

    # 1. Portfolio Performance
    try:
        from src.portfolio.manager import PortfolioManager

        manager = PortfolioManager()
        pv = manager.get_portfolio_value()

        total_value = pv.get("total_value", 0)
        total_cost = pv.get("total_cost", 0)
        pnl = pv.get("pnl", total_value - total_cost)
        pnl_pct = pv.get(
            "pnl_percent", (pnl / total_cost * 100) if total_cost > 0 else 0
        )
        num_positions = pv.get("num_positions", 0)

        msg_parts.append("💼 *PORTFOLIO*\n")
        if num_positions > 0:
            msg_parts.append(f"💰 Giá trị: {total_value:,.0f} VNĐ\n")
            msg_parts.append(f"💵 Vốn: {total_cost:,.0f} VNĐ\n")
            msg_parts.append(
                f"{'📈' if pnl >= 0 else '📉'} P&L: {pnl:+,.0f} VNĐ ({pnl_pct:+.2f}%)\n"
            )
            msg_parts.append(f"📊 Số mã: {num_positions}\n")

            # Chi tiết theo từng mã (dựa trên giá hiện có trong metadata nếu có)
            try:
                positions = manager.get_positions()
                if positions:
                    msg_parts.append("\n📋 *CHI TIẾT VỊ THẾ*\n")

                    # Sắp xếp theo P&L % giảm dần nếu có giá hiện tại
                    def calc_pnl_tuple(item):
                        sym, pos = item
                        shares = pos.get("shares", 0) or 0
                        avg_price = pos.get("avg_price", 0) or 0
                        last_price = (
                            pos.get("metadata", {}).get("last_price", avg_price)
                            or avg_price
                        )
                        cost = shares * avg_price
                        value = shares * last_price
                        pnl_abs = value - cost
                        pnl_percent = (pnl_abs / cost * 100) if cost > 0 else 0
                        return (pnl_percent, pnl_abs)

                    sorted_items = sorted(
                        positions.items(), key=calc_pnl_tuple, reverse=True
                    )
                    limit = min(len(sorted_items), 10)
                    for i in range(limit):
                        sym, pos = sorted_items[i]
                        shares = pos.get("shares", 0) or 0
                        avg_price = pos.get("avg_price", 0) or 0
                        last_price = (
                            pos.get("metadata", {}).get("last_price", avg_price)
                            or avg_price
                        )
                        cost = shares * avg_price
                        value = shares * last_price
                        pnl_abs = value - cost
                        pnl_percent = (pnl_abs / cost * 100) if cost > 0 else 0
                        emoji = "📈" if pnl_abs >= 0 else "📉"
                        msg_parts.append(
                            f"{emoji} {sym}: {pnl_abs:+,.0f} VNĐ ({pnl_percent:+.2f}%) | "
                            f"{shares} CP @ {avg_price:,.0f} → {last_price:,.0f}\n"
                        )
                    if len(sorted_items) > limit:
                        msg_parts.append(
                            f"... và {len(sorted_items) - limit} mã khác\n"
                        )
            except Exception:
                # Bỏ qua chi tiết nếu có lỗi nhỏ, vẫn giữ phần tổng
                pass
        else:
            msg_parts.append("Chưa có vị thế nào\n")
    except Exception as e:
        msg_parts.append("💼 *PORTFOLIO*\n")
        msg_parts.append(f"⚠️ Lỗi: {str(e)[:50]}\n")

    # 2. Market Regime
    try:
        from src.market.regime_proxy import ProxyMarketRegimeAnalyzer

        regime_analyzer = ProxyMarketRegimeAnalyzer()
        regime = regime_analyzer.analyze_market_regime()

        msg_parts.append("\n📈 *THỊ TRƯỜNG*\n")
        msg_parts.append(f"Trạng thái: {regime.get('regime', 'UNKNOWN')}\n")
        msg_parts.append(f"Confidence: {regime.get('confidence', 0):.0f}%\n")
        msg_parts.append(
            f"Tradeable: {'✅' if regime.get('tradeable', False) else '❌'}\n"
        )
    except Exception:
        logger.warning("Could not fetch market regime for summary")

    # 3. Hot News
    if get_hot_news:
        try:
            hot_news = get_hot_news(limit=3)
            if hot_news:
                msg_parts.append("\n🔥 *TIN TỨC*\n")
                for news in hot_news:
                    symbol = news.get("symbol", "_market")
                    title = news.get("title", "N/A")[:40]
                    sentiment = news.get("sentiment", 0)
                    emoji = "📈" if sentiment > 0 else "📉" if sentiment < 0 else "➖"
                    msg_parts.append(f"{emoji} {symbol}: {title}...\n")
        except Exception:
            logger.warning("Could not fetch hot news for summary")

    # 4. Paper Trading
    try:
        from src.portfolio.paper_trading import get_paper_account

        paper = get_paper_account()
        stats = paper.get_statistics()
        pv = paper.get_portfolio_value() or {}
        cash = paper.account.get("cash", 0) if hasattr(paper, "account") else 0
        current_value = pv.get("total_value", 0)
        total_equity = current_value + cash

        if stats:
            msg_parts.append("\n📝 *PAPER TRADING*\n")
            msg_parts.append(f"💵 Tiền mặt: {cash:,.0f} VNĐ\n")
            msg_parts.append(f"📈 Giá trị vị thế: {current_value:,.0f} VNĐ\n")
            msg_parts.append(f"💼 Tổng tài sản: {total_equity:,.0f} VNĐ\n")
            msg_parts.append(f"🔄 Giao dịch: {stats.get('total_trades', 0)}\n")
            msg_parts.append(
                f"📊 Return: {stats.get('current_return_pct', 0):+.2f}% | "
                f"Vị thế: {stats.get('num_positions', 0)} | "
                f"Phí: {stats.get('total_commission', 0):,.0f} VNĐ\n"
            )
    except Exception:
        logger.warning("Could not fetch paper trading stats for summary")

    # Footer
    msg_parts.append(f"\n{'='*30}\n")
    msg_parts.append("💡 Dùng /run để scan tín hiệu mới\n")
    msg_parts.append("💼 Dùng /portfolio để xem chi tiết\n")

    return "".join(msg_parts) if msg_parts else "📊 Summary sẽ có sau khi có dữ liệu."


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /summary - Summary cuối ngày"""
    try:
        summary = await generate_daily_summary()
        await update.message.reply_text(summary, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text("❌ Lỗi")


# ===== PAPER TRADING COMMANDS =====
async def paper_account_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /paper - Xem paper trading account"""
    try:
        from src.portfolio.paper_trading import get_paper_account

        account = get_paper_account()

        # Defensive: ensure essential fields exist
        if not hasattr(account, "format_account_summary"):
            await update.message.reply_text(
                "❌ PaperTradingAccount thiếu format_account_summary()"
            )
            return

        try:
            summary = account.format_account_summary()
        except Exception as e:
            summary = (
                "📊 Paper Trading Account\n"
                f"(không thể format chi tiết: {type(e).__name__}: {str(e)})"
            )

        keyboard = [
            [
                InlineKeyboardButton(
                    "📊 Trade History", callback_data="action:papertrades"
                ),
                InlineKeyboardButton("📈 Performance", callback_data="action:paperper"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Try Markdown first, then fallback to plain text if it fails
        try:
            await update.message.reply_text(
                summary, parse_mode="Markdown", reply_markup=reply_markup
            )
        except Exception:
            await update.message.reply_text(summary, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"/paper command failed: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")


async def paper_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /papertrades - Xem lịch sử giao dịch paper trading"""
    try:
        from src.portfolio.paper_trading import get_paper_account

        account = get_paper_account()
        trades = account.get_trade_history()

        if not trades:
            await update.message.reply_text(
                "📭 Chưa có giao dịch nào trong paper trading account."
            )
            return

        # Normalize trades to dicts (DB rows may be tuples or sqlite rows)
        def to_dict(t):
            try:
                if isinstance(t, dict):
                    return t
                if hasattr(t, "keys"):
                    return dict(t)
                if isinstance(t, (list, tuple)):
                    # Heuristic mapping from save_trade: (symbol, action, shares, price, total_value, trade_date, reason, metadata)
                    return {
                        "symbol": t[0] if len(t) > 0 else "",
                        "action": t[1] if len(t) > 1 else "",
                        "shares": t[2] if len(t) > 2 else 0,
                        "price": t[3] if len(t) > 3 else 0,
                        "timestamp": t[5] if len(t) > 5 else "",
                    }
            except Exception:
                pass
            return {
                "symbol": "",
                "action": "",
                "shares": 0,
                "price": 0,
                "timestamp": "",
            }

        normalized = [to_dict(t) for t in trades]

        # Show last 10 trades
        recent_trades = normalized[:10]
        msg = "📊 *Paper Trading History (10 gần nhất):*\n\n"

        for trade in recent_trades:
            try:
                action = str(trade.get("action", "")).upper()
                symbol = str(trade.get("symbol", ""))
                shares = int(trade.get("shares", 0) or 0)
                price = float(trade.get("price", 0) or 0)
                timestamp = str(trade.get("timestamp", ""))[:16]

                emoji = "🟢" if action == "BUY" else "🔴" if "SELL" in action else "⚪"
                msg += f"{emoji} {action} {shares} CP {symbol} @ {price:,.0f} VNĐ\n"
                if timestamp:
                    msg += f"   {timestamp}\n\n"
                else:
                    msg += "\n"
            except Exception:
                # Skip malformed trade rows but keep listing others
                continue

        # Try Markdown first, fallback to plain if it fails
        try:
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(msg)
    except Exception as e:
        logger.error(f"/papertrades command failed: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")


async def run_bot_async():
    """Hàm async chạy bot"""
    print("✅ Telegram Bot đang khởi động...")

    # Validate token before starting
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN.strip() == "":
        raise ValueError(
            "TELEGRAM_TOKEN is not set or empty. "
            "Please set TELEGRAM_TOKEN environment variable or in .env file. "
            "Get your token from https://t.me/Botfather"
        )

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Set bot instance cho notifications
    try:
        from src.notifications.telegram import set_bot_instance

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
        except ValueError as e:
            # Configuration error - don't retry
            print(f"❌ Lỗi cấu hình Telegram Bot: {str(e)}")
            print("💡 Vui lòng kiểm tra TELEGRAM_TOKEN trong .env file")
            break
        except TgNetworkError:
            print(f"⚠️ Mất kết nối Telegram. Thử lại sau {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
        except Exception as e:
            print(f"❌ Lỗi Telegram Bot: {str(e)}")
            import traceback

            traceback.print_exc()
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
        finally:
            loop.close()
