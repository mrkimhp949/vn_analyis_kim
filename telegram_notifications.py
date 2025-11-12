"""
Telegram Notifications Helper
Gửi notifications cho subscribers khi có tin tức/signals mới
"""
import asyncio
from typing import List, Optional
from telegram import Bot
from telegram.error import TelegramError

try:
    from telegram_subscriptions import SubscriptionManager
    from config import TELEGRAM_TOKEN
    subscription_manager = SubscriptionManager()
    bot_instance = None  # Will be set by main app
except ImportError:
    subscription_manager = None
    bot_instance = None


async def send_notification_to_subscribers(symbol: str, message: str, include_chart_button: bool = True):
    """
    Gửi notification cho tất cả users đăng ký nhận tin cho symbol
    
    Args:
        symbol: Mã cổ phiếu
        message: Nội dung message
        include_chart_button: Có thêm button xem chart không
    """
    if not subscription_manager or not bot_instance:
        return
    
    try:
        subscribers = subscription_manager.get_symbol_subscribers(symbol)
        
        if not subscribers:
            return
        
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        keyboard = None
        if include_chart_button:
            keyboard = [[
                InlineKeyboardButton("📈 Chart", url=f"https://www.tradingview.com/chart/?symbol=HOSE:{symbol}"),
            ]]
            reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
        else:
            reply_markup = None
        
        # Gửi cho từng subscriber
        for user_id in subscribers:
            try:
                await bot_instance.send_message(
                    chat_id=user_id,
                    text=message,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            except TelegramError as e:
                print(f"⚠️ Không thể gửi notification cho user {user_id}: {e}")
            except Exception as e:
                print(f"❌ Lỗi gửi notification: {e}")
    
    except Exception as e:
        print(f"❌ Lỗi send_notification_to_subscribers: {e}")


async def send_daily_summary_to_user(user_id: int):
    """Gửi daily summary cho một user"""
    if not bot_instance:
        return
    
    try:
        from tg_listener import generate_daily_summary
        summary = await generate_daily_summary()
        
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [[
            InlineKeyboardButton("📊 Quét tín hiệu", callback_data="action:run"),
            InlineKeyboardButton("💼 Portfolio", callback_data="action:portfolio"),
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await bot_instance.send_message(
            chat_id=user_id,
            text=summary,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    except Exception as e:
        print(f"❌ Lỗi gửi daily summary cho user {user_id}: {e}")


async def send_daily_summary_to_all():
    """Gửi daily summary cho tất cả users có subscription"""
    if not subscription_manager or not bot_instance:
        return
    
    try:
        # Lấy tất cả users có subscription
        all_users = set()
        for user_data in subscription_manager.subscriptions.get("users", {}).values():
            # This is a bit hacky, but we need to get user IDs
            pass
        
        # Get all user IDs from subscriptions
        user_ids = set()
        for symbol, subscribers in subscription_manager.subscriptions.get("symbol_subscribers", {}).items():
            user_ids.update(subscribers)
        for sector, subscribers in subscription_manager.subscriptions.get("sector_subscribers", {}).items():
            user_ids.update(subscribers)
        
        # Also include default chat_id if no subscriptions
        from config import CHAT_ID
        if CHAT_ID:
            user_ids.add(str(CHAT_ID))
        
        # Gửi cho từng user
        for user_id_str in user_ids:
            try:
                user_id = int(user_id_str)
                await send_daily_summary_to_user(user_id)
            except ValueError:
                continue
            except Exception as e:
                print(f"⚠️ Lỗi gửi summary cho user {user_id_str}: {e}")
    
    except Exception as e:
        print(f"❌ Lỗi send_daily_summary_to_all: {e}")


def set_bot_instance(bot: Bot):
    """Set bot instance từ main app"""
    global bot_instance
    bot_instance = bot

