"""
Telegram Notifications Helper
Gửi notifications cho subscribers khi có tin tức/signals mới
"""

from telegram import Bot
from telegram.error import TelegramError

try:
    from src.notifications.subscriptions import SubscriptionManager

    subscription_manager = SubscriptionManager()
    bot_instance = None  # Will be set by main app
except ImportError:
    subscription_manager = None
    bot_instance = None


async def send_notification_to_subscribers(
    symbol: str, message: str, include_chart_button: bool = True
):
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
            keyboard = [
                [
                    InlineKeyboardButton(
                        "📈 Chart",
                        url=f"https://www.tradingview.com/chart/?symbol=HOSE:{symbol}",
                    ),
                ]
            ]
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
                    parse_mode="Markdown",
                )
            except TelegramError:
                print(f"⚠️ Không thể gửi notification cho user {user_id}")
            except Exception:
                print("❌ Lỗi gửi notification")

    except Exception:
        print("❌ Lỗi send_notification_to_subscribers")


async def send_daily_summary_to_user(user_id: int):
    """Gửi daily summary cho một user"""
    if not bot_instance:
        return

    try:
        from src.notifications.listener import generate_daily_summary

        summary = await generate_daily_summary()

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = [
            [
                InlineKeyboardButton("📊 Quét tín hiệu", callback_data="action:run"),
                InlineKeyboardButton("💼 Portfolio", callback_data="action:portfolio"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await bot_instance.send_message(
            chat_id=user_id,
            text=summary,
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )
    except Exception:
        print(f"❌ Lỗi gửi daily summary cho user {user_id}")


async def send_daily_summary_to_all():
    """Gửi daily summary cho tất cả users có subscription"""
    if not bot_instance:
        print("⚠️ Bot instance không có, không thể gửi daily summary")
        return

    try:
        from src.config.legacy_config import CHAT_ID

        # Get all user IDs from subscriptions (if available)
        user_ids = set()

        if subscription_manager:
            try:
                # Get all user IDs from subscriptions
                for symbol, subscribers in subscription_manager.subscriptions.get(
                    "symbol_subscribers", {}
                ).items():
                    user_ids.update(subscribers)
                for sector, subscribers in subscription_manager.subscriptions.get(
                    "sector_subscribers", {}
                ).items():
                    user_ids.update(subscribers)
            except Exception as e:
                print(f"⚠️ Lỗi lấy subscribers: {e}")

        # Always include default chat_id (even if no subscription manager)
        if CHAT_ID:
            user_ids.add(str(CHAT_ID))

        if not user_ids:
            print("⚠️ Không có user nào để gửi daily summary")
            return

        print(f"📤 Gửi daily summary cho {len(user_ids)} user(s)...")

        # Gửi cho từng user
        for user_id_str in user_ids:
            try:
                user_id = int(user_id_str)
                await send_daily_summary_to_user(user_id)
                print(f"✅ Đã gửi daily summary cho user {user_id}")
            except ValueError:
                print(f"⚠️ Invalid user_id: {user_id_str}")
                continue
            except Exception as e:
                print(f"⚠️ Lỗi gửi summary cho user {user_id_str}: {type(e).__name__}: {str(e)}")

    except Exception as e:
        import traceback

        print(f"❌ Lỗi send_daily_summary_to_all: {type(e).__name__}: {str(e)}")
        traceback.print_exc()


def set_bot_instance(bot: Bot):
    """Set bot instance từ main app"""
    global bot_instance
    bot_instance = bot
