"""
Notification Service
Handles all Telegram notifications
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from telegram import Bot

logger = logging.getLogger(__name__)


class NotificationService:
    """
    Service for notification operations

    Responsibilities:
    - Send entry signals
    - Send exit signals
    - Send scan summaries
    - Send risk alerts
    - Format messages
    """

    def __init__(self, bot: Bot, chat_id: str):
        self.bot = bot
        self.chat_id = chat_id

        logger.info("✅ Notification Service initialized")

    async def send_scan_start(self, ticker_count: int, market_regime: Dict) -> None:
        """Send scan start notification"""
        try:
            regime_text = market_regime.get("regime", "UNKNOWN")
            confidence = market_regime.get("confidence", 50)

            message = (
                f"🔍 **BẮT ĐẦU QUÉT**\n\n"
                f"Số mã: {ticker_count}\n"
                f"Thị trường: *{regime_text}* ({confidence}%)\n"
                f"Thời gian: {datetime.now().strftime('%H:%M %d/%m/%Y')}"
            )

            await self.bot.send_message(
                chat_id=self.chat_id, text=message, parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Error sending scan start: {e}")

    async def send_entry_signal(self, signal_data: Dict) -> None:
        """Send entry signal notification"""
        try:
            symbol = signal_data["symbol"]
            signal = signal_data["signal"]
            position_size = signal_data["position_size"]

            # Calculate R:R
            risk = signal.entry_price - signal.stop_loss
            reward = signal.take_profit_targets[0] - signal.entry_price
            rr_ratio = reward / risk if risk > 0 else 0

            message = (
                f"🚀 **TÍN HIỆU MUA MỚI**\n\n"
                f"**Mã:** `{symbol}`\n"
                f"**Giá vào:** `{signal.entry_price:,.0f}`\n"
                f"**Stop Loss:** `{signal.stop_loss:,.0f}`\n"
                f"**Take Profit 1:** `{signal.take_profit_targets[0]:,.0f}`\n"
                f"**R:R:** `{rr_ratio:.2f}`\n\n"
                f"**Confidence:** {signal.confidence}%\n"
                f"**Strength:** {signal.strength.name}\n\n"
                f"**--- Quản lý vốn ---**\n"
                f"**Số CP:** `{position_size.shares:,}`\n"
                f"**Giá trị:** `{position_size.value:,.0f} VNĐ`\n"
                f"**Rủi ro:** `{position_size.risk_amount:,.0f} VNĐ` "
                f"({position_size.risk_percent:.2f}%)\n\n"
                f"**Lý do:** {', '.join(signal.reasons[:2])}"
            )

            await self.bot.send_message(
                chat_id=self.chat_id, text=message, parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Error sending entry signal: {e}")

    async def send_exit_signal(self, exit_data: Dict) -> None:
        """Send exit signal notification"""
        try:
            symbol = exit_data["symbol"]
            decision = exit_data["decision"]

            urgency_emoji = {5: "🚨🚨🚨", 4: "🚨🚨", 3: "⚠️", 2: "💡", 1: "ℹ️"}

            emoji = urgency_emoji.get(decision.urgency, "⚠️")

            message = (
                f"{emoji} **TÍN HIỆU THOÁT**\n\n"
                f"**Mã:** `{symbol}`\n"
                f"**Loại:** {decision.exit_type}\n"
                f"**Lý do:** {decision.exit_reason.value}\n"
                f"**Giá thoát:** `{decision.exit_price:,.0f}`\n"
                f"**P&L:** {decision.expected_pnl_percent:+.2f}%\n"
                f"**Urgency:** {decision.urgency}/5\n\n"
                f"{decision.message}"
            )

            await self.bot.send_message(
                chat_id=self.chat_id, text=message, parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Error sending exit signal: {e}")

    async def send_scan_summary(
        self,
        signal_count: int,
        exit_count: int,
        market_regime: Dict,
        portfolio_summary: Optional[str] = None,
    ) -> None:
        """Send scan summary"""
        try:
            regime_text = market_regime.get("regime", "N/A")
            confidence = market_regime.get("confidence", 0)

            message = (
                f"📊 **BÁO CÁO QUÉT**\n\n"
                f"Thời gian: {datetime.now().strftime('%H:%M %d/%m/%Y')}\n"
                f"Thị trường: *{regime_text}* ({confidence}%)\n\n"
                f"💡 Tín hiệu mua: **{signal_count}**\n"
                f"🚪 Tín hiệu thoát: **{exit_count}**\n"
            )

            if portfolio_summary:
                message += f"\n{portfolio_summary}"

            await self.bot.send_message(
                chat_id=self.chat_id, text=message, parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Error sending summary: {e}")

    async def send_risk_alert(self, alert_type: str, message: str) -> None:
        """Send risk alert"""
        try:
            alert_message = (
                f"🚨 **CẢNH BÁO RỦI RO**\n\n"
                f"**Loại:** {alert_type}\n"
                f"**Chi tiết:** {message}\n"
                f"**Thời gian:** {datetime.now().strftime('%H:%M %d/%m/%Y')}"
            )

            await self.bot.send_message(
                chat_id=self.chat_id, text=alert_message, parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Error sending risk alert: {e}")


def get_notification_service(bot: Bot, chat_id: str) -> NotificationService:
    """Get notification service instance"""
    return NotificationService(bot, chat_id)
