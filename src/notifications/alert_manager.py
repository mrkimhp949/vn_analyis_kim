# -*- coding: utf-8 -*-
"""
Alert & Notification Manager

Multi-channel notification system:
- Telegram alerts
- Email notifications
- Desktop notifications
- Webhook integrations

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from queue import Queue
import threading
import json

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels"""

    CRITICAL = 1  # Immediate action required
    HIGH = 2  # Important, needs attention
    MEDIUM = 3  # Normal alerts
    LOW = 4  # Informational
    DEBUG = 5  # Debug/verbose


class AlertType(Enum):
    """Alert types"""

    SIGNAL_BUY = "SIGNAL_BUY"
    SIGNAL_SELL = "SIGNAL_SELL"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TAKE_PROFIT_HIT = "TAKE_PROFIT_HIT"
    POSITION_ALERT = "POSITION_ALERT"
    RISK_WARNING = "RISK_WARNING"
    MARKET_ALERT = "MARKET_ALERT"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    EARNINGS_ALERT = "EARNINGS_ALERT"
    FOREIGN_FLOW = "FOREIGN_FLOW"


@dataclass
class Alert:
    """Alert data"""

    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str

    # Optional data
    symbol: Optional[str] = None
    price: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    alert_id: str = ""
    sent: bool = False
    channels_sent: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"{self.alert_type.value}_{self.timestamp.strftime('%Y%m%d%H%M%S')}"

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "type": self.alert_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "symbol": self.symbol,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


class BaseNotificationChannel(ABC):
    """Abstract base class for notification channels"""

    @property
    @abstractmethod
    def channel_name(self) -> str:
        pass

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        pass


class TelegramChannel(BaseNotificationChannel):
    """
    Telegram notification channel

    Setup:
    1. Create bot via @BotFather
    2. Get bot token
    3. Get chat_id from @userinfobot
    """

    def __init__(self, bot_token: str = "", chat_id: str = ""):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{bot_token}"

    @property
    def channel_name(self) -> str:
        return "TELEGRAM"

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            logger.warning("Telegram not configured")
            return False

        try:
            import requests

            # Format message
            emoji = self._get_emoji(alert.alert_type, alert.priority)

            text = f"{emoji} *{alert.title}*\n\n"
            text += f"{alert.message}\n"

            if alert.symbol:
                text += f"\n📊 Symbol: `{alert.symbol}`"
            if alert.price:
                text += f"\n💰 Price: `{alert.price:,.0f}` VND"

            text += f"\n\n⏰ {alert.timestamp.strftime('%H:%M:%S %d/%m/%Y')}"

            # Send message
            url = f"{self._base_url}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info(f"✅ Telegram alert sent: {alert.title}")
                return True
            else:
                logger.warning(f"Telegram send failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    def _get_emoji(self, alert_type: AlertType, priority: AlertPriority) -> str:
        """Get emoji for alert type"""
        emojis = {
            AlertType.SIGNAL_BUY: "🟢",
            AlertType.SIGNAL_SELL: "🔴",
            AlertType.ORDER_FILLED: "✅",
            AlertType.ORDER_REJECTED: "❌",
            AlertType.STOP_LOSS_HIT: "🛑",
            AlertType.TAKE_PROFIT_HIT: "🎯",
            AlertType.RISK_WARNING: "⚠️",
            AlertType.MARKET_ALERT: "📢",
            AlertType.SYSTEM_ERROR: "🚨",
            AlertType.EARNINGS_ALERT: "📅",
            AlertType.FOREIGN_FLOW: "🌍",
        }

        if priority == AlertPriority.CRITICAL:
            return "🚨"

        return emojis.get(alert_type, "📌")


class WebhookChannel(BaseNotificationChannel):
    """
    Webhook notification channel

    Sends JSON payload to configured URL
    """

    def __init__(self, webhook_url: str = "", headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}

    @property
    def channel_name(self) -> str:
        return "WEBHOOK"

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False

        try:
            import requests

            payload = alert.to_dict()
            response = requests.post(
                self.webhook_url, json=payload, headers=self.headers, timeout=10
            )

            return response.status_code in [200, 201, 202]

        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return False


class ConsoleChannel(BaseNotificationChannel):
    """Console/log notification channel"""

    @property
    def channel_name(self) -> str:
        return "CONSOLE"

    def is_configured(self) -> bool:
        return True

    def send(self, alert: Alert) -> bool:
        priority_colors = {
            AlertPriority.CRITICAL: "\033[91m",  # Red
            AlertPriority.HIGH: "\033[93m",  # Yellow
            AlertPriority.MEDIUM: "\033[94m",  # Blue
            AlertPriority.LOW: "\033[92m",  # Green
            AlertPriority.DEBUG: "\033[90m",  # Gray
        }

        reset = "\033[0m"
        color = priority_colors.get(alert.priority, "")

        print(f"\n{color}{'='*60}")
        print(f"🔔 [{alert.priority.name}] {alert.title}")
        print(f"{'='*60}{reset}")
        print(f"{alert.message}")
        if alert.symbol:
            print(f"Symbol: {alert.symbol}")
        if alert.price:
            print(f"Price: {alert.price:,.0f} VND")
        print(f"Time: {alert.timestamp.strftime('%H:%M:%S')}")
        print(f"{color}{'='*60}{reset}\n")

        return True


class AlertManager:
    """
    Central alert management system

    Features:
    - Multi-channel notifications
    - Priority-based filtering
    - Rate limiting
    - Alert history
    - Async sending
    """

    def __init__(self):
        self._channels: Dict[str, BaseNotificationChannel] = {}
        self._alert_history: List[Alert] = []
        self._max_history = 1000
        self._rate_limits: Dict[str, datetime] = {}
        self._rate_limit_seconds = 60  # Min seconds between same alerts

        # Async queue
        self._queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # Add console channel by default
        self.add_channel(ConsoleChannel())

    def add_channel(self, channel: BaseNotificationChannel) -> None:
        """Add notification channel"""
        self._channels[channel.channel_name] = channel
        logger.info(f"Added notification channel: {channel.channel_name}")

    def remove_channel(self, channel_name: str) -> None:
        """Remove notification channel"""
        if channel_name in self._channels:
            del self._channels[channel_name]

    def configure_telegram(self, bot_token: str, chat_id: str) -> None:
        """Configure Telegram notifications"""
        channel = TelegramChannel(bot_token, chat_id)
        if channel.is_configured():
            self.add_channel(channel)
            logger.info("✅ Telegram notifications configured")
        else:
            logger.warning("Telegram configuration incomplete")

    def configure_webhook(self, url: str, headers: Dict[str, str] = None) -> None:
        """Configure webhook notifications"""
        channel = WebhookChannel(url, headers)
        if channel.is_configured():
            self.add_channel(channel)
            logger.info("✅ Webhook notifications configured")

    def send_alert(
        self,
        alert_type: AlertType,
        title: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
        symbol: str = None,
        price: float = None,
        data: Dict = None,
        channels: List[str] = None,
    ) -> Alert:
        """
        Send alert to configured channels

        Args:
            alert_type: Type of alert
            title: Alert title
            message: Alert message
            priority: Alert priority
            symbol: Related symbol
            price: Related price
            data: Additional data
            channels: Specific channels (None = all)

        Returns:
            Alert object
        """
        alert = Alert(
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            symbol=symbol,
            price=price,
            data=data or {},
        )

        # Check rate limit
        rate_key = f"{alert_type.value}_{symbol or 'GENERAL'}"
        if self._is_rate_limited(rate_key):
            logger.debug(f"Alert rate limited: {rate_key}")
            return alert

        # Send to channels
        target_channels = channels or list(self._channels.keys())

        for channel_name in target_channels:
            if channel_name in self._channels:
                channel = self._channels[channel_name]

                # Skip low priority for some channels
                if channel_name == "TELEGRAM" and priority.value > AlertPriority.MEDIUM.value:
                    continue

                try:
                    if channel.send(alert):
                        alert.channels_sent.append(channel_name)
                except Exception as e:
                    logger.error(f"Channel {channel_name} error: {e}")

        alert.sent = len(alert.channels_sent) > 0

        # Update rate limit
        self._rate_limits[rate_key] = datetime.now()

        # Add to history
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history :]

        return alert

    def _is_rate_limited(self, key: str) -> bool:
        """Check if alert is rate limited"""
        if key not in self._rate_limits:
            return False

        last_sent = self._rate_limits[key]
        elapsed = (datetime.now() - last_sent).total_seconds()

        return elapsed < self._rate_limit_seconds

    # Convenience methods for common alerts
    def signal_buy(self, symbol: str, price: float, reason: str) -> Alert:
        """Send buy signal alert"""
        return self.send_alert(
            AlertType.SIGNAL_BUY,
            f"🟢 BUY Signal: {symbol}",
            f"Buy signal generated for {symbol}\n\nReason: {reason}",
            AlertPriority.HIGH,
            symbol=symbol,
            price=price,
        )

    def signal_sell(self, symbol: str, price: float, reason: str) -> Alert:
        """Send sell signal alert"""
        return self.send_alert(
            AlertType.SIGNAL_SELL,
            f"🔴 SELL Signal: {symbol}",
            f"Sell signal generated for {symbol}\n\nReason: {reason}",
            AlertPriority.HIGH,
            symbol=symbol,
            price=price,
        )

    def stop_loss_hit(self, symbol: str, price: float, loss_pct: float) -> Alert:
        """Send stop loss alert"""
        return self.send_alert(
            AlertType.STOP_LOSS_HIT,
            f"🛑 Stop Loss: {symbol}",
            f"Stop loss triggered for {symbol}\nLoss: {loss_pct:.2f}%",
            AlertPriority.CRITICAL,
            symbol=symbol,
            price=price,
        )

    def risk_warning(self, title: str, message: str, data: Dict = None) -> Alert:
        """Send risk warning alert"""
        return self.send_alert(
            AlertType.RISK_WARNING, f"⚠️ {title}", message, AlertPriority.HIGH, data=data
        )

    def earnings_alert(self, symbol: str, days_until: int, period: str) -> Alert:
        """Send earnings alert"""
        return self.send_alert(
            AlertType.EARNINGS_ALERT,
            f"📅 Earnings Alert: {symbol}",
            f"{symbol} earnings ({period}) in {days_until} days\n\nConsider reducing position or avoiding new entries.",
            AlertPriority.MEDIUM,
            symbol=symbol,
        )

    def foreign_flow_alert(self, direction: str, value: float) -> Alert:
        """Send foreign flow alert"""
        emoji = "🟢" if direction == "BUY" else "🔴"
        return self.send_alert(
            AlertType.FOREIGN_FLOW,
            f"{emoji} Foreign {direction}ing Alert",
            f"Significant foreign {direction.lower()}ing detected\nNet value: {value/1e9:.1f}B VND",
            AlertPriority.MEDIUM,
        )

    def get_history(self, limit: int = 50) -> List[Alert]:
        """Get alert history"""
        return self._alert_history[-limit:]


# Singleton instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get singleton alert manager"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


# Convenience functions
def send_alert(alert_type: AlertType, title: str, message: str, **kwargs) -> Alert:
    """Quick alert send"""
    return get_alert_manager().send_alert(alert_type, title, message, **kwargs)


def configure_telegram(bot_token: str, chat_id: str) -> None:
    """Configure Telegram"""
    get_alert_manager().configure_telegram(bot_token, chat_id)
