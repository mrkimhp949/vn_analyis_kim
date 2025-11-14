# -*- coding: utf-8 -*-
"""
Real-time Position Monitor
Giám sát positions liên tục và gửi alerts
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PriceAlert:
    """Alert về giá"""

    symbol: str
    alert_type: str  # 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP', 'REVERSAL'
    current_price: float
    trigger_price: float
    message: str
    urgency: str  # 'HIGH', 'MEDIUM', 'LOW'


class RealtimeMonitor:
    """
    Real-time monitor cho positions

    Features:
    - Check giá liên tục
    - Trailing stop updates
    - Alert khi hit stop loss/take profit
    - Reversal pattern detection
    """

    def __init__(
        self,
        check_interval: int = 300,  # 5 minutes
        alert_callback: Optional[callable] = None,
    ):
        """
        Args:
            check_interval: Interval giữa các lần check (seconds)
            alert_callback: Async function để gửi alerts
        """
        self.check_interval = check_interval
        self.alert_callback = alert_callback
        self.is_running = False
        self.last_prices = {}  # Track last known prices
        self.trailing_stops = {}  # Track trailing stops

    async def start_monitoring(self):
        """Bắt đầu monitoring"""
        self.is_running = True
        print(f"🔍 Starting real-time monitoring (interval: {self.check_interval}s)...")

        while self.is_running:
            try:
                await self._check_all_positions()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            # Wait for next check
            await asyncio.sleep(self.check_interval)

    def stop_monitoring(self):
        """Dừng monitoring"""
        self.is_running = False
        print("⏹️ Stopping real-time monitoring...")

    async def _check_all_positions(self):
        """Check tất cả positions"""
        from portfolio_manager import get_portfolio_manager
        from data_loader import load_data

        portfolio_manager = get_portfolio_manager()
        positions = portfolio_manager.get_positions()

        if not positions:
            return

        print(
            f"\n🔍 Checking {len(positions)} positions at {datetime.now().strftime('%H:%M:%S')}"
        )

        alerts = []

        for symbol, pos in positions.items():
            try:
                # Get current price
                df = load_data(symbol, lookback=5, use_cache=False)
                if df.empty:
                    continue

                current_price = df["close"].iloc[-1]
                entry_price = pos["avg_price"]

                # Update last price
                self.last_prices[symbol] = current_price

                # Check stop loss
                if pos.get("stop_loss"):
                    if current_price <= pos["stop_loss"]:
                        alerts.append(
                            PriceAlert(
                                symbol=symbol,
                                alert_type="STOP_LOSS",
                                current_price=current_price,
                                trigger_price=pos["stop_loss"],
                                message=f"🚨 {symbol} hit STOP LOSS! Price: {current_price:,.0f} <= SL: {pos['stop_loss']:,.0f}",
                                urgency="HIGH",
                            )
                        )

                # Check take profit
                if pos.get("take_profit"):
                    if current_price >= pos["take_profit"]:
                        alerts.append(
                            PriceAlert(
                                symbol=symbol,
                                alert_type="TAKE_PROFIT",
                                current_price=current_price,
                                trigger_price=pos["take_profit"],
                                message=f"🎯 {symbol} hit TAKE PROFIT! Price: {current_price:,.0f} >= TP: {pos['take_profit']:,.0f}",
                                urgency="MEDIUM",
                            )
                        )

                # Update trailing stop
                pnl_percent = ((current_price - entry_price) / entry_price) * 100

                if pnl_percent > 10:  # Profit > 10%
                    # Calculate trailing stop (2% below current price)
                    new_trailing_stop = current_price * 0.98

                    # Update if higher than current trailing stop
                    if (
                        symbol not in self.trailing_stops
                        or new_trailing_stop > self.trailing_stops[symbol]
                    ):
                        old_stop = self.trailing_stops.get(symbol, 0)
                        self.trailing_stops[symbol] = new_trailing_stop

                        print(
                            f"   📈 {symbol}: Updated trailing stop {old_stop:,.0f} -> {new_trailing_stop:,.0f}"
                        )

                    # Check if hit trailing stop
                    if current_price <= self.trailing_stops[symbol]:
                        alerts.append(
                            PriceAlert(
                                symbol=symbol,
                                alert_type="TRAILING_STOP",
                                current_price=current_price,
                                trigger_price=self.trailing_stops[symbol],
                                message=f"📉 {symbol} hit TRAILING STOP! Price: {current_price:,.0f} <= TS: {self.trailing_stops[symbol]:,.0f}",
                                urgency="HIGH",
                            )
                        )

                # Check reversal patterns
                if len(df) >= 3:
                    reversal_alert = self._check_reversal_pattern(
                        symbol, df, current_price, entry_price
                    )
                    if reversal_alert:
                        alerts.append(reversal_alert)

            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")

        # Send alerts
        if alerts:
            await self._send_alerts(alerts)

    def _check_reversal_pattern(
        self, symbol: str, df, current_price: float, entry_price: float
    ) -> Optional[PriceAlert]:
        """Check for reversal patterns"""
        # Simple reversal detection: 3 consecutive down candles after profit
        pnl_percent = ((current_price - entry_price) / entry_price) * 100

        if pnl_percent > 5:  # Only check if in profit
            last_3 = df.tail(3)

            # Check if all 3 are down candles
            down_candles = (last_3["close"] < last_3["open"]).sum()

            if down_candles == 3:
                return PriceAlert(
                    symbol=symbol,
                    alert_type="REVERSAL",
                    current_price=current_price,
                    trigger_price=entry_price,
                    message=f"⚠️ {symbol}: Possible reversal detected (3 down candles). Current: {current_price:,.0f}, P&L: {pnl_percent:+.1f}%",
                    urgency="MEDIUM",
                )

        return None

    async def _send_alerts(self, alerts: List[PriceAlert]):
        """Gửi alerts"""
        # Sort by urgency
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        alerts.sort(key=lambda x: urgency_order[x.urgency])

        for alert in alerts:
            print(f"   {alert.message}")

            # Send via callback if provided
            if self.alert_callback:
                try:
                    await self.alert_callback(alert)
                except Exception as e:
                    logger.error(f"Error sending alert: {e}")

    def get_monitoring_status(self) -> Dict:
        """Get monitoring status"""
        return {
            "is_running": self.is_running,
            "check_interval": self.check_interval,
            "monitored_symbols": len(self.last_prices),
            "trailing_stops": self.trailing_stops,
            "last_check": datetime.now().isoformat(),
        }


class MonitoringScheduler:
    """
    Scheduler cho monitoring
    Chỉ chạy trong giờ giao dịch
    """

    def __init__(self, monitor: RealtimeMonitor):
        self.monitor = monitor
        self.is_running = False

    async def start(self):
        """Start scheduler"""
        self.is_running = True
        print("📅 Starting monitoring scheduler...")

        while self.is_running:
            # Check if in trading hours
            if self._is_trading_hours():
                if not self.monitor.is_running:
                    # Start monitoring
                    asyncio.create_task(self.monitor.start_monitoring())
            else:
                if self.monitor.is_running:
                    # Stop monitoring
                    self.monitor.stop_monitoring()

            # Check every minute
            await asyncio.sleep(60)

    def stop(self):
        """Stop scheduler"""
        self.is_running = False
        self.monitor.stop_monitoring()

    def _is_trading_hours(self) -> bool:
        """Check if currently in trading hours"""
        from vn_trading_schedule import is_trading_hours

        try:
            return is_trading_hours()
        except Exception:
            # Fallback: check basic hours
            now = datetime.now()
            hour = now.hour
            minute = now.minute

            # Morning: 9:00 - 11:30
            if 9 <= hour < 11 or (hour == 11 and minute <= 30):
                return True

            # Afternoon: 13:00 - 15:00
            if 13 <= hour < 15:
                return True

            return False


# Integration with Telegram
async def telegram_alert_callback(alert: PriceAlert):
    """Callback để gửi alert qua Telegram"""
    try:
        from telegram import Bot
        from config import TELEGRAM_TOKEN, CHAT_ID

        bot = Bot(token=TELEGRAM_TOKEN)

        # Format message
        emoji = {
            "STOP_LOSS": "🚨",
            "TAKE_PROFIT": "🎯",
            "TRAILING_STOP": "📉",
            "REVERSAL": "⚠️",
        }.get(alert.alert_type, "📊")

        message = f"{emoji} *ALERT*\n\n{alert.message}\n\nUrgency: {alert.urgency}"

        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error sending Telegram alert: {e}")


# Singleton
_monitor = None


def get_realtime_monitor(
    check_interval: int = 300, with_telegram: bool = True
) -> RealtimeMonitor:
    """Get monitor singleton"""
    global _monitor
    if _monitor is None:
        alert_callback = telegram_alert_callback if with_telegram else None
        _monitor = RealtimeMonitor(
            check_interval=check_interval, alert_callback=alert_callback
        )
    return _monitor


# Test
if __name__ == "__main__":
    print("Testing Real-time Monitor...")

    async def test_monitor():
        # Create monitor without Telegram
        monitor = RealtimeMonitor(check_interval=10, alert_callback=None)

        # Run for 30 seconds
        task = asyncio.create_task(monitor.start_monitoring())
        await asyncio.sleep(30)
        monitor.stop_monitoring()
        await task

        # Get status
        status = monitor.get_monitoring_status()
        print(f"Status: {status}")

    asyncio.run(test_monitor())
    print("\n✅ Test completed!")
