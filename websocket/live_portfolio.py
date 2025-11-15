# -*- coding: utf-8 -*-
"""
Live Portfolio Monitor - Real-time portfolio tracking with WebSocket

Features:
- Real-time P&L updates
- Stop-loss/Take-profit alerts
- Position monitoring
- Risk alerts
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List

from src.strategies.exit_logic import ImprovedExitStrategy
from src.portfolio.manager import PortfolioManager

from websocket.price_feed import PriceFeedClient, PriceUpdate, WebSocketConfig

logger = logging.getLogger(__name__)


@dataclass
class PositionAlert:
    """Alert for position events"""

    symbol: str
    alert_type: str  # 'STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP', 'RISK_WARNING'
    message: str
    current_price: float
    position_pnl: float
    timestamp: datetime
    urgency: int  # 1-5 (5 = urgent)


class LivePortfolioMonitor:
    """
    Monitor portfolio in real-time using WebSocket price feed

    Features:
    - Auto-update P&L
    - Stop-loss/Take-profit monitoring
    - Trailing stop updates
    - Risk alerts
    - Telegram notifications (optional)
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager = None,
        websocket_config: WebSocketConfig = None,
        enable_alerts: bool = True,
    ):
        self.portfolio = portfolio_manager or PortfolioManager()
        self.price_feed = PriceFeedClient(websocket_config)
        self.exit_logic = ImprovedExitStrategy()
        self.enable_alerts = enable_alerts

        # Alert callbacks
        self.alert_callbacks: List[Callable[[PositionAlert], None]] = []

        # Performance tracking
        self.last_portfolio_value = 0
        self.daily_high = 0
        self.daily_low = float("inf")

        # Register price callback
        self.price_feed.add_price_callback(self.handle_price_update)

        logger.info("Live portfolio monitor initialized")

    def add_alert_callback(self, callback: Callable[[PositionAlert], None]):
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
        logger.info(f"Added alert callback: {callback.__name__}")

    def handle_price_update(self, update: PriceUpdate):
        """Handle real-time price updates"""
        symbol = update.symbol

        # Get position if exists
        positions = self.portfolio.get_positions()
        if symbol not in positions:
            return

        position = positions[symbol]

        # Calculate current P&L
        entry_price = position["avg_price"]
        current_price = update.price
        shares = position["shares"]

        pnl = (current_price - entry_price) * shares
        pnl_percent = ((current_price - entry_price) / entry_price) * 100

        # Check exit conditions
        self.check_exit_conditions(
            symbol=symbol,
            position=position,
            current_price=current_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
        )

        # Update portfolio value tracking
        self.update_portfolio_tracking()

    def check_exit_conditions(
        self,
        symbol: str,
        position: Dict,
        current_price: float,
        pnl: float,
        pnl_percent: float,
    ):
        """Check if any exit conditions are met"""

        # Get position details
        entry_price = position["avg_price"]
        position.get("entry_date", datetime.now())
        stop_loss = position.get("stop_loss", entry_price * 0.95)
        take_profit = position.get("take_profit", entry_price * 1.15)

        # Check stop loss
        if current_price <= stop_loss:
            self.send_alert(
                PositionAlert(
                    symbol=symbol,
                    alert_type="STOP_LOSS",
                    message=f"⛔ STOP LOSS HIT: {symbol} @ {current_price:,.0f} (Entry: {entry_price:,.0f})",
                    current_price=current_price,
                    position_pnl=pnl,
                    timestamp=datetime.now(),
                    urgency=5,
                )
            )

        # Check take profit
        if current_price >= take_profit:
            self.send_alert(
                PositionAlert(
                    symbol=symbol,
                    alert_type="TAKE_PROFIT",
                    message=f"🎯 TAKE PROFIT: {symbol} @ {current_price:,.0f} (Target: {take_profit:,.0f})",
                    current_price=current_price,
                    position_pnl=pnl,
                    timestamp=datetime.now(),
                    urgency=4,
                )
            )

        # Check trailing stop
        if pnl_percent >= 8.0:  # Activate trailing if up 8%
            trailing_stop = current_price * 0.95  # 5% below current

            if trailing_stop > stop_loss:
                self.send_alert(
                    PositionAlert(
                        symbol=symbol,
                        alert_type="TRAILING_STOP",
                        message=f"📈 UPDATE TRAILING STOP: {symbol} to {trailing_stop:,.0f}",
                        current_price=current_price,
                        position_pnl=pnl,
                        timestamp=datetime.now(),
                        urgency=2,
                    )
                )

        # Risk warning if losing > 2%
        if pnl_percent < -2.0:
            self.send_alert(
                PositionAlert(
                    symbol=symbol,
                    alert_type="RISK_WARNING",
                    message=f"⚠️ LOSS WARNING: {symbol} down {pnl_percent:.2f}% ({pnl:+,.0f} VND)",
                    current_price=current_price,
                    position_pnl=pnl,
                    timestamp=datetime.now(),
                    urgency=3,
                )
            )

    def send_alert(self, alert: PositionAlert):
        """Send alert through all registered callbacks"""
        if not self.enable_alerts:
            return

        logger.warning(f"[ALERT] {alert.message}")

        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception:
                logger.error("Error in alert callback")

    def update_portfolio_tracking(self):
        """Update portfolio performance tracking"""
        try:
            portfolio_value = self.portfolio.get_portfolio_value()
            current_value = portfolio_value.get("total_value", 0)

            # Update daily high/low
            if current_value > self.daily_high:
                self.daily_high = current_value

            if current_value < self.daily_low:
                self.daily_low = current_value

            self.last_portfolio_value = current_value

        except Exception:
            logger.error("Error updating portfolio tracking")

    def start(self, symbols: List[str] = None):
        """Start live monitoring"""
        # Get symbols from portfolio if not specified
        if symbols is None:
            positions = self.portfolio.get_positions()
            symbols = list(positions.keys())

        if not symbols:
            logger.warning("No symbols to monitor")
            return

        # Subscribe to price updates
        self.price_feed.subscribe(symbols)

        # Start WebSocket client
        self.price_feed.start()

        logger.info(f"✅ Live monitoring started for: {', '.join(symbols)}")

    async def stop(self):
        """Stop live monitoring"""
        await self.price_feed.stop()
        logger.info("Live monitoring stopped")

    def get_live_summary(self) -> Dict:
        """Get current portfolio summary with live prices"""
        positions = self.portfolio.get_positions()
        summary = {
            "timestamp": datetime.now(),
            "total_positions": len(positions),
            "portfolio_value": self.last_portfolio_value,
            "daily_high": self.daily_high,
            "daily_low": self.daily_low,
            "positions": {},
        }

        for symbol, position in positions.items():
            live_price = self.price_feed.get_price(symbol)

            if live_price:
                entry_price = position["avg_price"]
                shares = position["shares"]
                current_value = live_price.price * shares
                position_pnl = (live_price.price - entry_price) * shares
                position_pnl_pct = (
                    (live_price.price - entry_price) / entry_price
                ) * 100

                summary["positions"][symbol] = {
                    "shares": shares,
                    "entry_price": entry_price,
                    "current_price": live_price.price,
                    "current_value": current_value,
                    "pnl": position_pnl,
                    "pnl_percent": position_pnl_pct,
                    "change_today": live_price.change_percent,
                }

        return summary

    def print_live_summary(self):
        """Print formatted live summary"""
        summary = self.get_live_summary()

        print("\n" + "=" * 80)
        print(
            f"LIVE PORTFOLIO SUMMARY - {summary['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 80)

        print(f"\n📊 Portfolio Value: {summary['portfolio_value']:,.0f} VND")
        print(f"   Daily High: {summary['daily_high']:,.0f} VND")
        print(f"   Daily Low: {summary['daily_low']:,.0f} VND")

        print(f"\n💼 Positions ({summary['total_positions']}):")
        print("-" * 80)
        print(
            f"{'Symbol':<8} {'Shares':<8} {'Entry':<12} {'Current':<12} {'P&L':<15} {'Today %':<10}"
        )
        print("-" * 80)

        for symbol, pos in summary["positions"].items():
            pnl_str = f"+{pos['pnl']:,.0f}" if pos["pnl"] >= 0 else f"{pos['pnl']:,.0f}"
            _pnl_pct_str = f"({pos['pnl_percent']:+.2f}%)"  # noqa: F841
            today_str = f"{pos['change_today']:+.2f}%"

            print(
                f"{symbol:<8} {pos['shares']:<8} {pos['entry_price']:<12,.0f} "
                f"{pos['current_price']:<12,.0f} {pnl_str:<15} {today_str:<10}"
            )

        print("=" * 80 + "\n")


# Example alert callback
def telegram_alert(alert: PositionAlert):
    """Send alert to Telegram (placeholder)"""
    # In production, use python-telegram-bot to send message
    print(f"📱 TELEGRAM: {alert.message}")


def console_alert(alert: PositionAlert):
    """Print alert to console"""
    urgency_emoji = ["ℹ️", "📊", "⚠️", "🚨", "🔴"][min(alert.urgency - 1, 4)]
    print(f"{urgency_emoji} [{alert.timestamp.strftime('%H:%M:%S')}] {alert.message}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create monitor
    monitor = LivePortfolioMonitor()

    # Add alert callbacks
    monitor.add_alert_callback(console_alert)
    # monitor.add_alert_callback(telegram_alert)  # Uncomment for Telegram

    # Start monitoring
    monitor.start(["VCB", "HPG", "VHM"])

    # Print summary every 30 seconds
    try:
        while True:
            asyncio.run(asyncio.sleep(30))
            monitor.print_live_summary()
    except KeyboardInterrupt:
        print("\nStopping...")
        asyncio.run(monitor.stop())
