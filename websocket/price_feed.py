# -*- coding: utf-8 -*-
"""
WebSocket Price Feed - Real-time market data

Supports:
- SSI WebSocket API
- TCBS WebSocket API (when available)
- Auto-reconnection
- Price caching
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


@dataclass
class PriceUpdate:
    """Real-time price update"""

    symbol: str
    price: float
    volume: int
    change: float
    change_percent: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None


@dataclass
class WebSocketConfig:
    """WebSocket configuration"""

    url: str = "wss://fc-data.ssi.com.vn/realtime"
    reconnect_delay: int = 5  # seconds
    max_reconnect_attempts: int = 10
    heartbeat_interval: int = 30  # seconds
    subscribe_channels: List[str] = field(default_factory=list)


class PriceFeedClient:
    """
    WebSocket client for real-time price feeds

    Features:
    - Automatic reconnection
    - Heartbeat/ping-pong
    - Multiple symbol subscriptions
    - Callback-based updates
    """

    def __init__(self, config: WebSocketConfig = None):
        self.config = config or WebSocketConfig()
        self.websocket = None
        self.running = False
        self.connected = False

        # Price cache
        self.prices: Dict[str, PriceUpdate] = {}

        # Callbacks
        self.price_callbacks: List[Callable[[PriceUpdate], None]] = []

        # Reconnection
        self.reconnect_attempts = 0

        logger.info("Price feed client initialized")

    def add_price_callback(self, callback: Callable[[PriceUpdate], None]):
        """Add callback function for price updates"""
        self.price_callbacks.append(callback)
        logger.info(f"Added price callback: {callback.__name__}")

    def subscribe(self, symbols: List[str]):
        """Subscribe to symbols for real-time updates"""
        if not isinstance(symbols, list):
            symbols = [symbols]

        for symbol in symbols:
            if symbol not in self.config.subscribe_channels:
                self.config.subscribe_channels.append(symbol)

        logger.info(f"Subscribed to: {', '.join(symbols)}")

    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        if not isinstance(symbols, list):
            symbols = [symbols]

        for symbol in symbols:
            if symbol in self.config.subscribe_channels:
                self.config.subscribe_channels.remove(symbol)

        logger.info(f"Unsubscribed from: {', '.join(symbols)}")

    async def connect(self):
        """Connect to WebSocket server"""
        try:
            logger.info(f"Connecting to {self.config.url}...")
            self.websocket = await websockets.connect(
                self.config.url,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=10,
            )
            self.connected = True
            self.reconnect_attempts = 0
            logger.info("✅ Connected to price feed")

            # Send subscription message
            await self.send_subscription()

        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            raise

    async def send_subscription(self):
        """Send subscription message to server"""
        if not self.websocket or not self.config.subscribe_channels:
            return

        # SSI WebSocket subscription format
        subscription_msg = {
            "action": "subscribe",
            "channels": [f"X:{symbol}" for symbol in self.config.subscribe_channels],
        }

        try:
            await self.websocket.send(json.dumps(subscription_msg))
            logger.info(
                f"Sent subscription for {len(self.config.subscribe_channels)} symbols"
            )
        except Exception as e:
            logger.error(f"Failed to send subscription: {e}")

    async def handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)

            # Parse price update (SSI format)
            if "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    self.process_price_update(item)

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def process_price_update(self, data: Dict):
        """Process price update from WebSocket"""
        try:
            symbol = data.get("sym", data.get("symbol", ""))
            if not symbol:
                return

            # Create price update
            update = PriceUpdate(
                symbol=symbol,
                price=float(data.get("lastPrice", data.get("price", 0))),
                volume=int(data.get("totalVol", data.get("volume", 0))),
                change=float(data.get("change", 0)),
                change_percent=float(
                    data.get("changePc", data.get("changePercent", 0))
                ),
                timestamp=datetime.now(),
                bid=float(data.get("bid1", 0)) if "bid1" in data else None,
                ask=float(data.get("ask1", 0)) if "ask1" in data else None,
                high=float(data.get("high", 0)) if "high" in data else None,
                low=float(data.get("low", 0)) if "low" in data else None,
            )

            # Update cache
            self.prices[symbol] = update

            # Trigger callbacks
            for callback in self.price_callbacks:
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Error in callback {callback.__name__}: {e}")

        except Exception as e:
            logger.error(f"Error processing price update: {e}")

    async def receive_loop(self):
        """Main receive loop"""
        try:
            async for message in self.websocket:
                await self.handle_message(message)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.connected = False

        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
            self.connected = False

    async def reconnect(self):
        """Attempt to reconnect"""
        self.reconnect_attempts += 1

        if self.reconnect_attempts > self.config.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self.running = False
            return

        delay = self.config.reconnect_delay * self.reconnect_attempts
        logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts})...")

        await asyncio.sleep(delay)

        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            await self.reconnect()

    async def run(self):
        """Main run loop with auto-reconnection"""
        self.running = True

        while self.running:
            try:
                if not self.connected:
                    await self.connect()

                await self.receive_loop()

                # If we get here, connection was closed
                if self.running:
                    await self.reconnect()

            except Exception as e:
                logger.error(f"Error in run loop: {e}")
                if self.running:
                    await self.reconnect()

    def start(self):
        """Start the WebSocket client in a background thread"""

        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run())

        thread = threading.Thread(target=run_async_loop, daemon=True)
        thread.start()
        logger.info("WebSocket client started in background")

    async def stop(self):
        """Stop the WebSocket client"""
        logger.info("Stopping WebSocket client...")
        self.running = False

        if self.websocket:
            await self.websocket.close()

        self.connected = False
        logger.info("WebSocket client stopped")

    def get_price(self, symbol: str) -> Optional[PriceUpdate]:
        """Get latest price for a symbol"""
        return self.prices.get(symbol)

    def get_all_prices(self) -> Dict[str, PriceUpdate]:
        """Get all cached prices"""
        return self.prices.copy()


# Example callback function
def print_price_update(update: PriceUpdate):
    """Print price updates to console"""
    change_str = (
        f"+{update.change_percent:.2f}%"
        if update.change_percent >= 0
        else f"{update.change_percent:.2f}%"
    )
    print(
        f"[{update.timestamp.strftime('%H:%M:%S')}] {update.symbol}: {update.price:,.0f} VND ({change_str})"
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create client
    client = PriceFeedClient()

    # Add callback
    client.add_price_callback(print_price_update)

    # Subscribe to symbols
    client.subscribe(["VCB", "HPG", "VHM", "VNM", "VIC"])

    # Start (this runs in background)
    client.start()

    # Keep main thread alive
    try:
        while True:
            asyncio.run(asyncio.sleep(1))
    except KeyboardInterrupt:
        print("\nStopping...")
        asyncio.run(client.stop())
