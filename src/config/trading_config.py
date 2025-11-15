"""
Centralized Trading Configuration
All trading parameters in one place
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from src.config.exceptions import ConfigurationError


@dataclass
class DataConfig:
    """Data source configuration"""

    end_date: str = datetime.now().strftime("%Y-%m-%d")
    start_date: str = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    lookback: int = 200
    min_volume: int = 100000
    use_csv_tickers: bool = True  # Load from List.csv
    cache_enabled: bool = True

    @classmethod
    def from_env(cls):
        end_date = os.getenv("END_DATE", datetime.now().strftime("%Y-%m-%d"))
        start_date = os.getenv(
            "START_DATE",
            (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime(
                "%Y-%m-%d"
            ),
        )
        return cls(
            end_date=end_date,
            start_date=start_date,
            lookback=int(os.getenv("LOOKBACK", 200)),
            min_volume=int(os.getenv("MIN_VOLUME", 100000)),
            use_csv_tickers=os.getenv("USE_CSV_TICKERS", "true").lower() == "true",
            cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
        )

    def validate(self):
        """Validate data configuration"""
        if self.lookback < 50:
            raise ConfigurationError(
                f"lookback must be >= 50, got {self.lookback}",
                context={"config": "data", "field": "lookback", "value": self.lookback},
            )
        if self.min_volume < 0:
            raise ConfigurationError(
                f"min_volume must be >= 0, got {self.min_volume}",
                context={
                    "config": "data",
                    "field": "min_volume",
                    "value": self.min_volume,
                },
            )


@dataclass
class TradingConfig:
    """Trading strategy configuration"""

    # Scanning & Universe
    max_scan_universe: int = 40  # Max tickers to scan in detail
    watchlist_size: int = 100  # Size of the initial watchlist

    # Entry logic
    min_confidence: int = 60
    min_risk_reward: float = 2.0
    support_distance_percent: float = 3.0  # Max distance to support (%)

    # Exit logic
    stop_loss_percent: float = -7.0
    take_profit_percent: float = 15.0
    trailing_stop_percent: float = 3.0
    trailing_activation_percent: float = 8.0  # Activate trailing stop after 8% gain

    # Position sizing
    total_capital: float = 100_000_000  # 100M VND
    max_position_size: float = 0.15  # 15% of portfolio
    min_position_size: float = 0.05  # 5% of portfolio
    max_positions: int = 10

    # Risk management
    max_portfolio_risk: float = 0.20  # 20% max risk
    max_sector_exposure: float = 0.40  # 40% max per sector
    max_loss_per_day_pct: float = 5.0  # Max loss per day (%) for circuit breaker

    @classmethod
    def from_env(cls):
        return cls(
            max_scan_universe=int(os.getenv("MAX_SCAN_UNIVERSE", 40)),
            watchlist_size=int(os.getenv("WATCHLIST_SIZE", 100)),
            min_confidence=int(os.getenv("MIN_CONFIDENCE", 60)),
            min_risk_reward=float(os.getenv("MIN_RISK_REWARD", 2.0)),
            support_distance_percent=float(os.getenv("SUPPORT_DISTANCE_PERCENT", 3.0)),
            stop_loss_percent=float(os.getenv("STOP_LOSS_PERCENT", -7.0)),
            take_profit_percent=float(os.getenv("TAKE_PROFIT_PERCENT", 15.0)),
            trailing_stop_percent=float(os.getenv("TRAILING_STOP_PERCENT", 3.0)),
            trailing_activation_percent=float(
                os.getenv("TRAILING_ACTIVATION_PERCENT", 8.0)
            ),
            total_capital=float(os.getenv("TOTAL_CAPITAL", 100_000_000)),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", 0.15)),
            min_position_size=float(os.getenv("MIN_POSITION_SIZE", 0.05)),
            max_positions=int(os.getenv("MAX_POSITIONS", 10)),
            max_portfolio_risk=float(os.getenv("MAX_PORTFOLIO_RISK", 0.20)),
            max_sector_exposure=float(os.getenv("MAX_SECTOR_EXPOSURE", 0.40)),
            max_loss_per_day_pct=float(os.getenv("MAX_LOSS_PER_DAY_PCT", 5.0)),
        )

    def validate(self):
        """Validate trading configuration"""
        if not (0 <= self.min_confidence <= 100):
            raise ConfigurationError(
                f"min_confidence must be between 0 and 100, got {self.min_confidence}",
                context={
                    "config": "trading",
                    "field": "min_confidence",
                    "value": self.min_confidence,
                },
            )

        if self.min_risk_reward < 1.0:
            raise ConfigurationError(
                f"min_risk_reward must be >= 1.0, got {self.min_risk_reward}",
                context={
                    "config": "trading",
                    "field": "min_risk_reward",
                    "value": self.min_risk_reward,
                },
            )

        if not (0 < self.max_position_size <= 1.0):
            raise ConfigurationError(
                f"max_position_size must be between 0 and 1.0, got {self.max_position_size}",
                context={
                    "config": "trading",
                    "field": "max_position_size",
                    "value": self.max_position_size,
                },
            )

        if not (0 < self.min_position_size < self.max_position_size):
            raise ConfigurationError(
                f"min_position_size ({self.min_position_size}) must be < max_position_size ({self.max_position_size})",
                context={
                    "config": "trading",
                    "min": self.min_position_size,
                    "max": self.max_position_size,
                },
            )

        if self.max_positions < 1:
            raise ConfigurationError(
                f"max_positions must be >= 1, got {self.max_positions}",
                context={
                    "config": "trading",
                    "field": "max_positions",
                    "value": self.max_positions,
                },
            )

        if not (0 < self.max_portfolio_risk <= 1.0):
            raise ConfigurationError(
                f"max_portfolio_risk must be between 0 and 1.0, got {self.max_portfolio_risk}",
                context={
                    "config": "trading",
                    "field": "max_portfolio_risk",
                    "value": self.max_portfolio_risk,
                },
            )

        if not (0 < self.max_sector_exposure <= 1.0):
            raise ConfigurationError(
                f"max_sector_exposure must be between 0 and 1.0, got {self.max_sector_exposure}",
                context={
                    "config": "trading",
                    "field": "max_sector_exposure",
                    "value": self.max_sector_exposure,
                },
            )


@dataclass
class APIConfig:
    """API configuration"""

    tcbs_base_url: str = "https://apipubaws.tcbs.com.vn"
    tcbs_rate_limit: int = 10  # calls per second
    yahoo_rate_limit: int = 5
    request_timeout: int = 10

    @classmethod
    def from_env(cls):
        return cls(
            tcbs_base_url=os.getenv("TCBS_API_BASE", "https://apipubaws.tcbs.com.vn"),
            tcbs_rate_limit=int(os.getenv("TCBS_RATE_LIMIT", 10)),
            yahoo_rate_limit=int(os.getenv("YAHOO_RATE_LIMIT", 5)),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", 10)),
        )


@dataclass
class TelegramConfig:
    """Telegram bot configuration"""

    token: Optional[str] = None
    chat_id: Optional[str] = None
    enabled: bool = True

    @classmethod
    def from_env(cls):
        token = os.getenv("TELEGRAM_TOKEN")
        chat_id = os.getenv("CHAT_ID")

        return cls(token=token, chat_id=chat_id, enabled=bool(token and chat_id))

    def validate(self):
        """Validate telegram config"""
        if not self.token or self.token.strip() == "":
            raise ConfigurationError(
                "TELEGRAM_TOKEN not set or empty. Get your token from https://t.me/Botfather",
                context={
                    "config": "telegram",
                    "field": "token",
                    "value": self.token,
                    "help": "Set TELEGRAM_TOKEN in .env file or environment variable",
                },
            )
        if not self.chat_id or self.chat_id.strip() == "":
            raise ConfigurationError(
                "CHAT_ID not set or empty",
                context={
                    "config": "telegram",
                    "field": "chat_id",
                    "value": self.chat_id,
                    "help": "Set CHAT_ID in .env file or environment variable",
                },
            )


@dataclass
class ServerConfig:
    """Server configuration"""

    port: int = 8080
    host: str = "0.0.0.0"
    debug: bool = False

    @classmethod
    def from_env(cls):
        return cls(
            port=int(os.getenv("PORT", 8080)),
            host=os.getenv("HOST", "0.0.0.0"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )


class Config:
    """
    Master configuration class

    Usage:
        config = Config.load()
        print(config.trading.min_confidence)
        print(config.data.lookback)
    """

    def __init__(self):
        self.data = DataConfig.from_env()
        self.trading = TradingConfig.from_env()
        self.api = APIConfig.from_env()
        self.telegram = TelegramConfig.from_env()
        self.server = ServerConfig.from_env()

    @classmethod
    def load(cls):
        """Load configuration from environment"""
        # Load .env file if available
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        return cls()

    def validate(self):
        """Validate all configurations"""
        errors = []

        # Validate data config
        try:
            self.data.validate()
        except ConfigurationError as e:
            errors.append(str(e))

        # Validate trading config
        try:
            self.trading.validate()
        except ConfigurationError as e:
            errors.append(str(e))

        # Validate telegram if enabled
        if self.telegram.enabled:
            try:
                self.telegram.validate()
            except (ValueError, ConfigurationError) as e:
                errors.append(f"Telegram: {str(e)}")

        # Validate API config
        if self.api.tcbs_rate_limit < 1:
            errors.append(
                f"tcbs_rate_limit must be >= 1, got {self.api.tcbs_rate_limit}"
            )

        if self.api.request_timeout < 1:
            errors.append(
                f"request_timeout must be >= 1, got {self.api.request_timeout}"
            )

        # Validate server config
        if not (1024 <= self.server.port <= 65535):
            errors.append(
                f"port must be between 1024 and 65535, got {self.server.port}"
            )

        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {len(errors)} error(s)",
                context={"errors": errors},
            )

    def summary(self) -> str:
        """Get configuration summary"""
        lines = []
        lines.append("⚙️ CONFIGURATION")
        lines.append("=" * 50)

        lines.append("\n📊 Data:")
        lines.append(f"  Start Date: {self.data.start_date}")
        lines.append(f"  End Date: {self.data.end_date}")
        lines.append(f"  Lookback: {self.data.lookback}")
        lines.append(f"  Min Volume: {self.data.min_volume:,}")
        lines.append(f"  Use CSV Tickers: {self.data.use_csv_tickers}")
        lines.append(f"  Cache Enabled: {self.data.cache_enabled}")

        lines.append("\n💹 Trading:")
        lines.append(f"  Min Confidence: {self.trading.min_confidence}%")
        lines.append(f"  Stop Loss: {self.trading.stop_loss_percent}%")
        lines.append(f"  Take Profit: {self.trading.take_profit_percent}%")
        lines.append(f"  Max Position: {self.trading.max_position_size*100}%")
        lines.append(f"  Max Positions: {self.trading.max_positions}")

        lines.append("\n🌐 API:")
        lines.append(f"  TCBS Rate Limit: {self.api.tcbs_rate_limit} calls/sec")
        lines.append(f"  Timeout: {self.api.request_timeout}s")

        lines.append("\n📱 Telegram:")
        lines.append(f"  Enabled: {self.telegram.enabled}")
        if self.telegram.token:
            lines.append(f"  Token: {self.telegram.token[:10]}...")

        lines.append("\n🖥️ Server:")
        lines.append(f"  Port: {self.server.port}")
        lines.append(f"  Debug: {self.server.debug}")

        return "\n".join(lines)


# Singleton instance
_config = None


def get_config(validate: bool = True) -> Config:
    """
    Get configuration singleton

    Args:
        validate: Whether to validate config on load (default: True)

    Returns:
        Config instance

    Raises:
        ConfigurationError: If validation fails
    """
    global _config
    if _config is None:
        _config = Config.load()
        if validate:
            _config.validate()
    return _config


# Test
if __name__ == "__main__":
    print("Testing configuration...")

    config = Config.load()

    print(config.summary())

    try:
        config.validate()
        print("\n✅ Configuration valid!")
    except ValueError:
        print("\n❌ Configuration error")
