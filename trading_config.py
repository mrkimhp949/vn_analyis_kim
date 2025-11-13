"""
Centralized Trading Configuration
All trading parameters in one place
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Data source configuration"""
    lookback: int = 200
    min_volume: int = 100000
    use_csv_tickers: bool = True  # Load from List.csv
    cache_enabled: bool = True
    
    @classmethod
    def from_env(cls):
        return cls(
            lookback=int(os.getenv('LOOKBACK', 200)),
            min_volume=int(os.getenv('MIN_VOLUME', 100000)),
            use_csv_tickers=os.getenv('USE_CSV_TICKERS', 'true').lower() == 'true',
            cache_enabled=os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        )


@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    # Entry logic
    min_confidence: int = 60
    min_risk_reward: float = 2.0
    
    # Exit logic
    stop_loss_percent: float = -7.0
    take_profit_percent: float = 15.0
    trailing_stop_percent: float = 3.0
    
    # Position sizing
    max_position_size: float = 0.15  # 15% of portfolio
    min_position_size: float = 0.05  # 5% of portfolio
    max_positions: int = 10
    
    # Risk management
    max_portfolio_risk: float = 0.20  # 20% max risk
    max_sector_exposure: float = 0.40  # 40% max per sector
    
    @classmethod
    def from_env(cls):
        return cls(
            min_confidence=int(os.getenv('MIN_CONFIDENCE', 60)),
            min_risk_reward=float(os.getenv('MIN_RISK_REWARD', 2.0)),
            stop_loss_percent=float(os.getenv('STOP_LOSS_PERCENT', -7.0)),
            take_profit_percent=float(os.getenv('TAKE_PROFIT_PERCENT', 15.0)),
            trailing_stop_percent=float(os.getenv('TRAILING_STOP_PERCENT', 3.0)),
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', 0.15)),
            min_position_size=float(os.getenv('MIN_POSITION_SIZE', 0.05)),
            max_positions=int(os.getenv('MAX_POSITIONS', 10)),
            max_portfolio_risk=float(os.getenv('MAX_PORTFOLIO_RISK', 0.20)),
            max_sector_exposure=float(os.getenv('MAX_SECTOR_EXPOSURE', 0.40))
        )


@dataclass
class APIConfig:
    """API configuration"""
    tcbs_base_url: str = 'https://apipubaws.tcbs.com.vn'
    tcbs_rate_limit: int = 10  # calls per second
    yahoo_rate_limit: int = 5
    request_timeout: int = 10
    
    @classmethod
    def from_env(cls):
        return cls(
            tcbs_base_url=os.getenv('TCBS_API_BASE', 'https://apipubaws.tcbs.com.vn'),
            tcbs_rate_limit=int(os.getenv('TCBS_RATE_LIMIT', 10)),
            yahoo_rate_limit=int(os.getenv('YAHOO_RATE_LIMIT', 5)),
            request_timeout=int(os.getenv('REQUEST_TIMEOUT', 10))
        )


@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    token: Optional[str] = None
    chat_id: Optional[str] = None
    enabled: bool = True
    
    @classmethod
    def from_env(cls):
        token = os.getenv('TELEGRAM_TOKEN')
        chat_id = os.getenv('CHAT_ID')
        
        return cls(
            token=token,
            chat_id=chat_id,
            enabled=bool(token and chat_id)
        )
    
    def validate(self):
        """Validate telegram config"""
        if not self.token:
            raise ValueError("TELEGRAM_TOKEN not set")
        if not self.chat_id:
            raise ValueError("CHAT_ID not set")


@dataclass
class ServerConfig:
    """Server configuration"""
    port: int = 8080
    host: str = '0.0.0.0'
    debug: bool = False
    
    @classmethod
    def from_env(cls):
        return cls(
            port=int(os.getenv('PORT', 8080)),
            host=os.getenv('HOST', '0.0.0.0'),
            debug=os.getenv('DEBUG', 'false').lower() == 'true'
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
        
        # Validate telegram if enabled
        if self.telegram.enabled:
            try:
                self.telegram.validate()
            except ValueError as e:
                errors.append(f"Telegram: {e}")
        
        # Validate trading params
        if self.trading.min_confidence < 0 or self.trading.min_confidence > 100:
            errors.append("min_confidence must be between 0 and 100")
        
        if self.trading.max_position_size > 1.0:
            errors.append("max_position_size must be <= 1.0")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    def summary(self) -> str:
        """Get configuration summary"""
        lines = []
        lines.append("⚙️ CONFIGURATION")
        lines.append("=" * 50)
        
        lines.append("\n📊 Data:")
        lines.append(f"  Lookback: {self.data.lookback}")
        lines.append(f"  Min Volume: {self.data.min_volume:,}")
        lines.append(f"  Dynamic Tickers: {self.data.use_dynamic_tickers}")
        
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

def get_config() -> Config:
    """Get configuration singleton"""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


# Test
if __name__ == "__main__":
    print("Testing configuration...")
    
    config = Config.load()
    
    print(config.summary())
    
    try:
        config.validate()
        print("\n✅ Configuration valid!")
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
