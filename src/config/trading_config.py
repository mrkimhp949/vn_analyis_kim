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
            (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d"),
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
    # IMPROVED: Raised thresholds for better signal quality
    min_confidence: int = 50  # Raised from 40 for better quality signals
    min_risk_reward: float = 2.2  # Raised from 1.8 for better risk-adjusted returns
    support_distance_percent: float = (
        4.0  # Max distance to support (%) - widened from 3% for more opportunities
    )

    # Exit logic
    # IMPROVED: Optimized for VN market volatility
    stop_loss_percent: float = -6.0  # Tighter from -7% for Vietnam market
    take_profit_percent: float = 12.0  # More realistic from 15% for VN market
    trailing_stop_percent: float = 4.0  # Trail 4% below peak (from 3%)
    trailing_activation_percent: float = (
        6.0  # Activate trailing stop after 6% gain (lowered from 8%)
    )

    # Position sizing - IMPROVED v4.0
    # CRITICAL FIX: Ensure max_position_size * max_positions <= max_cash_allocation
    # Previous config was impossible: 12% * 10 = 120% > 100%!
    total_capital: float = 100_000_000  # 100M VND
    max_position_size: float = 0.07  # 7% of portfolio per position
    min_position_size: float = 0.03  # 3% of portfolio minimum (lowered for flexibility)
    max_positions: int = 10  # Max 10 positions
    max_cash_allocation: float = 0.70  # Max 70% invested, keep 30% cash buffer
    # VALIDATION: 7% * 10 = 70% = max_cash_allocation ✓

    # Risk management
    # IMPROVED: More conservative risk limits for Vietnam market
    max_portfolio_risk: float = 0.15  # 15% max risk (reduced from 20% for safety)
    max_sector_exposure: float = (
        0.30  # 30% max per sector (reduced from 40% for better diversification)
    )
    max_positions_per_sector: int = 3  # Max 3 positions per sector for diversification
    max_loss_per_day_pct: float = 3.0  # Max loss per day (%) for circuit breaker (reduced from 5%)

    # CRITICAL FIX: Magic numbers moved to config for easier tuning
    # Market regime adjustments
    bull_market_penalty_scale: float = 0.7  # Scale penalties down in bull market
    bear_market_penalty_scale: float = 1.2  # Scale penalties up in bear market
    high_volatility_penalty_scale: float = 1.3  # Scale penalties up in high volatility

    # Profit protection
    profit_protection_pct_low: float = 0.50  # Protect 50% of profit (3-5% range)
    profit_protection_pct_high: float = 0.60  # Protect 60% of profit (5-8% range)

    # Circuit breaker volatility adjustment
    circuit_breaker_volatility_tighten_factor: float = 0.75  # Tighten 25% in high vol

    # Technical-only signals
    min_technical_only_confidence: float = 40.0  # Lower threshold for technical signals

    # Per-symbol circuit breaker
    per_symbol_max_consecutive_losses: int = 3  # Block symbol after N consecutive losses
    per_symbol_min_win_rate: float = 0.30  # Block if win rate < 30% after 5 trades

    # VIETNAM MARKET-SPECIFIC FEATURES (NEW)
    # Price floor/ceiling limits (±7% daily limit for Vietnam stocks)
    vn_daily_price_limit_pct: float = 7.0  # Vietnam daily price limit ±7%
    vn_check_price_limits: bool = True  # Check if price near floor/ceiling before entry
    vn_avoid_floor_ceiling_pct: float = 2.0  # Avoid entry if within 2% of floor/ceiling

    # T+2 settlement (Vietnam market rule)
    vn_settlement_days: int = 2  # T+2 settlement for Vietnam
    vn_reserve_t2_cash: bool = True  # Reserve cash for T+2 settlement obligations
    vn_t2_cash_buffer_pct: float = 0.10  # 10% extra cash buffer for T+2

    # Liquidity considerations for Vietnam market
    vn_min_daily_value: float = 2_000_000_000  # 2B VND minimum daily trading value
    vn_max_position_pct_of_volume: float = 0.05  # Max 5% of daily volume for single position
    vn_require_continuous_trading: bool = True  # Avoid stocks with trading halts

    # Vietnam market hours and session management
    vn_trading_session_am_end: str = "11:30"  # Morning session ends
    vn_trading_session_pm_start: str = "13:00"  # Afternoon session starts
    vn_avoid_session_boundaries: bool = True  # Avoid trading near session boundaries
    vn_session_boundary_minutes: int = 5  # Minutes before/after session boundary to avoid

    @classmethod
    def from_env(cls):
        return cls(
            max_scan_universe=int(os.getenv("MAX_SCAN_UNIVERSE", 40)),
            watchlist_size=int(os.getenv("WATCHLIST_SIZE", 100)),
            min_confidence=int(os.getenv("MIN_CONFIDENCE", 45)),
            min_risk_reward=float(os.getenv("MIN_RISK_REWARD", 2.0)),
            support_distance_percent=float(os.getenv("SUPPORT_DISTANCE_PERCENT", 3.0)),
            stop_loss_percent=float(os.getenv("STOP_LOSS_PERCENT", -7.0)),
            take_profit_percent=float(os.getenv("TAKE_PROFIT_PERCENT", 15.0)),
            trailing_stop_percent=float(os.getenv("TRAILING_STOP_PERCENT", 3.0)),
            trailing_activation_percent=float(os.getenv("TRAILING_ACTIVATION_PERCENT", 8.0)),
            total_capital=float(os.getenv("TOTAL_CAPITAL", 100_000_000)),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", 0.08)),
            min_position_size=float(os.getenv("MIN_POSITION_SIZE", 0.05)),
            max_positions=int(os.getenv("MAX_POSITIONS", 10)),
            max_cash_allocation=float(os.getenv("MAX_CASH_ALLOCATION", 0.80)),
            max_portfolio_risk=float(os.getenv("MAX_PORTFOLIO_RISK", 0.20)),
            max_sector_exposure=float(os.getenv("MAX_SECTOR_EXPOSURE", 0.40)),
            max_positions_per_sector=int(os.getenv("MAX_POSITIONS_PER_SECTOR", 3)),
            max_loss_per_day_pct=float(os.getenv("MAX_LOSS_PER_DAY_PCT", 5.0)),
            # Magic numbers
            bull_market_penalty_scale=float(os.getenv("BULL_MARKET_PENALTY_SCALE", 0.7)),
            bear_market_penalty_scale=float(os.getenv("BEAR_MARKET_PENALTY_SCALE", 1.2)),
            high_volatility_penalty_scale=float(os.getenv("HIGH_VOLATILITY_PENALTY_SCALE", 1.3)),
            profit_protection_pct_low=float(os.getenv("PROFIT_PROTECTION_PCT_LOW", 0.50)),
            profit_protection_pct_high=float(os.getenv("PROFIT_PROTECTION_PCT_HIGH", 0.60)),
            circuit_breaker_volatility_tighten_factor=float(
                os.getenv("CIRCUIT_BREAKER_VOLATILITY_TIGHTEN_FACTOR", 0.75)
            ),
            min_technical_only_confidence=float(os.getenv("MIN_TECHNICAL_ONLY_CONFIDENCE", 40.0)),
            per_symbol_max_consecutive_losses=int(
                os.getenv("PER_SYMBOL_MAX_CONSECUTIVE_LOSSES", 3)
            ),
            per_symbol_min_win_rate=float(os.getenv("PER_SYMBOL_MIN_WIN_RATE", 0.30)),
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

        # CRITICAL: Cross-field validation
        self._validate_cross_field_consistency()

    def _validate_cross_field_consistency(self):
        """
        Validate cross-field relationships and logical consistency

        Critical checks:
        - max_position_size * max_positions <= 1.0 (can't allocate >100%)
        - max_sector_exposure compatible with max_positions
        - stop_loss_percent compatible with max_portfolio_risk
        """
        # Check 1: Total potential exposure with cash allocation
        # If all positions are at max size, total exposure can't exceed max_cash_allocation
        max_potential_exposure = self.max_position_size * self.max_positions

        if max_potential_exposure > self.max_cash_allocation:
            raise ConfigurationError(
                f"Impossible configuration: max_position_size ({self.max_position_size:.1%}) "
                f"* max_positions ({self.max_positions}) = {max_potential_exposure:.1%} "
                f"> max_cash_allocation ({self.max_cash_allocation:.1%})\n"
                f"Cannot allocate more than {self.max_cash_allocation:.1%} of capital!",
                context={
                    "config": "trading",
                    "max_position_size": self.max_position_size,
                    "max_positions": self.max_positions,
                    "max_cash_allocation": self.max_cash_allocation,
                    "max_potential_exposure": max_potential_exposure,
                    "suggestion": f"Reduce max_position_size to <= {self.max_cash_allocation / self.max_positions:.2f} "
                    f"or max_positions to <= {int(self.max_cash_allocation / self.max_position_size)}",
                },
            )

        # Check 2: Sector exposure must accommodate multiple positions
        # If max_sector_exposure < max_position_size, impossible to fill even one position
        if self.max_sector_exposure < self.max_position_size:
            raise ConfigurationError(
                f"Inconsistent configuration: max_sector_exposure ({self.max_sector_exposure:.1%}) "
                f"< max_position_size ({self.max_position_size:.1%})\n"
                f"Cannot create a position larger than sector limit!",
                context={
                    "config": "trading",
                    "max_sector_exposure": self.max_sector_exposure,
                    "max_position_size": self.max_position_size,
                    "suggestion": f"Increase max_sector_exposure to >= {self.max_position_size:.2f}",
                },
            )

        # Check 3: Portfolio risk must be achievable
        # If all positions hit stop loss, total loss = max_positions * stop_loss_percent * max_position_size
        # This should not exceed max_portfolio_risk
        max_loss_per_position = abs(self.stop_loss_percent) / 100.0  # Convert % to decimal
        max_portfolio_loss = self.max_positions * max_loss_per_position * self.max_position_size

        if max_portfolio_loss > self.max_portfolio_risk:
            raise ConfigurationError(
                f"Risk configuration unsafe: if all {self.max_positions} positions hit "
                f"{self.stop_loss_percent}% stop loss, total loss = {max_portfolio_loss:.1%} "
                f"exceeds max_portfolio_risk ({self.max_portfolio_risk:.1%})",
                context={
                    "config": "trading",
                    "max_positions": self.max_positions,
                    "stop_loss_percent": self.stop_loss_percent,
                    "max_position_size": self.max_position_size,
                    "max_portfolio_loss": max_portfolio_loss,
                    "max_portfolio_risk": self.max_portfolio_risk,
                    "suggestion": f"Reduce max_positions to <= {int(self.max_portfolio_risk / (max_loss_per_position * self.max_position_size))} "
                    f"or increase max_portfolio_risk to >= {max_portfolio_loss:.2f}",
                },
            )

        # Check 4: Max positions per sector validation
        # max_positions_per_sector must be reasonable given max_positions
        if self.max_positions_per_sector > self.max_positions:
            raise ConfigurationError(
                f"max_positions_per_sector ({self.max_positions_per_sector}) "
                f"> max_positions ({self.max_positions}). This doesn't make sense!",
                context={
                    "config": "trading",
                    "max_positions_per_sector": self.max_positions_per_sector,
                    "max_positions": self.max_positions,
                    "suggestion": f"Set max_positions_per_sector <= {self.max_positions}",
                },
            )

        # Check sector concentration: If max_positions_per_sector too high, limited diversification
        min_sectors_needed = (
            self.max_positions + self.max_positions_per_sector - 1
        ) // self.max_positions_per_sector
        if min_sectors_needed < 3:
            # Warning level - not blocking, but logged
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"⚠️ Limited sector diversification: max_positions ({self.max_positions}) "
                f"/ max_positions_per_sector ({self.max_positions_per_sector}) "
                f"= only {min_sectors_needed} sectors minimum. Consider reducing max_positions_per_sector."
            )

        # Check 5: Min position size must be practical
        # With total capital and max positions, min position size must be achievable
        min_capital_per_position = self.total_capital * self.min_position_size
        max_capital_per_position = self.total_capital * self.max_position_size

        # Each position needs at least 1 lot (100 shares) * reasonable price (e.g., 10,000 VND)
        min_practical_capital = 100 * 10_000  # 1M VND minimum

        if min_capital_per_position < min_practical_capital:
            raise ConfigurationError(
                f"min_position_size too small: {self.min_position_size:.1%} of "
                f"{self.total_capital:,.0f} VND = {min_capital_per_position:,.0f} VND "
                f"< {min_practical_capital:,.0f} VND (minimum 1 lot at 10k/share)",
                context={
                    "config": "trading",
                    "min_position_size": self.min_position_size,
                    "total_capital": self.total_capital,
                    "min_capital_per_position": min_capital_per_position,
                    "min_practical_capital": min_practical_capital,
                    "suggestion": f"Increase min_position_size to >= {min_practical_capital / self.total_capital:.4f} "
                    f"or increase total_capital",
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
            errors.append(f"tcbs_rate_limit must be >= 1, got {self.api.tcbs_rate_limit}")

        if self.api.request_timeout < 1:
            errors.append(f"request_timeout must be >= 1, got {self.api.request_timeout}")

        # Validate server config
        if not (1024 <= self.server.port <= 65535):
            errors.append(f"port must be between 1024 and 65535, got {self.server.port}")

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
