"""
Trading Constants - Centralized configuration values
Replaces magic numbers throughout the codebase
"""

# Risk Management Constants
DEFAULT_RISK_PER_TRADE = 0.02  # 2% risk per trade
DEFAULT_MAX_POSITION_SIZE = 0.15  # 15% max position size
DEFAULT_MIN_POSITION_SIZE = 0.05  # 5% min position size
DEFAULT_MAX_TOTAL_EXPOSURE = 0.60  # 60% max total exposure
DEFAULT_MAX_SECTOR_EXPOSURE = 0.40  # 40% max sector exposure
DEFAULT_MAX_DRAWDOWN = 0.15  # 15% max drawdown

# Technical Analysis Constants
DEFAULT_ATR_PERCENTAGE = 0.02  # 2% ATR fallback
DEFAULT_VOLUME_SURGE_MULTIPLIER = 1.5  # 1.5x volume surge threshold
DEFAULT_ROLLING_WINDOW = 20  # 20-period rolling window
DEFAULT_SHORT_MA_PERIOD = 20  # 20-period short moving average
DEFAULT_MEDIUM_MA_PERIOD = 50  # 50-period medium moving average
DEFAULT_LONG_MA_PERIOD = 200  # 200-period long moving average

# Risk/Reward Ratios
DEFAULT_MIN_RISK_REWARD = 1.5  # Minimum 1.5:1 risk/reward
DEFAULT_TAKE_PROFIT_RATIOS = [1.5, 3.0, 5.0]  # Multiple take profit levels

# Position Sizing Multipliers
MIN_POSITION_MULTIPLIER = 0.3  # Minimum position size multiplier
MAX_POSITION_MULTIPLIER = 1.5  # Maximum position size multiplier

# Market Regime Thresholds
BULL_MARKET_THRESHOLD = 0.02  # 2% threshold for bull market
BEAR_MARKET_THRESHOLD = -0.02  # -2% threshold for bear market
TREND_THRESHOLD = 0.02  # 2% trend threshold

# Time-based Constants
MAX_HOLDING_DAYS = 20  # Maximum holding period in days
EARLY_STOPPING_ROUNDS = 20  # ML early stopping rounds
CORRELATION_LOOKBACK_DAYS = 60  # Correlation calculation lookback

# Commission and Slippage (Vietnam Market)
# Vietnam total transaction costs: brokerage (0.15%) + tax (0.10%) + fees (0.10%) = ~0.35%
DEFAULT_COMMISSION_RATE = 0.0035  # 0.35% total transaction cost (commission + tax + fees)
DEFAULT_SLIPPAGE = 0.001  # 0.1% average slippage
TOTAL_TRANSACTION_COST = DEFAULT_COMMISSION_RATE + DEFAULT_SLIPPAGE  # 0.45% total per trade
ROUND_TRIP_COST = TOTAL_TRANSACTION_COST * 2  # 0.9% round trip (buy + sell)

# Vietnam Market Specific Constants
VIETNAM_PRICE_LIMIT_PERCENT = 0.07  # ±7% daily price limit (floor/ceiling)
VIETNAM_LOT_SIZE = 100  # Minimum trading lot size
VIETNAM_SETTLEMENT_DAYS = 2  # T+2 settlement
VIETNAM_TICK_SIZE = 10  # 10 VND minimum tick size for most stocks

# Stop Loss and Take Profit
DEFAULT_STOP_LOSS_LEVELS = [0.10, 0.15, 0.25]  # 10%, 15%, 25%
DEFAULT_TRAILING_STOP_ACTIVATION = 0.08  # 8% profit to activate trailing
DEFAULT_TRAILING_STOP_DISTANCE = 0.05  # 5% trailing distance
DEFAULT_TIME_DECAY_THRESHOLD = 0.02  # 2% time decay threshold

# Capital and Sizing
DEFAULT_TOTAL_CAPITAL = 100_000_000  # 100M VND default capital
DEFAULT_MIN_ROWS = 50  # Minimum rows for analysis
DEFAULT_VALIDATION_MIN_ROWS = 20  # Minimum rows for validation

# Volume Analysis
VOLUME_CONFIRMATION_THRESHOLD = 1.2  # 1.2x volume for confirmation
VOLUME_SURGE_THRESHOLD = 1.5  # 1.5x volume for surge
OBV_PERIODS = [5, 20]  # OBV calculation periods

# RSI and Momentum
RSI_OVERSOLD = 30  # RSI oversold level
RSI_OVERBOUGHT = 70  # RSI overbought level
RSI_NEUTRAL = 50  # RSI neutral level

# Market Regime Adjustments
BULL_MARKET_CONFIDENCE = 50  # Lower confidence in bull market
BULL_MARKET_RR = 1.5  # Lower R:R in bull market
BULL_MARKET_EXPOSURE = 0.70  # Higher exposure in bull market

BEAR_MARKET_CONFIDENCE = 65  # Higher confidence in bear market
BEAR_MARKET_RR = 2.0  # Higher R:R in bear market
BEAR_MARKET_EXPOSURE = 0.30  # Lower exposure in bear market

SIDEWAYS_MARKET_CONFIDENCE = 55  # Medium confidence in sideways market
SIDEWAYS_MARKET_RR = 1.8  # Medium R:R in sideways market
SIDEWAYS_MARKET_EXPOSURE = 0.50  # Medium exposure in sideways market

# ML Model Constants
ML_SIGNAL_WEIGHT = 1.5  # Weight for ML signals
TECH_SIGNAL_WEIGHT = 0.5  # Weight for technical signals

# Validation Thresholds
MAX_CORRELATION = 0.70  # Maximum correlation between positions
DIVERSIFICATION_PENALTY = 20  # Points deducted per warning

# Export all constants
__all__ = [
    # Risk Management
    "DEFAULT_RISK_PER_TRADE",
    "DEFAULT_MAX_POSITION_SIZE",
    "DEFAULT_MIN_POSITION_SIZE",
    "DEFAULT_MAX_TOTAL_EXPOSURE",
    "DEFAULT_MAX_SECTOR_EXPOSURE",
    "DEFAULT_MAX_DRAWDOWN",
    # Technical Analysis
    "DEFAULT_ATR_PERCENTAGE",
    "DEFAULT_VOLUME_SURGE_MULTIPLIER",
    "DEFAULT_ROLLING_WINDOW",
    "DEFAULT_SHORT_MA_PERIOD",
    "DEFAULT_MEDIUM_MA_PERIOD",
    "DEFAULT_LONG_MA_PERIOD",
    # Risk/Reward
    "DEFAULT_MIN_RISK_REWARD",
    "DEFAULT_TAKE_PROFIT_RATIOS",
    # Position Sizing
    "MIN_POSITION_MULTIPLIER",
    "MAX_POSITION_MULTIPLIER",
    # Market Regime
    "BULL_MARKET_THRESHOLD",
    "BEAR_MARKET_THRESHOLD",
    "TREND_THRESHOLD",
    # Time-based
    "MAX_HOLDING_DAYS",
    "EARLY_STOPPING_ROUNDS",
    "CORRELATION_LOOKBACK_DAYS",
    # Trading Costs
    "DEFAULT_COMMISSION_RATE",
    "DEFAULT_SLIPPAGE",
    "TOTAL_TRANSACTION_COST",
    "ROUND_TRIP_COST",
    # Vietnam Market
    "VIETNAM_PRICE_LIMIT_PERCENT",
    "VIETNAM_LOT_SIZE",
    "VIETNAM_SETTLEMENT_DAYS",
    "VIETNAM_TICK_SIZE",
    # Stop Loss/Take Profit
    "DEFAULT_STOP_LOSS_LEVELS",
    "DEFAULT_TRAILING_STOP_ACTIVATION",
    "DEFAULT_TRAILING_STOP_DISTANCE",
    "DEFAULT_TIME_DECAY_THRESHOLD",
    # Capital
    "DEFAULT_TOTAL_CAPITAL",
    "DEFAULT_MIN_ROWS",
    "DEFAULT_VALIDATION_MIN_ROWS",
    # Volume
    "VOLUME_CONFIRMATION_THRESHOLD",
    "VOLUME_SURGE_THRESHOLD",
    "OBV_PERIODS",
    # RSI
    "RSI_OVERSOLD",
    "RSI_OVERBOUGHT",
    "RSI_NEUTRAL",
    # Market Regime Adjustments
    "BULL_MARKET_CONFIDENCE",
    "BULL_MARKET_RR",
    "BULL_MARKET_EXPOSURE",
    "BEAR_MARKET_CONFIDENCE",
    "BEAR_MARKET_RR",
    "BEAR_MARKET_EXPOSURE",
    "SIDEWAYS_MARKET_CONFIDENCE",
    "SIDEWAYS_MARKET_RR",
    "SIDEWAYS_MARKET_EXPOSURE",
    # ML
    "ML_SIGNAL_WEIGHT",
    "TECH_SIGNAL_WEIGHT",
    # Validation
    "MAX_CORRELATION",
    "DIVERSIFICATION_PENALTY",
]
