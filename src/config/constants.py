"""
Trading Constants - Centralized configuration values
Replaces magic numbers throughout the codebase
"""

# Risk Management Constants - OPTIMIZED v3.0
DEFAULT_RISK_PER_TRADE = 0.015  # TIGHTENED: 1.5% risk per trade (was 2%)
DEFAULT_MAX_POSITION_SIZE = 0.12  # TIGHTENED: 12% max position size (was 15%)
DEFAULT_MIN_POSITION_SIZE = 0.03  # TIGHTENED: 3% min position size (was 5%)
DEFAULT_MAX_TOTAL_EXPOSURE = 0.60  # Max 60% total exposure
DEFAULT_MAX_SECTOR_EXPOSURE = 0.30  # TIGHTENED: 30% max sector exposure (was 40%)
DEFAULT_MAX_DRAWDOWN = 0.12  # TIGHTENED: 12% max drawdown (was 15%)

# Technical Analysis Constants
DEFAULT_ATR_PERCENTAGE = 0.02  # 2% ATR fallback
DEFAULT_VOLUME_SURGE_MULTIPLIER = 1.5  # 1.5x volume surge threshold
DEFAULT_ROLLING_WINDOW = 20  # 20-period rolling window
DEFAULT_SHORT_MA_PERIOD = 20  # 20-period short moving average
DEFAULT_MEDIUM_MA_PERIOD = 50  # 50-period medium moving average
DEFAULT_LONG_MA_PERIOD = 200  # 200-period long moving average

# Risk/Reward Ratios - TIGHTENED v3.0
DEFAULT_MIN_RISK_REWARD = 2.0  # TIGHTENED: Minimum 2:1 risk/reward (was 1.5)
DEFAULT_TAKE_PROFIT_RATIOS = [1.5, 2.5, 4.0]  # OPTIMIZED: Better R:R ratios

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
# UPDATED: More realistic transaction costs for Vietnam market
# Components:
#   - Brokerage fee: 0.15-0.30% (depending on broker)
#   - Stock transaction tax: 0.10% (sell only)
#   - Exchange fees: 0.03%
#   - Transfer fees: 0.02%
#   - Slippage: 0.20-0.30% (market orders, especially large orders)
# Total realistic per-trade cost: 0.50-0.65%
DEFAULT_COMMISSION_RATE = 0.0055  # 0.55% total transaction cost (commission + tax + fees)
DEFAULT_SLIPPAGE = 0.0025  # 0.25% realistic slippage for market orders
TOTAL_TRANSACTION_COST = DEFAULT_COMMISSION_RATE + DEFAULT_SLIPPAGE  # 0.80% total per trade
ROUND_TRIP_COST = TOTAL_TRANSACTION_COST * 2  # 1.6% round trip (buy + sell) - conservative estimate

# Alternative costs for different scenarios
OPTIMISTIC_ROUND_TRIP_COST = 0.012  # 1.2% with best execution and limit orders
REALISTIC_ROUND_TRIP_COST = 0.016  # 1.6% with market orders (default)
PESSIMISTIC_ROUND_TRIP_COST = 0.020  # 2.0% with poor execution and high slippage

# Vietnam Market Specific Constants
VIETNAM_PRICE_LIMIT_PERCENT = 0.07  # ±7% daily price limit (floor/ceiling)
VIETNAM_LOT_SIZE = 100  # Minimum trading lot size
VIETNAM_SETTLEMENT_DAYS = 2  # T+2 settlement
VIETNAM_TICK_SIZE = 10  # 10 VND minimum tick size for most stocks

# Stop Loss and Take Profit - TIGHTENED v3.0
DEFAULT_STOP_LOSS_LEVELS = [0.08, 0.15, 0.25]  # TIGHTENED: 8%, 15%, 25%
DEFAULT_TRAILING_STOP_ACTIVATION = 0.05  # TIGHTENED: 5% profit to activate trailing
DEFAULT_TRAILING_STOP_DISTANCE = 0.03  # TIGHTENED: 3% trailing distance
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

# Market Regime Adjustments - OPTIMIZED v3.0
BULL_MARKET_CONFIDENCE = 55  # TIGHTENED: Still need 55% confidence in bull
BULL_MARKET_RR = 1.8  # TIGHTENED: 1.8:1 R:R in bull market
BULL_MARKET_EXPOSURE = 0.65  # TIGHTENED: 65% max exposure in bull

BEAR_MARKET_CONFIDENCE = 70  # TIGHTENED: Need 70% confidence in bear
BEAR_MARKET_RR = 2.5  # TIGHTENED: 2.5:1 R:R in bear market
BEAR_MARKET_EXPOSURE = 0.25  # TIGHTENED: 25% max exposure in bear

SIDEWAYS_MARKET_CONFIDENCE = 60  # TIGHTENED: 60% confidence in sideways
SIDEWAYS_MARKET_RR = 2.0  # TIGHTENED: 2:1 R:R in sideways
SIDEWAYS_MARKET_EXPOSURE = 0.45  # TIGHTENED: 45% exposure in sideways

# ML Model Constants
ML_SIGNAL_WEIGHT = 1.5  # Weight for ML signals
TECH_SIGNAL_WEIGHT = 0.5  # Weight for technical signals

# Validation Thresholds - TIGHTENED v3.0
MAX_CORRELATION = 0.65  # TIGHTENED: Maximum 0.65 correlation between positions
DIVERSIFICATION_PENALTY = 25  # TIGHTENED: 25 points deducted per warning

# NEW: Vietnam Market Specific Thresholds
# IMPROVED: Tiered liquidity for different market caps
VN_MIN_LIQUIDITY_VALUE = 1_000_000_000  # LOWERED: 1B VND for small caps
VN_MID_CAP_LIQUIDITY_VALUE = 2_000_000_000  # 2B VND for mid caps
VN_LARGE_CAP_LIQUIDITY_VALUE = 5_000_000_000  # 5B VND for large caps
VN_CRITICAL_LIQUIDITY_VALUE = 500_000_000  # 500M VND critical minimum
VN_MIN_VOLUME = 50_000  # 50K shares minimum
VN_MAX_INTRADAY_RANGE = 5.0  # 5% max intraday range for entry
VN_OPTIMAL_ENTRY_TIMES = [(9, 30, 10, 30), (13, 30, 14, 30)]  # Optimal entry windows

# ATO/ATC Session Handling (Auction periods - high volatility)
VN_ATO_START = (9, 0)  # ATO: 9:00-9:15 - Opening auction
VN_ATO_END = (9, 15)
VN_ATC_START = (14, 30)  # ATC: 14:30-14:45 - Closing auction
VN_ATC_END = (14, 45)
VN_ATO_ATC_PENALTY = -15  # Reduced penalty for trading during auction periods (was -25)
VN_ALLOW_ATO_ATC_TRADING = True  # Allow trading during ATO/ATC (with confidence penalty)

# NEW: Beta-adjusted stop loss thresholds
# Higher beta stocks need wider stops to avoid premature exit
VN_STOP_LOSS_BASE = 0.06  # 6% base stop loss
VN_STOP_LOSS_HIGH_BETA = 0.08  # 8% for beta > 1.2
VN_STOP_LOSS_LOW_BETA = 0.05  # 5% for beta < 0.8
VN_HIGH_BETA_THRESHOLD = 1.2  # Beta threshold for wider stop
VN_LOW_BETA_THRESHOLD = 0.8  # Beta threshold for tighter stop

# NEW: Position size adjustments by liquidity tier
VN_SMALL_CAP_POSITION_MULT = 0.7  # 70% position size for small caps
VN_MID_CAP_POSITION_MULT = 0.85  # 85% position size for mid caps
VN_LARGE_CAP_POSITION_MULT = 1.0  # 100% position size for large caps

# Vietnam Price Limits
VN_CEILING_DISTANCE_THRESHOLD = 0.5  # 0.5% from ceiling = near limit
VN_FLOOR_DISTANCE_THRESHOLD = 0.5  # 0.5% from floor = near limit
VN_FLOOR_PENALTY = -20  # Penalty for trading near floor

# Entry Logic Thresholds
ENTRY_PULLBACK_MAX_PCT = 5.0  # Max pullback % from high
ENTRY_PULLBACK_MIN_PCT = 1.0  # Min pullback % to consider
ENTRY_BREAKOUT_VOLUME_MULT = 1.2  # Volume multiplier for breakout
ENTRY_LIMIT_ORDER_MIN_DIFF = 0.5  # Min price diff for limit order (%)

# Support/Resistance
SR_BOUNCE_THRESHOLD = 0.02  # 2% bounce from support
SR_RESISTANCE_CLOSE_THRESHOLD = 2.0  # 2% from resistance = too close
SR_SUPPORT_SUSTAINED_MOVE = 1.01  # 1% above 3-bar avg for sustained move
SR_VOLUME_CONFIRMATION_MULT = 1.2  # Volume multiplier for bounce confirmation

# Correlation Cache
CORRELATION_CACHE_TTL = 300  # 5 minutes TTL

# Technical Scoring (0-1 scale)
TECH_SCORE_HIGH = 1.0
TECH_SCORE_GOOD = 0.8
TECH_SCORE_MODERATE = 0.6
TECH_SCORE_LOW = 0.4
TECH_SCORE_POOR = 0.2

# Technical Confidence Threshold
TECH_ONLY_MIN_CONFIDENCE = 55  # Min confidence for technical-only signals

# Per-Symbol Performance Tracking
MIN_TRADES_FOR_POOR_PERFORMER = (
    5  # Minimum trades before labeling poor performer (increased from 3)
)
POOR_PERFORMER_WIN_RATE_THRESHOLD = 0.35  # Below 35% win rate = poor performer
POOR_PERFORMER_CONSECUTIVE_LOSSES = 2  # 2 consecutive losses threshold

# Foreign Flow (Smart Money) Integration
FOREIGN_FLOW_STRONG_BUY_BONUS = 10  # +10 confidence for strong foreign buying (score > 0.5)
FOREIGN_FLOW_MODERATE_BUY_BONUS = 5  # +5 confidence for moderate foreign buying (score > 0)
FOREIGN_FLOW_MODERATE_SELL_PENALTY = -5  # -5 confidence for moderate foreign selling (score < 0)
FOREIGN_FLOW_STRONG_SELL_PENALTY = -15  # -15 confidence for strong foreign selling (score < -0.5)

# Export all constants
__all__ = [
    # Vietnam Market Extended
    "VN_MIN_LIQUIDITY_VALUE",
    "VN_MID_CAP_LIQUIDITY_VALUE",
    "VN_LARGE_CAP_LIQUIDITY_VALUE",
    "VN_CRITICAL_LIQUIDITY_VALUE",
    "VN_CEILING_DISTANCE_THRESHOLD",
    "VN_FLOOR_DISTANCE_THRESHOLD",
    "VN_FLOOR_PENALTY",
    # ATO/ATC Session
    "VN_ATO_START",
    "VN_ATO_END",
    "VN_ATC_START",
    "VN_ATC_END",
    "VN_ATO_ATC_PENALTY",
    "VN_ALLOW_ATO_ATC_TRADING",
    # Beta-adjusted stop loss
    "VN_STOP_LOSS_BASE",
    "VN_STOP_LOSS_HIGH_BETA",
    "VN_STOP_LOSS_LOW_BETA",
    "VN_HIGH_BETA_THRESHOLD",
    "VN_LOW_BETA_THRESHOLD",
    # Position size by liquidity
    "VN_SMALL_CAP_POSITION_MULT",
    "VN_MID_CAP_POSITION_MULT",
    "VN_LARGE_CAP_POSITION_MULT",
    # Entry Logic
    "ENTRY_PULLBACK_MAX_PCT",
    "ENTRY_PULLBACK_MIN_PCT",
    "ENTRY_BREAKOUT_VOLUME_MULT",
    "ENTRY_LIMIT_ORDER_MIN_DIFF",
    # Support/Resistance
    "SR_BOUNCE_THRESHOLD",
    "SR_RESISTANCE_CLOSE_THRESHOLD",
    "SR_SUPPORT_SUSTAINED_MOVE",
    "SR_VOLUME_CONFIRMATION_MULT",
    # Correlation
    "CORRELATION_CACHE_TTL",
    # Technical Scoring
    "TECH_SCORE_HIGH",
    "TECH_SCORE_GOOD",
    "TECH_SCORE_MODERATE",
    "TECH_SCORE_LOW",
    "TECH_SCORE_POOR",
    "TECH_ONLY_MIN_CONFIDENCE",
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
    "OPTIMISTIC_ROUND_TRIP_COST",
    "REALISTIC_ROUND_TRIP_COST",
    "PESSIMISTIC_ROUND_TRIP_COST",
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
