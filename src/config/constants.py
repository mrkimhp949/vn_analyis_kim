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

# Adaptive Holding Days by Market Regime
HOLDING_DAYS_BULL_STRONG_TREND = 20  # Bull + ADX > 25
HOLDING_DAYS_BULL_WEAK_TREND = 15  # Bull + ADX <= 25
HOLDING_DAYS_SIDEWAYS_TREND = 12  # Sideways + ADX > 20
HOLDING_DAYS_SIDEWAYS_NO_TREND = 10  # Sideways + ADX <= 20
HOLDING_DAYS_BEAR_STRONG_TREND = 8  # Bear + ADX > 25
HOLDING_DAYS_BEAR_WEAK_TREND = 6  # Bear + ADX <= 25
HOLDING_DAYS_HIGH_VOLATILITY = 5  # High volatility - exit fast
HOLDING_DAYS_DEFAULT = 15  # Default fallback
ADX_STRONG_TREND_THRESHOLD = 25  # ADX threshold for strong trend
ADX_WEAK_TREND_THRESHOLD = 20  # ADX threshold for weak trend


def get_adaptive_holding_days(regime: str, adx: float = 20) -> int:
    """
    Get adaptive max holding days based on market regime and trend strength.

    Vietnam market characteristics:
    - T+2.5 settlement → minimum 3 days practical holding
    - High volatility → shorter holding to lock profits
    - Strong trends (ADX > 25) → can hold longer

    Args:
        regime: Market regime (BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY)
        adx: Average Directional Index (trend strength, default 20)

    Returns:
        Maximum holding days (5-20)
    """
    if regime == "BULL":
        if adx > ADX_STRONG_TREND_THRESHOLD:
            return HOLDING_DAYS_BULL_STRONG_TREND  # 20 days - strong trend
        return HOLDING_DAYS_BULL_WEAK_TREND  # 15 days - weak trend
    elif regime == "SIDEWAYS":
        if adx > ADX_WEAK_TREND_THRESHOLD:
            return HOLDING_DAYS_SIDEWAYS_TREND  # 12 days - some trend
        return HOLDING_DAYS_SIDEWAYS_NO_TREND  # 10 days - no trend
    elif regime == "BEAR":
        if adx > ADX_STRONG_TREND_THRESHOLD:
            return HOLDING_DAYS_BEAR_STRONG_TREND  # 8 days - strong downtrend
        return HOLDING_DAYS_BEAR_WEAK_TREND  # 6 days - exit faster
    elif regime == "HIGH_VOLATILITY":
        return HOLDING_DAYS_HIGH_VOLATILITY  # 5 days - very short
    else:
        return HOLDING_DAYS_DEFAULT  # 15 days default


# Commission and Slippage (Vietnam Market) - IMPROVED v4.0
# REALISTIC transaction costs for Vietnam market based on actual trading experience
#
# Cost breakdown per trade:
#   - Brokerage fee: 0.15-0.30% (depending on broker, VPS/SSI/TCBS etc.)
#   - Stock transaction tax: 0.10% (sell only - government tax)
#   - Exchange fees (HOSE/HNX): 0.03%
#   - Transfer/depository fees: 0.02%
#   - Slippage: 0.30-0.50% (market orders, especially for large orders or illiquid stocks)
#
# Detailed cost calculation:
VN_BROKERAGE_FEE = 0.0025  # 0.25% (average broker fee)
VN_STOCK_TAX = 0.0010  # 0.10% (sell only - government tax)
VN_EXCHANGE_FEE = 0.0003  # 0.03% (HOSE/HNX fee)
VN_TRANSFER_FEE = 0.0002  # 0.02% (depository fee)
VN_SLIPPAGE_MARKET_ORDER = 0.0040  # 0.40% (realistic slippage for market orders)
VN_SLIPPAGE_LIMIT_ORDER = 0.0015  # 0.15% (slippage for limit orders)

# Total cost per side:
# BUY: brokerage + exchange + transfer + slippage = 0.25% + 0.03% + 0.02% + 0.40% = 0.70%
VN_BUY_COST = (
    VN_BROKERAGE_FEE + VN_EXCHANGE_FEE + VN_TRANSFER_FEE + VN_SLIPPAGE_MARKET_ORDER
)  # 0.70%

# SELL: brokerage + tax + exchange + slippage = 0.25% + 0.10% + 0.03% + 0.40% = 0.78%
VN_SELL_COST = VN_BROKERAGE_FEE + VN_STOCK_TAX + VN_EXCHANGE_FEE + VN_SLIPPAGE_MARKET_ORDER  # 0.78%

# Round trip costs (buy + sell):
VN_ROUND_TRIP_COST_MARKET = VN_BUY_COST + VN_SELL_COST  # 1.48% with market orders
VN_ROUND_TRIP_COST_LIMIT = 0.0060 + 0.0053  # 1.13% with limit orders (lower slippage)

# Scenario-based costs for different trading styles:
VN_OPTIMISTIC_ROUND_TRIP = 0.0100  # 1.0% (best case: limit orders, low slippage, discount broker)
VN_REALISTIC_ROUND_TRIP = 0.0148  # 1.48% (realistic: market orders, normal execution)
VN_PESSIMISTIC_ROUND_TRIP = (
    0.0200  # 2.0% (worst case: high slippage, large orders, illiquid stocks)
)

# Default values - USE REALISTIC ESTIMATES
DEFAULT_COMMISSION_RATE = VN_BUY_COST  # 0.70% per trade (buy side)
DEFAULT_SLIPPAGE = VN_SLIPPAGE_MARKET_ORDER  # 0.40% slippage
TOTAL_TRANSACTION_COST = VN_BUY_COST  # 0.70% per trade
ROUND_TRIP_COST = VN_REALISTIC_ROUND_TRIP  # 1.48% round trip (IMPROVED from 1.6%)

# Dynamic Slippage by Liquidity Tier
VN_SLIPPAGE_VN30 = 0.003  # 0.3% for VN30 blue chips (highest liquidity)
VN_SLIPPAGE_LIQUID = 0.004  # 0.4% for liquid stocks (> 3B VND daily)
VN_SLIPPAGE_MEDIUM = 0.006  # 0.6% for medium liquidity (1-3B VND daily)
VN_SLIPPAGE_ILLIQUID = 0.010  # 1.0% for illiquid stocks (< 1B VND daily)

# VN30 Symbols (Top 30 largest market cap on HOSE)
VN30_SYMBOLS = [
    "ACB",
    "BCM",
    "BID",
    "BVH",
    "CTG",
    "FPT",
    "GAS",
    "GVR",
    "HDB",
    "HPG",
    "MBB",
    "MSN",
    "MWG",
    "PLX",
    "POW",
    "SAB",
    "SHB",
    "SSB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VNM",
    "VPB",
    "VRE",
]


def get_dynamic_slippage(symbol: str, liquidity_value: float) -> float:
    """
    Get dynamic slippage based on symbol and liquidity.

    VN30 stocks have highest liquidity and lowest slippage.
    Other stocks are tiered by daily trading value.

    Args:
        symbol: Stock symbol (e.g., "VNM", "HPG")
        liquidity_value: Average daily trading value in VND

    Returns:
        Slippage rate (0.003-0.010)
    """
    # VN30 blue chips have best liquidity
    if symbol.upper() in VN30_SYMBOLS:
        return VN_SLIPPAGE_VN30  # 0.3%

    # Tier by liquidity value
    if liquidity_value > 3_000_000_000:  # > 3B VND
        return VN_SLIPPAGE_LIQUID  # 0.4%
    elif liquidity_value > 1_000_000_000:  # 1-3B VND
        return VN_SLIPPAGE_MEDIUM  # 0.6%
    else:  # < 1B VND
        return VN_SLIPPAGE_ILLIQUID  # 1.0%


# Alternative costs for different scenarios (backward compatible)
OPTIMISTIC_ROUND_TRIP_COST = VN_OPTIMISTIC_ROUND_TRIP  # 1.0% with best execution
REALISTIC_ROUND_TRIP_COST = VN_REALISTIC_ROUND_TRIP  # 1.48% with market orders (default)
PESSIMISTIC_ROUND_TRIP_COST = VN_PESSIMISTIC_ROUND_TRIP  # 2.0% with poor execution

# Vietnam Market Specific Constants
VIETNAM_PRICE_LIMIT_PERCENT = 0.07  # ±7% daily price limit (floor/ceiling) - HOSE
VIETNAM_LOT_SIZE = 100  # Minimum trading lot size
VIETNAM_SETTLEMENT_DAYS = 2  # T+2 settlement (actually T+2.5)
VIETNAM_TICK_SIZE = 10  # 10 VND minimum tick size for most stocks

# Exchange-specific price limits
VN_HOSE_PRICE_LIMIT = 0.07  # ±7% for HOSE
VN_HNX_PRICE_LIMIT = 0.10  # ±10% for HNX
VN_UPCOM_PRICE_LIMIT = 0.15  # ±15% for UPCOM

# Tick sizes by price range (HOSE rules)
VN_TICK_LOW = 10  # Price < 10,000 VND
VN_TICK_MID = 50  # 10,000 <= Price < 50,000 VND
VN_TICK_HIGH = 100  # Price >= 50,000 VND

# ATO/ATC Session Settings
VN_ALLOW_ATO_ATC_TRADING = False  # Block trading during auction sessions by default
VN_ATO_ATC_PENALTY = -15  # Confidence penalty for ATO/ATC trading

# Gap Protection - IMPROVED v4.1
VN_GAP_DOWN_EXIT_THRESHOLD = -0.025  # TIGHTENED: Exit on 2.5% gap down (with profit)
VN_GAP_DOWN_EMERGENCY_THRESHOLD = -0.04  # TIGHTENED: Emergency exit on 4% gap down
VN_GAP_UP_PROFIT_TAKE_THRESHOLD = 0.04  # NEW: Consider profit taking on 4%+ gap up

# Distribution Volume
VN_DISTRIBUTION_VOLUME_MULT = 2.0  # Volume > 2x avg = distribution

# NEW v4.1: Session-based trading rules
VN_AVOID_FIRST_15_MINUTES = True  # Avoid trading in first 15 min (ATO volatility)
VN_AVOID_LAST_15_MINUTES = True  # Avoid trading in last 15 min (ATC volatility)
VN_LUNCH_GAP_PROTECTION = True  # Exit profitable positions before lunch
VN_LUNCH_EXIT_MIN_PROFIT = 0.02  # Min 2% profit to exit before lunch

# NEW v4.1: Intraday volatility limits
VN_MAX_INTRADAY_RANGE_FOR_ENTRY = 0.045  # TIGHTENED: Max 4.5% intraday range for entry
VN_HIGH_VOLATILITY_PENALTY = -10  # Confidence penalty for high intraday volatility

# NEW v4.1: Foreign flow thresholds (smart money tracking)
VN_FOREIGN_NET_BUY_BONUS = 8  # Confidence bonus for net foreign buying
VN_FOREIGN_NET_SELL_PENALTY = -12  # Confidence penalty for net foreign selling
VN_FOREIGN_FLOW_LOOKBACK_DAYS = 5  # Days to look back for foreign flow trend

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

# ML Confidence Thresholds for Dynamic Weighting
ML_HIGH_CONFIDENCE_THRESHOLD = 70  # High confidence threshold
ML_MEDIUM_CONFIDENCE_THRESHOLD = 60  # Medium confidence threshold
ML_HIGH_CONFIDENCE_WEIGHT = 1.5  # Weight for high confidence ML signals
ML_MEDIUM_CONFIDENCE_WEIGHT = 1.0  # Weight for medium confidence ML signals
ML_LOW_CONFIDENCE_WEIGHT = 0.5  # Weight for low confidence - trust technical more


def get_ml_signal_weight(ml_confidence: float) -> float:
    """
    Get dynamic ML signal weight based on confidence level.

    Higher confidence = trust ML more, lower confidence = trust technical more.

    Args:
        ml_confidence: ML model confidence score (0-100)

    Returns:
        Weight multiplier for ML signal (0.5-1.5)
    """
    if ml_confidence >= ML_HIGH_CONFIDENCE_THRESHOLD:
        return ML_HIGH_CONFIDENCE_WEIGHT  # 1.5 - High confidence
    elif ml_confidence >= ML_MEDIUM_CONFIDENCE_THRESHOLD:
        return ML_MEDIUM_CONFIDENCE_WEIGHT  # 1.0 - Medium confidence
    else:
        return ML_LOW_CONFIDENCE_WEIGHT  # 0.5 - Low confidence, trust technical more


# Validation Thresholds - TIGHTENED v3.0
MAX_CORRELATION = 0.65  # TIGHTENED: Maximum 0.65 correlation between positions
DIVERSIFICATION_PENALTY = 25  # TIGHTENED: 25 points deducted per warning

# NEW: Vietnam Market Specific Thresholds - OPTIMIZED v4.0
# IMPROVED: Tiered liquidity for different market caps - LOWERED to capture more opportunities
VN_MIN_LIQUIDITY_VALUE = 500_000_000  # LOWERED: 500M VND for small caps (was 1B)
VN_MID_CAP_LIQUIDITY_VALUE = 1_000_000_000  # LOWERED: 1B VND for mid caps (was 2B)
VN_LARGE_CAP_LIQUIDITY_VALUE = 3_000_000_000  # LOWERED: 3B VND for large caps (was 5B)
VN_CRITICAL_LIQUIDITY_VALUE = 300_000_000  # LOWERED: 300M VND critical minimum (was 500M)
VN_MIN_VOLUME = 25_000  # LOWERED: 25K shares minimum (was 50K)
VN_MAX_INTRADAY_RANGE = 5.0  # 5% max intraday range for entry
VN_OPTIMAL_ENTRY_TIMES = [(9, 30, 10, 30), (13, 30, 14, 30)]  # Optimal entry windows

# NEW: Micro cap tier for speculative plays
VN_MICRO_CAP_LIQUIDITY_VALUE = 300_000_000  # 300M VND for micro caps
VN_MICRO_CAP_MIN_VOLUME = 15_000  # 15K shares minimum for micro caps

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
    # Dynamic Functions
    "get_ml_signal_weight",
    "get_dynamic_slippage",
    "get_adaptive_holding_days",
    # Adaptive Holding Days Constants
    "HOLDING_DAYS_BULL_STRONG_TREND",
    "HOLDING_DAYS_BULL_WEAK_TREND",
    "HOLDING_DAYS_SIDEWAYS_TREND",
    "HOLDING_DAYS_SIDEWAYS_NO_TREND",
    "HOLDING_DAYS_BEAR_STRONG_TREND",
    "HOLDING_DAYS_BEAR_WEAK_TREND",
    "HOLDING_DAYS_HIGH_VOLATILITY",
    "HOLDING_DAYS_DEFAULT",
    "ADX_STRONG_TREND_THRESHOLD",
    "ADX_WEAK_TREND_THRESHOLD",
    # VN30 Symbols
    "VN30_SYMBOLS",
    # ML Confidence Weighting
    "ML_HIGH_CONFIDENCE_THRESHOLD",
    "ML_MEDIUM_CONFIDENCE_THRESHOLD",
    "ML_HIGH_CONFIDENCE_WEIGHT",
    "ML_MEDIUM_CONFIDENCE_WEIGHT",
    "ML_LOW_CONFIDENCE_WEIGHT",
    # Dynamic Slippage Tiers
    "VN_SLIPPAGE_VN30",
    "VN_SLIPPAGE_LIQUID",
    "VN_SLIPPAGE_MEDIUM",
    "VN_SLIPPAGE_ILLIQUID",
    # Vietnam Market Extended
    "VN_MIN_LIQUIDITY_VALUE",
    "VN_MID_CAP_LIQUIDITY_VALUE",
    "VN_LARGE_CAP_LIQUIDITY_VALUE",
    "VN_CRITICAL_LIQUIDITY_VALUE",
    "VN_MICRO_CAP_LIQUIDITY_VALUE",
    "VN_MICRO_CAP_MIN_VOLUME",
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
    # Transaction costs - Vietnam specific
    "VN_BROKERAGE_FEE",
    "VN_STOCK_TAX",
    "VN_EXCHANGE_FEE",
    "VN_TRANSFER_FEE",
    "VN_SLIPPAGE_MARKET_ORDER",
    "VN_SLIPPAGE_LIMIT_ORDER",
    "VN_BUY_COST",
    "VN_SELL_COST",
    "VN_ROUND_TRIP_COST_MARKET",
    "VN_ROUND_TRIP_COST_LIMIT",
    "VN_OPTIMISTIC_ROUND_TRIP",
    "VN_REALISTIC_ROUND_TRIP",
    "VN_PESSIMISTIC_ROUND_TRIP",
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
