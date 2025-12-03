"""Market analysis modules"""

from src.market.regime_detector import (
    MarketRegime,
    MarketRegimeDetector,
    MarketRegimeAnalyzer,  # Legacy alias
    EnhancedMarketRegime,
    EnhancedRegimeDetector,  # Alias for MarketRegimeDetector
    detect_regime,
    detect_enhanced_regime,
    get_regime_detector,
    get_enhanced_regime_detector,
    check_market_before_trading,
    get_market_position_adjustment,
)

ENHANCED_REGIME_AVAILABLE = True

# Session trading (ATO/ATC)
try:
    from src.market.session_trading import (
        SessionType,
        OrderType,
        SessionInfo,
        EntryTimingResult,
        SessionTradingManager,
        get_session_manager,
        get_current_session,
        analyze_entry_timing,
        is_optimal_entry_time,
    )

    SESSION_TRADING_AVAILABLE = True
except ImportError:
    SESSION_TRADING_AVAILABLE = False

# Vietnam market indicators
try:
    from src.market.vietnam_indicators import (
        VietnamMarketIndicators,
        get_vietnam_indicators,
    )

    VIETNAM_INDICATORS_AVAILABLE = True
except ImportError:
    VIETNAM_INDICATORS_AVAILABLE = False

__all__ = [
    # Core regime detection
    "MarketRegime",
    "MarketRegimeDetector",
    "MarketRegimeAnalyzer",  # Legacy alias
    "detect_regime",
    "get_regime_detector",
    "check_market_before_trading",
    "get_market_position_adjustment",
    # Enhanced regime detection (now unified)
    "EnhancedMarketRegime",
    "EnhancedRegimeDetector",
    "detect_enhanced_regime",
    "get_enhanced_regime_detector",
    "ENHANCED_REGIME_AVAILABLE",
    # Session trading
    "SessionType",
    "OrderType",
    "SessionInfo",
    "EntryTimingResult",
    "SessionTradingManager",
    "get_session_manager",
    "get_current_session",
    "analyze_entry_timing",
    "is_optimal_entry_time",
    "SESSION_TRADING_AVAILABLE",
    # Vietnam indicators
    "VietnamMarketIndicators",
    "get_vietnam_indicators",
    "VIETNAM_INDICATORS_AVAILABLE",
]
