# -*- coding: utf-8 -*-
"""
Vietnam Trading System 10/10

Complete trading system for Vietnam stock market with all features:

1. ✅ Real-time Data Integration
   - SSI, VNDirect API support
   - WebSocket streaming
   - Order book depth

2. ✅ Fundamental Analysis
   - P/E, P/B, ROE filters
   - Sector comparison
   - Financial health scoring

3. ✅ Earnings Calendar
   - Quarterly earnings tracking
   - Risk assessment before earnings
   - Position adjustment recommendations

4. ✅ Ex-Dividend Awareness
   - Ex-date tracking
   - Dividend yield analysis
   - Entry timing around dividends

5. ✅ Portfolio VaR
   - Historical VaR
   - Monte Carlo simulation
   - Stress testing scenarios

6. ✅ Broker Integration
   - Order placement
   - Position sync
   - Paper trading

7. ✅ Alert System
   - Telegram notifications
   - Webhook support
   - Multi-channel alerts

8. ✅ Vietnam Market Rules
   - Lot size (100 shares)
   - Tick size (10/50/100 VND)
   - Price limits (±7%/10%/15%)
   - T+2 settlement
   - ATO/ATC sessions

Author: Trading Bot Team
Version: 10.0.0
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORTS - All modules
# =============================================================================

# Vietnam Market Rules
from src.utils.vietnam_market import (
    VietnamMarketValidator,
    round_to_lot,
    get_tick_size,
    round_to_tick,
    get_exchange,
    get_price_limit,
    calculate_ceiling_floor,
    get_current_session,
    check_ato_atc_session,
    is_optimal_entry_time,
    validate_order,
    VN30_SYMBOLS,
    VN30_SECTORS,
)

# Session Trading
from src.market.session_trading import (
    SessionTradingManager,
    get_session_manager,
    analyze_entry_timing,
    is_optimal_entry_time as check_optimal_time,
)

# Settlement Timing
from src.strategies.settlement_timing import (
    SettlementTimingAnalyzer,
    get_settlement_analyzer,
)

# Market Regime
from src.market.regime_detector import (
    MarketRegimeDetector,
    get_regime_detector,
)

# Foreign Flow
from src.market.foreign_flow import (
    ForeignFlowAnalyzer,
    get_foreign_flow_analyzer,
)

# Risk Management
from src.strategies.risk_management import (
    EnhancedRiskManager,
)

# Exit Logic
from src.strategies.exit_logic import (
    ImprovedExitStrategy,
    ExitConfig,
    ExitReason,
)

# NEW: Real-time Data
try:
    from src.data.realtime_provider import (
        RealtimeDataManager,
        get_realtime_manager,
        setup_realtime_providers,
        RealtimeQuote,
        OrderBook,
    )

    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    logger.warning("Real-time provider not available")

# NEW: Fundamental Analysis
try:
    from src.data.fundamental_analyzer import (
        FundamentalAnalyzer,
        get_fundamental_analyzer,
        get_fundamental_score,
        FundamentalScore,
    )

    FUNDAMENTAL_AVAILABLE = True
except ImportError:
    FUNDAMENTAL_AVAILABLE = False
    logger.warning("Fundamental analyzer not available")

# NEW: Earnings Calendar
try:
    from src.data.earnings_calendar import (
        EarningsCalendarManager,
        get_earnings_manager,
        is_near_earnings,
        get_earnings_risk_multiplier,
        CorporateEvent,
    )

    EARNINGS_AVAILABLE = True
except ImportError:
    EARNINGS_AVAILABLE = False
    logger.warning("Earnings calendar not available")

# NEW: Portfolio VaR
try:
    from src.risk.portfolio_var import (
        PortfolioVaRCalculator,
        get_var_calculator,
        calculate_portfolio_var,
        run_stress_test,
        VaRResult,
        StressTestResult,
    )

    VAR_AVAILABLE = True
except ImportError:
    VAR_AVAILABLE = False
    logger.warning("Portfolio VaR not available")

# NEW: Broker Integration
try:
    from src.broker.base_broker import (
        BaseBroker,
        SimulatedBroker,
        get_paper_broker,
        Order,
        Position,
        OrderSide,
        OrderType,
        OrderStatus,
    )

    BROKER_AVAILABLE = True
except ImportError:
    BROKER_AVAILABLE = False
    logger.warning("Broker integration not available")

# NEW: Alert System
try:
    from src.notifications.alert_manager import (
        AlertManager,
        get_alert_manager,
        configure_telegram,
        send_alert,
        AlertType,
        AlertPriority,
    )

    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False
    logger.warning("Alert system not available")

# NEW: Enhanced Entry v2
try:
    from src.strategies.enhanced_entry_v2 import (
        EnhancedEntryLogicV2,
        get_enhanced_entry_v2,
        EnhancedEntryResult,
    )

    ENHANCED_ENTRY_AVAILABLE = True
except ImportError:
    ENHANCED_ENTRY_AVAILABLE = False
    logger.warning("Enhanced entry v2 not available")


# =============================================================================
# MAIN TRADING SYSTEM CLASS
# =============================================================================


class VietnamTradingSystem:
    """
    Complete Vietnam Trading System

    Integrates all components for 10/10 trading logic
    """

    def __init__(
        self,
        initial_capital: float = 100_000_000,
        enable_realtime: bool = True,
        enable_paper_trading: bool = True,
        telegram_token: str = "",
        telegram_chat_id: str = "",
    ):
        """
        Initialize trading system

        Args:
            initial_capital: Initial capital in VND
            enable_realtime: Enable real-time data
            enable_paper_trading: Enable paper trading
            telegram_token: Telegram bot token
            telegram_chat_id: Telegram chat ID
        """
        self.capital = initial_capital

        # Initialize components
        self._init_components(
            enable_realtime, enable_paper_trading, telegram_token, telegram_chat_id
        )

        logger.info("=" * 60)
        logger.info("🇻🇳 VIETNAM TRADING SYSTEM 10/10 INITIALIZED")
        logger.info("=" * 60)
        self._log_status()

    def _init_components(
        self,
        enable_realtime: bool,
        enable_paper_trading: bool,
        telegram_token: str,
        telegram_chat_id: str,
    ):
        """Initialize all components"""

        # Core components (always available)
        self.market_validator = VietnamMarketValidator()
        self.session_manager = get_session_manager()
        self.settlement_analyzer = get_settlement_analyzer()
        self.risk_manager = EnhancedRiskManager(total_capital=self.capital)
        self.exit_strategy = ImprovedExitStrategy()

        # Real-time data
        self.realtime_manager = None
        if enable_realtime and REALTIME_AVAILABLE:
            self.realtime_manager = get_realtime_manager()

        # Fundamental analyzer
        self.fundamental_analyzer = None
        if FUNDAMENTAL_AVAILABLE:
            self.fundamental_analyzer = get_fundamental_analyzer()

        # Earnings calendar
        self.earnings_manager = None
        if EARNINGS_AVAILABLE:
            self.earnings_manager = get_earnings_manager()

        # VaR calculator
        self.var_calculator = None
        if VAR_AVAILABLE:
            self.var_calculator = get_var_calculator()

        # Paper trading broker
        self.broker = None
        if enable_paper_trading and BROKER_AVAILABLE:
            self.broker = get_paper_broker(initial_cash=self.capital)

        # Alert manager
        self.alert_manager = None
        if ALERTS_AVAILABLE:
            self.alert_manager = get_alert_manager()
            if telegram_token and telegram_chat_id:
                self.alert_manager.configure_telegram(telegram_token, telegram_chat_id)

        # Enhanced entry logic
        self.entry_logic = None
        if ENHANCED_ENTRY_AVAILABLE:
            self.entry_logic = get_enhanced_entry_v2()

    def _log_status(self):
        """Log component status"""
        components = [
            ("Real-time Data", REALTIME_AVAILABLE and self.realtime_manager is not None),
            ("Fundamental Analysis", FUNDAMENTAL_AVAILABLE),
            ("Earnings Calendar", EARNINGS_AVAILABLE),
            ("Portfolio VaR", VAR_AVAILABLE),
            ("Broker Integration", BROKER_AVAILABLE and self.broker is not None),
            ("Alert System", ALERTS_AVAILABLE),
            ("Enhanced Entry", ENHANCED_ENTRY_AVAILABLE),
        ]

        for name, available in components:
            status = "✅" if available else "❌"
            logger.info(f"  {status} {name}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "capital": self.capital,
            "components": {
                "realtime": REALTIME_AVAILABLE and self.realtime_manager is not None,
                "fundamental": FUNDAMENTAL_AVAILABLE,
                "earnings": EARNINGS_AVAILABLE,
                "var": VAR_AVAILABLE,
                "broker": BROKER_AVAILABLE and self.broker is not None,
                "alerts": ALERTS_AVAILABLE,
                "enhanced_entry": ENHANCED_ENTRY_AVAILABLE,
            },
            "session": self.session_manager.get_current_session().__dict__,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_trading_system(
    capital: float = 100_000_000, telegram_token: str = "", telegram_chat_id: str = ""
) -> VietnamTradingSystem:
    """Create and configure trading system"""
    return VietnamTradingSystem(
        initial_capital=capital, telegram_token=telegram_token, telegram_chat_id=telegram_chat_id
    )


def get_feature_status() -> Dict[str, bool]:
    """Get status of all features"""
    return {
        "realtime_data": REALTIME_AVAILABLE,
        "fundamental_analysis": FUNDAMENTAL_AVAILABLE,
        "earnings_calendar": EARNINGS_AVAILABLE,
        "portfolio_var": VAR_AVAILABLE,
        "broker_integration": BROKER_AVAILABLE,
        "alert_system": ALERTS_AVAILABLE,
        "enhanced_entry": ENHANCED_ENTRY_AVAILABLE,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 70)
    print("🇻🇳 VIETNAM TRADING SYSTEM 10/10 - TEST")
    print("=" * 70 + "\n")

    # Check features
    print("📋 FEATURE STATUS:")
    print("-" * 40)
    features = get_feature_status()
    for feature, available in features.items():
        status = "✅" if available else "❌"
        print(f"  {status} {feature.replace('_', ' ').title()}")

    # Create system
    print("\n" + "-" * 40)
    print("🚀 INITIALIZING SYSTEM...")
    print("-" * 40)

    system = create_trading_system(capital=500_000_000)

    # Test session
    print("\n📊 CURRENT SESSION:")
    session = system.session_manager.get_current_session()
    print(f"  Session: {session.session_type.value}")
    print(f"  Entry Quality: {session.entry_quality}")
    print(f"  Risk Level: {session.risk_level}")

    # Test market validator
    print("\n🔍 MARKET VALIDATION TEST:")
    valid, details = validate_order("VCB", 150, 95123, "LO")
    print(f"  Order Valid: {valid}")
    print(f"  Corrected Shares: {details['corrected_shares']}")
    print(f"  Corrected Price: {details['corrected_price']:,.0f}")

    print("\n" + "=" * 70)
    print("✅ VIETNAM TRADING SYSTEM 10/10 - READY")
    print("=" * 70)
