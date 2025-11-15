"""
Dependency Injection Factory

This module provides factory functions to create properly configured
instances of services, avoiding tight coupling in the orchestrator.

✅ Uses refactored TradingOrchestrator with flexible constructor that supports
   both legacy mode (bot_instance, chat_id) and modern dependency injection.

Usage:
    from src.core.factory import create_orchestrator

    # Creates TradingOrchestrator with all dependencies injected
    orchestrator = create_orchestrator()

    # Legacy usage still works
    from telegram import Bot
    orchestrator = TradingOrchestrator(Bot(token), chat_id="123")

    # Modern factory usage
    orchestrator = create_orchestrator()
"""

import logging
from typing import Optional

from src.config.trading_config import TradingConfig

logger = logging.getLogger(__name__)


def create_data_loader():
    """
    Factory function to create data loader

    Returns:
        load_data function from loader module
    """
    from src.data import loader

    logger.debug("✅ Created data loader module")
    return loader


def create_ml_signal_generator(config: TradingConfig):
    """
    Factory function to create ML signal generator

    Args:
        config: Trading configuration

    Returns:
        MLSignalGenerator or EnhancedMLSignalGenerator
    """
    try:
        from src.ml.signals.enhanced import EnhancedMLSignalGenerator

        generator = EnhancedMLSignalGenerator(config)
        logger.debug("✅ Created EnhancedMLSignalGenerator")
        return generator
    except Exception:
        logger.warning("Could not create EnhancedMLSignalGenerator")
        from src.ml.signals.generator import MLSignalGenerator

        generator = MLSignalGenerator()  # No config parameter
        logger.debug("✅ Created MLSignalGenerator (fallback)")
        return generator


def create_strategy_manager(config: TradingConfig):
    """
    Factory function to create strategy manager

    Args:
        config: Trading configuration

    Returns:
        StrategyManager instance
    """
    from src.strategies.manager import StrategyManager

    manager = StrategyManager()  # No config parameter
    logger.debug("✅ Created StrategyManager")
    return manager


def create_portfolio_manager():
    """
    Factory function to create portfolio manager

    Returns:
        PortfolioManager singleton instance
    """
    from src.portfolio.manager import get_portfolio_manager

    manager = get_portfolio_manager()
    logger.debug("✅ Created PortfolioManager")
    return manager


def create_risk_service():
    """
    Factory function to create risk service

    Returns:
        RiskService instance
    """
    from src.services.risk_service import get_risk_service

    service = get_risk_service()
    logger.debug("✅ Created RiskService")
    return service


def create_entry_service(config: TradingConfig):
    """
    Factory function to create entry service

    Args:
        config: Trading configuration

    Returns:
        EntrySignalService instance
    """
    from src.services.entry_service import EntrySignalService

    service = EntrySignalService()  # No config parameter
    logger.debug("✅ Created EntrySignalService")
    return service


def create_exit_service():
    """
    Factory function to create exit service

    Returns:
        ExitManagementService instance
    """
    from src.services.exit_service import ExitManagementService

    service = ExitManagementService()
    logger.debug("✅ Created ExitManagementService")
    return service


def create_notification_service():
    """
    Factory function to create notification service

    Returns:
        Telegram notification module
    """
    from src.notifications import telegram

    logger.debug("✅ Created notification service")
    return telegram


def create_circuit_breaker():
    """
    Factory function to create circuit breaker

    Returns:
        CircuitBreaker instance
    """
    from src.risk.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker()
    logger.debug("✅ Created CircuitBreaker")
    return breaker


def create_paper_trading_account():
    """
    Factory function to create paper trading account

    Returns:
        PaperTradingAccount instance
    """
    from src.portfolio.paper_trading import PaperTradingAccount

    account = PaperTradingAccount()
    logger.debug("✅ Created PaperTradingAccount")
    return account


def create_orchestrator(config: Optional[TradingConfig] = None):
    """
    Main factory function to create fully configured orchestrator

    This demonstrates Dependency Injection pattern:
    - All dependencies are created here
    - Injected into orchestrator via constructor
    - Orchestrator doesn't know how to create dependencies

    Args:
        config: Trading configuration (optional, will create default if not provided)

    Returns:
        TradingOrchestrator instance with all dependencies injected

    Example:
        >>> orchestrator = create_orchestrator()
        >>> orchestrator.run_scan()
    """
    # Create config if not provided
    if config is None:
        config = TradingConfig()
        logger.info("Created default TradingConfig")

    # Create all dependencies
    logger.info("🏭 Creating orchestrator dependencies...")

    data_loader = create_data_loader()
    ml_generator = create_ml_signal_generator(config)
    strategy_manager = create_strategy_manager(config)
    portfolio_manager = create_portfolio_manager()
    risk_service = create_risk_service()
    entry_service = create_entry_service(config)
    exit_service = create_exit_service()
    notification_service = create_notification_service()
    circuit_breaker = create_circuit_breaker()
    paper_account = create_paper_trading_account()

    # Import refactored orchestrator with dependency injection support
    from src.core.orchestrator import TradingOrchestrator

    # Inject all dependencies using modern pattern
    orchestrator = TradingOrchestrator(
        # Legacy params (optional for backward compatibility)
        bot_instance=None,
        chat_id=None,
        vnindex_df=None,
        # Modern dependency injection
        config=config,
        data_loader=data_loader,
        ml_generator=ml_generator,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_service=risk_service,
        entry_service=entry_service,
        exit_service=exit_service,
        notification_service=notification_service,
        circuit_breaker=circuit_breaker,
        paper_account=paper_account,
    )

    logger.info("✅ Orchestrator created with dependency injection")

    return orchestrator


# Example for testing - create orchestrator with mocked dependencies
def create_test_orchestrator(
    config=None,
    data_loader=None,
    ml_generator=None,
    strategy_manager=None,
    portfolio_manager=None,
    risk_service=None,
    entry_service=None,
    exit_service=None,
    notification_service=None,
    circuit_breaker=None,
    paper_account=None,
):
    """
    Factory for testing - allows injecting mocks

    Any parameter set to None will use the real implementation.
    Pass mock objects to override specific dependencies.

    Example:
        >>> mock_ml = MagicMock()
        >>> test_orch = create_test_orchestrator(ml_generator=mock_ml)
        >>> # Now orchestrator uses mock ML generator
    """
    if config is None:
        config = TradingConfig()

    # Use provided mocks or create real instances
    data_loader = data_loader or create_data_loader()
    ml_generator = ml_generator or create_ml_signal_generator(config)
    strategy_manager = strategy_manager or create_strategy_manager(config)
    portfolio_manager = portfolio_manager or create_portfolio_manager()
    risk_service = risk_service or create_risk_service()
    entry_service = entry_service or create_entry_service(config)
    exit_service = exit_service or create_exit_service()
    notification_service = notification_service or create_notification_service()
    circuit_breaker = circuit_breaker or create_circuit_breaker()
    paper_account = paper_account or create_paper_trading_account()

    from src.core.orchestrator import TradingOrchestrator

    return TradingOrchestrator(
        # Legacy params
        bot_instance=None,
        chat_id="",
        vnindex_df=None,
        # Modern dependency injection
        config=config,
        data_loader=data_loader,
        ml_generator=ml_generator,
        strategy_manager=strategy_manager,
        portfolio_manager=portfolio_manager,
        risk_service=risk_service,
        entry_service=entry_service,
        exit_service=exit_service,
        notification_service=notification_service,
        circuit_breaker=circuit_breaker,
        paper_account=paper_account,
    )
