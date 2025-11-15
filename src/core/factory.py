"""
Dependency Injection Factory

This module provides factory functions to create properly configured
instances of services, avoiding tight coupling in the orchestrator.

Usage:
    from src.core.factory import create_orchestrator

    orchestrator = create_orchestrator()
    orchestrator.run_scan()
"""

import logging
from typing import Optional
from src.config.trading_config import TradingConfig

logger = logging.getLogger(__name__)


def create_data_loader():
    """
    Factory function to create data loader

    Returns:
        TCBSDataLoader instance
    """
    from src.data.loader import TCBSDataLoader

    loader = TCBSDataLoader()
    logger.debug("✅ Created TCBSDataLoader")
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
    except Exception as e:
        logger.warning(f"Could not create EnhancedMLSignalGenerator: {e}")
        from src.ml.signals.generator import MLSignalGenerator

        generator = MLSignalGenerator(config)
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

    manager = StrategyManager(config)
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
        EntryService instance
    """
    from src.services.entry_service import EntryService

    service = EntryService(config)
    logger.debug("✅ Created EntryService")
    return service


def create_exit_service():
    """
    Factory function to create exit service

    Returns:
        ExitService instance
    """
    from src.services.exit_service import ExitService

    service = ExitService()
    logger.debug("✅ Created ExitService")
    return service


def create_notification_service():
    """
    Factory function to create notification service

    Returns:
        NotificationService instance
    """
    from src.notifications.telegram import get_notification_service

    service = get_notification_service()
    logger.debug("✅ Created NotificationService")
    return service


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

    # Import orchestrator
    from src.core.orchestrator_v2 import TradingOrchestratorV2

    # Inject all dependencies
    orchestrator = TradingOrchestratorV2(
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

    from src.core.orchestrator_v2 import TradingOrchestratorV2

    return TradingOrchestratorV2(
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
