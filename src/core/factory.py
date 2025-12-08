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
from types import ModuleType
from typing import Any, Optional, Protocol, Union

from src.config.trading_config import TradingConfig

logger = logging.getLogger(__name__)

__all__ = [
    "create_orchestrator",
    "create_test_orchestrator",
    "create_data_loader",
    "create_ml_signal_generator",
    "create_strategy_manager",
    "create_portfolio_manager",
    "create_risk_service",
    "create_entry_service",
    "create_exit_service",
    "create_notification_service",
    "create_circuit_breaker",
    "create_paper_trading_account",
]


def create_data_loader() -> ModuleType:
    """
    Factory function to create data loader.

    Returns:
        ModuleType: load_data function from loader module
    """
    from src.data import loader

    logger.debug("✅ Created data loader module")
    return loader


def create_ml_signal_generator(config: TradingConfig) -> Any:
    """
    Factory function to create ML signal generator.

    Args:
        config: Trading configuration

    Returns:
        Any: MLSignalGenerator or EnhancedMLSignalGenerator instance
    """
    try:
        from src.ml.signals.enhanced_v2 import (
            EnhancedMLSignalGeneratorV2 as EnhancedMLSignalGenerator,
        )

        generator = EnhancedMLSignalGenerator(config)
        logger.debug("✅ Created EnhancedMLSignalGenerator")
        return generator
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"Could not create EnhancedMLSignalGenerator: {e}")
    except TypeError as e:
        logger.warning(f"EnhancedMLSignalGenerator init failed: {e}")

    # Fallback to basic generator
    from src.ml.signals.generator_v2 import MLSignalGeneratorV2

    generator = MLSignalGeneratorV2()
    logger.debug("✅ Created MLSignalGeneratorV2 (fallback)")
    return generator


def create_strategy_manager() -> Any:
    """
    Factory function to create strategy manager.

    Returns:
        Any: StrategyManager instance
    """
    from src.strategies.manager import StrategyManager

    manager = StrategyManager()
    logger.debug("✅ Created StrategyManager")
    return manager


def create_portfolio_manager() -> Any:
    """
    Factory function to create portfolio manager.

    Returns:
        Any: PortfolioManager singleton instance
    """
    from src.portfolio.manager import get_portfolio_manager

    manager = get_portfolio_manager()
    logger.debug("✅ Created PortfolioManager")
    return manager


def create_risk_service() -> Any:
    """
    Factory function to create risk service.

    Returns:
        Any: RiskService instance
    """
    from src.services.risk_service import get_risk_service

    service = get_risk_service()
    logger.debug("✅ Created RiskService")
    return service


def create_entry_service() -> Any:
    """
    Factory function to create entry service.

    Returns:
        Any: EntrySignalService instance
    """
    from src.services.entry_service import EntrySignalService

    service = EntrySignalService()
    logger.debug("✅ Created EntrySignalService")
    return service


def create_exit_service() -> Any:
    """
    Factory function to create exit service.

    Returns:
        Any: ExitManagementService instance
    """
    from src.services.exit_service import ExitManagementService

    service = ExitManagementService()
    logger.debug("✅ Created ExitManagementService")
    return service


def create_notification_service() -> ModuleType:
    """
    Factory function to create notification service.

    Returns:
        ModuleType: Telegram notification module
    """
    from src.notifications import telegram

    logger.debug("✅ Created notification service")
    return telegram


def create_circuit_breaker() -> Any:
    """
    Factory function to create circuit breaker.

    Returns:
        Any: CircuitBreaker instance
    """
    from src.risk.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker()
    logger.debug("✅ Created CircuitBreaker")
    return breaker


def create_paper_trading_account() -> Any:
    """
    Factory function to create paper trading account.

    Returns:
        Any: PaperTradingAccount instance
    """
    from src.portfolio.paper_trading import PaperTradingAccount

    account = PaperTradingAccount()
    logger.debug("✅ Created PaperTradingAccount")
    return account


def create_orchestrator(config: Optional[TradingConfig] = None) -> Any:
    """
    Main factory function to create fully configured orchestrator.

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
    return _create_orchestrator_internal(config=config)


def create_test_orchestrator(
    config: Optional[TradingConfig] = None,
    data_loader: Optional[Any] = None,
    ml_generator: Optional[Any] = None,
    strategy_manager: Optional[Any] = None,
    portfolio_manager: Optional[Any] = None,
    risk_service: Optional[Any] = None,
    entry_service: Optional[Any] = None,
    exit_service: Optional[Any] = None,
    notification_service: Optional[Any] = None,
    circuit_breaker: Optional[Any] = None,
    paper_account: Optional[Any] = None,
) -> Any:
    """
    Factory for testing - allows injecting mocks.

    Any parameter set to None will use the real implementation.
    Pass mock objects to override specific dependencies.

    Args:
        config: Trading configuration (optional)
        data_loader: Mock data loader (optional)
        ml_generator: Mock ML signal generator (optional)
        strategy_manager: Mock strategy manager (optional)
        portfolio_manager: Mock portfolio manager (optional)
        risk_service: Mock risk service (optional)
        entry_service: Mock entry service (optional)
        exit_service: Mock exit service (optional)
        notification_service: Mock notification service (optional)
        circuit_breaker: Mock circuit breaker (optional)
        paper_account: Mock paper trading account (optional)

    Returns:
        Any: TradingOrchestrator instance with injected dependencies

    Example:
        >>> mock_ml = MagicMock()
        >>> test_orch = create_test_orchestrator(ml_generator=mock_ml)
        >>> # Now orchestrator uses mock ML generator
    """
    return _create_orchestrator_internal(
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


def _create_orchestrator_internal(
    config: Optional[TradingConfig] = None,
    data_loader: Optional[Any] = None,
    ml_generator: Optional[Any] = None,
    strategy_manager: Optional[Any] = None,
    portfolio_manager: Optional[Any] = None,
    risk_service: Optional[Any] = None,
    entry_service: Optional[Any] = None,
    exit_service: Optional[Any] = None,
    notification_service: Optional[Any] = None,
    circuit_breaker: Optional[Any] = None,
    paper_account: Optional[Any] = None,
) -> Any:
    """
    Internal factory function to create orchestrator with optional dependency overrides.

    This is the shared implementation for both create_orchestrator and create_test_orchestrator.

    Args:
        config: Trading configuration (optional, will create default if not provided)
        data_loader: Data loader instance or None to create default
        ml_generator: ML signal generator instance or None to create default
        strategy_manager: Strategy manager instance or None to create default
        portfolio_manager: Portfolio manager instance or None to create default
        risk_service: Risk service instance or None to create default
        entry_service: Entry service instance or None to create default
        exit_service: Exit service instance or None to create default
        notification_service: Notification service instance or None to create default
        circuit_breaker: Circuit breaker instance or None to create default
        paper_account: Paper trading account instance or None to create default

    Returns:
        Any: TradingOrchestrator instance with all dependencies injected
    """
    # Create config if not provided
    if config is None:
        config = TradingConfig()
        logger.info("Created default TradingConfig")

    # Create dependencies - use provided or create new
    logger.info("🏭 Creating orchestrator dependencies...")

    data_loader = data_loader or create_data_loader()
    ml_generator = ml_generator or create_ml_signal_generator(config)
    strategy_manager = strategy_manager or create_strategy_manager()
    portfolio_manager = portfolio_manager or create_portfolio_manager()
    risk_service = risk_service or create_risk_service()
    entry_service = entry_service or create_entry_service()
    exit_service = exit_service or create_exit_service()
    notification_service = notification_service or create_notification_service()
    circuit_breaker = circuit_breaker or create_circuit_breaker()
    paper_account = paper_account or create_paper_trading_account()

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
