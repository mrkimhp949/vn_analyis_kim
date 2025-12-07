# -*- coding: utf-8 -*-
"""
Broker Integration Module

Provides broker API integration for:
- Order placement and management
- Position tracking
- Account information
- Paper trading simulation

Supported Brokers:
- SSI Securities (ssi.com.vn)
- VNDirect Securities (vndirect.com.vn)
- Simulated/Paper Trading
"""

from src.broker.base_broker import (
    BaseBroker,
    SimulatedBroker,
    SimulationConfig,
    Order,
    Position,
    AccountInfo,
    OrderSide,
    OrderType,
    OrderStatus,
    get_paper_broker,
)

from src.broker.ssi_broker import (
    SSIBroker,
    create_ssi_broker,
)

from src.broker.vndirect_broker import (
    VNDirectBroker,
    create_vndirect_broker,
)

__all__ = [
    # Base classes
    "BaseBroker",
    "SimulatedBroker",
    "SimulationConfig",
    "Order",
    "Position",
    "AccountInfo",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "get_paper_broker",
    # SSI Broker
    "SSIBroker",
    "create_ssi_broker",
    # VNDirect Broker
    "VNDirectBroker",
    "create_vndirect_broker",
]


def get_broker(
    broker_type: str, account_id: str, credentials: dict, is_paper: bool = True
) -> BaseBroker:
    """
    Factory function to create broker instance

    Args:
        broker_type: "SSI", "VNDIRECT", or "SIMULATED"
        account_id: Trading account ID
        credentials: Dict with API credentials
        is_paper: Use paper trading mode

    Returns:
        Broker instance

    Example:
        broker = get_broker(
            "SSI",
            "123456789",
            {"consumer_id": "xxx", "consumer_secret": "yyy"},
            is_paper=True
        )
    """
    broker_type = broker_type.upper()

    if broker_type == "SSI":
        return create_ssi_broker(
            account_id=account_id,
            consumer_id=credentials.get("consumer_id", ""),
            consumer_secret=credentials.get("consumer_secret", ""),
            is_paper=is_paper,
        )
    elif broker_type == "VNDIRECT":
        return create_vndirect_broker(
            account_id=account_id,
            username=credentials.get("username", ""),
            password=credentials.get("password", ""),
            is_paper=is_paper,
            otp_callback=credentials.get("otp_callback"),
        )
    elif broker_type == "SIMULATED":
        return get_paper_broker(initial_cash=credentials.get("initial_cash", 100_000_000))
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")
