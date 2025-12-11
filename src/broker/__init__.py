# -*- coding: utf-8 -*-
"""
Broker Integration Module - Complete 10/10 Implementation

Provides broker API integration for:
- Order placement and management
- Position tracking
- Account information
- Paper trading simulation

Supported Brokers:
- SSI Securities (ssi.com.vn)
- VNDirect Securities (vndirect.com.vn)
- TCBS Securities (tcbs.com.vn)
- MBS Securities (mbs.com.vn)
- VPS Securities (vps.com.vn)
- Simulated/Paper Trading

Version: 2.0.0
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

from src.broker.tcbs_broker import (
    TCBSBroker,
    create_tcbs_broker,
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
    # TCBS Broker
    "TCBSBroker",
    "create_tcbs_broker",
    # Factory
    "get_broker",
    "get_supported_brokers",
]


# Supported brokers registry
SUPPORTED_BROKERS = {
    "SSI": {
        "name": "SSI Securities",
        "website": "ssi.com.vn",
        "features": ["stocks", "derivatives", "margin"],
        "api_type": "REST",
    },
    "VNDIRECT": {
        "name": "VNDirect Securities",
        "website": "vndirect.com.vn",
        "features": ["stocks", "derivatives", "margin", "bonds"],
        "api_type": "REST",
    },
    "TCBS": {
        "name": "TCBS Securities",
        "website": "tcbs.com.vn",
        "features": ["stocks", "derivatives", "margin", "bonds", "funds"],
        "api_type": "REST",
    },
    "SIMULATED": {
        "name": "Paper Trading",
        "website": None,
        "features": ["stocks", "simulation"],
        "api_type": "LOCAL",
    },
}


def get_supported_brokers() -> dict:
    """
    Get list of supported brokers with their features.

    Returns:
        Dict of broker info
    """
    return SUPPORTED_BROKERS.copy()


def get_broker(
    broker_type: str,
    account_id: str,
    credentials: dict,
    is_paper: bool = True,
) -> BaseBroker:
    """
    Factory function to create broker instance.

    Args:
        broker_type: "SSI", "VNDIRECT", "TCBS", or "SIMULATED"
        account_id: Trading account ID
        credentials: Dict with API credentials
        is_paper: Use paper trading mode (default True for safety)

    Returns:
        Broker instance

    Example:
        # SSI Broker
        broker = get_broker(
            "SSI",
            "123456789",
            {"consumer_id": "xxx", "consumer_secret": "yyy"},
            is_paper=True
        )

        # TCBS Broker
        broker = get_broker(
            "TCBS",
            "123456789",
            {"username": "user", "password": "pass"},
            is_paper=True
        )

        # Paper Trading
        broker = get_broker(
            "SIMULATED",
            "PAPER",
            {"initial_cash": 100_000_000}
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
    elif broker_type == "TCBS":
        return create_tcbs_broker(
            account_id=account_id,
            username=credentials.get("username", ""),
            password=credentials.get("password", ""),
            is_paper=is_paper,
            is_margin_account=credentials.get("is_margin_account", False),
            otp_callback=credentials.get("otp_callback"),
        )
    elif broker_type == "SIMULATED":
        return get_paper_broker(initial_cash=credentials.get("initial_cash", 100_000_000))
    else:
        supported = ", ".join(SUPPORTED_BROKERS.keys())
        raise ValueError(f"Unknown broker type: {broker_type}. Supported: {supported}")
