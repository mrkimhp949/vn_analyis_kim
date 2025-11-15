"""
Custom Exceptions for Trading Bot
Structured error handling với context
"""

from typing import Any, Dict, Optional


class TradingBotError(Exception):
    """Base exception for trading bot"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self):
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message


class ConfigurationError(TradingBotError):
    """Error in configuration"""


class DataLoadError(TradingBotError):
    """Error loading market data"""


class DataQualityError(TradingBotError):
    """Error in data quality (outliers, missing data, etc.)"""


class SignalValidationError(TradingBotError):
    """Error validating trading signal"""


class PositionSizingError(TradingBotError):
    """Error calculating position size"""


class RiskManagementError(TradingBotError):
    """Error in risk management (limits exceeded, etc.)"""


class APIConnectionError(TradingBotError):
    """Error connecting to API"""


class ModelPredictionError(TradingBotError):
    """Error in ML model prediction"""


class PortfolioError(TradingBotError):
    """Error in portfolio operations"""
