"""
Safe DataFrame Utilities

Provides safe access methods for DataFrame operations to prevent
IndexError and other common pandas-related crashes.
"""

import logging
from typing import Any, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def safe_get_latest(df: pd.DataFrame, column: str, default: Any = None) -> Any:
    """
    Safely get the latest value from a DataFrame column.

    Args:
        df: DataFrame to access
        column: Column name to get value from
        default: Default value if access fails

    Returns:
        Latest value or default
    """
    try:
        if len(df) == 0:
            logger.debug(f"DataFrame is empty, returning default for {column}")
            return default

        if column not in df.columns:
            logger.debug(f"Column {column} not found, returning default")
            return default

        value = df[column].iloc[-1]

        # Check for NaN
        if pd.isna(value):
            logger.debug(f"NaN value in {column}, returning default")
            return default

        return value

    except Exception as e:
        logger.warning(f"Error accessing {column}: {e}, returning default")
        return default


def safe_get_range(
    df: pd.DataFrame,
    column: str,
    start: int = -20,
    end: Optional[int] = None,
    default: Any = None,
) -> Any:
    """
    Safely get a range of values from DataFrame column.

    Args:
        df: DataFrame to access
        column: Column name
        start: Start index (negative for from end)
        end: End index (None for until end)
        default: Default value if access fails

    Returns:
        Series slice or default
    """
    try:
        if len(df) == 0 or column not in df.columns:
            return default

        if abs(start) > len(df):
            logger.debug(f"Start index {start} out of bounds for {len(df)} rows")
            return default

        if end is None:
            return df[column].iloc[start:]
        else:
            return df[column].iloc[start:end]

    except (IndexError, KeyError, AttributeError) as e:
        logger.warning(f"Error accessing range {column}[{start}:{end}]: {e}")
        return default


def safe_rolling_operation(
    df: pd.DataFrame,
    column: str,
    window: int,
    operation: str = "mean",
    default: Any = None,
) -> Any:
    """
    Safely perform rolling operations on DataFrame column.

    Args:
        df: DataFrame to operate on
        column: Column name
        window: Rolling window size
        operation: Operation ('mean', 'min', 'max', 'std', etc.)
        default: Default value if operation fails

    Returns:
        Result of rolling operation or default
    """
    try:
        if len(df) < window or column not in df.columns:
            return default

        rolling_obj = df[column].rolling(window)

        if operation == "mean":
            result = rolling_obj.mean().iloc[-1]
        elif operation == "min":
            result = rolling_obj.min().iloc[-1]
        elif operation == "max":
            result = rolling_obj.max().iloc[-1]
        elif operation == "std":
            result = rolling_obj.std().iloc[-1]
        else:
            logger.error(f"Unsupported operation: {operation}")
            return default

        return result if not pd.isna(result) else default

    except Exception as e:
        logger.warning(f"Error in rolling {operation} for {column}: {e}")
        return default


def validate_dataframe_basic(
    df: pd.DataFrame, min_rows: int = 1, required_columns: Optional[list] = None
) -> bool:
    """
    Basic DataFrame validation.

    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        required_columns: List of required column names

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if DataFrame exists and is not None
        if df is None:
            logger.error("DataFrame is None")
            return False

        # Check minimum rows
        if len(df) < min_rows:
            logger.error(f"DataFrame has {len(df)} rows, minimum {min_rows} required")
            return False

        # Check required columns
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating DataFrame: {e}")
        return False


# Convenience functions for common OHLCV operations
def safe_get_close_price(df: pd.DataFrame, default: float = 0.0) -> float:
    """Get latest close price safely"""
    return safe_get_latest(df, "close", default)


def safe_get_volume(df: pd.DataFrame, default: float = 0.0) -> float:
    """Get latest volume safely"""
    return safe_get_latest(df, "volume", default)


def safe_get_support_resistance(df: pd.DataFrame, lookback: int = 20) -> tuple:
    """Get support and resistance levels safely"""
    support = safe_rolling_operation(df, "low", lookback, "min")
    resistance = safe_rolling_operation(df, "high", lookback, "max")
    return support, resistance


# Export commonly used functions
__all__ = [
    "safe_get_latest",
    "safe_get_range",
    "safe_rolling_operation",
    "validate_dataframe_basic",
    "safe_get_close_price",
    "safe_get_volume",
    "safe_get_support_resistance",
]
