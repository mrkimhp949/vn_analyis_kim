"""
Type Hints Examples for Trading Bot
Demonstrates proper typing for all major components
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# TYPE ALIASES
# ============================================================================

Symbol = str
Price = float
Shares = int
Confidence = float  # 0-100
Timestamp = str  # ISO format


# ============================================================================
# DATA CLASSES WITH TYPE HINTS
# ============================================================================


@dataclass
class Position:
    """Trading position with full type annotations"""

    symbol: Symbol
    shares: Shares
    avg_price: Price
    entry_date: Timestamp
    entry_value: float
    stop_loss: Optional[Price] = None
    take_profit: Optional[Price] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TradingSignal:
    """Trading signal with type annotations"""

    symbol: Symbol
    signal_type: str  # 'BUY' | 'SELL' | 'HOLD'
    confidence: Confidence
    entry_price: Price
    stop_loss: Price
    take_profit_targets: List[Price]
    reasons: List[str]
    timestamp: Timestamp


# ============================================================================
# FUNCTION TYPE HINTS EXAMPLES
# ============================================================================


def load_data(
    symbol: Symbol, lookback: int = 200, use_cache: bool = True
) -> pd.DataFrame:
    """
    Load market data with type hints

    Args:
        symbol: Stock symbol (e.g., 'VCB', 'HPG')
        lookback: Number of days to look back
        use_cache: Whether to use cached data

    Returns:
        DataFrame with OHLCV data

    Raises:
        DataLoadError: If data cannot be loaded
    """
    ...


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with additional indicator columns
    """
    ...


def analyze_entry(
    df: pd.DataFrame,
    ml_signal: Dict[str, Any],
    market_regime: Optional[Dict[str, Any]] = None,
) -> TradingSignal:
    """
    Analyze entry opportunity

    Args:
        df: Market data with indicators
        ml_signal: ML model prediction
        market_regime: Current market regime (optional)

    Returns:
        Trading signal with entry recommendation
    """
    ...


def calculate_position_size(
    symbol: Symbol,
    entry_price: Price,
    stop_loss: Price,
    portfolio_value: float,
    max_risk_per_trade: float = 0.02,
) -> Tuple[Shares, float]:
    """
    Calculate position size

    Args:
        symbol: Stock symbol
        entry_price: Entry price
        stop_loss: Stop loss price
        portfolio_value: Total portfolio value
        max_risk_per_trade: Maximum risk per trade (default 2%)

    Returns:
        Tuple of (shares, position_value)
    """
    ...


def validate_signal(
    signal: TradingSignal, portfolio_positions: Dict[Symbol, Position]
) -> Tuple[bool, str]:
    """
    Validate trading signal

    Args:
        signal: Trading signal to validate
        portfolio_positions: Current portfolio positions

    Returns:
        Tuple of (is_valid, reason)
    """
    ...


# ============================================================================
# ML MODEL TYPE HINTS
# ============================================================================


def train_model(
    X_train: Union[pd.DataFrame, NDArray[np.float64]],
    y_train: Union[pd.Series, NDArray[np.int64]],
) -> None:
    """
    Train ML model

    Args:
        X_train: Training features
        y_train: Training labels
    """
    ...


def predict(X: Union[pd.DataFrame, NDArray[np.float64]]) -> NDArray[np.float64]:
    """
    Make predictions

    Args:
        X: Features for prediction

    Returns:
        Array of predictions (probabilities 0-1)
    """
    ...


# ============================================================================
# PORTFOLIO MANAGEMENT TYPE HINTS
# ============================================================================


def add_position(
    symbol: Symbol,
    shares: Shares,
    entry_price: Price,
    stop_loss: Optional[Price] = None,
    take_profit: Optional[Price] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Add position to portfolio

    Args:
        symbol: Stock symbol
        shares: Number of shares
        entry_price: Entry price
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        metadata: Additional metadata (optional)

    Raises:
        PortfolioError: If validation fails
    """
    ...


def get_portfolio_value() -> Dict[str, Union[float, int]]:
    """
    Get current portfolio value

    Returns:
        Dictionary with portfolio metrics:
        - total_value: Current total value
        - total_cost: Total cost basis
        - pnl: Profit/loss
        - pnl_percent: P&L percentage
        - num_positions: Number of positions
    """
    ...


def get_positions() -> Dict[Symbol, Position]:
    """
    Get all active positions

    Returns:
        Dictionary mapping symbols to Position objects
    """
    ...


# ============================================================================
# RISK MANAGEMENT TYPE HINTS
# ============================================================================


def check_risk_limits(
    positions: Dict[Symbol, Position], new_position: Position
) -> Tuple[bool, List[str]]:
    """
    Check if new position violates risk limits

    Args:
        positions: Current positions
        new_position: Proposed new position

    Returns:
        Tuple of (is_valid, violation_messages)
    """
    ...


def calculate_portfolio_risk(positions: Dict[Symbol, Position]) -> Dict[str, float]:
    """
    Calculate portfolio risk metrics

    Args:
        positions: Current portfolio positions

    Returns:
        Dictionary with risk metrics:
        - total_risk: Total portfolio risk
        - sector_exposure: Max sector exposure
        - correlation_risk: Correlation risk score
    """
    ...


# ============================================================================
# CONFIGURATION TYPE HINTS
# ============================================================================


@dataclass
class TradingConfig:
    """Trading configuration with type hints"""

    min_confidence: Confidence
    min_risk_reward: float
    support_distance_percent: float
    stop_loss_percent: float
    take_profit_percent: float
    trailing_stop_percent: float
    trailing_activation_percent: float
    max_position_size: float
    min_position_size: float
    max_positions: int
    max_portfolio_risk: float
    max_sector_exposure: float
    max_loss_per_day_pct: float

    def validate(self) -> None:
        """Validate configuration"""
        ...


# ============================================================================
# DATABASE TYPE HINTS
# ============================================================================


def save_position(
    symbol: Symbol,
    shares: Shares,
    avg_price: Price,
    entry_date: Timestamp,
    entry_value: float,
    stop_loss: Optional[Price] = None,
    take_profit: Optional[Price] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save position to database"""
    ...


def get_trades(
    symbol: Optional[Symbol] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Get trade history

    Args:
        symbol: Filter by symbol (optional)
        start_date: Start date (optional)
        end_date: End date (optional)
        limit: Maximum number of trades

    Returns:
        List of trade dictionaries
    """
    ...


# ============================================================================
# ASYNC TYPE HINTS
# ============================================================================


async def send_telegram_message(
    chat_id: str, message: str, parse_mode: Optional[str] = None
) -> bool:
    """
    Send Telegram message

    Args:
        chat_id: Telegram chat ID
        message: Message text
        parse_mode: Parse mode (e.g., 'Markdown')

    Returns:
        True if message sent successfully
    """
    ...


async def fetch_market_data(
    symbols: List[Symbol], timeout: int = 10
) -> Dict[Symbol, pd.DataFrame]:
    """
    Fetch market data for multiple symbols

    Args:
        symbols: List of stock symbols
        timeout: Request timeout in seconds

    Returns:
        Dictionary mapping symbols to DataFrames
    """
    ...


if __name__ == "__main__":
    print("✅ Type hints examples defined successfully")
    print("Use these patterns throughout the codebase for proper typing")
