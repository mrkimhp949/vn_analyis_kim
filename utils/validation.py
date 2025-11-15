"""
Data Validation Utilities
Comprehensive validation for trading data
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from exceptions import DataQualityError
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for trading data quality"""
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    OPTIONAL_COLUMNS = ['atr', 'rsi', 'macd', 'ema20', 'ema50']
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        min_rows: int = 50,
        check_columns: Optional[List[str]] = None
    ) -> None:
        """
        Validate DataFrame quality
        
        Args:
            df: DataFrame to validate
            min_rows: Minimum required rows
            check_columns: Columns to check (default: REQUIRED_COLUMNS)
            
        Raises:
            DataQualityError: If validation fails
        """
        if check_columns is None:
            check_columns = DataValidator.REQUIRED_COLUMNS
        
        # Check 1: Empty DataFrame
        if df.empty:
            raise DataQualityError("DataFrame is empty")
        
        # Check 2: Minimum rows
        if len(df) < min_rows:
            raise DataQualityError(
                f"Insufficient data: {len(df)} rows < {min_rows} required",
                context={'rows': len(df), 'required': min_rows}
            )
        
        # Check 3: Required columns
        missing_cols = [col for col in check_columns if col not in df.columns]
        if missing_cols:
            raise DataQualityError(
                f"Missing required columns: {missing_cols}",
                context={'missing': missing_cols, 'available': list(df.columns)}
            )
        
        # Check 4: NaN in latest row
        latest = df.iloc[-1]
        nan_cols = [col for col in check_columns if pd.isna(latest[col])]
        if nan_cols:
            raise DataQualityError(
                f"NaN values in latest row: {nan_cols}",
                context={'nan_columns': nan_cols}
            )
        
        # Check 5: Invalid values
        DataValidator._validate_values(df, check_columns)
        
        logger.debug(f"✅ DataFrame validation passed: {len(df)} rows")
    
    @staticmethod
    def _validate_values(df: pd.DataFrame, columns: List[str]) -> None:
        """Validate value ranges"""
        latest = df.iloc[-1]
        
        # Price columns must be positive
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in columns and latest[col] <= 0:
                raise DataQualityError(
                    f"Invalid {col}: {latest[col]} <= 0",
                    context={col: latest[col]}
                )
        
        # Volume must be positive
        if 'volume' in columns and latest['volume'] <= 0:
            raise DataQualityError(
                f"Invalid volume: {latest['volume']} <= 0",
                context={'volume': latest['volume']}
            )
        
        # High >= Low
        if 'high' in columns and 'low' in columns:
            if latest['high'] < latest['low']:
                raise DataQualityError(
                    f"High {latest['high']} < Low {latest['low']}",
                    context={'high': latest['high'], 'low': latest['low']}
                )
        
        # RSI range
        if 'rsi' in df.columns and pd.notna(latest.get('rsi')):
            rsi = latest['rsi']
            if not (0 <= rsi <= 100):
                logger.warning(f"RSI out of range: {rsi}")
    
    @staticmethod
    def validate_price(price: float, name: str = "price") -> float:
        """
        Validate a single price value
        
        Args:
            price: Price to validate
            name: Name for error message
            
        Returns:
            Validated price
            
        Raises:
            ValueError: If price is invalid
        """
        if pd.isna(price):
            raise ValueError(f"{name} is NaN")
        
        if not isinstance(price, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(price)}")
        
        if price <= 0:
            raise ValueError(f"{name} must be positive, got {price}")
        
        return float(price)
    
    @staticmethod
    def validate_shares(shares: int) -> int:
        """Validate share count"""
        if not isinstance(shares, int):
            raise ValueError(f"Shares must be integer, got {type(shares)}")
        
        if shares <= 0:
            raise ValueError(f"Shares must be positive, got {shares}")
        
        if shares % 100 != 0:
            logger.warning(f"Shares {shares} not in lots of 100")
        
        return shares
    
    @staticmethod
    def validate_percentage(value: float, name: str = "percentage") -> float:
        """Validate percentage value (0-1)"""
        if pd.isna(value):
            raise ValueError(f"{name} is NaN")
        
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        
        if not (0 <= value <= 1):
            raise ValueError(f"{name} must be 0-1, got {value}")
        
        return float(value)
    
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize DataFrame by handling common issues
        
        Args:
            df: DataFrame to sanitize
            
        Returns:
            Sanitized DataFrame
        """
        df = df.copy()
        
        # Remove duplicate index
        df = df[~df.index.duplicated(keep='last')]
        
        # Forward fill NaN (limited)
        df = df.fillna(method='ffill', limit=3)
        
        # Drop rows with remaining NaN in critical columns
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        available_critical = [col for col in critical_cols if col in df.columns]
        df = df.dropna(subset=available_critical)
        
        # Ensure positive values
        for col in available_critical:
            df[col] = df[col].abs()
            df.loc[df[col] == 0, col] = df[col].replace(0, np.nan).fillna(method='ffill')
        
        return df


class InputValidator:
    """Validator for function inputs"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate stock symbol"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        symbol = symbol.strip().upper()
        
        if len(symbol) < 2 or len(symbol) > 10:
            raise ValueError(f"Symbol length invalid: {symbol}")
        
        return symbol
    
    @staticmethod
    def validate_confidence(confidence: int) -> int:
        """Validate confidence score"""
        if not isinstance(confidence, (int, float)):
            raise ValueError(f"Confidence must be numeric, got {type(confidence)}")
        
        confidence = int(confidence)
        
        if not (0 <= confidence <= 100):
            raise ValueError(f"Confidence must be 0-100, got {confidence}")
        
        return confidence
    
    @staticmethod
    def validate_risk_reward(rr: float) -> float:
        """Validate risk/reward ratio"""
        if not isinstance(rr, (int, float)):
            raise ValueError(f"R:R must be numeric, got {type(rr)}")
        
        if rr < 0:
            raise ValueError(f"R:R must be positive, got {rr}")
        
        return float(rr)


# Singleton instances
data_validator = DataValidator()
input_validator = InputValidator()


if __name__ == "__main__":
    print("Testing DataValidator...")
    
    # Test 1: Valid DataFrame
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [99, 100, 101],
        'close': [103, 104, 105],
        'volume': [1000, 1100, 1200]
    })
    
    try:
        data_validator.validate_dataframe(df, min_rows=3)
        print("✅ Test 1 passed: Valid DataFrame")
    except DataQualityError as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Empty DataFrame
    df_empty = pd.DataFrame()
    try:
        data_validator.validate_dataframe(df_empty)
        print("❌ Test 2 failed: Should raise error for empty DataFrame")
    except DataQualityError:
        print("✅ Test 2 passed: Empty DataFrame rejected")
    
    # Test 3: Invalid price
    try:
        data_validator.validate_price(-100, "test_price")
        print("❌ Test 3 failed: Should raise error for negative price")
    except ValueError:
        print("✅ Test 3 passed: Negative price rejected")
    
    # Test 4: Valid confidence
    try:
        conf = input_validator.validate_confidence(75)
        assert conf == 75
        print("✅ Test 4 passed: Valid confidence")
    except ValueError as e:
        print(f"❌ Test 4 failed: {e}")
    
    print("\n✅ All validation tests completed!")
