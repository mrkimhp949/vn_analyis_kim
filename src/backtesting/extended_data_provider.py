# -*- coding: utf-8 -*-
"""
Extended Backtesting Support with Long-term Data

This module provides support for backtesting with 3-5 years of historical data:
- Extended data providers for long-term data
- Data validation and cleaning
- UPCoM exchange support
- Performance optimization for large datasets

Usage:
    from src.backtesting.extended_data_provider import (
        ExtendedDataProvider,
        get_extended_data_provider,
    )
    
    provider = get_extended_data_provider()
    
    # Get 5 years of data
    data = provider.get_historical_data(
        symbol="VNM",
        years=5,
        exchange="HOSE",
    )
"""

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


class ExchangeType(Enum):
    """Vietnam stock exchanges."""

    HOSE = "HOSE"  # Ho Chi Minh Stock Exchange
    HNX = "HNX"  # Hanoi Stock Exchange
    UPCOM = "UPCOM"  # Unlisted Public Company Market
    ALL = "ALL"  # All exchanges


# Exchange characteristics
EXCHANGE_CONFIG = {
    ExchangeType.HOSE: {
        "price_limit": 0.07,  # ±7%
        "lot_size": 100,
        "trading_hours": {
            "ato_start": "09:00",
            "ato_end": "09:15",
            "morning_start": "09:15",
            "morning_end": "11:30",
            "afternoon_start": "13:00",
            "afternoon_end": "14:30",
            "atc_start": "14:30",
            "atc_end": "14:45",
        },
        "established": 2000,
        "min_price": 1000,
    },
    ExchangeType.HNX: {
        "price_limit": 0.10,  # ±10%
        "lot_size": 100,
        "trading_hours": {
            "ato_start": "09:00",
            "ato_end": "09:15",
            "morning_start": "09:15",
            "morning_end": "11:30",
            "afternoon_start": "13:00",
            "afternoon_end": "14:30",
            "atc_start": "14:30",
            "atc_end": "15:00",
        },
        "established": 2005,
        "min_price": 100,
    },
    ExchangeType.UPCOM: {
        "price_limit": 0.15,  # ±15%
        "lot_size": 100,  # Standard lot
        "odd_lot_allowed": True,  # Odd lots allowed for some transactions
        "trading_hours": {
            "start": "09:00",
            "end": "15:00",
        },
        "established": 2009,
        "min_price": 100,
    },
}


# Data providers API endpoints
DATA_PROVIDERS = {
    "vndirect": {
        "base_url": "https://finfo-api.vndirect.com.vn",
        "historical": "/v4/stock_prices",
        "max_years": 10,
    },
    "cafef": {
        "base_url": "https://s.cafef.vn",
        "historical": "/ajax/PageNew/DataHistory/PriceHistory.ashx",
        "max_years": 10,
    },
    "tcbs": {
        "base_url": "https://apipubaws.tcbs.com.vn",
        "historical": "/stock-insight/v1/stock/bars-long-term",
        "max_years": 5,
    },
    "vnstock": {
        "type": "library",
        "max_years": 20,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class DataQualityReport:
    """Report on data quality after validation."""

    symbol: str
    exchange: ExchangeType

    # Coverage
    total_days: int = 0
    trading_days: int = 0
    missing_days: int = 0
    coverage_pct: float = 0.0

    # Quality issues
    zero_volume_days: int = 0
    price_gaps: int = 0  # Days with gap > 10%
    negative_prices: int = 0
    duplicate_dates: int = 0

    # Date range
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    # Status
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "total_days": self.total_days,
            "trading_days": self.trading_days,
            "coverage_pct": self.coverage_pct,
            "issues": {
                "zero_volume": self.zero_volume_days,
                "price_gaps": self.price_gaps,
                "missing_days": self.missing_days,
            },
            "date_range": {
                "start": self.start_date.isoformat() if self.start_date else None,
                "end": self.end_date.isoformat() if self.end_date else None,
            },
            "is_valid": self.is_valid,
            "warnings": self.warnings,
        }


@dataclass
class ExtendedOHLCV:
    """Extended OHLCV data with additional fields."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    # Extended fields
    adjusted_close: Optional[float] = None
    foreign_buy_volume: Optional[int] = None
    foreign_sell_volume: Optional[int] = None
    trading_value: Optional[float] = None

    # Corporate actions
    dividend: Optional[float] = None
    split_ratio: Optional[float] = None


# =============================================================================
# DATA VALIDATOR
# =============================================================================


class DataValidator:
    """Validate and clean historical data."""

    @staticmethod
    def validate_ohlcv(
        df: pd.DataFrame,
        symbol: str,
        exchange: ExchangeType = ExchangeType.HOSE,
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Validate and clean OHLCV data.

        Args:
            df: Raw DataFrame with OHLCV data
            symbol: Stock symbol
            exchange: Exchange type

        Returns:
            (cleaned_df, quality_report)
        """
        report = DataQualityReport(symbol=symbol, exchange=exchange)

        if df is None or len(df) == 0:
            report.is_valid = False
            report.warnings.append("Empty or null DataFrame")
            return pd.DataFrame(), report

        df = df.copy()

        # Ensure date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.columns = ["date"] + list(df.columns[1:])
            df["date"] = pd.to_datetime(df["date"])
        else:
            report.is_valid = False
            report.warnings.append("No date column found")
            return df, report

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        # Check for duplicates
        duplicates = df["date"].duplicated().sum()
        if duplicates > 0:
            report.duplicate_dates = duplicates
            report.warnings.append(f"Found {duplicates} duplicate dates")
            df = df.drop_duplicates(subset=["date"], keep="last")

        # Date range
        report.start_date = df["date"].min().date()
        report.end_date = df["date"].max().date()
        report.trading_days = len(df)

        # Calculate expected trading days
        date_range = (report.end_date - report.start_date).days
        expected_trading_days = int(date_range * 252 / 365)  # Approx trading days
        report.total_days = date_range
        report.missing_days = max(0, expected_trading_days - report.trading_days)
        report.coverage_pct = (report.trading_days / max(expected_trading_days, 1)) * 100

        # Check for required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            report.is_valid = False
            report.warnings.append(f"Missing columns: {missing_cols}")
            return df, report

        # Check for negative prices
        for col in ["open", "high", "low", "close"]:
            negatives = (df[col] <= 0).sum()
            if negatives > 0:
                report.negative_prices += negatives
                df.loc[df[col] <= 0, col] = np.nan

        if report.negative_prices > 0:
            report.warnings.append(f"Found {report.negative_prices} negative/zero prices")

        # Check for zero volume
        report.zero_volume_days = (df["volume"] == 0).sum()
        if report.zero_volume_days > report.trading_days * 0.1:
            report.warnings.append(f"High zero volume days: {report.zero_volume_days}")

        # Check for price gaps
        price_change = df["close"].pct_change().abs()
        price_limit = EXCHANGE_CONFIG[exchange]["price_limit"]
        report.price_gaps = (price_change > price_limit * 1.5).sum()  # Gaps > 150% of limit

        if report.price_gaps > 0:
            report.warnings.append(f"Found {report.price_gaps} unusual price gaps")

        # OHLC consistency check
        ohlc_issues = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        ).sum()

        if ohlc_issues > 0:
            report.warnings.append(f"Found {ohlc_issues} OHLC consistency issues")
            # Fix OHLC consistency
            df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
            df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

        # Forward fill NaN values
        df = df.ffill().bfill()

        # Final validation
        if df.isnull().any().any():
            report.warnings.append("Data still contains NaN after cleaning")

        report.is_valid = len(report.warnings) < 5 and report.coverage_pct >= 80

        return df, report

    @staticmethod
    def adjust_for_corporate_actions(
        df: pd.DataFrame,
        dividends: Optional[pd.DataFrame] = None,
        splits: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Adjust prices for dividends and stock splits.

        Args:
            df: OHLCV DataFrame
            dividends: DataFrame with dividend info
            splits: DataFrame with split info

        Returns:
            Adjusted DataFrame
        """
        df = df.copy()

        if splits is not None and len(splits) > 0:
            for _, split in splits.iterrows():
                split_date = pd.to_datetime(split["date"])
                ratio = split["ratio"]  # e.g., 2 for 2:1 split

                mask = df["date"] < split_date
                for col in ["open", "high", "low", "close"]:
                    df.loc[mask, col] = df.loc[mask, col] / ratio
                df.loc[mask, "volume"] = df.loc[mask, "volume"] * ratio

        if dividends is not None and len(dividends) > 0:
            for _, div in dividends.iterrows():
                div_date = pd.to_datetime(div["date"])
                div_amount = div["amount"]

                mask = df["date"] < div_date
                for col in ["open", "high", "low", "close"]:
                    df.loc[mask, col] = df.loc[mask, col] - div_amount

        return df


# =============================================================================
# UPCOM SUPPORT
# =============================================================================


class UPCoMDataProvider:
    """Data provider specifically for UPCoM stocks."""

    # UPCoM-specific characteristics
    UPCOM_CHARACTERISTICS = {
        "price_limit": 0.15,
        "min_price_step": 100,  # VND
        "trading_method": "continuous",  # Continuous trading
        "odd_lot_board": True,
    }

    # Major UPCoM stocks (for reference)
    MAJOR_UPCOM_STOCKS = [
        "ACV",  # Airports Corporation
        "MCH",  # Masan Consumer
        "BSR",  # Binh Son Refining
        "OIL",  # PVN Oil
        "LPB",  # LienViet Post Bank
        "ABB",  # ABBank
        "DGC",  # DGC
        "YEG",  # Yeah1
        "VEA",  # VEAM
        "HVN",  # Vietnam Airlines
    ]

    def __init__(self):
        self._session = requests.Session()
        self._cache: Dict[str, pd.DataFrame] = {}

    def is_upcom_stock(self, symbol: str) -> bool:
        """Check if a symbol is listed on UPCoM."""
        symbol = symbol.upper()

        # Check cache first
        if hasattr(self, "_upcom_symbols"):
            return symbol in self._upcom_symbols

        # Try to determine from data source
        # This is a simplified check - in production, should use official listing
        return symbol in self.MAJOR_UPCOM_STOCKS

    def get_upcom_price_limits(self, reference_price: float) -> Tuple[float, float]:
        """
        Calculate price limits for UPCoM stock.

        Args:
            reference_price: Reference price (previous close)

        Returns:
            (floor_price, ceiling_price)
        """
        limit = self.UPCOM_CHARACTERISTICS["price_limit"]

        ceiling = reference_price * (1 + limit)
        floor = reference_price * (1 - limit)

        # Round to price step
        step = self.UPCOM_CHARACTERISTICS["min_price_step"]
        ceiling = int(ceiling / step) * step
        floor = int(floor / step) * step + step

        return floor, ceiling

    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """Get historical data for UPCoM stock."""
        symbol = symbol.upper()
        cache_key = f"{symbol}_{start_date}_{end_date}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try vnstock first
        try:
            from vnstock import Vnstock

            stock = Vnstock().stock(symbol=symbol, source="TCBS")
            df = stock.quote.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

            if df is not None and len(df) > 0:
                self._cache[cache_key] = df
                return df

        except Exception as e:
            logger.debug(f"vnstock failed for UPCoM {symbol}: {e}")

        # Fallback to other sources
        return None


# =============================================================================
# EXTENDED DATA PROVIDER
# =============================================================================


class ExtendedDataProvider:
    """
    Extended Data Provider for Long-term Backtesting

    Features:
    - Multiple data source integration
    - Data validation and cleaning
    - Caching for performance
    - Support for all Vietnam exchanges (HOSE, HNX, UPCoM)
    - Corporate action adjustments

    Usage:
        provider = ExtendedDataProvider()

        # Get 5 years of data
        df = provider.get_historical_data("VNM", years=5)

        # Get UPCoM stock
        df = provider.get_historical_data("ACV", exchange=ExchangeType.UPCOM)
    """

    CACHE_DIR = Path("data_cache/historical")

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_days: int = 7,
        validate_data: bool = True,
    ):
        self._cache_enabled = cache_enabled
        self._cache_days = cache_days
        self._validate_data = validate_data
        self._lock = RLock()

        # Initialize providers
        self._upcom_provider = UPCoMDataProvider()
        self._validator = DataValidator()

        # Initialize session
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            }
        )

        # Ensure cache directory exists
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("📊 Extended Data Provider initialized")

    def get_historical_data(
        self,
        symbol: str,
        years: int = 5,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        exchange: Optional[ExchangeType] = None,
        adjust_for_splits: bool = True,
        validate: bool = None,
    ) -> Tuple[Optional[pd.DataFrame], DataQualityReport]:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Stock symbol
            years: Number of years to fetch (if start_date not specified)
            start_date: Start date
            end_date: End date
            exchange: Exchange type (auto-detected if None)
            adjust_for_splits: Adjust for stock splits
            validate: Validate data (uses default if None)

        Returns:
            (DataFrame, DataQualityReport)
        """
        symbol = symbol.upper()

        if validate is None:
            validate = self._validate_data

        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=years * 365)

        # Auto-detect exchange if not specified
        if exchange is None:
            exchange = self._detect_exchange(symbol)

        # Check cache
        if self._cache_enabled:
            cached = self._load_from_cache(symbol, start_date, end_date)
            if cached is not None:
                if validate:
                    return self._validator.validate_ohlcv(cached, symbol, exchange)
                return cached, DataQualityReport(symbol=symbol, exchange=exchange)

        # Fetch from providers
        df = None

        # Try vnstock first (supports all exchanges)
        df = self._fetch_from_vnstock(symbol, start_date, end_date)

        # Fallback to VNDirect
        if df is None:
            df = self._fetch_from_vndirect(symbol, start_date, end_date)

        # Fallback to TCBS
        if df is None:
            df = self._fetch_from_tcbs(symbol, start_date, end_date)

        # Special handling for UPCoM
        if df is None and exchange == ExchangeType.UPCOM:
            df = self._upcom_provider.get_historical_data(symbol, start_date, end_date)

        if df is None:
            logger.warning(f"Could not fetch data for {symbol}")
            return None, DataQualityReport(
                symbol=symbol,
                exchange=exchange,
                is_valid=False,
                warnings=["No data available from any source"],
            )

        # Cache the data
        if self._cache_enabled:
            self._save_to_cache(symbol, df)

        # Validate and return
        if validate:
            return self._validator.validate_ohlcv(df, symbol, exchange)

        return df, DataQualityReport(symbol=symbol, exchange=exchange)

    def get_batch_historical_data(
        self, symbols: List[str], years: int = 5, **kwargs
    ) -> Dict[str, Tuple[Optional[pd.DataFrame], DataQualityReport]]:
        """
        Get historical data for multiple symbols.

        Args:
            symbols: List of stock symbols
            years: Number of years to fetch
            **kwargs: Additional arguments for get_historical_data

        Returns:
            Dict mapping symbol to (DataFrame, Report)
        """
        results = {}

        for symbol in symbols:
            try:
                df, report = self.get_historical_data(symbol, years=years, **kwargs)
                results[symbol] = (df, report)
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
                results[symbol] = (
                    None,
                    DataQualityReport(
                        symbol=symbol,
                        exchange=ExchangeType.HOSE,
                        is_valid=False,
                        warnings=[str(e)],
                    ),
                )

            # Rate limiting
            time.sleep(0.5)

        return results

    def get_index_data(
        self,
        index: str = "VNINDEX",
        years: int = 5,
    ) -> Optional[pd.DataFrame]:
        """Get historical data for an index."""
        return self.get_historical_data(index, years=years)[0]

    def get_available_symbols(
        self,
        exchange: ExchangeType = ExchangeType.ALL,
    ) -> List[str]:
        """Get list of available symbols."""
        try:
            from vnstock import Vnstock

            stock = Vnstock()
            listing = stock.stock().listing.all_symbols()

            if exchange == ExchangeType.ALL:
                return listing["ticker"].tolist()
            else:
                return listing[listing["exchange"] == exchange.value]["ticker"].tolist()

        except Exception as e:
            logger.warning(f"Could not get symbol list: {e}")
            return []

    def _detect_exchange(self, symbol: str) -> ExchangeType:
        """Auto-detect exchange for a symbol."""
        symbol = symbol.upper()

        # Check UPCoM
        if self._upcom_provider.is_upcom_stock(symbol):
            return ExchangeType.UPCOM

        # Known HNX stocks
        hnx_stocks = {"SHB", "NVB", "PVS", "ACB", "SHS", "VCS", "CEO", "NVL", "IDC"}
        if symbol in hnx_stocks:
            return ExchangeType.HNX

        # Default to HOSE
        return ExchangeType.HOSE

    def _fetch_from_vnstock(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from vnstock library."""
        try:
            from vnstock import Vnstock

            stock = Vnstock().stock(symbol=symbol, source="TCBS")
            df = stock.quote.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

            if df is not None and len(df) > 0:
                # Standardize column names
                df = df.rename(
                    columns={
                        "time": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
                logger.info(f"✅ Fetched {len(df)} days for {symbol} from vnstock")
                return df

        except Exception as e:
            logger.debug(f"vnstock failed for {symbol}: {e}")

        return None

    def _fetch_from_vndirect(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from VNDirect API."""
        try:
            base_url = DATA_PROVIDERS["vndirect"]["base_url"]

            response = self._session.get(
                f"{base_url}/v4/stock_prices",
                params={
                    "q": f"code:{symbol}~date:gte:{start_date}~date:lte:{end_date}",
                    "size": 5000,
                    "sort": "date",
                },
                timeout=30,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if not data.get("data"):
                return None

            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"])

            # Rename columns
            column_map = {
                "basicPrice": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "nmVolume": "volume",
            }
            df = df.rename(columns=column_map)

            logger.info(f"✅ Fetched {len(df)} days for {symbol} from VNDirect")
            return df

        except Exception as e:
            logger.debug(f"VNDirect failed for {symbol}: {e}")

        return None

    def _fetch_from_tcbs(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from TCBS API."""
        try:
            base_url = DATA_PROVIDERS["tcbs"]["base_url"]

            response = self._session.get(
                f"{base_url}/stock-insight/v1/stock/bars-long-term",
                params={
                    "ticker": symbol,
                    "type": "stock",
                    "resolution": "D",
                    "from": int(datetime.combine(start_date, datetime.min.time()).timestamp()),
                    "to": int(datetime.combine(end_date, datetime.max.time()).timestamp()),
                },
                timeout=30,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if not data or "data" not in data:
                return None

            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["tradingDate"])

            column_map = {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
            df = df.rename(columns=column_map)

            logger.info(f"✅ Fetched {len(df)} days for {symbol} from TCBS")
            return df

        except Exception as e:
            logger.debug(f"TCBS failed for {symbol}: {e}")

        return None

    def _load_from_cache(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        cache_file = self.CACHE_DIR / f"{symbol}.parquet"

        if not cache_file.exists():
            return None

        try:
            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.days > self._cache_days:
                return None

            df = pd.read_parquet(cache_file)

            # Filter to requested date range
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

            if len(df) > 0:
                logger.debug(f"Loaded {len(df)} days for {symbol} from cache")
                return df

        except Exception as e:
            logger.debug(f"Cache load failed for {symbol}: {e}")

        return None

    def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        """Save data to cache."""
        try:
            cache_file = self.CACHE_DIR / f"{symbol}.parquet"
            df.to_parquet(cache_file, index=False)
            logger.debug(f"Cached {len(df)} days for {symbol}")
        except Exception as e:
            logger.debug(f"Cache save failed for {symbol}: {e}")

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache for a symbol or all symbols."""
        if symbol:
            cache_file = self.CACHE_DIR / f"{symbol}.parquet"
            if cache_file.exists():
                cache_file.unlink()
        else:
            for cache_file in self.CACHE_DIR.glob("*.parquet"):
                cache_file.unlink()

    def get_data_quality_summary(
        self,
        symbols: List[str],
        years: int = 5,
    ) -> pd.DataFrame:
        """Get data quality summary for multiple symbols."""
        reports = []

        for symbol in symbols:
            _, report = self.get_historical_data(symbol, years=years)
            reports.append(
                {
                    "symbol": report.symbol,
                    "exchange": report.exchange.value,
                    "trading_days": report.trading_days,
                    "coverage_pct": report.coverage_pct,
                    "missing_days": report.missing_days,
                    "zero_volume": report.zero_volume_days,
                    "price_gaps": report.price_gaps,
                    "is_valid": report.is_valid,
                    "warnings_count": len(report.warnings),
                }
            )

        return pd.DataFrame(reports)


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_provider_instance: Optional[ExtendedDataProvider] = None
_provider_lock = RLock()


def get_extended_data_provider() -> ExtendedDataProvider:
    """Get singleton instance of the data provider."""
    global _provider_instance
    with _provider_lock:
        if _provider_instance is None:
            _provider_instance = ExtendedDataProvider()
        return _provider_instance


def reset_data_provider():
    """Reset the singleton instance (for testing)."""
    global _provider_instance
    with _provider_lock:
        _provider_instance = None


def get_long_term_data(symbol: str, years: int = 5, **kwargs) -> Optional[pd.DataFrame]:
    """Convenience function to get long-term historical data."""
    df, _ = get_extended_data_provider().get_historical_data(symbol, years=years, **kwargs)
    return df


def get_upcom_data(
    symbol: str,
    years: int = 3,
) -> Optional[pd.DataFrame]:
    """Convenience function to get UPCoM stock data."""
    df, _ = get_extended_data_provider().get_historical_data(
        symbol, years=years, exchange=ExchangeType.UPCOM
    )
    return df


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING EXTENDED DATA PROVIDER")
    print("=" * 60)

    provider = get_extended_data_provider()

    # Test symbols
    test_symbols = ["VNM", "VCB", "HPG"]

    print("\n📊 Fetching 3-year historical data:")
    print("-" * 60)

    for symbol in test_symbols:
        df, report = provider.get_historical_data(symbol, years=3)

        if df is not None:
            print(f"\n{symbol}:")
            print(f"  Trading days: {report.trading_days}")
            print(f"  Coverage: {report.coverage_pct:.1f}%")
            print(f"  Date range: {report.start_date} to {report.end_date}")
            print(f"  Valid: {report.is_valid}")
            if report.warnings:
                print(f"  Warnings: {report.warnings[:3]}")
        else:
            print(f"\n{symbol}: No data available")

    # Test UPCoM
    print("\n📊 Testing UPCoM stock:")
    print("-" * 60)

    upcom_symbol = "ACV"
    df, report = provider.get_historical_data(upcom_symbol, years=2, exchange=ExchangeType.UPCOM)

    if df is not None:
        print(f"\n{upcom_symbol} (UPCoM):")
        print(f"  Trading days: {report.trading_days}")
        print(f"  Coverage: {report.coverage_pct:.1f}%")
    else:
        print(f"\n{upcom_symbol}: No data available")

    print("\n" + "=" * 60)
