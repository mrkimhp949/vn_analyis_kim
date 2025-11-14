"""
Data Quality Checks
Validate market data before using for trading decisions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from exceptions import DataQualityError, DataLoadError


class DataQualityChecker:
    """
    Check data quality for trading decisions

    Features:
    - Outlier detection
    - Missing data detection
    - Data freshness validation
    - Volume anomalies
    - Price gaps
    """

    def __init__(
        self,
        max_price_change: float = 0.15,  # 15% max daily change
        min_data_points: int = 50,
        max_data_age_hours: int = 24,
        min_volume_ratio: float = 0.1,
    ):  # 10% of average
        self.max_price_change = max_price_change
        self.min_data_points = min_data_points
        self.max_data_age_hours = max_data_age_hours
        self.min_volume_ratio = min_volume_ratio

    def validate(self, df: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """
        Validate data quality

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for context

        Returns:
            Dict with validation results:
            {
                'valid': bool,
                'checks': {
                    'has_sufficient_data': bool,
                    'has_outliers': bool,
                    'has_missing': bool,
                    'has_gaps': bool,
                    'volume_valid': bool,
                    'price_valid': bool,
                    'data_fresh': bool
                },
                'issues': List[str],
                'warnings': List[str]
            }

        Raises:
            DataQualityError: If critical issues found
        """
        if df.empty:
            raise DataQualityError(
                f"Empty dataframe for {symbol}",
                context={"symbol": symbol, "check": "empty"},
            )

        results = {"valid": True, "checks": {}, "issues": [], "warnings": []}

        # Check 1: Sufficient data
        has_sufficient = len(df) >= self.min_data_points
        results["checks"]["has_sufficient_data"] = has_sufficient
        if not has_sufficient:
            results["issues"].append(
                f"Insufficient data: {len(df)} points, need {self.min_data_points}"
            )
            results["valid"] = False

        # Check 2: Missing values
        has_missing = df.isnull().any().any()
        results["checks"]["has_missing"] = has_missing
        if has_missing:
            missing_cols = df.columns[df.isnull().any()].tolist()
            results["warnings"].append(f"Missing values in: {', '.join(missing_cols)}")

        # Check 3: Outliers
        outlier_check = self._check_outliers(df)
        results["checks"]["has_outliers"] = outlier_check["has_outliers"]
        if outlier_check["has_outliers"]:
            results["warnings"].extend(outlier_check["warnings"])

        # Check 4: Price gaps
        gap_check = self._check_gaps(df)
        results["checks"]["has_gaps"] = gap_check["has_gaps"]
        if gap_check["has_gaps"]:
            results["warnings"].extend(gap_check["warnings"])

        # Check 5: Volume validation
        volume_check = self._check_volume(df)
        results["checks"]["volume_valid"] = volume_check["valid"]
        if not volume_check["valid"]:
            results["warnings"].extend(volume_check["warnings"])

        # Check 6: Price validation
        price_check = self._check_prices(df)
        results["checks"]["price_valid"] = price_check["valid"]
        if not price_check["valid"]:
            results["issues"].extend(price_check["issues"])
            results["valid"] = False

        # Check 7: Data freshness (if time column exists)
        if "time" in df.columns:
            freshness_check = self._check_freshness(df)
            results["checks"]["data_fresh"] = freshness_check["fresh"]
            if not freshness_check["fresh"]:
                results["warnings"].append(freshness_check["warning"])

        return results

    def _check_outliers(self, df: pd.DataFrame) -> Dict:
        """Check for price outliers"""
        if len(df) < 2:
            return {"has_outliers": False, "warnings": []}

        warnings = []
        has_outliers = False

        # Check daily price changes
        if "close" in df.columns:
            price_changes = df["close"].pct_change().abs()

            # Flag extreme changes
            extreme_changes = price_changes > self.max_price_change
            if extreme_changes.any():
                has_outliers = True
                num_outliers = extreme_changes.sum()
                max_change = price_changes.max() * 100
                warnings.append(
                    f"Price outliers: {num_outliers} days with >{self.max_price_change*100:.0f}% change "
                    f"(max: {max_change:.1f}%)"
                )

        return {"has_outliers": has_outliers, "warnings": warnings}

    def _check_gaps(self, df: pd.DataFrame) -> Dict:
        """Check for data gaps (missing days)"""
        if "time" not in df.columns or len(df) < 2:
            return {"has_gaps": False, "warnings": []}

        warnings = []
        has_gaps = False

        try:
            # Convert time to datetime if needed
            times = pd.to_datetime(df["time"])
            times = times.sort_values()

            # Check for gaps > 1 day
            time_diffs = times.diff()
            large_gaps = time_diffs > timedelta(days=2)

            if large_gaps.any():
                has_gaps = True
                num_gaps = large_gaps.sum()
                max_gap = time_diffs.max().days
                warnings.append(
                    f"Data gaps: {num_gaps} gaps > 2 days (max: {max_gap} days)"
                )
        except Exception:
            # Can't check gaps if time format is invalid
            pass

        return {"has_gaps": has_gaps, "warnings": warnings}

    def _check_volume(self, df: pd.DataFrame) -> Dict:
        """Check volume data"""
        if "volume" not in df.columns:
            return {"valid": True, "warnings": []}

        warnings = []
        valid = True

        # Check for zero or negative volume
        invalid_volume = (df["volume"] <= 0).any()
        if invalid_volume:
            num_invalid = (df["volume"] <= 0).sum()
            warnings.append(f"Invalid volume: {num_invalid} rows with volume <= 0")
            valid = False

        # Check for unusually low volume
        if len(df) >= 20:
            avg_volume = df["volume"].rolling(20).mean().iloc[-1]
            current_volume = df["volume"].iloc[-1]

            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                if volume_ratio < self.min_volume_ratio:
                    warnings.append(
                        f"Low volume: current ({current_volume:,.0f}) is "
                        f"{volume_ratio*100:.1f}% of 20-day average ({avg_volume:,.0f})"
                    )

        return {"valid": valid, "warnings": warnings}

    def _check_prices(self, df: pd.DataFrame) -> Dict:
        """Check price data validity"""
        price_cols = ["open", "high", "low", "close"]
        issues = []
        valid = True

        for col in price_cols:
            if col not in df.columns:
                continue

            # Check for zero or negative prices
            if (df[col] <= 0).any():
                num_invalid = (df[col] <= 0).sum()
                issues.append(f"{col}: {num_invalid} rows with price <= 0")
                valid = False

        # Check OHLC relationships
        if all(col in df.columns for col in price_cols):
            # High should be >= all others
            invalid_high = (df["high"] < df[["open", "low", "close"]].max(axis=1)).any()
            if invalid_high:
                issues.append("high price < max(open, low, close) in some rows")
                valid = False

            # Low should be <= all others
            invalid_low = (df["low"] > df[["open", "high", "close"]].min(axis=1)).any()
            if invalid_low:
                issues.append("low price > min(open, high, close) in some rows")
                valid = False

        return {"valid": valid, "issues": issues}

    def _check_freshness(self, df: pd.DataFrame) -> Dict:
        """Check if data is fresh enough"""
        if "time" not in df.columns or len(df) == 0:
            return {"fresh": True, "warning": ""}

        try:
            latest_time = pd.to_datetime(df["time"].iloc[-1])
            age_hours = (datetime.now() - latest_time).total_seconds() / 3600

            if age_hours > self.max_data_age_hours:
                return {
                    "fresh": False,
                    "warning": f"Data is {age_hours:.1f} hours old (max: {self.max_data_age_hours}h)",
                }
        except Exception:
            pass

        return {"fresh": True, "warning": ""}

    def clean_data(
        self, df: pd.DataFrame, method: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        Clean data by handling missing values

        Args:
            df: DataFrame to clean
            method: 'forward_fill', 'backward_fill', or 'drop'

        Returns:
            Cleaned DataFrame
        """
        cleaned = df.copy()

        if method == "forward_fill":
            cleaned = cleaned.fillna(method="ffill")
        elif method == "backward_fill":
            cleaned = cleaned.fillna(method="bfill")
        elif method == "drop":
            cleaned = cleaned.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

        return cleaned


# Singleton instance
_quality_checker = None


def get_quality_checker() -> DataQualityChecker:
    """Get data quality checker singleton"""
    global _quality_checker
    if _quality_checker is None:
        _quality_checker = DataQualityChecker()
    return _quality_checker
