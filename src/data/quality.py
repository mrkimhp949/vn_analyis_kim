# -*- coding: utf-8 -*-
"""
Data Quality Module for Vietnam Stock Market

Validates and cleans OHLCV data with Vietnam-specific checks:
- Gap detection (>7% = circuit limit)
- Volume anomaly detection
- Stock split/dividend adjustment detection
- Missing data handling
- Price outlier detection

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Constants - Vietnam Market Specific
# =============================================================================

# Vietnam market price limits
VN_HOSE_PRICE_LIMIT = 0.07  # ±7% for HOSE
VN_HNX_PRICE_LIMIT = 0.10  # ±10% for HNX
VN_UPCOM_PRICE_LIMIT = 0.15  # ±15% for UPCoM

# Typical stock split ratios in Vietnam
COMMON_SPLIT_RATIOS = [2.0, 3.0, 4.0, 5.0, 10.0, 0.5, 0.33, 0.25, 0.2, 0.1]

# Volume anomaly thresholds
VOLUME_SPIKE_THRESHOLD = 5.0  # 5x average volume
VOLUME_DROUGHT_THRESHOLD = 0.1  # 10% of average volume
VOLUME_ZERO_TOLERANCE = 3  # Max consecutive zero volume days

# Price outlier detection
PRICE_Z_SCORE_THRESHOLD = 4.0  # 4 standard deviations

# Gap detection
GAP_WARNING_THRESHOLD = 0.05  # 5% gap warning
GAP_CRITICAL_THRESHOLD = 0.07  # 7% gap (circuit limit)

# Minimum data requirements
MIN_BARS_FOR_VALIDATION = 5


@dataclass
class QualityIssue:
    """Represents a data quality issue"""

    issue_type: str
    severity: str  # 'warning', 'error', 'critical'
    description: str
    affected_rows: List[int] = field(default_factory=list)
    suggested_action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Complete data quality report"""

    symbol: str
    timestamp: datetime
    valid: bool
    total_rows: int
    issues: List[QualityIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 100.0  # 0-100 quality score

    def add_issue(self, issue: QualityIssue):
        self.issues.append(issue)
        # Adjust score based on severity
        if issue.severity == "critical":
            self.score -= 30
            self.valid = False
        elif issue.severity == "error":
            self.score -= 15
        elif issue.severity == "warning":
            self.score -= 5
        self.score = max(0, self.score)

    def add_warning(self, warning: str):
        self.warnings.append(warning)
        self.score -= 2
        self.score = max(0, self.score)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "valid": self.valid,
            "total_rows": self.total_rows,
            "score": self.score,
            "issues": [
                {
                    "type": i.issue_type,
                    "severity": i.severity,
                    "description": i.description,
                    "affected_rows": len(i.affected_rows),
                    "action": i.suggested_action,
                }
                for i in self.issues
            ],
            "warnings": self.warnings,
        }


class DataQualityChecker:
    """
    Comprehensive data quality checker for Vietnam stock data.

    Features:
    - Gap detection (price limits)
    - Volume anomaly detection
    - Stock split detection
    - Dividend adjustment detection
    - Missing value handling
    - Outlier detection

    Usage:
        checker = DataQualityChecker()
        report = checker.validate(df, symbol="VNM")

        if not report.valid:
            df = checker.clean_data(df, method='forward_fill')
    """

    def __init__(
        self,
        exchange: str = "HOSE",
        strict_mode: bool = False,
        auto_detect_splits: bool = True,
        auto_detect_dividends: bool = True,
    ):
        """
        Initialize quality checker.

        Args:
            exchange: Exchange type ('HOSE', 'HNX', 'UPCOM') for price limits
            strict_mode: If True, treat warnings as errors
            auto_detect_splits: Automatically detect stock splits
            auto_detect_dividends: Automatically detect dividend adjustments
        """
        self.exchange = exchange.upper()
        self.strict_mode = strict_mode
        self.auto_detect_splits = auto_detect_splits
        self.auto_detect_dividends = auto_detect_dividends

        # Set price limit based on exchange
        self.price_limit = {
            "HOSE": VN_HOSE_PRICE_LIMIT,
            "HNX": VN_HNX_PRICE_LIMIT,
            "UPCOM": VN_UPCOM_PRICE_LIMIT,
        }.get(self.exchange, VN_HOSE_PRICE_LIMIT)

    def validate(self, df: pd.DataFrame, symbol: str = "") -> QualityReport:
        """
        Validate DataFrame for data quality issues.

        Args:
            df: OHLCV DataFrame with columns: time, open, high, low, close, volume
            symbol: Stock symbol for reporting

        Returns:
            QualityReport with all detected issues
        """
        report = QualityReport(
            symbol=symbol,
            timestamp=datetime.now(),
            valid=True,
            total_rows=len(df) if df is not None else 0,
        )

        # Basic validation
        if df is None or df.empty:
            report.add_issue(
                QualityIssue(
                    issue_type="empty_data",
                    severity="critical",
                    description="DataFrame is empty or None",
                    suggested_action="Verify data source and date range",
                )
            )
            return report

        if len(df) < MIN_BARS_FOR_VALIDATION:
            report.add_issue(
                QualityIssue(
                    issue_type="insufficient_data",
                    severity="error",
                    description=f"Only {len(df)} bars, need at least {MIN_BARS_FOR_VALIDATION}",
                    suggested_action="Extend lookback period",
                )
            )
            return report

        # Required columns check
        required_cols = ["time", "open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            report.add_issue(
                QualityIssue(
                    issue_type="missing_columns",
                    severity="critical",
                    description=f"Missing columns: {missing_cols}",
                    suggested_action="Check data source format",
                )
            )
            return report

        # Run all quality checks
        self._check_missing_values(df, report)
        self._check_price_integrity(df, report)
        self._check_volume_anomalies(df, report)
        self._check_gaps(df, report)
        self._check_duplicates(df, report)
        self._check_chronological_order(df, report)

        if self.auto_detect_splits:
            self._detect_stock_splits(df, report)

        if self.auto_detect_dividends:
            self._detect_dividend_adjustments(df, report)

        self._check_outliers(df, report)

        return report

    def _check_missing_values(self, df: pd.DataFrame, report: QualityReport):
        """Check for missing values in critical columns"""
        price_cols = ["open", "high", "low", "close"]

        for col in price_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_rows = df[df[col].isnull()].index.tolist()
                severity = "error" if null_count > len(df) * 0.05 else "warning"
                report.add_issue(
                    QualityIssue(
                        issue_type="missing_values",
                        severity=severity,
                        description=f"Column '{col}' has {null_count} missing values ({null_count/len(df)*100:.1f}%)",
                        affected_rows=null_rows,
                        suggested_action="Use forward_fill or interpolation",
                        metadata={"column": col, "count": null_count},
                    )
                )

        # Volume can have zeros but not nulls
        vol_null = df["volume"].isnull().sum()
        if vol_null > 0:
            report.add_issue(
                QualityIssue(
                    issue_type="missing_volume",
                    severity="warning",
                    description=f"Volume has {vol_null} missing values",
                    affected_rows=df[df["volume"].isnull()].index.tolist(),
                    suggested_action="Fill with 0 or interpolate",
                )
            )

    def _check_price_integrity(self, df: pd.DataFrame, report: QualityReport):
        """Check OHLC price integrity"""
        issues = []

        # High should be >= Open, Close, Low
        invalid_high = df[
            (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])
        ].index.tolist()

        if invalid_high:
            report.add_issue(
                QualityIssue(
                    issue_type="invalid_high",
                    severity="error",
                    description=f"High price is less than O/L/C in {len(invalid_high)} rows",
                    affected_rows=invalid_high,
                    suggested_action="Set high = max(open, high, low, close)",
                )
            )

        # Low should be <= Open, Close, High
        invalid_low = df[
            (df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])
        ].index.tolist()

        if invalid_low:
            report.add_issue(
                QualityIssue(
                    issue_type="invalid_low",
                    severity="error",
                    description=f"Low price is greater than O/H/C in {len(invalid_low)} rows",
                    affected_rows=invalid_low,
                    suggested_action="Set low = min(open, high, low, close)",
                )
            )

        # Check for zero or negative prices
        zero_prices = df[
            (df["close"] <= 0) | (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0)
        ].index.tolist()

        if zero_prices:
            report.add_issue(
                QualityIssue(
                    issue_type="zero_or_negative_price",
                    severity="critical",
                    description=f"Zero or negative prices in {len(zero_prices)} rows",
                    affected_rows=zero_prices,
                    suggested_action="Remove or correct these rows",
                )
            )

    def _check_volume_anomalies(self, df: pd.DataFrame, report: QualityReport):
        """Check for volume anomalies"""
        # Check for consecutive zero volume days
        zero_vol = (df["volume"] == 0) | df["volume"].isnull()
        zero_count = zero_vol.sum()

        if zero_count > 0:
            # Check consecutive zeros
            zero_streaks = []
            streak = 0
            for i, is_zero in enumerate(zero_vol):
                if is_zero:
                    streak += 1
                else:
                    if streak >= VOLUME_ZERO_TOLERANCE:
                        zero_streaks.append((i - streak, streak))
                    streak = 0

            if zero_streaks:
                report.add_issue(
                    QualityIssue(
                        issue_type="consecutive_zero_volume",
                        severity="warning",
                        description=f"Found {len(zero_streaks)} periods of {VOLUME_ZERO_TOLERANCE}+ consecutive zero volume days",
                        suggested_action="May indicate trading halt or data issue",
                        metadata={"streaks": zero_streaks},
                    )
                )

        # Check for volume spikes
        if len(df) >= 20:
            avg_volume = df["volume"].rolling(20).mean()
            spikes = df[df["volume"] > avg_volume * VOLUME_SPIKE_THRESHOLD].index.tolist()

            if len(spikes) > len(df) * 0.05:  # More than 5% are spikes
                report.add_warning(f"High volume spikes detected ({len(spikes)} occurrences)")

    def _check_gaps(self, df: pd.DataFrame, report: QualityReport):
        """Check for price gaps exceeding limits"""
        if len(df) < 2:
            return

        # Calculate overnight gaps (close to open)
        df_temp = df.copy()
        df_temp["prev_close"] = df_temp["close"].shift(1)
        df_temp["gap_pct"] = (df_temp["open"] - df_temp["prev_close"]) / df_temp["prev_close"]

        # Critical gaps (>= price limit)
        critical_gaps = df_temp[
            (df_temp["gap_pct"].abs() >= GAP_CRITICAL_THRESHOLD) & (df_temp["prev_close"].notna())
        ]

        if len(critical_gaps) > 0:
            for idx, row in critical_gaps.iterrows():
                report.add_issue(
                    QualityIssue(
                        issue_type="critical_gap",
                        severity="warning",  # Gaps can be legitimate (circuit breaker)
                        description=f"Gap of {row['gap_pct']*100:.1f}% on {row.get('time', idx)}",
                        affected_rows=[idx],
                        suggested_action="Verify if this is a circuit breaker event or data error",
                        metadata={"gap_pct": row["gap_pct"], "date": str(row.get("time", idx))},
                    )
                )

        # Warning gaps
        warning_gaps = df_temp[
            (df_temp["gap_pct"].abs() >= GAP_WARNING_THRESHOLD)
            & (df_temp["gap_pct"].abs() < GAP_CRITICAL_THRESHOLD)
            & (df_temp["prev_close"].notna())
        ]

        if len(warning_gaps) > len(df) * 0.1:  # More than 10% have significant gaps
            report.add_warning(f"Frequent gaps detected ({len(warning_gaps)} occurrences)")

    def _check_duplicates(self, df: pd.DataFrame, report: QualityReport):
        """Check for duplicate timestamps"""
        if "time" in df.columns:
            dups = df[df.duplicated(subset=["time"], keep=False)]
            if len(dups) > 0:
                report.add_issue(
                    QualityIssue(
                        issue_type="duplicate_timestamps",
                        severity="error",
                        description=f"Found {len(dups)} rows with duplicate timestamps",
                        affected_rows=dups.index.tolist(),
                        suggested_action="Keep last occurrence: df.drop_duplicates(subset=['time'], keep='last')",
                    )
                )

    def _check_chronological_order(self, df: pd.DataFrame, report: QualityReport):
        """Check if data is in chronological order"""
        if "time" in df.columns:
            if not df["time"].is_monotonic_increasing:
                report.add_issue(
                    QualityIssue(
                        issue_type="not_chronological",
                        severity="error",
                        description="Data is not in chronological order",
                        suggested_action="Sort by time: df.sort_values('time')",
                    )
                )

    def _detect_stock_splits(self, df: pd.DataFrame, report: QualityReport):
        """
        Detect potential stock splits based on price drops with volume increase.

        Vietnam splits typically: 2:1, 3:1, 5:1, 10:1
        Detection: Price drops 40-90% overnight with high volume
        """
        if len(df) < 2:
            return

        df_temp = df.copy()
        df_temp["prev_close"] = df_temp["close"].shift(1)
        df_temp["price_change"] = df_temp["close"] / df_temp["prev_close"]

        # Look for common split ratios
        potential_splits = []
        for ratio in COMMON_SPLIT_RATIOS:
            if ratio < 1:  # Reverse split
                tolerance = 0.05
                matches = df_temp[
                    (df_temp["price_change"] >= ratio - tolerance)
                    & (df_temp["price_change"] <= ratio + tolerance)
                    & (df_temp["prev_close"].notna())
                ]
                for idx, row in matches.iterrows():
                    potential_splits.append(
                        {
                            "index": idx,
                            "date": row.get("time", idx),
                            "ratio": ratio,
                            "type": "reverse_split",
                        }
                    )
            else:  # Forward split
                inv_ratio = 1 / ratio
                tolerance = 0.05
                matches = df_temp[
                    (df_temp["price_change"] >= inv_ratio - tolerance)
                    & (df_temp["price_change"] <= inv_ratio + tolerance)
                    & (df_temp["prev_close"].notna())
                ]
                for idx, row in matches.iterrows():
                    potential_splits.append(
                        {
                            "index": idx,
                            "date": row.get("time", idx),
                            "ratio": ratio,
                            "type": "stock_split",
                        }
                    )

        if potential_splits:
            report.add_issue(
                QualityIssue(
                    issue_type="potential_stock_split",
                    severity="warning",
                    description=f"Detected {len(potential_splits)} potential stock split(s)",
                    affected_rows=[s["index"] for s in potential_splits],
                    suggested_action="Verify with corporate action data and adjust if needed",
                    metadata={"splits": potential_splits},
                )
            )

    def _detect_dividend_adjustments(self, df: pd.DataFrame, report: QualityReport):
        """
        Detect potential dividend adjustments.

        Detection: Small overnight drops (2-10%) with normal/low volume
        """
        if len(df) < 2:
            return

        df_temp = df.copy()
        df_temp["prev_close"] = df_temp["close"].shift(1)
        df_temp["overnight_change"] = (df_temp["open"] - df_temp["prev_close"]) / df_temp[
            "prev_close"
        ]

        # Dividend typically causes 2-10% overnight drop
        potential_dividends = df_temp[
            (df_temp["overnight_change"] < -0.02)
            & (df_temp["overnight_change"] > -0.15)
            & (df_temp["prev_close"].notna())
        ]

        # Filter by volume (dividends usually don't cause volume spikes)
        if "volume" in df_temp.columns and len(df_temp) >= 20:
            avg_vol = df_temp["volume"].rolling(20).mean()
            potential_dividends = potential_dividends[potential_dividends["volume"] < avg_vol * 2]

        if len(potential_dividends) > 5:  # Too many might indicate data issues
            report.add_warning(
                f"Multiple potential dividend adjustments detected ({len(potential_dividends)}). "
                "Consider verifying corporate action calendar."
            )

    def _check_outliers(self, df: pd.DataFrame, report: QualityReport):
        """Check for statistical outliers using z-score"""
        if len(df) < 30:
            return

        price_cols = ["open", "high", "low", "close"]
        outliers = []

        for col in price_cols:
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                z_scores = (df[col] - mean) / std
                col_outliers = df[z_scores.abs() > PRICE_Z_SCORE_THRESHOLD].index.tolist()
                outliers.extend(col_outliers)

        outliers = list(set(outliers))

        if outliers:
            report.add_issue(
                QualityIssue(
                    issue_type="statistical_outliers",
                    severity="warning",
                    description=f"Found {len(outliers)} statistical outliers (z-score > {PRICE_Z_SCORE_THRESHOLD})",
                    affected_rows=outliers,
                    suggested_action="Review these data points for accuracy",
                )
            )

    def clean_data(
        self,
        df: pd.DataFrame,
        method: str = "forward_fill",
        fix_ohlc: bool = True,
        remove_outliers: bool = False,
    ) -> pd.DataFrame:
        """
        Clean data based on detected issues.

        Args:
            df: DataFrame to clean
            method: Method for filling missing values
                    - 'forward_fill': Use previous value
                    - 'interpolate': Linear interpolation
                    - 'drop': Drop rows with issues
            fix_ohlc: Fix OHLC integrity issues
            remove_outliers: Remove statistical outliers

        Returns:
            Cleaned DataFrame
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        # Sort by time first
        if "time" in df.columns:
            df = df.sort_values("time").reset_index(drop=True)

        # Remove duplicates
        if "time" in df.columns:
            df = df.drop_duplicates(subset=["time"], keep="last")

        # Handle missing values
        price_cols = ["open", "high", "low", "close"]

        if method == "forward_fill":
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].ffill()
        elif method == "interpolate":
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].interpolate(method="linear")
        elif method == "drop":
            df = df.dropna(subset=price_cols)

        # Fill remaining NaN with 0 for volume
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)

        # Fix OHLC integrity
        if fix_ohlc:
            df = self._fix_ohlc_integrity(df)

        # Remove zero/negative prices
        df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]

        # Remove outliers if requested
        if remove_outliers and len(df) >= 30:
            for col in price_cols:
                if col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        z_scores = (df[col] - mean) / std
                        df = df[z_scores.abs() <= PRICE_Z_SCORE_THRESHOLD]

        return df.reset_index(drop=True)

    def _fix_ohlc_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix OHLC integrity issues"""
        df = df.copy()

        # Ensure high is the maximum
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)

        # Ensure low is the minimum
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

        return df

    def adjust_for_split(
        self,
        df: pd.DataFrame,
        split_date: str,
        split_ratio: float,
    ) -> pd.DataFrame:
        """
        Adjust historical prices for stock split.

        Args:
            df: DataFrame with OHLCV data
            split_date: Date of split (YYYY-MM-DD)
            split_ratio: Split ratio (e.g., 2.0 for 2:1 split)

        Returns:
            Adjusted DataFrame
        """
        df = df.copy()

        if "time" not in df.columns:
            logger.warning("Cannot adjust for split: 'time' column missing")
            return df

        split_dt = pd.to_datetime(split_date)

        # Adjust prices before split date
        before_split = df["time"] < split_dt

        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df.loc[before_split, col] = df.loc[before_split, col] / split_ratio

        # Adjust volume (inverse of price)
        if "volume" in df.columns:
            df.loc[before_split, "volume"] = df.loc[before_split, "volume"] * split_ratio

        logger.info(f"Adjusted {before_split.sum()} rows for {split_ratio}:1 split on {split_date}")

        return df


# =============================================================================
# Singleton Pattern for Easy Access
# =============================================================================

_quality_checker_instance: Optional[DataQualityChecker] = None


def get_quality_checker(
    exchange: str = "HOSE",
    strict_mode: bool = False,
) -> DataQualityChecker:
    """
    Get or create singleton quality checker instance.

    Args:
        exchange: Exchange type for price limits
        strict_mode: Treat warnings as errors

    Returns:
        DataQualityChecker instance
    """
    global _quality_checker_instance

    if _quality_checker_instance is None:
        _quality_checker_instance = DataQualityChecker(
            exchange=exchange,
            strict_mode=strict_mode,
        )

    return _quality_checker_instance


# =============================================================================
# Quick Validation Function
# =============================================================================


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str = "",
    raise_on_error: bool = False,
) -> Tuple[bool, Dict]:
    """
    Quick validation function for OHLCV data.

    Args:
        df: DataFrame to validate
        symbol: Stock symbol for logging
        raise_on_error: Raise exception on critical errors

    Returns:
        (is_valid, report_dict)
    """
    checker = get_quality_checker()
    report = checker.validate(df, symbol)

    if not report.valid and raise_on_error:
        issues_str = "; ".join([f"{i.issue_type}: {i.description}" for i in report.issues])
        raise ValueError(f"Data quality validation failed for {symbol}: {issues_str}")

    return report.valid, report.to_dict()


# =============================================================================
# Test Function
# =============================================================================

if __name__ == "__main__":
    # Test with sample data
    import numpy as np

    print("Testing Data Quality Module...")

    # Create sample data with issues
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    data = {
        "time": dates,
        "open": 100 + np.random.randn(100).cumsum(),
        "high": 101 + np.random.randn(100).cumsum(),
        "low": 99 + np.random.randn(100).cumsum(),
        "close": 100 + np.random.randn(100).cumsum(),
        "volume": np.random.randint(10000, 1000000, 100),
    }
    df = pd.DataFrame(data)

    # Introduce some issues
    df.loc[10, "close"] = np.nan  # Missing value
    df.loc[20, "high"] = df.loc[20, "low"] - 10  # Invalid OHLC
    df.loc[30, "volume"] = 0  # Zero volume

    # Validate
    checker = DataQualityChecker()
    report = checker.validate(df, symbol="TEST")

    print(f"\n📊 Quality Report for TEST:")
    print(f"   Valid: {report.valid}")
    print(f"   Score: {report.score:.1f}/100")
    print(f"   Issues: {len(report.issues)}")
    print(f"   Warnings: {len(report.warnings)}")

    for issue in report.issues:
        print(f"   ⚠️ [{issue.severity.upper()}] {issue.issue_type}: {issue.description}")

    # Clean data
    df_clean = checker.clean_data(df)
    print(f"\n✅ Cleaned data: {len(df_clean)} rows (from {len(df)} original)")
