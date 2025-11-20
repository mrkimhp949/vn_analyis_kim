"""
CSV Fundamental Data Provider
Reads P/E, P/B and other ratios from CSV file
Simple, reliable solution for fundamental data
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

from src.data.fundamental_data import FundamentalData, FundamentalDataProvider

logger = logging.getLogger(__name__)


class CSVFundamentalProvider(FundamentalDataProvider):
    """
    Provider that reads fundamental data from CSV file

    CSV Format:
    Symbol,PE,PB,ROE,ROA,DebtToEquity,EPS,MarketCap,LastUpdate
    VNM,18.5,5.2,25.3,12.1,0.35,5200,95000000000000,2025-11-20
    """

    def __init__(self, csv_path: str = "data/fundamental_ratios.csv", max_age_days: int = 30):
        """
        Initialize CSV provider

        Args:
            csv_path: Path to CSV file
            max_age_days: Maximum age of data in days before warning
        """
        self.csv_path = csv_path
        self.max_age_days = max_age_days
        self.data_cache = None
        self.last_load_time = None

        # Create directory if doesn't exist
        csv_dir = os.path.dirname(csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

        logger.info(f"📊 CSV Fundamental Provider initialized: {csv_path}")

    def _load_csv(self) -> pd.DataFrame:
        """Load CSV file into memory"""
        try:
            if not os.path.exists(self.csv_path):
                logger.warning(f"⚠️ CSV file not found: {self.csv_path}")
                return pd.DataFrame()

            df = pd.read_csv(self.csv_path)

            # Validate required columns
            required_cols = ['Symbol', 'PE', 'PB']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.error(f"❌ CSV missing required columns: {missing_cols}")
                return pd.DataFrame()

            # Convert LastUpdate to datetime if exists
            if 'LastUpdate' in df.columns:
                df['LastUpdate'] = pd.to_datetime(df['LastUpdate'], errors='coerce')

            logger.info(f"✅ Loaded {len(df)} symbols from CSV")
            return df

        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()

    def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental data from CSV"""
        try:
            # Reload CSV if cache is stale (older than 1 hour)
            if (
                self.data_cache is None
                or self.last_load_time is None
                or (datetime.now() - self.last_load_time).total_seconds() > 3600
            ):
                self.data_cache = self._load_csv()
                self.last_load_time = datetime.now()

            if self.data_cache.empty:
                return None

            # Find symbol in CSV
            row = self.data_cache[self.data_cache['Symbol'] == symbol]

            if row.empty:
                logger.debug(f"⚠️ {symbol} not found in CSV")
                return None

            row = row.iloc[0]

            # Check data freshness
            if 'LastUpdate' in row.index and pd.notna(row['LastUpdate']):
                age_days = (datetime.now() - row['LastUpdate']).days
                if age_days > self.max_age_days:
                    logger.warning(
                        f"⚠️ Data for {symbol} is {age_days} days old (max: {self.max_age_days})"
                    )

            # Create FundamentalData object
            fundamental = FundamentalData(
                symbol=symbol,
                pe_ratio=float(row['PE']) if pd.notna(row.get('PE')) else None,
                pb_ratio=float(row['PB']) if pd.notna(row.get('PB')) else None,
                roe=float(row['ROE']) if pd.notna(row.get('ROE')) else None,
                roa=float(row['ROA']) if pd.notna(row.get('ROA')) else None,
                debt_to_equity=(
                    float(row['DebtToEquity']) if pd.notna(row.get('DebtToEquity')) else None
                ),
                eps=float(row['EPS']) if pd.notna(row.get('EPS')) else None,
                market_cap=float(row['MarketCap']) if pd.notna(row.get('MarketCap')) else None,
                source="CSV",
                timestamp=(
                    row['LastUpdate']
                    if 'LastUpdate' in row.index and pd.notna(row['LastUpdate'])
                    else datetime.now()
                ),
            )

            logger.debug(
                f"✅ Got {symbol} from CSV: P/E={fundamental.pe_ratio}, P/B={fundamental.pb_ratio}"
            )
            return fundamental

        except Exception as e:
            logger.error(f"❌ Error getting {symbol} from CSV: {e}")
            return None

    def get_earnings_date(self, symbol: str) -> Optional[Dict]:
        """Earnings dates not available from CSV"""
        return None


# Convenience function
def get_csv_fundamental_data(
    symbol: str, csv_path: str = "data/fundamental_ratios.csv"
) -> Optional[FundamentalData]:
    """Quick function to get data from CSV"""
    provider = CSVFundamentalProvider(csv_path=csv_path)
    return provider.get_fundamental_data(symbol)


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("🧪 TESTING CSV FUNDAMENTAL PROVIDER")
    print("=" * 70 + "\n")

    test_symbols = ["VNM", "VCB", "FPT", "INVALID"]

    for symbol in test_symbols:
        print(f"📊 Testing {symbol}...")
        data = get_csv_fundamental_data(symbol)

        if data and data.is_valid():
            print(f"✅ {symbol}:")
            print(f"   P/E: {data.pe_ratio}")
            print(f"   P/B: {data.pb_ratio}")
            print(f"   ROE: {data.roe}%")
            print(f"   Source: {data.source}")
        else:
            print(f"❌ No data for {symbol}")
        print()

    print("=" * 70)
