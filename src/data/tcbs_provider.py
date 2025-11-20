"""
TCBS Provider for fundamental data using vnstock package
Fallback to VCI source when TCBS is unavailable
"""

import logging
from typing import Optional, Dict
from datetime import datetime

try:
    from vnstock import Vnstock
    VNSTOCK_AVAILABLE = True
except ImportError:
    VNSTOCK_AVAILABLE = False

from src.data.fundamental_data import FundamentalData, FundamentalDataProvider

logger = logging.getLogger(__name__)


class TCBSProvider(FundamentalDataProvider):
    """
    TCBS/VCI Provider using vnstock package
    Falls back to VCI when TCBS is unavailable

    Sources priority:
    1. VCI (Viet Capital Securities) - Most reliable
    2. TCBS (Techcom Securities) - Backup
    """

    def __init__(self, timeout: int = 10, source: str = 'VCI'):
        """
        Initialize TCBS provider

        Args:
            timeout: Request timeout in seconds
            source: Data source ('VCI', 'TCBS', 'MSN', etc.)
        """
        if not VNSTOCK_AVAILABLE:
            raise ImportError(
                "vnstock package not installed. Run: pip install vnstock"
            )

        self.timeout = timeout
        self.source = source
        self.vnstock = Vnstock()
        logger.info(f"📊 TCBS Provider initialized with source: {source}")

    def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """
        Get fundamental data using vnstock

        Args:
            symbol: Stock symbol (e.g., 'VNM', 'FPT')

        Returns:
            FundamentalData object or None if unavailable
        """
        try:
            # Initialize stock object
            stock = self.vnstock.stock(symbol=symbol, source=self.source)

            # Get financial ratios (yearly data, Vietnamese language)
            ratios = stock.finance.ratio(period='year', lang='vi')

            if ratios.empty or len(ratios) == 0:
                logger.warning(f"No ratio data for {symbol} from {self.source}")
                return None

            # Get latest ratios (most recent year/quarter)
            latest = ratios.iloc[-1]

            # Extract P/E ratio
            pe_ratio = self._extract_value(
                latest,
                ['pe', 'priceToEarning', 'PE', 'p/e', 'price_to_earning']
            )

            # Extract P/B ratio
            pb_ratio = self._extract_value(
                latest,
                ['pb', 'priceToBook', 'PB', 'p/b', 'price_to_book']
            )

            # Extract other ratios
            roe = self._extract_value(latest, ['roe', 'ROE', 'returnOnEquity'])
            roa = self._extract_value(latest, ['roa', 'ROA', 'returnOnAssets'])
            debt_to_equity = self._extract_value(
                latest,
                ['de', 'debtToEquity', 'DE', 'debt_equity']
            )
            eps = self._extract_value(latest, ['eps', 'EPS', 'earningsPerShare'])

            # Try to get market cap
            market_cap = self._extract_value(
                latest,
                ['marketCap', 'market_cap', 'von_hoa', 'vonHoa']
            )

            # Create FundamentalData object
            fundamental = FundamentalData(
                symbol=symbol,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                roe=roe,
                roa=roa,
                debt_to_equity=debt_to_equity,
                eps=eps,
                market_cap=market_cap,
                source=f"TCBS/{self.source}",
                timestamp=datetime.now(),
            )

            # Check if we got at least P/E or P/B
            if pe_ratio is not None or pb_ratio is not None:
                logger.info(
                    f"✅ Got fundamental data for {symbol} from {self.source}: "
                    f"P/E={pe_ratio}, P/B={pb_ratio}"
                )
                return fundamental
            else:
                logger.warning(
                    f"⚠️ {symbol} data from {self.source} missing P/E and P/B"
                )
                return None

        except Exception as e:
            logger.warning(f"⚠️ {self.source} error for {symbol}: {e}")

            # If VCI failed, try TCBS as fallback
            if self.source == 'VCI':
                logger.info(f"🔄 Trying TCBS fallback for {symbol}...")
                try:
                    fallback_provider = TCBSProvider(
                        timeout=self.timeout,
                        source='TCBS'
                    )
                    return fallback_provider.get_fundamental_data(symbol)
                except Exception as fallback_error:
                    logger.warning(f"⚠️ TCBS fallback also failed: {fallback_error}")

            return None

    def _extract_value(self, row, possible_keys: list) -> Optional[float]:
        """
        Extract numeric value from pandas Series using multiple possible key names

        Args:
            row: Pandas Series (one row of data)
            possible_keys: List of possible column names to try

        Returns:
            Float value or None if not found/invalid
        """
        for key in possible_keys:
            # Try exact match
            if key in row.index:
                val = row[key]
                if self._is_valid_number(val):
                    try:
                        return float(val)
                    except:
                        pass

            # Try case-insensitive match
            for col in row.index:
                if col.lower() == key.lower():
                    val = row[col]
                    if self._is_valid_number(val):
                        try:
                            return float(val)
                        except:
                            pass

        return None

    def _is_valid_number(self, val) -> bool:
        """Check if value is a valid number (not NaN, None, empty string)"""
        if val is None:
            return False

        val_str = str(val).lower().strip()

        if val_str in ['nan', 'none', '', 'null', 'n/a', '-']:
            return False

        return True

    def get_earnings_date(self, symbol: str) -> Optional[Dict]:
        """
        Get earnings dates

        Note: VCI/TCBS may not provide earnings calendar through vnstock

        Returns:
            None (not implemented)
        """
        logger.warning(f"⚠️ Earnings dates not available from {self.source} source")
        return None


# Convenience function
def get_tcbs_fundamental_data(symbol: str, source: str = 'VCI') -> Optional[FundamentalData]:
    """
    Quick function to get fundamental data from TCBS/VCI

    Args:
        symbol: Stock symbol
        source: Data source ('VCI' or 'TCBS')

    Returns:
        FundamentalData or None
    """
    provider = TCBSProvider(source=source)
    return provider.get_fundamental_data(symbol)


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("🧪 TESTING TCBS PROVIDER")
    print("=" * 70 + "\n")

    test_symbols = ["VNM", "VCB", "FPT", "HPG"]

    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")

        data = get_tcbs_fundamental_data(symbol, source='VCI')

        if data and data.is_valid():
            print(f"✅ {symbol} Fundamental Data:")
            print(f"   Source: {data.source}")
            print(f"   P/E: {data.pe_ratio}")
            print(f"   P/B: {data.pb_ratio}")
            print(f"   ROE: {data.roe}%")
            print(f"   ROA: {data.roa}%")
            print(f"   Debt/Equity: {data.debt_to_equity}")
            print(f"   EPS: {data.eps}")
            if data.market_cap:
                print(f"   Market Cap: {data.market_cap:,.0f} VND")
        else:
            print(f"❌ No data available for {symbol}")

    print("\n" + "=" * 70)
