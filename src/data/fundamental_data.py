"""
Fundamental Data Integration
Provides P/E ratio, debt ratio, earnings dates from multiple sources
Supports: CSV, VNDirect, SSI, FiinTrade APIs
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class FundamentalData:
    """Container for fundamental data"""

    symbol: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    debt_ratio: Optional[float] = None  # Total debt / Total assets
    debt_to_equity: Optional[float] = None  # Total debt / Total equity
    roe: Optional[float] = None  # Return on equity
    roa: Optional[float] = None  # Return on assets
    revenue_growth: Optional[float] = None  # YoY revenue growth
    profit_margin: Optional[float] = None  # Net profit margin
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None  # Market cap in VND
    eps: Optional[float] = None  # Earnings per share
    next_earnings_date: Optional[datetime] = None
    last_earnings_date: Optional[datetime] = None
    source: Optional[str] = None  # Data source (VNDirect, SSI, etc.)
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def is_valid(self) -> bool:
        """Check if fundamental data is valid and fresh"""
        if self.timestamp is None:
            return False

        # Data older than 7 days is stale
        if (datetime.now() - self.timestamp).days > 7:
            return False

        # At least one fundamental metric must be available
        return any(
            [
                self.pe_ratio is not None,
                self.debt_ratio is not None,
                self.roe is not None,
                self.eps is not None,
            ]
        )


class FundamentalDataProvider(ABC):
    """Abstract base class for fundamental data providers"""

    @abstractmethod
    def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental data for a symbol"""
        pass

    @abstractmethod
    def get_earnings_date(self, symbol: str) -> Optional[Dict]:
        """Get earnings dates for a symbol"""
        pass


class VNDirectProvider(FundamentalDataProvider):
    """
    VNDirect API Provider
    Free API: https://finfo-api.vndirect.com.vn
    """

    def __init__(self, timeout: int = 10):
        self.base_url = "https://finfo-api.vndirect.com.vn"
        self.timeout = timeout

    def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental data from VNDirect"""
        try:
            # Get stock info
            url = f"{self.base_url}/v4/stocks"
            params = {"q": f"code:{symbol}", "size": 1}

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            if not data.get("data") or len(data["data"]) == 0:
                logger.warning(f"No fundamental data found for {symbol} from VNDirect")
                return None

            stock_data = data["data"][0]

            # Get financial ratios
            ratios_url = f"{self.base_url}/v4/ratios"
            ratios_params = {"q": f"code:{symbol}", "size": 1}

            ratios_response = requests.get(ratios_url, params=ratios_params, timeout=self.timeout)
            ratios_response.raise_for_status()
            ratios_data = ratios_response.json()

            ratios = ratios_data.get("data", [{}])[0] if ratios_data.get("data") else {}

            # Parse data
            fundamental = FundamentalData(
                symbol=symbol,
                pe_ratio=stock_data.get("pE"),
                pb_ratio=stock_data.get("pB"),
                eps=stock_data.get("ePS"),
                market_cap=stock_data.get("marketCap"),
                # From ratios endpoint
                debt_to_equity=ratios.get("debtOnEquity"),
                roe=ratios.get("roe"),
                roa=ratios.get("roa"),
                profit_margin=ratios.get("netProfitMargin"),
                dividend_yield=ratios.get("dividendYield"),
                source="VNDirect",
            )

            # Calculate debt ratio from debt/equity if available
            if fundamental.debt_to_equity is not None:
                # debt_ratio = debt / (debt + equity) = (debt/equity) / (1 + debt/equity)
                fundamental.debt_ratio = fundamental.debt_to_equity / (
                    1 + fundamental.debt_to_equity
                )

            logger.debug(f"✅ Got fundamental data for {symbol} from VNDirect")
            return fundamental

        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ VNDirect API timeout for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ VNDirect API error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing VNDirect data for {symbol}: {e}")
            return None

    def get_earnings_date(self, symbol: str) -> Optional[Dict]:
        """Get earnings dates from VNDirect"""
        try:
            # VNDirect financial reports endpoint
            url = f"{self.base_url}/v4/financial_statements"
            params = {"q": f"code:{symbol}", "size": 5}  # Get last 5 reports

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            if not data.get("data") or len(data["data"]) == 0:
                return None

            reports = data["data"]

            # Find last earnings date
            last_report = reports[0]
            last_earnings_date = last_report.get("reportDate")

            if last_earnings_date:
                last_earnings_date = datetime.strptime(last_earnings_date, "%Y-%m-%d")

            # Estimate next earnings (quarterly)
            next_earnings_date = None
            if last_earnings_date:
                # Assume quarterly reports (every 3 months)
                next_earnings_date = last_earnings_date + timedelta(days=90)

            return {
                "last_earnings_date": last_earnings_date,
                "next_earnings_date": next_earnings_date,
                "source": "VNDirect",
            }

        except Exception as e:
            logger.warning(f"⚠️ Error getting earnings dates for {symbol}: {e}")
            return None


class SSIProvider(FundamentalDataProvider):
    """
    SSI API Provider
    Note: SSI API requires authentication for detailed fundamental data
    This is a simplified version using publicly available data
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.base_url = "https://iboard.ssi.com.vn/dchart/api"
        self.api_key = api_key
        self.timeout = timeout

    def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """
        Get fundamental data from SSI
        Note: Limited without API key
        """
        try:
            # SSI public API (limited data)
            url = f"{self.base_url}/company/{symbol}/fundamental"

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(url, headers=headers, timeout=self.timeout)

            if response.status_code == 401:
                logger.warning("⚠️ SSI API requires authentication")
                return None

            response.raise_for_status()
            data = response.json()

            # Parse SSI data format (adjust based on actual API response)
            fundamental = FundamentalData(
                symbol=symbol,
                pe_ratio=data.get("pe"),
                pb_ratio=data.get("pb"),
                roe=data.get("roe"),
                roa=data.get("roa"),
                debt_to_equity=data.get("de"),
                eps=data.get("eps"),
                market_cap=data.get("marketCap"),
                source="SSI",
            )

            logger.debug(f"✅ Got fundamental data for {symbol} from SSI")
            return fundamental

        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ SSI API error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing SSI data for {symbol}: {e}")
            return None

    def get_earnings_date(self, symbol: str) -> Optional[Dict]:
        """Get earnings dates from SSI"""
        # SSI API for earnings calendar requires authentication
        logger.warning("⚠️ SSI earnings calendar requires API authentication")
        return None


class FiinTradeProvider(FundamentalDataProvider):
    """
    FiinTrade API Provider
    Note: FiinTrade is a paid service, requires API key
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.base_url = "https://api.fiintrade.vn"
        self.api_key = api_key
        self.timeout = timeout

    def get_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental data from FiinTrade"""
        if not self.api_key:
            logger.warning("⚠️ FiinTrade API key not provided")
            return None

        try:
            url = f"{self.base_url}/stock/{symbol}/fundamental"
            headers = {"X-API-KEY": self.api_key}

            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            fundamental = FundamentalData(
                symbol=symbol,
                pe_ratio=data.get("peRatio"),
                pb_ratio=data.get("pbRatio"),
                debt_ratio=data.get("debtRatio"),
                roe=data.get("roe"),
                roa=data.get("roa"),
                revenue_growth=data.get("revenueGrowth"),
                profit_margin=data.get("profitMargin"),
                eps=data.get("eps"),
                market_cap=data.get("marketCap"),
                dividend_yield=data.get("dividendYield"),
                source="FiinTrade",
            )

            logger.debug(f"✅ Got fundamental data for {symbol} from FiinTrade")
            return fundamental

        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ FiinTrade API error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing FiinTrade data for {symbol}: {e}")
            return None

    def get_earnings_date(self, symbol: str) -> Optional[Dict]:
        """Get earnings dates from FiinTrade"""
        if not self.api_key:
            return None

        try:
            url = f"{self.base_url}/stock/{symbol}/earnings-calendar"
            headers = {"X-API-KEY": self.api_key}

            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            return {
                "last_earnings_date": (
                    datetime.fromisoformat(data.get("lastEarningsDate"))
                    if data.get("lastEarningsDate")
                    else None
                ),
                "next_earnings_date": (
                    datetime.fromisoformat(data.get("nextEarningsDate"))
                    if data.get("nextEarningsDate")
                    else None
                ),
                "source": "FiinTrade",
            }

        except Exception as e:
            logger.warning(f"⚠️ Error getting FiinTrade earnings for {symbol}: {e}")
            return None


class FundamentalDataManager:
    """
    Manager for fundamental data from multiple sources
    Uses fallback strategy: CSV -> VNDirect -> SSI -> FiinTrade
    Caches data to minimize API calls
    """

    def __init__(
        self,
        csv_enabled: bool = True,
        csv_path: str = "data/fundamental_ratios.csv",
        vndirect_enabled: bool = False,
        ssi_enabled: bool = False,
        fiintrade_enabled: bool = False,
        ssi_api_key: Optional[str] = None,
        fiintrade_api_key: Optional[str] = None,
        cache_ttl_hours: int = 24,
    ):
        self.providers = []

        # Initialize providers in priority order
        # CSV first (most reliable for production)
        if csv_enabled:
            try:
                from src.data.csv_fundamental_provider import CSVFundamentalProvider
                self.providers.append(CSVFundamentalProvider(csv_path=csv_path))
            except ImportError:
                logger.warning("⚠️ CSV Fundamental Provider not available")

        if vndirect_enabled:
            self.providers.append(VNDirectProvider())

        if ssi_enabled:
            self.providers.append(SSIProvider(api_key=ssi_api_key))

        if fiintrade_enabled:
            self.providers.append(FiinTradeProvider(api_key=fiintrade_api_key))

        self.cache: Dict[str, FundamentalData] = {}
        self.cache_ttl_hours = cache_ttl_hours

        logger.info(f"📊 Fundamental Data Manager initialized with {len(self.providers)} providers")

    def get_fundamental_data(
        self, symbol: str, use_cache: bool = True
    ) -> Optional[FundamentalData]:
        """
        Get fundamental data with fallback across multiple providers

        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data

        Returns:
            FundamentalData or None
        """
        # Check cache first
        if use_cache and symbol in self.cache:
            cached = self.cache[symbol]
            age_hours = (datetime.now() - cached.timestamp).total_seconds() / 3600
            if age_hours < self.cache_ttl_hours and cached.is_valid():
                logger.debug(
                    f"📦 Using cached fundamental data for {symbol} (age: {age_hours:.1f}h)"
                )
                return cached

        # Try each provider until we get valid data
        for provider in self.providers:
            try:
                data = provider.get_fundamental_data(symbol)
                if data and data.is_valid():
                    self.cache[symbol] = data
                    logger.info(f"✅ Got fundamental data for {symbol} from {data.source}")
                    return data
            except Exception as e:
                logger.warning(f"⚠️ Provider {provider.__class__.__name__} failed for {symbol}: {e}")
                continue

        logger.warning(f"❌ No fundamental data available for {symbol} from any provider")
        return None

    def get_earnings_date(self, symbol: str) -> Optional[Dict]:
        """Get earnings dates with fallback"""
        for provider in self.providers:
            try:
                earnings = provider.get_earnings_date(symbol)
                if earnings:
                    return earnings
            except Exception as e:
                logger.warning(f"⚠️ Earnings date fetch failed for {symbol}: {e}")
                continue

        return None

    def clear_cache(self):
        """Clear fundamental data cache"""
        self.cache.clear()
        logger.info("🗑️ Fundamental data cache cleared")


# Singleton instance
_fundamental_manager = None


def get_fundamental_manager(
    csv_enabled: bool = True,
    csv_path: str = "data/fundamental_ratios.csv",
    vndirect_enabled: bool = False,
    ssi_api_key: Optional[str] = None,
    fiintrade_api_key: Optional[str] = None,
) -> FundamentalDataManager:
    """Get fundamental data manager singleton"""
    global _fundamental_manager
    if _fundamental_manager is None:
        _fundamental_manager = FundamentalDataManager(
            csv_enabled=csv_enabled,
            csv_path=csv_path,
            vndirect_enabled=vndirect_enabled,
            ssi_enabled=bool(ssi_api_key),
            fiintrade_enabled=bool(fiintrade_api_key),
            ssi_api_key=ssi_api_key,
            fiintrade_api_key=fiintrade_api_key,
        )
    return _fundamental_manager


# Convenience functions
def get_fundamental_data(symbol: str) -> Optional[FundamentalData]:
    """Get fundamental data for a symbol"""
    manager = get_fundamental_manager()
    return manager.get_fundamental_data(symbol)


def get_earnings_date(symbol: str) -> Optional[Dict]:
    """Get earnings dates for a symbol"""
    manager = get_fundamental_manager()
    return manager.get_earnings_date(symbol)


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("\n" + "=" * 70)
    print("🧪 TESTING FUNDAMENTAL DATA INTEGRATION")
    print("=" * 70 + "\n")

    # Test with VNM (Vinamilk)
    symbol = "VNM"

    print(f"📊 Fetching fundamental data for {symbol}...")
    data = get_fundamental_data(symbol)

    if data:
        print(f"\n✅ Fundamental Data for {symbol}:")
        print(f"  Source: {data.source}")
        print(f"  P/E Ratio: {data.pe_ratio}")
        print(f"  P/B Ratio: {data.pb_ratio}")
        print(f"  Debt Ratio: {data.debt_ratio}")
        print(f"  Debt/Equity: {data.debt_to_equity}")
        print(f"  ROE: {data.roe}%")
        print(f"  ROA: {data.roa}%")
        print(f"  EPS: {data.eps}")
        print(
            f"  Market Cap: {data.market_cap:,.0f} VND" if data.market_cap else "  Market Cap: N/A"
        )
        print(f"  Valid: {data.is_valid()}")
    else:
        print(f"❌ No fundamental data available for {symbol}")

    print(f"\n📅 Fetching earnings dates for {symbol}...")
    earnings = get_earnings_date(symbol)

    if earnings:
        print(f"\n✅ Earnings Dates for {symbol}:")
        print(f"  Last Earnings: {earnings.get('last_earnings_date')}")
        print(f"  Next Earnings: {earnings.get('next_earnings_date')}")
        print(f"  Source: {earnings.get('source')}")
    else:
        print(f"❌ No earnings data available for {symbol}")

    print("\n" + "=" * 70)
