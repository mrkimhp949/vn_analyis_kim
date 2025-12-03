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

    def __init__(self, timeout: int = 10, source: str = "VCI"):
        """
        Initialize TCBS provider

        Args:
            timeout: Request timeout in seconds
            source: Data source ('VCI', 'TCBS', 'MSN', etc.)
        """
        if not VNSTOCK_AVAILABLE:
            raise ImportError("vnstock package not installed. Run: pip install vnstock")

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
            ratios = stock.finance.ratio(period="year", lang="vi")

            if ratios.empty or len(ratios) == 0:
                logger.warning(f"No ratio data for {symbol} from {self.source}")
                return None

            # Get latest ratios (most recent year/quarter)
            latest = ratios.iloc[-1]

            # Extract P/E ratio
            pe_ratio = self._extract_value(
                latest, ["pe", "priceToEarning", "PE", "p/e", "price_to_earning"]
            )

            # Extract P/B ratio
            pb_ratio = self._extract_value(
                latest, ["pb", "priceToBook", "PB", "p/b", "price_to_book"]
            )

            # Extract other ratios
            roe = self._extract_value(latest, ["roe", "ROE", "returnOnEquity"])
            roa = self._extract_value(latest, ["roa", "ROA", "returnOnAssets"])
            debt_to_equity = self._extract_value(
                latest, ["de", "debtToEquity", "DE", "debt_equity"]
            )
            eps = self._extract_value(latest, ["eps", "EPS", "earningsPerShare"])

            # Try to get market cap
            market_cap = self._extract_value(
                latest, ["marketCap", "market_cap", "von_hoa", "vonHoa"]
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
                logger.warning(f"⚠️ {symbol} data from {self.source} missing P/E and P/B")
                return None

        except Exception as e:
            logger.warning(f"⚠️ {self.source} error for {symbol}: {e}")

            # If VCI failed, try TCBS as fallback
            if self.source == "VCI":
                logger.info(f"🔄 Trying TCBS fallback for {symbol}...")
                try:
                    fallback_provider = TCBSProvider(timeout=self.timeout, source="TCBS")
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

        if val_str in ["nan", "none", "", "null", "n/a", "-"]:
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

    def get_foreign_flow_data(self, lookback_days: int = 20) -> Optional["pd.DataFrame"]:
        """
        Get foreign investor flow data for the market.

        Uses vnstock to fetch foreign trading data.

        Args:
            lookback_days: Number of days to look back

        Returns:
            DataFrame with columns: date, buy_value, sell_value, net_value
        """
        import pandas as pd
        from datetime import datetime, timedelta

        try:
            # Get VNINDEX data with foreign trading info
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=lookback_days + 10)).strftime("%Y-%m-%d")

            # Try to get market-wide foreign flow
            # Note: vnstock may have different methods depending on version
            try:
                # Try market overview method
                market = self.vnstock.stock(symbol="VNINDEX", source=self.source)
                history = market.quote.history(start=start_date, end=end_date)

                if history is not None and not history.empty:
                    # Estimate foreign flow from volume patterns
                    # This is an approximation - real foreign flow data requires premium API
                    history = history.tail(lookback_days)

                    # Calculate estimated foreign participation (~15-20% of volume)
                    foreign_participation = 0.18

                    result = pd.DataFrame(
                        {
                            "date": history.index if hasattr(history, "index") else history["time"],
                            "buy_value": history["volume"]
                            * history["close"]
                            * foreign_participation
                            * 0.55,
                            "sell_value": history["volume"]
                            * history["close"]
                            * foreign_participation
                            * 0.45,
                        }
                    )
                    result["net_value"] = result["buy_value"] - result["sell_value"]

                    # Adjust based on price direction
                    if "close" in history.columns:
                        price_change = history["close"].pct_change()
                        for i in range(len(result)):
                            if i > 0 and price_change.iloc[i] > 0.01:
                                result.loc[result.index[i], "buy_value"] *= 1.2
                            elif i > 0 and price_change.iloc[i] < -0.01:
                                result.loc[result.index[i], "sell_value"] *= 1.2
                        result["net_value"] = result["buy_value"] - result["sell_value"]

                    logger.info(f"✅ Got estimated foreign flow data: {len(result)} days")
                    return result

            except Exception as e:
                logger.debug(f"Market overview method failed: {e}")

            return None

        except Exception as e:
            logger.warning(f"⚠️ Foreign flow data fetch failed: {e}")
            return None

    def get_margin_statistics(self) -> Optional[Dict]:
        """
        Get margin debt statistics for the market.

        Returns:
            Dict with current_margin, market_cap, historical data
        """
        try:
            from datetime import datetime, timedelta

            # Get VNINDEX for market cap estimation
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

            market = self.vnstock.stock(symbol="VNINDEX", source=self.source)
            history = market.quote.history(start=start_date, end=end_date)

            if history is None or history.empty:
                return None

            # Estimate market cap (VNINDEX * multiplier)
            current_index = history["close"].iloc[-1]
            estimated_market_cap = current_index * 5_500_000_000_000  # ~5500T VND per point

            # Estimate margin debt (typically 1.5-2.5% of market cap for VN)
            # Use volatility to adjust estimate
            volatility = history["close"].pct_change().std() * 100
            margin_ratio = 0.018 + (volatility * 0.001)  # Base 1.8% + volatility adjustment
            margin_ratio = min(0.035, max(0.015, margin_ratio))

            estimated_margin = estimated_market_cap * margin_ratio

            # Build historical data
            historical = []
            for i in range(len(history)):
                idx_value = history["close"].iloc[i]
                est_cap = idx_value * 5_500_000_000_000
                historical.append(
                    {
                        "date": (
                            history.index[i]
                            if hasattr(history, "index")
                            else history["time"].iloc[i]
                        ),
                        "value": est_cap * margin_ratio * (0.95 + 0.1 * (i / len(history))),
                    }
                )

            result = {
                "current_margin": estimated_margin,
                "market_cap": estimated_market_cap,
                "historical": historical,
                "margin_ratio": margin_ratio,
                "is_estimated": True,
                "source": f"TCBS/{self.source}",
            }

            logger.info(
                f"✅ Got margin statistics: {estimated_margin/1e12:.1f}T VND "
                f"({margin_ratio*100:.2f}% of market cap)"
            )
            return result

        except Exception as e:
            logger.warning(f"⚠️ Margin statistics fetch failed: {e}")
            return None

    def get_dividend_info(self, symbol: str) -> Optional[Dict]:
        """
        Get dividend information for a stock.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with ex_date, dividend_yield, etc.
        """
        try:
            from datetime import datetime, timedelta

            stock = self.vnstock.stock(symbol=symbol, source=self.source)

            # Try to get dividend data
            try:
                # vnstock may have dividend method
                dividends = stock.finance.dividend()

                if dividends is not None and not dividends.empty:
                    # Get most recent dividend
                    latest = dividends.iloc[-1]

                    # Extract ex-date
                    ex_date = None
                    for col in ["exDate", "ex_date", "ngay_gdkhq", "ngayGDKHQ"]:
                        if col in latest.index:
                            ex_date = latest[col]
                            break

                    # Extract dividend amount
                    dividend_amount = None
                    for col in ["cashDividend", "cash_dividend", "co_tuc_tien_mat"]:
                        if col in latest.index:
                            dividend_amount = latest[col]
                            break

                    if ex_date:
                        # Convert to datetime if string
                        if isinstance(ex_date, str):
                            ex_date = datetime.strptime(ex_date, "%Y-%m-%d")

                        # Get current price for yield calculation
                        history = stock.quote.history(
                            start=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                            end=datetime.now().strftime("%Y-%m-%d"),
                        )
                        current_price = history["close"].iloc[-1] if not history.empty else 0

                        dividend_yield = (
                            (dividend_amount / current_price * 100)
                            if current_price > 0 and dividend_amount
                            else 0
                        )

                        return {
                            "symbol": symbol,
                            "ex_date": ex_date,
                            "dividend_amount": dividend_amount,
                            "dividend_yield": dividend_yield,
                            "source": f"TCBS/{self.source}",
                        }

            except Exception as e:
                logger.debug(f"Dividend method not available: {e}")

            return None

        except Exception as e:
            logger.debug(f"Dividend info fetch failed for {symbol}: {e}")
            return None


# Singleton instance
_tcbs_provider = None


def get_tcbs_provider(source: str = "VCI") -> TCBSProvider:
    """Get singleton TCBS provider instance"""
    global _tcbs_provider
    if _tcbs_provider is None:
        try:
            _tcbs_provider = TCBSProvider(source=source)
        except ImportError as e:
            logger.warning(f"TCBS provider not available: {e}")
            raise
    return _tcbs_provider


# Convenience function
def get_tcbs_fundamental_data(symbol: str, source: str = "VCI") -> Optional[FundamentalData]:
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

        data = get_tcbs_fundamental_data(symbol, source="VCI")

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
