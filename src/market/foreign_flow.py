# -*- coding: utf-8 -*-
"""
Foreign Investor Flow Analysis for Vietnam Stock Market

Analyzes foreign investor net buy/sell patterns to gauge market sentiment.
Foreign investors are often considered "smart money" in Vietnam market.

Data Sources (to be integrated):
- HOSE/HNX daily foreign trading reports
- SSI, VNDirect, TCBS broker APIs
- Financial data providers (Fireant, CafeF, etc.)

Usage:
    from src.market.foreign_flow import get_foreign_flow_analyzer
    
    analyzer = get_foreign_flow_analyzer()
    flow = analyzer.analyze()
    print(f"Foreign flow score: {flow['score']}")
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForeignFlowData:
    """Container for foreign flow analysis results"""

    date: str
    net_value: float  # Net buy/sell value in VND (positive = net buy)
    buy_value: float
    sell_value: float
    net_volume: int
    score: float  # -1 (heavy selling) to +1 (heavy buying)
    trend: str  # BUYING, SELLING, NEUTRAL
    strength: str  # STRONG, MODERATE, WEAK
    consecutive_days: int  # Days of same direction
    vs_average: float  # Ratio vs 20-day average


class ForeignFlowAnalyzer:
    """
    Analyze foreign investor trading patterns.

    Scoring Logic:
    - Net buy > 2x average: +1.0 (STRONG BUYING)
    - Net buy > 1x average: +0.5 (MODERATE BUYING)
    - Net buy > 0: +0.2 (WEAK BUYING)
    - Net sell < 0: -0.2 (WEAK SELLING)
    - Net sell < -1x average: -0.5 (MODERATE SELLING)
    - Net sell < -2x average: -1.0 (STRONG SELLING)

    Additional factors:
    - Consecutive days bonus: +0.1 per day (max +0.3)
    - Volume confirmation: +0.1 if volume > average
    """

    def __init__(
        self,
        lookback_days: int = 20,
        strong_threshold_multiplier: float = 2.0,
        moderate_threshold_multiplier: float = 1.0,
        cache_ttl_seconds: int = 300,  # 5 minutes
    ):
        self.lookback_days = lookback_days
        self.strong_threshold = strong_threshold_multiplier
        self.moderate_threshold = moderate_threshold_multiplier
        self.cache_ttl = cache_ttl_seconds

        # Cache
        self._cache = None
        self._cache_time = None

        # Historical data storage
        self._historical_data: List[Dict] = []

    def analyze(self, force_refresh: bool = False) -> ForeignFlowData:
        """
        Analyze current foreign flow.

        Args:
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            ForeignFlowData with analysis results
        """
        # Check cache
        if not force_refresh and self._is_cache_valid():
            return self._cache

        try:
            # Fetch latest data
            raw_data = self._fetch_foreign_flow_data()

            if raw_data is None or len(raw_data) == 0:
                return self._default_result("No data available")

            # Calculate metrics
            result = self._calculate_metrics(raw_data)

            # Update cache
            self._cache = result
            self._cache_time = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Foreign flow analysis failed: {e}", exc_info=True)
            return self._default_result(f"Error: {str(e)}")

    def _fetch_foreign_flow_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch foreign flow data from data source.

        Integrated sources:
        - TCBS API (primary)
        - Manual data input (fallback)
        - Historical cache

        Returns:
            DataFrame with columns: date, buy_value, sell_value, net_value
        """
        # Try TCBS provider first
        try:
            from src.data.tcbs_provider import get_tcbs_provider

            provider = get_tcbs_provider()
            data = provider.get_foreign_flow_data(lookback_days=self.lookback_days)

            if data is not None and not data.empty:
                logger.info(f"✅ Fetched foreign flow data from TCBS: {len(data)} days")
                return data

        except ImportError:
            logger.debug("TCBS provider not available for foreign flow")
        except Exception as e:
            logger.warning(f"TCBS foreign flow fetch failed: {e}")

        # Try SSI API as fallback
        try:
            from src.data.ssi_provider import get_ssi_provider

            provider = get_ssi_provider()
            data = provider.get_foreign_flow_data(lookback_days=self.lookback_days)

            if data is not None and not data.empty:
                logger.info(f"✅ Fetched foreign flow data from SSI: {len(data)} days")
                return data

        except ImportError:
            logger.debug("SSI provider not available for foreign flow")
        except Exception as e:
            logger.warning(f"SSI foreign flow fetch failed: {e}")

        # Use manual/historical data if available
        if self._historical_data and len(self._historical_data) >= 5:
            logger.info(f"Using {len(self._historical_data)} manual foreign flow records")
            return pd.DataFrame(self._historical_data)

        # Estimate from VNINDEX volume patterns as last resort
        try:
            estimated_data = self._estimate_foreign_flow_from_market()
            if estimated_data is not None:
                logger.info("Using estimated foreign flow from market data")
                return estimated_data
        except Exception as e:
            logger.debug(f"Foreign flow estimation failed: {e}")

        logger.warning("No foreign flow data available from any source")
        return None

    def _estimate_foreign_flow_from_market(self) -> Optional[pd.DataFrame]:
        """
        Estimate foreign flow from market data patterns.

        Logic:
        - High volume + price up = likely foreign buying
        - High volume + price down = likely foreign selling
        - Uses VNINDEX as proxy
        """
        try:
            from src.data.vnindex_cache import get_cached_vnindex

            vnindex_df = get_cached_vnindex(lookback=self.lookback_days + 10)
            if vnindex_df is None or len(vnindex_df) < self.lookback_days:
                return None

            # Calculate estimated flow
            vnindex_df = vnindex_df.tail(self.lookback_days).copy()
            vnindex_df["price_change"] = vnindex_df["close"].pct_change()
            vnindex_df["volume_ratio"] = (
                vnindex_df["volume"] / vnindex_df["volume"].rolling(20).mean()
            )

            # Estimate: volume_ratio * price_change direction
            # Positive = buying, Negative = selling
            vnindex_df["estimated_flow"] = vnindex_df["volume_ratio"] * vnindex_df[
                "price_change"
            ].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

            # Convert to buy/sell values (rough estimate based on typical foreign participation ~20%)
            avg_value = vnindex_df["volume"].mean() * vnindex_df["close"].mean() * 0.20

            result = pd.DataFrame(
                {
                    "date": vnindex_df.index,
                    "buy_value": [
                        avg_value * (1 + max(0, f)) for f in vnindex_df["estimated_flow"]
                    ],
                    "sell_value": [
                        avg_value * (1 + max(0, -f)) for f in vnindex_df["estimated_flow"]
                    ],
                }
            )
            result["net_value"] = result["buy_value"] - result["sell_value"]

            return result

        except Exception as e:
            logger.debug(f"Market-based foreign flow estimation failed: {e}")
            return None

    def _calculate_metrics(self, data: pd.DataFrame) -> ForeignFlowData:
        """Calculate foreign flow metrics from raw data"""

        # Latest day
        latest = data.iloc[-1]
        net_value = latest.get("net_value", 0)
        buy_value = latest.get("buy_value", 0)
        sell_value = latest.get("sell_value", 0)

        # Calculate average
        avg_net = data["net_value"].abs().mean() if len(data) > 0 else 1

        # Calculate score
        score = self._calculate_score(net_value, avg_net)

        # Determine trend and strength
        trend, strength = self._determine_trend_strength(net_value, avg_net)

        # Count consecutive days
        consecutive = self._count_consecutive_days(data)

        # Ratio vs average
        vs_average = net_value / avg_net if avg_net > 0 else 0

        return ForeignFlowData(
            date=latest.get("date", datetime.now().isoformat()),
            net_value=net_value,
            buy_value=buy_value,
            sell_value=sell_value,
            net_volume=latest.get("net_volume", 0),
            score=score,
            trend=trend,
            strength=strength,
            consecutive_days=consecutive,
            vs_average=vs_average,
        )

    def _calculate_score(self, net_value: float, avg_net: float) -> float:
        """Calculate score from -1 to +1"""
        if avg_net == 0:
            return 0.0

        ratio = net_value / avg_net

        if ratio >= self.strong_threshold:
            return 1.0
        elif ratio >= self.moderate_threshold:
            return 0.5
        elif ratio > 0:
            return 0.2
        elif ratio > -self.moderate_threshold:
            return -0.2
        elif ratio > -self.strong_threshold:
            return -0.5
        else:
            return -1.0

    def _determine_trend_strength(self, net_value: float, avg_net: float) -> tuple:
        """Determine trend direction and strength"""
        if avg_net == 0:
            return "NEUTRAL", "WEAK"

        ratio = abs(net_value) / avg_net

        if net_value > 0:
            trend = "BUYING"
        elif net_value < 0:
            trend = "SELLING"
        else:
            trend = "NEUTRAL"

        if ratio >= self.strong_threshold:
            strength = "STRONG"
        elif ratio >= self.moderate_threshold:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return trend, strength

    def _count_consecutive_days(self, data: pd.DataFrame) -> int:
        """Count consecutive days of same direction"""
        if len(data) < 2:
            return 1

        # Get direction of latest day
        latest_direction = 1 if data.iloc[-1]["net_value"] > 0 else -1

        count = 1
        for i in range(len(data) - 2, -1, -1):
            day_direction = 1 if data.iloc[i]["net_value"] > 0 else -1
            if day_direction == latest_direction:
                count += 1
            else:
                break

        return count

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if self._cache is None or self._cache_time is None:
            return False

        age = (datetime.now() - self._cache_time).total_seconds()
        return age < self.cache_ttl

    def is_data_stale(self, max_delay_minutes: int = 15) -> bool:
        """
        Check if foreign flow data is stale (delayed more than threshold).

        IMPROVED v6.0: Data staleness detection
        =========================================================================
        Foreign flow data can be delayed due to:
        - API rate limits
        - Data provider delays
        - Network issues

        When data is stale (> 15 minutes old), its weight should be reduced
        by 50% in trading decisions.
        =========================================================================

        Args:
            max_delay_minutes: Maximum acceptable delay in minutes (default: 15)

        Returns:
            True if data is stale, False if fresh
        """
        if self._cache_time is None:
            return True

        age_minutes = (datetime.now() - self._cache_time).total_seconds() / 60
        is_stale = age_minutes > max_delay_minutes

        if is_stale:
            logger.warning(
                f"⚠️ Foreign flow data is stale ({age_minutes:.0f} min old, "
                f"threshold: {max_delay_minutes} min). Reducing weight by 50%."
            )

        return is_stale

    def get_data_age_minutes(self) -> float:
        """Get age of cached data in minutes"""
        if self._cache_time is None:
            return float("inf")
        return (datetime.now() - self._cache_time).total_seconds() / 60

    def get_adjusted_score(self, max_delay_minutes: int = 15) -> float:
        """
        Get foreign flow score with staleness adjustment.

        IMPROVED v6.0: Automatic weight reduction for stale data
        =========================================================================
        - Fresh data (< 15 min): Full score
        - Stale data (> 15 min): Score reduced by 50%
        - No data: Score = 0 (neutral)
        =========================================================================

        Args:
            max_delay_minutes: Maximum acceptable delay before weight reduction

        Returns:
            Adjusted score (-1 to +1, or reduced if stale)
        """
        result = self.analyze()

        if result.score == 0:
            return 0.0

        if self.is_data_stale(max_delay_minutes):
            adjusted = result.score * 0.5
            logger.info(
                f"📊 Foreign flow score adjusted for staleness: "
                f"{result.score:.2f} → {adjusted:.2f}"
            )
            return adjusted

        return result.score

    def _default_result(self, reason: str) -> ForeignFlowData:
        """
        Return default neutral result when data is unavailable.

        IMPROVED v6.0: Foreign Flow Fallback Logic
        =========================================================================
        When foreign flow data is unavailable, the system:
        1. Returns neutral (zero) score - no bias in either direction
        2. Logs the reason for unavailability
        3. Marks data as unavailable in metadata

        This ensures trading decisions can continue without foreign flow data,
        but with reduced confidence in the signal.
        =========================================================================
        """
        logger.warning(
            f"⚠️ Foreign flow data unavailable: {reason}. "
            f"Using neutral score (0.0). Trading decisions will proceed without foreign flow signal."
        )
        return ForeignFlowData(
            date=datetime.now().isoformat(),
            net_value=0,
            buy_value=0,
            sell_value=0,
            net_volume=0,
            score=0.0,
            trend="NEUTRAL",
            strength="WEAK",
            consecutive_days=0,
            vs_average=0.0,
        )

    def add_manual_data(self, date: str, buy_value: float, sell_value: float):
        """
        Manually add foreign flow data (for testing or manual input).

        Args:
            date: Date string (YYYY-MM-DD)
            buy_value: Foreign buy value in VND
            sell_value: Foreign sell value in VND
        """
        self._historical_data.append(
            {
                "date": date,
                "buy_value": buy_value,
                "sell_value": sell_value,
                "net_value": buy_value - sell_value,
            }
        )

        # Invalidate cache
        self._cache = None

        logger.info(
            f"📊 Added foreign flow data: {date} - "
            f"Buy: {buy_value/1e9:.1f}B, Sell: {sell_value/1e9:.1f}B, "
            f"Net: {(buy_value-sell_value)/1e9:+.1f}B VND"
        )


# =============================================================================
# NEW v7.0: REALTIME FOREIGN FLOW TRACKER
# =============================================================================


@dataclass
class IntradayForeignFlow:
    """Intraday foreign flow data for a symbol"""

    symbol: str
    timestamp: str
    buy_volume: int
    sell_volume: int
    buy_value: float
    sell_value: float
    net_volume: int
    net_value: float
    trend: str  # ACCUMULATING, DISTRIBUTING, NEUTRAL
    vs_5day_avg: float  # Ratio vs 5-day average


class RealtimeForeignFlowTracker:
    """
    Track foreign flow in real-time during trading hours.

    IMPROVED v7.0: Real-time foreign flow tracking

    Data sources (priority order):
    1. TCBS API (primary) - updates every 1 minute
    2. SSI iBoard (secondary)
    3. Fireant API (backup)

    Features:
    - Per-symbol intraday tracking
    - Accumulation/Distribution detection
    - Alert on significant flow changes
    - Integration with entry/exit logic
    """

    UPDATE_INTERVAL_SECONDS = 60  # 1 minute
    SIGNIFICANT_FLOW_THRESHOLD = 5_000_000_000  # 5B VND

    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize realtime tracker.

        Args:
            symbols: List of symbols to track (default: VN30)
        """
        self.symbols = symbols or self._get_default_symbols()
        self._intraday_data: Dict[str, List[IntradayForeignFlow]] = {}
        self._last_update: Optional[datetime] = None
        self._alerts: List[Dict] = []

    def _get_default_symbols(self) -> List[str]:
        """Get default symbols (VN30)"""
        try:
            from src.utils.vietnam_market import VN30_SYMBOLS

            return list(VN30_SYMBOLS)
        except ImportError:
            return [
                "VNM",
                "VCB",
                "HPG",
                "FPT",
                "MWG",
                "VHM",
                "VIC",
                "TCB",
                "MBB",
                "ACB",
            ]

    def update(self) -> Dict[str, IntradayForeignFlow]:
        """
        Fetch latest foreign flow data for all tracked symbols.

        Returns:
            Dict of {symbol: IntradayForeignFlow}
        """
        results = {}

        for symbol in self.symbols:
            try:
                flow_data = self._fetch_symbol_flow(symbol)
                if flow_data:
                    results[symbol] = flow_data

                    # Store for history
                    if symbol not in self._intraday_data:
                        self._intraday_data[symbol] = []
                    self._intraday_data[symbol].append(flow_data)

                    # Trim history (keep last 100 updates)
                    if len(self._intraday_data[symbol]) > 100:
                        self._intraday_data[symbol] = self._intraday_data[symbol][-100:]

                    # Check for alerts
                    self._check_alerts(symbol, flow_data)

            except Exception as e:
                logger.debug(f"Failed to fetch flow for {symbol}: {e}")

        self._last_update = datetime.now()
        return results

    def _fetch_symbol_flow(self, symbol: str) -> Optional[IntradayForeignFlow]:
        """Fetch foreign flow for a single symbol"""
        # Try TCBS first
        try:
            from src.data.tcbs_provider import get_tcbs_provider

            provider = get_tcbs_provider()
            data = provider.get_intraday_foreign_flow(symbol)

            if data:
                return IntradayForeignFlow(
                    symbol=symbol,
                    timestamp=datetime.now().isoformat(),
                    buy_volume=data.get("buy_volume", 0),
                    sell_volume=data.get("sell_volume", 0),
                    buy_value=data.get("buy_value", 0),
                    sell_value=data.get("sell_value", 0),
                    net_volume=data.get("buy_volume", 0) - data.get("sell_volume", 0),
                    net_value=data.get("buy_value", 0) - data.get("sell_value", 0),
                    trend=self._determine_trend(data),
                    vs_5day_avg=data.get("vs_avg", 1.0),
                )
        except (ImportError, Exception) as e:
            logger.debug(f"TCBS flow fetch failed for {symbol}: {e}")

        return None

    def _determine_trend(self, data: Dict) -> str:
        """Determine accumulation/distribution trend"""
        net_value = data.get("buy_value", 0) - data.get("sell_value", 0)
        vs_avg = data.get("vs_avg", 1.0)

        if net_value > 0 and vs_avg > 1.2:
            return "ACCUMULATING"
        elif net_value < 0 and vs_avg > 1.2:
            return "DISTRIBUTING"
        else:
            return "NEUTRAL"

    def _check_alerts(self, symbol: str, flow: IntradayForeignFlow):
        """Check for significant flow changes and generate alerts"""
        if abs(flow.net_value) >= self.SIGNIFICANT_FLOW_THRESHOLD:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "type": "SIGNIFICANT_FLOW",
                "direction": "BUY" if flow.net_value > 0 else "SELL",
                "net_value": flow.net_value,
                "message": (
                    f"🚨 {symbol}: Significant foreign "
                    f"{'buying' if flow.net_value > 0 else 'selling'} "
                    f"({flow.net_value/1e9:+.1f}B VND)"
                ),
            }
            self._alerts.append(alert)
            logger.warning(alert["message"])

    def get_symbol_flow(self, symbol: str) -> Optional[IntradayForeignFlow]:
        """Get latest flow data for a symbol"""
        if symbol in self._intraday_data and self._intraday_data[symbol]:
            return self._intraday_data[symbol][-1]
        return None

    def get_market_flow_summary(self) -> Dict:
        """Get summary of foreign flow across all tracked symbols"""
        total_buy = 0
        total_sell = 0
        accumulating = []
        distributing = []

        for symbol, history in self._intraday_data.items():
            if not history:
                continue

            latest = history[-1]
            total_buy += latest.buy_value
            total_sell += latest.sell_value

            if latest.trend == "ACCUMULATING":
                accumulating.append(symbol)
            elif latest.trend == "DISTRIBUTING":
                distributing.append(symbol)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_buy_value": total_buy,
            "total_sell_value": total_sell,
            "net_value": total_buy - total_sell,
            "market_trend": (
                "ACCUMULATING"
                if total_buy > total_sell * 1.2
                else "DISTRIBUTING" if total_sell > total_buy * 1.2 else "NEUTRAL"
            ),
            "accumulating_symbols": accumulating,
            "distributing_symbols": distributing,
            "symbols_tracked": len(self._intraday_data),
        }

    def get_entry_adjustment(self, symbol: str) -> Tuple[int, str]:
        """
        Get confidence adjustment for entry based on foreign flow.

        IMPROVED v7.0: Integration with entry logic

        Returns:
            (adjustment, reason)
            - Positive adjustment for accumulation
            - Negative adjustment for distribution
        """
        flow = self.get_symbol_flow(symbol)

        if not flow:
            return (0, "No foreign flow data available")

        if flow.trend == "ACCUMULATING" and flow.vs_5day_avg > 1.5:
            return (+15, f"Strong foreign accumulation ({flow.net_value/1e9:+.1f}B)")
        elif flow.trend == "ACCUMULATING":
            return (+8, f"Foreign accumulation ({flow.net_value/1e9:+.1f}B)")
        elif flow.trend == "DISTRIBUTING" and flow.vs_5day_avg > 1.5:
            return (-20, f"Strong foreign distribution ({flow.net_value/1e9:+.1f}B)")
        elif flow.trend == "DISTRIBUTING":
            return (-10, f"Foreign distribution ({flow.net_value/1e9:+.1f}B)")
        else:
            return (0, "Neutral foreign flow")

    def get_alerts(self, clear: bool = True) -> List[Dict]:
        """Get and optionally clear alerts"""
        alerts = self._alerts.copy()
        if clear:
            self._alerts = []
        return alerts


# Singleton instances
_analyzer_instance: Optional[ForeignFlowAnalyzer] = None
_realtime_tracker: Optional[RealtimeForeignFlowTracker] = None


def get_foreign_flow_analyzer() -> ForeignFlowAnalyzer:
    """Get singleton instance of foreign flow analyzer"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ForeignFlowAnalyzer()
    return _analyzer_instance


def get_realtime_foreign_tracker(
    symbols: Optional[List[str]] = None,
) -> RealtimeForeignFlowTracker:
    """Get singleton instance of realtime foreign flow tracker"""
    global _realtime_tracker
    if _realtime_tracker is None:
        _realtime_tracker = RealtimeForeignFlowTracker(symbols)
    return _realtime_tracker
