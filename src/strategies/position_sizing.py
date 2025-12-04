"""
Enhanced Position Sizing với Kelly Criterion và Portfolio Risk
Version 4.0 - Refactored for production quality

Features:
- Kelly Criterion (half-Kelly for safety)
- Portfolio-level risk limits
- Correlation-based adjustments with LRU cache
- Sector exposure limits
- Market regime awareness
- Circuit breaker integration
- Vietnam market compliance (lot size 100)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Tuple

import pandas as pd

from src.config.constants import (
    CORRELATION_LOOKBACK_DAYS,
    DEFAULT_MAX_POSITION_SIZE,
    DEFAULT_MAX_SECTOR_EXPOSURE,
    DEFAULT_MAX_TOTAL_EXPOSURE,
    DEFAULT_MIN_POSITION_SIZE,
    DEFAULT_RISK_PER_TRADE,
    DEFAULT_TOTAL_CAPITAL,
    MAX_CORRELATION,
    VIETNAM_LOT_SIZE,
)
from src.config.exceptions import RiskManagementError

if TYPE_CHECKING:
    from src.risk.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Position Sizing Specific
# =============================================================================


class PositionSizingConstants:
    """Centralized constants for position sizing calculations."""

    # Risk thresholds
    MIN_RISK_PERCENT: float = 0.01  # 1% minimum risk per share
    DEFAULT_RISK_PERCENT: float = 0.02  # 2% default risk if stop too tight

    # Kelly Criterion
    MAX_KELLY_PERCENT: float = 0.25  # Max 25% of capital via Kelly
    DEFAULT_KELLY_FRACTION: float = 0.5  # Half-Kelly for safety
    MIN_KELLY_FALLBACK: float = 0.01  # 1% minimum for negative Kelly (v2.0 behavior)

    # Correlation
    HIGH_CORRELATION_THRESHOLD: float = 0.70
    MEDIUM_CORRELATION_THRESHOLD: float = 0.50
    HIGH_CORRELATION_ADJUSTMENT: float = 0.50  # Reduce 50%
    MEDIUM_CORRELATION_ADJUSTMENT: float = 0.75  # Reduce 25%

    # Sector limits
    SECTOR_HIGH_COUNT: int = 3  # 3+ positions = high concentration
    SECTOR_MEDIUM_COUNT: int = 2  # 2 positions = medium concentration
    SECTOR_HIGH_ADJUSTMENT: float = 0.70  # Reduce 30%
    SECTOR_MEDIUM_ADJUSTMENT: float = 0.85  # Reduce 15%

    # DCA levels - IMPROVED v4.2 for Vietnam Market
    # Vietnam market characteristics:
    # - ±7% daily price limit means 1-3% DCA levels can hit within same day
    # - Transaction cost ~1.48% round trip reduces DCA effectiveness
    # - Wider levels (2%, 4%, 6%) provide better cost-adjusted entries
    # - Each DCA level should exceed transaction cost to be profitable
    #
    # RECOMMENDATION: Consider disabling DCA for VN market due to:
    # 1. High transaction costs (1.48% round trip)
    # 2. T+2 settlement ties up capital
    # 3. Narrow DCA levels get hit too quickly in volatile market
    DCA_LEVEL_1_PERCENT: float = 0.50  # 50% at first level
    DCA_LEVEL_2_PERCENT: float = 0.30  # 30% at second level
    DCA_LEVEL_3_PERCENT: float = 0.20  # 20% at third level
    DCA_LEVEL_1_DISCOUNT: float = 0.98  # WIDENED: 2% below entry (was 1%)
    DCA_LEVEL_2_DISCOUNT: float = 0.96  # WIDENED: 4% below entry (was 2%)
    DCA_LEVEL_3_DISCOUNT: float = 0.94  # WIDENED: 6% below entry (was 3%)

    # DCA configuration flags
    DCA_ENABLED: bool = True  # Set False to disable DCA for VN market
    DCA_MIN_PROFIT_THRESHOLD: float = 0.02  # Min 2% expected profit after costs

    # Cache settings
    CORRELATION_CACHE_TTL: int = 3600  # 1 hour
    CORRELATION_CACHE_MAXSIZE: int = 500

    # Risk multiplier bounds - DOCUMENTED v4.2
    # These bounds control position size scaling based on signal quality
    #
    # MIN_RISK_MULTIPLIER = 0.5 rationale:
    # - Even weak signals get 50% of base position
    # - Prevents over-reduction that makes positions too small
    # - 50% of 1.5% risk = 0.75% risk per trade (still meaningful)
    # - Allows participation in uncertain markets with reduced exposure
    #
    # MAX_RISK_MULTIPLIER = 1.2 rationale:
    # - Strong signals get max 20% boost over base position
    # - Conservative cap prevents overconfidence in any single trade
    # - 120% of 1.5% risk = 1.8% risk per trade (within 2% guideline)
    # - Balances conviction with risk management
    #
    # Combined with Kelly Criterion, actual position sizes are further
    # constrained by win rate and risk/reward statistics.
    MIN_RISK_MULTIPLIER: float = 0.5
    MAX_RISK_MULTIPLIER: float = 1.2

    # Circuit breaker
    CAUTION_MODE_MULTIPLIER: float = 0.5  # Reduce 50% in caution mode


# =============================================================================
# PROTOCOLS - Dependency Injection Interfaces
# =============================================================================


class DataLoaderProtocol(Protocol):
    """Protocol for data loading dependency."""

    def __call__(self, symbol: str, lookback: int = 60) -> Optional[pd.DataFrame]: ...


class RegimeDetectorProtocol(Protocol):
    """Protocol for market regime detection."""

    def __call__(self, df: pd.DataFrame) -> object: ...


class CircuitBreakerProtocol(Protocol):
    """Protocol for circuit breaker."""

    def is_caution_mode(self) -> bool: ...


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class EnhancedPositionSize:
    """Container cho kết quả position sizing với Kelly."""

    shares: int
    value: float
    risk_amount: float
    risk_percent: float
    max_loss: float
    position_percent: float
    kelly_percent: float
    recommended_entries: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adjustments: Dict[str, float] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if position is valid for trading."""
        return self.shares > 0 and self.shares >= VIETNAM_LOT_SIZE


@dataclass
class MarketRegimeInfo:
    """Structured market regime information."""

    regime: str = "SIDEWAYS"
    confidence: float = 50.0
    tradeable: bool = True
    description: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "MarketRegimeInfo":
        """Create from dictionary."""
        if not data:
            return cls()
        return cls(
            regime=data.get("regime", "SIDEWAYS"),
            confidence=data.get("confidence", 50.0),
            tradeable=data.get("tradeable", True),
            description=data.get("description", ""),
        )


# =============================================================================
# CORRELATION CACHE - Thread-safe LRU Cache
# =============================================================================


class CorrelationCache:
    """Thread-safe LRU cache for correlation values."""

    def __init__(
        self,
        ttl: int = PositionSizingConstants.CORRELATION_CACHE_TTL,
        maxsize: int = PositionSizingConstants.CORRELATION_CACHE_MAXSIZE,
    ):
        self._cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def _make_key(self, symbol1: str, symbol2: str) -> Tuple[str, str]:
        """Create order-independent cache key."""
        return tuple(sorted([symbol1, symbol2]))

    def get(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Get cached correlation if valid."""
        key = self._make_key(symbol1, symbol2)

        with self._lock:
            if key in self._cache:
                corr, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    return corr
                # Expired - remove
                del self._cache[key]

            self._misses += 1
            return None

    def set(self, symbol1: str, symbol2: str, correlation: float) -> None:
        """Store correlation in cache."""
        key = self._make_key(symbol1, symbol2)

        with self._lock:
            self._cache[key] = (correlation, time.time())
            self._prune_if_needed()

    def _prune_if_needed(self) -> None:
        """Prune cache if over maxsize (must hold lock)."""
        if len(self._cache) <= self._maxsize:
            return

        current_time = time.time()

        # Remove expired first
        expired = [k for k, (_, ts) in self._cache.items() if current_time - ts > self._ttl]
        for k in expired:
            del self._cache[k]

        # If still over, remove oldest
        while len(self._cache) > self._maxsize:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# =============================================================================
# MAIN CLASS - EnhancedPositionSizer
# =============================================================================


class EnhancedPositionSizer:
    """
    Enhanced Position Sizing với:
    1. Kelly Criterion (half-Kelly for safety)
    2. Portfolio-level risk limits
    3. Correlation-based adjustments
    4. Sector exposure limits
    5. Win rate based sizing
    6. Market regime awareness
    7. Circuit breaker integration
    """

    def __init__(
        self,
        total_capital: float = DEFAULT_TOTAL_CAPITAL,
        max_risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
        max_position_size: float = DEFAULT_MAX_POSITION_SIZE,
        min_position_size: float = DEFAULT_MIN_POSITION_SIZE,
        max_total_exposure: float = DEFAULT_MAX_TOTAL_EXPOSURE,
        max_portfolio_risk: float = 0.15,
        max_sector_exposure: float = DEFAULT_MAX_SECTOR_EXPOSURE,
        use_kelly: bool = True,
        kelly_fraction: float = PositionSizingConstants.DEFAULT_KELLY_FRACTION,
        max_correlation_threshold: float = MAX_CORRELATION,
        volatility_adjustment: bool = True,
        # Dependency injection
        data_loader: Optional[DataLoaderProtocol] = None,
        regime_detector: Optional[RegimeDetectorProtocol] = None,
        circuit_breaker: Optional[CircuitBreakerProtocol] = None,
    ):
        # Configuration
        self.total_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_total_exposure = max_total_exposure
        self.max_portfolio_risk = max_portfolio_risk
        self.max_sector_exposure = max_sector_exposure
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction
        self.max_correlation_threshold = max_correlation_threshold
        self.volatility_adjustment = volatility_adjustment

        # Dependencies (lazy-loaded if not provided)
        self._data_loader = data_loader
        self._regime_detector = regime_detector
        self._circuit_breaker = circuit_breaker

        # State tracking
        self.current_positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.sector_exposure: Dict[str, float] = {}

        # Cache
        self._correlation_cache = CorrelationCache()

    # =========================================================================
    # DEPENDENCY GETTERS (Lazy Loading)
    # =========================================================================

    def _get_data_loader(self) -> DataLoaderProtocol:
        """Get data loader, lazy-loading if needed."""
        if self._data_loader is None:
            from src.data.loader import load_data

            self._data_loader = load_data
        return self._data_loader

    def _get_circuit_breaker(self) -> Optional[CircuitBreakerProtocol]:
        """Get circuit breaker, lazy-loading if needed."""
        if self._circuit_breaker is None:
            try:
                from src.risk.circuit_breaker import get_circuit_breaker

                self._circuit_breaker = get_circuit_breaker()
            except (ImportError, Exception) as e:
                logger.debug(f"Circuit breaker not available: {e}")
                return None
        return self._circuit_breaker

    def _detect_market_regime(self) -> Optional[MarketRegimeInfo]:
        """Auto-detect market regime from VNINDEX."""
        try:
            from src.data.vnindex_cache import get_cached_vnindex
            from src.market.regime_detector import detect_regime

            vnindex_df = get_cached_vnindex(lookback=250)
            if vnindex_df is None or vnindex_df.empty:
                logger.warning("Could not load VNINDEX for regime detection")
                return None

            regime_obj = detect_regime(vnindex_df)
            regime_info = MarketRegimeInfo(
                regime=regime_obj.regime,
                confidence=regime_obj.confidence,
                tradeable=regime_obj.tradeable,
                description=regime_obj.description,
            )

            logger.info(
                f"🔍 Auto-detected regime: {regime_info.regime} "
                f"(confidence: {regime_info.confidence:.1f}%)"
            )
            return regime_info

        except Exception as e:
            logger.warning(f"Auto regime detection failed: {e}")
            return None

    # =========================================================================
    # MAIN CALCULATION METHOD
    # =========================================================================

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: int,
        signal_strength: str = "MODERATE",
        market_regime: Optional[Dict] = None,
        sector: Optional[str] = None,
        portfolio_risk: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        auto_detect_regime: bool = True,
    ) -> EnhancedPositionSize:
        """
        Calculate position size với Kelly Criterion và portfolio context.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit target
            confidence: Signal confidence (0-100)
            signal_strength: VERY_STRONG, STRONG, MODERATE, WEAK, VERY_WEAK
            market_regime: Market regime info (auto-detected if None)
            sector: Stock sector for exposure tracking
            portfolio_risk: Current portfolio risk percentage
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average win/loss ratio
            auto_detect_regime: Auto-detect regime if not provided

        Returns:
            EnhancedPositionSize with calculated values
        """
        warnings: List[str] = []
        adjustments: Dict[str, float] = {}

        # Parse market regime
        regime_info = self._resolve_market_regime(
            market_regime, auto_detect_regime, adjustments, warnings
        )

        # Pre-trade validations
        self._validate_portfolio_risk(portfolio_risk)
        self._validate_sector_exposure(sector)

        # Check available capital
        available_capital = self._get_available_capital()
        if available_capital <= 0:
            return self._zero_position("Exposure limit reached", warnings)

        # Calculate risk per share with protection
        risk_per_share = self._calculate_risk_per_share(entry_price, stop_loss, warnings)
        if risk_per_share <= 0:
            return self._zero_position("Invalid stop loss", warnings)

        # Calculate base shares using multiple methods
        base_shares = self._calculate_base_shares(
            entry_price=entry_price,
            risk_per_share=risk_per_share,
            confidence=confidence,
            signal_strength=signal_strength,
            regime_info=regime_info,
            win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss_ratio,
            adjustments=adjustments,
        )

        # Apply portfolio adjustments
        adjusted_shares = self._apply_portfolio_adjustments(
            base_shares=base_shares,
            symbol=symbol,
            sector=sector,
            portfolio_risk=portfolio_risk,
            adjustments=adjustments,
            warnings=warnings,
        )

        # Enforce limits and round to lot size
        final_shares = self._enforce_limits(
            shares=adjusted_shares,
            entry_price=entry_price,
            available_capital=available_capital,
            risk_per_share=risk_per_share,
            warnings=warnings,
        )

        if final_shares <= 0:
            return self._zero_position("Position size = 0 after calculations", warnings)

        # Build result
        return self._build_result(
            shares=final_shares,
            entry_price=entry_price,
            risk_per_share=risk_per_share,
            kelly_percent=adjustments.get("kelly", 0.0),
            warnings=warnings,
            adjustments=adjustments,
        )

    # =========================================================================
    # PRIVATE HELPER METHODS - Validation
    # =========================================================================

    def _resolve_market_regime(
        self,
        market_regime: Optional[Dict],
        auto_detect: bool,
        adjustments: Dict[str, float],
        warnings: List[str],
    ) -> MarketRegimeInfo:
        """Resolve market regime from input or auto-detection."""
        if market_regime:
            return MarketRegimeInfo.from_dict(market_regime)

        if auto_detect:
            detected = self._detect_market_regime()
            if detected:
                adjustments["regime_auto_detected"] = 1.0
                adjustments["regime"] = hash(detected.regime) % 100  # For tracking

                if not detected.tradeable:
                    warnings.append(f"⚠️ Market regime {detected.regime} is NOT TRADEABLE")
                return detected

        return MarketRegimeInfo()

    def _validate_portfolio_risk(self, portfolio_risk: Optional[float]) -> None:
        """Validate portfolio risk is within limits."""
        if portfolio_risk is not None and portfolio_risk >= self.max_portfolio_risk:
            raise RiskManagementError(
                f"Portfolio risk ({portfolio_risk*100:.1f}%) exceeds limit "
                f"({self.max_portfolio_risk*100:.1f}%)",
                context={
                    "portfolio_risk": portfolio_risk,
                    "limit": self.max_portfolio_risk,
                },
            )

    def _validate_sector_exposure(self, sector: Optional[str]) -> None:
        """Validate sector exposure is within limits."""
        if not sector:
            return

        sector_exp = self._get_sector_exposure(sector)
        if sector_exp >= self.max_sector_exposure:
            raise RiskManagementError(
                f"Sector {sector} exposure ({sector_exp*100:.1f}%) exceeds limit "
                f"({self.max_sector_exposure*100:.1f}%)",
                context={
                    "sector": sector,
                    "exposure": sector_exp,
                    "limit": self.max_sector_exposure,
                },
            )

    def _get_available_capital(self) -> float:
        """
        Calculate available capital for new positions.

        IMPROVEMENT: Integrates T+2 settlement tracking for Vietnam market.
        Only settled cash is available for new purchases.
        """
        current_exposure = self._calculate_current_exposure()
        max_exposure_value = self.total_capital * self.max_total_exposure

        # Calculate base available capital
        base_available = max_exposure_value - current_exposure

        # IMPROVEMENT: Account for T+2 settlement
        # Only use settled cash for new positions
        try:
            from src.portfolio.settlement import get_settlement_tracker

            settlement = get_settlement_tracker()
            cash_info = settlement.get_available_cash(base_available)
            settled_available = cash_info.get("available_cash", base_available)

            if settled_available < base_available:
                pending = base_available - settled_available
                logger.debug(
                    f"💰 Available capital adjusted for T+2 settlement: "
                    f"{base_available:,.0f} → {settled_available:,.0f} "
                    f"(pending: {pending:,.0f})"
                )

            return settled_available

        except ImportError:
            logger.debug("Settlement tracker not available, using base calculation")
            return base_available
        except Exception as e:
            logger.warning(f"Settlement check failed: {e}, using base calculation")
            return base_available

    def _calculate_risk_per_share(
        self,
        entry_price: float,
        stop_loss: float,
        warnings: List[str],
    ) -> float:
        """Calculate risk per share with minimum enforcement."""
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return 0.0

        # Enforce minimum risk (prevent division by tiny numbers)
        min_risk = entry_price * PositionSizingConstants.MIN_RISK_PERCENT
        if risk_per_share < min_risk:
            enforced_risk = entry_price * PositionSizingConstants.DEFAULT_RISK_PERCENT
            logger.warning(
                f"⚠️ Risk per share too small ({risk_per_share:.0f}). "
                f"Enforcing {PositionSizingConstants.DEFAULT_RISK_PERCENT*100:.0f}% "
                f"minimum: {enforced_risk:.0f}"
            )
            warnings.append(
                f"Stop loss too tight, adjusted to "
                f"{PositionSizingConstants.DEFAULT_RISK_PERCENT*100:.0f}% risk"
            )
            return enforced_risk

        return risk_per_share

    # =========================================================================
    # PRIVATE HELPER METHODS - Position Calculation
    # =========================================================================

    def _calculate_base_shares(
        self,
        entry_price: float,
        risk_per_share: float,
        confidence: int,
        signal_strength: str,
        regime_info: MarketRegimeInfo,
        win_rate: Optional[float],
        avg_win_loss_ratio: Optional[float],
        adjustments: Dict[str, float],
    ) -> int:
        """Calculate base shares using risk-based and Kelly methods."""

        # Method 1: Risk-based sizing
        base_risk_amount = self.total_capital * self.max_risk_per_trade
        risk_multiplier = self._calculate_risk_multiplier(confidence, signal_strength, regime_info)
        adjustments["risk_multiplier"] = risk_multiplier

        adjusted_risk_amount = base_risk_amount * risk_multiplier
        shares_by_risk = int(adjusted_risk_amount / risk_per_share)

        # Method 2: Kelly Criterion (if data available)
        shares_by_kelly = 0
        kelly_percent = 0.0

        if self.use_kelly and win_rate and avg_win_loss_ratio:
            kelly_percent = self._calculate_kelly(win_rate, avg_win_loss_ratio)
            adjustments["kelly"] = kelly_percent

            if kelly_percent > 0:
                kelly_capital = self.total_capital * kelly_percent
                shares_by_kelly = int(kelly_capital / entry_price)
                adjustments["kelly_shares"] = shares_by_kelly

        # Combine: Use minimum of both methods (conservative)
        if shares_by_kelly > 0:
            return min(shares_by_risk, shares_by_kelly)
        return shares_by_risk

    def _calculate_kelly(
        self,
        win_rate: float,
        avg_win_loss_ratio: float,
    ) -> float:
        """
        Calculate Kelly Criterion percentage.

        Formula: K = W - (1-W)/R
        Where: W = win rate, R = average win / average loss

        Returns half-Kelly for safety, clamped to reasonable range.
        """
        # Validation
        if avg_win_loss_ratio <= 0:
            logger.warning(
                f"⚠️ Invalid avg_win_loss_ratio: {avg_win_loss_ratio:.3f}. "
                "Using conservative sizing."
            )
            return 0.0

        if win_rate <= 0 or win_rate >= 1:
            logger.warning(f"⚠️ Invalid win_rate: {win_rate:.3f}. Using conservative sizing.")
            return 0.0

        if win_rate < 0.3:
            logger.warning(f"⚠️ Low win rate: {win_rate:.1%}. Review strategy.")

        # Calculate Kelly
        kelly = win_rate - ((1 - win_rate) / avg_win_loss_ratio)

        logger.debug(
            f"📊 Kelly: win_rate={win_rate:.1%}, " f"W/L={avg_win_loss_ratio:.2f}, raw={kelly:.1%}"
        )

        # Handle negative Kelly (strategy has negative expected value)
        if kelly < 0:
            logger.warning(
                f"⚠️ NEGATIVE Kelly ({kelly:.1%})! Strategy has negative EV. "
                f"Win rate: {win_rate:.1%}, W/L: {avg_win_loss_ratio:.2f}. "
                f"Returning minimum {PositionSizingConstants.MIN_KELLY_FALLBACK:.1%} fallback."
            )
            return PositionSizingConstants.MIN_KELLY_FALLBACK  # v2.0: Return minimum instead of 0

        # Apply half-Kelly for safety
        half_kelly = kelly * self.kelly_fraction

        if kelly > 0.5:
            logger.warning(f"⚠️ Very high Kelly ({kelly:.1%}). Clamping to 25%.")

        # Clamp to reasonable range
        final_kelly = max(0.0, min(half_kelly, PositionSizingConstants.MAX_KELLY_PERCENT))

        logger.info(
            f"✅ Kelly sizing: {final_kelly:.1%} "
            f"(win={win_rate:.1%}, W/L={avg_win_loss_ratio:.2f})"
        )

        return final_kelly

    def _calculate_risk_multiplier(
        self,
        confidence: int,
        signal_strength: str,
        regime_info: MarketRegimeInfo,
    ) -> float:
        """Calculate risk multiplier based on confidence, signal, and regime."""

        # Base multiplier from confidence
        if confidence >= 80:
            base = 1.1
        elif confidence >= 70:
            base = 1.0
        elif confidence >= 60:
            base = 0.8
        else:
            base = 0.6

        # Signal strength multiplier
        strength_multipliers = {
            "VERY_STRONG": 1.1,
            "STRONG": 1.0,
            "MODERATE": 0.9,
            "WEAK": 0.7,
            "VERY_WEAK": 0.5,
        }
        strength_mult = strength_multipliers.get(signal_strength, 0.9)

        # Market regime multiplier
        regime_mult = self._get_regime_multiplier(regime_info, confidence)

        # Combine and clamp
        combined = base * strength_mult * regime_mult
        return max(
            PositionSizingConstants.MIN_RISK_MULTIPLIER,
            min(combined, PositionSizingConstants.MAX_RISK_MULTIPLIER),
        )

    def _get_regime_multiplier(
        self,
        regime_info: Optional[MarketRegimeInfo],
        confidence: int,
    ) -> float:
        """Get position multiplier based on market regime."""
        # Handle None regime_info
        if regime_info is None:
            return 1.0  # Default multiplier when no regime info

        # Handle dict input (for backward compatibility with tests)
        if isinstance(regime_info, dict):
            regime = regime_info.get("regime", "SIDEWAYS")
            regime_confidence = regime_info.get("confidence", 50.0)
        else:
            regime = regime_info.regime
            regime_confidence = regime_info.confidence

        if regime == "BULL":
            return 1.1

        if regime == "BEAR":
            # Variable bear multiplier based on regime confidence
            if regime_confidence < 70:
                # Weak bear - less defensive
                bear_mult = 0.7
                logger.info(
                    f"📊 Weak bear ({regime_confidence:.0f}% conf) - "
                    f"{bear_mult:.1f}x multiplier"
                )
            else:
                # Strong bear - very defensive
                bear_mult = 0.5
                logger.warning(
                    f"🐻 Strong bear ({regime_confidence:.0f}% conf) - "
                    f"{bear_mult:.1f}x multiplier"
                )

            # High confidence signals get less reduction
            if confidence >= 80:
                bear_mult = min(bear_mult * 1.2, 0.8)
                logger.info(
                    f"✨ High confidence ({confidence}%) in bear - " f"adjusted to {bear_mult:.2f}x"
                )

            return bear_mult

        if regime == "HIGH_VOLATILITY":
            return 0.6

        # SIDEWAYS or unknown
        return 0.8

    # =========================================================================
    # PRIVATE HELPER METHODS - Portfolio Adjustments
    # =========================================================================

    def _apply_portfolio_adjustments(
        self,
        base_shares: int,
        symbol: str,
        sector: Optional[str],
        portfolio_risk: Optional[float],
        adjustments: Dict[str, float],
        warnings: List[str],
    ) -> int:
        """Apply portfolio-level adjustments to position size."""

        # 1. Portfolio risk adjustment
        portfolio_adj = 1.0
        if portfolio_risk is not None:
            remaining_risk = self.max_portfolio_risk - portfolio_risk
            portfolio_adj = min(1.0, remaining_risk / self.max_portfolio_risk)
            adjustments["portfolio_risk_adj"] = portfolio_adj

        # 2. Sector exposure adjustment
        sector_adj = 1.0
        if sector:
            sector_exp = self._get_sector_exposure(sector)
            remaining_sector = self.max_sector_exposure - sector_exp
            sector_adj = min(1.0, remaining_sector / self.max_sector_exposure)
            adjustments["sector_adj"] = sector_adj

        # 3. Correlation adjustment
        correlation_adj = self._calculate_correlation_adjustment(symbol, sector)
        adjustments["correlation_adj"] = correlation_adj

        # 4. Circuit breaker adjustment
        circuit_adj = self._apply_circuit_breaker_adjustment(adjustments, warnings)

        # Apply all adjustments
        total_adj = portfolio_adj * sector_adj * correlation_adj * circuit_adj
        return int(base_shares * total_adj)

    def _apply_circuit_breaker_adjustment(
        self,
        adjustments: Dict[str, float],
        warnings: List[str],
    ) -> float:
        """Apply circuit breaker caution mode adjustment."""
        circuit_breaker = self._get_circuit_breaker()

        if circuit_breaker and circuit_breaker.is_caution_mode():
            mult = PositionSizingConstants.CAUTION_MODE_MULTIPLIER
            adjustments["caution_mode_adj"] = mult
            warnings.append(
                f"⚠️ Circuit breaker caution mode: " f"Position reduced by {(1-mult)*100:.0f}%"
            )
            logger.warning("🚨 Circuit breaker caution mode active")
            return mult

        return 1.0

    def _calculate_correlation_adjustment(
        self,
        symbol: str,
        sector: Optional[str],
    ) -> float:
        """
        Calculate correlation-based position adjustment.

        Uses actual price correlation with existing positions.
        Falls back to sector-based if correlation calc fails.
        """
        if not self.current_positions:
            return 1.0

        # Try real correlation calculation
        correlations = []
        for pos_symbol in self.current_positions:
            if pos_symbol == symbol:
                continue

            corr = self._calculate_correlation(symbol, pos_symbol)
            if corr is not None:
                correlations.append(abs(corr))

        # Use correlation-based adjustment if we have data
        if correlations:
            avg_corr = sum(correlations) / len(correlations)
            logger.info(
                f"📊 Avg correlation for {symbol}: {avg_corr:.3f} "
                f"({len(correlations)} positions)"
            )

            if avg_corr > PositionSizingConstants.HIGH_CORRELATION_THRESHOLD:
                logger.warning(f"⚠️ High correlation ({avg_corr:.2f}) - reducing 50%")
                return PositionSizingConstants.HIGH_CORRELATION_ADJUSTMENT

            if avg_corr > PositionSizingConstants.MEDIUM_CORRELATION_THRESHOLD:
                logger.info(f"Medium correlation ({avg_corr:.2f}) - reducing 25%")
                return PositionSizingConstants.MEDIUM_CORRELATION_ADJUSTMENT

            return 1.0

        # Fallback: sector-based correlation
        return self._sector_based_correlation_adjustment(sector)

    def _sector_based_correlation_adjustment(self, sector: Optional[str]) -> float:
        """Fallback sector-based correlation adjustment."""
        if not sector:
            return 1.0

        same_sector_count = sum(
            1 for pos in self.current_positions.values() if pos.get("sector") == sector
        )

        if same_sector_count >= PositionSizingConstants.SECTOR_HIGH_COUNT:
            logger.warning(f"⚠️ {same_sector_count} positions in {sector} - reducing 30%")
            return PositionSizingConstants.SECTOR_HIGH_ADJUSTMENT

        if same_sector_count >= PositionSizingConstants.SECTOR_MEDIUM_COUNT:
            logger.info(f"{same_sector_count} positions in {sector} - reducing 15%")
            return PositionSizingConstants.SECTOR_MEDIUM_ADJUSTMENT

        return 1.0

    def _calculate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        days: int = CORRELATION_LOOKBACK_DAYS,
    ) -> Optional[float]:
        """
        Calculate correlation coefficient with caching.

        Returns correlation (-1 to 1) or None if calculation fails.
        """
        # Check cache first
        cached = self._correlation_cache.get(symbol1, symbol2)
        if cached is not None:
            logger.debug(f"📦 Cache hit: {symbol1}-{symbol2} = {cached:.3f}")
            return cached

        # Calculate correlation
        try:
            load_data = self._get_data_loader()

            df1 = load_data(symbol1, lookback=days)
            df2 = load_data(symbol2, lookback=days)

            if df1 is None or df2 is None or len(df1) < 10 or len(df2) < 10:
                logger.warning(f"Insufficient data for correlation: {symbol1}-{symbol2}")
                return None

            # Merge on date
            merged = pd.merge(
                df1[["date", "close"]],
                df2[["date", "close"]],
                on="date",
                suffixes=("_1", "_2"),
            )

            if len(merged) < 10:
                logger.warning(f"Insufficient overlap for {symbol1}-{symbol2}: {len(merged)}")
                return None

            # Calculate and cache
            corr = merged["close_1"].corr(merged["close_2"])

            if pd.isna(corr):
                return None

            self._correlation_cache.set(symbol1, symbol2, corr)
            logger.debug(f"📊 Calculated {symbol1}-{symbol2}: {corr:.3f}")

            return corr

        except Exception as e:
            logger.warning(f"Correlation calc failed {symbol1}-{symbol2}: {e}")
            return None

    # =========================================================================
    # PRIVATE HELPER METHODS - Limits & Finalization
    # =========================================================================

    def _enforce_limits(
        self,
        shares: int,
        entry_price: float,
        available_capital: float,
        risk_per_share: float,
        warnings: List[str],
    ) -> int:
        """Enforce position limits and round to lot size."""

        # Calculate limits
        max_by_capital = int((self.total_capital * self.max_position_size) / entry_price)
        max_by_available = int(available_capital / entry_price)
        min_shares = int((self.total_capital * self.min_position_size) / entry_price)

        # Apply limits
        final_shares = min(shares, max_by_capital, max_by_available)
        final_shares = max(final_shares, min_shares) if final_shares > 0 else 0

        if final_shares <= 0:
            return 0

        # Round to Vietnam lot size
        final_shares = (final_shares // VIETNAM_LOT_SIZE) * VIETNAM_LOT_SIZE

        # Enforce minimum lot
        if final_shares < VIETNAM_LOT_SIZE:
            logger.warning(
                f"⚠️ Position {final_shares} < lot size {VIETNAM_LOT_SIZE}. " f"Adjusting to 1 lot."
            )
            final_shares = VIETNAM_LOT_SIZE

        # Validate risk limit
        final_shares = self._validate_risk_limit(
            final_shares, entry_price, risk_per_share, warnings
        )

        logger.debug(
            f"✅ Final: {final_shares} shares " f"({final_shares // VIETNAM_LOT_SIZE} lots)"
        )

        return final_shares

    def _validate_risk_limit(
        self,
        shares: int,
        entry_price: float,
        risk_per_share: float,
        warnings: List[str],
    ) -> int:
        """Ensure position doesn't exceed risk limit."""
        max_loss = shares * risk_per_share
        risk_percent = max_loss / self.total_capital

        if risk_percent > self.max_risk_per_trade:
            # Reduce to stay within limit
            max_safe_shares = int((self.total_capital * self.max_risk_per_trade) / risk_per_share)
            shares = min(shares, max_safe_shares)
            shares = max((shares // VIETNAM_LOT_SIZE) * VIETNAM_LOT_SIZE, VIETNAM_LOT_SIZE)
            warnings.append(f"Reduced shares to keep risk <= {self.max_risk_per_trade*100:.1f}%")

        return shares

    def _build_result(
        self,
        shares: int,
        entry_price: float,
        risk_per_share: float,
        kelly_percent: float,
        warnings: List[str],
        adjustments: Dict[str, float],
    ) -> EnhancedPositionSize:
        """Build final result object."""
        position_value = shares * entry_price
        position_percent = (position_value / self.total_capital) * 100
        max_loss = shares * risk_per_share
        risk_percent = (max_loss / self.total_capital) * 100

        # Add warnings for large positions
        if position_percent > self.max_position_size * 100 * 0.8:
            warnings.append(f"Large position: {position_percent:.1f}%")

        if risk_percent > self.max_risk_per_trade * 100 * 0.8:
            warnings.append(f"High risk: {risk_percent:.2f}%")

        return EnhancedPositionSize(
            shares=shares,
            value=position_value,
            risk_amount=max_loss,
            risk_percent=risk_percent,
            max_loss=max_loss,
            position_percent=position_percent,
            kelly_percent=kelly_percent * 100,
            recommended_entries=self._calculate_dca_entries(entry_price, shares),
            warnings=warnings,
            adjustments=adjustments,
        )

    # =========================================================================
    # PRIVATE HELPER METHODS - Utilities
    # =========================================================================

    def _get_sector_exposure(self, sector: str) -> float:
        """Get current sector exposure as percentage of capital."""
        if not sector or self.total_capital <= 0:
            return 0.0

        sector_value = sum(
            pos["shares"] * pos.get("current_price", 0)
            for pos in self.current_positions.values()
            if pos.get("sector") == sector
        )

        return sector_value / self.total_capital

    def _calculate_current_exposure(self) -> float:
        """Calculate current total exposure value."""
        return sum(
            pos["shares"] * pos.get("current_price", 0) for pos in self.current_positions.values()
        )

    def _calculate_dca_entries(
        self,
        base_price: float,
        total_shares: int,
    ) -> List[Dict]:
        """
        Calculate DCA entry levels for Vietnam market.

        IMPROVED v4.2:
        - Widened DCA levels (2%, 4%, 6%) to account for transaction costs
        - DCA can be disabled via DCA_ENABLED flag
        - Each level must exceed transaction cost threshold to be worthwhile

        Vietnam market considerations:
        - ±7% daily limit means narrow DCA levels hit too quickly
        - 1.48% round trip cost reduces DCA effectiveness
        - T+2 settlement ties up capital for each DCA entry
        """
        c = PositionSizingConstants

        # Check if DCA is enabled
        if not c.DCA_ENABLED:
            logger.info(
                "📊 DCA disabled for VN market (high transaction costs). "
                "Using single entry strategy."
            )
            return [
                {
                    "level": 1,
                    "price": round(base_price, -2),
                    "shares": total_shares,
                    "percent": 100,
                    "note": "Single entry (DCA disabled)",
                }
            ]

        def round_shares(pct: float) -> int:
            shares = int((total_shares * pct // VIETNAM_LOT_SIZE) * VIETNAM_LOT_SIZE)
            return max(shares, VIETNAM_LOT_SIZE) if shares > 0 else 0

        # Calculate DCA levels with widened discounts
        dca_entries = [
            {
                "level": 1,
                "price": round(base_price * c.DCA_LEVEL_1_DISCOUNT, -2),
                "shares": round_shares(c.DCA_LEVEL_1_PERCENT),
                "percent": int(c.DCA_LEVEL_1_PERCENT * 100),
                "discount": f"{(1 - c.DCA_LEVEL_1_DISCOUNT) * 100:.0f}%",
            },
            {
                "level": 2,
                "price": round(base_price * c.DCA_LEVEL_2_DISCOUNT, -2),
                "shares": round_shares(c.DCA_LEVEL_2_PERCENT),
                "percent": int(c.DCA_LEVEL_2_PERCENT * 100),
                "discount": f"{(1 - c.DCA_LEVEL_2_DISCOUNT) * 100:.0f}%",
            },
            {
                "level": 3,
                "price": round(base_price * c.DCA_LEVEL_3_DISCOUNT, -2),
                "shares": round_shares(c.DCA_LEVEL_3_PERCENT),
                "percent": int(c.DCA_LEVEL_3_PERCENT * 100),
                "discount": f"{(1 - c.DCA_LEVEL_3_DISCOUNT) * 100:.0f}%",
            },
        ]

        logger.debug(
            f"📊 DCA entries calculated: "
            f"L1={dca_entries[0]['price']:,.0f} ({dca_entries[0]['discount']}), "
            f"L2={dca_entries[1]['price']:,.0f} ({dca_entries[1]['discount']}), "
            f"L3={dca_entries[2]['price']:,.0f} ({dca_entries[2]['discount']})"
        )

        return dca_entries

    def _zero_position(
        self,
        reason: str,
        warnings: List[str],
    ) -> EnhancedPositionSize:
        """Return zero position with reason."""
        warnings.append(reason)
        return EnhancedPositionSize(
            shares=0,
            value=0,
            risk_amount=0,
            risk_percent=0,
            max_loss=0,
            position_percent=0,
            kelly_percent=0,
            recommended_entries=[],
            warnings=warnings,
            adjustments={},
        )

    # =========================================================================
    # PUBLIC METHODS - Position Management
    # =========================================================================

    def add_position(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        current_price: float,
        sector: Optional[str] = None,
    ) -> None:
        """Add a position to tracking."""
        self.current_positions[symbol] = {
            "shares": shares,
            "entry_price": entry_price,
            "current_price": current_price,
            "sector": sector,
        }

    def remove_position(self, symbol: str) -> None:
        """Remove a position from tracking."""
        self.current_positions.pop(symbol, None)

    def update_position_price(self, symbol: str, current_price: float) -> None:
        """Update current price for a position."""
        if symbol in self.current_positions:
            self.current_positions[symbol]["current_price"] = current_price

    def clear_positions(self) -> None:
        """Clear all tracked positions."""
        self.current_positions.clear()

    def get_portfolio_summary(self) -> Dict:
        """Get summary of current portfolio."""
        total_value = self._calculate_current_exposure()
        exposure_pct = (total_value / self.total_capital * 100) if self.total_capital > 0 else 0

        return {
            "total_capital": self.total_capital,
            "total_exposure": total_value,
            "exposure_percent": exposure_pct,
            "position_count": len(self.current_positions),
            "available_capital": self._get_available_capital(),
            "correlation_cache_hit_rate": self._correlation_cache.hit_rate,
        }
