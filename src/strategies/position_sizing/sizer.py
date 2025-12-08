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

Module Structure (v5.0):
- position_sizing/constants.py: Configuration constants
- position_sizing/protocols.py: DI interfaces
- position_sizing/models.py: Data classes
- position_sizing/cache.py: Correlation cache
- position_sizing/sizer.py: Main EnhancedPositionSizer (this file)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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

# Import from same package using relative imports
from .cache import CorrelationCache
from .constants import PositionSizingConstants
from .models import (
    EnhancedPositionSize,
    MarketRegimeInfo,
    PositionSize,
)
from .protocols import (
    CircuitBreakerProtocol,
    DataLoaderProtocol,
    RegimeDetectorProtocol,
)

if TYPE_CHECKING:
    from src.risk.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Re-export all for backward compatibility
__all__ = [
    "PositionSizingConstants",
    "DataLoaderProtocol",
    "RegimeDetectorProtocol",
    "CircuitBreakerProtocol",
    "EnhancedPositionSize",
    "MarketRegimeInfo",
    "PositionSize",
    "CorrelationCache",
    "EnhancedPositionSizer",
]


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

        # IMPROVED v5.0: Get avg daily value for liquidity tier
        avg_daily_value = self._get_avg_daily_value(symbol)
        if avg_daily_value:
            tier_name, tier_config = self._get_liquidity_tier(symbol, avg_daily_value)
            adjustments["liquidity_tier"] = tier_name
            adjustments["tier_max_position"] = tier_config["max_position_pct"]

        # Enforce limits and round to lot size
        final_shares = self._enforce_limits(
            shares=adjusted_shares,
            entry_price=entry_price,
            available_capital=available_capital,
            risk_per_share=risk_per_share,
            warnings=warnings,
            symbol=symbol,
            avg_daily_value=avg_daily_value,
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
        """
        Calculate base shares using risk-based and Kelly methods.

        IMPROVED v5.0:
        - 3-step calculation: Base size → Apply adjustments → Enforce limits
        - Regime-aware Kelly with cost adjustment
        - Liquidity tier-based position limits
        """

        # Method 1: Risk-based sizing
        base_risk_amount = self.total_capital * self.max_risk_per_trade
        risk_multiplier = self._calculate_risk_multiplier(confidence, signal_strength, regime_info)
        adjustments["risk_multiplier"] = risk_multiplier

        adjusted_risk_amount = base_risk_amount * risk_multiplier
        shares_by_risk = int(adjusted_risk_amount / risk_per_share)

        # Method 2: Kelly Criterion (if data available)
        # IMPROVED v5.0: Pass market regime for regime-aware Kelly
        shares_by_kelly = 0
        kelly_percent = 0.0

        if self.use_kelly and win_rate and avg_win_loss_ratio:
            # Extract regime string for Kelly calculation
            market_regime_str = None
            if regime_info:
                if isinstance(regime_info, dict):
                    market_regime_str = regime_info.get("regime")
                else:
                    market_regime_str = regime_info.regime

            kelly_percent = self._calculate_kelly(
                win_rate, avg_win_loss_ratio, market_regime=market_regime_str
            )
            adjustments["kelly"] = kelly_percent
            adjustments["kelly_regime"] = market_regime_str or "N/A"

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
        market_regime: Optional[str] = None,
    ) -> float:
        """
        Calculate Kelly Criterion percentage with regime-aware adjustment.

        IMPROVED v5.0:
        - Regime-aware Kelly fraction (BULL=0.5, BEAR=0.25, HIGH_VOL=0.125)
        - Transaction cost adjustment for Vietnam market (1.48%)
        - Cost-adjusted win/loss ratio for realistic sizing

        Formula: K = W - (1-W)/R_adjusted
        Where:
            W = win rate
            R_adjusted = (avg_win - cost) / (avg_loss + cost)

        Returns regime-adjusted Kelly, clamped to reasonable range.
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

        # IMPROVED v5.0: Adjust W/L ratio for transaction costs
        # Assume average win = avg_win_loss_ratio * average_loss
        # Cost reduces wins and increases losses
        cost_pct = PositionSizingConstants.VN_TRANSACTION_COST  # 1.48%

        # Cost-adjusted ratio: (win - cost) / (loss + cost)
        # If avg_win_loss_ratio = 2.0, and cost = 1.48%
        # Adjusted = (2.0 - 0.0148) / (1.0 + 0.0148) ≈ 1.96
        cost_adjusted_ratio = (avg_win_loss_ratio - cost_pct) / (1.0 + cost_pct)

        if cost_adjusted_ratio <= 0:
            logger.warning(
                f"⚠️ Cost-adjusted W/L ratio <= 0 ({cost_adjusted_ratio:.3f}). "
                f"Original: {avg_win_loss_ratio:.2f}, cost: {cost_pct:.2%}. "
                "Transaction costs exceed expected profit."
            )
            return PositionSizingConstants.MIN_KELLY_FALLBACK

        # Calculate Kelly with cost-adjusted ratio
        kelly = win_rate - ((1 - win_rate) / cost_adjusted_ratio)

        logger.debug(
            f"📊 Kelly: win_rate={win_rate:.1%}, "
            f"W/L={avg_win_loss_ratio:.2f} (cost-adj: {cost_adjusted_ratio:.2f}), "
            f"raw={kelly:.1%}"
        )

        # Handle negative Kelly (strategy has negative expected value)
        if kelly < 0:
            logger.warning(
                f"⚠️ NEGATIVE Kelly ({kelly:.1%})! Strategy has negative EV. "
                f"Win rate: {win_rate:.1%}, W/L: {avg_win_loss_ratio:.2f}. "
                f"Returning minimum {PositionSizingConstants.MIN_KELLY_FALLBACK:.1%} fallback."
            )
            return PositionSizingConstants.MIN_KELLY_FALLBACK

        # IMPROVED v5.0: Regime-aware Kelly fraction
        if market_regime:
            kelly_fraction = PositionSizingConstants.REGIME_KELLY_FRACTIONS.get(
                market_regime, self.kelly_fraction
            )
            logger.debug(f"📊 Regime-aware Kelly fraction: {kelly_fraction:.2f} ({market_regime})")
        else:
            kelly_fraction = self.kelly_fraction

        # IMPROVED v6.1: Performance-based Kelly adjustment
        # Reduce Kelly if recent performance is poor
        performance_factor = self._get_performance_based_kelly_factor(win_rate, avg_win_loss_ratio)
        kelly_fraction *= performance_factor

        if performance_factor < 1.0:
            logger.info(
                f"📉 Performance-based Kelly reduction: {performance_factor:.2f}x "
                f"(win_rate={win_rate:.1%}, W/L={avg_win_loss_ratio:.2f})"
            )

        # Apply regime-adjusted Kelly fraction
        adjusted_kelly = kelly * kelly_fraction

        if kelly > 0.5:
            logger.warning(f"⚠️ Very high Kelly ({kelly:.1%}). Clamping to 25%.")

        # Clamp to reasonable range
        final_kelly = max(0.0, min(adjusted_kelly, PositionSizingConstants.MAX_KELLY_PERCENT))

        logger.info(
            f"✅ Kelly sizing: {final_kelly:.1%} "
            f"(win={win_rate:.1%}, W/L={avg_win_loss_ratio:.2f}, "
            f"regime={market_regime or 'N/A'}, fraction={kelly_fraction:.2f})"
        )

        return final_kelly

    def _get_performance_based_kelly_factor(
        self,
        win_rate: float,
        avg_win_loss_ratio: float,
    ) -> float:
        """
        Get Kelly adjustment factor based on recent performance.

        IMPROVED v6.1: Performance-based Kelly adjustment for Vietnam market.

        Rationale:
        - If win rate < 40%, reduce Kelly by 30% (underperforming)
        - If W/L ratio < 1.5, reduce Kelly by 20% (poor R:R)
        - If both conditions, reduce by 44% (0.7 * 0.8)
        - Never increase Kelly based on performance (avoid overconfidence)

        Args:
            win_rate: Recent win rate (0-1)
            avg_win_loss_ratio: Recent average win/loss ratio

        Returns:
            Factor to multiply Kelly by (0.56 to 1.0)
        """
        factor = 1.0

        # Reduce if win rate is poor
        if win_rate < 0.40:
            factor *= 0.7  # -30%
            logger.debug(f"📉 Low win rate ({win_rate:.1%}): Kelly factor -30%")
        elif win_rate < 0.45:
            factor *= 0.85  # -15%
            logger.debug(f"📉 Below-average win rate ({win_rate:.1%}): Kelly factor -15%")

        # Reduce if R:R is poor
        if avg_win_loss_ratio < 1.5:
            factor *= 0.8  # -20%
            logger.debug(f"📉 Low W/L ratio ({avg_win_loss_ratio:.2f}): Kelly factor -20%")
        elif avg_win_loss_ratio < 1.8:
            factor *= 0.9  # -10%
            logger.debug(
                f"📉 Below-average W/L ratio ({avg_win_loss_ratio:.2f}): Kelly factor -10%"
            )

        # Minimum factor to prevent over-reduction
        return max(0.5, factor)

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

    def _get_avg_daily_value(self, symbol: str) -> Optional[float]:
        """
        Get average daily trading value for a symbol.

        IMPROVED v5.0: Used for liquidity tier determination.

        Args:
            symbol: Stock symbol

        Returns:
            Average daily value in VND, or None if unavailable
        """
        try:
            load_data = self._get_data_loader()
            df = load_data(symbol, lookback=20)

            if df is None or df.empty or len(df) < 5:
                logger.debug(f"Insufficient data for {symbol} liquidity calculation")
                return None

            if "volume" not in df.columns or "close" not in df.columns:
                return None

            avg_volume = df["volume"].tail(20).mean()
            avg_price = df["close"].tail(20).mean()
            avg_daily_value = avg_volume * avg_price

            logger.debug(f"📊 {symbol} avg daily value: {avg_daily_value/1e9:.2f}B VND")
            return avg_daily_value

        except Exception as e:
            logger.debug(f"Could not calculate avg daily value for {symbol}: {e}")
            return None

    def _get_liquidity_tier(
        self,
        symbol: str,
        avg_daily_value: float,
    ) -> Tuple[str, Dict]:
        """
        Determine liquidity tier for a symbol.

        IMPROVED v6.0: Enhanced liquidity risk management for Vietnam market
        =========================================================================
        4-tier system with additional critical liquidity handling:
        - VN30: Blue chips (highest liquidity) - 15% max position
        - LARGE_CAP: Large caps outside VN30 (> 5B VND) - 12% max position
        - MID_CAP: Mid caps (3-5B VND) - 10% max position
        - SMALL_CAP: Small caps (< 3B VND) - 6% max position
        - CRITICAL: Below 500M VND - 3% max position, 2% of daily volume limit

        Additional rules for illiquid stocks:
        - Position limited to 2% of average daily volume
        - 1.0% slippage applied (vs 0.4% for liquid stocks)
        - Exit recommendation: split into multiple orders if > 5% of daily volume
        =========================================================================

        Args:
            symbol: Stock symbol
            avg_daily_value: Average daily trading value in VND

        Returns:
            Tuple of (tier_name, tier_config)
        """
        from src.config.constants import VN30_SYMBOLS, VN_CRITICAL_LIQUIDITY_VALUE

        tiers = PositionSizingConstants.LIQUIDITY_TIERS

        # VN30 blue chips get highest tier
        if symbol.upper() in VN30_SYMBOLS:
            return ("VN30", tiers["VN30"])

        # IMPROVED v6.0: Critical liquidity check (< 500M VND)
        # These stocks have severe liquidity risk
        if avg_daily_value < VN_CRITICAL_LIQUIDITY_VALUE:
            logger.warning(
                f"⚠️ {symbol}: CRITICAL LIQUIDITY ({avg_daily_value/1e6:.0f}M VND < "
                f"{VN_CRITICAL_LIQUIDITY_VALUE/1e6:.0f}M threshold). "
                f"Position limited to 3% max, 2% of daily volume."
            )
            return (
                "CRITICAL",
                {
                    "max_position_pct": 0.03,  # 3% max position
                    "min_daily_value": 0,
                    "slippage": 0.015,  # 1.5% slippage for critical liquidity
                    "max_volume_pct": 0.02,  # Max 2% of daily volume
                },
            )

        # Tier by daily trading value
        if avg_daily_value >= tiers["LARGE_CAP"]["min_daily_value"]:
            return ("LARGE_CAP", tiers["LARGE_CAP"])
        elif avg_daily_value >= tiers["MID_CAP"]["min_daily_value"]:
            return ("MID_CAP", tiers["MID_CAP"])
        else:
            return ("SMALL_CAP", tiers["SMALL_CAP"])

    def check_exit_liquidity(
        self,
        symbol: str,
        shares: int,
        avg_daily_volume: float,
    ) -> Dict:
        """
        Check if exit position has liquidity risk.

        IMPROVED v6.0: Exit liquidity risk assessment
        =========================================================================
        When exiting a position, check if it's too large relative to daily volume.

        Rules:
        - Position > 5% of daily volume: Recommend splitting into multiple orders
        - Position > 10% of daily volume: High risk, may take multiple days to exit
        - Position > 20% of daily volume: Critical risk, significant price impact
        =========================================================================

        Args:
            symbol: Stock symbol
            shares: Number of shares to exit
            avg_daily_volume: Average daily volume in shares

        Returns:
            Dict with risk assessment and recommendations
        """
        if avg_daily_volume <= 0:
            return {
                "risk_level": "UNKNOWN",
                "volume_pct": 0,
                "recommendation": "Unable to assess - no volume data",
                "split_orders": False,
            }

        volume_pct = shares / avg_daily_volume

        if volume_pct > 0.20:
            return {
                "risk_level": "CRITICAL",
                "volume_pct": volume_pct,
                "recommendation": (
                    f"⚠️ CRITICAL: Position is {volume_pct:.1%} of daily volume. "
                    f"Recommend splitting into 4+ orders over multiple days. "
                    f"Expect significant price impact."
                ),
                "split_orders": True,
                "suggested_splits": max(4, int(volume_pct / 0.05)),
            }
        elif volume_pct > 0.10:
            return {
                "risk_level": "HIGH",
                "volume_pct": volume_pct,
                "recommendation": (
                    f"⚠️ HIGH: Position is {volume_pct:.1%} of daily volume. "
                    f"Recommend splitting into 2-3 orders. "
                    f"May take 1-2 days to fully exit."
                ),
                "split_orders": True,
                "suggested_splits": max(2, int(volume_pct / 0.05)),
            }
        elif volume_pct > 0.05:
            return {
                "risk_level": "MODERATE",
                "volume_pct": volume_pct,
                "recommendation": (
                    f"📊 MODERATE: Position is {volume_pct:.1%} of daily volume. "
                    f"Consider splitting into 2 orders for better execution."
                ),
                "split_orders": True,
                "suggested_splits": 2,
            }
        else:
            return {
                "risk_level": "LOW",
                "volume_pct": volume_pct,
                "recommendation": f"✅ Position is {volume_pct:.1%} of daily volume - safe to exit in single order.",
                "split_orders": False,
                "suggested_splits": 1,
            }

    def _enforce_limits(
        self,
        shares: int,
        entry_price: float,
        available_capital: float,
        risk_per_share: float,
        warnings: List[str],
        symbol: Optional[str] = None,
        avg_daily_value: Optional[float] = None,
    ) -> int:
        """
        Enforce position limits and round to lot size.

        IMPROVED v5.0: Liquidity tier-based position limits
        - VN30: 15% max position
        - LARGE_CAP: 12% max position
        - MID_CAP: 10% max position
        - SMALL_CAP: 6% max position
        """

        # IMPROVED v5.0: Get liquidity tier-based max position
        if symbol and avg_daily_value is not None:
            tier_name, tier_config = self._get_liquidity_tier(symbol, avg_daily_value)
            tier_max_position = tier_config["max_position_pct"]
            logger.debug(f"📊 Liquidity tier: {tier_name} → max position {tier_max_position:.0%}")
        else:
            tier_name = "DEFAULT"
            tier_max_position = self.max_position_size

        # Use the more restrictive of tier limit and configured limit
        effective_max_position = min(tier_max_position, self.max_position_size)

        # Calculate limits
        max_by_capital = int((self.total_capital * effective_max_position) / entry_price)
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

        # Add tier info to warnings if position was limited
        if shares > final_shares and tier_name != "DEFAULT":
            warnings.append(
                f"Position limited by {tier_name} tier (max {effective_max_position:.0%})"
            )

        logger.debug(
            f"✅ Final: {final_shares} shares "
            f"({final_shares // VIETNAM_LOT_SIZE} lots, tier={tier_name})"
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

    # =========================================================================
    # NEW v6.0: ODD-LOT TRADING INTEGRATION
    # =========================================================================

    def calculate_odd_lot_position(
        self,
        symbol: str,
        entry_price: float,
        target_value: float,
        expected_return_pct: float = 5.0,
    ) -> Dict:
        """
        Calculate position size for odd-lot trading (1-99 shares).

        IMPROVED v6.0: Odd-lot Trading Implementation
        =========================================================================
        Odd-lot trading in Vietnam market:
        - Enabled since 2021 for quantities 1-99 shares
        - Higher spread premium (0.5% wider)
        - Minimum commission applies (11,000 VND)
        - Useful for small portfolios or selling remaining shares
        =========================================================================

        Args:
            symbol: Stock symbol
            entry_price: Entry price per share
            target_value: Target position value in VND
            expected_return_pct: Expected return percentage

        Returns:
            Dict with odd-lot position details and cost analysis
        """
        from src.config.constants import (
            VN_ODD_LOT_ENABLED,
            VN_ODD_LOT_MIN_QTY,
            VN_ODD_LOT_MAX_QTY,
            VN_ODD_LOT_SPREAD_PREMIUM,
            VN_ODD_LOT_MIN_COMMISSION,
        )

        if not VN_ODD_LOT_ENABLED:
            return {
                "enabled": False,
                "reason": "Odd-lot trading is disabled",
                "recommendation": "Use standard lot (100 shares minimum)",
            }

        # Calculate shares
        raw_shares = int(target_value / entry_price)

        if raw_shares >= 100:
            return {
                "enabled": True,
                "is_odd_lot": False,
                "shares": (raw_shares // 100) * 100,
                "recommendation": "Use standard lot trading",
            }

        if raw_shares < VN_ODD_LOT_MIN_QTY:
            return {
                "enabled": True,
                "is_odd_lot": False,
                "shares": 0,
                "reason": "Position value too small for any shares",
            }

        # Calculate costs for odd-lot
        gross_value = raw_shares * entry_price
        commission = max(gross_value * 0.0025, VN_ODD_LOT_MIN_COMMISSION)  # 0.25% or min
        spread_cost = gross_value * VN_ODD_LOT_SPREAD_PREMIUM
        total_cost = commission + spread_cost
        cost_pct = (total_cost / gross_value) * 100 if gross_value > 0 else 0

        # Check if trade is worthwhile
        is_worthwhile = expected_return_pct > cost_pct

        result = {
            "enabled": True,
            "is_odd_lot": True,
            "shares": raw_shares,
            "gross_value": gross_value,
            "commission": commission,
            "spread_cost": spread_cost,
            "total_cost": total_cost,
            "cost_pct": cost_pct,
            "is_worthwhile": is_worthwhile,
            "net_return_pct": expected_return_pct - cost_pct,
        }

        if not is_worthwhile:
            result["warning"] = (
                f"⚠️ Odd-lot costs ({cost_pct:.2f}%) exceed expected return "
                f"({expected_return_pct:.2f}%). Consider larger position."
            )
            logger.warning(result["warning"])

        return result

    # =========================================================================
    # NEW v6.0: MARGIN TRADING INTEGRATION
    # =========================================================================

    def calculate_margin_position(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: int,
        use_margin: bool = True,
        margin_ratio: float = 0.50,
    ) -> Dict:
        """
        Calculate position size with margin trading integration.

        IMPROVED v6.0: Margin Trading Integration
        =========================================================================
        Vietnam margin trading rules:
        - Initial margin: 50% (can borrow up to 50% of position value)
        - Maintenance margin: 35%
        - Margin call: 30%
        - Force liquidation: 25%
        =========================================================================

        Args:
            symbol: Stock symbol
            entry_price: Entry price per share
            stop_loss: Stop loss price
            confidence: Signal confidence (0-100)
            use_margin: Whether to use margin
            margin_ratio: Margin ratio (default 50%)

        Returns:
            Dict with margin-adjusted position details
        """
        from src.config.constants import (
            VN_INITIAL_MARGIN,
            VN_MAINTENANCE_MARGIN,
            VN_MARGIN_WARNING_LEVEL,
            VN_MARGIN_CALL_LEVEL,
        )

        # Get margin manager
        try:
            from src.risk.margin_manager import MarginManager

            margin_mgr = MarginManager()
            account_state = margin_mgr.get_account_state()
        except ImportError:
            logger.warning("MarginManager not available, using cash-only calculation")
            account_state = None

        # Calculate base position (without margin)
        base_result = self.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=entry_price * 1.10,  # 10% target
            confidence=confidence,
        )

        if not use_margin or account_state is None:
            return {
                "use_margin": False,
                "base_shares": base_result.shares,
                "margin_shares": base_result.shares,
                "buying_power": self.total_capital,
                "margin_status": "CASH_ONLY",
            }

        # Check margin status
        margin_status = account_state.status.value
        equity_ratio = account_state.equity_ratio

        # Adjust position based on margin status
        margin_multiplier = 1.0
        warnings = []

        if equity_ratio < VN_MARGIN_CALL_LEVEL:
            # Margin call - block new positions
            return {
                "use_margin": True,
                "base_shares": base_result.shares,
                "margin_shares": 0,
                "buying_power": 0,
                "margin_status": margin_status,
                "blocked": True,
                "reason": f"Margin call active (equity: {equity_ratio:.1%}). No new positions allowed.",
            }
        elif equity_ratio < VN_MARGIN_WARNING_LEVEL:
            # Warning level - reduce position by 50%
            margin_multiplier = 0.5
            warnings.append(
                f"⚠️ Margin warning (equity: {equity_ratio:.1%}). Position reduced by 50%."
            )
        elif equity_ratio < VN_MAINTENANCE_MARGIN + 0.05:
            # Near maintenance - reduce position by 25%
            margin_multiplier = 0.75
            warnings.append(
                f"⚠️ Near maintenance margin (equity: {equity_ratio:.1%}). Position reduced by 25%."
            )

        # Calculate margin-enhanced position
        # With 50% margin, buying power is 2x cash
        margin_buying_power = account_state.buying_power
        max_margin_shares = int(margin_buying_power / entry_price)

        # Apply multiplier and limits
        margin_shares = int(base_result.shares * (1 + margin_ratio) * margin_multiplier)
        margin_shares = min(margin_shares, max_margin_shares)
        margin_shares = (margin_shares // VIETNAM_LOT_SIZE) * VIETNAM_LOT_SIZE

        # Calculate borrowed amount
        cash_portion = base_result.shares * entry_price
        borrowed_amount = max(0, (margin_shares * entry_price) - cash_portion)

        return {
            "use_margin": True,
            "base_shares": base_result.shares,
            "margin_shares": margin_shares,
            "cash_portion": cash_portion,
            "borrowed_amount": borrowed_amount,
            "buying_power": margin_buying_power,
            "margin_status": margin_status,
            "equity_ratio": equity_ratio,
            "margin_multiplier": margin_multiplier,
            "warnings": warnings,
        }

    # =========================================================================
    # NEW v6.0: T+0 INTRADAY TRADING VALIDATION
    # =========================================================================

    def validate_t0_trading(
        self,
        symbol: str,
        quantity: int,
        price: float,
        account_value: float,
    ) -> Dict:
        """
        Validate T+0 intraday trading capability.

        IMPROVED v6.0: T+0 Intraday Trading Validation
        =========================================================================
        Vietnam T+0 rules:
        - Only for margin accounts
        - Minimum account value: 50M VND
        - Max trades per day: 20
        - Max daily loss: 2%
        - Min holding time: 5 minutes (wash trade prevention)
        =========================================================================

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Trade price
            account_value: Total account value

        Returns:
            Dict with T+0 validation result
        """
        from src.config.constants import (
            VN_T0_ENABLED,
            VN_T0_MIN_ACCOUNT_VALUE,
            VN_T0_MAX_TRADES_PER_DAY,
            VN_T0_MAX_LOSS_PCT,
            VN_T0_MIN_HOLDING_MINUTES,
        )

        result = {
            "t0_enabled": VN_T0_ENABLED,
            "can_trade_t0": False,
            "validations": [],
            "warnings": [],
        }

        if not VN_T0_ENABLED:
            result["reason"] = "T+0 trading is disabled in configuration"
            return result

        # Check 1: Account value
        if account_value < VN_T0_MIN_ACCOUNT_VALUE:
            result["validations"].append(
                {
                    "check": "account_value",
                    "passed": False,
                    "message": f"Account value ({account_value/1e6:.0f}M) < minimum ({VN_T0_MIN_ACCOUNT_VALUE/1e6:.0f}M)",
                }
            )
            result["reason"] = "Account value below T+0 minimum"
            return result
        else:
            result["validations"].append(
                {
                    "check": "account_value",
                    "passed": True,
                    "message": f"Account value ({account_value/1e6:.0f}M) meets minimum",
                }
            )

        # Check 2: Broker API support
        # Note: Broker validation is optional - T+0 can proceed without broker verification
        # In production, broker should be configured and available
        try:
            from src.broker import get_broker

            # get_broker requires configuration - skip if not configured
            # This check is informational only
            result["validations"].append(
                {
                    "check": "broker_support",
                    "passed": True,
                    "message": "Broker module available (verification skipped - requires configuration)",
                }
            )
        except (ImportError, TypeError) as e:
            result["warnings"].append(f"Broker verification skipped: {str(e)}")

        # Check 3: Daily trade count
        try:
            from src.portfolio.intraday_trading import get_intraday_tracker

            tracker = get_intraday_tracker()
            stats = tracker.get_stats()

            if stats.total_trades >= VN_T0_MAX_TRADES_PER_DAY:
                result["validations"].append(
                    {
                        "check": "daily_trades",
                        "passed": False,
                        "message": f"Daily trade limit reached ({stats.total_trades}/{VN_T0_MAX_TRADES_PER_DAY})",
                    }
                )
                result["reason"] = "Daily T+0 trade limit reached"
                return result
            else:
                result["validations"].append(
                    {
                        "check": "daily_trades",
                        "passed": True,
                        "message": f"Trade count OK ({stats.total_trades}/{VN_T0_MAX_TRADES_PER_DAY})",
                    }
                )

            # Check 4: Daily loss limit
            daily_loss_pct = abs(stats.net_pnl / account_value) if stats.net_pnl < 0 else 0
            if daily_loss_pct >= VN_T0_MAX_LOSS_PCT:
                result["validations"].append(
                    {
                        "check": "daily_loss",
                        "passed": False,
                        "message": f"Daily loss limit reached ({daily_loss_pct:.1%} >= {VN_T0_MAX_LOSS_PCT:.1%})",
                    }
                )
                result["reason"] = "Daily T+0 loss limit reached"
                return result
            else:
                result["validations"].append(
                    {
                        "check": "daily_loss",
                        "passed": True,
                        "message": f"Daily loss OK ({daily_loss_pct:.1%} < {VN_T0_MAX_LOSS_PCT:.1%})",
                    }
                )

        except ImportError:
            result["warnings"].append("Intraday tracker not available - cannot verify trade limits")

        # All checks passed
        result["can_trade_t0"] = True
        result["min_holding_minutes"] = VN_T0_MIN_HOLDING_MINUTES

        return result

    # =========================================================================
    # NEW v6.0: WARRANT/ETF SPECIAL INSTRUMENT HANDLING
    # =========================================================================

    def calculate_special_instrument_position(
        self,
        symbol: str,
        entry_price: float,
        confidence: int,
        instrument_type: str = "AUTO",
    ) -> Dict:
        """
        Calculate position size for special instruments (warrants, ETFs).

        IMPROVED v6.0: Warrant/ETF Specific Logic
        =========================================================================
        Warrants:
        - ±50% daily price limit (vs ±7% for stocks)
        - T+0 settlement
        - Don't trade if < 3 days to expiry
        - Warning if < 30 days to expiry
        - Max 5% portfolio allocation

        ETFs:
        - ±7% daily price limit (same as stocks)
        - Some allow short selling
        - Track premium/discount to NAV
        =========================================================================

        Args:
            symbol: Instrument symbol
            entry_price: Entry price
            confidence: Signal confidence (0-100)
            instrument_type: "WARRANT", "ETF", or "AUTO" (auto-detect)

        Returns:
            Dict with special instrument position details
        """
        try:
            from src.strategies.special_instruments import (
                get_instrument_handler,
                InstrumentType,
            )

            handler = get_instrument_handler()
        except ImportError:
            logger.warning("Special instruments module not available")
            return {
                "instrument_type": "UNKNOWN",
                "error": "Special instruments module not available",
            }

        # Auto-detect instrument type
        if instrument_type == "AUTO":
            detected_type = handler.detect_instrument_type(symbol)
        else:
            detected_type = InstrumentType[instrument_type]

        result = {
            "symbol": symbol,
            "instrument_type": detected_type.value,
            "entry_price": entry_price,
            "confidence": confidence,
        }

        # Get price limits
        limits = handler.get_price_limits(symbol, entry_price)
        result["price_limits"] = limits

        if detected_type == InstrumentType.WARRANT:
            # Warrant-specific logic
            from src.strategies.special_instruments import get_warrant_logic

            warrant_logic = get_warrant_logic()

            # Check if tradeable
            from src.strategies.special_instruments import WarrantInfo
            from datetime import datetime, timedelta

            # Create basic warrant info (in production, fetch from data source)
            warrant_info = WarrantInfo(
                symbol=symbol,
                underlying=warrant_logic.get_underlying(symbol) or "UNKNOWN",
                issuer="UNKNOWN",
                exercise_price=entry_price * 0.9,  # Estimate
                exercise_ratio=1.0,
                expiry_date=datetime.now() + timedelta(days=60),  # Estimate
                warrant_type="CALL",
                warrant_price=entry_price,
            )

            is_tradeable, warnings = warrant_logic.check_tradeable(warrant_info, entry_price)
            result["is_tradeable"] = is_tradeable
            result["warnings"] = warnings

            if is_tradeable:
                # Calculate position (max 5% for warrants)
                max_allocation = self.total_capital * 0.05
                shares = warrant_logic.calculate_position_size(
                    self.total_capital, entry_price, confidence
                )
                result["shares"] = shares
                result["max_allocation"] = max_allocation
                result["stop_loss"] = warrant_logic.calculate_stop_loss(entry_price, warrant_info)
            else:
                result["shares"] = 0
                result["blocked_reason"] = warnings[0] if warnings else "Not tradeable"

        elif detected_type == InstrumentType.ETF:
            # ETF-specific logic
            from src.strategies.special_instruments import get_etf_logic

            etf_logic = get_etf_logic()

            etf_info = etf_logic.get_etf_info(symbol)
            result["etf_info"] = etf_info
            result["can_short"] = etf_logic.can_short(symbol)

            # Standard position sizing for ETFs
            base_result = self.calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=entry_price * 0.95,  # 5% stop
                take_profit=entry_price * 1.10,  # 10% target
                confidence=confidence,
            )
            result["shares"] = base_result.shares
            result["position_value"] = base_result.value

        else:
            # Regular stock - use standard calculation
            base_result = self.calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=entry_price * 0.93,  # 7% stop (floor)
                take_profit=entry_price * 1.10,
                confidence=confidence,
            )
            result["shares"] = base_result.shares
            result["position_value"] = base_result.value

        return result
