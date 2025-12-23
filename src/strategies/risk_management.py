"""
Enhanced Risk Management
Quản lý rủi ro nâng cao với volatility adjustment và correlation penalty
"""


class RiskManager:
    """Base Risk Manager class"""

    def __init__(self, total_capital=100_000_000, max_position_pct=0.2, risk_per_trade_pct=0.02):
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.risk_per_trade_pct = risk_per_trade_pct

    def _normalize_price(self, price: float) -> float:
        """Normalize price to VND (vnstock returns prices in thousands)."""
        try:
            from src.utils.vietnam_market import normalize_price_to_vnd
            return normalize_price_to_vnd(price)
        except ImportError:
            # Fallback: if price < 1000, assume it's in thousands
            if 0 < price < 1000:
                return price * 1000
            return price

    def calculate_position_size(self, price, atr, confidence, signal="BUY"):
        """Base position sizing calculation"""
        # Normalize price to VND
        price = self._normalize_price(price)
        atr = self._normalize_price(atr) if atr < 1000 else atr
        
        # Calculate max risk per trade
        max_risk = self.total_capital * self.risk_per_trade_pct

        # Calculate risk per share (using ATR as stop loss distance)
        risk_per_share = atr * 2  # 2x ATR stop loss

        # Calculate number of shares
        shares = int(max_risk / risk_per_share) if risk_per_share > 0 else 0

        # Check against max position size
        max_shares_by_capital = int((self.total_capital * self.max_position_pct) / price)
        shares = min(shares, max_shares_by_capital)

        # Round to lot of 100
        if shares > 0:
            shares = max((shares // 100) * 100, 100)

        return {
            "shares": shares,
            "value": shares * price,
            "risk_per_share": risk_per_share,
            "max_loss": risk_per_share * shares,
        }

    def suggest_limit_orders(self, current_price, atr, signal="BUY"):
        """Suggest limit order prices"""
        if signal == "BUY":
            return {
                "aggressive": round(current_price - (atr * 0.3), -2),
                "moderate": round(current_price - (atr * 0.5), -2),
                "conservative": round(current_price - (atr * 0.7), -2),
            }
        else:
            return {
                "aggressive": round(current_price + (atr * 0.3), -2),
                "moderate": round(current_price + (atr * 0.5), -2),
                "conservative": round(current_price + (atr * 0.7), -2),
            }

    def format_recommendation(self, symbol, result, position_info, limit_prices, df=None):
        """Format recommendation message"""
        return f"Position recommendation for {symbol}"


class EnhancedRiskManager(RiskManager):
    """
    Enhanced Risk Manager with Vietnam market specific features.

    IMPROVED v4.2:
    - T+2 settlement capital management
    - Lot size validation (100 shares)
    - Tick size awareness
    - Session-based position sizing
    - Foreign flow integration
    """

    def suggest_limit_orders(self, current_price, atr, signal="BUY"):
        """Override method từ class cha để tránh lỗi"""
        return super().suggest_limit_orders(current_price, atr, signal)

    def format_recommendation(self, symbol, result, position_info, limit_prices, df=None):
        """Override method từ class cha"""
        return super().format_recommendation(symbol, result, position_info, limit_prices, df)

    def __init__(self, total_capital=100_000_000, max_position_pct=0.2, risk_per_trade_pct=0.02):
        super().__init__(total_capital, max_position_pct, risk_per_trade_pct)
        self.volatility_adjustment = True
        self.correlation_penalty = True
        self.market_regime_adjustment = True

        # NEW v4.2: Vietnam market specific settings
        self.vn_lot_size = 100  # Minimum trading unit
        self.vn_transaction_cost = 0.015  # ~1.5% round trip
        self.t2_capital_buffer = 0.10  # 10% buffer for T+2 settlements

        # NEW v5.0: Dynamic slippage integration
        self.use_dynamic_slippage = True

        # NEW v5.1: Liquidity-based position sizing
        self.use_liquidity_based_sizing = True
        self.max_volume_participation = 0.10  # Max 10% of daily volume
        self.liquidity_tiers = {
            "mega": {"min_value": 50_000_000_000, "max_participation": 0.05},  # 5% for mega caps
            "large": {"min_value": 10_000_000_000, "max_participation": 0.08},  # 8% for large caps
            "mid": {"min_value": 2_000_000_000, "max_participation": 0.10},  # 10% for mid caps
            "small": {"min_value": 500_000_000, "max_participation": 0.15},  # 15% for small caps
            "micro": {
                "min_value": 0,
                "max_participation": 0.20,
            },  # 20% for micro caps (but higher slippage)
        }

    def calculate_enhanced_position_size(
        self,
        symbol,
        price,
        atr,
        confidence,
        signal="BUY",
        market_volatility=0.02,
        market_regime=None,
        force_regime_refresh: bool = False,
        avg_daily_volume: int = 0,  # NEW: For dynamic slippage calculation
    ):
        """
        Position sizing nâng cao với nhiều yếu tố điều chỉnh

        Args:
            symbol: Mã cổ phiếu
            price: Giá hiện tại
            atr: Average True Range
            confidence: Độ tin cậy tín hiệu (0-100)
            signal: 'BUY' hoặc 'SELL'
            market_volatility: Độ biến động thị trường (VIX proxy)
            market_regime: Dict từ regime_detector (IMPROVEMENT #4)
            force_regime_refresh: Force refresh VNINDEX cache trước khi detect regime
                                  (nên dùng = True cho lệnh đầu tiên của trading session)
            avg_daily_volume: Average daily volume for slippage calculation (NEW v5.0)
        """
        # Base position size từ class cha
        base_position = super().calculate_position_size(price, atr, confidence, signal)

        if base_position["shares"] == 0:
            return base_position

        # ĐIỀU CHỈNH 1: Market Volatility
        if self.volatility_adjustment:
            volatility_factor = self._calculate_volatility_factor(market_volatility)
            base_position["shares"] = int(base_position["shares"] * volatility_factor)
            base_position["value"] = base_position["shares"] * price
            base_position["max_loss"] = base_position["risk_per_share"] * base_position["shares"]
            print(f"  📉 Volatility adjustment: {volatility_factor:.2f}x")

        # ĐIỀU CHỈNH 2: Confidence-based Sizing
        confidence_factor = self._calculate_confidence_factor(confidence)
        base_position["shares"] = int(base_position["shares"] * confidence_factor)
        base_position["value"] = base_position["shares"] * price
        base_position["max_loss"] = base_position["risk_per_share"] * base_position["shares"]
        print(f"  🎯 Confidence adjustment: {confidence_factor:.2f}x")

        # ĐIỀU CHỈNH 3: Market Regime (IMPROVEMENT #4 - Tích hợp với regime_detector)
        if self.market_regime_adjustment:
            regime_factor = self._calculate_market_regime_factor(
                market_regime, force_refresh=force_regime_refresh
            )
            base_position["shares"] = int(base_position["shares"] * regime_factor)
            base_position["value"] = base_position["shares"] * price
            base_position["max_loss"] = base_position["risk_per_share"] * base_position["shares"]

        # Đảm bảo không vượt quá giới hạn
        max_shares_by_capital = int((self.total_capital * self.max_position_pct) / price)
        base_position["shares"] = min(base_position["shares"], max_shares_by_capital)

        # NEW v5.1: Liquidity-based position sizing - limit to % of daily volume
        if self.use_liquidity_based_sizing and avg_daily_volume > 0:
            liquidity_result = self._apply_liquidity_based_sizing(
                shares=base_position["shares"],
                price=price,
                avg_daily_volume=avg_daily_volume,
                symbol=symbol,
            )
            if liquidity_result["adjusted"]:
                base_position["shares"] = liquidity_result["shares"]
                base_position["value"] = base_position["shares"] * price
                base_position["max_loss"] = (
                    base_position["risk_per_share"] * base_position["shares"]
                )
                base_position["liquidity_cap"] = liquidity_result
                print(
                    f"  📊 Liquidity cap: {liquidity_result['participation_pct']*100:.1f}% of daily volume "
                    f"(tier: {liquidity_result['tier']})"
                )

        # NEW v5.0: Dynamic slippage and execution cost calculation
        if self.use_dynamic_slippage and avg_daily_volume > 0:
            slippage_result = self._calculate_dynamic_execution_cost(
                symbol=symbol,
                order_value=base_position["value"],
                avg_daily_volume=avg_daily_volume,
                avg_price=price,
                signal=signal,
            )
            base_position["execution_cost"] = slippage_result
            base_position["effective_value"] = (
                slippage_result.get("effective_price", price) * base_position["shares"]
            )

            # Adjust position if slippage is too high
            if slippage_result.get("total_cost_pct", 0) > 0.025:  # > 2.5% total cost
                # Reduce position to lower market impact
                reduction_factor = 0.025 / slippage_result["total_cost_pct"]
                reduction_factor = max(0.5, min(1.0, reduction_factor))  # Cap at 50% reduction
                base_position["shares"] = int(base_position["shares"] * reduction_factor)
                base_position["shares"] = self._round_to_vn_lot(base_position["shares"])
                base_position["value"] = base_position["shares"] * price
                print(
                    f"  💰 Slippage adjustment: {reduction_factor:.2f}x (cost was {slippage_result['total_cost_pct']*100:.1f}%)"
                )

            # Add recommendation to position info
            base_position["order_recommendation"] = slippage_result.get("recommendation", "")

        return base_position

    def _apply_liquidity_based_sizing(
        self,
        shares: int,
        price: float,
        avg_daily_volume: int,
        symbol: str,
    ) -> dict:
        """
        Apply liquidity-based position sizing.

        NEW v5.1 IMPROVEMENT: Scale position size based on % of daily volume
        to minimize market impact and slippage.

        Liquidity tiers:
        - Mega cap (>50B VND/day): Max 5% of daily volume
        - Large cap (10-50B VND/day): Max 8% of daily volume
        - Mid cap (2-10B VND/day): Max 10% of daily volume
        - Small cap (0.5-2B VND/day): Max 15% of daily volume
        - Micro cap (<0.5B VND/day): Max 20% of daily volume

        Args:
            shares: Calculated number of shares
            price: Current price
            avg_daily_volume: Average daily trading volume
            symbol: Stock symbol for logging

        Returns:
            Dict with adjusted shares, tier, participation_pct, adjusted flag
        """
        result = {
            "shares": shares,
            "original_shares": shares,
            "tier": "unknown",
            "participation_pct": 0.0,
            "max_participation": self.max_volume_participation,
            "adjusted": False,
            "reason": None,
        }

        if avg_daily_volume <= 0:
            return result

        # Calculate daily trading value
        avg_daily_value = avg_daily_volume * price

        # Determine liquidity tier
        tier = "micro"
        max_participation = self.liquidity_tiers["micro"]["max_participation"]

        for tier_name, tier_config in self.liquidity_tiers.items():
            if avg_daily_value >= tier_config["min_value"]:
                tier = tier_name
                max_participation = tier_config["max_participation"]
                break

        result["tier"] = tier
        result["max_participation"] = max_participation

        # Calculate current participation
        current_participation = shares / avg_daily_volume
        result["participation_pct"] = current_participation

        # Apply cap if exceeds max participation
        if current_participation > max_participation:
            max_shares = int(avg_daily_volume * max_participation)
            max_shares = self._round_to_vn_lot(max_shares)

            # Ensure minimum position
            if max_shares < self.vn_lot_size:
                max_shares = self.vn_lot_size

            result["shares"] = min(shares, max_shares)
            result["adjusted"] = True
            result["reason"] = (
                f"Capped to {max_participation*100:.0f}% of daily volume ({tier} cap)"
            )

            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"[{symbol}] Liquidity cap applied: {shares} -> {result['shares']} shares "
                f"({current_participation*100:.1f}% -> {max_participation*100:.0f}% of volume)"
            )

        return result

    def _calculate_dynamic_execution_cost(
        self,
        symbol: str,
        order_value: float,
        avg_daily_volume: int,
        avg_price: float,
        signal: str = "BUY",
    ) -> dict:
        """
        Calculate dynamic execution cost using the new slippage model.

        Args:
            symbol: Stock symbol
            order_value: Order value in VND
            avg_daily_volume: Average daily volume in shares
            avg_price: Average price per share
            signal: 'BUY' or 'SELL'

        Returns:
            Dict with execution cost breakdown
        """
        try:
            from src.config.constants import estimate_execution_cost

            return estimate_execution_cost(
                symbol=symbol,
                order_value=order_value,
                avg_daily_volume=avg_daily_volume,
                avg_price=avg_price,
                side=signal,
                is_market_order=True,  # Assume market order for conservative estimate
            )
        except ImportError:
            # Fallback to simple estimate
            return {
                "total_cost_pct": self.vn_transaction_cost,
                "total_cost_vnd": order_value * self.vn_transaction_cost,
                "effective_price": (
                    avg_price * (1 + self.vn_transaction_cost)
                    if signal == "BUY"
                    else avg_price * (1 - self.vn_transaction_cost)
                ),
                "recommendation": "UNKNOWN",
            }

    def _round_to_vn_lot(self, shares: int) -> int:
        """Round shares to Vietnam lot size (100 shares minimum)."""
        if shares <= 0:
            return 0
        rounded = (shares // self.vn_lot_size) * self.vn_lot_size
        return max(self.vn_lot_size, rounded)

    def calculate_available_capital(self, pending_settlements: list = None) -> float:
        """
        Calculate available capital considering T+2 pending settlements.

        Vietnam T+2 settlement means capital from sells is locked for 2 days.
        This method accounts for pending settlements to avoid over-trading.

        Args:
            pending_settlements: List of pending settlement amounts

        Returns:
            Available capital for new trades
        """
        pending = sum(pending_settlements) if pending_settlements else 0
        buffer = self.total_capital * self.t2_capital_buffer
        available = self.total_capital - pending - buffer
        return max(0, available)

    def _calculate_volatility_factor(self, market_volatility):
        """Điều chỉnh theo độ biến động thị trường"""
        # market_volatility: 0.01 (thấp) -> 0.05 (cao)
        if market_volatility < 0.015:
            # Low volatility -> tăng position
            return 1.2
        elif market_volatility < 0.03:
            # Normal volatility -> giữ nguyên
            return 1.0
        elif market_volatility < 0.05:
            # High volatility -> giảm position
            return 0.7
        else:
            # Very high volatility -> giảm mạnh
            return 0.5

    def _calculate_confidence_factor(self, confidence):
        """Điều chỉnh theo độ tin cậy tín hiệu"""
        if confidence >= 80:
            return 1.2  # High confidence -> tăng position
        elif confidence >= 60:
            return 1.0  # Medium confidence -> giữ nguyên
        elif confidence >= 40:
            return 0.7  # Low confidence -> giảm position
        else:
            return 0.3  # Very low confidence -> giảm mạnh

    def _calculate_market_regime_factor(
        self, market_regime: dict = None, force_refresh: bool = False
    ):
        """
        Điều chỉnh theo regime thị trường từ regime_detector.py

        IMPROVEMENT #4: Tích hợp với MarketRegimeDetector thay vì tính toán riêng
        Position sizing phản ánh đúng market conditions

        IMPROVED v4.1: Vietnam market specific adjustments
        - VN market has ±7% daily limit, so regime impact is amplified
        - Foreign flow is critical indicator for VN market
        - T+2 settlement affects position sizing in volatile regimes

        IMPROVED v4.2: Better cache handling and conservative fallback
        - Force refresh option để đảm bảo data fresh trước trading session
        - Conservative fallback (0.5) thay vì neutral (1.0) khi không detect được
        - Tránh risk trade trong bear market với bull position size

        Args:
            market_regime: Dict từ regime_detector.detect() hoặc None để tự detect
            force_refresh: Force refresh VNIndex cache (dùng trước trading session)

        Returns:
            float: Factor điều chỉnh position size (0.25 - 1.15)
        """
        # Conservative default - giảm 50% position nếu không detect được regime
        # Tránh risk trade trong bear market với bull position size
        CONSERVATIVE_DEFAULT_FACTOR = 0.5

        try:
            # Nếu không có market_regime, tự detect
            if market_regime is None:
                from src.market.regime_detector import get_regime_detector
                from src.data.vnindex_cache import get_cached_vnindex

                # IMPROVED v4.2: Force refresh để đảm bảo data không stale
                # Đặc biệt quan trọng trước mỗi trading session
                vnindex_data = get_cached_vnindex(lookback=250, force_refresh=force_refresh)

                if vnindex_data is not None and len(vnindex_data) >= 200:
                    detector = get_regime_detector()
                    regime_result = detector.detect(vnindex_data)
                    market_regime = {
                        "regime": regime_result.regime,
                        "confidence": regime_result.confidence,
                        "tradeable": regime_result.tradeable,
                        "components": regime_result.components,
                    }
                else:
                    # IMPROVED v4.2: Log warning khi data không đủ
                    bars = len(vnindex_data) if vnindex_data is not None else 0
                    print(
                        f"  ⚠️ VNINDEX data insufficient ({bars} bars), "
                        f"using conservative factor: {CONSERVATIVE_DEFAULT_FACTOR}"
                    )

            # IMPROVED v4.2: Conservative fallback thay vì neutral
            if market_regime is None:
                print(
                    f"  ⚠️ Cannot detect market regime, "
                    f"using conservative factor: {CONSERVATIVE_DEFAULT_FACTOR}"
                )
                return CONSERVATIVE_DEFAULT_FACTOR

            regime = market_regime.get("regime", "SIDEWAYS")
            confidence = market_regime.get("confidence", 50)
            tradeable = market_regime.get("tradeable", True)
            components = market_regime.get("components", {})

            # Không tradeable -> giảm mạnh position
            if not tradeable:
                print(f"  🚫 Market not tradeable (regime: {regime})")
                return 0.25  # TIGHTENED: 25% instead of 30%

            # Điều chỉnh theo regime và confidence
            # IMPROVED v4.1: More conservative for VN market
            if regime == "BULL":
                # Bull market: tăng position, scale theo confidence
                # VN market: Be more conservative even in bull
                if confidence >= 70:
                    factor = 1.15  # TIGHTENED: Strong bull -> tăng 15% (was 20%)
                elif confidence >= 50:
                    factor = 1.05  # TIGHTENED: Moderate bull -> tăng 5% (was 10%)
                else:
                    factor = 0.95  # TIGHTENED: Weak bull -> giảm nhẹ 5%

            elif regime == "BEAR":
                # Bear market: giảm mạnh position
                # VN market: ±7% limit means bear can be brutal
                if confidence >= 70:
                    factor = 0.35  # TIGHTENED: Strong bear -> giảm 65% (was 60%)
                elif confidence >= 50:
                    factor = 0.45  # TIGHTENED: Moderate bear -> giảm 55% (was 50%)
                else:
                    factor = 0.55  # TIGHTENED: Weak bear -> giảm 45% (was 40%)

            elif regime == "HIGH_VOLATILITY":
                # High volatility: giảm position để quản lý risk
                # VN market: High vol + ±7% limit = very dangerous
                volatility = components.get("volatility", 0.5)
                if volatility > 0.8:
                    factor = 0.25  # TIGHTENED: Extreme volatility -> giảm 75% (was 70%)
                else:
                    factor = 0.40  # TIGHTENED: High volatility -> giảm 60% (was 50%)

            else:  # SIDEWAYS
                # Sideways: giảm nhẹ, tùy thuộc vào volatility
                volatility = components.get("volatility", 0.5)
                if volatility > 0.5:
                    factor = 0.65  # TIGHTENED: Sideways + high vol -> giảm 35% (was 30%)
                else:
                    factor = 0.80  # TIGHTENED: Sideways + low vol -> giảm 20% (was 15%)

            # Điều chỉnh thêm theo sector rotation và foreign flow nếu có
            sector_score = components.get("sector_rotation", 0)
            foreign_score = components.get("foreign_flow", 0)

            # Bonus/penalty từ sector rotation (-0.1 to +0.1)
            if sector_score > 0.3:
                factor += 0.05  # Leading sectors -> bonus nhỏ
            elif sector_score < -0.3:
                factor -= 0.05  # Lagging sectors -> penalty nhỏ

            # IMPROVED v4.1: Foreign flow is critical for VN market
            # Foreign investors often lead market direction
            if foreign_score > 0.5:
                factor += 0.08  # INCREASED: Strong foreign buying -> bonus 8%
            elif foreign_score > 0.3:
                factor += 0.05  # Foreign buying -> bonus nhỏ
            elif foreign_score < -0.5:
                factor -= 0.10  # INCREASED: Strong foreign selling -> penalty 10%
            elif foreign_score < -0.3:
                factor -= 0.06  # Foreign selling -> penalty nhỏ

            # Clamp factor trong range hợp lý
            factor = max(0.25, min(1.15, factor))

            print(f"  🌡️ Market regime: {regime} (conf: {confidence:.0f}%) -> factor: {factor:.2f}")
            return factor

        except ImportError as e:
            print(
                f"  ⚠️ Regime detector not available: {e}, "
                f"using conservative factor: {CONSERVATIVE_DEFAULT_FACTOR}"
            )
            return CONSERVATIVE_DEFAULT_FACTOR
        except Exception as e:
            print(
                f"  ⚠️ Error calculating regime factor: {e}, "
                f"using conservative factor: {CONSERVATIVE_DEFAULT_FACTOR}"
            )
            return CONSERVATIVE_DEFAULT_FACTOR

    def suggest_enhanced_limit_orders(self, current_price, atr, signal="BUY", confidence=50):
        """Đề xuất limit orders nâng cao với Vietnam tick size validation"""
        super().suggest_limit_orders(current_price, atr, signal)

        # Điều chỉnh theo confidence - Higher confidence = more aggressive (closer to price)
        # Invert confidence: high confidence (80%) -> small discount, low confidence (30%) -> large discount
        confidence_factor = 1.0 - (confidence / 100) * 0.5  # Maps 0% -> 1.0, 100% -> 0.5

        if signal == "BUY":
            prices = {
                "aggressive": self._round_to_vn_tick(
                    current_price - (atr * 0.3 * confidence_factor)
                ),
                "moderate": self._round_to_vn_tick(current_price - (atr * 0.5 * confidence_factor)),
                "conservative": self._round_to_vn_tick(
                    current_price - (atr * 0.7 * confidence_factor)
                ),
                "note": f"Giá mua điều chỉnh theo confidence: {confidence}%",
            }
        else:
            prices = {
                "aggressive": self._round_to_vn_tick(
                    current_price + (atr * 0.3 * confidence_factor)
                ),
                "moderate": self._round_to_vn_tick(current_price + (atr * 0.5 * confidence_factor)),
                "conservative": self._round_to_vn_tick(
                    current_price + (atr * 0.7 * confidence_factor)
                ),
                "note": f"Giá bán điều chỉnh theo confidence: {confidence}%",
            }
        return prices

    def _round_to_vn_tick(self, price: float) -> float:
        """
        Round price to valid Vietnam tick size.

        Vietnam tick sizes (HOSE):
        - Price < 10,000 VND: Tick = 10 VND
        - 10,000 <= Price < 50,000 VND: Tick = 50 VND
        - Price >= 50,000 VND: Tick = 100 VND
        """
        if price < 10_000:
            tick = 10
        elif price < 50_000:
            tick = 50
        else:
            tick = 100
        return round(price / tick) * tick

    def get_session_position_multiplier(self) -> float:
        """
        Get position size multiplier based on current trading session.

        Vietnam market sessions:
        - ATO (9:00-9:15): High volatility -> 0.7x
        - Morning optimal (9:30-10:30): Best time -> 1.0x
        - Pre-lunch (11:00-11:30): Selling pressure -> 0.8x
        - Afternoon optimal (13:30-14:15): Good time -> 1.0x
        - ATC (14:30-14:45): High volatility -> 0.7x

        Returns:
            Position size multiplier (0.7 to 1.0)
        """
        try:
            from src.market.session_trading import get_session_manager

            manager = get_session_manager()
            timing = manager.analyze_entry_timing()
            return timing.position_size_multiplier
        except ImportError:
            return 1.0
        except Exception as e:
            print(f"  ⚠️ Session multiplier error: {e}")
            return 1.0

    def prepare_trading_session(self) -> dict:
        """
        Prepare for a new trading session by refreshing market data.

        IMPROVED v4.2: Force refresh VNINDEX cache và detect regime mới
        Nên gọi function này trước khi bắt đầu trading session để:
        1. Đảm bảo VNINDEX data không stale
        2. Detect market regime với data mới nhất
        3. Tránh risk trade trong bear market với bull position size

        Returns:
            dict: Session preparation result với regime info
        """
        from src.data.vnindex_cache import (
            get_cached_vnindex,
            invalidate_vnindex_cache,
            get_vnindex_cache_info,
        )

        result = {
            "success": False,
            "regime": None,
            "regime_factor": 0.5,  # Conservative default
            "vnindex_bars": 0,
            "message": "",
        }

        try:
            # Step 1: Invalidate old cache
            invalidate_vnindex_cache()
            print("🔄 Invalidated old VNINDEX cache")

            # Step 2: Force refresh VNINDEX data
            vnindex_data = get_cached_vnindex(lookback=250, force_refresh=True)

            if vnindex_data is None or len(vnindex_data) < 200:
                bars = len(vnindex_data) if vnindex_data is not None else 0
                result["message"] = (
                    f"VNINDEX data insufficient ({bars} bars), " "using conservative mode"
                )
                result["vnindex_bars"] = bars
                print(f"⚠️ {result['message']}")
                return result

            result["vnindex_bars"] = len(vnindex_data)

            # Step 3: Detect market regime with fresh data
            from src.market.regime_detector import get_regime_detector

            detector = get_regime_detector()
            regime_result = detector.detect(vnindex_data)

            market_regime = {
                "regime": regime_result.regime,
                "confidence": regime_result.confidence,
                "tradeable": regime_result.tradeable,
                "components": regime_result.components,
            }

            # Step 4: Calculate regime factor
            regime_factor = self._calculate_market_regime_factor(market_regime)

            result["success"] = True
            result["regime"] = market_regime
            result["regime_factor"] = regime_factor
            result["message"] = (
                f"Session prepared: {regime_result.regime} "
                f"(conf: {regime_result.confidence:.0f}%), "
                f"factor: {regime_factor:.2f}"
            )

            print(f"✅ {result['message']}")

            # Step 5: Log cache info
            cache_info = get_vnindex_cache_info()
            print(f"📦 VNINDEX cache: {cache_info['bars']} bars, TTL: {cache_info['ttl_seconds']}s")

            return result

        except ImportError as e:
            result["message"] = f"Module not available: {e}"
            print(f"⚠️ {result['message']}")
            return result
        except Exception as e:
            result["message"] = f"Session preparation failed: {e}"
            print(f"⚠️ {result['message']}")
            return result
