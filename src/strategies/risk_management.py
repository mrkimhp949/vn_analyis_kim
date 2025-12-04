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

    def calculate_position_size(self, price, atr, confidence, signal="BUY"):
        """Base position sizing calculation"""
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

    def calculate_enhanced_position_size(
        self,
        symbol,
        price,
        atr,
        confidence,
        signal="BUY",
        market_volatility=0.02,
        market_regime=None,
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
            regime_factor = self._calculate_market_regime_factor(market_regime)
            base_position["shares"] = int(base_position["shares"] * regime_factor)
            base_position["value"] = base_position["shares"] * price
            base_position["max_loss"] = base_position["risk_per_share"] * base_position["shares"]

        # Đảm bảo không vượt quá giới hạn
        max_shares_by_capital = int((self.total_capital * self.max_position_pct) / price)
        base_position["shares"] = min(base_position["shares"], max_shares_by_capital)

        # IMPROVED v4.2: Vietnam lot size validation
        if base_position["shares"] > 0:
            base_position["shares"] = self._round_to_vn_lot(base_position["shares"])
            base_position["value"] = base_position["shares"] * price

        # IMPROVED v4.2: Add transaction cost estimate
        base_position["estimated_cost"] = base_position["value"] * self.vn_transaction_cost
        base_position["net_value"] = base_position["value"] - base_position["estimated_cost"]

        return base_position

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

    def _calculate_market_regime_factor(self, market_regime: dict = None):
        """
        Điều chỉnh theo regime thị trường từ regime_detector.py

        IMPROVEMENT #4: Tích hợp với MarketRegimeDetector thay vì tính toán riêng
        Position sizing phản ánh đúng market conditions

        IMPROVED v4.1: Vietnam market specific adjustments
        - VN market has ±7% daily limit, so regime impact is amplified
        - Foreign flow is critical indicator for VN market
        - T+2 settlement affects position sizing in volatile regimes

        Args:
            market_regime: Dict từ regime_detector.detect() hoặc None để tự detect

        Returns:
            float: Factor điều chỉnh position size (0.25 - 1.15)
        """
        try:
            # Nếu không có market_regime, tự detect
            if market_regime is None:
                from src.market.regime_detector import get_regime_detector
                from src.data.vnindex_cache import get_cached_vnindex

                vnindex_data = get_cached_vnindex(lookback=250)
                if vnindex_data is not None and len(vnindex_data) >= 200:
                    detector = get_regime_detector()
                    regime_result = detector.detect(vnindex_data)
                    market_regime = {
                        "regime": regime_result.regime,
                        "confidence": regime_result.confidence,
                        "tradeable": regime_result.tradeable,
                        "components": regime_result.components,
                    }

            if market_regime is None:
                return 1.0  # Default nếu không detect được

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
            print(f"  ⚠️ Regime detector not available: {e}")
            return 1.0
        except Exception as e:
            print(f"  ⚠️ Error calculating regime factor: {e}")
            return 1.0

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
