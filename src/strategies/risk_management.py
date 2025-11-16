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

    def calculate_enhanced_position_size(
        self, symbol, price, atr, confidence, signal="BUY", market_volatility=0.02
    ):
        """
        Position sizing nâng cao với nhiều yếu tố điều chỉnh

        Args:
            market_volatility: Độ biến động thị trường (VIX proxy)
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

        # ĐIỀU CHỈNH 3: Market Regime
        if self.market_regime_adjustment:
            regime_factor = self._calculate_market_regime_factor()
            base_position["shares"] = int(base_position["shares"] * regime_factor)
            base_position["value"] = base_position["shares"] * price
            base_position["max_loss"] = base_position["risk_per_share"] * base_position["shares"]
            print(f"  🌡️ Market regime adjustment: {regime_factor:.2f}x")

        # Đảm bảo không vượt quá giới hạn
        max_shares_by_capital = int((self.total_capital * self.max_position_pct) / price)
        base_position["shares"] = min(base_position["shares"], max_shares_by_capital)

        # Làm tròn đến lot 100
        if base_position["shares"] > 0:
            base_position["shares"] = max((base_position["shares"] // 100) * 100, 100)
            base_position["value"] = base_position["shares"] * price

        return base_position

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

    def _calculate_market_regime_factor(self):
        """Điều chỉnh theo regime thị trường (giả lập)"""
        # Trong thực tế, cần phân tích trend VNINDEX
        # Tạm thời giả lập:
        try:
            from src.data.loader import load_data
            from utils.dataframe_utils import safe_get_latest

            vnindex_data = load_data("VNINDEX", lookback=50)
            if len(vnindex_data) > 20:
                current_close = safe_get_latest(vnindex_data, "close", 0)
                past_close = (
                    vnindex_data["close"].iloc[-20] if len(vnindex_data) > 20 else current_close
                )
                market_trend = (current_close / past_close - 1) if past_close > 0 else 0

                if market_trend > 0.02:
                    return 1.1  # Bull market -> tăng nhẹ
                elif market_trend < -0.02:
                    return 0.6  # Bear market -> giảm mạnh
                else:
                    return 0.8  # Sideway -> giảm nhẹ
        except Exception:
            pass

        return 1.0  # Default

    def suggest_enhanced_limit_orders(self, current_price, atr, signal="BUY", confidence=50):
        """Đề xuất limit orders nâng cao"""
        super().suggest_limit_orders(current_price, atr, signal)

        # Điều chỉnh theo confidence - Higher confidence = more aggressive (closer to price)
        # Invert confidence: high confidence (80%) -> small discount, low confidence (30%) -> large discount
        confidence_factor = 1.0 - (confidence / 100) * 0.5  # Maps 0% -> 1.0, 100% -> 0.5

        if signal == "BUY":
            return {
                "aggressive": round(current_price - (atr * 0.3 * confidence_factor), -2),
                "moderate": round(current_price - (atr * 0.5 * confidence_factor), -2),
                "conservative": round(current_price - (atr * 0.7 * confidence_factor), -2),
                "note": f"Giá mua điều chỉnh theo confidence: {confidence}%",
            }
        else:
            return {
                "aggressive": round(current_price + (atr * 0.3 * confidence_factor), -2),
                "moderate": round(current_price + (atr * 0.5 * confidence_factor), -2),
                "conservative": round(current_price + (atr * 0.7 * confidence_factor), -2),
                "note": f"Giá bán điều chỉnh theo confidence: {confidence}%",
            }
