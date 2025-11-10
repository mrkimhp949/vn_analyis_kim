"""
Risk Management - Position Sizing & Price Targets
"""

import numpy as np
import pandas as pd

class RiskManager:
    def __init__(self, 
                 total_capital=100_000_000,
                 max_position_pct=0.2,      # Tối đa 20% vốn/cổ phiếu
                 risk_per_trade_pct=0.02,   # Rủi ro 2% vốn/giao dịch
                 max_portfolio_risk=0.10):  # Tổng rủi ro danh mục 10%
        """
        total_capital: Tổng vốn (VNĐ)
        max_position_pct: % vốn tối đa cho 1 cổ phiếu
        risk_per_trade_pct: % vốn rủi ro cho 1 lệnh
        max_portfolio_risk: % rủi ro tối đa toàn danh mục
        """
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_portfolio_risk = max_portfolio_risk
    
    def calculate_position_size(self, 
                                current_price, 
                                atr,
                                confidence,
                                signal='BUY'):
        """
        Tính số lượng cổ phiếu nên mua/bán
        
        Phương pháp: ATR-based Position Sizing
        - Stop Loss = 2 * ATR
        - Position Size = (Risk Amount) / (Stop Loss Distance)
        
        Returns:
            dict: {
                'shares': Số CP,
                'value': Giá trị (VNĐ),
                'stop_loss': Giá cắt lỗ,
                'take_profit_1': Chốt lời lần 1,
                'take_profit_2': Chốt lời lần 2,
                'risk_reward_ratio': Tỷ lệ R:R
            }
        """
        
        # 1. Tính Stop Loss dựa trên ATR
        atr_multiplier = 2.0  # Standard: 2 ATR
        stop_loss_distance = atr * atr_multiplier
        
        if signal == 'BUY':
            stop_loss = current_price - stop_loss_distance
        else:  # SELL
            stop_loss = current_price + stop_loss_distance
        
        # 2. Tính Risk Amount (điều chỉnh theo confidence)
        # Confidence cao → rủi ro nhiều hơn
        confidence_factor = confidence / 100
        risk_amount = self.total_capital * self.risk_per_trade_pct * confidence_factor
        
        # 3. Position Size = Risk / Stop Loss Distance
        shares = int(risk_amount / stop_loss_distance)
        
        # 4. Giới hạn theo % vốn
        max_shares_by_capital = int((self.total_capital * self.max_position_pct) / current_price)
        shares = min(shares, max_shares_by_capital)
        
        # Đảm bảo ít nhất 100 CP (1 lô)
        shares = max(shares // 100 * 100, 100)
        
        position_value = shares * current_price
        
        # 5. Take Profit targets
        # TP1 = 1.5:1 R:R (chốt 50%)
        # TP2 = 3:1 R:R (chốt 50% còn lại)
        if signal == 'BUY':
            tp1 = current_price + (stop_loss_distance * 1.5)
            tp2 = current_price + (stop_loss_distance * 3.0)
        else:
            tp1 = current_price - (stop_loss_distance * 1.5)
            tp2 = current_price - (stop_loss_distance * 3.0)
        
        # 6. Risk:Reward Ratio
        risk_per_share = abs(current_price - stop_loss)
        reward_per_share = abs(tp2 - current_price)
        risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        
        return {
            'shares': shares,
            'value': position_value,
            'price_entry': current_price,
            'stop_loss': round(stop_loss, -2),  # Làm tròn 100
            'take_profit_1': round(tp1, -2),
            'take_profit_2': round(tp2, -2),
            'risk_per_share': risk_per_share,
            'reward_per_share': reward_per_share,
            'risk_reward_ratio': risk_reward,
            'max_loss': risk_per_share * shares,
            'expected_profit_tp2': reward_per_share * shares
        }
    
    def suggest_limit_orders(self, current_price, atr, signal='BUY'):
        """
        Đề xuất giá đặt lệnh limit (tối ưu hơn market order)
        
        Returns:
            dict: {
                'aggressive': Giá tích cực (gần market),
                'moderate': Giá trung bình,
                'conservative': Giá thận trọng (giảm giá tốt hơn)
            }
        """
        
        # Dùng 0.3 - 0.7 ATR để đặt limit
        if signal == 'BUY':
            return {
                'aggressive': round(current_price - (atr * 0.3), -2),
                'moderate': round(current_price - (atr * 0.5), -2),
                'conservative': round(current_price - (atr * 0.7), -2),
                'note': 'Giá mua thấp hơn → Lợi nhuận cao hơn'
            }
        else:  # SELL
            return {
                'aggressive': round(current_price + (atr * 0.3), -2),
                'moderate': round(current_price + (atr * 0.5), -2),
                'conservative': round(current_price + (atr * 0.7), -2),
                'note': 'Giá bán cao hơn → Lợi nhuận cao hơn'
            }
    
    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        """
        Kelly Criterion: Tính % vốn tối ưu cho mỗi lệnh
        
        Formula: K = W - [(1-W) / R]
        W = Win rate
        R = Avg Win / Avg Loss
        
        Returns:
            float: % vốn nên đầu tư (0-1)
        """
        if avg_loss == 0:
            return 0
        
        R = avg_win / avg_loss
        K = win_rate - ((1 - win_rate) / R)
        
        # Kelly thường quá aggressive, dùng Half Kelly
        half_kelly = K / 2
        
        # Giới hạn 0-25%
        return max(0, min(half_kelly, 0.25))
    
    def adjust_for_portfolio_risk(self, current_positions):
        """
        Điều chỉnh position size dựa trên tổng rủi ro danh mục
        
        Args:
            current_positions: list of dict với 'value' và 'risk'
        
        Returns:
            float: Hệ số điều chỉnh (0-1)
        """
        total_position_value = sum(p.get('value', 0) for p in current_positions)
        total_risk = sum(p.get('risk', 0) for p in current_positions)
        
        portfolio_risk_pct = total_risk / self.total_capital
        
        # Nếu rủi ro vượt ngưỡng, giảm position size mới
        if portfolio_risk_pct >= self.max_portfolio_risk:
            return 0  # Không mở lệnh mới
        
        # Tính hệ số điều chỉnh
        remaining_risk_capacity = self.max_portfolio_risk - portfolio_risk_pct
        adjustment_factor = remaining_risk_capacity / self.max_portfolio_risk
        
        return adjustment_factor
    
    def format_recommendation(self, symbol, result, position_info, limit_prices, df=None):
        """
        Format thành message dễ đọc với NHIỀU THÔNG TIN
        """
        signal = result['signal']
        confidence = result['confidence']
        
        if signal == 'BUY':
            emoji = '🟢'
            action = 'MUA'
            color = '🚀'
        else:
            emoji = '🔴'
            action = 'BÁN'
            color = '⚠️'
        
        # Tính % thay đổi giá
        price_change_pct = 0
        if df is not None and len(df) > 1:
            prev_close = df.iloc[-2]['close']
            curr_close = df.iloc[-1]['close']
            price_change_pct = ((curr_close - prev_close) / prev_close) * 100
        
        price_change_emoji = '📈' if price_change_pct > 0 else '📉'
        
        # Header
        msg = f"""
{color}═══════════════════════════════════════{color}
{emoji} **[{symbol}] TÍN HIỆU {action}** {emoji}
Độ tin cậy: **{confidence}%** {'⭐' * (confidence // 20)}
{color}═══════════════════════════════════════{color}

📊 **THÔNG TIN GIÁ:**
├─ Giá hiện tại: **{position_info['price_entry']:,.0f}** VNĐ
├─ Thay đổi: {price_change_emoji} **{price_change_pct:+.2f}%** (hôm qua)
├─ Volume: {df.iloc[-1]['volume']:,.0f} if df is not None else 'N/A'
└─ ATR (biến động): {df.iloc[-1]['atr']:,.0f} VNĐ if df is not None else 'N/A'

💰 **KHUYẾN NGHỊ VÀO LỆNH:**
├─ Số lượng: **{position_info['shares']:,}** cổ phiếu ({position_info['shares']//100} lô)
├─ Giá trị: **{position_info['value']:,.0f}** VNĐ
└─ % vốn: {(position_info['value']/self.total_capital)*100:.1f}%

📌 **GIÁ ĐẶT LỆNH (Limit Order):**
{'├─ 🔥 Aggressive:' if signal=='BUY' else '├─ 🔥 Aggressive:'} {limit_prices['aggressive']:,.0f} VNĐ (nhanh)
{'├─ ⭐ Moderate:' if signal=='BUY' else '├─ ⭐ Moderate:'} **{limit_prices['moderate']:,.0f}** VNĐ (khuyến nghị)
{'└─ 🎯 Conservative:' if signal=='BUY' else '└─ 🎯 Conservative:'} {limit_prices['conservative']:,.0f} VNĐ (tối ưu)
💡 {limit_prices['note']}

🛡️ **QUẢN TRỊ RỦI RO:**
├─ Stop Loss: **{position_info['stop_loss']:,.0f}** VNĐ ({abs((position_info['stop_loss']-position_info['price_entry'])/position_info['price_entry'])*100:.1f}%)
├─ Lỗ tối đa: **{position_info['max_loss']:,.0f}** VNĐ
└─ Risk/Share: {position_info['risk_per_share']:,.0f} VNĐ

🎯 **MỤC TIÊU LỢI NHUẬN:**
├─ TP1 (50%): **{position_info['take_profit_1']:,.0f}** VNĐ (+{abs((position_info['take_profit_1']-position_info['price_entry'])/position_info['price_entry'])*100:.1f}%)
├─ TP2 (50%): **{position_info['take_profit_2']:,.0f}** VNĐ (+{abs((position_info['take_profit_2']-position_info['price_entry'])/position_info['price_entry'])*100:.1f}%)
├─ Lợi nhuận kỳ vọng: **{position_info['expected_profit_tp2']:,.0f}** VNĐ
└─ Reward/Share: {position_info['reward_per_share']:,.0f} VNĐ

📈 **TỶ LỆ RISK:REWARD:**
⚖️ **1 : {position_info['risk_reward_ratio']:.1f}** {'✅ Tốt' if position_info['risk_reward_ratio'] >= 2 else '⚠️ Chấp nhận được' if position_info['risk_reward_ratio'] >= 1.5 else '❌ Rủi ro cao'}

🤖 **PHÂN TÍCH KỸ THUẬT:**
├─ ML Score: **{result['ml_score']:.3f}** {'(Rất tích cực)' if result['ml_score'] > 0.7 else '(Tích cực)' if result['ml_score'] > 0.6 else '(Trung lập)' if result['ml_score'] > 0.4 else '(Tiêu cực)'}
├─ RSI: **{result['rsi']:.1f}** {'(Quá mua)' if result['rsi'] > 70 else '(Quá bán)' if result['rsi'] < 30 else '(Trung bình)'}
├─ EMA Trend: **{result['ema_trend']}** {'📈' if result['ema_trend'] == 'UP' else '📉'}
"""
        
        # Thêm MACD nếu có
        if df is not None:
            macd_diff = df.iloc[-1]['macd_diff']
            macd_status = 'Bullish' if macd_diff > 0 else 'Bearish'
            msg += f"├─ MACD: **{macd_status}** ({macd_diff:,.0f})\n"
        
        msg += f"└─ Lý do: {result['reason']}\n"
        
        # Footer với hướng dẫn
        msg += f"""
{color}═══════════════════════════════════════{color}
📋 **HƯỚNG DẪN:**
1. Đặt lệnh Limit tại giá Moderate
2. Đặt Stop Loss ngay sau khi khớp
3. Chốt 50% tại TP1, trailing stop 50% còn lại
4. Không all-in, chỉ dùng {(position_info['value']/self.total_capital)*100:.1f}% vốn

⏰ Thời gian: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{color}═══════════════════════════════════════{color}
"""
        
        return msg.strip()