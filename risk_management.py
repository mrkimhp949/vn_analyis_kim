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
        """
        # 1. Tính Stop Loss dựa trên ATR
        atr_multiplier = 2.0  # Standard: 2 ATR
        stop_loss_distance = atr * atr_multiplier if atr is not None else 0
        
        if signal == 'BUY':
            stop_loss = current_price - stop_loss_distance
        else:  # SELL
            stop_loss = current_price + stop_loss_distance
        
        # 2. Tính Risk Amount (điều chỉnh theo confidence)
        confidence_factor = max(0.0, min(1.0, confidence / 100 if confidence is not None else 0))
        risk_amount = self.total_capital * self.risk_per_trade_pct * confidence_factor
        
        # 3. Position Size = Risk / Stop Loss Distance
        shares = int(risk_amount / stop_loss_distance) if stop_loss_distance > 0 else 0
        
        # 4. Giới hạn theo % vốn
        max_shares_by_capital = int((self.total_capital * self.max_position_pct) / current_price) if current_price > 0 else 0
        shares = min(shares, max_shares_by_capital)
        
        # Đảm bảo ít nhất 100 CP (1 lô)
        if shares <= 0:
            shares = 0
        else:
            shares = max((shares // 100) * 100, 100)
        
        position_value = shares * current_price
        
        # 5. Take Profit targets
        if signal == 'BUY':
            tp1 = current_price + (stop_loss_distance * 1.5)
            tp2 = current_price + (stop_loss_distance * 3.0)
        else:
            tp1 = current_price - (stop_loss_distance * 1.5)
            tp2 = current_price - (stop_loss_distance * 3.0)
        
        # 6. Risk:Reward Ratio
        risk_per_share = abs(current_price - stop_loss) if stop_loss is not None else 0
        reward_per_share = abs(tp2 - current_price) if tp2 is not None else 0
        risk_reward = (reward_per_share / risk_per_share) if risk_per_share > 0 else 0
        
        return {
            'shares': shares,
            'value': position_value,
            'price_entry': current_price,
            'stop_loss': round(stop_loss, -2) if stop_loss is not None else 0,
            'take_profit_1': round(tp1, -2) if tp1 is not None else 0,
            'take_profit_2': round(tp2, -2) if tp2 is not None else 0,
            'risk_per_share': risk_per_share,
            'reward_per_share': reward_per_share,
            'risk_reward_ratio': risk_reward,
            'max_loss': risk_per_share * shares,
            'expected_profit_tp2': reward_per_share * shares
        }
    
    def suggest_limit_orders(self, current_price, atr, signal='BUY'):
        """
        Đề xuất giá đặt lệnh limit
        """
        if atr is None:
            atr = 0
        if signal == 'BUY':
            return {
                'aggressive': round(current_price - (atr * 0.3), -2),
                'moderate': round(current_price - (atr * 0.5), -2),
                'conservative': round(current_price - (atr * 0.7), -2),
                'note': 'Giá mua thấp hơn → Lợi nhuận cao hơn'
            }
        else:
            return {
                'aggressive': round(current_price + (atr * 0.3), -2),
                'moderate': round(current_price + (atr * 0.5), -2),
                'conservative': round(current_price + (atr * 0.7), -2),
                'note': 'Giá bán cao hơn → Lợi nhuận cao hơn'
            }
    
    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        if avg_loss == 0:
            return 0
        
        R = avg_win / avg_loss if avg_loss != 0 else 0
        K = win_rate - ((1 - win_rate) / R) if R != 0 else 0
        half_kelly = K / 2
        return max(0, min(half_kelly, 0.25))
    
    def adjust_for_portfolio_risk(self, current_positions):
        total_position_value = sum(p.get('value', 0) for p in current_positions)
        total_risk = sum(p.get('risk', 0) for p in current_positions)
        
        portfolio_risk_pct = total_risk / self.total_capital if self.total_capital > 0 else 0
        
        if portfolio_risk_pct >= self.max_portfolio_risk:
            return 0
        
        remaining_risk_capacity = self.max_portfolio_risk - portfolio_risk_pct
        adjustment_factor = remaining_risk_capacity / self.max_portfolio_risk if self.max_portfolio_risk > 0 else 0
        
        return adjustment_factor
    
    def format_recommendation(self, symbol, result, position_info, limit_prices, df=None):
        """
        Format message (giữ nguyên như trước)
        """
        signal = result.get('signal', 'HOLD')
        confidence = result.get('confidence', 0)
        
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
            prev_close = df.iloc[-2].get('close', None)
            curr_close = df.iloc[-1].get('close', None)
            try:
                if prev_close and curr_close:
                    price_change_pct = ((curr_close - prev_close) / prev_close) * 100
            except Exception:
                price_change_pct = 0
        
        price_change_emoji = '📈' if price_change_pct > 0 else '📉'

        # Prepare safe strings for volume / atr
        if df is not None and len(df) > 0:
            try:
                volume_str = f"{int(df.iloc[-1].get('volume', 0)):,}"
            except Exception:
                volume_str = "N/A"
            try:
                atr_str = f"{df.iloc[-1].get('atr', 0):,.0f}"
            except Exception:
                atr_str = "N/A"
        else:
            volume_str = "N/A"
            atr_str = "N/A"
        
        # Header
        msg = f"""
{color}═══════════════════════════════════════{color}
{emoji} **[{symbol}] TÍN HIỆU {action}** {emoji}
Độ tin cậy: **{confidence}%** {'⭐' * (int(confidence) // 20)}
{color}═══════════════════════════════════════{color}

📊 **THÔNG TIN GIÁ:**
├─ Giá hiện tại: **{position_info.get('price_entry', 0):,.0f}** VNĐ
├─ Thay đổi: {price_change_emoji} **{price_change_pct:+.2f}%** (hôm qua)
├─ Volume: {volume_str}
└─ ATR (biến động): {atr_str} VNĐ
"""

        # BUY vs SELL blocks
        if signal == 'BUY':
            msg += f"""
💰 **KHUYẾN NGHỊ VÀO LỆNH:**
├─ Số lượng: **{position_info.get('shares', 0):,}** cổ phiếu ({int(position_info.get('shares', 0))//100} lô)
├─ Giá trị: **{position_info.get('value', 0):,.0f}** VNĐ
└─ % vốn: {(position_info.get('value', 0)/self.total_capital)*100:.1f}%
"""
        else:
            msg += f"""
🔴 **KHUYẾN NGHỊ BÁN:**
├─ Nếu đang nắm: Bán **TOÀN BỘ** vị thế
└─ Lý do: Tín hiệu kỹ thuật tiêu cực
"""

        # Common risk section
        stop_loss = position_info.get('stop_loss', 0)
        price_entry = position_info.get('price_entry', 1)
        try:
            sl_pct = abs((stop_loss - price_entry)/price_entry)*100
        except Exception:
            sl_pct = 0

        msg += f"""
🛡️ **QUẢN TRỊ RỦI RO:**
├─ Stop Loss: **{stop_loss:,.0f}** VNĐ ({sl_pct:.1f}%)
├─ Lỗ tối đa: **{position_info.get('max_loss', 0):,.0f}** VNĐ
└─ Risk/Share: {position_info.get('risk_per_share', 0):,.0f} VNĐ
"""
        # Footer
        msg += f"\n⏰ Thời gian: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        return msg.strip()