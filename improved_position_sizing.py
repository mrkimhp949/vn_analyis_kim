# -*- coding: utf-8 -*-
"""
improved_position_sizing.py - Conservative Position Sizing
Quản lý vốn an toàn hơn cho người mới bắt đầu
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Container cho kết quả position sizing"""
    shares: int
    value: float
    risk_amount: float
    risk_percent: float
    max_loss: float
    position_percent: float
    recommended_entries: list  # List các mức giá để DCA
    warnings: list


class ConservativePositionSizer:
    """
    Position sizing theo nguyên tắc BẢO TOÀN VỐN
    
    Nguyên tắc:
    1. Không bao giờ risk >2% vốn cho 1 lệnh
    2. Không bao giờ đầu tư >10% vốn vào 1 mã
    3. Tổng exposure không quá 60% vốn
    4. Phải có ít nhất 8-10 mã khác nhau
    5. Scale position theo confidence và market regime
    """
    
    def __init__(self,
                 total_capital: float = 100_000_000,
                 max_risk_per_trade: float = 0.02,      # 2% max risk
                 max_position_size: float = 0.10,        # 10% max position
                 max_total_exposure: float = 0.60,       # 60% max trong thị trường
                 min_positions: int = 8,                 # Ít nhất 8 mã
                 emergency_stop_loss: float = 0.15):     # Cut all nếu mất 15%
        
        self.total_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.min_positions = min_positions
        self.emergency_stop_loss = emergency_stop_loss
        
        # Tracking
        self.current_positions = {}  # {symbol: position_data}
        self.realized_pnl = 0
        self.unrealized_pnl = 0
    
    def calculate_position_size(self,
                               symbol: str,
                               entry_price: float,
                               stop_loss: float,
                               confidence: int,
                               signal_strength: str = 'MODERATE',
                               market_regime: Optional[Dict] = None) -> PositionSize:
        """
        Tính position size an toàn
        
        Args:
            symbol: Mã cổ phiếu
            entry_price: Giá dự kiến vào
            stop_loss: Giá stop loss
            confidence: 0-100
            signal_strength: 'VERY_STRONG', 'STRONG', 'MODERATE', 'WEAK'
            market_regime: Market regime info
            
        Returns:
            PositionSize object
        """
        
        warnings = []
        
        # =================================================================
        # CHECK 1: Có còn vốn để vào lệnh không?
        # =================================================================
        current_exposure = self._calculate_current_exposure()
        available_capital = self.total_capital * self.max_total_exposure - current_exposure
        
        if available_capital <= 0:
            logger.warning(f"⚠️ Đã dùng hết {self.max_total_exposure*100}% vốn")
            return self._zero_position(f"Exposure đã đạt limit ({current_exposure:,.0f} VNĐ)")
        
        # =================================================================
        # CHECK 2: Risk per trade
        # =================================================================
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return self._zero_position("Stop loss không hợp lệ")
        
        # Base risk amount (2% vốn)
        base_risk_amount = self.total_capital * self.max_risk_per_trade
        
        # Adjust risk theo confidence và signal strength
        risk_multiplier = self._calculate_risk_multiplier(
            confidence, 
            signal_strength, 
            market_regime
        )
        
        adjusted_risk_amount = base_risk_amount * risk_multiplier
        
        # Số cổ phiếu dựa trên risk
        shares_by_risk = int(adjusted_risk_amount / risk_per_share)
        
        # =================================================================
        # CHECK 3: Position size limit (10% vốn)
        # =================================================================
        max_shares_by_capital = int(
            (self.total_capital * self.max_position_size) / entry_price
        )
        
        # =================================================================
        # CHECK 4: Available capital
        # =================================================================
        max_shares_by_available = int(available_capital / entry_price)
        
        # =================================================================
        # TAKE THE MINIMUM
        # =================================================================
        shares = min(shares_by_risk, max_shares_by_capital, max_shares_by_available)
        
        # Round to lot of 100
        if shares > 0:
            shares = max((shares // 100) * 100, 100)
        else:
            return self._zero_position("Position size = 0 sau khi tính toán")
        
        # =================================================================
        # CHECK 5: Diversification
        # =================================================================
        num_positions = len(self.current_positions) + 1  # +1 for this new position
        
        if num_positions < self.min_positions:
            # Nếu chưa đủ diversification, giảm position size
            diversification_factor = num_positions / self.min_positions
            
            if diversification_factor < 0.8:
                shares = int(shares * 0.7)  # Giảm 30%
                shares = max((shares // 100) * 100, 100)
                warnings.append(
                    f"⚠️ Giảm 30% position vì chưa đa dạng hóa "
                    f"({num_positions}/{self.min_positions} mã)"
                )
        
        # =================================================================
        # CALCULATE FINAL VALUES
        # =================================================================
        position_value = shares * entry_price
        position_percent = (position_value / self.total_capital) * 100
        max_loss = shares * risk_per_share
        risk_percent = (max_loss / self.total_capital) * 100
        
        # =================================================================
        # CHECK 6: Final safety checks
        # =================================================================
        if risk_percent > self.max_risk_per_trade * 100:
            # Giảm shares để risk không vượt quá
            max_safe_shares = int(
                (self.total_capital * self.max_risk_per_trade) / risk_per_share
            )
            shares = min(shares, max_safe_shares)
            shares = max((shares // 100) * 100, 100)
            
            position_value = shares * entry_price
            position_percent = (position_value / self.total_capital) * 100
            max_loss = shares * risk_per_share
            risk_percent = (max_loss / self.total_capital) * 100
            
            warnings.append(f"⚠️ Đã giảm shares để risk <= {self.max_risk_per_trade*100}%")
        
        # =================================================================
        # DCA ENTRIES (Dollar Cost Averaging)
        # =================================================================
        recommended_entries = self._calculate_dca_entries(entry_price, shares)
        
        # =================================================================
        # WARNINGS
        # =================================================================
        if position_percent > 8:
            warnings.append(f"⚠️ Position size cao: {position_percent:.1f}%")
        
        if risk_percent > 1.5:
            warnings.append(f"⚠️ Risk cao: {risk_percent:.2f}%")
        
        return PositionSize(
            shares=shares,
            value=position_value,
            risk_amount=max_loss,
            risk_percent=risk_percent,
            max_loss=max_loss,
            position_percent=position_percent,
            recommended_entries=recommended_entries,
            warnings=warnings
        )
    
    def _calculate_risk_multiplier(self,
                                   confidence: int,
                                   signal_strength: str,
                                   market_regime: Optional[Dict]) -> float:
        """
        Tính multiplier cho risk amount dựa trên confidence và conditions
        
        Returns:
            0.5 - 1.2 (không bao giờ >1.2 để an toàn)
        """
        # Base multiplier từ confidence
        if confidence >= 80:
            base = 1.1
        elif confidence >= 70:
            base = 1.0
        elif confidence >= 60:
            base = 0.8
        else:
            base = 0.6
        
        # Adjust theo signal strength
        strength_multipliers = {
            'VERY_STRONG': 1.1,
            'STRONG': 1.0,
            'MODERATE': 0.9,
            'WEAK': 0.7,
            'VERY_WEAK': 0.5
        }
        
        strength_mult = strength_multipliers.get(signal_strength, 0.9)
        
        # Adjust theo market regime
        regime_mult = 1.0
        if market_regime:
            regime = market_regime.get('regime', 'SIDEWAYS')
            if regime == 'BULL':
                regime_mult = 1.1
            elif regime == 'BEAR':
                regime_mult = 0.5  # Giảm mạnh trong bear
            elif regime == 'HIGH_VOLATILITY':
                regime_mult = 0.6
            else:  # SIDEWAYS
                regime_mult = 0.8
        
        # Combine
        multiplier = base * strength_mult * regime_mult
        
        # Clamp to safe range
        return max(0.5, min(multiplier, 1.2))
    
    def _calculate_dca_entries(self, 
                              base_price: float, 
                              total_shares: int) -> list:
        """
        Tính các mức giá để DCA (Dollar Cost Averaging)
        
        Ví dụ:
        - Entry 1 (50%): base_price - 1%
        - Entry 2 (30%): base_price - 2%
        - Entry 3 (20%): base_price - 3%
        """
        entries = [
            {
                'level': 1,
                'price': round(base_price * 0.99, -2),  # -1%
                'shares': int((total_shares * 0.5 // 100) * 100),
                'percent': 50
            },
            {
                'level': 2,
                'price': round(base_price * 0.98, -2),  # -2%
                'shares': int((total_shares * 0.3 // 100) * 100),
                'percent': 30
            },
            {
                'level': 3,
                'price': round(base_price * 0.97, -2),  # -3%
                'shares': int((total_shares * 0.2 // 100) * 100),
                'percent': 20
            }
        ]
        
        return entries
    
    def _calculate_current_exposure(self) -> float:
        """Tính tổng value các positions hiện tại"""
        return sum(
            pos['shares'] * pos['current_price'] 
            for pos in self.current_positions.values()
        )
    
    def _zero_position(self, reason: str) -> PositionSize:
        """Return zero position"""
        return PositionSize(
            shares=0,
            value=0,
            risk_amount=0,
            risk_percent=0,
            max_loss=0,
            position_percent=0,
            recommended_entries=[],
            warnings=[reason]
        )
    
    # ========================================================================
    # PORTFOLIO MANAGEMENT
    # ========================================================================
    
    def add_position(self, symbol: str, shares: int, entry_price: float):
        """Thêm position mới"""
        self.current_positions[symbol] = {
            'shares': shares,
            'entry_price': entry_price,
            'current_price': entry_price,
            'unrealized_pnl': 0
        }
    
    def update_position_price(self, symbol: str, current_price: float):
        """Cập nhật giá hiện tại"""
        if symbol in self.current_positions:
            pos = self.current_positions[symbol]
            pos['current_price'] = current_price
            pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['shares']
    
    def close_position(self, symbol: str, exit_price: float):
        """Đóng position"""
        if symbol in self.current_positions:
            pos = self.current_positions[symbol]
            pnl = (exit_price - pos['entry_price']) * pos['shares']
            self.realized_pnl += pnl
            del self.current_positions[symbol]
            
            logger.info(f"✅ Đã đóng {symbol}: PnL = {pnl:,.0f} VNĐ")
    
    def get_portfolio_status(self) -> Dict:
        """Lấy status portfolio"""
        total_value = self._calculate_current_exposure()
        cash = self.total_capital - total_value
        
        # Tính unrealized PnL
        unrealized_pnl = sum(
            pos['unrealized_pnl'] for pos in self.current_positions.values()
        )
        
        total_pnl = self.realized_pnl + unrealized_pnl
        total_return = (total_pnl / self.total_capital) * 100
        
        # Check emergency stop
        emergency_triggered = total_return <= -self.emergency_stop_loss * 100
        
        return {
            'total_capital': self.total_capital,
            'cash': cash,
            'invested': total_value,
            'exposure_percent': (total_value / self.total_capital) * 100,
            'num_positions': len(self.current_positions),
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'emergency_triggered': emergency_triggered,
            'positions': self.current_positions
        }
    
    def print_portfolio_status(self):
        """In portfolio status"""
        status = self.get_portfolio_status()
        
        print("\n" + "="*70)
        print("💼 PORTFOLIO STATUS")
        print("="*70)
        print(f"💰 Vốn:            {status['total_capital']:>20,.0f} VNĐ")
        print(f"💵 Cash:           {status['cash']:>20,.0f} VNĐ")
        print(f"📊 Đã đầu tư:      {status['invested']:>20,.0f} VNĐ ({status['exposure_percent']:.1f}%)")
        print(f"📋 Số vị thế:      {status['num_positions']:>20}")
        print(f"💹 Realized PnL:   {status['realized_pnl']:>20,.0f} VNĐ")
        print(f"📈 Unrealized PnL: {status['unrealized_pnl']:>20,.0f} VNĐ")
        print(f"🎯 Total Return:   {status['total_return']:>20,.2f}%")
        
        if status['emergency_triggered']:
            print(f"\n🚨 CẢNH BÁO: ĐÃ MẤT {abs(status['total_return']):.1f}%")
            print(f"⛔ KHUYẾN NGHỊ: ĐÓNG TẤT CẢ VỊ THẾ VÀ ĐÁNH GIÁ LẠI CHIẾN LƯỢC")
        
        print("="*70)
    
    def format_position_message(self, 
                               symbol: str, 
                               position: PositionSize,
                               entry_price: float) -> str:
        """Format position size thành message"""
        
        if position.shares == 0:
            return f"⏭️ **{symbol}** - Không vào lệnh\n" \
                   f"Lý do: {', '.join(position.warnings)}"
        
        msg = f"💰 **POSITION SIZING - {symbol}**\n\n"
        
        msg += f"📊 **Size:**\n"
        msg += f"  • Shares: {position.shares:,} ({position.shares//100} lô)\n"
        msg += f"  • Value: {position.value:,.0f} VNĐ\n"
        msg += f"  • % Portfolio: {position.position_percent:.2f}%\n\n"
        
        msg += f"🎲 **Risk:**\n"
        msg += f"  • Max Loss: {position.max_loss:,.0f} VNĐ\n"
        msg += f"  • Risk %: {position.risk_percent:.2f}%\n\n"
        
        if position.recommended_entries:
            msg += f"📍 **DCA Entries (Khuyến nghị):**\n"
            for entry in position.recommended_entries:
                msg += f"  {entry['level']}. {entry['price']:,.0f} VNĐ: "
                msg += f"{entry['shares']:,} shares ({entry['percent']}%)\n"
            msg += "\n"
        
        if position.warnings:
            msg += f"⚠️ **Warnings:**\n"
            for warning in position.warnings:
                msg += f"  • {warning}\n"
        
        return msg


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING CONSERVATIVE POSITION SIZER")
    print("="*70 + "\n")
    
    # Initialize với vốn 100 triệu
    sizer = ConservativePositionSizer(
        total_capital=100_000_000,
        max_risk_per_trade=0.02,
        max_position_size=0.10,
        max_total_exposure=0.60
    )
    
    # Test case 1: High confidence, strong signal
    print("\n📊 TEST CASE 1: High Confidence Signal")
    position1 = sizer.calculate_position_size(
        symbol='VNM',
        entry_price=80_000,
        stop_loss=76_000,  # -5%
        confidence=85,
        signal_strength='VERY_STRONG',
        market_regime={'regime': 'BULL', 'tradeable': True}
    )
    
    print(sizer.format_position_message('VNM', position1, 80_000))
    
    # Add position
    sizer.add_position('VNM', position1.shares, 80_000)
    
    # Test case 2: Medium confidence
    print("\n📊 TEST CASE 2: Medium Confidence Signal")
    position2 = sizer.calculate_position_size(
        symbol='VCB',
        entry_price=90_000,
        stop_loss=85_500,  # -5%
        confidence=65,
        signal_strength='MODERATE',
        market_regime={'regime': 'SIDEWAYS', 'tradeable': True}
    )
    
    print(sizer.format_position_message('VCB', position2, 90_000))
    
    # Add position
    sizer.add_position('VCB', position2.shares, 90_000)
    
    # Print portfolio status
    sizer.print_portfolio_status()
    
    print("\n✅ Testing complete!")