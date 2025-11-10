"""
Portfolio Management - Quản lý danh mục đầu tư
"""

import pandas as pd
import json
import os
from datetime import datetime

class PortfolioManager:
    def __init__(self, initial_capital=100_000_000, max_position_size=0.3):
        """
        initial_capital: Vốn ban đầu
        max_position_size: Tỷ lệ tối đa mỗi cổ phiếu (30%)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_size = max_position_size
        self.positions = {}  # {symbol: {shares, avg_price, value}}
        self.transaction_history = []
        self.portfolio_file = 'portfolio.json'
        
        # Load portfolio if exists
        self.load_portfolio()
    
    def buy(self, symbol, price, confidence, max_amount=None):
        """
        Mua cổ phiếu
        
        Returns:
            dict: Thông tin giao dịch hoặc None nếu không mua
        """
        if max_amount is None:
            # Tính số tiền tối đa có thể mua (dựa trên confidence)
            max_amount = self.cash * self.max_position_size * (confidence / 100)
        
        # Số cổ phiếu có thể mua
        shares = int(max_amount / price)
        
        if shares <= 0 or max_amount > self.cash:
            return None
        
        cost = shares * price
        
        # Cập nhật cash
        self.cash -= cost
        
        # Cập nhật position
        if symbol in self.positions:
            old_shares = self.positions[symbol]['shares']
            old_avg_price = self.positions[symbol]['avg_price']
            
            new_shares = old_shares + shares
            new_avg_price = ((old_shares * old_avg_price) + cost) / new_shares
            
            self.positions[symbol] = {
                'shares': new_shares,
                'avg_price': new_avg_price,
                'current_price': price,
                'value': new_shares * price,
                'pnl': (price - new_avg_price) * new_shares,
                'pnl_pct': ((price - new_avg_price) / new_avg_price) * 100
            }
        else:
            self.positions[symbol] = {
                'shares': shares,
                'avg_price': price,
                'current_price': price,
                'value': shares * price,
                'pnl': 0,
                'pnl_pct': 0
            }
        
        # Lưu transaction
        transaction = {
            'date': datetime.now().isoformat(),
            'type': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'value': cost,
            'confidence': confidence
        }
        self.transaction_history.append(transaction)
        
        self.save_portfolio()
        
        return transaction
    
    def sell(self, symbol, price, shares=None):
        """
        Bán cổ phiếu
        
        shares: Số cổ phiếu bán (None = bán hết)
        
        Returns:
            dict: Thông tin giao dịch hoặc None nếu không bán
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        if shares is None:
            shares = position['shares']
        
        if shares > position['shares']:
            shares = position['shares']
        
        revenue = shares * price
        
        # Cập nhật cash
        self.cash += revenue
        
        # Tính P&L
        pnl = (price - position['avg_price']) * shares
        pnl_pct = ((price - position['avg_price']) / position['avg_price']) * 100
        
        # Cập nhật hoặc xóa position
        if shares == position['shares']:
            del self.positions[symbol]
        else:
            self.positions[symbol]['shares'] -= shares
            self.positions[symbol]['value'] = self.positions[symbol]['shares'] * price
            self.positions[symbol]['current_price'] = price
        
        # Lưu transaction
        transaction = {
            'date': datetime.now().isoformat(),
            'type': 'SELL',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'value': revenue,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }
        self.transaction_history.append(transaction)
        
        self.save_portfolio()
        
        return transaction
    
    def update_prices(self, prices_dict):
        """
        Cập nhật giá hiện tại cho tất cả positions
        
        prices_dict: {symbol: current_price}
        """
        for symbol, price in prices_dict.items():
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos['current_price'] = price
                pos['value'] = pos['shares'] * price
                pos['pnl'] = (price - pos['avg_price']) * pos['shares']
                pos['pnl_pct'] = ((price - pos['avg_price']) / pos['avg_price']) * 100
        
        self.save_portfolio()
    
    def get_portfolio_value(self):
        """Tính tổng giá trị danh mục"""
        positions_value = sum(pos['value'] for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_portfolio_summary(self):
        """Lấy tóm tắt danh mục"""
        total_value = self.get_portfolio_value()
        positions_value = sum(pos['value'] for pos in self.positions.values())
        total_pnl = sum(pos['pnl'] for pos in self.positions.values())
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_pnl': total_pnl,
            'total_return': ((total_value - self.initial_capital) / self.initial_capital) * 100,
            'positions': self.positions,
            'num_positions': len(self.positions)
        }
    
    def print_portfolio(self):
        """In thông tin danh mục"""
        summary = self.get_portfolio_summary()
        
        print("\n" + "="*70)
        print("💼 DANH MỤC ĐẦU TƯ")
        print("="*70)
        print(f"💰 Tổng giá trị:       {summary['total_value']:>20,.0f} VNĐ")
        print(f"💵 Tiền mặt:           {summary['cash']:>20,.0f} VNĐ")
        print(f"📊 Giá trị cổ phiếu:   {summary['positions_value']:>20,.0f} VNĐ")
        print(f"📈 Lợi nhuận:          {summary['total_pnl']:>20,.0f} VNĐ ({summary['total_return']:>6.2f}%)")
        print(f"📋 Số cổ phiếu nắm:    {summary['num_positions']:>20}")
        print("="*70)
        
        if len(self.positions) > 0:
            print("\n📊 CHI TIẾT VỊ THẾ:")
            print("-"*70)
            
            for symbol, pos in self.positions.items():
                print(f"\n{symbol}:")
                print(f"  Số CP:        {pos['shares']:>10,}")
                print(f"  Giá TB:       {pos['avg_price']:>10,.0f} VNĐ")
                print(f"  Giá HT:       {pos['current_price']:>10,.0f} VNĐ")
                print(f"  Giá trị:      {pos['value']:>10,.0f} VNĐ")
                print(f"  P&L:          {pos['pnl']:>10,.0f} VNĐ ({pos['pnl_pct']:>6.2f}%)")
        
        print("\n" + "="*70 + "\n")
    
    def save_portfolio(self):
        """Lưu portfolio vào file"""
        data = {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': self.positions,
            'transaction_history': self.transaction_history
        }
        
        with open(self.portfolio_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_portfolio(self):
        """Load portfolio từ file"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                    self.cash = data.get('cash', self.initial_capital)
                    self.positions = data.get('positions', {})
                    self.transaction_history = data.get('transaction_history', [])
                print("✅ Đã load portfolio từ file")
            except Exception as e:
                print(f"⚠️ Không load được portfolio: {e}")
    
    def reset_portfolio(self):
        """Reset portfolio về ban đầu"""
        self.cash = self.initial_capital
        self.positions = {}
        self.transaction_history = []
        self.save_portfolio()
        print("✅ Đã reset portfolio")