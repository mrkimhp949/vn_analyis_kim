# [file name]: portfolio_manager.py
# [file content begin]
# -*- coding: utf-8 -*-
"""
Portfolio Manager - Quản lý danh mục thực tế
Kết nối với tài khoản chứng khoán (giả lập)
"""

import json
import os
from datetime import datetime
from portfolio_analyzer import PortfolioAnalyzer

class PortfolioManager:
    def __init__(self, portfolio_file='my_portfolio.json'):
        self.portfolio_file = portfolio_file
        self.analyzer = PortfolioAnalyzer()
        self.load_portfolio()
    
    def load_portfolio(self):
        """Load portfolio từ file"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                    self.portfolio = json.load(f)
                print(f"✅ Đã load portfolio từ {self.portfolio_file}")
            except Exception as e:
                print(f"❌ Lỗi load portfolio: {e}")
                self.portfolio = {}
        else:
            self.portfolio = {}
            print("ℹ️ Chưa có portfolio, tạo mới")
    
    def save_portfolio(self):
        """Lưu portfolio vào file"""
        try:
            with open(self.portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(self.portfolio, f, indent=2, ensure_ascii=False)
            print(f"✅ Đã lưu portfolio vào {self.portfolio_file}")
        except Exception as e:
            print(f"❌ Lỗi lưu portfolio: {e}")
    
    def add_stock(self, symbol, shares, price, date=None):
        """Thêm cổ phiếu vào portfolio"""
        if date is None:
            date = datetime.now().isoformat()
        
        if symbol in self.portfolio:
            # Cập nhật nếu đã có
            current = self.portfolio[symbol]
            total_shares = current['shares'] + shares
            total_cost = (current['shares'] * current['avg_price']) + (shares * price)
            new_avg_price = total_cost / total_shares
            
            self.portfolio[symbol] = {
                'shares': total_shares,
                'avg_price': new_avg_price,
                'last_updated': date,
                'transactions': current.get('transactions', []) + [{
                    'date': date, 'type': 'BUY', 'shares': shares, 'price': price
                }]
            }
        else:
            # Thêm mới
            self.portfolio[symbol] = {
                'shares': shares,
                'avg_price': price,
                'last_updated': date,
                'transactions': [{
                    'date': date, 'type': 'BUY', 'shares': shares, 'price': price
                }]
            }
        
        self.save_portfolio()
        print(f"✅ Đã thêm {shares} CP {symbol} vào portfolio")
    
    def remove_stock(self, symbol, shares=None, price=None, date=None):
        """Bán cổ phiếu khỏi portfolio"""
        if date is None:
            date = datetime.now().isoformat()
        
        if symbol not in self.portfolio:
            print(f"❌ Không có {symbol} trong portfolio")
            return False
        
        current = self.portfolio[symbol]
        
        if shares is None:
            shares = current['shares']  # Bán hết
        
        if shares > current['shares']:
            print(f"❌ Không đủ cổ phiếu để bán")
            return False
        
        # Ghi nhận giao dịch
        transaction = {
            'date': date,
            'type': 'SELL', 
            'shares': shares,
            'price': price if price else 0
        }
        
        if 'transactions' in current:
            current['transactions'].append(transaction)
        else:
            current['transactions'] = [transaction]
        
        # Cập nhật số lượng
        if shares == current['shares']:
            # Bán hết -> xóa khỏi portfolio
            del self.portfolio[symbol]
        else:
            # Bán 1 phần
            current['shares'] -= shares
            current['last_updated'] = date
        
        self.save_portfolio()
        print(f"✅ Đã bán {shares} CP {symbol} khỏi portfolio")
        return True
    
    def get_current_holdings(self):
        """Lấy danh sách cổ phiếu đang nắm giữ"""
        return {
            symbol: {
                'shares': data['shares'],
                'avg_price': data['avg_price']
            }
            for symbol, data in self.portfolio.items()
        }
    
    def get_portfolio_summary(self):
        """Lấy tổng quan portfolio"""
        holdings = self.get_current_holdings()
        total_value = 0
        total_invested = 0
        
        # Lấy giá hiện tại và tính toán
        from data_loader import load_data
        
        for symbol, holding in holdings.items():
            try:
                df = load_data(symbol, lookback=5)
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    current_value = holding['shares'] * current_price
                    invested_value = holding['shares'] * holding['avg_price']
                    
                    total_value += current_value
                    total_invested += invested_value
            except:
                continue
        
        return {
            'total_stocks': len(holdings),
            'total_portfolio_value': total_value,
            'total_invested': total_invested,
            'total_pnl': total_value - total_invested,
            'total_return_percent': ((total_value - total_invested) / total_invested * 100) if total_invested > 0 else 0,
            'holdings': holdings
        }
    
    def analyze_portfolio(self):
        """Phân tích portfolio và đề xuất"""
        current_holdings = self.get_current_holdings()
        return self.analyzer.analyze_current_portfolio(current_holdings)
    
    def get_detailed_analysis(self):
        """Lấy phân tích chi tiết với report formatted"""
        analysis = self.analyze_portfolio()
        return self.analyzer.format_analysis_report(analysis)
    
    def print_portfolio_status(self):
        """In trạng thái portfolio"""
        summary = self.get_portfolio_summary()
        holdings = self.get_current_holdings()
        
        print("\n" + "="*60)
        print("💼 MY PORTFOLIO STATUS")
        print("="*60)
        
        print(f"📊 Tổng số mã: {summary['total_stocks']}")
        print(f"💰 Tổng giá trị: {summary['total_portfolio_value']:,.0f} VNĐ")
        print(f"💵 Tổng đầu tư: {summary['total_invested']:,.0f} VNĐ")
        print(f"📈 Lợi nhuận: {summary['total_pnl']:,.0f} VNĐ ({summary['total_return_percent']:.2f}%)")
        
        print(f"\n📋 Chi tiết holdings:")
        for symbol, holding in holdings.items():
            print(f"  • {symbol}: {holding['shares']:,} CP - Giá TB: {holding['avg_price']:,.0f}")

# Integration với Telegram
def send_portfolio_update_to_telegram():
    """Gửi cập nhật portfolio qua Telegram"""
    try:
        from telegram import Bot
        from config import TELEGRAM_TOKEN, CHAT_ID
        
        manager = PortfolioManager()
        analysis_report = manager.get_detailed_analysis()
        
        bot = Bot(token=TELEGRAM_TOKEN)
        
        # Chia nhỏ message nếu quá dài
        if len(analysis_report) > 4000:
            parts = [analysis_report[i:i+4000] for i in range(0, len(analysis_report), 4000)]
            for part in parts:
                bot.send_message(chat_id=CHAT_ID, text=part, parse_mode='Markdown')
        else:
            bot.send_message(chat_id=CHAT_ID, text=analysis_report, parse_mode='Markdown')
            
        print("✅ Đã gửi portfolio update qua Telegram")
        
    except Exception as e:
        print(f"❌ Lỗi gửi Telegram: {e}")

# Test
if __name__ == "__main__":
    print("🧪 Testing Portfolio Manager...")
    
    manager = PortfolioManager()
    
    # Thêm vài cổ phiếu mẫu
    manager.add_stock('ACB', 500, 25000)
    manager.add_stock('VNM', 300, 80000)
    manager.add_stock('HPG', 400, 45000)
    
    # In trạng thái
    manager.print_portfolio_status()
    
    # Phân tích
    print("\n" + "="*60)
    print("📊 PORTFOLIO ANALYSIS")
    print("="*60)
    analysis = manager.get_detailed_analysis()
    print(analysis)
# [file content end]