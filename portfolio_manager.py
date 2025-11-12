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
from typing import Optional
from portfolio_analyzer import PortfolioAnalyzer
from portfolio_history import PortfolioHistoryTracker
from risk_metrics import (
    calculate_sector_exposure,
    check_sector_overweight,
    calculate_portfolio_correlation_risk,
    get_diversification_recommendation
)

class PortfolioManager:
    def __init__(self, portfolio_file='my_portfolio.json'):
        self.portfolio_file = portfolio_file
        self.analyzer = PortfolioAnalyzer()
        self.history_tracker = PortfolioHistoryTracker()
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
        
        # Record history
        self._record_daily_snapshot()
        
        print(f"✅ Đã thêm {shares} CP {symbol} vào portfolio")
    
    def remove_stock(self, symbol, shares=None, price=None, date=None):
        """Bán cổ phiếu khỏi portfolio

        Returns:
            tuple[bool, str]: (success, message)
        """
        if date is None:
            date = datetime.now().isoformat()
        
        if symbol not in self.portfolio:
            msg = f"Không có {symbol} trong portfolio"
            print(f"❌ {msg}")
            return False, msg
        
        current = self.portfolio[symbol]
        
        if shares is None:
            shares = current['shares']  # Bán hết
        
        if shares > current['shares']:
            msg = f"Không đủ cổ phiếu để bán (hiện có {current['shares']:,} CP)"
            print(f"❌ {msg}")
            return False, msg
        
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
        
        # Record history
        self._record_daily_snapshot()
        
        msg = f"Đã bán {shares:,} CP {symbol}"
        print(f"✅ {msg}")
        return True, msg
    
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
        """Lấy tổng quan portfolio với risk metrics"""
        holdings = self.get_current_holdings()
        total_value = 0
        total_invested = 0
        
        # Lấy giá hiện tại và tính toán
        from data_loader import load_data
        
        holdings_with_prices = {}
        for symbol, holding in holdings.items():
            try:
                df = load_data(symbol, lookback=5)
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    current_value = holding['shares'] * current_price
                    invested_value = holding['shares'] * holding['avg_price']
                    
                    total_value += current_value
                    total_invested += invested_value
                    
                    holdings_with_prices[symbol] = {
                        **holding,
                        'current_price': current_price,
                        'current_value': current_value,
                        'entry_value': invested_value,
                        'pnl_amount': current_value - invested_value,
                        'pnl_percent': ((current_price - holding['avg_price']) / holding['avg_price'] * 100) if holding['avg_price'] > 0 else 0
                    }
            except:
                continue
        
        # Calculate sector exposure
        sector_exposure = calculate_sector_exposure(holdings_with_prices)
        
        # Check for overweight sectors
        overweight_sectors = check_sector_overweight(sector_exposure, max_sector_pct=40.0)
        
        # Calculate correlation risk
        symbols = list(holdings.keys())
        correlation_risk = calculate_portfolio_correlation_risk(symbols) if len(symbols) >= 2 else None
        
        return {
            'total_stocks': len(holdings),
            'total_portfolio_value': total_value,
            'total_invested': total_invested,
            'total_pnl': total_value - total_invested,
            'total_return_percent': ((total_value - total_invested) / total_invested * 100) if total_invested > 0 else 0,
            'holdings': holdings_with_prices,
            'sector_exposure': sector_exposure,
            'overweight_sectors': overweight_sectors,
            'correlation_risk': correlation_risk,
        }
    
    def _record_daily_snapshot(self):
        """Ghi lại snapshot portfolio trong ngày"""
        try:
            summary = self.get_portfolio_summary()
            holdings = summary.get('holdings', {})
            
            # Format holdings for history
            holdings_data = {}
            for symbol, data in holdings.items():
                holdings_data[symbol] = {
                    'shares': data.get('shares', 0),
                    'current_price': data.get('current_price', 0),
                    'value': data.get('current_value', 0),
                    'pnl': data.get('pnl_amount', 0),
                }
            
            portfolio_data = {
                'total_value': summary.get('total_portfolio_value', 0),
                'total_invested': summary.get('total_invested', 0),
                'total_pnl': summary.get('total_pnl', 0),
                'total_return_pct': summary.get('total_return_percent', 0),
                'num_positions': summary.get('total_stocks', 0),
                'holdings': holdings_data,
                'sector_exposure': summary.get('sector_exposure', {}),
            }
            
            self.history_tracker.record_daily_snapshot(portfolio_data)
        except Exception as e:
            print(f"⚠️ Lỗi ghi snapshot: {e}")
    
    def analyze_portfolio(self):
        """Phân tích portfolio và đề xuất"""
        current_holdings = self.get_current_holdings()
        return self.analyzer.analyze_current_portfolio(current_holdings)
    
    def get_detailed_analysis(self):
        """Lấy phân tích chi tiết với report formatted và risk metrics"""
        analysis = self.analyze_portfolio()
        report = self.analyzer.format_analysis_report(analysis)
        
        # Add risk metrics
        summary = self.get_portfolio_summary()
        diversification = get_diversification_recommendation(summary.get('holdings', {}))
        
        # Append risk section
        risk_section = "\n\n" + "="*60 + "\n"
        risk_section += "⚠️ RISK ANALYSIS\n"
        risk_section += "="*60 + "\n\n"
        
        # Sector exposure
        if summary.get('overweight_sectors'):
            risk_section += "📊 Sector Exposure:\n"
            for sector, pct in summary['overweight_sectors']:
                risk_section += f"  ⚠️ {sector}: {pct:.1f}% (vượt 40%)\n"
            risk_section += "\n"
        
        # Correlation risk
        if summary.get('correlation_risk'):
            corr_risk = summary['correlation_risk']
            risk_section += f"🔗 Correlation Risk:\n"
            risk_section += f"  Avg Correlation: {corr_risk.get('avg_correlation', 0):.2f}\n"
            risk_section += f"  Risk Score: {corr_risk.get('risk_score', 0)}/100\n"
            risk_section += f"  {corr_risk.get('recommendation', '')}\n\n"
        
        # Diversification recommendations
        if diversification.get('warnings'):
            risk_section += "💡 Recommendations:\n"
            for rec in diversification.get('recommendations', [])[:3]:
                risk_section += f"  • {rec}\n"
        
        return report + risk_section
    
    def get_performance_history(self, days: Optional[int] = None):
        """Lấy performance history"""
        return self.history_tracker.get_performance_metrics(days)
    
    def get_equity_curve(self, days: Optional[int] = None):
        """Lấy equity curve"""
        return self.history_tracker.get_equity_curve(days)
    
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