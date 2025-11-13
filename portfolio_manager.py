"""
Portfolio Manager - Quản lý portfolio với SQLite
Thay thế JSON files bằng database
"""
from datetime import datetime
from typing import Dict, List, Optional
from database import get_db
from monitoring import get_performance_monitor
from trading_config import get_config


class PortfolioManager:
    """
    Quản lý portfolio với SQLite database
    
    Features:
    - Lưu positions vào database thay vì JSON
    - Track performance metrics
    - Portfolio history
    """
    
    def __init__(self):
        self.db = get_db()
        self.monitor = get_performance_monitor()
        self.config = get_config()
    
    def get_positions(self) -> Dict:
        """Get all active positions"""
        return self.db.get_positions()
    
    def add_position(self, symbol: str, shares: int, entry_price: float,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None,
                    metadata: Optional[Dict] = None):
        """Add new position"""
        entry_date = datetime.now().isoformat()
        entry_value = shares * entry_price
        
        self.db.save_position(
            symbol=symbol,
            shares=shares,
            avg_price=entry_price,
            entry_date=entry_date,
            entry_value=entry_value,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata
        )
        
        # Log trade
        self.db.save_trade(
            symbol=symbol,
            action='BUY',
            shares=shares,
            price=entry_price,
            total_value=entry_value,
            trade_date=entry_date,
            reason='Entry signal',
            metadata=metadata
        )
        
        print(f"✅ Added position: {symbol} - {shares} shares @ {entry_price:,.0f}")
    
    def close_position(self, symbol: str, exit_price: float, reason: str = 'Exit signal'):
        """Close position"""
        positions = self.db.get_positions()
        
        if symbol not in positions:
            print(f"⚠️ Position {symbol} not found")
            return
        
        pos = positions[symbol]
        shares = pos['shares']
        entry_price = pos['avg_price']
        entry_date = pos['entry_date']
        
        # Calculate P&L
        exit_value = shares * exit_price
        entry_value = shares * entry_price
        pnl = exit_value - entry_value
        pnl_percent = (pnl / entry_value) * 100
        
        # Track in performance monitor
        self.monitor.track_trade(
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares,
            entry_date=entry_date,
            exit_date=datetime.now().isoformat()
        )
        
        # Log trade
        self.db.save_trade(
            symbol=symbol,
            action='SELL',
            shares=shares,
            price=exit_price,
            total_value=exit_value,
            trade_date=datetime.now().isoformat(),
            reason=reason,
            metadata={'pnl': pnl, 'pnl_percent': pnl_percent}
        )
        
        # Delete position
        self.db.delete_position(symbol)
        
        print(f"✅ Closed position: {symbol} - P&L: {pnl:+,.0f} ({pnl_percent:+.1f}%)")
    
    def update_position_price(self, symbol: str, current_price: float):
        """Update current price for position"""
        positions = self.db.get_positions()
        
        if symbol not in positions:
            return
        
        pos = positions[symbol]
        metadata = pos.get('metadata', {})
        metadata['last_price'] = current_price
        metadata['last_updated'] = datetime.now().isoformat()
        
        self.db.save_position(
            symbol=symbol,
            shares=pos['shares'],
            avg_price=pos['avg_price'],
            entry_date=pos['entry_date'],
            entry_value=pos['entry_value'],
            stop_loss=pos.get('stop_loss'),
            take_profit=pos.get('take_profit'),
            metadata=metadata
        )
    
    def get_portfolio_value(self) -> Dict:
        """Calculate current portfolio value"""
        positions = self.db.get_positions()
        
        total_value = 0
        total_cost = 0
        
        for symbol, pos in positions.items():
            shares = pos['shares']
            entry_price = pos['avg_price']
            current_price = pos.get('metadata', {}).get('last_price', entry_price)
            
            total_cost += shares * entry_price
            total_value += shares * current_price
        
        pnl = total_value - total_cost
        pnl_percent = (pnl / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'num_positions': len(positions)
        }
    
    def save_portfolio_snapshot(self):
        """Save current portfolio snapshot"""
        portfolio = self.get_portfolio_value()
        
        self.db.save_portfolio_snapshot(
            date=datetime.now().isoformat(),
            total_value=portfolio['total_value'],
            total_cost=portfolio['total_cost'],
            pnl=portfolio['pnl'],
            pnl_percent=portfolio['pnl_percent'],
            num_positions=portfolio['num_positions']
        )
        
        print(f"📸 Saved portfolio snapshot: {portfolio['total_value']:,.0f} VNĐ")
    
    def get_detailed_analysis(self) -> str:
        """Get detailed portfolio analysis"""
        positions = self.db.get_positions()
        portfolio = self.get_portfolio_value()
        metrics = self.monitor.get_metrics()
        
        lines = []
        lines.append("📊 *PORTFOLIO ANALYSIS*")
        lines.append("=" * 40)
        
        # Portfolio summary
        lines.append(f"\n💰 *Portfolio Value:* {portfolio['total_value']:,.0f} VNĐ")
        lines.append(f"💵 *Total Cost:* {portfolio['total_cost']:,.0f} VNĐ")
        lines.append(f"📈 *P&L:* {portfolio['pnl']:+,.0f} VNĐ ({portfolio['pnl_percent']:+.1f}%)")
        lines.append(f"📦 *Positions:* {portfolio['num_positions']}")
        
        # Individual positions
        if positions:
            lines.append(f"\n🎯 *POSITIONS:*")
            for symbol, pos in positions.items():
                shares = pos['shares']
                entry_price = pos['avg_price']
                current_price = pos.get('metadata', {}).get('last_price', entry_price)
                
                pos_value = shares * current_price
                pos_cost = shares * entry_price
                pos_pnl = pos_value - pos_cost
                pos_pnl_pct = (pos_pnl / pos_cost * 100) if pos_cost > 0 else 0
                
                lines.append(f"• {symbol}: {shares:,} CP @ {entry_price:,.0f}")
                lines.append(f"  Current: {current_price:,.0f} | P&L: {pos_pnl:+,.0f} ({pos_pnl_pct:+.1f}%)")
        
        # Performance metrics
        if metrics['total_trades'] > 0:
            lines.append(f"\n📊 *PERFORMANCE:*")
            lines.append(f"• Total Trades: {metrics['total_trades']}")
            lines.append(f"• Win Rate: {metrics['win_rate']:.1f}%")
            lines.append(f"• Avg Profit: {metrics['avg_profit']:,.0f} VNĐ")
            lines.append(f"• Avg Loss: {metrics['avg_loss']:,.0f} VNĐ")
            lines.append(f"• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        return "\n".join(lines)


# Singleton
_manager = None

def get_portfolio_manager() -> PortfolioManager:
    """Get portfolio manager singleton"""
    global _manager
    if _manager is None:
        _manager = PortfolioManager()
    return _manager


if __name__ == "__main__":
    print("Testing Portfolio Manager...")
    
    manager = PortfolioManager()
    
    # Test add position
    manager.add_position('VCB', 100, 60000, stop_loss=57000, take_profit=66000)
    
    # Test get positions
    positions = manager.get_positions()
    print(f"Positions: {positions}")
    
    # Test portfolio value
    portfolio = manager.get_portfolio_value()
    print(f"Portfolio: {portfolio}")
    
    # Test analysis
    analysis = manager.get_detailed_analysis()
    print(analysis)
    
    print("\n✅ Portfolio Manager test completed!")
