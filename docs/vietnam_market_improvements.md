# Vietnam Market Trading Improvements - Đề Xuất Cải Tiến

## 1. T+2.5 Settlement Tracking (HIGH PRIORITY)

### Vấn đề hiện tại
- Chưa track pending settlements properly
- Có thể mua quá khả năng thanh toán

### Giải pháp đề xuất
```python
# Thêm vào src/portfolio/settlement.py
class SettlementTracker:
    """Track T+2 settlement for Vietnam market"""
    
    def __init__(self):
        self.pending_settlements = {}  # {date: amount}
        
    def record_buy(self, amount: float, trade_date: date):
        """Record pending settlement from buy order"""
        settlement_date = self._get_settlement_date(trade_date)
        self.pending_settlements[settlement_date] = \
            self.pending_settlements.get(settlement_date, 0) + amount
            
    def get_available_cash(self, total_cash: float) -> float:
        """Get cash available for new trades (excluding pending)"""
        today = date.today()
        pending = sum(
            amt for dt, amt in self.pending_settlements.items() 
            if dt > today
        )
        return max(0, total_cash - pending)
```

## 2. Enhanced Gap Protection (HIGH PRIORITY)

### Vấn đề hiện tại
- Gap down -2.5% có thể quá loose
- Chưa có gap up profit taking

### Giải pháp đề xuất
```python
# Cải tiến trong exit_logic.py
VN_GAP_DOWN_THRESHOLDS = {
    "PROFITABLE": -0.02,    # -2% gap down khi có lời → exit
    "BREAKEVEN": -0.025,    # -2.5% gap down khi hòa vốn → exit  
    "LOSING": -0.03,        # -3% gap down khi đang lỗ → hold (đã lỗ rồi)
    "EMERGENCY": -0.05,     # -5% gap down → emergency exit bất kể P&L
}

VN_GAP_UP_THRESHOLDS = {
    "PROFIT_TAKE_50": 0.04,  # +4% gap up → chốt 50%
    "PROFIT_TAKE_ALL": 0.06, # +6% gap up → chốt hết (near ceiling)
}
```

## 3. Sector Rotation Strategy (MEDIUM PRIORITY)

### Vấn đề hiện tại
- Có sector mapping nhưng chưa tận dụng

### Giải pháp đề xuất
```python
# Thêm vào src/market/sector_rotation.py
class SectorRotationAnalyzer:
    """Analyze sector momentum for rotation strategy"""
    
    SECTOR_INDICES = {
        "BANKING": ["VCB", "BID", "CTG", "TCB", "MBB"],
        "REAL_ESTATE": ["VHM", "VIC", "NVL", "DXG"],
        "TECHNOLOGY": ["FPT", "CMG"],
        # ...
    }
    
    def get_sector_momentum(self, lookback_days: int = 20) -> Dict[str, float]:
        """Calculate momentum score for each sector"""
        # Return: {"BANKING": 0.8, "REAL_ESTATE": -0.3, ...}
        
    def get_rotation_signal(self) -> Dict:
        """Get sector rotation recommendation"""
        momentum = self.get_sector_momentum()
        
        # Overweight sectors with momentum > 0.5
        # Underweight sectors with momentum < -0.3
        return {
            "overweight": [s for s, m in momentum.items() if m > 0.5],
            "underweight": [s for s, m in momentum.items() if m < -0.3],
            "neutral": [s for s, m in momentum.items() if -0.3 <= m <= 0.5],
        }
```

## 4. Intraday T+0 Trading (MEDIUM PRIORITY)

### Vấn đề hiện tại
- Constants có VN_T0_ENABLED nhưng chưa implement

### Giải pháp đề xuất
```python
# Thêm vào src/portfolio/intraday_trading.py
class IntradayManager:
    """Manage T+0 intraday trading for margin accounts"""
    
    def __init__(self, margin_manager: MarginManager):
        self.margin_manager = margin_manager
        self.intraday_trades = []
        self.daily_pnl = 0.0
        
    def can_open_intraday(self, symbol: str, quantity: int, price: float) -> Tuple[bool, str]:
        """Check if can open intraday position"""
        # Check 1: Account value >= 50M VND
        # Check 2: Daily trades < 20
        # Check 3: Daily loss < 2%
        # Check 4: Min holding time 5 minutes
        
    def close_all_intraday(self):
        """Force close all intraday positions before market close"""
        # Must close before 14:30 (ATC)
```

## 5. Foreign Flow Real-time Integration (MEDIUM PRIORITY)

### Vấn đề hiện tại
- Có analyzer nhưng chưa integrate vào entry logic

### Giải pháp đề xuất
```python
# Cải tiến trong entry_logic.py
def _apply_foreign_flow_filter(self, ...):
    """Apply foreign flow signal to entry decision"""
    from src.market.foreign_flow import get_foreign_flow_analyzer
    
    analyzer = get_foreign_flow_analyzer()
    
    # Use adjusted score (accounts for staleness)
    score = analyzer.get_adjusted_score(max_delay_minutes=15)
    
    if score > 0.5:
        # Strong foreign buying → bonus confidence
        self._add_adjustment(adjustments, breakdown, "foreign_flow", +10, 
                           "Strong foreign net buying")
    elif score < -0.5:
        # Strong foreign selling → penalty
        self._add_adjustment(adjustments, breakdown, "foreign_flow", -15,
                           "Strong foreign net selling - smart money exiting")
```

## 6. Liquidity-Aware Order Execution (LOW PRIORITY)

### Vấn đề hiện tại
- Chưa có TWAP/VWAP execution cho large orders

### Giải pháp đề xuất
```python
# Thêm vào src/execution/smart_order.py
class SmartOrderExecutor:
    """Smart order execution for Vietnam market"""
    
    def execute_large_order(
        self, 
        symbol: str, 
        quantity: int, 
        side: str,
        max_participation_rate: float = 0.05  # Max 5% of volume
    ) -> List[Order]:
        """Split large order into smaller chunks"""
        
        avg_volume = self._get_avg_volume(symbol)
        max_per_order = int(avg_volume * max_participation_rate)
        
        # Round to lot size
        max_per_order = (max_per_order // 100) * 100
        
        orders = []
        remaining = quantity
        
        while remaining > 0:
            order_qty = min(remaining, max_per_order)
            orders.append(self._place_order(symbol, order_qty, side))
            remaining -= order_qty
            
            # Wait between orders to avoid market impact
            time.sleep(30)  # 30 seconds
            
        return orders
```

## 7. Enhanced ATO/ATC Handling (LOW PRIORITY)

### Vấn đề hiện tại
- Có penalty nhưng chưa có strategy tận dụng ATO/ATC

### Giải pháp đề xuất
```python
# Thêm vào src/strategies/auction_strategy.py
class AuctionStrategy:
    """Strategy for ATO/ATC auction sessions"""
    
    def should_use_ato(self, symbol: str, signal: Dict) -> bool:
        """Determine if should use ATO order"""
        # Use ATO when:
        # 1. Strong overnight news/catalyst
        # 2. Gap up expected (foreign buying overnight)
        # 3. High confidence signal (>80%)
        
    def should_use_atc(self, symbol: str, position: Dict) -> bool:
        """Determine if should use ATC order"""
        # Use ATC when:
        # 1. Want to close position at closing price
        # 2. Avoid overnight risk
        # 3. Friday afternoon (T+2 over weekend)
```

## Implementation Priority

| Priority | Feature | Effort | Impact | Status |
|----------|---------|--------|--------|--------|
| HIGH | T+2.5 Settlement Tracking | Medium | High | ✅ DONE |
| HIGH | Enhanced Gap Protection | Low | High | ✅ DONE |
| MEDIUM | Sector Rotation | Medium | Medium | ✅ DONE |
| MEDIUM | Smart Order Execution (TWAP/VWAP) | High | Medium | ✅ DONE |
| MEDIUM | ATO/ATC Strategy | Medium | Medium | ✅ DONE |
| MEDIUM | Foreign Flow Real-time | Medium | Medium | ✅ DONE |
| LOW | T+0 Intraday Trading | High | Low | ❌ SKIPPED |

## Implemented Features

### 1. Smart Order Execution (`src/execution/smart_order.py`)
- **TWAP**: Time-Weighted Average Price - splits orders over time
- **VWAP**: Volume-Weighted Average Price - follows volume profile
- **Iceberg**: Hidden size orders
- **Participation Rate**: Limits market impact to % of volume

### 2. ATO/ATC Auction Strategy (`src/strategies/auction_strategy.py`)
- Analyze overnight news and global markets for ATO
- Check intraday momentum for ATC
- Friday exit recommendations (weekend risk)
- Foreign flow integration

### 3. Foreign Flow Provider (`src/data/foreign_flow_provider.py`)
- Multiple data sources (TCBS, SSI, CafeF)
- Automatic failover
- Real-time background refresh
- Caching with TTL

## Kết Luận

Project đã có nền tảng hoàn chỉnh cho Vietnam market với:
- ✅ Lot size, tick size, price limits
- ✅ Transaction costs realistic (1.48% round trip)
- ✅ Circuit breaker với sector awareness
- ✅ Regime-aware position sizing
- ✅ T+2.5 settlement tracking
- ✅ Gap protection logic
- ✅ Sector rotation strategy
- ✅ Foreign flow integration
- ✅ Smart order execution (TWAP/VWAP)
- ✅ ATO/ATC auction strategy
