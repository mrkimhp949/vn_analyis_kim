# Risk Improvements v5.1

## Tổng Quan Các Rủi Ro Đã Xử Lý

### 1. ✅ Confidence Threshold Quá Thấp (45%)

**Vấn đề:** Confidence threshold 45% tạo nhiều false signals.

**Giải pháp:**
- Tăng minimum confidence lên 55% cho tất cả profiles
- File: `scripts/adjust_thresholds.py`

```python
# Trước
"aggressive": {"BULL": {"min_confidence": 40}, "SIDEWAYS": {"min_confidence": 45}}

# Sau  
"aggressive": {"BULL": {"min_confidence": 55}, "SIDEWAYS": {"min_confidence": 55}}
```

### 2. ✅ Max 3 Consecutive Losses - Sideway Market

**Vấn đề:** Quá aggressive trong sideway market, miss mean-reversion opportunities.

**Giải pháp:** Regime-aware consecutive loss limits
- File: `src/risk/circuit_breaker.py` - method `update_market_regime()`
- File: `src/risk/circuit_breaker_db.py` - `REGIME_CONSECUTIVE_LOSS_LIMITS`

```python
REGIME_CONSECUTIVE_LOSS_LIMITS = {
    "BULL": 4,      # More lenient
    "SIDEWAYS": 3,  # Standard (allows mean-reversion)
    "BEAR": 2,      # Strict
    "HIGH_VOLATILITY": 2,
}
```

### 3. ✅ Floor Bounce Wait 30 Minutes

**Vấn đề:** 30 phút có thể không đủ trong panic selling.

**Giải pháp:** Dynamic wait time với volume confirmation
- File: `src/utils/vietnam_market.py` - function `check_floor_bounce_enhanced()`
- File: `src/config/constants.py` - new constants

```python
VN_FLOOR_BOUNCE_MAX_WAIT_MINUTES = 30      # Base wait
VN_FLOOR_BOUNCE_EXTENDED_WAIT_MINUTES = 60 # High volatility
VN_FLOOR_BOUNCE_MIN_VOLUME_RATIO = 1.5     # Volume confirmation
VN_FLOOR_BOUNCE_PANIC_VOLUME_RATIO = 3.0   # Panic detection
```

### 4. ✅ DCA Strategy Không Phù Hợp VN Market

**Vấn đề:** Transaction cost 1.48% làm DCA không hiệu quả.

**Giải pháp:** Disable DCA by default
- File: `src/strategies/position_sizing.py`

```python
DCA_ENABLED: bool = False  # DISABLED for VN market
DCA_MIN_PROFIT_THRESHOLD: float = 0.03  # Tightened to 3%
```

### 5. ✅ Circuit Breaker File-Based (Single Point of Failure)

**Vấn đề:** JSON file có thể corrupt, race conditions với multiple bots.

**Giải pháp:** Database-backed storage với fallback
- File: `src/risk/circuit_breaker_db.py` - class `CircuitBreakerDB`

Features:
- SQLite database với ACID transactions
- Distributed locking cho multiple bots
- Automatic fallback to JSON file
- Historical data retention

### 6. ✅ Hardcoded VN30 Symbols

**Vấn đề:** VN30 thay đổi quarterly, cần manual update.

**Giải pháp:** Fetch từ API với caching
- File: `src/utils/vn30_fetcher.py` - class `VN30Fetcher`

Features:
- Multiple API sources (SSI, TCBS, VNDirect)
- Local caching (24h TTL)
- Fallback to hardcoded list
- Change detection for quarterly rebalancing

## Cách Sử Dụng

### 1. Regime-Aware Circuit Breaker

```python
from src.risk.circuit_breaker import get_circuit_breaker

breaker = get_circuit_breaker()
breaker.update_market_regime("SIDEWAYS")  # Adjusts consecutive loss limit
```

### 2. Enhanced Floor Bounce Check

```python
from src.utils.vietnam_market import check_floor_bounce_enhanced

is_valid, message, strength, wait_time = check_floor_bounce_enhanced(
    current_price=45000,
    day_low=43500,
    reference_price=46500,
    symbol="VNM",
    volume_ratio=2.0,
    minutes_at_floor=35,
    market_volatility=0.025,
)
```

### 3. VN30 Symbols Fetcher

```python
from src.utils.vn30_fetcher import get_vn30_symbols, is_vn30_symbol

symbols = get_vn30_symbols()  # Fetches from API with cache
is_blue_chip = is_vn30_symbol("VNM")  # True
```

### 4. Database-Backed Circuit Breaker

```python
from src.risk.circuit_breaker_db import CircuitBreakerDB

db = CircuitBreakerDB(db_path="trading_bot.db")
stats = db.load_stats()
db.save_stats(stats)

# Distributed locking
if db.acquire_lock("trading", bot_id="bot1"):
    # Safe to trade
    db.release_lock("trading", bot_id="bot1")
```

## Testing

```bash
# Test VN30 fetcher
python src/utils/vn30_fetcher.py

# Test circuit breaker
python src/risk/circuit_breaker.py
```

## Migration Notes

1. **DCA disabled by default** - Nếu muốn enable, set `DCA_ENABLED = True` trong `position_sizing.py`
2. **Confidence thresholds tăng** - Review signals sau khi update
3. **Circuit breaker DB** - Tự động fallback to JSON nếu DB fail
