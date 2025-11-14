# LUỒNG HOẠT ĐỘNG VÀ CHỨC NĂNG CÁC FILE PYTHON

## 📋 MỤC LỤC

1. [Core System - Hệ thống chính](#1-core-system)
2. [Data & Configuration - Dữ liệu & Cấu hình](#2-data--configuration)
3. [Machine Learning - Học máy](#3-machine-learning)
4. [Trading Logic - Logic giao dịch](#4-trading-logic)
5. [Portfolio Management - Quản lý danh mục](#5-portfolio-management)
6. [Risk Management - Quản lý rủi ro](#6-risk-management)
7. [Market Analysis - Phân tích thị trường](#7-market-analysis)
8. [Notifications & Monitoring - Thông báo & Giám sát](#8-notifications--monitoring)
9. [Utilities - Tiện ích](#9-utilities)
10. [Testing & Migration - Kiểm thử & Di chuyển](#10-testing--migration)

---

## 1. CORE SYSTEM - Hệ thống chính

### 📄 `main.py`
**Chức năng:** File khởi động chính của ứng dụng
- Khởi tạo FastAPI web server
- Quản lý lifecycle của bot (startup/shutdown)
- Chạy scheduler tự động theo lịch giao dịch VN
- Cung cấp REST API endpoints

**Luồng hoạt động:**
```
1. Load environment variables & config
2. Khởi động FastAPI app
3. Start background services:
   - Scheduler thread (chạy bot theo lịch)
   - Telegram bot listener thread
4. Expose API endpoints:
   - GET /health - Health check
   - POST /run-bot - Chạy bot thủ công
   - GET /portfolio - Xem portfolio
```

**Lịch chạy tự động:**
- Thứ 7 20:00: Phân tích ngành
- Thứ 2-6 9:15: Quét tín hiệu (sau khi mở cửa 15 phút)
- Thứ 6 14:45: Kiểm tra portfolio
- 15:10: Ghi lại PnL hàng ngày
- 15:15: Gửi daily summary


### 📄 `bot_runner_improved.py`
**Chức năng:** Engine chính để quét và phân tích tín hiệu giao dịch
- Quét tất cả mã cổ phiếu từ List.csv
- Phân tích ML signals
- Áp dụng entry/exit logic
- Gửi khuyến nghị qua Telegram

**Luồng hoạt động:**
```
1. Check market regime (có nên trade không?)
2. Load danh sách tickers từ List.csv
3. Check exit cho các vị thế đang nắm giữ
4. Scan tín hiệu mua mới:
   - Load data cho từng mã
   - Phân tích ML signal
   - Áp dụng improved entry logic
   - Tính position sizing
   - Gửi khuyến nghị qua Telegram
5. Kiểm tra portfolio và đề xuất
6. Gửi summary
```

**Tích hợp:**
- `market_regime_proxy.py` - Kiểm tra tình trạng thị trường
- `ml_signals.py` - Tín hiệu ML
- `improved_entry_logic.py` - Logic vào lệnh
- `improved_exit_logic.py` - Logic thoát lệnh
- `improved_position_sizing.py` - Tính kích thước vị thế
- `portfolio_manager.py` - Quản lý portfolio

---

## 2. DATA & CONFIGURATION - Dữ liệu & Cấu hình

### 📄 `config.py` (DEPRECATED)
**Chức năng:** File cấu hình cũ (giữ lại để backward compatibility)
- Load environment variables
- Định nghĩa constants
- **⚠️ Nên dùng `trading_config.py` thay thế**

### 📄 `trading_config.py`
**Chức năng:** Cấu hình tập trung mới
- Quản lý tất cả config trong một nơi
- Hỗ trợ validation
- Dễ dàng override bằng environment variables

**Các config class:**
- `DataConfig` - Cấu hình data source
- `TradingConfig` - Cấu hình chiến lược trading
- `APIConfig` - Cấu hình API
- `TelegramConfig` - Cấu hình Telegram bot
- `ServerConfig` - Cấu hình server

**Sử dụng:**
```python
from trading_config import get_config
config = get_config()
print(config.trading.min_confidence)
```

### 📄 `data_loader.py`
**Chức năng:** Load dữ liệu giá từ TCBS API
- Fetch OHLCV data từ TCBS
- Cache dữ liệu để tránh gọi API nhiều lần
- Rate limiting để tránh bị block

**Luồng hoạt động:**
```
1. Check cache (nếu có và còn fresh)
2. Nếu không có cache:
   - Gọi TCBS API với rate limiting
   - Parse response
   - Validate data
   - Save cache
3. Return DataFrame với columns: time, open, high, low, close, volume
```

**API sử dụng:**
- TCBS Stock Bars API: `https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term`

### 📄 `ticker_loader.py`
**Chức năng:** Load danh sách tickers từ List.csv
- Đọc file List.csv
- Clean và validate tickers
- Cung cấp singleton instance

**Sử dụng:**
```python
from ticker_loader import get_ticker_loader
loader = get_ticker_loader()
tickers = loader.all_tickers  # List tất cả mã
```

### 📄 `database.py`
**Chức năng:** SQLite database thay thế JSON files
- Lưu trữ positions, trades, portfolio history
- Cung cấp API để CRUD operations
- Migration từ JSON sang SQLite

**Tables:**
- `positions` - Vị thế đang nắm giữ
- `portfolio_history` - Lịch sử portfolio
- `trades` - Lịch sử giao dịch
- `paper_trades` - Paper trading
- `signals_cache` - Cache tín hiệu

**Sử dụng:**
```python
from database import get_db
db = get_db()
positions = db.get_positions()
```

### 📄 `incremental_cache.py`
**Chức năng:** Cache system với incremental updates
- Lưu cache theo symbol và date
- Chỉ fetch dữ liệu mới (từ last_update đến hiện tại)
- Merge với cache cũ
- Tự động xóa cache cũ

---

## 3. MACHINE LEARNING - Học máy

### 📄 `ml_models.py`
**Chức năng:** ML models (Random Forest)
- Load/save trained models
- Predict xác suất giá tăng
- Tạo dummy models nếu chưa có

**Models:**
- Random Forest Classifier
- StandardScaler cho feature scaling

**Sử dụng:**
```python
from ml_models import MLPredictor
predictor = MLPredictor()
predictor.load_models()
predictions = predictor.predict(X)
```

### 📄 `ml_signals.py`
**Chức năng:** Tạo tín hiệu giao dịch từ ML + Technical Analysis
- Kết hợp ML predictions với technical indicators
- Ensemble decision making
- Fallback sang technical analysis nếu ML fail

**Luồng hoạt động:**
```
1. Add ML features vào DataFrame
2. Kiểm tra đủ features không
3. Nếu đủ:
   - ML prediction
   - Technical analysis
   - Ensemble decision (kết hợp cả 2)
4. Nếu không đủ:
   - Fallback sang pure technical analysis
5. Return signal: BUY/SELL/HOLD với confidence
```

### 📄 `features.py`
**Chức năng:** Tạo features cho ML models
- Tính toán 18 technical indicators
- Chuẩn bị data cho training/prediction

**Features (18 total):**
1. Moving Averages: sma20, ema20, ema50
2. RSI: rsi, rsi_signal
3. ATR: atr
4. MACD: macd, macd_signal, macd_diff, macd_signal_line
5. Bollinger Bands: bb_width, bb_position
6. Momentum: momentum_5, momentum_10, momentum_20
7. Volume: volume_ratio, volume_surge
8. Volatility: volatility_20

### 📄 `train_models.py`
**Chức năng:** Script để train ML models
- Load data từ nhiều cổ phiếu
- Combine data
- Train Random Forest
- Save models

**Chạy:**
```bash
python train_models.py
```

### 📄 `retrain_models.py`
**Chức năng:** Script để train lại models với đúng 18 features
- Kiểm tra features
- Clean data
- Train và save

---

## 4. TRADING LOGIC - Logic giao dịch

### 📄 `improved_entry_logic.py`
**Chức năng:** Logic vào lệnh nâng cao với multiple filters
- Trend filter
- Support/Resistance check
- Volume confirmation
- Risk/Reward check
- Market regime check
- Volatility filter

**Filters:**
1. Market Regime - Thị trường phải OK
2. Trend Alignment - Phải theo xu hướng
3. Support/Resistance - Vào gần support
4. Volume Confirmation - Volume tăng khi breakout
5. Volatility Check - Không vào khi vol quá cao
6. RSI Check - Không overbought
7. Price Action - Pattern bullish

**Output:**
```python
EntrySignal(
    should_enter=True/False,
    confidence=0-100,
    strength=SignalStrength,
    entry_price=...,
    stop_loss=...,
    take_profit_targets=[TP1, TP2, TP3],
    reasons=[...],
    warnings=[...]
)
```

### 📄 `improved_exit_logic.py`
**Chức năng:** Chiến lược thoát lệnh chuyên nghiệp
- Trailing stop
- Take profit bậc thang
- Time-based exit
- Market protection
- Pattern recognition

**Exit reasons:**
1. Stop Loss Hit
2. Trailing Stop
3. Take Profit (1, 2, 3)
4. ML Signal SELL
5. Time Decay (sideway quá lâu)
6. Market Crash Protection
7. Bearish Reversal Pattern
8. Support Breakdown

**Take Profit Strategy:**
- TP1 (10%): Chốt 30% position
- TP2 (15%): Chốt 50% còn lại
- TP3 (25%): Chốt 100% còn lại

### 📄 `improved_position_sizing.py`
**Chức năng:** Position sizing theo nguyên tắc BẢO TOÀN VỐN
- Không bao giờ risk >2% vốn cho 1 lệnh
- Không bao giờ đầu tư >10% vốn vào 1 mã
- Tổng exposure không quá 60% vốn
- Scale position theo confidence và market regime

**Nguyên tắc:**
```
1. Risk per trade: 2% max
2. Position size: 10% max
3. Total exposure: 60% max
4. Min positions: 8-10 mã
5. Diversification required
```

**Adjustments:**
- Volatility adjustment
- Confidence-based sizing
- Market regime adjustment
- Correlation penalty


---

## 5. PORTFOLIO MANAGEMENT - Quản lý danh mục

### 📄 `portfolio_manager.py`
**Chức năng:** Quản lý portfolio với SQLite database
- Lưu/load positions
- Track performance metrics
- Portfolio history
- Detailed analysis

**Chức năng chính:**
```python
# Thêm position
manager.add_position(symbol, shares, entry_price, stop_loss, take_profit)

# Đóng position
manager.close_position(symbol, exit_price, reason)

# Lấy portfolio value
portfolio = manager.get_portfolio_value()

# Phân tích chi tiết
analysis = manager.get_detailed_analysis()
```

**Tích hợp:**
- `database.py` - Lưu trữ data
- `monitoring.py` - Track performance
- `trading_config.py` - Config

### 📄 `portfolio_analyzer.py`
**Chức năng:** Phân tích portfolio hiện tại và đề xuất
- Phân tích từng cổ phiếu đang nắm giữ
- Đề xuất BÁN/GIỮ cho mã đang có
- Tìm cơ hội MUA MỚI cho mã chưa có
- Tính sector exposure
- Regime adjustment
- Optimal allocation (HRP/Risk Budgeting)

**Luồng hoạt động:**
```
1. Phân tích market regime
2. Phân tích từng mã đang nắm giữ:
   - ML signal
   - Exit decision
   - Recommendation (SELL/HOLD)
3. Tìm cơ hội mua mới (CHỈ từ config):
   - Scan tất cả mã trong TICKERS
   - Loại trừ mã đã nắm
   - Entry signal analysis
4. Tổng kết portfolio:
   - Sector exposure
   - Regime adjustment
   - Optimal allocation
```

### 📄 `portfolio_optimizer.py`
**Chức năng:** Tối ưu hóa phân bổ portfolio
- Hierarchical Risk Parity (HRP)
- Risk Budgeting
- Correlation analysis

**Methods:**
- `optimize_weights()` - Tính trọng số tối ưu
- HRP allocation - Phân bổ theo cấu trúc phân cấp
- Risk Budgeting - Phân bổ theo rủi ro

### 📄 `portfolio_regime_adjuster.py`
**Chức năng:** Điều chỉnh portfolio theo market regime
- Tính target cash ratio theo regime
- Đề xuất bán để tăng tiền mặt
- Ưu tiên bán mã có recommendation SELL

**Target cash ratio:**
- BULL: 15%
- SIDEWAYS: 35%
- BEAR: 70%
- HIGH_VOLATILITY: 60%

### 📄 `portfolio_history.py`
**Chức năng:** Theo dõi lịch sử portfolio
- Lưu daily snapshots
- Equity curve
- Performance metrics
- Export to CSV

**Metrics:**
- Total return
- Daily avg return
- Volatility
- Sharpe ratio
- Max drawdown
- Win rate

### 📄 `paper_trading.py`
**Chức năng:** Mô phỏng giao dịch (không dùng tiền thật)
- Tài khoản paper trading
- Thực thi lệnh với slippage và fees
- Tracking PnL
- So sánh với real portfolio

**Features:**
- Initial capital: 100M VNĐ
- Commission: 0.15% mỗi chiều
- Slippage: 0.1%
- Trade history
- Performance metrics

---

## 6. RISK MANAGEMENT - Quản lý rủi ro

### 📄 `risk_management.py`
**Chức năng:** Position sizing & price targets
- Tính số lượng cổ phiếu nên mua/bán
- Stop loss dựa trên ATR
- Take profit targets
- Risk/Reward ratio

**Nguyên tắc:**
```
- Max position: 20% vốn
- Risk per trade: 2% vốn
- Max portfolio risk: 10%
- Stop loss: 2 ATR
- Take profit: 1.5R và 3R
```

### 📄 `enhanced_risk_management.py`
**Chức năng:** Risk management nâng cao
- Volatility adjustment
- Correlation penalty
- Market regime adjustment

**Adjustments:**
- Market volatility factor
- Confidence factor
- Market regime factor

### 📄 `risk_metrics.py`
**Chức năng:** Tính toán các metrics rủi ro
- Sector exposure
- Correlation matrix
- Distance correlation
- Copula correlation
- Diversification score

**Functions:**
```python
# Sector exposure
exposure = calculate_sector_exposure(holdings)

# Correlation risk
risk = calculate_portfolio_correlation_risk(symbols)

# Diversification recommendation
rec = get_diversification_recommendation(holdings)
```

---

## 7. MARKET ANALYSIS - Phân tích thị trường

### 📄 `market_regime.py`
**Chức năng:** Phát hiện tình trạng thị trường
- Phân tích VNINDEX
- Xác định regime: BULL/BEAR/SIDEWAYS/HIGH_VOLATILITY
- Quyết định có nên trade không
- HMM-based regime detection (optional)

**Indicators:**
- Weekly change
- Trend (SMA20 vs SMA50)
- Volatility (ATR)
- Market breadth

**Decision:**
```
KHÔNG TRADE khi:
- Bear market
- High volatility
- Weekly change < -5%

CÓ THỂ TRADE khi:
- Bull market
- Sideways nhẹ
```

### 📄 `market_regime_proxy.py`
**Chức năng:** Phân tích market regime qua blue-chip stocks
- Phân tích tất cả mã từ config
- Tính buy rate
- Xác định regime dựa trên % mã BUY

**Logic:**
```
Buy rate >= 60% → BULL
Buy rate >= 40% → SIDEWAYS
Buy rate < 40% → BEAR
```

### 📄 `multi_timeframe.py`
**Chức năng:** Phân tích đa khung thời gian
- Daily analysis
- Weekly analysis (giả lập từ daily)
- Combined signal

**Weight:**
- Daily: 60%
- Weekly: 40%

### 📄 `improved_sector_analysis.py` (DEPRECATED)
**Chức năng:** Phân tích ngành (không còn dùng)
- ⚠️ Không còn phân tích theo ngành
- Tất cả mã được scan trực tiếp từ List.csv

### 📄 `news_analyzer.py`
**Chức năng:** Phân tích tin tức và sentiment
- Fetch tin tức từ nhiều nguồn (RSS)
- Sentiment analysis
- Topic classification
- Hot news detection

**Sources:**
- VnExpress Chứng khoán
- CafeF
- Vietstock
- Bloomberg (filtered by Vietnam keywords)
- Reuters (filtered by Vietnam keywords)

**Features:**
- Sentiment score (-1 to +1)
- Topic classification (dividend, litigation, macro, earnings)
- Hot news tracking
- Source frequency monitoring

---

## 8. NOTIFICATIONS & MONITORING - Thông báo & Giám sát

### 📄 `telegram_notifications.py`
**Chức năng:** Gửi notifications cho subscribers
- Gửi notification cho users đăng ký
- Daily summary
- Chart buttons

**Functions:**
```python
# Gửi cho subscribers của symbol
await send_notification_to_subscribers(symbol, message)

# Gửi daily summary cho tất cả
await send_daily_summary_to_all()
```

### 📄 `telegram_subscriptions.py`
**Chức năng:** Quản lý đăng ký nhận tin
- Subscribe/unsubscribe symbol
- Subscribe/unsubscribe sector
- Lưu subscriptions vào JSON

**Data structure:**
```json
{
  "users": {
    "user_id": {
      "symbols": ["VNM", "VCB"],
      "sectors": ["BANKING"]
    }
  },
  "symbol_subscribers": {
    "VNM": ["user_id1", "user_id2"]
  }
}
```

### 📄 `tg_listener.py`
**Chức năng:** Telegram bot listener
- Xử lý commands
- Inline buttons
- Callback queries

**Commands:**
- `/start` - Khởi động bot
- `/run` - Quét tín hiệu
- `/portfolio` - Xem portfolio
- `/addstock` - Thêm cổ phiếu
- `/sellstock` - Bán cổ phiếu
- `/news` - Tin tức
- `/subscribe` - Đăng ký nhận tin
- `/summary` - Daily summary
- `/paper` - Paper trading account

### 📄 `monitoring.py`
**Chức năng:** Performance monitoring & metrics
- Track trading performance
- System health monitoring
- API call tracking

**Metrics:**
- Win rate
- Average profit/loss
- Sharpe ratio
- Maximum drawdown
- Total return

### 📄 `api_monitor.py`
**Chức năng:** API monitoring với ping check
- Ping check endpoints
- Retry policy
- Failure tracking
- Alerts khi fail nhiều lần

**Features:**
- Health status tracking
- Consecutive failure detection
- Response time monitoring
- Alert callback

### 📄 `model_monitor.py`
**Chức năng:** Model versioning và drift monitoring
- Record training runs
- Check model drift
- Version tracking

---

## 9. UTILITIES - Tiện ích

### 📄 `rate_limiter.py`
**Chức năng:** Rate limiting cho API calls
- Token bucket algorithm
- Prevent hitting API rate limits

**Usage:**
```python
from rate_limiter import tcbs_limiter

@tcbs_limiter.limit
def api_call():
    return requests.get(url)
```

### 📄 `logging_config.py`
**Chức năng:** Cấu hình logging toàn hệ thống
- File handler
- Console handler
- UTF-8 encoding
- Log level configuration

### 📄 `suppress_warnings.py`
**Chức năng:** Suppress các warnings không cần thiết
- TensorFlow warnings
- FutureWarning từ transformers
- DeprecationWarning

**Import ở đầu file:**
```python
import suppress_warnings  # noqa: F401
```

### 📄 `vn_trading_schedule.py`
**Chức năng:** Lịch giao dịch Việt Nam
- Kiểm tra giờ giao dịch
- Kiểm tra ngày giao dịch (không phải T3/T7)
- Lấy giờ giao dịch tiếp theo

**Trading hours:**
- Morning: 9:00 - 11:30
- Afternoon: 13:00 - 15:00
- Non-trading days: Thứ 3 và Thứ 7

---

## 10. TESTING & MIGRATION - Kiểm thử & Di chuyển

### 📄 `test_improvements.py`
**Chức năng:** Test script cho improvements
- Test configuration
- Test database
- Test portfolio manager
- Test monitoring
- Test backward compatibility

**Chạy:**
```bash
python test_improvements.py
```

### 📄 `migrate_json_to_db.py`
**Chức năng:** Migration từ JSON sang SQLite
- Backup JSON files
- Migrate positions
- Migrate portfolio history
- Verify migration

**Chạy:**
```bash
python migrate_json_to_db.py
```

### 📄 `migrate_to_sqlite.py`
**Chức năng:** Migration script đơn giản hơn
- Migrate JSON to SQLite
- Show current data

### 📄 `validate_list_csv.py`
**Chức năng:** Validate tickers trong List.csv
- Kiểm tra từng ticker với TCBS API
- Tìm tickers không hợp lệ
- Save results

**Chạy:**
```bash
python validate_list_csv.py [sample_size] [skip_first]
```

### 📄 `backtest.py`
**Chức năng:** Script chạy backtest
- Menu-driven interface
- Backtest 1 cổ phiếu
- Backtest nhiều cổ phiếu
- So sánh các threshold

**Chạy:**
```bash
python backtest.py
```

### 📄 `run_backtest.py`
**Chức năng:** Backtesting engine
- Mô phỏng trading với ML + Risk Management
- Tính metrics
- Vẽ biểu đồ
- Export results

**Features:**
- Initial capital: 100M VNĐ
- Commission: 0.15%
- Slippage: 0.1%
- ML-based signals
- Conservative position sizing
- Stop loss & take profit

**Metrics:**
- Total return
- Buy & hold comparison
- Win rate
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Profit factor
- Max drawdown


---

## 📊 LUỒNG HOẠT ĐỘNG TỔNG QUAN

### 🔄 Luồng chính khi bot chạy

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN.PY - Khởi động                      │
│  - Load config                                              │
│  - Start FastAPI server                                     │
│  - Start scheduler thread                                   │
│  - Start Telegram bot thread                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              SCHEDULER - Chạy theo lịch VN                  │
│  - Thứ 7 20:00: Phân tích ngành                            │
│  - Thứ 2-6 9:15: Quét tín hiệu                             │
│  - Thứ 6 14:45: Kiểm tra portfolio                         │
│  - 15:10: Ghi PnL                                           │
│  - 15:15: Gửi daily summary                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         BOT_RUNNER_IMPROVED.PY - Quét tín hiệu              │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Check Market │  │ Check Exits  │  │ Scan New     │
│ Regime       │  │ (Positions)  │  │ Entries      │
└──────────────┘  └──────────────┘  └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────────────────────────────────────────┐
│         MARKET_REGIME_PROXY.PY                   │
│  - Phân tích tất cả mã từ config                │
│  - Tính buy rate                                 │
│  - Xác định BULL/BEAR/SIDEWAYS                  │
└──────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────┐
│         IMPROVED_EXIT_LOGIC.PY                   │
│  - Check stop loss                               │
│  - Check trailing stop                           │
│  - Check take profit                             │
│  - Check ML signal SELL                          │
│  - Check reversal patterns                       │
└──────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────┐
│         Scan từng mã trong List.csv              │
└──────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ DATA_LOADER  │  │ ML_SIGNALS   │  │ ENTRY_LOGIC  │
│ Load OHLCV   │  │ ML + Tech    │  │ Filters      │
└──────────────┘  └──────────────┘  └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────┐
│         IMPROVED_POSITION_SIZING.PY              │
│  - Tính shares dựa trên risk                    │
│  - Adjust theo confidence                        │
│  - Adjust theo market regime                     │
│  - Adjust theo volatility                        │
└──────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────┐
│         Gửi khuyến nghị qua Telegram             │
│  - Entry price                                   │
│  - Stop loss                                     │
│  - Take profit targets                           │
│  - Position size                                 │
│  - Reasons & warnings                            │
└──────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────┐
│         PORTFOLIO_MANAGER.PY                     │
│  - Save position to database                     │
│  - Track performance                             │
│  - Update portfolio history                      │
└──────────────────────────────────────────────────┘
```

### 🎯 Luồng phân tích một mã cổ phiếu

```
┌─────────────────────────────────────────────────┐
│  1. Load Data (data_loader.py)                 │
│     - Check cache                               │
│     - Fetch from TCBS API if needed             │
│     - Return OHLCV DataFrame                    │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  2. Add Features (features.py)                  │
│     - Calculate 18 technical indicators         │
│     - SMA, EMA, RSI, MACD, BB, etc.            │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  3. ML Analysis (ml_signals.py)                 │
│     - ML prediction (Random Forest)             │
│     - Technical analysis                        │
│     - Ensemble decision                         │
│     - Output: BUY/SELL/HOLD + confidence        │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  4. Entry Logic (improved_entry_logic.py)       │
│     - Apply 7 filters:                          │
│       1. Market regime                          │
│       2. Trend alignment                        │
│       3. Support/Resistance                     │
│       4. Volume confirmation                    │
│       5. Volatility check                       │
│       6. RSI check                              │
│       7. Price action                           │
│     - Calculate adjusted confidence             │
│     - Calculate entry price, SL, TP             │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  5. Position Sizing (improved_position_sizing)  │
│     - Calculate shares based on risk            │
│     - Adjust by confidence                      │
│     - Adjust by market regime                   │
│     - Adjust by volatility                      │
│     - Ensure diversification                    │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  6. News Analysis (news_analyzer.py) [Optional] │
│     - Fetch news from multiple sources          │
│     - Sentiment analysis                        │
│     - Adjust confidence based on sentiment      │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  7. Final Decision                              │
│     - Should enter: Yes/No                      │
│     - Entry signal with all details             │
│     - Send to Telegram                          │
└─────────────────────────────────────────────────┘
```

### 💼 Luồng quản lý portfolio

```
┌─────────────────────────────────────────────────┐
│  User adds stock via Telegram                   │
│  /addstock VNM 500 80000                        │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  PORTFOLIO_MANAGER.PY                           │
│  - Validate input                               │
│  - Calculate entry value                        │
│  - Save to database                             │
│  - Track in monitoring                          │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  DATABASE.PY                                    │
│  - Insert into positions table                  │
│  - Log trade in trades table                    │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  Daily: Check portfolio                         │
│  - Load all positions                           │
│  - Update current prices                        │
│  - Calculate P&L                                │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  PORTFOLIO_ANALYZER.PY                          │
│  - Analyze each holding                         │
│  - Check exit signals                           │
│  - Recommend SELL/HOLD                          │
│  - Find new buy opportunities                   │
│  - Calculate sector exposure                    │
│  - Regime adjustment                            │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  Send analysis to Telegram                      │
│  - Current holdings                             │
│  - Sell recommendations                         │
│  - Hold recommendations                         │
│  - New buy opportunities                        │
│  - Portfolio summary                            │
└─────────────────────────────────────────────────┘
```

---

## 🔧 CÁC ĐIỂM QUAN TRỌNG CẦN LƯU Ý

### ✅ Đã cải tiến
1. **Không còn phân tích theo ngành** - Tất cả mã được scan trực tiếp từ List.csv
2. **Database thay vì JSON** - Sử dụng SQLite để lưu trữ data
3. **Improved entry/exit logic** - Logic vào/thoát lệnh chuyên nghiệp hơn
4. **Conservative position sizing** - Quản lý vốn an toàn
5. **Market regime detection** - Chỉ trade khi thị trường phù hợp
6. **News integration** - Tích hợp tin tức và sentiment
7. **Paper trading** - Mô phỏng giao dịch để test
8. **Portfolio optimization** - HRP và Risk Budgeting

### ⚠️ Deprecated files
- `config.py` - Dùng `trading_config.py` thay thế
- `improved_sector_analysis.py` - Không còn dùng phân tích ngành
- JSON files - Đã migrate sang SQLite

### 🔄 Migration path
```
1. Chạy test_improvements.py để kiểm tra
2. Chạy migrate_json_to_db.py để migrate data
3. Backup JSON files vào json_backup/
4. Xóa JSON files sau khi verify
```

### 📝 Cấu hình quan trọng

**Environment variables (.env):**
```
TELEGRAM_TOKEN=your_token
CHAT_ID=your_chat_id
LOOKBACK=200
MIN_CONFIDENCE=60
MAX_POSITION_SIZE=0.10
```

**List.csv:**
- Chứa tất cả mã cổ phiếu cần scan
- Format: Mỗi dòng một mã
- Ví dụ: VNM, VCB, FPT, HPG, ...

**Models:**
- Random Forest: `models/random_forest.pkl`
- Scaler: `models/scaler.pkl`
- Model info: `models/model_info.json`

### 🚀 Khởi động hệ thống

**Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Train models (nếu chưa có)
python retrain_models.py

# Test
python test_improvements.py

# Run bot
python main.py
```

**Production:**
```bash
# Set environment variables
export PORT=8080
export TELEGRAM_TOKEN=...
export CHAT_ID=...

# Run with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8080
```

---

## 📚 TÀI LIỆU THAM KHẢO

### Các file markdown khác:
- `README.md` - Hướng dẫn tổng quan
- `SETUP.md` - Hướng dẫn cài đặt
- `SETUP_COMPLETE.md` - Checklist setup
- `MIGRATION_GUIDE.md` - Hướng dẫn migration
- `SYSTEM_ARCHITECTURE.md` - Kiến trúc hệ thống
- `SYSTEM_REVIEW_CHANGES.md` - Các thay đổi gần đây
- `MEDIUM_PRIORITY_IMPROVEMENTS.md` - Cải tiến ưu tiên trung bình
- `WARNINGS_SUPPRESSION.md` - Hướng dẫn suppress warnings

### API Documentation:
- FastAPI Docs: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

### Telegram Commands:
```
/start - Khởi động bot
/run - Quét tín hiệu
/portfolio - Xem portfolio
/addstock SYMBOL SHARES PRICE - Thêm cổ phiếu
/sellstock SYMBOL [SHARES] - Bán cổ phiếu
/news SYMBOL - Tin tức
/subscribe SYMBOL - Đăng ký nhận tin
/summary - Daily summary
/paper - Paper trading account
```

---

## 🎓 KẾT LUẬN

Hệ thống trading bot này là một giải pháp hoàn chỉnh cho việc:
- ✅ Tự động quét và phân tích tín hiệu giao dịch
- ✅ Quản lý portfolio chuyên nghiệp
- ✅ Quản lý rủi ro an toàn
- ✅ Tích hợp ML và technical analysis
- ✅ Thông báo qua Telegram
- ✅ Backtest và paper trading
- ✅ Monitoring và performance tracking

**Điểm mạnh:**
- Logic vào/thoát lệnh chuyên nghiệp
- Position sizing bảo toàn vốn
- Market regime detection
- Đa dạng hóa portfolio
- Tích hợp tin tức
- Database thay vì JSON

**Lưu ý:**
- Đây là công cụ hỗ trợ, không phải lời khuyên đầu tư
- Luôn kiểm tra kỹ trước khi thực hiện giao dịch
- Sử dụng paper trading để test trước
- Quản lý rủi ro nghiêm ngặt

---

**📅 Cập nhật:** 2025-11-14
**👨‍💻 Tác giả:** Trading Bot Team
**📧 Liên hệ:** Qua Telegram bot

