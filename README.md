# 🤖 Vietnam Stock Trading Bot

**Signal Generator & Portfolio Tracker cho thị trường chứng khoán Việt Nam**

Bot này phân tích thị trường, tạo tín hiệu giao dịch và gửi qua Telegram. Bạn quyết định có mua/bán hay không và execute thủ công qua broker app.

---

## ✨ TÍNH NĂNG CHÍNH

### 📊 Signal Generation
- **ML Ensemble Models**: RandomForest + XGBoost + LSTM
- **Technical Analysis**: RSI, MACD, ATR, Bollinger Bands, Momentum
- **News Sentiment**: PhoBERT/ViBERT cho phân tích sentiment tiếng Việt
- **Sector Analysis**: Phân tích ngành và chọn top sectors
- **Market Regime**: Phân tích thị trường (BULL/BEAR/SIDEWAYS)

### 💼 Portfolio Management
- **Track Holdings**: Theo dõi portfolio với PnL real-time
- **Risk Metrics**: Sector exposure, correlation matrix, diversification
- **Performance Tracking**: Equity curve, Sharpe ratio, Max drawdown
- **History**: Lưu lịch sử portfolio để phân tích

### 📝 Paper Trading
- **Simulate Trades**: Mô phỏng thực thi lệnh với slippage & commission
- **Track Performance**: Theo dõi PnL và so sánh với real portfolio
- **Trade History**: Lưu lịch sử giao dịch

### 📱 Telegram Integration
- **Real-time Signals**: Nhận tín hiệu ngay khi bot phát hiện
- **Portfolio Updates**: Cập nhật portfolio và risk metrics
- **Daily Summaries**: Tóm tắt cuối ngày
- **News Alerts**: Tin tức nóng và sentiment
- **Subscriptions**: Đăng ký nhận tin cho mã/ngành cụ thể

### 🔧 Infrastructure
- **Incremental Cache**: Cache dữ liệu để tối ưu performance
- **API Monitoring**: Giám sát API với retry và alerts
- **VN Trading Schedule**: Chỉ chạy trong giờ giao dịch VN (9:00-11:30, 13:00-15:00)
- **Error Handling**: Xử lý lỗi và logging đầy đủ

---

## 🚀 QUICK START

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/vn_analysis_bot.git
cd vn_analysis_bot
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Configuration

Tạo file `config.py` hoặc set environment variables:

```python
# config.py
TELEGRAM_TOKEN = "your_telegram_bot_token"
CHAT_ID = "your_telegram_chat_id"
```

Hoặc dùng `.env`:

```bash
TELEGRAM_TOKEN=your_telegram_bot_token
CHAT_ID=your_telegram_chat_id
```

### 4. Run Bot

```bash
python main.py
```

Bot sẽ:
- Khởi động FastAPI server (port 8000)
- Chạy Telegram bot
- Chạy scheduler theo giờ giao dịch VN

---

## 📱 TELEGRAM COMMANDS

### Signal & Analysis
- `/run` - Quét tín hiệu ngay
- `/status` - Trạng thái bot

### Portfolio
- `/portfolio` - Xem portfolio với risk analysis
- `/addstock SYMBOL SHARES PRICE` - Thêm cổ phiếu vào portfolio
- `/sellstock SYMBOL [SHARES]` - Bán cổ phiếu

### News & Information
- `/news SYMBOL` - Tin tức và sentiment cho mã
- `/summary` - Summary cuối ngày

### Subscriptions
- `/subscribe SYMBOL` - Đăng ký nhận tin cho mã
- `/unsubscribe SYMBOL` - Hủy đăng ký
- `/mysubs` - Xem đăng ký của tôi

### Paper Trading
- `/paper` - Xem paper trading account
- `/papertrades` - Lịch sử paper trading

---

## 🔄 WORKFLOW

### Hàng ngày:

1. **Bot quét thị trường** (9:15, 13:30, 14:30)
   - Phân tích ML + Technical + Sentiment
   - Chọn top sectors
   - Quét mã cổ phiếu

2. **Bot gửi signals qua Telegram**
   - Confidence, reasons, risk metrics
   - Entry price, stop-loss, take-profit
   - Position size suggestion

3. **Bạn review signals**
   - Xem confidence, reasons
   - Phân tích thêm nếu cần

4. **Bạn quyết định**
   - Mua/không mua dựa trên bot + phân tích của bạn
   - Execute thủ công qua broker app (TCBS, VPS, SSI, etc.)

5. **Bot track portfolio**
   - Nếu bạn add stocks vào bot
   - Tính PnL, risk metrics
   - Gửi updates

6. **Bot monitor exits**
   - Check stop-loss, take-profit
   - Gửi thông báo khi cần thoát

---

## 📊 FEATURES CHI TIẾT

### ML Models
- **RandomForest**: Feature importance, non-linear patterns
- **XGBoost**: Gradient boosting với regularization
- **LSTM**: Time-series patterns và sequences
- **Ensemble**: Weighted average của tất cả models

### Technical Indicators
- **Trend**: SMA, EMA, MACD
- **Momentum**: RSI, Momentum
- **Volatility**: ATR, Bollinger Bands
- **Volume**: Volume analysis, OBV
- **Patterns**: Candlestick patterns

### Risk Management
- **Position Sizing**: Conservative với risk per trade 2%
- **Stop Loss**: Dựa trên ATR và volatility
- **Take Profit**: Multiple levels (TP1, TP2, TP3)
- **Sector Exposure**: Giới hạn 40% per sector
- **Correlation**: Tránh mã tương quan cao (>0.7)

### News Analysis
- **Sources**: VnExpress, CafeF, VnEconomy, Bloomberg, Reuters
- **Sentiment**: PhoBERT/ViBERT cho tiếng Việt
- **Topics**: Dividend, litigation, macro, earnings
- **Hot News**: Phát hiện tin nóng dựa trên frequency

---

## 🛠️ DEPLOYMENT

### Local Development

```bash
python main.py
```

### Render.com Deployment

1. **Fork repository**
2. **Go to [Render.com](https://render.com)**
3. **Click "New +" → "Web Service"**
4. **Connect GitHub repository**
5. **Configure:**
   - **Name:** `vn-trading-bot`
   - **Environment:** `Python 3.11`
   - **Region:** `Singapore`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python main.py`
6. **Add Environment Variables:**
   - `TELEGRAM_TOKEN` = your_bot_token
   - `CHAT_ID` = your_chat_id
7. **Deploy!**

### Docker (Optional)

```bash
docker build -t vn-trading-bot .
docker run -e TELEGRAM_TOKEN=xxx -e CHAT_ID=xxx vn-trading-bot
```

---

## 📁 PROJECT STRUCTURE

```
vn_analysis_bot/
├── main.py                 # Entry point, FastAPI + Scheduler
├── bot_runner_improved.py  # Main bot logic
├── tg_listener.py          # Telegram bot handlers
├── data_loader.py          # Data fetching với cache
├── ml_pipeline/            # ML models và training
│   ├── model_trainer.py
│   ├── sentiment_model.py
│   └── data_manager.py
├── portfolio_manager.py    # Portfolio tracking
├── paper_trading.py        # Paper trading system
├── risk_metrics.py         # Risk analysis
├── news_analyzer.py        # News sentiment
├── incremental_cache.py     # Incremental data cache
├── api_monitor.py           # API monitoring
├── vn_trading_schedule.py  # VN trading hours
└── requirements.txt         # Dependencies
```

---

## ⚙️ CONFIGURATION

### Trading Hours (VN)
- **Morning**: 9:00 - 11:30
- **Afternoon**: 13:00 - 15:00
- **Non-trading days**: Thứ 3 và Thứ 7

### Scheduler
- **9:15** - Signal scan
- **9:30, 13:30, 14:30** - News refresh
- **14:45** (Thứ 6) - Portfolio check
- **15:10** - Record PnL
- **15:15** - Daily summary
- **20:00** (Thứ 7) - Sector analysis

---

## 📈 PERFORMANCE METRICS

Bot track các metrics sau:
- **Total Return**: Lợi nhuận tổng
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Mức giảm tối đa
- **Win Rate**: Tỷ lệ lệnh thắng
- **Sortino Ratio**: Downside risk
- **Calmar Ratio**: Return vs max drawdown
- **Profit Factor**: Lợi nhuận / Lỗ

---

## ⚠️ LƯU Ý QUAN TRỌNG

### Bot này:
- ✅ **Gửi signals** qua Telegram
- ✅ **Track portfolio** nếu bạn add stocks
- ✅ **Paper trading** để test strategy
- ❌ **KHÔNG tự động trade**

### Bạn:
- ✅ **Quyết định** có mua/bán hay không
- ✅ **Execute** orders thủ công qua broker app
- ✅ **Quản lý risk** của riêng bạn

### Rủi ro:
- ⚠️ Bot signals không phải lúc nào cũng đúng
- ⚠️ Trading có rủi ro, chỉ đầu tư số tiền bạn có thể chấp nhận mất
- ⚠️ Luôn có risk management riêng

---

## 🤝 CONTRIBUTING

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 LICENSE

This project is licensed under the MIT License.

---

## 📞 SUPPORT

Nếu có câu hỏi hoặc vấn đề:
- Mở issue trên GitHub
- Check documentation trong code
- Review logs để debug

---

## 🎯 ROADMAP

### Đã hoàn thành ✅
- ML ensemble models
- Technical analysis
- News sentiment
- Portfolio tracking
- Paper trading
- Risk metrics
- Telegram integration
- Incremental cache
- API monitoring
- VN trading schedule

### Tương lai 🔮
- Real-time price feeds (WebSocket)
- More ML models (Transformer, etc.)
- Advanced backtesting
- Multi-timeframe analysis
- Options trading signals

---

**Happy Trading! 📈**

*Lưu ý: Bot này chỉ là công cụ hỗ trợ. Bạn vẫn là người quyết định cuối cùng. Trading có rủi ro, hãy đầu tư có trách nhiệm.*
