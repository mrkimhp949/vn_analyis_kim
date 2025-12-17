# 🇻🇳 Vietnam Stock Trading Bot

Hệ thống giao dịch tự động cho thị trường chứng khoán Việt Nam với Machine Learning và phân tích kỹ thuật.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## ✨ Tính năng chính

### 📊 Phân tích thị trường
- **Real-time Data** - Tích hợp vnstock, SSI, VNDirect API
- **Foreign Flow Analysis** - Theo dõi dòng tiền khối ngoại
- **Market Regime Detection** - Nhận diện trạng thái thị trường (Bull/Bear/Sideway)
- **Sector Rotation** - Phân tích luân chuyển ngành

### 🤖 Machine Learning
- **ML Signal Generator V3** - Ensemble models với accuracy 65-70%
  - Random Forest, Gradient Boosting, XGBoost
  - Microstructure features
  - Confidence calibration
  - Walk-forward validation
- **Sentiment Analysis** - Phân tích tin tức với PhoBERT

### 📈 Chiến lược giao dịch
- **Entry Timing Filter** - Tối ưu thời điểm vào lệnh
- **Settlement Timing** - Quản lý T+2 settlement
- **Session Trading** - ATO/ATC session analysis
- **Risk Management** - Portfolio VaR, stress testing

### 🏛️ Vietnam Market Rules
- ✅ Lot size (100 cổ phiếu)
- ✅ Tick size (10/50/100 VND)
- ✅ Price limits (±7%/10%/15%)
- ✅ T+2 settlement
- ✅ ATO/ATC sessions

### 📱 Thông báo & Monitoring
- Telegram Bot notifications
- Prometheus metrics
- Grafana dashboard
- WebSocket streaming

## 🚀 Cài đặt

### Yêu cầu
- Python 3.11+
- pip hoặc conda

### Cài đặt nhanh

```bash
# Clone repository
git clone https://github.com/your-repo/vn_analysis_kim.git
cd vn_analysis_kim

# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt

# Copy file cấu hình
cp .env.example .env
# Chỉnh sửa .env với API keys của bạn
```

### Cài đặt với Docker

```bash
docker-compose up -d
```

## 📁 Cấu trúc project

```
vn_analysis_kim/
├── src/                    # Source code chính
│   ├── api/                # FastAPI endpoints
│   ├── broker/             # Broker integration (SSI, VNDirect)
│   ├── data/               # Data loaders & providers
│   ├── market/             # Market analysis (foreign flow, regime)
│   ├── ml/                 # Machine learning models
│   ├── monitoring/         # Prometheus metrics
│   ├── portfolio/          # Portfolio management
│   ├── risk/               # Risk management (VaR, position sizing)
│   ├── strategies/         # Trading strategies
│   └── utils/              # Utilities & helpers
├── backtesting/            # Backtesting engine
├── scripts/                # Utility scripts
├── tests/                  # Unit & integration tests
├── data/                   # Data files
├── models/                 # Trained ML models
└── docs/                   # Documentation
```

## 💻 Sử dụng

### Chạy Trading Bot

```bash
# Chạy API server
python -m uvicorn src.api.main:app --reload --port 8000

# Chạy paper trading
python scripts/run_backtest.py --mode paper

# Chạy backtest
python scripts/comprehensive_backtest.py
```

### Sử dụng ML Signal Generator

```python
from src.ml.signals.enhanced_v2 import EnhancedMLSignalGeneratorV2
from src.data.loader import load_data

# Load data
df = load_data("FPT", lookback=100)

# Generate signal
generator = EnhancedMLSignalGeneratorV2()
result = generator.analyze(df, symbol="FPT")

print(f"Signal: {result['signal']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Phân tích Foreign Flow

```python
from src.market.foreign_flow import get_foreign_flow_analyzer

analyzer = get_foreign_flow_analyzer()
flow = analyzer.analyze()

print(f"Score: {flow['score']}")  # -1 to +1
print(f"Trend: {flow['trend']}")  # BUYING, SELLING, NEUTRAL
```

### Data Loader (vnstock)

```python
from src.data.loader import load_data

# Load stock data
df = load_data(
    symbol="VNM",
    start_date="2024-01-01",
    end_date="2024-12-17"
)

# Hoặc sử dụng lookback
df = load_data("FPT", lookback=60)  # 60 ngày gần nhất
```

## 🧪 Testing

```bash
# Chạy tất cả tests
pytest

# Chạy với coverage
pytest --cov=src --cov-report=html

# Chạy tests cụ thể
pytest tests/unit/test_ml_signals.py -v
```

## 📊 Backtesting

```bash
# Backtest đơn giản
python scripts/backtest.py --symbol FPT --start 2024-01-01

# Comprehensive backtest với ML
python scripts/comprehensive_backtest.py

# Walk-forward validation
python scripts/walk_forward_test.py
```

## ⚙️ Cấu hình

### Environment Variables (.env)

```env
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Broker API (optional)
SSI_API_KEY=your_ssi_key
VNDIRECT_API_KEY=your_vndirect_key

# Database
DATABASE_URL=sqlite:///trading.db

# ML Settings
ML_CONFIDENCE_THRESHOLD=0.55
USE_ML_V3=true
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| ML Signal Accuracy | 65-70% |
| Backtest Win Rate | 55-60% |
| Avg Profit/Trade | 2-3% |
| Max Drawdown | <15% |

## 🛠️ Scripts hữu ích

| Script | Mô tả |
|--------|-------|
| `train_models.py` | Train ML models |
| `validate_list_csv.py` | Validate danh sách cổ phiếu |
| `filter_quality_tickers.py` | Lọc cổ phiếu chất lượng |
| `view_metrics.py` | Xem metrics hiện tại |
| `dashboard_app.py` | Chạy dashboard |

## 🔧 Troubleshooting

### Lỗi TCBS API 404
TCBS API đã ngừng hoạt động. Hệ thống đã chuyển sang sử dụng `vnstock` với VCI source.

### Lỗi vnstock rate limit
```python
# Đợi giữa các request
import time
time.sleep(1)  # Đợi 1 giây
```

### Thiếu data cho symbol
Một số mã có thể đã bị hủy niêm yết hoặc không có trên VCI. Kiểm tra tại [https://vci.vn](https://vci.vn).

## 📚 Documentation

- [API Reference](docs/api.md)
- [ML Models Guide](docs/ml_models.md)
- [Backtesting Guide](docs/backtesting.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## 📄 License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## ⚠️ Disclaimer

Phần mềm này chỉ dành cho mục đích nghiên cứu và giáo dục. **KHÔNG** phải là lời khuyên đầu tư. Giao dịch chứng khoán có rủi ro cao. Bạn chịu hoàn toàn trách nhiệm với các quyết định đầu tư của mình.

---

Made with ❤️ for Vietnam Stock Market
