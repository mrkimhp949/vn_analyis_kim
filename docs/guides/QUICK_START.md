# ⚡ QUICK START GUIDE

## 🚀 Cài đặt nhanh (5 phút)

### Windows:
```cmd
quick_install.bat
```

### Linux/macOS:
```bash
bash quick_install.sh
```

---

## 📦 Cài đặt thủ công

### 1. Tạo virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 2. Cài đặt dependencies

**Chọn 1 trong 3:**

```bash
# Minimal (Recommended cho lần đầu)
pip install -r requirements-minimal.txt

# Enhanced (Recommended cho production)
pip install -r requirements-enhanced.txt

# Advanced (Full features)
pip install -r requirements-advanced.txt
```

### 3. Cấu hình

```bash
# Copy .env
cp .env.example .env

# Edit với credentials của bạn
nano .env  # hoặc notepad .env
```

**Cần thiết lập:**
- `TELEGRAM_TOKEN` - Lấy từ @BotFather
- `CHAT_ID` - Chat ID của bạn

### 4. Khởi tạo database

```bash
python -c "from database import get_db; get_db()"
```

### 5. Chạy bot

```bash
python main.py
```

---

## ✅ Kiểm tra

### Health check:
```bash
curl http://localhost:8080/health
```

### Metrics:
```bash
curl http://localhost:8080/metrics
```

### Dashboard (nếu cài advanced):
```bash
streamlit run dashboard_app.py
# Access: http://localhost:8501
```

---

## 🔧 Cấu hình nâng cao

### Generate API keys:
```bash
python auth.py
```

Copy keys vào `.env`:
```
API_KEYS=key1,key2,key3
```

### Setup backup:
```bash
# AWS S3
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export S3_BACKUP_BUCKET=your-bucket

# Google Cloud
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
export GCS_BACKUP_BUCKET=your-bucket
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov

# Specific test
pytest tests/unit/test_position_sizing.py -v
```

---

## 📊 Features

### ✅ Đã cài (Minimal):
- Trading bot core
- ML predictions (RF, XGB, LGB)
- Portfolio management
- Telegram notifications
- Basic monitoring

### ✅ Đã cài (Enhanced):
- + Authentication & Security
- + Prometheus metrics
- + Database backup
- + Model monitoring
- + Health checks

### ✅ Đã cài (Advanced):
- + Web dashboard (Streamlit)
- + LSTM predictions
- + Real-time data streaming
- + Advanced analytics

---

## 🆘 Troubleshooting

### Lỗi: "sqlite3 not found"
**Giải pháp:** sqlite3 là built-in, không cần install

### Lỗi: "TensorFlow not compatible"
**Giải pháp:** Dùng Python 3.11 và TensorFlow 2.15+
```bash
pip install tensorflow==2.15.0
```

### Lỗi: "Module not found"
**Giải pháp:** Reinstall dependencies
```bash
pip install -r requirements-minimal.txt --force-reinstall
```

### Bot không chạy
**Kiểm tra:**
1. Python version: `python --version` (should be 3.11.x)
2. Virtual env activated: `which python`
3. .env configured: `cat .env`
4. Database exists: `ls -la trading.db`

---

## 📚 Documentation

- `INSTALLATION_GUIDE.md` - Chi tiết cài đặt
- `IMPLEMENTATION_GUIDE.md` - Hướng dẫn triển khai
- `CRITICAL_IMPROVEMENTS_SUMMARY.md` - Các cải tiến critical
- `IMPORTANT_IMPROVEMENTS_SUMMARY.md` - Các cải tiến important
- `README.md` - Tổng quan dự án

---

## 🎯 Next Steps

Sau khi cài đặt thành công:

1. ✅ Đọc documentation
2. ✅ Chạy tests
3. ✅ Configure .env
4. ✅ Generate API keys
5. ✅ Start bot
6. ✅ Monitor metrics
7. ✅ Check dashboard

---

## 💡 Tips

### Development:
```bash
# Format code
black .
isort .

# Lint
flake8 .

# Type check
mypy .
```

### Production:
```bash
# Run with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

# Run with systemd
sudo systemctl start trading-bot

# Check logs
tail -f logs/trading_bot.log
```

### Monitoring:
```bash
# Watch metrics
watch -n 5 'curl -s http://localhost:8080/metrics | grep trading'

# Watch health
watch -n 10 'curl -s http://localhost:8080/health | jq'
```

---

## ✅ Checklist

- [ ] Python 3.11 installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] .env configured
- [ ] Database initialized
- [ ] Tests passing
- [ ] Bot running
- [ ] Health check OK
- [ ] Telegram working
- [ ] Metrics accessible

---

**Happy Trading! 📈**

Need help? Check `INSTALLATION_GUIDE.md` for detailed instructions.
