# 🚀 Deployment Guide

Complete guide to deploy the Vietnam Trading Bot to production.

---

## 📋 **PRE-DEPLOYMENT CHECKLIST**

Before deploying, ensure you have:

- [ ] Telegram Bot Token from [@BotFather](https://t.me/BotFather)
- [ ] Telegram Chat ID (get from `/start` command to your bot)
- [ ] Server/hosting platform (Render, Heroku, VPS, etc.)
- [ ] Git repository access
- [ ] Environment variables prepared

---

## 🔧 **ENVIRONMENT VARIABLES**

### **Required Variables**

```bash
# Telegram Configuration
TELEGRAM_TOKEN=your_telegram_bot_token_here
CHAT_ID=your_telegram_chat_id_here
```

### **Optional Configuration**

```bash
# Trading Parameters
MIN_CONFIDENCE=60              # Minimum signal confidence (0-100)
MIN_RISK_REWARD=2.0           # Minimum risk/reward ratio
SUPPORT_DISTANCE_PERCENT=3.0  # Max distance to support (%)
MAX_POSITIONS=10              # Maximum concurrent positions
MAX_SECTOR_EXPOSURE=0.40      # Max 40% per sector

# Risk Management
STOP_LOSS_PERCENT=-7.0        # Default stop loss (%)
TAKE_PROFIT_PERCENT=15.0      # Default take profit (%)
TRAILING_STOP_PERCENT=3.0     # Trailing stop distance (%)
MAX_LOSS_PER_DAY_PCT=5.0      # Circuit breaker threshold (%)

# Position Sizing
MAX_POSITION_SIZE=0.15        # Max 15% per position
MIN_POSITION_SIZE=0.05        # Min 5% per position

# Data Configuration
LOOKBACK=200                  # Historical data lookback period
MIN_VOLUME=100000             # Minimum volume filter
USE_CSV_TICKERS=true          # Load tickers from List.csv

# API Configuration
TCBS_RATE_LIMIT=10            # TCBS API calls per second
REQUEST_TIMEOUT=10            # API request timeout (seconds)

# Server Configuration
PORT=8080                     # Server port
DEBUG=false                   # Debug mode (true/false)
```

---

## 🌐 **DEPLOYMENT OPTIONS**

### **Option 1: Render.com (Recommended)**

**Advantages:**
- Free tier available
- Auto-deploy from GitHub
- Built-in monitoring
- Easy environment variable management

**Steps:**

1. **Fork Repository**
   ```bash
   # Fork on GitHub
   https://github.com/mrkimhp949/vn_analyis_kim
   ```

2. **Create New Web Service on Render**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure Service**
   ```yaml
   Name: vn-trading-bot
   Environment: Python 3.11
   Region: Singapore (closest to Vietnam)
   Branch: main
   Build Command: pip install -r requirements.txt
   Start Command: python main.py
   ```

4. **Add Environment Variables**
   - Go to "Environment" tab
   - Add `TELEGRAM_TOKEN` and `CHAT_ID`
   - Add any optional variables

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)
   - Check logs for startup messages

6. **Verify Deployment**
   ```bash
   curl https://your-app.onrender.com/health
   ```

---

### **Option 2: Heroku**

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku

   # Ubuntu
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Login and Create App**
   ```bash
   heroku login
   heroku create vn-trading-bot
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set TELEGRAM_TOKEN=your_token
   heroku config:set CHAT_ID=your_chat_id
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **Check Logs**
   ```bash
   heroku logs --tail
   ```

---

### **Option 3: Docker (Any Platform)**

1. **Build Image**
   ```bash
   docker build -t vn-trading-bot .
   ```

2. **Run Container**
   ```bash
   docker run -d \
     --name trading-bot \
     -e TELEGRAM_TOKEN=your_token \
     -e CHAT_ID=your_chat_id \
     -p 8080:8080 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     vn-trading-bot
   ```

3. **Check Status**
   ```bash
   docker logs -f trading-bot
   ```

---

### **Option 4: Ubuntu VPS**

1. **Setup Server**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install Python 3.11
   sudo apt install python3.11 python3.11-venv python3-pip -y

   # Install Git
   sudo apt install git -y
   ```

2. **Clone Repository**
   ```bash
   git clone https://github.com/mrkimhp949/vn_analyis_kim.git
   cd vn_analyis_kim
   ```

3. **Setup Virtual Environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   # Create .env file
   cat > .env << EOF
   TELEGRAM_TOKEN=your_token
   CHAT_ID=your_chat_id
   EOF
   ```

5. **Setup Systemd Service**
   ```bash
   sudo nano /etc/systemd/system/trading-bot.service
   ```

   ```ini
   [Unit]
   Description=Vietnam Trading Bot
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/vn_analyis_kim
   Environment="PATH=/home/ubuntu/vn_analyis_kim/venv/bin"
   ExecStart=/home/ubuntu/vn_analyis_kim/venv/bin/python main.py
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

6. **Start Service**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable trading-bot
   sudo systemctl start trading-bot
   sudo systemctl status trading-bot
   ```

7. **View Logs**
   ```bash
   sudo journalctl -u trading-bot -f
   ```

---

## ✅ **POST-DEPLOYMENT VERIFICATION**

### 1. **Check API Health**
```bash
# Replace with your deployment URL
curl https://your-app.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-14T10:30:00",
  "data_loader": "OK",
  "ml_models": "OK"
}
```

### 2. **Run Health Check**
```bash
# If deployed locally or on VPS
python health_check.py
```

### 3. **Test Telegram Bot**
Send `/status` command to your Telegram bot. Expected response:
```
🤖 Bot Status
Status: Online ✅
Uptime: 5 minutes
Positions: 0
```

### 4. **View Metrics**
```bash
python scripts/view_metrics.py
```

### 5. **Monitor Logs**
Check logs for:
- ✅ Configuration validated successfully
- ✅ Scheduler started
- ✅ Telegram bot started
- ❌ No error messages

---

## 🔍 **MONITORING**

### **Automated Health Checks**

Add to cron (VPS) or use monitoring service:

```bash
# Check every 5 minutes
*/5 * * * * /path/to/venv/bin/python /path/to/health_check.py
```

### **Log Monitoring**

```bash
# Tail logs
tail -f logs/trading_bot.log

# Or systemd logs
sudo journalctl -u trading-bot -f
```

### **Performance Metrics**

```bash
# View daily
python scripts/view_metrics.py

# Export to JSON
python scripts/view_metrics.py --export metrics_$(date +%Y%m%d).json
```

---

## 🐛 **TROUBLESHOOTING**

### **Bot Won't Start**

```bash
# Check configuration
python -c "from trading_config import get_config; get_config(validate=True)"

# Check dependencies
pip install -r requirements.txt

# Check environment variables
printenv | grep TELEGRAM
```

### **Configuration Errors**

```
❌ FATAL: Configuration validation failed
```

**Solution:**
```bash
export TELEGRAM_TOKEN="your_token_here"
export CHAT_ID="your_chat_id_here"
```

### **Database Locked**

```
database is locked
```

**Solution:** Already fixed with WAL mode, but if persists:
```bash
# Check for stale locks
fuser trading.db
kill -9 <pid>
```

### **API Errors**

```
TCBS API returned status 429
```

**Solution:** Retry logic already implemented with exponential backoff

### **No Signals Generated**

**Check:**
1. Market hours (VN: 9:00-11:30, 13:00-15:00)
2. List.csv has valid tickers
3. ML models loaded: `python -c "from ml_models import MLPredictor; MLPredictor().load_models()"`

---

## 🔄 **UPDATES & MAINTENANCE**

### **Update Code**

```bash
# Pull latest changes
git pull origin main

# Restart service
sudo systemctl restart trading-bot

# Or Docker
docker restart trading-bot

# Or Render (auto-deploys from GitHub)
git push origin main
```

### **Backup Data**

```bash
# Backup database
cp trading.db trading_backup_$(date +%Y%m%d).db

# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

### **Clean Cache**

```bash
# Remove old cache (older than 7 days)
find data_cache -name "*.pkl" -mtime +7 -delete

# Or clear all
rm -rf data_cache/*.pkl
```

---

## 📊 **PERFORMANCE OPTIMIZATION**

### **For High-Frequency Scanning**

```bash
# Increase TCBS rate limit (if you have premium API)
export TCBS_RATE_LIMIT=20

# Reduce lookback period
export LOOKBACK=100
```

### **For Resource-Constrained Servers**

```bash
# Limit max positions
export MAX_POSITIONS=5

# Disable heavy ML features
# (Use dummy models - automatic fallback)
```

---

## 🔒 **SECURITY BEST PRACTICES**

1. **Never commit credentials**
   ```bash
   # .gitignore already includes
   .env
   *.db
   ```

2. **Use environment variables**
   - Store in hosting platform's secret manager
   - Never hardcode in code

3. **Secure API access**
   - Use HTTPS for all API calls
   - Enable rate limiting

4. **Regular updates**
   ```bash
   # Update dependencies monthly
   pip install -U -r requirements.txt
   ```

5. **Monitor logs**
   - Check for suspicious activity
   - Set up alerts for errors

---

## 📞 **SUPPORT**

**Issues:** https://github.com/mrkimhp949/vn_analyis_kim/issues

**Documentation:** Check README.md and code comments

**Health Check:** `python health_check.py`

**Metrics:** `python scripts/view_metrics.py`

---

**Good luck with your deployment! 🚀📈**
