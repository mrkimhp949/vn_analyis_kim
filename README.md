# 🤖 Trading Bot - Deployment Guide

## 🚀 Quick Deploy on Render

### Method 1: One-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Method 2: Manual Deploy

1. **Fork this repository**
2. **Go to [Render.com](https://render.com)**
3. **Click "New +" → "Web Service"**
4. **Connect your GitHub repository**
5. **Configure:**

   - **Name:** `trading-bot`
   - **Environment:** `Python`
   - **Region:** `Singapore`
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python main.py`

6. **Add Environment Variables:**

   - `TELEGRAM_TOKEN` = your_bot_token
   - `CHAT_ID` = your_chat_id

7. **Click "Create Web Service"**

### Method 3: Using Render Blueprint

1. **Fork this repository**
2. **In Render Dashboard, click "New +" → "Blueprint"**
3. **Connect your repository**
4. **Render will automatically detect `render.yaml`**

## 🔧 Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your values

# Run locally
python main.py
```
