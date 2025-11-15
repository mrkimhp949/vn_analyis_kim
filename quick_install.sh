#!/bin/bash
# Quick Installation Script for Trading Bot
# Usage: bash quick_install.sh

set -e  # Exit on error

echo "=================================================="
echo "🚀 Trading Bot - Quick Installation"
echo "=================================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found: Python $python_version"

if [[ ! "$python_version" =~ ^3\.11\. ]]; then
    echo "⚠️  Warning: Python 3.11 recommended, you have $python_version"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv
echo "   ✅ Virtual environment created"

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "   ✅ pip upgraded"

# Ask which requirements to install
echo ""
echo "📦 Which version do you want to install?"
echo "   1) Minimal (Basic features only)"
echo "   2) Enhanced (Recommended - with monitoring, backup, etc.)"
echo "   3) Advanced (Full features - with LSTM, Dashboard, etc.)"
read -p "   Enter choice (1-3): " choice

case $choice in
    1)
        requirements_file="requirements-minimal.txt"
        echo "   Installing minimal version..."
        ;;
    2)
        requirements_file="requirements-enhanced.txt"
        echo "   Installing enhanced version..."
        ;;
    3)
        requirements_file="requirements-advanced.txt"
        echo "   Installing advanced version (this may take a while)..."
        ;;
    *)
        echo "   Invalid choice, using enhanced version..."
        requirements_file="requirements-enhanced.txt"
        ;;
esac

# Install dependencies
echo ""
echo "📥 Installing dependencies from $requirements_file..."
pip install -r $requirements_file
echo "   ✅ Dependencies installed"

# Setup .env
echo ""
echo "⚙️  Setting up configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "   ✅ .env file created"
    echo "   ⚠️  Please edit .env with your Telegram credentials"
else
    echo "   ℹ️  .env already exists, skipping"
fi

# Initialize database
echo ""
echo "🗄️  Initializing database..."
python3 -c "from database import get_db; get_db()" 2>/dev/null
echo "   ✅ Database initialized"

# Run tests
echo ""
read -p "🧪 Run tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Running tests..."
    pytest tests/ -v --tb=short || echo "   ⚠️  Some tests failed (this is OK for first install)"
fi

# Generate API keys
echo ""
read -p "🔑 Generate API keys? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Generating API keys..."
    python3 auth.py
    echo ""
    echo "   ⚠️  Add these keys to your .env file:"
    echo "   API_KEYS=key1,key2,key3"
fi

# Summary
echo ""
echo "=================================================="
echo "✅ Installation Complete!"
echo "=================================================="
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Edit .env file with your credentials:"
echo "   nano .env"
echo ""
echo "2. Start the bot:"
echo "   python main.py"
echo ""
echo "3. (Optional) Start dashboard:"
echo "   streamlit run dashboard_app.py"
echo ""
echo "4. Access:"
echo "   - Bot API: http://localhost:8080"
echo "   - Health: http://localhost:8080/health"
echo "   - Metrics: http://localhost:8080/metrics"
echo "   - Dashboard: http://localhost:8501 (if installed)"
echo ""
echo "📚 Documentation:"
echo "   - INSTALLATION_GUIDE.md"
echo "   - IMPLEMENTATION_GUIDE.md"
echo "   - README.md"
echo ""
echo "Happy Trading! 📈"
echo ""
