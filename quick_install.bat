@echo off
REM Quick Installation Script for Trading Bot (Windows)
REM Usage: quick_install.bat

echo ==================================================
echo 🚀 Trading Bot - Quick Installation (Windows)
echo ==================================================
echo.

REM Check Python version
echo 📋 Checking Python version...
python --version
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
echo    ✅ Virtual environment created
echo.

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat
echo    ✅ Virtual environment activated
echo.

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo    ✅ pip upgraded
echo.

REM Ask which requirements to install
echo 📦 Which version do you want to install?
echo    1) Minimal (Basic features only)
echo    2) Enhanced (Recommended - with monitoring, backup, etc.)
echo    3) Advanced (Full features - with LSTM, Dashboard, etc.)
set /p choice="   Enter choice (1-3): "

if "%choice%"=="1" (
    set requirements_file=requirements-minimal.txt
    echo    Installing minimal version...
) else if "%choice%"=="2" (
    set requirements_file=requirements-enhanced.txt
    echo    Installing enhanced version...
) else if "%choice%"=="3" (
    set requirements_file=requirements-advanced.txt
    echo    Installing advanced version ^(this may take a while^)...
) else (
    set requirements_file=requirements-enhanced.txt
    echo    Invalid choice, using enhanced version...
)

REM Install dependencies
echo.
echo 📥 Installing dependencies from %requirements_file%...
pip install -r %requirements_file%
echo    ✅ Dependencies installed
echo.

REM Setup .env
echo ⚙️  Setting up configuration...
if not exist .env (
    copy .env.example .env >nul
    echo    ✅ .env file created
    echo    ⚠️  Please edit .env with your Telegram credentials
) else (
    echo    ℹ️  .env already exists, skipping
)
echo.

REM Initialize database
echo 🗄️  Initializing database...
python -c "from database import get_db; get_db()" 2>nul
echo    ✅ Database initialized
echo.

REM Run tests
set /p run_tests="🧪 Run tests? (y/n): "
if /i "%run_tests%"=="y" (
    echo    Running tests...
    pytest tests/ -v --tb=short
)
echo.

REM Generate API keys
set /p gen_keys="🔑 Generate API keys? (y/n): "
if /i "%gen_keys%"=="y" (
    echo    Generating API keys...
    python auth.py
    echo.
    echo    ⚠️  Add these keys to your .env file:
    echo    API_KEYS=key1,key2,key3
)
echo.

REM Summary
echo ==================================================
echo ✅ Installation Complete!
echo ==================================================
echo.
echo 📝 Next steps:
echo.
echo 1. Edit .env file with your credentials:
echo    notepad .env
echo.
echo 2. Start the bot:
echo    python main.py
echo.
echo 3. ^(Optional^) Start dashboard:
echo    streamlit run dashboard_app.py
echo.
echo 4. Access:
echo    - Bot API: http://localhost:8080
echo    - Health: http://localhost:8080/health
echo    - Metrics: http://localhost:8080/metrics
echo    - Dashboard: http://localhost:8501 ^(if installed^)
echo.
echo 📚 Documentation:
echo    - INSTALLATION_GUIDE.md
echo    - IMPLEMENTATION_GUIDE.md
echo    - README.md
echo.
echo Happy Trading! 📈
echo.

pause
