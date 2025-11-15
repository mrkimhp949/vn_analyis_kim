# 🔄 MIGRATION GUIDE - Import Updates

**Version:** 2.0  
**Date:** 15/11/2025

---

## 📋 OVERVIEW

This guide helps you update imports after project restructuring.

---

## 🔄 IMPORT CHANGES

### Core Modules

**Before:**
```python
from orchestrator import TradingOrchestrator
from bot_runner_improved import run_bot_sync
```

**After:**
```python
from src.core.orchestrator_v2 import TradingOrchestratorV2
from src.core.bot_runner import run_bot_sync
```

---

### Services

**Before:**
```python
from services.risk_service import RiskManagementService
from services.entry_service import EntrySignalService
```

**After:**
```python
from src.services.risk_service import RiskManagementService
from src.services.entry_service import EntrySignalService
# Or use convenience imports
from src.services import get_risk_service, get_entry_service
```

---

### Strategies

**Before:**
```python
from improved_entry_logic import ImprovedEntryLogic
from improved_exit_logic import ImprovedExitStrategy
from position_sizing_enhanced import EnhancedPositionSizer
```

**After:**
```python
from src.strategies.entry_logic import ImprovedEntryLogic
from src.strategies.exit_logic import ImprovedExitStrategy
from src.strategies.position_sizing import EnhancedPositionSizer
```

---

### Machine Learning

**Before:**
```python
from ml_models import MLPredictor
from ml_signals_enhanced import EnhancedMLSignalGenerator
from features_enhanced import add_enhanced_features
```

**After:**
```python
from src.ml.models.predictor import MLPredictor
from src.ml.signals.enhanced import EnhancedMLSignalGenerator
from src.ml.features.enhanced import add_enhanced_features
```

---

### Data Management

**Before:**
```python
from data_loader import load_data
from database import get_db
from ticker_loader import get_ticker_loader
```

**After:**
```python
from src.data.loader import load_data
from src.data.database import get_db
from src.data.ticker_loader import get_ticker_loader
```

---

### Portfolio

**Before:**
```python
from portfolio_manager import PortfolioManager, get_portfolio_manager
from portfolio_analyzer import PortfolioAnalyzer
from paper_trading import get_paper_account
```

**After:**
```python
from src.portfolio.manager import PortfolioManager, get_portfolio_manager
from src.portfolio.analyzer import PortfolioAnalyzer
from src.portfolio.paper_trading import get_paper_account
```

---

### Risk Management

**Before:**
```python
from circuit_breaker import get_circuit_breaker
from emergency_stop import get_emergency_stop
from risk_metrics import RiskMetrics
```

**After:**
```python
from src.risk.circuit_breaker import get_circuit_breaker
from src.risk.emergency_stop import get_emergency_stop
from src.risk.metrics import RiskMetrics
```

---

### Market Analysis

**Before:**
```python
from market_regime_proxy import ProxyMarketRegimeAnalyzer
from improved_sector_analysis import analyze_sectors
from vn_trading_schedule import is_trading_hour
```

**After:**
```python
from src.market.regime_proxy import ProxyMarketRegimeAnalyzer
from src.market.sector_analysis import analyze_sectors
from src.market.schedule import is_trading_hour
```

---

### Monitoring

**Before:**
```python
from monitoring import get_performance_monitor
from monitoring_enhanced import get_enhanced_monitor
```

**After:**
```python
from src.monitoring.performance import get_performance_monitor
from src.monitoring.enhanced import get_enhanced_monitor
```

---

### Utilities

**Before:**
```python
from utils.indicators import StopLossCalculator
from utils.validation import DataValidator
from logging_config import setup_logging
```

**After:**
```python
from src.utils.indicators import StopLossCalculator
from src.utils.validation import DataValidator
from src.utils.logging_config import setup_logging
```

---

### Configuration

**Before:**
```python
from config import TICKERS, CHAT_ID
from trading_config import get_config
from exceptions import TradingError
```

**After:**
```python
from src.config.legacy_config import TICKERS, CHAT_ID
from src.config.trading_config import get_config
from src.config.exceptions import TradingError
```

---

## 🔧 QUICK FIND & REPLACE

Use these patterns for bulk updates:

```bash
# Core
find . -name "*.py" -exec sed -i 's/from orchestrator import/from src.core.orchestrator import/g' {} +
find . -name "*.py" -exec sed -i 's/from bot_runner_improved import/from src.core.bot_runner import/g' {} +

# Services
find . -name "*.py" -exec sed -i 's/from services\./from src.services./g' {} +

# Strategies
find . -name "*.py" -exec sed -i 's/from improved_entry_logic import/from src.strategies.entry_logic import/g' {} +
find . -name "*.py" -exec sed -i 's/from improved_exit_logic import/from src.strategies.exit_logic import/g' {} +

# ML
find . -name "*.py" -exec sed -i 's/from ml_signals_enhanced import/from src.ml.signals.enhanced import/g' {} +
find . -name "*.py" -exec sed -i 's/from ml_models_enhanced import/from src.ml.models.ensemble import/g' {} +

# Data
find . -name "*.py" -exec sed -i 's/from data_loader import/from src.data.loader import/g' {} +
find . -name "*.py" -exec sed -i 's/from database import/from src.data.database import/g' {} +

# Portfolio
find . -name "*.py" -exec sed -i 's/from portfolio_manager import/from src.portfolio.manager import/g' {} +

# Risk
find . -name "*.py" -exec sed -i 's/from circuit_breaker import/from src.risk.circuit_breaker import/g' {} +

# Utils
find . -name "*.py" -exec sed -i 's/from utils\./from src.utils./g' {} +

# Config
find . -name "*.py" -exec sed -i 's/from trading_config import/from src.config.trading_config import/g' {} +
```

---

## ✅ VERIFICATION

After updating imports, verify:

```bash
# 1. Check syntax
python -m py_compile src/**/*.py

# 2. Run tests
pytest tests/ -v

# 3. Try importing
python -c "from src.core import TradingOrchestratorV2; print('OK')"

# 4. Check for old imports
grep -r "from orchestrator import" src/
grep -r "from portfolio_manager import" src/
```

---

## 🐛 TROUBLESHOOTING

### Issue: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'orchestrator'
```

**Solution:**
```python
# Change:
from orchestrator import TradingOrchestrator

# To:
from src.core.orchestrator import TradingOrchestrator
```

### Issue: Circular Import

**Error:**
```
ImportError: cannot import name 'X' from partially initialized module
```

**Solution:**
- Move import inside function
- Use `from typing import TYPE_CHECKING`
- Restructure dependencies

### Issue: Old Files Still Referenced

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'config.py'
```

**Solution:**
```python
# Update to new location
from src.config.legacy_config import TICKERS
```

---

## 📝 CHECKLIST

- [ ] Update all imports in `src/`
- [ ] Update all imports in `tests/`
- [ ] Update all imports in `scripts/`
- [ ] Run syntax check
- [ ] Run tests
- [ ] Verify no old imports remain
- [ ] Update documentation
- [ ] Commit changes

---

**Last Updated:** 15/11/2025  
**Status:** Active
