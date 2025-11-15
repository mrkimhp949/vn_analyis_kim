# 📁 NEW PROJECT STRUCTURE

## Overview

Restructured project following Python best practices for better scalability and maintainability.

---

## 🎯 New Structure

```
vn_trading_bot/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── .env
├── .env.example
├── .gitignore
├── pytest.ini
│
├── src/                          # Main source code
│   ├── __init__.py
│   │
│   ├── core/                     # Core business logic
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # Old orchestrator (legacy)
│   │   ├── orchestrator_v2.py    # New orchestrator
│   │   └── bot_runner.py         # Bot runner
│   │
│   ├── services/                 # Business services
│   │   ├── __init__.py
│   │   ├── risk_service.py
│   │   ├── entry_service.py
│   │   ├── exit_service.py
│   │   └── notification_service.py
│   │
│   ├── strategies/               # Trading strategies
│   │   ├── __init__.py
│   │   ├── entry_logic.py
│   │   ├── exit_logic.py
│   │   ├── position_sizing.py
│   │   └── risk_management.py
│   │
│   ├── ml/                       # Machine Learning
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── ensemble.py
│   │   │   ├── lstm.py
│   │   │   └── predictor.py
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── technical.py
│   │   │   └── enhanced.py
│   │   ├── signals/
│   │   │   ├── __init__.py
│   │   │   ├── generator.py
│   │   │   └── enhanced.py
│   │   └── training/
│   │       ├── __init__.py
│   │       ├── trainer.py
│   │       └── pipeline.py
│   │
│   ├── data/                     # Data management
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── cache.py
│   │   ├── database.py
│   │   └── ticker_loader.py
│   │
│   ├── portfolio/                # Portfolio management
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   ├── analyzer.py
│   │   ├── lock.py
│   │   └── paper_trading.py
│   │
│   ├── risk/                     # Risk management
│   │   ├── __init__.py
│   │   ├── circuit_breaker.py
│   │   ├── emergency_stop.py
│   │   ├── metrics.py
│   │   └── validator.py
│   │
│   ├── market/                   # Market analysis
│   │   ├── __init__.py
│   │   ├── regime.py
│   │   ├── sector_analysis.py
│   │   └── schedule.py
│   │
│   ├── monitoring/               # Monitoring & metrics
│   │   ├── __init__.py
│   │   ├── performance.py
│   │   ├── enhanced.py
│   │   └── health.py
│   │
│   ├── notifications/            # Notifications
│   │   ├── __init__.py
│   │   ├── telegram.py
│   │   └── subscriptions.py
│   │
│   ├── api/                      # API & Web
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── bot.py
│   │   │   ├── portfolio.py
│   │   │   └── health.py
│   │   └── auth.py
│   │
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── indicators.py
│   │   ├── validation.py
│   │   ├── logging_config.py
│   │   └── suppress_warnings.py
│   │
│   └── config/                   # Configuration
│       ├── __init__.py
│       ├── trading_config.py
│       ├── settings.py
│       └── exceptions.py
│
├── tests/                        # Tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_critical_fixes.py
│   │   ├── test_services.py
│   │   ├── test_strategies.py
│   │   └── test_portfolio.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_orchestrator.py
│   │   └── test_api.py
│   └── fixtures/
│       ├── __init__.py
│       └── sample_data.py
│
├── scripts/                      # Utility scripts
│   ├── migrate_structure.py     # Migration script
│   ├── train_models.py
│   ├── backtest.py
│   ├── validate_data.py
│   └── quick_install.sh
│
├── docs/                         # Documentation
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── DEPLOYMENT.md
│   ├── analysis/
│   │   ├── PHAN_TICH_DU_AN.md
│   │   └── improvements/
│   └── guides/
│       ├── QUICK_START.md
│       └── INSTALLATION.md
│
├── data/                         # Data files
│   ├── tickers/
│   │   └── List.csv
│   ├── models/
│   │   └── README.md
│   └── cache/
│
├── logs/                         # Log files
│   └── .gitkeep
│
├── backups/                      # Backups
│   └── .gitkeep
│
└── notebooks/                    # Jupyter notebooks
    ├── analysis.ipynb
    └── backtesting.ipynb
```

---

## 🎯 Key Improvements

### 1. Clear Separation of Concerns
- **src/core/** - Core orchestration logic
- **src/services/** - Business services
- **src/strategies/** - Trading strategies
- **src/ml/** - Machine learning components
- **src/data/** - Data management
- **src/portfolio/** - Portfolio management
- **src/risk/** - Risk management
- **src/market/** - Market analysis
- **src/monitoring/** - Monitoring & metrics
- **src/api/** - API endpoints
- **src/utils/** - Utilities
- **src/config/** - Configuration

### 2. Better Organization
- All source code in `src/`
- All tests in `tests/`
- All docs in `docs/`
- All scripts in `scripts/`
- Clear module boundaries

### 3. Scalability
- Easy to add new services
- Easy to add new strategies
- Easy to add new ML models
- Easy to add new features

### 4. Maintainability
- Clear file locations
- Logical grouping
- Easy to navigate
- Easy to understand

---

## 📦 Module Descriptions

### src/core/
Core orchestration and bot runner logic. Entry point for the trading bot.

### src/services/
Business services following service-oriented architecture. Each service has single responsibility.

### src/strategies/
Trading strategies including entry logic, exit logic, position sizing, and risk management.

### src/ml/
Machine learning components organized by function:
- **models/** - ML model implementations
- **features/** - Feature engineering
- **signals/** - Signal generation
- **training/** - Training pipelines

### src/data/
Data management including loaders, cache, database, and ticker management.

### src/portfolio/
Portfolio management including manager, analyzer, lock, and paper trading.

### src/risk/
Risk management components including circuit breaker, emergency stop, and validators.

### src/market/
Market analysis including regime detection, sector analysis, and trading schedule.

### src/monitoring/
Monitoring and metrics including performance tracking and health checks.

### src/notifications/
Notification systems including Telegram bot and subscriptions.

### src/api/
FastAPI application with routes for bot control, portfolio management, and health checks.

### src/utils/
Utility functions including indicators, validation, logging, and warnings suppression.

### src/config/
Configuration management including trading config, settings, and exceptions.

---

## 🔄 Migration Strategy

### Phase 1: Create New Structure
1. Create all directories
2. Create __init__.py files
3. Set up imports

### Phase 2: Move Files
1. Move files to new locations
2. Update imports
3. Update references

### Phase 3: Update Tests
1. Update test imports
2. Update test paths
3. Verify all tests pass

### Phase 4: Update Documentation
1. Update README
2. Update guides
3. Update API docs

### Phase 5: Cleanup
1. Remove old files
2. Update .gitignore
3. Final verification

---

## 🚀 Benefits

### For Development:
- ✅ Clear module boundaries
- ✅ Easy to find files
- ✅ Easy to add features
- ✅ Better IDE support

### For Testing:
- ✅ Clear test organization
- ✅ Easy to write tests
- ✅ Better test coverage
- ✅ Faster test execution

### For Deployment:
- ✅ Clean package structure
- ✅ Easy to install
- ✅ Better dependency management
- ✅ Easier Docker setup

### For Maintenance:
- ✅ Easy to understand
- ✅ Easy to modify
- ✅ Easy to debug
- ✅ Better documentation

---

## 📝 Next Steps

1. Review structure
2. Run migration script
3. Update imports
4. Run tests
5. Update documentation
6. Deploy

---

**Status:** Ready for migration  
**Estimated Time:** 2-3 hours  
**Risk:** Low (automated migration)
