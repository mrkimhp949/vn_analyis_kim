# 🏗️ ARCHITECTURE GUIDE

**Version:** 2.0  
**Last Updated:** 15/11/2025

---

## 📋 TABLE OF CONTENTS

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Module Descriptions](#module-descriptions)
4. [Data Flow](#data-flow)
5. [Service Architecture](#service-architecture)
6. [Best Practices](#best-practices)

---

## 🎯 OVERVIEW

VN Trading Bot follows a **Service-Oriented Architecture (SOA)** with clear separation of concerns. The project is organized into logical modules for better maintainability and scalability.

### Key Principles:
- **Modularity:** Each module has a single responsibility
- **Loose Coupling:** Modules communicate through well-defined interfaces
- **High Cohesion:** Related functionality grouped together
- **Testability:** Easy to test individual components
- **Scalability:** Easy to add new features

---

## 📁 PROJECT STRUCTURE

```
vn_trading_bot/
├── src/                    # Source code
│   ├── core/              # Core orchestration
│   ├── services/          # Business services
│   ├── strategies/        # Trading strategies
│   ├── ml/                # Machine learning
│   ├── data/              # Data management
│   ├── portfolio/         # Portfolio management
│   ├── risk/              # Risk management
│   ├── market/            # Market analysis
│   ├── monitoring/        # Monitoring & metrics
│   ├── notifications/     # Notifications
│   ├── api/               # API endpoints
│   ├── utils/             # Utilities
│   └── config/            # Configuration
│
├── tests/                 # Tests
├── scripts/               # Utility scripts
├── docs/                  # Documentation
├── data/                  # Data files
└── logs/                  # Log files
```

---

## 📦 MODULE DESCRIPTIONS

### 1. src/core/ - Core Orchestration

**Purpose:** Main orchestration logic and bot runner

**Files:**
- `orchestrator.py` - Legacy orchestrator (V1)
- `orchestrator_v2.py` - New service-oriented orchestrator
- `bot_runner.py` - Bot runner and entry point

**Responsibilities:**
- Coordinate services
- Manage scan workflow
- Handle errors gracefully
- Control execution flow

**Example:**
```python
from src.core import TradingOrchestratorV2

orchestrator = TradingOrchestratorV2(bot, chat_id)
await orchestrator.run_scan(market_regime)
```

---

### 2. src/services/ - Business Services

**Purpose:** Business logic services following SOA

**Files:**
- `risk_service.py` - Risk management service
- `entry_service.py` - Entry signal service
- `exit_service.py` - Exit management service
- `notification_service.py` - Notification service

**Responsibilities:**
- Encapsulate business logic
- Provide clean interfaces
- Handle service-specific errors
- Maintain service state

**Example:**
```python
from src.services import get_risk_service

risk_service = get_risk_service()
can_trade, reason = await risk_service.can_trade()
```

---

### 3. src/strategies/ - Trading Strategies

**Purpose:** Trading strategy implementations

**Files:**
- `entry_logic.py` - Entry signal logic
- `exit_logic.py` - Exit signal logic
- `position_sizing.py` - Position sizing logic
- `risk_management.py` - Risk management logic
- `manager.py` - Strategy manager

**Responsibilities:**
- Implement trading strategies
- Calculate entry/exit signals
- Determine position sizes
- Manage risk parameters

**Example:**
```python
from src.strategies import ImprovedEntryLogic

entry_logic = ImprovedEntryLogic()
signal = entry_logic.analyze_entry(df, ml_signal, market_regime)
```

---

### 4. src/ml/ - Machine Learning

**Purpose:** ML models and signal generation

**Structure:**
```
ml/
├── models/          # ML model implementations
├── features/        # Feature engineering
├── signals/         # Signal generation
└── training/        # Training pipelines
```

**Responsibilities:**
- Train ML models
- Generate features
- Produce trading signals
- Monitor model performance

**Example:**
```python
from src.ml.signals import EnhancedMLSignalGenerator

ml_gen = EnhancedMLSignalGenerator()
signal = ml_gen.analyze(df, vnindex_df)
```

---

### 5. src/data/ - Data Management

**Purpose:** Data loading, caching, and storage

**Files:**
- `loader.py` - Data loading
- `database.py` - Database operations
- `ticker_loader.py` - Ticker management
- `cache.py` - Caching layer

**Responsibilities:**
- Load market data
- Manage database
- Handle caching
- Validate data quality

**Example:**
```python
from src.data import load_data

df = load_data('VNM', lookback=200)
```

---

### 6. src/portfolio/ - Portfolio Management

**Purpose:** Portfolio tracking and management

**Files:**
- `manager.py` - Portfolio manager
- `analyzer.py` - Portfolio analyzer
- `lock.py` - Portfolio lock
- `paper_trading.py` - Paper trading

**Responsibilities:**
- Track positions
- Calculate P&L
- Manage trades
- Analyze performance

**Example:**
```python
from src.portfolio import get_portfolio_manager

pm = get_portfolio_manager()
pm.add_position('VNM', 100, 80_000)
```

---

### 7. src/risk/ - Risk Management

**Purpose:** Risk controls and validation

**Files:**
- `circuit_breaker.py` - Circuit breaker
- `emergency_stop.py` - Emergency stop
- `metrics.py` - Risk metrics
- `validator.py` - Risk validator

**Responsibilities:**
- Monitor risk levels
- Enforce limits
- Trigger stops
- Track metrics

**Example:**
```python
from src.risk import get_circuit_breaker

cb = get_circuit_breaker()
can_trade, reason = cb.can_trade()
```

---

### 8. src/market/ - Market Analysis

**Purpose:** Market regime and sector analysis

**Files:**
- `regime.py` - Market regime detection
- `regime_proxy.py` - Proxy regime analyzer
- `sector_analysis.py` - Sector analysis
- `schedule.py` - Trading schedule

**Responsibilities:**
- Detect market regime
- Analyze sectors
- Check trading hours
- Validate trading days

**Example:**
```python
from src.market import ProxyMarketRegimeAnalyzer

analyzer = ProxyMarketRegimeAnalyzer()
regime = analyzer.analyze_market_regime(vnindex_df)
```

---

### 9. src/monitoring/ - Monitoring & Metrics

**Purpose:** Performance monitoring and metrics

**Files:**
- `performance.py` - Performance tracking
- `enhanced.py` - Enhanced monitoring
- `health.py` - Health checks
- `prometheus.py` - Prometheus metrics

**Responsibilities:**
- Track performance
- Collect metrics
- Health checks
- Export metrics

**Example:**
```python
from src.monitoring import get_performance_monitor

monitor = get_performance_monitor()
metrics = monitor.get_metrics()
```

---

### 10. src/notifications/ - Notifications

**Purpose:** Notification systems

**Files:**
- `telegram.py` - Telegram notifications
- `subscriptions.py` - Subscription management
- `listener.py` - Telegram listener

**Responsibilities:**
- Send notifications
- Manage subscriptions
- Handle commands
- Format messages

**Example:**
```python
from src.notifications import send_telegram_message

await send_telegram_message(chat_id, "Signal detected!")
```

---

### 11. src/api/ - API Endpoints

**Purpose:** FastAPI application

**Structure:**
```
api/
├── main.py          # FastAPI app
├── auth.py          # Authentication
└── routes/          # API routes
    ├── bot.py
    ├── portfolio.py
    └── health.py
```

**Responsibilities:**
- Expose REST API
- Handle authentication
- Manage routes
- Return responses

**Example:**
```python
# Start API server
python -m src.api.main
```

---

### 12. src/utils/ - Utilities

**Purpose:** Utility functions

**Files:**
- `indicators.py` - Technical indicators
- `validation.py` - Data validation
- `logging_config.py` - Logging setup
- `suppress_warnings.py` - Warning suppression

**Responsibilities:**
- Provide utilities
- Validate data
- Configure logging
- Helper functions

**Example:**
```python
from src.utils import StopLossCalculator

sl, reason = StopLossCalculator.calculate_stop_loss(...)
```

---

### 13. src/config/ - Configuration

**Purpose:** Configuration management

**Files:**
- `trading_config.py` - Trading configuration
- `settings.py` - Application settings
- `exceptions.py` - Custom exceptions

**Responsibilities:**
- Manage configuration
- Validate settings
- Define exceptions
- Environment variables

**Example:**
```python
from src.config import get_config

config = get_config()
print(config.trading.min_confidence)
```

---

## 🔄 DATA FLOW

### 1. Scan Workflow

```
┌─────────────────┐
│  Bot Runner     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Orchestrator V2 │
└────────┬────────┘
         │
         ├──────────────────────────────────┐
         │                                  │
         ▼                                  ▼
┌─────────────────┐              ┌─────────────────┐
│  Risk Service   │              │  Data Loader    │
└────────┬────────┘              └────────┬────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────┐              ┌─────────────────┐
│  Entry Service  │◄─────────────│  ML Signals     │
└────────┬────────┘              └─────────────────┘
         │
         ▼
┌─────────────────┐
│  Exit Service   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Notification    │
│    Service      │
└─────────────────┘
```

### 2. Entry Signal Flow

```
Ticker → Data Loader → ML Analysis → Entry Logic → Position Sizing → Notification
```

### 3. Exit Signal Flow

```
Position → Data Loader → ML Analysis → Exit Logic → Execution → Notification
```

---

## 🏛️ SERVICE ARCHITECTURE

### Service Communication

Services communicate through:
1. **Direct calls** - For synchronous operations
2. **Async/await** - For I/O operations
3. **Dependency injection** - For loose coupling
4. **Singleton pattern** - For shared state

### Service Lifecycle

```python
# 1. Initialization
service = get_service()

# 2. Operation
result = await service.do_something()

# 3. Cleanup (if needed)
service.cleanup()
```

### Error Handling

```python
try:
    result = await service.operation()
except ServiceError as e:
    logger.error(f"Service error: {e}")
    # Handle gracefully
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    # Escalate
```

---

## 📝 BEST PRACTICES

### 1. Module Organization

✅ **DO:**
- Keep related code together
- Use clear module names
- Create __init__.py files
- Document module purpose

❌ **DON'T:**
- Mix unrelated code
- Use vague names
- Create circular imports
- Leave modules undocumented

### 2. Import Statements

✅ **DO:**
```python
# Absolute imports
from src.services import get_risk_service
from src.utils import DataValidator

# Relative imports within package
from .models import Predictor
from ..utils import helper_function
```

❌ **DON'T:**
```python
# Avoid wildcard imports
from src.services import *

# Avoid deep relative imports
from ....utils import something
```

### 3. Service Design

✅ **DO:**
- Single responsibility
- Clear interfaces
- Dependency injection
- Error handling

❌ **DON'T:**
- Multiple responsibilities
- Tight coupling
- Global state
- Silent failures

### 4. Testing

✅ **DO:**
- Test each module
- Mock dependencies
- Test edge cases
- Use fixtures

❌ **DON'T:**
- Test multiple modules together
- Use real dependencies
- Only test happy path
- Duplicate test code

---

## 🚀 GETTING STARTED

### 1. Installation

```bash
# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### 2. Running the Bot

```bash
# Using Python module
python -m src.api.main

# Or using entry point
trading-bot
```

### 3. Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/unit/test_services.py

# With coverage
pytest tests/ --cov=src
```

---

## 📚 ADDITIONAL RESOURCES

- [Deployment Guide](DEPLOYMENT.md)
- [API Documentation](API.md)
- [Quick Start Guide](guides/QUICK_START.md)
- [Installation Guide](guides/INSTALLATION.md)

---

**Maintained by:** Development Team  
**Last Review:** 15/11/2025  
**Status:** Active
