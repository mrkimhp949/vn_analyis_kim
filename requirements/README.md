# 📦 Requirements Guide

## Overview

Dependencies are organized by feature for flexible installation.

---

## 📁 Files

### `requirements.txt` (Root)
**Main requirements file** - Includes all core dependencies for production use.

```bash
pip install -r requirements.txt
```

**Includes:**
- Core dependencies
- ML models (XGBoost, LightGBM)
- Monitoring tools
- Basic testing

**Size:** ~500MB

---

### `requirements/base.txt`
**Minimal installation** - Only core dependencies.

```bash
pip install -r requirements/base.txt
```

**Includes:**
- Data processing (pandas, numpy)
- API framework (FastAPI)
- Telegram bot
- Technical analysis

**Size:** ~200MB  
**Use for:** Lightweight deployments, testing

---

### `requirements/ml.txt`
**Machine learning** - Core + ML models.

```bash
pip install -r requirements/ml.txt
```

**Includes:**
- Base requirements
- scikit-learn
- XGBoost, LightGBM
- SHAP (interpretability)
- HMM (market regime)

**Size:** ~400MB  
**Use for:** Trading with ML signals

---

### `requirements/monitoring.txt`
**Monitoring & metrics** - Core + monitoring tools.

```bash
pip install -r requirements/monitoring.txt
```

**Includes:**
- Base requirements
- Prometheus client
- WebSocket support

**Size:** ~250MB  
**Use for:** Production monitoring

---

### `requirements/dev.txt`
**Development tools** - Everything + dev tools.

```bash
pip install -r requirements/dev.txt
```

**Includes:**
- All above requirements
- pytest (testing)
- black, isort (formatting)
- flake8, mypy (linting)
- bandit, safety (security)

**Size:** ~600MB  
**Use for:** Development, testing, CI/CD

---

### `requirements/optional.txt`
**Optional features** - Large or specialized dependencies.

**Commented out by default.** Uncomment what you need:

```bash
# Edit requirements/optional.txt to uncomment features
pip install -r requirements/optional.txt
```

**Available:**
- TensorFlow (~2GB) - LSTM models
- Streamlit - Web dashboard
- Plotly, Seaborn - Visualization
- Optuna - Hyperparameter tuning
- Docker, Kubernetes - Deployment
- Elasticsearch - Log aggregation

---

## 🚀 Installation Scenarios

### Scenario 1: Production Trading Bot
```bash
pip install -r requirements.txt
```

### Scenario 2: Lightweight Bot (No ML)
```bash
pip install -r requirements/base.txt
```

### Scenario 3: Development
```bash
pip install -r requirements/dev.txt
```

### Scenario 4: With LSTM Models
```bash
pip install -r requirements.txt
# Uncomment tensorflow in requirements/optional.txt
pip install tensorflow==2.15.0
```

### Scenario 5: With Dashboard
```bash
pip install -r requirements.txt
pip install streamlit plotly seaborn
```

---

## 🔄 Updating Dependencies

### Check for updates:
```bash
pip list --outdated
```

### Update specific package:
```bash
pip install --upgrade package-name
```

### Update all:
```bash
pip install --upgrade -r requirements.txt
```

---

## 🐛 Troubleshooting

### Issue: Conflicting dependencies
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install fresh
pip install -r requirements.txt
```

### Issue: TensorFlow installation fails
```bash
# Install CPU version
pip install tensorflow-cpu==2.15.0

# Or skip TensorFlow
# (Bot works without LSTM models)
```

### Issue: Slow installation
```bash
# Use pip cache
pip install -r requirements.txt --cache-dir ~/.pip/cache

# Or use conda
conda install --file requirements.txt
```

---

## 📊 Size Comparison

| File | Size | Install Time | Use Case |
|------|------|--------------|----------|
| base.txt | ~200MB | 2-3 min | Minimal |
| ml.txt | ~400MB | 4-5 min | Trading |
| monitoring.txt | ~250MB | 3-4 min | Monitoring |
| requirements.txt | ~500MB | 5-7 min | Production |
| dev.txt | ~600MB | 7-10 min | Development |
| + TensorFlow | ~2.5GB | 15-20 min | LSTM models |

---

## 🎯 Recommendations

### For Production:
```bash
pip install -r requirements.txt
```

### For Development:
```bash
pip install -r requirements/dev.txt
```

### For Testing:
```bash
pip install -r requirements/base.txt
pip install pytest pytest-cov
```

### For CI/CD:
```bash
pip install -r requirements/dev.txt
```

---

**Last Updated:** 15/11/2025  
**Python Version:** 3.11+
