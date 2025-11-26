# 🤖 ML Integration Modules

Complete MLOps pipeline for automated model management, monitoring, and retraining.

---

## 📦 Modules Overview

### 1. Model Version Manager (`model_version_manager.py`)

**Quản lý versioning và deployment cho ML models**

**Features:**
- ✅ Model registry with metadata
- ✅ Semantic versioning (v1.2.3)
- ✅ Auto-promotion for better models
- ✅ Model comparison
- ✅ Rollback capability
- ✅ Performance tracking

**Usage:**
```python
from src.ml.model_version_manager import get_model_version_manager

manager = get_model_version_manager()

# Register model
model_id = manager.register_model(
    model=trained_model,
    model_type="xgboost",
    version="1.2.0",
    train_metrics={'accuracy': 0.75, 'auc': 0.82},
    val_metrics={'accuracy': 0.72, 'auc': 0.80},
    feature_names=['rsi', 'macd', ...],
    training_period="2023-01-01 to 2024-01-01",
    auto_activate=True
)

# Load active model
model = manager.load_model()

# Compare models
comparison = manager.compare_models()
```

---

### 2. Feature Drift Detector (`feature_drift_detector.py`)

**Phát hiện distribution drift trong features**

**Features:**
- ✅ PSI (Population Stability Index)
- ✅ KS test (Kolmogorov-Smirnov)
- ✅ Baseline tracking
- ✅ Per-feature drift analysis
- ✅ Retraining recommendations

**Drift Thresholds:**
- PSI < 0.1: No drift
- PSI 0.1-0.2: Low drift (monitor)
- PSI 0.2-0.25: Medium drift (investigate)
- PSI > 0.25: High drift (retrain!)
- PSI > 0.5: Critical drift (model broken)

**Usage:**
```python
from src.ml.feature_drift_detector import get_drift_detector

detector = get_drift_detector()

# Set baseline from training data
detector.set_baseline(train_features_df, feature_names)

# Detect drift in production
drift_report = detector.detect_drift(current_features_df)

if drift_report.requires_retraining:
    print(f"🚨 DRIFT! PSI: {drift_report.overall_drift_score:.3f}")
    print(f"   {drift_report.features_with_drift} features drifted")
    # Trigger retraining
```

---

### 3. Model Explainer (`model_explainer.py`)

**SHAP-based model explainability**

**Features:**
- ✅ SHAP integration
- ✅ Feature contribution analysis
- ✅ Top positive/negative features
- ✅ Global feature importance
- ✅ Individual prediction explanations

**Usage:**
```python
from src.ml.model_explainer import ModelExplainer

explainer = ModelExplainer(
    model=trained_model,
    model_type="xgboost",
    feature_names=feature_list
)

# Explain prediction
explanation = explainer.explain_prediction(
    features=stock_features,
    prediction="BUY",
    confidence=75
)

print(explainer.format_explanation(explanation))
# Output:
# ✅ TOP POSITIVE CONTRIBUTORS:
#    • rsi_14: +0.152
#    • macd_histogram: +0.089
```

**Requirements:**
```bash
pip install shap
```

---

### 4. A/B Testing Framework (`ab_testing.py`)

**Statistical comparison of ML vs Technical signals**

**Features:**
- ✅ ML vs Technical comparison
- ✅ Hash-based allocation (deterministic)
- ✅ Statistical significance testing
- ✅ Wilson confidence intervals
- ✅ Winner selection
- ✅ Automated recommendations

**Usage:**
```python
from src.ml.ab_testing import get_ab_testing_framework, ABTestConfig, SignalSource

framework = get_ab_testing_framework()

# Start test
config = ABTestConfig(
    test_id="test_001",
    name="ML vs Technical Q1 2024",
    ml_allocation=0.50,
    technical_allocation=0.50,
    primary_metric="win_rate"
)
framework.start_test(config)

# Allocate symbols
for symbol in symbols:
    variant = framework.allocate_signal(symbol)

    if variant == SignalSource.ML_MODEL:
        signal = ml_generator.analyze(df)
    else:
        signal = technical_generator.analyze(df)

    framework.track_signal(variant, symbol, signal, confidence)

# Track results
framework.track_trade_result(
    variant=SignalSource.ML_MODEL,
    symbol="VNM",
    is_win=True,
    profit_pct=5.2,
    holding_days=12
)

# Get summary
summary = framework.get_summary()
print(summary.recommendation)
```

---

### 5. Automated Retraining Pipeline (`retraining_pipeline.py`) ⭐

**End-to-end automated model retraining**

**Features:**
- ✅ 5 trigger types (Drift, Performance, Scheduled, Manual, New Data)
- ✅ Full ML pipeline (Data → Train → Evaluate → Deploy)
- ✅ Multi-model training (XGBoost, LightGBM, RF)
- ✅ Auto-deployment (if improvement >= threshold)
- ✅ Model comparison
- ✅ Rollback support

**Triggers:**
1. **Drift**: PSI > 0.25
2. **Performance**: Win rate < 60%
3. **Scheduled**: Every 30 days
4. **Manual**: User-triggered
5. **New Data**: Significant new data available

**Usage:**
```python
from src.ml.retraining_pipeline import get_retraining_pipeline, RetrainingTrigger

pipeline = get_retraining_pipeline()

# Check triggers
should_retrain, trigger, reason = pipeline.check_triggers(
    current_performance={'win_rate': 0.55},
    current_features=recent_features_df
)

if should_retrain:
    # Auto-retrain
    result = pipeline.run_retraining(trigger, reason)

    if result.training_successful:
        print(f"✅ New model: {result.new_model_id}")
        print(f"Val accuracy: {result.val_accuracy:.1%}")
        print(f"Improvement: {result.improvement_pct:+.1f}%")
        print(f"Deployed: {result.deployed}")
```

---

### 6. Retraining Scheduler (`retraining_scheduler.py`) ⭐

**Background monitoring and automated retraining**

**Features:**
- ✅ Background scheduler (check every 6 hours)
- ✅ Automatic drift + performance monitoring
- ✅ Telegram notifications
- ✅ Manual trigger
- ✅ Status dashboard
- ✅ Emergency stop

**Usage:**
```python
from src.ml.retraining_scheduler import get_retraining_scheduler

scheduler = get_retraining_scheduler()

# Start background scheduler
scheduler.start()
# Will check every 6 hours and auto-retrain if needed

# Manual trigger
result = scheduler.manual_trigger(reason="Found issue")

# Get status
status = scheduler.get_status()
print(f"Running: {status['is_running']}")
print(f"Last check: {status['last_check_time']}")
print(f"Total retrainings: {status['total_retrainings']}")

# Stop scheduler
scheduler.stop()
```

---

## 🔄 Complete ML Ops Workflow

```
1. MONITORING (Continuous)
   ├─ Feature Drift Detection
   ├─ Performance Monitoring
   └─ Scheduled Checks (every 6h)
          ↓
2. TRIGGER DETECTION
   ├─ Drift > 0.25? → RETRAIN
   ├─ Win rate < 60%? → RETRAIN
   ├─ 30 days passed? → RETRAIN
   └─ Manual trigger? → RETRAIN
          ↓
3. DATA PREPARATION
   ├─ Fetch recent data
   ├─ Feature engineering
   ├─ Train/Val/Test split
   └─ Set new baseline
          ↓
4. MODEL TRAINING
   ├─ Train XGBoost
   ├─ Train LightGBM
   ├─ Train RandomForest
   └─ Select best model
          ↓
5. EVALUATION
   ├─ Calculate metrics
   ├─ Compare with current
   └─ Decide deployment
          ↓
6. DEPLOYMENT
   ├─ Register model
   ├─ Auto-activate if better
   └─ Log to registry
          ↓
7. NOTIFICATION
   ├─ Telegram alert
   ├─ Update dashboard
   └─ Log history
```

---

## 📊 Module Dependencies

```
model_version_manager.py (Core)
         ↑
         |
feature_drift_detector.py
         ↑
         |
retraining_pipeline.py
         ↑
         |
retraining_scheduler.py (Orchestrator)
         ↑
         |
    Production
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install xgboost lightgbm scikit-learn shap scipy schedule
```

### 2. Initialize

```python
# Start retraining scheduler in production
from src.ml.retraining_scheduler import get_retraining_scheduler

scheduler = get_retraining_scheduler()
scheduler.start()

# Scheduler will now:
# - Check drift every 6 hours
# - Monitor performance
# - Auto-retrain when needed
# - Send Telegram alerts
```

### 3. Monitor

```python
# Check status anytime
status = scheduler.get_status()

# Manual trigger if needed
result = scheduler.manual_trigger(reason="Emergency retraining")

# View model history
from src.ml.model_version_manager import get_model_version_manager
manager = get_model_version_manager()
comparison = manager.compare_models()
```

---

## 📝 Best Practices

### 1. Feature Drift Monitoring
- Set baseline from clean training data
- Check drift weekly
- Investigate PSI > 0.2
- Retrain when PSI > 0.25

### 2. Model Versioning
- Use semantic versioning (major.minor.patch)
- Increment minor for retraining
- Increment major for architecture changes
- Tag models with metadata

### 3. A/B Testing
- Run for minimum 30 trades per variant
- Use 50/50 allocation for fairness
- Wait for statistical significance (p < 0.05)
- Document learnings

### 4. Retraining
- Schedule monthly retraining
- Monitor performance continuously
- Auto-deploy only if improvement >= 3%
- Keep previous model for rollback

### 5. Monitoring
- Set up Telegram notifications
- Track retraining history
- Monitor drift trends
- Alert on failures

---

## 🐛 Troubleshooting

### Issue: Retraining fails with "Insufficient data"
**Solution:** Check `min_training_samples` in RetrainingConfig. Default is 1000 samples.

### Issue: Drift always detected
**Solution:** Baseline may be stale. Re-set baseline with recent data.

### Issue: Scheduler not triggering
**Solution:** Check scheduler status. Ensure it's running with `scheduler.get_status()`.

### Issue: Model not deploying
**Solution:** Check improvement threshold. Default is 3%. New model must be 3% better to deploy.

### Issue: SHAP errors
**Solution:** Install SHAP: `pip install shap`. Ensure model type is supported (XGBoost, LightGBM, RF).

---

## 📚 References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [PSI Calculation](https://en.wikipedia.org/wiki/Population_stability_index)
- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [MLOps Best Practices](https://ml-ops.org/)

---

## 🎯 Performance

| Module | Latency | Memory | Notes |
|--------|---------|--------|-------|
| Version Manager | <1ms | Low | In-memory registry |
| Drift Detector | ~100ms | Medium | PSI calculation |
| Model Explainer | ~500ms | High | SHAP values |
| A/B Testing | <1ms | Low | Hash-based |
| Retraining | ~10min | High | Full training |
| Scheduler | Background | Low | Thread-safe |

---

## ✅ Checklist

Before production deployment:

- [ ] Set baseline for drift detection
- [ ] Configure retraining thresholds
- [ ] Set up Telegram notifications
- [ ] Test manual retraining
- [ ] Verify model registry working
- [ ] Start background scheduler
- [ ] Monitor for 1 week
- [ ] Review first auto-retraining
- [ ] Document model versions
- [ ] Set up alerting

---

**Status**: ✅ Production-ready

**Last Updated**: 2024-01-26

**Maintainer**: ML Ops Team
