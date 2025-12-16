# 🤖 ML Pipeline Documentation

## Tổng quan

ML Pipeline của Vietnam Stock Trading Bot bao gồm các thành phần chính:

1. **Model Registry** - Quản lý và versioning models
2. **Training Pipeline** - Automated training và retraining
3. **Feature Selection** - SHAP-based feature selection
4. **Ensemble Models** - Stacking và voting ensemble
5. **Volatility Forecaster** - Dự báo biến động
6. **Sentiment Model** - Phân tích sentiment tin tức

## 📁 Cấu trúc thư mục

```
ml_pipeline/
├── __init__.py
├── model_registry.py      # Model versioning & management
├── training_pipeline.py   # Automated training
├── feature_selection.py   # SHAP-based selection
├── stacking_ensemble.py   # Meta-model stacking
├── volatility_forecaster.py
├── sentiment_model.py
└── data_manager.py

models/
├── registry.json          # Registry metadata
├── rf_v2.pkl             # Random Forest
├── xgb_v2.pkl            # XGBoost
├── lgb_v2.pkl            # LightGBM
├── ensemble_v2.pkl       # Ensemble model
├── stacking_v2.pkl       # Stacking meta-model
├── scaler_v2.pkl         # Feature scaler
├── model_info_v2.json    # Model metadata
├── feature_importance_v2.csv
├── backups/              # Model backups
└── staging/              # Staging models
```

## 🚀 Quick Start

### 1. Training mới model

```python
from ml_pipeline.training_pipeline import MLTrainingPipeline, TrainingConfig
from ml_pipeline.model_registry import get_registry

# Khởi tạo pipeline
pipeline = MLTrainingPipeline()

# Cấu hình training
config = TrainingConfig(
    model_type="random_forest",
    feature_columns=["rsi", "macd", "bb_pct", "vol_ratio", ...],
    target_column="target",
    n_cv_splits=5,
)

# Load training data
df = load_training_data()  # Your data loading function

# Train model
result = pipeline.train_model(df, config)

if result.success:
    print(f"Model version: {result.model_version}")
    print(f"ROC-AUC: {result.metrics.roc_auc:.4f}")
```

### 2. Load production model

```python
from ml_pipeline.model_registry import get_registry, ModelStage

registry = get_registry()

# Load production model
model, model_version = registry.load_model(
    model_type="random_forest",
    stage=ModelStage.PRODUCTION
)

# Predict
predictions = model.predict(X_new)
```

### 3. Retrain production model

```python
from ml_pipeline.training_pipeline import MLTrainingPipeline

pipeline = MLTrainingPipeline()

# Retrain với data mới
result = pipeline.retrain_production_model(
    df=new_data,
    model_type="random_forest",
    promote_if_better=True,
    min_improvement=0.01  # Improve 1% ROC-AUC
)
```

## 📊 Model Registry

### Concepts

- **Model Type**: Loại model (random_forest, xgboost, lightgbm, ...)
- **Version**: Semantic versioning (v1.0.0, v1.0.1, v1.1.0, v2.0.0)
- **Stage**: Lifecycle stage (development, staging, production, archived)
- **Metrics**: Performance metrics (accuracy, precision, recall, f1, roc_auc)

### Model Lifecycle

```
development → staging → production → archived
     ↓                       ↓
  delete                  rollback
```

### API Reference

```python
from ml_pipeline.model_registry import (
    ModelRegistry,
    ModelMetrics,
    ModelStage,
    ModelType,
    get_registry
)

registry = get_registry()

# Register model
version = registry.register_model(
    model=trained_model,
    model_type="random_forest",
    metrics=ModelMetrics(accuracy=0.65, roc_auc=0.72),
    version_bump="minor",  # v1.0.0 → v1.1.0
    description="Improved with new features",
    feature_columns=[...],
    tags=["vietnam", "vn30"]
)

# Promote to production
registry.promote_model(
    model_type="random_forest",
    version="v1.1.0",
    to_stage=ModelStage.PRODUCTION
)

# Rollback
registry.rollback_model(
    model_type="random_forest",
    to_version="v1.0.0"
)

# Compare versions
comparison = registry.compare_models(
    model_type="random_forest",
    versions=["v1.0.0", "v1.1.0", "v1.2.0"]
)

# List models
models = registry.list_models(
    model_type="random_forest",
    stage=ModelStage.PRODUCTION
)

# Get stats
stats = registry.get_registry_stats()
```

## 🔄 Training Pipeline

### Features

1. **Data Validation** - Kiểm tra data trước training
2. **Preprocessing** - Scaling, missing values handling
3. **Cross-Validation** - TimeSeriesSplit
4. **Hyperparameter Tuning** - Grid search (optional)
5. **Auto Registration** - Tự động đăng ký vào registry

### Training Config

```python
@dataclass
class TrainingConfig:
    model_type: str           # "random_forest", "xgboost", ...
    feature_columns: List[str]
    target_column: str = "target"
    test_size: float = 0.2
    n_cv_splits: int = 5
    tune_hyperparameters: bool = False
    random_state: int = 42
    min_samples: int = 1000
    min_positive_ratio: float = 0.2
    max_positive_ratio: float = 0.8
```

### Train Ensemble

```python
results = pipeline.train_ensemble(
    df=training_data,
    config=base_config,
    model_types=["random_forest", "xgboost", "lightgbm"],
)

for model_type, result in results.items():
    if result.success:
        print(f"{model_type}: ROC-AUC = {result.metrics.roc_auc:.4f}")
```

## 📈 Feature Selection

### SHAP-based Selection

```python
from ml_pipeline.feature_selection import select_features_with_shap

selected_features, importance_df = select_features_with_shap(
    df=training_data,
    feature_columns=all_features,
    target_column="target",
    top_k=30,  # Chọn top 30 features
    correlation_threshold=0.92,  # Loại features trùng lặp
)
```

### Feature Importance

File `models/feature_importance_v2.csv` chứa SHAP importance của mỗi feature:

| Feature | SHAP Importance |
|---------|-----------------|
| rsi | 0.0823 |
| macd_hist | 0.0756 |
| bb_pct | 0.0698 |
| vol_ratio | 0.0645 |
| ... | ... |

## 🤖 Ensemble Models

### Stacking Meta-Model

```python
from ml_pipeline.stacking_ensemble import StackingMetaModel

# Create stacking model
stacking = StackingMetaModel(model_type="lightgbm")

# Train với predictions từ base models
# meta_features shape: [n_samples, n_base_models]
stacking.fit(meta_features, y, feature_names=["rf", "xgb", "lgb"])

# Predict
final_predictions = stacking.predict_proba(new_meta_features)
```

### Voting Ensemble

Trong `src/signal/ensemble.py`:

```python
from src.signal.ensemble import EnsemblePredictor

predictor = EnsemblePredictor()
predictions = predictor.predict(df)
```

## 📉 Volatility Forecaster

```python
from ml_pipeline.volatility_forecaster import VolatilityForecaster

forecaster = VolatilityForecaster()
forecaster.fit(price_data)

# Predict volatility
volatility = forecaster.predict(recent_data)
```

## 💬 Sentiment Model

```python
from ml_pipeline.sentiment_model import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Analyze news
sentiment = analyzer.analyze("Cổ phiếu VNM tăng mạnh nhờ kết quả kinh doanh tốt")
# Returns: {"label": "positive", "score": 0.85}
```

## ⏰ Scheduled Retraining

### Setup Retraining Schedule

```python
from ml_pipeline.training_pipeline import RetrainingScheduler

scheduler = RetrainingScheduler(pipeline)

# Retrain mỗi 7 ngày
scheduler.add_schedule(
    model_type="random_forest",
    interval_days=7,
    min_samples_for_retrain=1000
)

# Check và retrain
result = scheduler.check_and_retrain("random_forest", new_data)
```

### Cron Job Example

```bash
# Retrain mỗi Chủ Nhật lúc 2:00 AM
0 2 * * 0 cd /path/to/project && python scripts/retrain_models.py
```

## 🔍 Model Monitoring

### Metrics to Track

1. **Prediction Drift** - Distribution thay đổi của predictions
2. **Feature Drift** - Distribution thay đổi của features
3. **Performance Drift** - Accuracy/ROC-AUC giảm theo thời gian

### Alerts

```python
# Trong src/monitoring/prometheus_metrics.py

# Alert khi ROC-AUC < 0.55
ml_model_roc_auc < 0.55

# Alert khi prediction confidence thấp
avg(ml_prediction_confidence) < 0.6
```

## 🧪 Testing Models

### Unit Tests

```bash
pytest tests/test_ml_pipeline.py -v
```

### Backtesting

```python
from backtesting.engine import BacktestEngine

engine = BacktestEngine()
results = engine.run(
    symbols=["VNM", "VIC", "VHM"],
    start_date="2024-01-01",
    end_date="2024-12-01"
)
```

### Walk-Forward Validation

```python
from backtesting.walk_forward import WalkForwardOptimizer

optimizer = WalkForwardOptimizer()
results = optimizer.run(
    data=historical_data,
    train_size=252,  # 1 năm
    test_size=63,    # 1 quý
)
```

## 📋 Best Practices

### 1. Data Quality

- ✅ Loại bỏ missing values trước training
- ✅ Handle outliers (clip extreme values)
- ✅ Feature scaling (StandardScaler)
- ✅ Check class balance

### 2. Model Selection

- ✅ Sử dụng TimeSeriesSplit cho cross-validation
- ✅ Compare multiple models trước khi chọn
- ✅ Ensemble thường tốt hơn single model

### 3. Versioning

- ✅ Bump major version khi thay đổi features
- ✅ Bump minor version khi retrain với data mới
- ✅ Bump patch version khi fix bugs

### 4. Monitoring

- ✅ Track performance metrics hàng ngày
- ✅ Set alerts cho performance drift
- ✅ Log predictions cho analysis

## ❓ Troubleshooting

### Model không load được

```python
# Check registry
registry = get_registry()
print(registry.list_models())

# Check production models
print(registry.get_production_models())
```

### Performance giảm

1. Check feature drift
2. Retrain với data mới
3. Rollback nếu model mới kém hơn

### Memory issues

```python
# Sử dụng sampling cho training
df_sample = df.sample(n=10000, random_state=42)
```

## 📚 References

- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [MLflow](https://mlflow.org/) (Future integration)
