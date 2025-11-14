# 📁 Models Directory

This directory contains trained ML models for the trading bot.

## 📊 Model Files

### Enhanced Models (New)
- `rf_enhanced.pkl` - Random Forest model
- `xgb_enhanced.pkl` - XGBoost model
- `lgb_enhanced.pkl` - LightGBM model
- `scaler_enhanced.pkl` - Feature scaler
- `feature_importance.pkl` - Feature importance scores
- `model_info_enhanced.json` - Model metadata

### Legacy Models (Old)
- `random_forest.pkl` - Old Random Forest model
- `scaler.pkl` - Old scaler
- `model_info.json` - Old metadata

## 🚀 Training Models

### Quick Training
```bash
python train_enhanced_models.py
```

### With Hyperparameter Tuning
```bash
python train_enhanced_models.py --tune
```

### Custom Parameters
```bash
python train_enhanced_models.py --max-symbols 100 --lookback 1000
```

## 📊 Model Performance

### Enhanced Models (Expected)
- **Accuracy:** 58-62%
- **Precision:** 56-60%
- **Recall:** 54-58%
- **F1-Score:** 55-59%
- **ROC-AUC:** 0.58-0.63

### Legacy Models
- **Accuracy:** 52-55%
- **Precision:** 50-53%
- **Recall:** 48-52%
- **F1-Score:** 49-52%
- **ROC-AUC:** 0.52-0.55

## 🔄 Retraining

Models should be retrained regularly:
- **Frequency:** Weekly or monthly
- **Reason:** Market conditions change
- **Command:** `python train_enhanced_models.py`

## 📁 File Sizes (Approximate)

- `rf_enhanced.pkl`: 50-100 MB
- `xgb_enhanced.pkl`: 10-20 MB
- `lgb_enhanced.pkl`: 5-10 MB
- `scaler_enhanced.pkl`: < 1 MB
- `feature_importance.pkl`: < 1 MB

## ⚠️ Important Notes

1. **Don't commit large model files to git**
   - Add to `.gitignore`
   - Store separately if needed

2. **Backup models before retraining**
   - Copy to `models_backup/`
   - Keep old versions

3. **Validate after training**
   - Run `python test_ml_improvements.py`
   - Check accuracy metrics

4. **Monitor model drift**
   - Track accuracy over time
   - Retrain if performance drops

## 🔍 Model Info

Check model metadata:
```python
import json

with open('models/model_info_enhanced.json', 'r') as f:
    info = json.load(f)
    print(info)
```

Output:
```json
{
  "expected_features": 28,
  "feature_names": ["sma20", "ema20", ...],
  "models_available": {
    "rf": true,
    "xgb": true,
    "lgb": true
  },
  "saved_at": "2024-01-15T10:30:00"
}
```

## 📚 More Info

- **Training Guide:** `../train_enhanced_models.py`
- **Documentation:** `../ML_IMPROVEMENTS.md`
- **Quick Start:** `../QUICKSTART_ML.md`
