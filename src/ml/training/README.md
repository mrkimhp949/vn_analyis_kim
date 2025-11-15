# ML Training Scripts

This directory contains training scripts for different ML model types.

## Files

### `basic_trainer.py`
- **Purpose**: Train basic ML models
- **Models**: Random Forest + LSTM
- **Features**: Basic technical indicators
- **Output**: `random_forest.pkl`, `lstm_model.h5`, `scaler.pkl`
- **Usage**: `python -m src.ml.training.basic_trainer`

### `advanced_trainer.py`
- **Purpose**: Train advanced ML models with enhanced features
- **Models**: Random Forest, XGBoost, LightGBM ensemble
- **Features**: Enhanced technical + market regime + volatility features
- **Output**: `rf_enhanced.pkl`, `xgb_enhanced.pkl`, `lgb_enhanced.pkl`, `scaler_enhanced.pkl`
- **Advanced Features**:
  - Hyperparameter tuning
  - Feature importance analysis
  - SHAP explanations
  - Cross-validation
  - Ensemble methods
- **Usage**: `python -m src.ml.training.advanced_trainer`

## Recommendations

- **For beginners**: Start with `basic_trainer.py`
- **For production**: Use `advanced_trainer.py` with `--tune` flag
- **For quick testing**: Use `basic_trainer.py` (faster training)
- **For best performance**: Use `advanced_trainer.py` (better accuracy)

## Example Commands

```bash
# Basic training
python -m src.ml.training.basic_trainer

# Advanced training
python -m src.ml.training.advanced_trainer

# Advanced training with hyperparameter tuning
python -m src.ml.training.advanced_trainer --tune

# Advanced training with specific symbols
python -m src.ml.training.advanced_trainer --symbols "VNM,VCB,FPT" --max-symbols 10
```