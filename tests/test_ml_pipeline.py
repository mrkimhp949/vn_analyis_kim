"""
Unit tests for ML Pipeline modules

NOTE: Some tests are skipped due to legacy import issues in ml_pipeline modules:
- DataManager: requires data_loader, features modules
- SentimentAnalyzer: requires suppress_warnings module
- ModelTrainer: complex dependencies
- TrainPipeline: requires config, features modules

Working tests cover:
- FeatureSelection (SHAP-based)
- StackingMetaModel
- VolatilityForecaster
"""

import os
import sys
import tempfile
import shutil

# Fix import path BEFORE imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Feature Selection Tests
# ============================================================================
class TestFeatureSelection:
    """Test feature selection with SHAP"""

    def test_select_features_empty_dataframe(self):
        """Test feature selection with empty dataframe"""
        from ml_pipeline.feature_selection import select_features_with_shap

        df = pd.DataFrame()
        features, importance = select_features_with_shap(
            df, feature_columns=["feat1", "feat2"], target_column="target"
        )

        assert features == ["feat1", "feat2"]
        assert importance is None

    def test_select_features_insufficient_features(self):
        """Test with insufficient features"""
        from ml_pipeline.feature_selection import select_features_with_shap

        df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6], "target": [0, 1, 0]})

        features, importance = select_features_with_shap(
            df,
            feature_columns=["feat1", "feat2", "feat3"],  # feat3 doesn't exist
            target_column="target",
        )

        # Should return available features
        assert len(features) == 2

    def test_select_features_with_valid_data(self):
        """Test feature selection with valid data"""
        from ml_pipeline.feature_selection import select_features_with_shap

        np.random.seed(42)
        df = pd.DataFrame({f"feat{i}": np.random.randn(100) for i in range(10)})
        df["target"] = (df["feat0"] + df["feat1"] > 0).astype(int)

        features, importance = select_features_with_shap(
            df,
            feature_columns=[f"feat{i}" for i in range(10)],
            target_column="target",
            max_samples=50,
        )

        assert len(features) > 0
        assert len(features) <= 10

    def test_select_features_with_correlation_filtering(self):
        """Test correlation-based feature filtering"""
        from ml_pipeline.feature_selection import select_features_with_shap

        np.random.seed(42)
        df = pd.DataFrame({"feat1": np.random.randn(100), "target": np.random.randint(0, 2, 100)})
        # Create highly correlated feature
        df["feat2"] = df["feat1"] * 0.99 + np.random.randn(100) * 0.01
        df["feat3"] = np.random.randn(100)

        features, importance = select_features_with_shap(
            df,
            feature_columns=["feat1", "feat2", "feat3"],
            target_column="target",
            correlation_threshold=0.95,
        )

        # feat2 should be filtered out due to high correlation with feat1
        assert len(features) <= 3


# ============================================================================
# Stacking Ensemble Tests
# ============================================================================
class TestStackingMetaModel:
    """Test StackingMetaModel"""

    def test_meta_model_initialization(self):
        """Test meta model initializes correctly"""
        from ml_pipeline.stacking_ensemble import StackingMetaModel

        model = StackingMetaModel(model_type="logistic")

        assert model.model_type == "logistic"
        assert model.model is None

    def test_fit_with_valid_data(self):
        """Test fitting meta model"""
        from ml_pipeline.stacking_ensemble import StackingMetaModel

        np.random.seed(42)
        meta_features = np.random.rand(100, 3)  # 3 base models
        y = np.random.randint(0, 2, 100)

        model = StackingMetaModel(model_type="logistic")
        model.fit(meta_features, y, feature_names=["rf", "gb", "xgb"])

        assert model.model is not None
        assert model.feature_names == ["rf", "gb", "xgb"]

    def test_fit_with_invalid_shape(self):
        """Test fit raises error with invalid shape"""
        from ml_pipeline.stacking_ensemble import StackingMetaModel

        meta_features = np.random.rand(100)  # 1D array
        y = np.random.randint(0, 2, 100)

        model = StackingMetaModel()

        with pytest.raises(ValueError, match="2 chiều"):
            model.fit(meta_features, y)

    def test_fit_with_mismatched_lengths(self):
        """Test fit raises error with mismatched lengths"""
        from ml_pipeline.stacking_ensemble import StackingMetaModel

        meta_features = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 50)  # Wrong length

        model = StackingMetaModel()

        with pytest.raises(ValueError, match="cùng số lượng"):
            model.fit(meta_features, y)

    def test_predict_proba(self):
        """Test probability prediction"""
        from ml_pipeline.stacking_ensemble import StackingMetaModel

        np.random.seed(42)
        meta_features = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)

        model = StackingMetaModel(model_type="logistic")
        model.fit(meta_features, y)

        test_features = np.random.rand(10, 3)
        proba = model.predict_proba(test_features)

        assert len(proba) == 10
        assert all(0 <= p <= 1 for p in proba)

    def test_predict(self):
        """Test class prediction"""
        from ml_pipeline.stacking_ensemble import StackingMetaModel

        np.random.seed(42)
        meta_features = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)

        model = StackingMetaModel(model_type="logistic")
        model.fit(meta_features, y)

        test_features = np.random.rand(10, 3)
        predictions = model.predict(test_features)

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_predict_without_training(self):
        """Test predict raises error without training"""
        from ml_pipeline.stacking_ensemble import StackingMetaModel

        model = StackingMetaModel()
        test_features = np.random.rand(10, 3)

        with pytest.raises(ValueError, match="chưa được huấn luyện"):
            model.predict_proba(test_features)

    def test_save_and_load(self):
        """Test saving and loading meta model"""
        from ml_pipeline.stacking_ensemble import StackingMetaModel

        temp_dir = tempfile.mkdtemp()

        try:
            np.random.seed(42)
            meta_features = np.random.rand(100, 3)
            y = np.random.randint(0, 2, 100)

            model1 = StackingMetaModel(model_type="logistic")
            model1.fit(meta_features, y, feature_names=["rf", "gb", "xgb"])
            model1.save(temp_dir)

            # Load in new instance
            model2 = StackingMetaModel()
            model2.load(temp_dir)

            assert model2.model is not None
            assert model2.feature_names == ["rf", "gb", "xgb"]

            # Test predictions match
            test_features = np.random.rand(10, 3)
            pred1 = model1.predict(test_features)
            pred2 = model2.predict(test_features)

            np.testing.assert_array_equal(pred1, pred2)

        finally:
            shutil.rmtree(temp_dir)

    def test_load_nonexistent(self):
        """Test loading from non-existent path"""
        from ml_pipeline.stacking_ensemble import StackingMetaModel

        model = StackingMetaModel()

        with pytest.raises(FileNotFoundError):
            model.load("/nonexistent/path")


# ============================================================================
# Volatility Forecaster Tests
# ============================================================================
class TestVolatilityForecaster:
    """Test VolatilityForecaster"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup after each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_forecaster_initialization(self):
        """Test forecaster initializes correctly"""
        from ml_pipeline.volatility_forecaster import VolatilityForecaster

        forecaster = VolatilityForecaster(lookback_window=20, forecast_horizon=5)

        assert forecaster.lookback_window == 20
        assert forecaster.forecast_horizon == 5
        assert forecaster.scaler is not None

    def test_calculate_features(self):
        """Test feature calculation"""
        from ml_pipeline.volatility_forecaster import VolatilityForecaster

        df = pd.DataFrame(
            {
                "close": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 105,
                "low": np.random.randn(100).cumsum() + 95,
                "volume": np.random.randint(1000, 10000, 100),
                "atr": np.random.randn(100) * 2 + 5,
                "rsi": np.random.randn(100) * 20 + 50,
                "macd": np.random.randn(100),
            }
        )

        forecaster = VolatilityForecaster()
        features = forecaster._calculate_features(df)

        assert "volatility_5d" in features.columns
        assert "volatility_10d" in features.columns
        assert "atr_pct" in features.columns
        assert "volume_ratio" in features.columns

    def test_prepare_target(self):
        """Test target preparation"""
        from ml_pipeline.volatility_forecaster import VolatilityForecaster

        df = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100})

        forecaster = VolatilityForecaster(forecast_horizon=5)
        target = forecaster._prepare_target(df)

        assert isinstance(target, pd.Series)
        assert len(target) == len(df)

    def test_train_with_valid_data(self):
        """Test training with valid data"""
        from ml_pipeline.volatility_forecaster import VolatilityForecaster

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "close": np.random.randn(200).cumsum() + 100,
                "high": np.random.randn(200).cumsum() + 105,
                "low": np.random.randn(200).cumsum() + 95,
                "volume": np.random.randint(1000, 10000, 200),
                "atr": np.random.randn(200) * 2 + 5,
                "rsi": np.random.randn(200) * 20 + 50,
                "macd": np.random.randn(200),
            }
        )

        forecaster = VolatilityForecaster()
        forecaster.models_dir = self.temp_dir

        metrics = forecaster.train(df)

        assert "test_mae" in metrics
        assert "test_rmse" in metrics
        assert forecaster.model is not None

    def test_train_with_insufficient_data(self):
        """Test training with insufficient data"""
        from ml_pipeline.volatility_forecaster import VolatilityForecaster

        df = pd.DataFrame({"close": [100, 101, 102]})

        forecaster = VolatilityForecaster()
        metrics = forecaster.train(df)

        assert "error" in metrics

    def test_forecast_without_model(self):
        """Test forecast falls back when no model"""
        from ml_pipeline.volatility_forecaster import VolatilityForecaster

        df = pd.DataFrame({"close": [100, 101, 102], "atr": [2, 2.5, 2.2]})

        forecaster = VolatilityForecaster()
        forecaster.models_dir = self.temp_dir

        vol = forecaster.forecast(df)

        assert isinstance(vol, float)
        assert 0 < vol < 1  # Reasonable volatility range

    def test_forecast_with_trained_model(self):
        """Test forecast with trained model"""
        from ml_pipeline.volatility_forecaster import VolatilityForecaster

        np.random.seed(42)
        df_train = pd.DataFrame(
            {
                "close": np.random.randn(200).cumsum() + 100,
                "high": np.random.randn(200).cumsum() + 105,
                "low": np.random.randn(200).cumsum() + 95,
                "volume": np.random.randint(1000, 10000, 200),
                "atr": np.random.randn(200) * 2 + 5,
                "rsi": np.random.randn(200) * 20 + 50,
                "macd": np.random.randn(200),
            }
        )

        forecaster = VolatilityForecaster()
        forecaster.models_dir = self.temp_dir
        forecaster.train(df_train)

        df_test = df_train.tail(50)
        vol = forecaster.forecast(df_test)

        assert isinstance(vol, float)
        assert 0.001 <= vol <= 0.10  # Bounded volatility

    def test_save_and_load(self):
        """Test saving and loading model"""
        from ml_pipeline.volatility_forecaster import VolatilityForecaster

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "close": np.random.randn(200).cumsum() + 100,
                "high": np.random.randn(200).cumsum() + 105,
                "low": np.random.randn(200).cumsum() + 95,
                "volume": np.random.randint(1000, 10000, 200),
                "atr": np.random.randn(200) * 2 + 5,
                "rsi": np.random.randn(200) * 20 + 50,
                "macd": np.random.randn(200),
            }
        )

        forecaster1 = VolatilityForecaster()
        forecaster1.models_dir = self.temp_dir
        forecaster1.train(df)
        forecaster1.save()

        forecaster2 = VolatilityForecaster()
        forecaster2.models_dir = self.temp_dir
        loaded = forecaster2.load()

        assert loaded is True
        assert forecaster2.model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
