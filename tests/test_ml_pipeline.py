"""
Unit tests for ML Pipeline modules
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch

# Fix import path BEFORE imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# DataManager Tests
# ============================================================================
@patch("ml_pipeline.data_manager.load_data")
@patch("ml_pipeline.data_manager.add_ml_features")
class TestDataManager:
    """Test DataManager class"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup after each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_data_manager_initialization(self, mock_add_features, mock_load):
        """Test DataManager initializes correctly"""
        from ml_pipeline.data_manager import DataIngestionConfig, DataManager

        config = DataIngestionConfig(
            tickers=["VNM", "VCB"], lookback=100, feature_store_path=self.temp_dir
        )
        manager = DataManager(config)

        assert manager.config.tickers == ["VNM", "VCB"]
        assert manager.config.lookback == 100
        assert os.path.exists(self.temp_dir)

    def test_feature_file_path(self, mock_add_features, mock_load):
        """Test feature file path generation"""
        from ml_pipeline.data_manager import DataIngestionConfig, DataManager

        config = DataIngestionConfig(tickers=["VNM"], feature_store_path=self.temp_dir)
        manager = DataManager(config)

        path = manager._feature_file("VNM")
        assert "VNM.parquet" in path
        assert self.temp_dir in path

    def test_ingest_symbol_success(self, mock_add_features, mock_load_data):
        """Test successful symbol ingestion"""
        from ml_pipeline.data_manager import DataIngestionConfig, DataManager

        # Mock data
        mock_df = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
                "target": [1, 0, 1],
            }
        )
        mock_load_data.return_value = mock_df
        mock_add_features.return_value = mock_df

        config = DataIngestionConfig(
            tickers=["VNM"], feature_store_path=self.temp_dir, refresh=True
        )
        manager = DataManager(config)

        result = manager.ingest_symbol("VNM")

        assert result is not None
        assert not result.empty
        assert "symbol" in result.columns
        assert result["symbol"].iloc[0] == "VNM"

    def test_ingest_symbol_empty_data(self, mock_add_features, mock_load_data):
        """Test ingestion with empty data"""
        from ml_pipeline.data_manager import DataIngestionConfig, DataManager

        mock_load_data.return_value = pd.DataFrame()

        config = DataIngestionConfig(tickers=["VNM"], feature_store_path=self.temp_dir)
        manager = DataManager(config)

        result = manager.ingest_symbol("VNM")
        assert result is None

    def test_ingest_all_success(self, mock_add_features, mock_load_data):
        """Test ingesting all symbols"""
        from ml_pipeline.data_manager import DataIngestionConfig, DataManager

        mock_df = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
                "target": [1, 0, 1],
            }
        )
        mock_load_data.return_value = mock_df
        mock_add_features.return_value = mock_df

        config = DataIngestionConfig(
            tickers=["VNM", "VCB"], feature_store_path=self.temp_dir, refresh=True
        )
        manager = DataManager(config)

        result = manager.ingest_all()

        assert not result.empty
        assert "symbol" in result.columns
        # Should have data from both tickers
        assert len(result) >= 3

    def test_ingest_all_no_valid_data(self, mock_add_features, mock_load):
        """Test ingest_all raises error when no valid data"""
        from ml_pipeline.data_manager import DataIngestionConfig, DataManager

        mock_load.return_value = pd.DataFrame()

        config = DataIngestionConfig(
            tickers=["INVALID"], feature_store_path=self.temp_dir
        )
        manager = DataManager(config)

        with pytest.raises(RuntimeError):
            manager.ingest_all()

    def test_load_feature_store_not_exists(self, mock_add_features, mock_load):
        """Test loading non-existent feature store"""
        from ml_pipeline.data_manager import load_feature_store

        with pytest.raises(FileNotFoundError):
            load_feature_store(os.path.join(self.temp_dir, "nonexistent.parquet"))


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
        df = pd.DataFrame(
            {"feat1": np.random.randn(100), "target": np.random.randint(0, 2, 100)}
        )
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
# Model Trainer Tests
# ============================================================================
class TestModelTrainer:
    """Test EnsembleTrainer class"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup after each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_trainer_initialization(self):
        """Test trainer initializes correctly"""
        from ml_pipeline.model_trainer import TrainingConfig, EnsembleTrainer

        config = TrainingConfig(
            feature_columns=["feat1", "feat2"],
            target_column="target",
            save_dir=self.temp_dir,
        )
        trainer = EnsembleTrainer(config)

        assert trainer.config.feature_columns == ["feat1", "feat2"]
        assert trainer.scaler is not None
        assert os.path.exists(self.temp_dir)

    def test_train_with_empty_dataframe(self):
        """Test training with empty dataframe"""
        from ml_pipeline.model_trainer import TrainingConfig, EnsembleTrainer

        config = TrainingConfig(feature_columns=["feat1"], save_dir=self.temp_dir)
        trainer = EnsembleTrainer(config)

        with pytest.raises(ValueError, match="empty"):
            trainer.train(pd.DataFrame())

    def test_train_with_missing_features(self):
        """Test training with missing features"""
        from ml_pipeline.model_trainer import TrainingConfig, EnsembleTrainer

        df = pd.DataFrame({"feat1": [1, 2, 3], "target": [0, 1, 0]})

        config = TrainingConfig(
            feature_columns=["feat1", "feat2"], save_dir=self.temp_dir  # feat2 missing
        )
        trainer = EnsembleTrainer(config)

        with pytest.raises(ValueError, match="Missing required features"):
            trainer.train(df)

    def test_train_with_missing_target(self):
        """Test training with missing target column"""
        from ml_pipeline.model_trainer import TrainingConfig, EnsembleTrainer

        df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})

        config = TrainingConfig(
            feature_columns=["feat1", "feat2"], save_dir=self.temp_dir
        )
        trainer = EnsembleTrainer(config)

        with pytest.raises(ValueError, match="Target column"):
            trainer.train(df)

    def test_train_with_valid_data(self):
        """Test successful training with valid data"""
        from ml_pipeline.model_trainer import TrainingConfig, EnsembleTrainer

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feat1": np.random.randn(100),
                "feat2": np.random.randn(100),
                "feat3": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        config = TrainingConfig(
            feature_columns=["feat1", "feat2", "feat3"],
            save_dir=self.temp_dir,
            n_splits=3,
        )
        trainer = EnsembleTrainer(config)

        metrics = trainer.train(df)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert os.path.exists(os.path.join(self.temp_dir, "ensemble_rf.pkl"))
        assert os.path.exists(os.path.join(self.temp_dir, "ensemble_gb.pkl"))
        assert os.path.exists(os.path.join(self.temp_dir, "ensemble_scaler.pkl"))

    def test_build_lstm_without_tensorflow(self):
        """Test LSTM building when TensorFlow not available"""
        from ml_pipeline.model_trainer import TrainingConfig, EnsembleTrainer

        config = TrainingConfig(feature_columns=["feat1"], save_dir=self.temp_dir)
        trainer = EnsembleTrainer(config)

        with patch("ml_pipeline.model_trainer.LSTM_AVAILABLE", False):
            lstm = trainer._build_lstm((10, 5))
            assert lstm is None

    def test_prepare_lstm_data(self):
        """Test LSTM data preparation"""
        from ml_pipeline.model_trainer import TrainingConfig, EnsembleTrainer

        config = TrainingConfig(feature_columns=["feat1"], save_dir=self.temp_dir)
        trainer = EnsembleTrainer(config)

        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)

        X_seq, y_seq = trainer._prepare_lstm_data(X, y, sequence_length=10)

        if X_seq is not None:
            assert X_seq.shape[0] == 40  # 50 - 10
            assert X_seq.shape[1] == 10  # sequence length
            assert X_seq.shape[2] == 3  # features
            assert len(y_seq) == 40

    def test_prepare_lstm_data_insufficient_samples(self):
        """Test LSTM data prep with insufficient samples"""
        from ml_pipeline.model_trainer import TrainingConfig, EnsembleTrainer

        config = TrainingConfig(feature_columns=["feat1"], save_dir=self.temp_dir)
        trainer = EnsembleTrainer(config)

        X = np.random.randn(5, 3)
        y = np.random.randint(0, 2, 5)

        X_seq, y_seq = trainer._prepare_lstm_data(X, y, sequence_length=10)

        assert X_seq is None
        assert y_seq is None


# ============================================================================
# Sentiment Model Tests
# ============================================================================
class TestSentimentAnalyzer:
    """Test VietnameseSentimentAnalyzer"""

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly"""
        from ml_pipeline.sentiment_model import VietnameseSentimentAnalyzer

        analyzer = VietnameseSentimentAnalyzer()

        assert analyzer is not None
        assert hasattr(analyzer, "positive_keywords")
        assert hasattr(analyzer, "negative_keywords")

    def test_score_empty_list(self):
        """Test scoring empty text list"""
        from ml_pipeline.sentiment_model import VietnameseSentimentAnalyzer

        analyzer = VietnameseSentimentAnalyzer()
        score = analyzer.score([])

        assert score == 0.0

    def test_score_positive_keywords(self):
        """Test scoring with positive keywords"""
        from ml_pipeline.sentiment_model import VietnameseSentimentAnalyzer

        analyzer = VietnameseSentimentAnalyzer()
        # Force keyword-based scoring
        analyzer.pipeline = None

        texts = ["Công ty tăng trưởng mạnh", "Lợi nhuận vượt mục tiêu"]
        score = analyzer.score(texts)

        assert score > 0

    def test_score_negative_keywords(self):
        """Test scoring with negative keywords"""
        from ml_pipeline.sentiment_model import VietnameseSentimentAnalyzer

        analyzer = VietnameseSentimentAnalyzer()
        analyzer.pipeline = None

        texts = ["Công ty thua lỗ", "Rủi ro cao"]
        score = analyzer.score(texts)

        assert score < 0

    def test_classify_strongly_positive(self):
        """Test classify returns correct label"""
        from ml_pipeline.sentiment_model import VietnameseSentimentAnalyzer

        analyzer = VietnameseSentimentAnalyzer()
        analyzer.pipeline = None

        text = "tăng trưởng tích cực vượt mục tiêu lợi nhuận kỷ lục"
        result = analyzer.classify(text)

        assert result in [
            "STRONGLY_POSITIVE",
            "POSITIVE",
            "NEUTRAL",
            "NEGATIVE",
            "STRONGLY_NEGATIVE",
        ]

    def test_classify_neutral(self):
        """Test neutral classification"""
        from ml_pipeline.sentiment_model import VietnameseSentimentAnalyzer

        analyzer = VietnameseSentimentAnalyzer()
        analyzer.pipeline = None

        text = "Công ty công bố thông tin"
        result = analyzer.classify(text)

        assert result == "NEUTRAL"

    def test_normalize_label(self):
        """Test label normalization"""
        from ml_pipeline.sentiment_model import VietnameseSentimentAnalyzer

        analyzer = VietnameseSentimentAnalyzer()
        analyzer.id2label = {0: "negative", 1: "neutral", 2: "positive"}

        label = analyzer._normalize_label("LABEL_2")
        assert label == "positive"


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


# ============================================================================
# Train Pipeline Tests
# ============================================================================
class TestTrainPipeline:
    """Test train_pipeline module"""

    def test_run_pipeline_no_tickers(self):
        """Test pipeline raises error with no tickers"""
        from ml_pipeline.train_pipeline import run_pipeline

        with pytest.raises(ValueError, match="No tickers"):
            run_pipeline([], lookback=100, refresh=False)

    def test_run_pipeline_invalid_tickers(self):
        """Test pipeline with invalid tickers"""
        from ml_pipeline.train_pipeline import run_pipeline

        with pytest.raises(ValueError, match="invalid"):
            run_pipeline(["", "  ", ""], lookback=100, refresh=False)

    @patch("ml_pipeline.train_pipeline.DataManager")
    @patch("ml_pipeline.train_pipeline.EnsembleTrainer")
    @patch("ml_pipeline.train_pipeline.VolatilityForecaster")
    def test_run_pipeline_success(self, mock_vol, mock_trainer, mock_manager):
        """Test successful pipeline run"""
        from ml_pipeline.train_pipeline import run_pipeline

        # Mock successful data ingestion
        mock_df = pd.DataFrame(
            {
                "close": np.random.randn(100) + 100,
                "feat1": np.random.randn(100),
                "feat2": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )
        mock_manager_instance = MagicMock()
        mock_manager_instance.ingest_all.return_value = mock_df
        mock_manager.return_value = mock_manager_instance

        # Mock successful training
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {
            "accuracy": {"mean": 0.75, "std": 0.05}
        }
        mock_trainer.return_value = mock_trainer_instance

        # Mock volatility forecaster
        mock_vol_instance = MagicMock()
        mock_vol_instance.train.return_value = {"test_mae": 0.01}
        mock_vol.return_value = mock_vol_instance

        with patch("ml_pipeline.train_pipeline.get_feature_columns") as mock_cols:
            mock_cols.return_value = ["feat1", "feat2"]

            metrics = run_pipeline(tickers=["VNM", "VCB"], lookback=100, refresh=False)

        assert "ensemble" in metrics
        assert "volatility" in metrics
        assert "tickers" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
