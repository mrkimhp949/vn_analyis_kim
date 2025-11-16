"""
Unit tests for ML Pipeline Model Trainer
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Import the classes to test
from ml_pipeline.model_trainer import TrainingConfig, EnsembleTrainer


class TestTrainingConfig:
    """Test TrainingConfig dataclass"""

    def test_training_config_creation(self):
        """Test creating TrainingConfig with required parameters"""
        feature_columns = ["feature1", "feature2", "feature3"]
        config = TrainingConfig(feature_columns=feature_columns)

        assert config.feature_columns == feature_columns
        assert config.target_column == "target"  # default value
        assert config.save_dir == "models"  # default value
        assert config.n_splits == 5  # default value

    def test_training_config_custom_values(self):
        """Test creating TrainingConfig with custom values"""
        feature_columns = ["rsi", "macd", "volume"]
        config = TrainingConfig(
            feature_columns=feature_columns,
            target_column="signal",
            save_dir="custom_models",
            n_splits=3,
        )

        assert config.feature_columns == feature_columns
        assert config.target_column == "signal"
        assert config.save_dir == "custom_models"
        assert config.n_splits == 3


class TestEnsembleTrainer:
    """Test EnsembleTrainer class"""

    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return TrainingConfig(
            feature_columns=["feature1", "feature2", "feature3"],
            target_column="target",
            save_dir="test_models",
            n_splits=3,
        )

    @pytest.fixture
    def trainer(self, config):
        """Create trainer instance with mocked os.makedirs"""
        with patch("ml_pipeline.model_trainer.os.makedirs"):
            trainer = EnsembleTrainer(config)
            return trainer

    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 100

        data = {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }

        return pd.DataFrame(data)

    def test_trainer_initialization(self, trainer, config):
        """Test trainer initialization"""
        assert trainer.config == config
        assert trainer.scaler is not None
        assert trainer.lstm_model is None
        assert trainer.lstm_sequence_length is None
        assert trainer.xgb_model is None

    def test_split_method(self, trainer, sample_data):
        """Test _split method for time series cross validation"""
        with patch("ml_pipeline.model_trainer.TimeSeriesSplit") as mock_tscv:
            # Mock the split method
            mock_split_instance = Mock()
            mock_split_instance.split.return_value = [
                (np.array([0, 1, 2]), np.array([3, 4])),
                (np.array([0, 1, 2, 3, 4]), np.array([5, 6])),
            ]
            mock_tscv.return_value = mock_split_instance

            splits = trainer._split(sample_data)

            assert len(splits) == 2
            assert all(len(split) == 4 for split in splits)  # X_train, X_test, y_train, y_test

    def test_prepare_lstm_data_sufficient_data(self, trainer):
        """Test _prepare_lstm_data with sufficient data"""
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, 20)
        sequence_length = 5

        X_seq, y_seq = trainer._prepare_lstm_data(X, y, sequence_length)

        assert X_seq is not None
        assert y_seq is not None
        assert X_seq.shape == (15, 5, 3)  # (20-5, sequence_length, features)
        assert y_seq.shape == (15,)

    def test_prepare_lstm_data_insufficient_data(self, trainer):
        """Test _prepare_lstm_data with insufficient data"""
        X = np.random.randn(5, 3)
        y = np.random.randint(0, 2, 5)
        sequence_length = 10

        X_seq, y_seq = trainer._prepare_lstm_data(X, y, sequence_length)

        assert X_seq is None
        assert y_seq is None

    def test_build_lstm_success(self, trainer):
        """Test successful LSTM model building"""
        import ml_pipeline.model_trainer as mt

        # Create mock layers
        mock_lstm_layer_1 = Mock()
        mock_dropout_layer_1 = Mock()
        mock_lstm_layer_2 = Mock()
        mock_dropout_layer_2 = Mock()
        mock_dense_layer_1 = Mock()
        mock_dense_layer_2 = Mock()

        # Create mock model
        mock_model = Mock()
        mock_model.compile = Mock()

        # Mock the layer constructors to return our mock layers
        mock_lstm = Mock(side_effect=[mock_lstm_layer_1, mock_lstm_layer_2])
        mock_dropout = Mock(side_effect=[mock_dropout_layer_1, mock_dropout_layer_2])
        mock_dense = Mock(side_effect=[mock_dense_layer_1, mock_dense_layer_2])
        mock_sequential = Mock(return_value=mock_model)

        # Temporarily set the mocked classes on the module
        old_lstm_available = getattr(mt, "LSTM_AVAILABLE", False)
        old_sequential = getattr(mt, "Sequential", None)
        old_lstm = getattr(mt, "LSTM", None)
        old_dropout = getattr(mt, "Dropout", None)
        old_dense = getattr(mt, "Dense", None)

        try:
            mt.LSTM_AVAILABLE = True
            mt.Sequential = mock_sequential
            mt.LSTM = mock_lstm
            mt.Dropout = mock_dropout
            mt.Dense = mock_dense

            result = trainer._build_lstm((10, 3))

            # Verify the model was created and compiled
            assert result == mock_model
            mock_sequential.assert_called_once()

            # Verify the layers were created
            assert mock_lstm.call_count == 2
            assert mock_dropout.call_count == 2
            assert mock_dense.call_count == 2

            # Verify compile was called
            mock_model.compile.assert_called_once_with(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
        finally:
            # Restore original values
            mt.LSTM_AVAILABLE = old_lstm_available
            if old_sequential is not None:
                mt.Sequential = old_sequential
            elif hasattr(mt, "Sequential"):
                delattr(mt, "Sequential")
            if old_lstm is not None:
                mt.LSTM = old_lstm
            elif hasattr(mt, "LSTM"):
                delattr(mt, "LSTM")
            if old_dropout is not None:
                mt.Dropout = old_dropout
            elif hasattr(mt, "Dropout"):
                delattr(mt, "Dropout")
            if old_dense is not None:
                mt.Dense = old_dense
            elif hasattr(mt, "Dense"):
                delattr(mt, "Dense")

    @patch("ml_pipeline.model_trainer.LSTM_AVAILABLE", False)
    def test_build_lstm_not_available(self, trainer):
        """Test LSTM building when LSTM is not available"""
        result = trainer._build_lstm((10, 3))
        assert result is None

    def test_train_empty_dataframe(self, trainer):
        """Test training with empty dataframe"""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Training dataframe is empty"):
            trainer.train(empty_df)

    def test_train_missing_features(self, trainer):
        """Test training with missing required features"""
        df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})

        with pytest.raises(ValueError, match="Missing required features"):
            trainer.train(df)

    def test_train_missing_target(self, trainer, config):
        """Test training with missing target column"""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]})

        with pytest.raises(ValueError, match="Target column 'target' is missing"):
            trainer.train(df)

    @patch("ml_pipeline.model_trainer.joblib.dump")
    @patch("ml_pipeline.model_trainer.TimeSeriesSplit")
    @patch("ml_pipeline.model_trainer.RandomForestClassifier")
    @patch("ml_pipeline.model_trainer.GradientBoostingClassifier")
    @patch("ml_pipeline.model_trainer.StandardScaler")
    @patch("ml_pipeline.model_trainer.XGBOOST_AVAILABLE", False)
    @patch("ml_pipeline.model_trainer.LSTM_AVAILABLE", False)
    def test_train_basic_models_only(
        self,
        mock_scaler,
        mock_gb,
        mock_rf,
        mock_tscv,
        mock_joblib,
        trainer,
        sample_data,
    ):
        """Test training with only basic models (RF + GB)"""
        # Mock TimeSeriesSplit
        mock_split_instance = Mock()
        mock_split_instance.split.return_value = [
            (np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9]))
        ]
        mock_tscv.return_value = mock_split_instance

        # Mock StandardScaler
        mock_scaler_instance = Mock()
        mock_scaler_instance.fit_transform.return_value = np.random.randn(5, 3)
        mock_scaler_instance.transform.return_value = np.random.randn(5, 3)
        mock_scaler.return_value = mock_scaler_instance

        # Mock RandomForest
        mock_rf_instance = Mock()
        mock_rf_instance.predict_proba.return_value = np.array(
            [[0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9]]
        )
        mock_rf.return_value = mock_rf_instance

        # Mock GradientBoosting
        mock_gb_instance = Mock()
        mock_gb_instance.predict_proba.return_value = np.array(
            [[0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.4, 0.6], [0.0, 1.0]]
        )
        mock_gb.return_value = mock_gb_instance

        result = trainer.train(sample_data)

        # Verify models were created and trained
        mock_rf.assert_called()
        mock_gb.assert_called()
        mock_rf_instance.fit.assert_called()
        mock_gb_instance.fit.assert_called()

        # Verify result structure
        assert isinstance(result, dict)

    @patch("ml_pipeline.model_trainer.joblib.dump")
    @patch("ml_pipeline.model_trainer.TimeSeriesSplit")
    @patch("ml_pipeline.model_trainer.RandomForestClassifier")
    @patch("ml_pipeline.model_trainer.GradientBoostingClassifier")
    @patch("ml_pipeline.model_trainer.StandardScaler")
    def test_train_with_market_regime(
        self,
        mock_scaler,
        mock_gb,
        mock_rf,
        mock_tscv,
        mock_joblib,
        trainer,
        sample_data,
    ):
        """Test training with market regime parameter"""
        # Setup mocks similar to previous test
        mock_split_instance = Mock()
        mock_split_instance.split.return_value = [
            (np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9]))
        ]
        mock_tscv.return_value = mock_split_instance

        mock_scaler_instance = Mock()
        mock_scaler_instance.fit_transform.return_value = np.random.randn(5, 3)
        mock_scaler_instance.transform.return_value = np.random.randn(5, 3)
        mock_scaler.return_value = mock_scaler_instance

        mock_rf_instance = Mock()
        mock_rf_instance.predict_proba.return_value = np.array(
            [[0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9]]
        )
        mock_rf.return_value = mock_rf_instance

        mock_gb_instance = Mock()
        mock_gb_instance.predict_proba.return_value = np.array(
            [[0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.4, 0.6], [0.0, 1.0]]
        )
        mock_gb.return_value = mock_gb_instance

        result = trainer.train(sample_data, market_regime="BULLISH")

        assert isinstance(result, dict)

    def test_train_logs_info_messages(self, trainer, sample_data, caplog):
        """Test that training logs appropriate info messages"""
        with patch("ml_pipeline.model_trainer.TimeSeriesSplit") as mock_tscv, patch(
            "ml_pipeline.model_trainer.RandomForestClassifier"
        ), patch("ml_pipeline.model_trainer.GradientBoostingClassifier"), patch(
            "ml_pipeline.model_trainer.StandardScaler"
        ):
            # Mock minimal setup
            mock_split_instance = Mock()
            mock_split_instance.split.return_value = [(np.array([0, 1, 2]), np.array([3, 4]))]
            mock_tscv.return_value = mock_split_instance

            with caplog.at_level("INFO"):
                try:
                    trainer.train(sample_data)
                except:
                    pass  # We're just testing logging, not full execution

            # Check that info messages were logged
            assert any("Training ensemble" in record.message for record in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
