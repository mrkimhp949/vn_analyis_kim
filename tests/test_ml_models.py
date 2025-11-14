"""
Unit tests for ML Models
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models import MLPredictor
from exceptions import ModelPredictionError


class TestMLPredictor:
    """Test MLPredictor class"""

    def setup_method(self):
        """Setup for each test"""
        self.predictor = MLPredictor()

    def test_predictor_initialization(self):
        """Test predictor initializes correctly"""
        assert self.predictor.expected_features == 18
        assert self.predictor.models_dir == "models"
        assert self.predictor.scaler is not None

    def test_create_dummy_models(self):
        """Test dummy model creation"""
        self.predictor.create_dummy_models()

        assert self.predictor.rf_model is not None
        assert hasattr(self.predictor.scaler, "mean_")
        assert len(self.predictor.scaler.mean_) == 18

    def test_predict_with_correct_features(self):
        """Test prediction with correct number of features"""
        self.predictor.create_dummy_models()

        # Create test data with 18 features
        X_test = np.random.randn(10, 18)

        predictions = self.predictor.predict(X_test)

        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions)

    def test_predict_with_wrong_features(self):
        """Test prediction fails with wrong number of features"""
        self.predictor.create_dummy_models()

        # Create test data with wrong number of features
        X_test = np.random.randn(10, 15)  # Wrong: 15 instead of 18

        with pytest.raises(ValueError) as exc_info:
            self.predictor.predict(X_test)

        assert "Feature mismatch" in str(exc_info.value)
        assert "15" in str(exc_info.value)
        assert "18" in str(exc_info.value)

    def test_predict_with_dataframe(self):
        """Test prediction with pandas DataFrame"""
        self.predictor.create_dummy_models()

        # Create DataFrame with 18 features
        df = pd.DataFrame(np.random.randn(10, 18))

        predictions = self.predictor.predict(df)

        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions)

    def test_predict_empty_input(self):
        """Test prediction with empty input"""
        self.predictor.create_dummy_models()

        X_test = np.array([])

        predictions = self.predictor.predict(X_test)

        assert len(predictions) == 0

    def test_train_with_correct_features(self):
        """Test training with correct number of features"""
        X_train = np.random.randn(100, 18)
        y_train = np.random.randint(0, 2, 100)

        # Should not raise exception
        self.predictor.train_random_forest(X_train, y_train)

        assert self.predictor.rf_model is not None

    def test_train_with_wrong_features(self):
        """Test training fails with wrong number of features"""
        X_train = np.random.randn(100, 15)  # Wrong: 15 instead of 18
        y_train = np.random.randint(0, 2, 100)

        with pytest.raises(ModelPredictionError) as exc_info:
            self.predictor.train_random_forest(X_train, y_train)

        assert "Feature count mismatch" in str(exc_info.value)

    def test_load_models_creates_dummy_if_not_exists(self):
        """Test load_models creates dummy models if files don't exist"""
        # Ensure models directory exists but is empty
        import shutil

        if os.path.exists("models"):
            # Backup existing models
            if os.path.exists("models_backup"):
                shutil.rmtree("models_backup")
            shutil.copytree("models", "models_backup")
            shutil.rmtree("models")

        try:
            predictor = MLPredictor()
            loaded = predictor.load_models()

            assert loaded is True
            assert predictor.rf_model is not None

        finally:
            # Restore original models
            if os.path.exists("models_backup"):
                if os.path.exists("models"):
                    shutil.rmtree("models")
                shutil.copytree("models_backup", "models")
                shutil.rmtree("models_backup")

    def test_save_and_load_models(self):
        """Test saving and loading models"""
        import tempfile
        import shutil

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Create and save models
            predictor1 = MLPredictor()
            predictor1.models_dir = temp_dir
            predictor1.create_dummy_models()
            predictor1.save_models()

            # Load models in new predictor
            predictor2 = MLPredictor()
            predictor2.models_dir = temp_dir
            loaded = predictor2.load_models()

            assert loaded is True
            assert predictor2.rf_model is not None
            assert hasattr(predictor2.scaler, "mean_")

            # Test predictions are consistent
            X_test = np.random.randn(5, 18)
            pred1 = predictor1.predict(X_test)
            pred2 = predictor2.predict(X_test)

            np.testing.assert_array_almost_equal(pred1, pred2)

        finally:
            shutil.rmtree(temp_dir)

    def test_model_info_saved_with_metadata(self):
        """Test model info file contains correct metadata"""
        import tempfile
        import shutil
        import json

        temp_dir = tempfile.mkdtemp()

        try:
            predictor = MLPredictor()
            predictor.models_dir = temp_dir
            predictor.create_dummy_models()
            predictor.save_models()

            # Check model_info.json exists and has correct data
            info_path = os.path.join(temp_dir, "model_info.json")
            assert os.path.exists(info_path)

            with open(info_path, "r") as f:
                info = json.load(f)

            assert "expected_features" in info
            assert info["expected_features"] == 18
            assert "saved_at" in info

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
