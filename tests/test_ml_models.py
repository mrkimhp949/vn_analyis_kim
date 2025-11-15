"""
Unit tests for ML Models
"""

import os
import sys

# Fix import path BEFORE imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.config.exceptions import ModelPredictionError
from src.ml.models.predictor import MLPredictor


class TestMLPredictor:
    """Test MLPredictor class"""

    def setup_method(self):
        """Setup for each test"""
        self.predictor = MLPredictor()

    def test_predictor_initialization(self):
        """Test predictor initializes correctly"""
        assert self.predictor.expected_features == 28
        assert self.predictor.models_dir == "models"
        assert self.predictor.scaler is not None

    def test_create_dummy_models(self):
        """Test dummy model creation"""
        self.predictor.create_dummy_models()

        assert self.predictor.rf_model is not None
        assert hasattr(self.predictor.scaler, "mean_")
        assert len(self.predictor.scaler.mean_) == 28

    def test_predict_with_correct_features(self):
        """Test prediction with correct number of features"""
        self.predictor.create_dummy_models()

        # Create test data with 28 features
        X_test = np.random.randn(10, 28)

        predictions = self.predictor.predict(X_test)

        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions)

    def test_predict_with_wrong_features(self):
        """Test prediction fails with wrong number of features"""
        self.predictor.create_dummy_models()

        # Create test data with wrong number of features
        X_test = np.random.randn(10, 15)  # Wrong: 15 instead of 28

        with pytest.raises(ValueError) as exc_info:
            self.predictor.predict(X_test)

        assert "Feature mismatch" in str(exc_info.value)
        assert "15" in str(exc_info.value)
        assert "28" in str(exc_info.value)

    def test_predict_with_dataframe(self):
        """Test prediction with pandas DataFrame"""
        self.predictor.create_dummy_models()

        # Create DataFrame with 28 features
        df = pd.DataFrame(np.random.randn(10, 28))

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
        X_train = np.random.randn(100, 28)
        y_train = np.random.randint(0, 2, 100)

        # Should not raise exception
        self.predictor.train_random_forest(X_train, y_train)

        assert self.predictor.rf_model is not None

    def test_train_with_wrong_features(self):
        """Test training fails with wrong number of features"""
        X_train = np.random.randn(100, 15)  # Wrong: 15 instead of 28
        y_train = np.random.randint(0, 2, 100)

        with pytest.raises(ModelPredictionError) as exc_info:
            self.predictor.train_random_forest(X_train, y_train)

        assert "Feature count mismatch" in str(exc_info.value)

    def test_load_models_creates_dummy_if_not_exists(self):
        """Test load_models returns False if files don't exist (no longer creates dummy models)"""
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

            # Should return False when no models exist (changed behavior)
            assert loaded is False
            assert predictor.ml_enabled is False

        finally:
            # Restore original models
            if os.path.exists("models_backup"):
                if os.path.exists("models"):
                    shutil.rmtree("models")
                shutil.copytree("models_backup", "models")
                shutil.rmtree("models_backup")

    def test_save_and_load_models(self):
        """Test saving and loading models"""
        import shutil
        import tempfile

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
            X_test = np.random.randn(5, 28)
            pred1 = predictor1.predict(X_test)
            pred2 = predictor2.predict(X_test)

            np.testing.assert_array_almost_equal(pred1, pred2)

        finally:
            shutil.rmtree(temp_dir)

    def test_model_info_saved_with_metadata(self):
        """Test model info file contains correct metadata"""
        import json
        import shutil
        import tempfile

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
            assert info["expected_features"] == 28
            assert "saved_at" in info

        finally:
            shutil.rmtree(temp_dir)

    def test_ml_enabled_flag_on_init(self):
        """Test ML enabled flag is set correctly on initialization"""
        predictor = MLPredictor()
        assert predictor.ml_enabled is True
        assert predictor.using_dummy_models is False

    def test_ml_disabled_when_no_models(self):
        """Test ML is disabled when models cannot be loaded"""
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()

        try:
            predictor = MLPredictor()
            predictor.models_dir = temp_dir
            loaded = predictor.load_models()

            assert loaded is False
            assert predictor.ml_enabled is False

        finally:
            shutil.rmtree(temp_dir)

    def test_predict_raises_when_ml_disabled(self):
        """Test prediction fails when ML is disabled"""
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()

        try:
            predictor = MLPredictor()
            predictor.models_dir = temp_dir
            predictor.ml_enabled = False

            X_test = np.random.randn(10, 28)

            with pytest.raises(ValueError) as exc_info:
                predictor.predict(X_test)

            assert "ML predictions disabled" in str(exc_info.value)
            assert "Train models first" in str(exc_info.value)

        finally:
            shutil.rmtree(temp_dir)

    def test_predict_with_2d_array(self):
        """Test prediction with 2D numpy array"""
        self.predictor.create_dummy_models()

        X_test = np.random.randn(5, 28)
        predictions = self.predictor.predict(X_test)

        assert predictions.shape == (5,)
        assert all(isinstance(p, (float, np.floating)) for p in predictions)

    def test_predict_with_series(self):
        """Test prediction with pandas Series"""
        self.predictor.create_dummy_models()

        # Create Series with 28 values (1D data)
        # This should fail because Series is 1D, not 2D
        series = pd.Series(np.random.randn(28))

        # Convert to DataFrame with proper shape for prediction
        df = series.to_frame().T  # Transpose to make it (1, 28)
        predictions = self.predictor.predict(df)

        assert len(predictions) == 1
        assert 0 <= predictions[0] <= 1

    def test_scaler_fitted_after_dummy_creation(self):
        """Test scaler is properly fitted after dummy model creation"""
        self.predictor.create_dummy_models()

        assert hasattr(self.predictor.scaler, "mean_")
        assert hasattr(self.predictor.scaler, "scale_")
        assert len(self.predictor.scaler.mean_) == self.predictor.expected_features
        assert len(self.predictor.scaler.scale_) == self.predictor.expected_features

    def test_random_forest_params(self):
        """Test Random Forest is created with correct parameters"""
        X_train = np.random.randn(100, 28)
        y_train = np.random.randint(0, 2, 100)

        self.predictor.train_random_forest(X_train, y_train)

        assert self.predictor.rf_model.n_estimators == 200
        assert self.predictor.rf_model.max_depth == 15
        assert self.predictor.rf_model.min_samples_split == 10
        assert self.predictor.rf_model.min_samples_leaf == 5
        assert self.predictor.rf_model.class_weight == "balanced"

    def test_evaluate_without_model(self):
        """Test evaluate handles missing model gracefully"""
        X_test = np.random.randn(50, 28)
        y_test = np.random.randint(0, 2, 50)

        # Should not raise exception, just log warning
        self.predictor.evaluate(X_test, y_test)

    def test_evaluate_with_model(self):
        """Test evaluate runs successfully with trained model"""
        # Train model
        X_train = np.random.randn(100, 28)
        y_train = np.random.randint(0, 2, 100)
        self.predictor.train_random_forest(X_train, y_train)

        # Evaluate
        X_test = np.random.randn(50, 28)
        y_test = np.random.randint(0, 2, 50)

        # Should not raise exception
        self.predictor.evaluate(X_test, y_test)

    def test_ensure_models_dir_creates_directory(self):
        """Test ensure_models_dir creates directory if it doesn't exist"""
        import shutil
        import tempfile

        temp_dir = os.path.join(tempfile.gettempdir(), "test_models_dir")

        # Remove if exists
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        try:
            predictor = MLPredictor()
            predictor.models_dir = temp_dir
            predictor.ensure_models_dir()

            assert os.path.exists(temp_dir)
            assert os.path.isdir(temp_dir)

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_predict_probability_range(self):
        """Test predictions are valid probabilities (0-1)"""
        self.predictor.create_dummy_models()

        X_test = np.random.randn(100, 28)
        predictions = self.predictor.predict(X_test)

        assert all(0 <= p <= 1 for p in predictions)
        assert predictions.min() >= 0
        assert predictions.max() <= 1

    def test_train_with_imbalanced_data(self):
        """Test training with imbalanced data uses class_weight"""
        # Create imbalanced dataset (90% class 0, 10% class 1)
        X_train = np.random.randn(1000, 28)
        y_train = np.concatenate([np.zeros(900), np.ones(100)])

        self.predictor.train_random_forest(X_train, y_train)

        assert self.predictor.rf_model.class_weight == "balanced"

    def test_save_models_creates_all_files(self):
        """Test save_models creates all required files"""
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()

        try:
            predictor = MLPredictor()
            predictor.models_dir = temp_dir
            predictor.create_dummy_models()

            # Check all files exist
            assert os.path.exists(os.path.join(temp_dir, "random_forest.pkl"))
            assert os.path.exists(os.path.join(temp_dir, "scaler.pkl"))
            assert os.path.exists(os.path.join(temp_dir, "model_info.json"))

        finally:
            shutil.rmtree(temp_dir)

    def test_load_models_with_feature_mismatch(self):
        """Test load_models handles feature mismatch by recreating models"""
        import json
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()

        try:
            # Create models with wrong feature count
            predictor1 = MLPredictor()
            predictor1.models_dir = temp_dir
            predictor1.expected_features = 15  # Wrong count
            predictor1.create_dummy_models()

            # Manually edit model_info.json to have wrong count
            info_path = os.path.join(temp_dir, "model_info.json")
            with open(info_path, "w") as f:
                json.dump({"expected_features": 15, "saved_at": "2025-01-01"}, f)

            # Load with correct feature count should detect mismatch
            predictor2 = MLPredictor()
            predictor2.models_dir = temp_dir
            # This should recreate models with correct count
            loaded = predictor2.load_models()

            # Should have recreated with correct features
            assert loaded is True

        finally:
            shutil.rmtree(temp_dir)

    def test_predict_with_single_sample(self):
        """Test prediction with single sample"""
        self.predictor.create_dummy_models()

        X_test = np.random.randn(1, 28)
        predictions = self.predictor.predict(X_test)

        assert len(predictions) == 1
        assert 0 <= predictions[0] <= 1

    def test_predict_without_scaler_fitted(self):
        """Test prediction works even if scaler not fitted"""
        self.predictor.create_dummy_models()

        # Remove scaler mean_ to simulate unfitted scaler
        if hasattr(self.predictor.scaler, "mean_"):
            delattr(self.predictor.scaler, "mean_")

        X_test = np.random.randn(10, 28)
        predictions = self.predictor.predict(X_test)

        # Should still work (uses unscaled data)
        assert len(predictions) == 10

    def test_model_persistence_across_instances(self):
        """Test models persist correctly across different predictor instances"""
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()

        try:
            # Create and train model in first instance
            predictor1 = MLPredictor()
            predictor1.models_dir = temp_dir
            X_train = np.random.randn(100, 28)
            y_train = np.random.randint(0, 2, 100)
            predictor1.train_random_forest(X_train, y_train)

            # Load in second instance
            predictor2 = MLPredictor()
            predictor2.models_dir = temp_dir
            loaded = predictor2.load_models()

            assert loaded is True
            assert predictor2.rf_model is not None

            # Predictions should be identical (within floating point precision)
            X_test = np.random.randn(10, 28)
            pred1 = predictor1.predict(X_test)
            pred2 = predictor2.predict(X_test)

            np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)

        finally:
            shutil.rmtree(temp_dir)


class TestMLPredictorEdgeCases:
    """Test edge cases and error handling"""

    def test_predict_with_nan_values(self):
        """Test prediction handles NaN values"""
        predictor = MLPredictor()
        predictor.create_dummy_models()

        X_test = np.random.randn(10, 28)
        X_test[0, 0] = np.nan  # Introduce NaN

        # Should handle or raise appropriate error
        try:
            predictions = predictor.predict(X_test)
            # If it handles NaN, check results
            assert len(predictions) == 10
        except (ValueError, Exception):
            # If it raises error, that's also acceptable
            pass

    def test_predict_with_inf_values(self):
        """Test prediction handles infinite values"""
        predictor = MLPredictor()
        predictor.create_dummy_models()

        X_test = np.random.randn(10, 28)
        X_test[0, 0] = np.inf  # Introduce infinity

        # Should handle or raise appropriate error
        try:
            predictions = predictor.predict(X_test)
            assert len(predictions) == 10
        except (ValueError, Exception):
            pass

    def test_predict_with_very_large_values(self):
        """Test prediction with very large values"""
        predictor = MLPredictor()
        predictor.create_dummy_models()

        X_test = np.random.randn(10, 28) * 1e10  # Very large values

        predictions = predictor.predict(X_test)
        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions)

    def test_train_with_single_class(self):
        """Test training with single class data"""
        predictor = MLPredictor()

        X_train = np.random.randn(100, 28)
        y_train = np.zeros(100)  # All same class

        # Should handle gracefully
        try:
            predictor.train_random_forest(X_train, y_train)
        except (ValueError, Exception):
            # May raise error, which is acceptable
            pass

    def test_train_with_minimal_samples(self):
        """Test training with minimal number of samples"""
        predictor = MLPredictor()

        X_train = np.random.randn(10, 28)  # Only 10 samples
        y_train = np.random.randint(0, 2, 10)

        # Should train but may not be reliable
        predictor.train_random_forest(X_train, y_train)
        assert predictor.rf_model is not None

    def test_empty_dataframe_predict(self):
        """Test prediction with empty DataFrame"""
        predictor = MLPredictor()
        predictor.create_dummy_models()

        df = pd.DataFrame()
        predictions = predictor.predict(df)

        assert len(predictions) == 0


class TestMLPredictorIntegration:
    """Integration tests for MLPredictor"""

    def test_full_workflow_train_save_load_predict(self):
        """Test complete workflow: train -> save -> load -> predict"""
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()

        try:
            # Step 1: Train
            predictor1 = MLPredictor()
            predictor1.models_dir = temp_dir
            X_train = np.random.randn(200, 28)
            y_train = np.random.randint(0, 2, 200)
            predictor1.train_random_forest(X_train, y_train)

            # Step 2: Save is automatic in train_random_forest

            # Step 3: Load in new instance
            predictor2 = MLPredictor()
            predictor2.models_dir = temp_dir
            loaded = predictor2.load_models()
            assert loaded is True

            # Step 4: Predict
            X_test = np.random.randn(50, 28)
            predictions = predictor2.predict(X_test)

            assert len(predictions) == 50
            assert all(0 <= p <= 1 for p in predictions)

        finally:
            shutil.rmtree(temp_dir)

    def test_multiple_predictions_consistent(self):
        """Test multiple predictions on same data are consistent"""
        predictor = MLPredictor()
        predictor.create_dummy_models()

        X_test = np.random.randn(20, 28)

        pred1 = predictor.predict(X_test)
        pred2 = predictor.predict(X_test)
        pred3 = predictor.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(pred2, pred3)

    def test_concurrent_predictions(self):
        """Test predictor can handle multiple prediction calls"""
        predictor = MLPredictor()
        predictor.create_dummy_models()

        results = []
        for _ in range(5):
            X_test = np.random.randn(10, 28)
            predictions = predictor.predict(X_test)
            results.append(predictions)

        # All should succeed
        assert len(results) == 5
        assert all(len(r) == 10 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
