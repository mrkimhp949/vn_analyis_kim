"""
Unit tests for ML Pipeline Model Trainer

NOTE: These tests are currently SKIPPED because they reference outdated architecture.
The old ml_pipeline.model_trainer module (TrainingConfig, EnsembleTrainer)
has been replaced by src.ml.models.predictor (MLPredictor).

TODO: Rewrite tests for MLPredictor class
"""

import pytest

# Skip all tests in this file with explanation
pytestmark = pytest.mark.skip(
    reason="Outdated tests for ml_pipeline.model_trainer which no longer exists. "
    "Refactored to src.ml.models.predictor.MLPredictor. "
    "Tests need to be rewritten for new architecture."
)


class TestModelTrainerPlaceholder:
    """Placeholder for future tests"""

    def test_placeholder(self):
        """This test is skipped - see module docstring"""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
