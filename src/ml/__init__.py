"""
ML Package - Machine Learning for Vietnam Stock Trading

Modules:
- vietnam_ml_integration: Complete ML integration for Vietnam market (10/10)
- integration_bridge: Bridge between ML and entry logic
- signals/: ML signal generators (V1, V2, V3)
- models/: ML model implementations (ensemble, LSTM)
- monitor: Model performance monitoring
- retraining_pipeline: Automated retraining
"""

# Vietnam ML Integration (10/10) - Primary interface
try:
    from src.ml.vietnam_ml_integration import (
        VietnamMLIntegration,
        VietnamMarketSession,
        SignalQuality,
        VietnamMLFeatures,
        MLIntegrationResult,
        ConfidenceCalibrator,
        VietnamFeatureGenerator,
        MLPredictionTracker,
        get_vietnam_ml_integration,
        reset_vietnam_ml_integration,
    )

    VIETNAM_ML_AVAILABLE = True
except ImportError:
    VIETNAM_ML_AVAILABLE = False

# Integration Bridge
try:
    from src.ml.integration_bridge import (
        MLIntegrationBridge,
        get_ml_integration_bridge,
        reset_ml_integration_bridge,
        enhance_entry_signal_with_ml,
    )

    INTEGRATION_BRIDGE_AVAILABLE = True
except ImportError:
    INTEGRATION_BRIDGE_AVAILABLE = False

# Export list
__all__ = [
    # Availability flags
    "VIETNAM_ML_AVAILABLE",
    "INTEGRATION_BRIDGE_AVAILABLE",
]

if VIETNAM_ML_AVAILABLE:
    __all__.extend(
        [
            "VietnamMLIntegration",
            "VietnamMarketSession",
            "SignalQuality",
            "VietnamMLFeatures",
            "MLIntegrationResult",
            "ConfidenceCalibrator",
            "VietnamFeatureGenerator",
            "MLPredictionTracker",
            "get_vietnam_ml_integration",
            "reset_vietnam_ml_integration",
        ]
    )

if INTEGRATION_BRIDGE_AVAILABLE:
    __all__.extend(
        [
            "MLIntegrationBridge",
            "get_ml_integration_bridge",
            "reset_ml_integration_bridge",
            "enhance_entry_signal_with_ml",
        ]
    )
