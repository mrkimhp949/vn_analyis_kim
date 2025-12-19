"""
ML Package - Machine Learning for Vietnam Stock Trading

Modules:
- vietnam_ml_integration: Base ML integration for Vietnam market
- enhanced_ml_integration: Enhanced ML integration v5.0 (10/10 rating)
- integration_bridge: Bridge between ML and entry logic
- signals/: ML signal generators (V1, V2, V3)
- models/: ML model implementations (ensemble, LSTM)
- monitor: Model performance monitoring
- retraining_pipeline: Automated retraining
- model_explainer: SHAP-based prediction explainability
- feature_drift_detector: Feature distribution drift detection

VERSION: 5.0.0 - Enhanced ML Integration with:
- Walk-Forward Validation
- Performance Decay Detection
- Advanced Confidence Calibration (Platt/Isotonic/Temperature scaling)
- Risk-Adjusted Metrics (Sharpe, Sortino, Information Ratio)
- Regime-Specific Model Selection
- Prediction Explainability
- Real-time Performance Tracking
"""

# Vietnam ML Integration (Base) - Core interface
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

# Enhanced ML Integration v5.0 (10/10) - Full-featured interface
try:
    from src.ml.enhanced_ml_integration import (
        EnhancedMLIntegration,
        WalkForwardValidator,
        PerformanceDecayDetector,
        AdvancedConfidenceCalibrator,
        RiskAdjustedMetrics,
        RegimeModelSelector,
        ModelHealth,
        PerformanceMetrics,
        WalkForwardResult,
        DecayDetection,
        get_enhanced_ml_integration,
        reset_enhanced_ml_integration,
        ACCURACY_TARGET_MINIMUM,
        ACCURACY_TARGET_ACCEPTABLE,
        ACCURACY_TARGET_GOOD,
        ACCURACY_TARGET_EXCELLENT,
    )

    ENHANCED_ML_AVAILABLE = True
except ImportError:
    ENHANCED_ML_AVAILABLE = False

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

# Model Explainer
try:
    from src.ml.model_explainer import (
        ModelExplainer,
        PredictionExplanation,
        SHAP_AVAILABLE,
    )

    MODEL_EXPLAINER_AVAILABLE = True
except ImportError:
    MODEL_EXPLAINER_AVAILABLE = False

# Feature Drift Detector
try:
    from src.ml.feature_drift_detector import (
        FeatureDriftDetector,
        DriftMetrics,
        DriftReport,
    )

    DRIFT_DETECTOR_AVAILABLE = True
except ImportError:
    DRIFT_DETECTOR_AVAILABLE = False

# Export list
__all__ = [
    # Availability flags
    "VIETNAM_ML_AVAILABLE",
    "ENHANCED_ML_AVAILABLE",
    "INTEGRATION_BRIDGE_AVAILABLE",
    "MODEL_EXPLAINER_AVAILABLE",
    "DRIFT_DETECTOR_AVAILABLE",
    # Enhanced ML (Primary - 10/10)
    "EnhancedMLIntegration",
    "get_enhanced_ml_integration",
    "reset_enhanced_ml_integration",
    # Base Vietnam ML
    "VietnamMLIntegration",
    "get_vietnam_ml_integration",
    # Utilities
    "WalkForwardValidator",
    "PerformanceDecayDetector",
    "AdvancedConfidenceCalibrator",
    "RiskAdjustedMetrics",
    "RegimeModelSelector",
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
