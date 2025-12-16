"""
ML Signals Package

Generator hierarchy:
- V3 (generator_v3.py): Ensemble + Microstructure features, 65-70% accuracy target
- V2 (generator_v2.py): Single model with V2 features, 58-62% accuracy
- V1 (enhanced.py): Basic features, ~55% accuracy

Usage:
    # Recommended: Auto-select best available
    from src.ml.signals.enhanced_v2 import EnhancedMLSignalGeneratorV2
    generator = EnhancedMLSignalGeneratorV2()  # Will use V3 -> V2 -> V1
    
    # Direct V3 access
    from src.ml.signals.generator_v3 import EnhancedMLSignalGeneratorV3
    generator = EnhancedMLSignalGeneratorV3()
"""

from src.ml.signals.enhanced_v2 import (
    EnhancedMLSignalGeneratorV2,
    get_enhanced_signal_generator,
)

__all__ = [
    "EnhancedMLSignalGeneratorV2",
    "get_enhanced_signal_generator",
]
