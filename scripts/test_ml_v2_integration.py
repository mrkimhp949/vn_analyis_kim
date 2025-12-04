#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test ML V2 Integration with Entry Logic
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

from src.data.loader import load_data
from src.ml.signals.enhanced_v2 import EnhancedMLSignalGeneratorV2


def test_signal_generator():
    """Test ML V2 signal generator"""
    print("\n" + "=" * 60)
    print("🧪 TEST 1: ML Signal Generator V2")
    print("=" * 60)

    generator = EnhancedMLSignalGeneratorV2(use_v2=True)

    print(f"✅ Using V2: {generator.use_v2}")
    print(f"✅ Model loaded: {generator.model_loaded}")

    # Test with multiple symbols
    symbols = ["VNM", "FPT", "VIC", "HPG", "MWG"]

    index_df = load_data("VNINDEX", lookback=200, is_index=True)

    print("\n📊 Signals:")
    for symbol in symbols:
        df = load_data(symbol, lookback=200)
        if df is not None:
            result = generator.analyze(df, index_df, symbol=symbol)
            print(f"   {symbol}: {result['signal']} ({result['confidence']}%)")

    return True


def test_entry_service_import():
    """Test entry service import with V2"""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Entry Service Import")
    print("=" * 60)

    try:
        from src.services.entry_service import EntrySignalService

        print("✅ EntrySignalService imported successfully")

        # Check if using V2
        service = EntrySignalService()
        is_v2 = hasattr(service.ml_generator, "use_v2") and service.ml_generator.use_v2
        print(f"✅ Using V2 generator: {is_v2}")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_entry_logic_with_v2():
    """Test entry logic with V2 signals"""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Entry Logic with V2 Signals")
    print("=" * 60)

    from src.strategies.entry_logic import ImprovedEntryLogic
    from src.ml.signals.enhanced_v2 import EnhancedMLSignalGeneratorV2
    from src.ml.features.technical import add_ml_features

    # Initialize
    generator = EnhancedMLSignalGeneratorV2(use_v2=True)
    entry_logic = ImprovedEntryLogic(
        min_confidence=50,
        min_risk_reward=1.5,
        require_trend_alignment=False,
    )

    # Load data
    symbol = "VNM"
    df = load_data(symbol, lookback=200)
    index_df = load_data("VNINDEX", lookback=200, is_index=True)

    # Add features for entry logic
    df = add_ml_features(df, index_df)

    # Get ML signal
    ml_signal = generator.analyze(df, index_df, symbol=symbol)
    print(f"\n📊 ML Signal: {ml_signal['signal']} ({ml_signal['confidence']}%)")

    # Run entry logic
    entry_signal = entry_logic.analyze_entry(
        df=df,
        ml_signal=ml_signal,
        symbol=symbol,
    )

    print(f"\n📊 Entry Signal:")
    print(f"   Should Enter: {entry_signal.should_enter}")
    print(f"   Signal Type: {entry_signal.signal_type}")
    print(f"   Confidence: {entry_signal.confidence}%")
    print(f"   Reasons: {entry_signal.reasons[:3] if entry_signal.reasons else 'None'}")

    return True


def test_batch_signals():
    """Test batch signal generation"""
    print("\n" + "=" * 60)
    print("🧪 TEST 4: Batch Signal Generation")
    print("=" * 60)

    from src.ml.signals.generator_v2 import MLSignalGeneratorV2

    generator = MLSignalGeneratorV2(model_name="rf")

    symbols = ["VNM", "FPT", "VIC", "VHM", "HPG", "MWG", "MSN", "VCB", "TCB", "VPB"]

    results = generator.batch_generate_signals(symbols, lookback=200)

    print("\n📊 Batch Results:")
    buy_signals = []
    for symbol, signal in results.items():
        status = signal["signal"]
        conf = signal["confidence"]
        print(f"   {symbol}: {status} ({conf}%)")
        if status == "BUY" and conf >= 55:
            buy_signals.append((symbol, conf))

    print(f"\n🎯 BUY Signals (conf >= 55%): {len(buy_signals)}")
    for symbol, conf in sorted(buy_signals, key=lambda x: -x[1]):
        print(f"   {symbol}: {conf}%")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 ML V2 INTEGRATION TEST")
    print("=" * 60)

    tests = [
        ("Signal Generator", test_signal_generator),
        ("Entry Service Import", test_entry_service_import),
        ("Entry Logic with V2", test_entry_logic_with_v2),
        ("Batch Signals", test_batch_signals),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")

    all_passed = all(r[1] for r in results)
    print(f"\n{'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
