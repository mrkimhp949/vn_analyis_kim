"""
Demo script for improvements:
1. Automatic Market Regime Detection
2. Feature Importance Analysis
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def demo_market_regime_detection():
    """Demo automatic market regime detection"""
    print("\n" + "=" * 80)
    print("🔍 DEMO 1: AUTOMATIC MARKET REGIME DETECTION")
    print("=" * 80 + "\n")

    from src.data.loader import load_data
    from src.market.regime_detector import MarketRegimeDetector

    # Load VN-Index data
    print("📊 Loading VN-Index data...")
    vnindex_df = load_data("VNINDEX", lookback=250, is_index=True)

    if vnindex_df is None or vnindex_df.empty:
        print("❌ Could not load VN-Index data")
        return

    print(f"✅ Loaded {len(vnindex_df)} bars of VN-Index data\n")

    # Create detector
    detector = MarketRegimeDetector(
        bull_threshold=0.6,
        bear_threshold=-0.6,
        volatility_threshold=0.7,
        min_confidence=50.0,
    )

    # Detect regime
    print("🔍 Detecting market regime...")
    regime = detector.detect(vnindex_df)

    print("\n" + "-" * 80)
    print("📊 MARKET REGIME ANALYSIS")
    print("-" * 80)
    print(f"Regime:      {regime.regime}")
    print(f"Confidence:  {regime.confidence:.1f}%")
    print(f"Tradeable:   {'✅ YES' if regime.tradeable else '❌ NO'}")
    print(f"Description: {regime.description}")
    print("\nComponents:")
    for key, value in regime.components.items():
        print(f"  - {key:20s}: {value:+.4f}")
    print("-" * 80 + "\n")

    # Interpretation
    print("💡 INTERPRETATION:")
    if regime.regime == "BULL":
        print("  🚀 Market is in BULL mode - Good for long positions")
        print("  ✅ Increase position sizes (1.1x multiplier)")
    elif regime.regime == "BEAR":
        print("  📉 Market is in BEAR mode - Avoid long positions")
        print("  ⚠️ Reduce position sizes significantly (0.5x multiplier)")
    elif regime.regime == "SIDEWAYS":
        print("  ↔️  Market is SIDEWAYS - Trade with caution")
        print("  ⚙️  Use moderate position sizes (0.8x multiplier)")
    elif regime.regime == "HIGH_VOLATILITY":
        print("  ⚡ Market has HIGH VOLATILITY - Very risky")
        print("  🛑 Reduce position sizes or avoid trading (0.6x multiplier)")

    if not regime.tradeable:
        print("\n⚠️  WARNING: Market is marked as NOT TRADEABLE")
        print("   Consider sitting out or using very small positions\n")

    return regime


def demo_feature_importance():
    """Demo feature importance analysis"""
    print("\n" + "=" * 80)
    print("📊 DEMO 2: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80 + "\n")

    from src.ml.models.predictor import MLPredictor

    # Create predictor
    predictor = MLPredictor()

    # Try to load existing models
    print("📦 Loading ML models...")
    if predictor.load_models():
        print("✅ Models loaded successfully\n")

        # Check if feature importance is available
        if predictor.feature_importance is not None:
            print("📊 FEATURE IMPORTANCE (from saved model):")
            print("-" * 80)

            # Display top 15 features
            top_n = 15
            for idx, row in predictor.feature_importance.head(top_n).iterrows():
                bar_length = int(row["importance"] * 50)
                bar = "█" * bar_length
                print(
                    f"{idx+1:2d}. {row['feature']:25s} "
                    f"{bar} {row['importance']:6.4f} "
                    f"(Cum: {row['cumulative']:6.1%})"
                )

            print("-" * 80)

            # Calculate how many features for 80%
            threshold_80 = predictor.feature_importance[
                predictor.feature_importance["cumulative"] <= 0.80
            ]
            if len(threshold_80) < len(predictor.feature_importance):
                threshold_80 = predictor.feature_importance.iloc[: len(threshold_80) + 1]

            print(f"\n💡 INSIGHT:")
            print(
                f"  - Top {len(threshold_80)} features explain "
                f"{threshold_80['cumulative'].iloc[-1]:.1%} of variance"
            )
            print(
                f"  - Could reduce from {len(predictor.feature_importance)} "
                f"to {len(threshold_80)} features"
            )
            print(
                f"  - This would speed up training/prediction and reduce overfitting risk\n"
            )

            # Show least important features
            print("\n🗑️  LEAST IMPORTANT FEATURES (candidates for removal):")
            print("-" * 80)
            bottom_5 = predictor.feature_importance.tail(5)
            for idx, row in bottom_5.iterrows():
                print(f"  - {row['feature']:25s} Importance: {row['importance']:6.4f}")
            print("-" * 80 + "\n")

        else:
            print("⚠️  Feature importance not available in saved models")
            print("   Train a new model to generate feature importance:\n")
            print("   $ python scripts/train_models.py\n")

    else:
        print("❌ Could not load models")
        print("   Train models first:")
        print("   $ python scripts/train_models.py\n")


def demo_position_sizing_with_regime():
    """Demo position sizing with auto regime detection"""
    print("\n" + "=" * 80)
    print("💰 DEMO 3: POSITION SIZING WITH AUTO REGIME DETECTION")
    print("=" * 80 + "\n")

    from src.strategies.position_sizing import EnhancedPositionSizer

    # Create position sizer
    sizer = EnhancedPositionSizer(
        total_capital=100_000_000,  # 100M VND
        max_risk_per_trade=0.02,  # 2%
        max_position_size=0.15,  # 15%
        use_kelly=True,
        kelly_fraction=0.5,
    )

    # Example trade
    symbol = "VNM"
    entry_price = 85_000
    stop_loss = 82_000
    take_profit = 91_000
    confidence = 75

    print(f"📈 Example Trade: {symbol}")
    print(f"  Entry Price:   {entry_price:,} VND")
    print(f"  Stop Loss:     {stop_loss:,} VND  (-{(1-stop_loss/entry_price)*100:.1f}%)")
    print(
        f"  Take Profit:   {take_profit:,} VND  (+{(take_profit/entry_price-1)*100:.1f}%)"
    )
    print(f"  Confidence:    {confidence}%")
    print()

    print("🔍 Calculating position size WITH auto regime detection...")

    # Calculate position with auto regime detection
    try:
        position = sizer.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            signal_strength="STRONG",
            win_rate=0.55,  # 55% win rate
            avg_win_loss_ratio=1.8,  # 1.8:1 win/loss ratio
            auto_detect_regime=True,  # AUTO-DETECT REGIME
        )

        print("\n" + "-" * 80)
        print("💰 POSITION SIZING RESULT")
        print("-" * 80)
        print(f"Shares:           {position.shares:,}")
        print(f"Value:            {position.value:,.0f} VND  ({position.position_percent:.2f}%)")
        print(f"Risk Amount:      {position.risk_amount:,.0f} VND  ({position.risk_percent:.2f}%)")
        print(f"Max Loss:         {position.max_loss:,.0f} VND")
        print(f"Kelly Percent:    {position.kelly_percent:.2f}%")

        print("\nAdjustments Applied:")
        for key, value in position.adjustments.items():
            if isinstance(value, float):
                print(f"  - {key:25s}: {value:.4f}")
            else:
                print(f"  - {key:25s}: {value}")

        if position.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in position.warnings:
                print(f"  - {warning}")

        print("\nRecommended DCA Entries:")
        for entry in position.recommended_entries:
            print(
                f"  Level {entry['level']}: {entry['price']:,} VND "
                f"({entry['shares']:,} shares, {entry['percent']}%)"
            )

        print("-" * 80 + "\n")

    except Exception as e:
        print(f"❌ Error calculating position: {e}\n")
        import traceback

        traceback.print_exc()


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("🎯 IMPROVEMENTS DEMO")
    print("=" * 80)
    print()
    print("This script demonstrates 2 key improvements:")
    print("  1. Automatic Market Regime Detection")
    print("  2. Feature Importance Analysis & Selection")
    print()

    try:
        # Demo 1: Market Regime Detection
        regime = demo_market_regime_detection()

        # Demo 2: Feature Importance
        demo_feature_importance()

        # Demo 3: Position Sizing with Regime
        demo_position_sizing_with_regime()

        print("\n" + "=" * 80)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

        print("💡 NEXT STEPS:")
        print("  1. Train models to get feature importance:")
        print("     $ python scripts/train_models.py")
        print()
        print("  2. Use auto regime detection in your trading bot:")
        print("     position = sizer.calculate_position_size(..., auto_detect_regime=True)")
        print()
        print("  3. Review feature importance to optimize your features")
        print()

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
