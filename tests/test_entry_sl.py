import numpy as np
import pandas as pd
from src.ml.features.technical import add_ml_features
from src.strategies.entry_logic import ImprovedEntryLogic


def test_stop_loss_below_entry():
    # Create synthetic price data with realistic volatility
    n = 250
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n)

    # Add random noise to create realistic price movements
    np.random.seed(42)
    trend = np.linspace(10000, 11000, n)
    noise = np.random.normal(0, 100, n)  # Add volatility
    close = trend + noise

    # Ensure high/low/open are realistic
    high = close + np.abs(np.random.normal(50, 20, n))
    low = close - np.abs(np.random.normal(50, 20, n))
    openp = close - np.random.normal(0, 30, n)
    # FIXED: Use higher volume to pass Vietnam liquidity check (2B VND min)
    # With price ~10k, need 200k+ shares: 200k * 10k = 2B VND avg_value
    volume = np.random.uniform(200_000, 250_000, n)

    df = pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )

    # Add indicators (ATR, EMA, etc.)
    df = add_ml_features(df)

    # Use higher ML confidence so adjusted confidence stays above threshold in this synthetic scenario
    ml_signal = {"signal": "BUY", "confidence": 90}

    logic = ImprovedEntryLogic(
        min_confidence=60,
        min_risk_reward=0.4,  # Lower threshold for synthetic test data (actual R:R ~0.50 after transaction costs)
        require_trend_alignment=False,
        require_volume_confirmation=False,
        # Use liquidity thresholds that match test data (2B+ VND, 200k+ volume)
        min_liquidity_value=2_000_000_000,  # 2B VND (Vietnam market requirement)
        min_avg_volume=150_000,  # 150k shares minimum
    )

    signal = logic.analyze_entry(df, ml_signal)

    # Ensure we produced an entry and that stop loss is below entry price
    assert signal.should_enter is True, "Expected should_enter True"
    assert (
        signal.stop_loss < signal.entry_price
    ), f"Stop loss {signal.stop_loss} must be < entry {signal.entry_price}"
