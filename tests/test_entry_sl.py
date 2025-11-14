import pandas as pd
import numpy as np
from datetime import datetime

from improved_entry_logic import ImprovedEntryLogic
from features import add_ml_features


def test_stop_loss_below_entry():
    # Create synthetic price data (uptrend) so filters pass
    n = 250
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n)
    close = np.linspace(10000, 11000, n)
    openp = close - 5
    high = close + 10
    low = close - 10
    volume = np.full(n, 1000)

    df = pd.DataFrame({
        'open': openp,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # Add indicators (ATR, EMA, etc.)
    df = add_ml_features(df)

    # Use higher ML confidence so adjusted confidence stays above threshold in this synthetic scenario
    ml_signal = {'signal': 'BUY', 'confidence': 90}

    logic = ImprovedEntryLogic(
        min_confidence=60,
        min_risk_reward=1.5,
        require_trend_alignment=False,
        require_volume_confirmation=False
    )

    signal = logic.analyze_entry(df, ml_signal)

    # Ensure we produced an entry and that stop loss is below entry price
    assert signal.should_enter is True, "Expected should_enter True"
    assert signal.stop_loss < signal.entry_price, f"Stop loss {signal.stop_loss} must be < entry {signal.entry_price}"
