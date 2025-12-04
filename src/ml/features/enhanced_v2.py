# -*- coding: utf-8 -*-
"""
Enhanced Features V2 - Improved for 65%+ accuracy target
Key improvements:
1. Better target definition (forward returns with threshold)
2. Predictive features (not just lagging indicators)
3. Market regime features
4. Momentum quality indicators
5. Mean reversion signals
6. NEW: Price action patterns
7. NEW: Liquidity features
8. NEW: Sentiment proxy features
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

# =============================================================================
# TARGET DEFINITION - KEY IMPROVEMENT
# =============================================================================


def create_improved_target(
    df: pd.DataFrame,
    forward_days: int = 5,
    profit_threshold: float = 0.02,
    use_risk_adjusted: bool = True,
) -> pd.Series:
    """
    Create improved target variable.

    Old target: next day up/down (too noisy, ~50% random)
    New target: 5-day forward return > 2% (more predictable pattern)

    Args:
        df: DataFrame with OHLCV
        forward_days: Days to look forward (default 5)
        profit_threshold: Min return to be classified as BUY (default 2%)
        use_risk_adjusted: Adjust threshold by volatility

    Returns:
        Series with target (1=BUY opportunity, 0=NO)
    """
    # Calculate forward returns
    forward_return = df["close"].pct_change(forward_days).shift(-forward_days)

    if use_risk_adjusted:
        # Adjust threshold by recent volatility
        volatility = df["close"].pct_change().rolling(20).std()
        # Higher volatility = higher threshold needed
        adjusted_threshold = profit_threshold * (1 + volatility / volatility.mean())
        adjusted_threshold = adjusted_threshold.clip(0.01, 0.05)  # 1% to 5%
        target = (forward_return > adjusted_threshold).astype(int)
    else:
        target = (forward_return > profit_threshold).astype(int)

    return target


def create_multi_horizon_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create targets for multiple horizons - helps model learn different patterns.
    IMPROVED: Risk-adjusted returns and drawdown consideration.
    """
    df = df.copy()

    # Calculate volatility for risk adjustment
    vol_20 = df["close"].pct_change().rolling(20).std()
    vol_factor = (vol_20 / vol_20.rolling(50).mean()).clip(0.5, 2.0).fillna(1.0)

    # Short-term (3 days) - momentum
    ret_3d = df["close"].pct_change(3).shift(-3)
    threshold_3d = 0.015 * vol_factor
    df["target_3d"] = (ret_3d > threshold_3d).astype(int)

    # Medium-term (5 days) - main target
    ret_5d = df["close"].pct_change(5).shift(-5)
    threshold_5d = 0.02 * vol_factor
    df["target_5d"] = (ret_5d > threshold_5d).astype(int)

    # Longer-term (10 days) - trend
    ret_10d = df["close"].pct_change(10).shift(-10)
    threshold_10d = 0.03 * vol_factor
    df["target_10d"] = (ret_10d > threshold_10d).astype(int)

    # NEW: Max drawdown check - avoid entries before big drops
    future_low_5d = df["low"].rolling(5).min().shift(-5)
    max_dd_5d = (future_low_5d - df["close"]) / df["close"]
    no_big_drawdown = (max_dd_5d > -0.05).astype(int)  # No >5% drawdown

    # Combined target: at least 2 of 3 horizons positive AND no big drawdown
    df["target"] = (
        ((df["target_3d"] + df["target_5d"] + df["target_10d"]) >= 2) & (no_big_drawdown == 1)
    ).astype(int)

    return df


# =============================================================================
# PREDICTIVE FEATURES - Not just lagging indicators
# =============================================================================


def add_predictive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that have predictive power, not just describe current state.
    """
    df = df.copy()

    # =========================================================================
    # 1. MOMENTUM QUALITY - Is momentum sustainable?
    # =========================================================================

    # Price momentum
    df["mom_5"] = df["close"].pct_change(5)
    df["mom_10"] = df["close"].pct_change(10)
    df["mom_20"] = df["close"].pct_change(20)

    # Momentum acceleration (2nd derivative)
    df["mom_accel"] = df["mom_5"] - df["mom_5"].shift(5)

    # Momentum consistency (how many of last 5 days were up) - optimized
    close_diff = df["close"].diff()
    df["up_days_5"] = (close_diff > 0).rolling(5).sum() / 5
    df["up_days_10"] = (close_diff > 0).rolling(10).sum() / 10

    # Momentum vs Volume (confirmed momentum)
    vol_change = df["volume"].pct_change(5)
    df["mom_vol_confirm"] = (df["mom_5"] * vol_change).clip(-1, 1)

    # =========================================================================
    # 2. MEAN REVERSION SIGNALS
    # =========================================================================

    # Distance from moving averages (normalized)
    df["dist_sma20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(
        20
    ).std()
    df["dist_sma50"] = (df["close"] - df["close"].rolling(50).mean()) / df["close"].rolling(
        50
    ).std()

    # RSI extremes (oversold = potential bounce)
    rsi = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["rsi"] = rsi
    df["rsi_oversold"] = (rsi < 30).astype(int)
    df["rsi_overbought"] = (rsi > 70).astype(int)
    df["rsi_neutral"] = ((rsi >= 40) & (rsi <= 60)).astype(int)

    # Bollinger Band position
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_pct"] = (df["close"] - bb.bollinger_lband()) / (
        bb.bollinger_hband() - bb.bollinger_lband()
    )
    df["bb_squeeze"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]

    # =========================================================================
    # 3. VOLUME PATTERNS - Leading indicator
    # =========================================================================

    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"]

    # Volume trend (accumulation/distribution)
    df["vol_trend"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()

    # Price-Volume divergence (bearish if price up but volume down)
    price_dir = np.sign(df["close"].diff(5))
    vol_dir = np.sign(df["volume"].diff(5))
    df["pv_divergence"] = (price_dir != vol_dir).astype(int)

    # OBV trend
    obv = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["obv_trend"] = (obv > obv.rolling(20).mean()).astype(int)

    # =========================================================================
    # 4. VOLATILITY REGIME
    # =========================================================================

    # ATR normalized
    atr = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()
    df["atr"] = atr
    df["atr_pct"] = atr / df["close"]

    # Volatility regime (high/low)
    vol_20 = df["close"].pct_change().rolling(20).std()
    vol_50 = df["close"].pct_change().rolling(50).std()
    df["vol_regime"] = (vol_20 > vol_50).astype(int)  # 1 = expanding volatility

    # Volatility percentile - optimized (avoid lambda)
    df["vol_percentile"] = vol_20.rolling(100).rank(pct=True)

    # =========================================================================
    # 5. TREND STRENGTH
    # =========================================================================

    # ADX
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()
    df["adx_strong"] = (df["adx"] > 25).astype(int)

    # EMA alignment
    ema_10 = df["close"].ewm(span=10).mean()
    ema_20 = df["close"].ewm(span=20).mean()
    ema_50 = df["close"].ewm(span=50).mean()

    df["ema_aligned_bull"] = ((ema_10 > ema_20) & (ema_20 > ema_50)).astype(int)
    df["ema_aligned_bear"] = ((ema_10 < ema_20) & (ema_20 < ema_50)).astype(int)

    # Trend consistency
    df["above_ema20"] = (df["close"] > ema_20).astype(int)
    df["above_ema20_streak"] = df["above_ema20"].rolling(10).sum() / 10

    # =========================================================================
    # 6. SUPPORT/RESISTANCE PROXIMITY
    # =========================================================================

    # Recent high/low
    high_20 = df["high"].rolling(20).max()
    low_20 = df["low"].rolling(20).min()

    df["near_high"] = (df["close"] > high_20 * 0.98).astype(int)  # Within 2% of high
    df["near_low"] = (df["close"] < low_20 * 1.02).astype(int)  # Within 2% of low

    # Breakout potential
    df["range_position"] = (df["close"] - low_20) / (high_20 - low_20)

    # =========================================================================
    # 7. MACD SIGNALS
    # =========================================================================

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["macd_cross_up"] = (
        (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)
    df["macd_positive"] = (df["macd_hist"] > 0).astype(int)

    # =========================================================================
    # 8. PRICE ACTION PATTERNS (NEW - High predictive power)
    # =========================================================================

    # Candlestick body size
    body = abs(df["close"] - df["open"])
    full_range = df["high"] - df["low"]
    df["body_ratio"] = (body / full_range.replace(0, np.nan)).fillna(0.5)

    # Upper/Lower shadows
    upper_shadow = df["high"] - df[["close", "open"]].max(axis=1)
    lower_shadow = df[["close", "open"]].min(axis=1) - df["low"]
    df["upper_shadow_ratio"] = (upper_shadow / full_range.replace(0, np.nan)).fillna(0)
    df["lower_shadow_ratio"] = (lower_shadow / full_range.replace(0, np.nan)).fillna(0)

    # Bullish/Bearish candle
    df["bullish_candle"] = (df["close"] > df["open"]).astype(int)

    # Consecutive up/down days
    df["consec_up"] = df["bullish_candle"].rolling(5).sum()
    df["consec_down"] = 5 - df["consec_up"]

    # Gap detection
    df["gap_up"] = (df["open"] > df["high"].shift(1)).astype(int)
    df["gap_down"] = (df["open"] < df["low"].shift(1)).astype(int)

    # Inside bar (consolidation)
    df["inside_bar"] = (
        (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    ).astype(int)

    # =========================================================================
    # 9. LIQUIDITY FEATURES (NEW)
    # =========================================================================

    # Volume-weighted price stability
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_20"] = (typical_price * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["price_vs_vwap"] = (df["close"] - df["vwap_20"]) / df["vwap_20"]

    # Liquidity score (volume * price range)
    df["liquidity_score"] = (df["volume"] * full_range / df["close"]).rolling(10).mean()
    df["liquidity_score"] = df["liquidity_score"] / df["liquidity_score"].rolling(50).mean()

    # Volume spike detection
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["vol_spike"] = ((df["volume"] - vol_mean) / vol_std.replace(0, 1)).clip(-3, 3)

    # =========================================================================
    # 10. SENTIMENT PROXY FEATURES (NEW)
    # =========================================================================

    # Buying pressure (close position in range)
    df["buying_pressure"] = (df["close"] - df["low"]) / full_range.replace(0, np.nan)
    df["buying_pressure"] = df["buying_pressure"].fillna(0.5)

    # Accumulation/Distribution momentum
    ad = ((2 * df["close"] - df["low"] - df["high"]) / full_range.replace(0, np.nan)) * df["volume"]
    df["ad_momentum"] = ad.rolling(10).mean() / ad.rolling(30).mean()
    df["ad_momentum"] = df["ad_momentum"].fillna(1).clip(0.5, 2)

    # Smart money indicator (volume on up vs down days)
    up_vol = df["volume"].where(df["close"] > df["close"].shift(1), 0)
    down_vol = df["volume"].where(df["close"] <= df["close"].shift(1), 0)
    df["smart_money"] = up_vol.rolling(10).sum() / (up_vol.rolling(10).sum() + down_vol.rolling(10).sum() + 1)

    # =========================================================================
    # 11. PATTERN RECOGNITION FEATURES (NEW)
    # =========================================================================

    # Higher highs / Lower lows
    df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)
    df["hh_count_5"] = df["higher_high"].rolling(5).sum()
    df["ll_count_5"] = df["lower_low"].rolling(5).sum()

    # Trend structure
    df["uptrend_structure"] = ((df["hh_count_5"] >= 3) & (df["ll_count_5"] <= 2)).astype(int)
    df["downtrend_structure"] = ((df["ll_count_5"] >= 3) & (df["hh_count_5"] <= 2)).astype(int)

    # Price compression (low volatility before breakout)
    range_5 = (df["high"].rolling(5).max() - df["low"].rolling(5).min()) / df["close"]
    range_20 = (df["high"].rolling(20).max() - df["low"].rolling(20).min()) / df["close"]
    df["price_compression"] = (range_5 / range_20.replace(0, np.nan)).fillna(1)

    return df


# =============================================================================
# MARKET REGIME FEATURES
# =============================================================================


def add_market_regime_features(
    df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Add market regime features - crucial for conditional predictions.
    """
    df = df.copy()

    if index_df is not None and not index_df.empty:
        try:
            # Merge index data
            index_df = index_df.copy()
            index_df = index_df.rename(columns={"close": "index_close", "volume": "index_volume"})

            df = pd.merge(
                df, index_df[["time", "index_close", "index_volume"]], on="time", how="left"
            )

            # Forward fill missing index data
            df["index_close"] = df["index_close"].ffill()
            df["index_volume"] = df["index_volume"].ffill()

            # Index momentum
            df["index_mom_5"] = df["index_close"].pct_change(5)
            df["index_mom_20"] = df["index_close"].pct_change(20)

            # Index trend
            index_ema20 = df["index_close"].ewm(span=20).mean()
            df["index_above_ema20"] = (df["index_close"] > index_ema20).astype(int)

            # Relative strength vs index
            stock_return = df["close"].pct_change(20)
            index_return = df["index_close"].pct_change(20)
            df["relative_strength"] = stock_return - index_return
            df["outperforming"] = (df["relative_strength"] > 0).astype(int)

            # Beta (rolling)
            stock_ret = df["close"].pct_change()
            index_ret = df["index_close"].pct_change()

            cov = stock_ret.rolling(60).cov(index_ret)
            var = index_ret.rolling(60).var()
            df["beta"] = (cov / var).clip(0, 3)

            # Market regime classification
            df["bull_market"] = (df["index_mom_20"] > 0.02).astype(int)
            df["bear_market"] = (df["index_mom_20"] < -0.02).astype(int)

        except Exception as e:
            logger.warning(f"Market regime features failed: {e}")
            df["index_mom_5"] = 0
            df["index_mom_20"] = 0
            df["index_above_ema20"] = 1
            df["relative_strength"] = 0
            df["outperforming"] = 0
            df["beta"] = 1
            df["bull_market"] = 0
            df["bear_market"] = 0
    else:
        # Default values when no index data
        df["index_mom_5"] = 0
        df["index_mom_20"] = 0
        df["index_above_ema20"] = 1
        df["relative_strength"] = 0
        df["outperforming"] = 0
        df["beta"] = 1
        df["bull_market"] = 0
        df["bear_market"] = 0

    return df


# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================


def add_enhanced_features_v2(
    df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None, target_type: str = "multi_horizon"
) -> pd.DataFrame:
    """
    Main function to add all enhanced features V2.

    Args:
        df: Stock DataFrame with OHLCV
        index_df: Index DataFrame (VNINDEX)
        target_type: 'simple', 'threshold', or 'multi_horizon'

    Returns:
        DataFrame with all features and target
    """
    if df.empty or len(df) < 60:
        logger.warning("Insufficient data for feature engineering")
        return df

    df = df.copy()

    # 1. Add predictive features
    df = add_predictive_features(df)

    # 2. Add market regime features
    df = add_market_regime_features(df, index_df)

    # 3. Create target
    if target_type == "multi_horizon":
        df = create_multi_horizon_target(df)
    elif target_type == "threshold":
        df["target"] = create_improved_target(df, forward_days=5, profit_threshold=0.02)
    else:
        # Simple next-day (old method - not recommended)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # 4. Fill NaN values
    feature_cols = get_feature_columns_v2()

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Forward fill then backward fill
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # Fill remaining NaN with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    # Replace inf values
    df = df.replace([np.inf, -np.inf], 0)

    return df


def get_feature_columns_v2() -> list:
    """
    Get list of feature columns for V2 model.
    Carefully selected for predictive power.
    UPDATED: Added 20 new high-predictive features (total 63)
    """
    features = [
        # Momentum (7)
        "mom_5",
        "mom_10",
        "mom_20",
        "mom_accel",
        "up_days_5",
        "up_days_10",
        "mom_vol_confirm",
        # Mean reversion (7)
        "dist_sma20",
        "dist_sma50",
        "rsi",
        "rsi_oversold",
        "rsi_overbought",
        "bb_pct",
        "bb_squeeze",
        # Volume (5)
        "vol_ratio",
        "vol_trend",
        "pv_divergence",
        "obv_trend",
        # Volatility (4)
        "atr_pct",
        "vol_regime",
        "vol_percentile",
        # Trend (6)
        "adx",
        "adx_strong",
        "ema_aligned_bull",
        "ema_aligned_bear",
        "above_ema20",
        "above_ema20_streak",
        # Support/Resistance (3)
        "near_high",
        "near_low",
        "range_position",
        # MACD (3)
        "macd_hist",
        "macd_cross_up",
        "macd_positive",
        # Market regime (8)
        "index_mom_5",
        "index_mom_20",
        "index_above_ema20",
        "relative_strength",
        "outperforming",
        "beta",
        "bull_market",
        "bear_market",
        # NEW: Price action patterns (10)
        "body_ratio",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "bullish_candle",
        "consec_up",
        "consec_down",
        "gap_up",
        "gap_down",
        "inside_bar",
        # NEW: Liquidity (4)
        "price_vs_vwap",
        "liquidity_score",
        "vol_spike",
        # NEW: Sentiment proxy (3)
        "buying_pressure",
        "ad_momentum",
        "smart_money",
        # NEW: Pattern recognition (6)
        "hh_count_5",
        "ll_count_5",
        "uptrend_structure",
        "downtrend_structure",
        "price_compression",
    ]

    return features


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from src.data.loader import load_data

    print("\n" + "=" * 70)
    print("🧪 TESTING ENHANCED FEATURES V2")
    print("=" * 70)

    # Load data
    symbol = "VNM"
    df = load_data(symbol, lookback=300)
    index_df = load_data("VNINDEX", lookback=300, is_index=True)

    # Add features
    df_enhanced = add_enhanced_features_v2(df, index_df, target_type="multi_horizon")

    # Check
    feature_cols = get_feature_columns_v2()
    print(f"\n📊 Total features: {len(feature_cols)}")
    print(f"📊 Features available: {sum(col in df_enhanced.columns for col in feature_cols)}")

    # Check NaN
    nan_count = df_enhanced[feature_cols].isna().sum().sum()
    print(f"📊 NaN values: {nan_count}")

    # Check target distribution
    if "target" in df_enhanced.columns:
        target_dist = df_enhanced["target"].value_counts(normalize=True)
        print(f"\n📊 Target distribution:")
        print(f"   Class 0: {target_dist.get(0, 0)*100:.1f}%")
        print(f"   Class 1: {target_dist.get(1, 0)*100:.1f}%")

    print("\n✅ Testing complete!")
