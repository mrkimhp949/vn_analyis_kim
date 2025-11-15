# -*- coding: utf-8 -*-
"""
Enhanced Features cho ML models
Thêm 10+ features mới để cải thiện accuracy
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


def add_enhanced_features(
    df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Thêm enhanced features cho ML models

    New features:
    1. Price momentum indicators
    2. Volume-price relationship
    3. Volatility regime
    4. Market microstructure
    5. Relative strength improvements
    6. Lag features
    7. Rolling statistics
    8. Candlestick patterns
    9. Support/Resistance levels
    10. Trend strength

    Args:
        df: DataFrame của cổ phiếu
        index_df: DataFrame của chỉ số (VNINDEX)

    Returns:
        DataFrame với enhanced features
    """
    if df.empty or len(df) < 50:
        logger.warning("Không đủ dữ liệu để tính features")
        return df

    df = df.copy()

    # ========================================================================
    # BASE FEATURES (từ features.py cũ)
    # ========================================================================

    # 1. Moving Averages
    df["sma20"] = df["close"].rolling(20).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()

    # 2. RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["rsi_signal"] = (df["rsi"] > 30) & (df["rsi"] < 70)

    # 3. ATR
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    # 4. MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_dif"] = macd.macd_diff()
    df["macd_signal_line"] = (df["macd"] > df["macd_signal"]).astype(int)

    # 5. Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
    df["bb_position"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

    # 6. Momentum
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_20"] = df["close"].pct_change(20)

    # 7. Volume
    df["volume_sma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma20"]
    df["volume_surge"] = (df["volume_ratio"] > 1.5).astype(int)

    # 8. Volatility
    df["volatility_20"] = df["close"].pct_change().rolling(20).std()

    # ========================================================================
    # NEW FEATURES
    # ========================================================================

    # 9. PRICE MOMENTUM INDICATORS
    df["roc_5"] = ta.momentum.ROCIndicator(df["close"], window=5).roc()
    df["roc_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()
    df["stoch_k"] = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"]
    ).stoch()
    df["stoch_d"] = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"]
    ).stoch_signal()

    # 10. VOLUME-PRICE RELATIONSHIP
    # OBV (On-Balance Volume)
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        df["close"], df["volume"]
    ).on_balance_volume()
    df["obv_ema"] = df["obv"].ewm(span=20).mean()
    df["obv_signal"] = (df["obv"] > df["obv_ema"]).astype(int)

    # Volume-Weighted Average Price (VWAP)
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]

    # 11. VOLATILITY REGIME
    # Historical Volatility
    df["hv_10"] = df["close"].pct_change().rolling(10).std() * np.sqrt(252)
    df["hv_20"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
    df["hv_ratio"] = df["hv_10"] / df["hv_20"]

    # ATR Percentile
    df["atr_percentile"] = (
        df["atr"].rolling(50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    # 12. MARKET MICROSTRUCTURE
    # High-Low Range
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["hl_range_ma"] = df["hl_range"].rolling(20).mean()

    # Close position in range
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

    # Gap
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    df["gap_filled"] = (
        ((df["gap"] > 0) & (df["low"] <= df["close"].shift(1)))
        | ((df["gap"] < 0) & (df["high"] >= df["close"].shift(1)))
    ).astype(int)

    # 13. RELATIVE STRENGTH (IMPROVED)
    if index_df is not None and not index_df.empty:
        try:
            # Merge by time
            merged = pd.merge(
                df[["time", "close"]],
                index_df[["time", "close"]],
                on="time",
                suffixes=("_stock", "_index"),
            )

            if not merged.empty:
                # Calculate returns
                merged["return_stock"] = merged["close_stock"].pct_change()
                merged["return_index"] = merged["close_index"].pct_change()

                # RS = cumulative return ratio
                merged["cum_return_stock"] = (1 + merged["return_stock"]).cumprod()
                merged["cum_return_index"] = (1 + merged["return_index"]).cumprod()
                merged["rs"] = merged["cum_return_stock"] / merged["cum_return_index"]

                # RS momentum (rate of change of RS)
                merged["rs_momentum"] = merged["rs"].pct_change(10)

                # Merge back
                df = df.merge(
                    merged[["time", "rs", "rs_momentum"]], on="time", how="left"
                )

                df["rs"].fillna(1.0, inplace=True)
                df["rs_momentum"].fillna(0.0, inplace=True)
            else:
                df["rs"] = 1.0
                df["rs_momentum"] = 0.0
        except Exception:
            logger.warning("Error calculating RS")
            df["rs"] = 1.0
            df["rs_momentum"] = 0.0
    else:
        df["rs"] = 1.0
        df["rs_momentum"] = 0.0

    # 14. LAG FEATURES
    for lag in [1, 2, 3, 5]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
        df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag)

    # 15. ROLLING STATISTICS
    # Rolling mean reversion
    df["price_vs_sma20"] = (df["close"] - df["sma20"]) / df["sma20"]
    df["price_vs_ema50"] = (df["close"] - df["ema50"]) / df["ema50"]

    # Rolling Z-score
    rolling_mean = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std()
    df["zscore"] = (df["close"] - rolling_mean) / rolling_std

    # 16. CANDLESTICK PATTERNS (Simplified)
    # Doji
    body = abs(df["close"] - df["open"])
    range_hl = df["high"] - df["low"]
    df["is_doji"] = (body / range_hl < 0.1).astype(int)

    # Hammer/Hanging Man
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    df["is_hammer"] = ((lower_shadow > body * 2) & (upper_shadow < body * 0.5)).astype(
        int
    )

    # Engulfing
    prev_body = abs(df["close"].shift(1) - df["open"].shift(1))
    df["is_bullish_engulfing"] = (
        (df["close"] > df["open"])  # Current bullish
        & (df["close"].shift(1) < df["open"].shift(1))  # Prev bearish
        & (body > prev_body * 1.5)  # Engulfs previous
    ).astype(int)

    # 17. SUPPORT/RESISTANCE LEVELS
    # Recent high/low
    df["resistance_20"] = df["high"].rolling(20).max()
    df["support_20"] = df["low"].rolling(20).min()
    df["distance_to_resistance"] = (df["resistance_20"] - df["close"]) / df["close"]
    df["distance_to_support"] = (df["close"] - df["support_20"]) / df["close"]

    # 18. TREND STRENGTH
    # ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # EMA alignment (trend confirmation)
    df["ema_alignment"] = (df["ema20"] > df["ema50"]).astype(int)

    # ========================================================================
    # TARGET
    # ========================================================================
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # ========================================================================
    # FILL NaN
    # ========================================================================
    # Forward fill then backward fill
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.fillna(0, inplace=True)

    return df


def get_feature_columns() -> list:
    """
    Danh sách tất cả features cho ML

    Returns:
        List of feature names (28 features)
    """
    base_features = [
        # Base (18)
        "sma20",
        "ema20",
        "ema50",
        "rsi",
        "rsi_signal",
        "atr",
        "macd",
        "macd_signal",
        "macd_dif",
        "macd_signal_line",
        "bb_width",
        "bb_position",
        "momentum_5",
        "momentum_10",
        "momentum_20",
        "volume_ratio",
        "volume_surge",
        "volatility_20",
    ]

    new_features = [
        # New (10)
        "roc_10",  # Rate of change
        "stoch_k",  # Stochastic
        "obv_signal",  # On-Balance Volume signal
        "price_vs_vwap",  # Price vs VWAP
        "hv_ratio",  # Volatility regime
        "close_position",  # Close position in range
        "rs",  # Relative strength
        "rs_momentum",  # RS momentum
        "price_vs_sma20",  # Mean reversion
        "adx",  # Trend strength
    ]

    return base_features + new_features


def get_all_feature_columns() -> list:
    """
    Danh sách TẤT CẢ features (bao gồm cả lag features)
    Dùng cho feature importance analysis
    """
    base = get_feature_columns()

    # Add lag features
    lag_features = []
    for lag in [1, 2, 3, 5]:
        lag_features.extend([f"close_lag_{lag}", f"volume_lag_{lag}", f"rsi_lag_{lag}"])

    # Add pattern features
    pattern_features = [
        "is_doji",
        "is_hammer",
        "is_bullish_engulfing",
        "distance_to_resistance",
        "distance_to_support",
        "ema_alignment",
    ]

    return base + lag_features + pattern_features


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from src.data.loader import load_data
    from utils.dataframe_utils import safe_get_latest

    print("\n" + "=" * 70)
    print("🧪 TESTING ENHANCED FEATURES")
    print("=" * 70 + "\n")

    # Load data
    symbol = "VNM"
    df = load_data(symbol, lookback=200)

    # Load index
    index_df = load_data("VNINDEX", lookback=200, is_index=True)

    # Add features
    df_enhanced = add_enhanced_features(df, index_df)

    # Check features
    feature_cols = get_feature_columns()
    print(f"📊 Total features: {len(feature_cols)}")
    print(
        f"📊 Features available: {sum(col in df_enhanced.columns for col in feature_cols)}"
    )

    # Check for NaN
    nan_count = df_enhanced[feature_cols].isna().sum().sum()
    print(f"📊 NaN values: {nan_count}")

    # Show sample
    print("\n📊 Sample data:")
    print(df_enhanced[feature_cols].tail())

    print("\n✅ Testing complete!")
