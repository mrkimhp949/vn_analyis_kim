# [file name]: features.py
# [file content begin]
from typing import Optional

import pandas as pd
import ta
from utils.dataframe_utils import safe_get_latest, safe_rolling_operation


def add_ml_features(df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Thêm features cho ML models.

    Args:
        df: DataFrame của cổ phiếu
        index_df: DataFrame của chỉ số (e.g., VNINDEX) để tính RS

    Returns:
        DataFrame với các features đã được thêm
    """
    if df.empty or len(df) < 50:
        print("⚠️ Không đủ dữ liệu để tính features")
        return df

    df = df.copy()

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

    # 12. RELATIVE STRENGTH (RS) - TÍNH TOÁN THỰC SỰ
    if index_df is not None and not index_df.empty:
        # Hợp nhất dữ liệu theo ngày
        merged_df = pd.merge(
            df[["time", "close"]],
            index_df[["time", "close"]],
            on="time",
            suffixes=("_stock", "_index"),
        )

        if not merged_df.empty:
            # Tính performance của stock và index
            from utils.dataframe_utils import safe_get_latest

            stock_perf = (
                safe_get_latest(merged_df, "close_stock", 0) / merged_df["close_stock"].iloc[0]
            )
            index_perf = (
                safe_get_latest(merged_df, "close_index", 0) / merged_df["close_index"].iloc[0]
            )

            # RS = perf_stock / perf_index
            rs_value = stock_perf / index_perf if index_perf != 0 else 1.0
            df["relative_strength"] = rs_value
        else:
            df["relative_strength"] = 1.0  # Fallback
    else:
        # Fallback nếu không có index_df
        df["relative_strength"] = df["momentum_20"].fillna(0) + 1.0

    # 13. LAG FEATURES (NEW)
    for lag in [1, 2, 3]:
        df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag)
        df[f"macd_diff_lag_{lag}"] = df["macd_dif"].shift(lag)
        df[f"volume_ratio_lag_{lag}"] = df["volume_ratio"].shift(lag)

    # Target: Price direction next day
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Fill NaN values and ensure all features are float64
    feature_cols = get_feature_columns()
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float64')

    return df


def get_feature_columns():
    """Danh sách features cho ML - CẬP NHẬT VỚI FEATURES MỚI"""
    base_features = [
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
        "relative_strength",  # Feature mới
    ]

    # Thêm lag features
    lag_features = []
    for lag in [1, 2, 3]:
        lag_features.append(f"rsi_lag_{lag}")
        lag_features.append(f"macd_diff_lag_{lag}")
        lag_features.append(f"volume_ratio_lag_{lag}")

    return base_features + lag_features


# [file content end]
