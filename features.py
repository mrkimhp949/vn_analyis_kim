# [file name]: features.py
# [file content begin]
import pandas as pd
import numpy as np
import ta
from typing import Optional

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
    
    # ... (các features khác giữ nguyên) ...

    # 12. RELATIVE STRENGTH (RS) - TÍNH TOÁN THỰC SỰ
    if index_df is not None and not index_df.empty:
        # Hợp nhất dữ liệu theo ngày
        merged_df = pd.merge(df[['time', 'close']], index_df[['time', 'close']], on='time', suffixes=('_stock', '_index'))
        
        if not merged_df.empty:
            # Tính performance của stock và index
            stock_perf = merged_df['close_stock'].iloc[-1] / merged_df['close_stock'].iloc[0]
            index_perf = merged_df['close_index'].iloc[-1] / merged_df['close_index'].iloc[0]
            
            # RS = perf_stock / perf_index
            rs_value = stock_perf / index_perf if index_perf != 0 else 1.0
            df['relative_strength'] = rs_value
        else:
            df['relative_strength'] = 1.0 # Fallback
    else:
        # Fallback nếu không có index_df
        df['relative_strength'] = df['momentum_20'].fillna(0) + 1.0

    # 13. LAG FEATURES (NEW)
    for lag in [1, 2, 3]:
        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        df[f'macd_diff_lag_{lag}'] = df['macd_diff'].shift(lag)
        df[f'volume_ratio_lag_{lag}'] = df['volume_ratio'].shift(lag)

    # Target: Price direction next day
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Fill NaN values
    # ... (giữ nguyên) ...
    
    return df

def get_feature_columns():
    """Danh sách features cho ML - CẬP NHẬT VỚI FEATURES MỚI"""
    base_features = [
        'sma20', 'ema20', 'ema50', 
        'rsi', 'rsi_signal',
        'atr',
        'macd', 'macd_signal', 'macd_diff', 'macd_signal_line',
        'bb_width', 'bb_position',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volume_ratio', 'volume_surge',
        'volatility_20',
        'relative_strength' # Feature mới
    ]
    
    # Thêm lag features
    lag_features = []
    for lag in [1, 2, 3]:
        lag_features.append(f'rsi_lag_{lag}')
        lag_features.append(f'macd_diff_lag_{lag}')
        lag_features.append(f'volume_ratio_lag_{lag}')
        
    return base_features + lag_features
# [file content end]