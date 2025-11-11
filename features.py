# [file name]: features.py
# [file content begin]
import pandas as pd
import numpy as np
import ta

def add_ml_features(df):
    """Thêm features cho ML models - FIXED VERSION"""
    if df.empty or len(df) < 50:
        print("⚠️ Không đủ dữ liệu để tính features")
        return df
        
    # Tạo bản copy để tránh warning
    df = df.copy()
    
    # ==================== CÁC FEATURES CƠ BẢN - ĐẢM BẢO TÍNH ĐƯỢC ====================
    
    # 1. MOVING AVERAGES (Luôn tính được)
    df['sma20'] = df['close'].rolling(20).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # 2. RSI (Luôn tính được)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    df['rsi_signal'] = np.where(df['rsi'] > 70, 1, np.where(df['rsi'] < 30, -1, 0))
    
    # 3. ATR (Cần high, low) - Fallback nếu thiếu
    if all(col in df.columns for col in ['high', 'low', 'close']):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        df['atr'] = true_range.rolling(window=14).mean()
    else:
        df['atr'] = df['close'].rolling(14).std()  # Fallback
    
    # 4. MACD (Luôn tính được từ close)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['macd_signal_line'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
    
    # 5. BOLLINGER BANDS (Luôn tính được)
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_high'] = df['bb_mid'] + (bb_std * 2)
    df['bb_low'] = df['bb_mid'] - (bb_std * 2)
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    df['bb_width'] = df['bb_width'].fillna(0)
    df['bb_position'] = df['bb_position'].fillna(0.5)
    
    # 6. MOMENTUM (Luôn tính được)
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)
    
    # 7. PRICE VS 52W HIGH/LOW (Giả lập)
    rolling_max = df['close'].rolling(252).max()
    rolling_min = df['close'].rolling(252).min()
    df['price_vs_52w_high'] = (df['close'] / rolling_max - 1).fillna(0)
    df['price_vs_52w_low'] = (df['close'] / rolling_min - 1).fillna(0)
    
    # 8. VOLUME FEATURES
    if 'volume' in df.columns:
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
        df['volume_surge'] = np.where(df['volume_ratio'] > 1.5, 1, 0)
    else:
        df['volume_ratio'] = 1.0
        df['volume_surge'] = 0
    
    # 9. SUPPORT/RESISTANCE (Giả lập)
    df['distance_to_support'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
    df['distance_to_resistance'] = (df['high'].rolling(20).max() - df['close']) / df['close']
    
    # 10. VOLATILITY (Luôn tính được)
    df['volatility_20'] = df['close'].pct_change().rolling(20).std().fillna(0)
    df['volatility_50'] = df['close'].pct_change().rolling(50).std().fillna(0)
    df['volatility_ratio'] = (df['volatility_20'] / df['volatility_50']).replace([np.inf, -np.inf], 1).fillna(1)
    
    # 11. CANDLESTICK FEATURES
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
    else:
        df['body_size'] = 0
        df['upper_shadow'] = 0
        df['lower_shadow'] = 0
    
    # 12. RELATIVE STRENGTH (Giả lập)
    # Trong thực tế cần so sánh với index, tạm thời dùng momentum
    df['relative_strength'] = df['momentum_20'].fillna(0)
    
    # Target: Price direction next day
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'target':
            df[col] = df[col].fillna(df[col].mean() if df[col].notna().any() else 0)
    
    # KIỂM TRA FEATURES CUỐI CÙNG
    feature_cols = get_feature_columns()
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    print(f"✅ Generated {len(available_features)}/{len(feature_cols)} features")
    if missing_features:
        print(f"⚠️ Still missing: {missing_features}")
    
    return df

def get_feature_columns():
    """Danh sách features cho ML - CẬP NHẬT ĐÚNG 18 FEATURES"""
    # CHỈ GIỮ LẠI 18 FEATURES QUAN TRỌNG NHẤT
    return [
        'sma20', 'ema20', 'ema50', 
        'rsi', 'rsi_signal',
        'atr',
        'macd', 'macd_signal', 'macd_diff', 'macd_signal_line',
        'bb_width', 'bb_position',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volume_ratio', 'volume_surge',
        'volatility_20'
    ]
# [file content end]