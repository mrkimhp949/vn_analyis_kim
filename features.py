import pandas as pd
import numpy as np
import ta

def add_ml_features(df):
    """Thêm features cho ML models - ĐẢM BẢO ĐỦ 18 FEATURES"""
    if df.empty or len(df) < 50:
        print("⚠️ Không đủ dữ liệu để tính features")
        return df
        
    # Tạo bản copy để tránh warning
    df = df.copy()
    
    # ==================== CÁC FEATURES CƠ BẢN - ĐẢM BẢO TÍNH ĐƯỢC ====================
    
    # 1. MOVING AVERAGES (Luôn tính được)
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # 2. RSI (Luôn tính được)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)  # Default 50 nếu không tính được
    
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
    
    # 5. BOLLINGER BANDS (Luôn tính được)
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_high'] = df['bb_mid'] + (bb_std * 2)
    df['bb_low'] = df['bb_mid'] - (bb_std * 2)
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    df['bb_width'] = df['bb_width'].fillna(0)
    
    # 6. MOMENTUM (Luôn tính được)
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)
    
    # 7. VOLUME RATIO (Fallback nếu không có volume)
    if 'volume' in df.columns:
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
    else:
        df['volume_ratio'] = 1.0  # Default
    
    # 8. PRICE POSITION (Cần high, low) - Fallback
    if all(col in df.columns for col in ['high', 'low']):
        high_low_range = df['high'] - df['low']
        high_low_range = high_low_range.replace(0, 1)  # Tránh chia cho 0
        df['price_position'] = (df['close'] - df['low']) / high_low_range
    else:
        df['price_position'] = 0.5  # Default middle
    
    # 9. VOLATILITY (Luôn tính được)
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['volatility'] = df['volatility'].fillna(0)
    
    # 10. ADX (Cần high, low) - Fallback
    if all(col in df.columns for col in ['high', 'low', 'close']):
        try:
            # Tính ADX đơn giản
            plus_dm = df['high'].diff()
            minus_dm = -df['low'].diff()
            tr = true_range  # Đã tính ở ATR
            
            plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
            minus_di = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(14).mean()
        except:
            df['adx'] = 25  # Default neutral
    else:
        df['adx'] = 25  # Default neutral
    
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
    """Danh sách features cho ML - BẢN NÂNG CẤP"""
    basic_features = [
        'ema20', 'ema50', 'sma20', 'rsi', 'rsi_signal', 'atr'
    ]
    
    advanced_features = [
        'momentum_5', 'momentum_10', 'momentum_20',
        'price_vs_52w_high', 'price_vs_52w_low',
        'volume_ratio', 'volume_surge',
        'distance_to_support', 'distance_to_resistance', 
        'volatility_20', 'volatility_50', 'volatility_ratio',
        'adx', 'body_size', 'upper_shadow', 'lower_shadow',
        'relative_strength'
    ]
    
    technical_features = [
        'macd', 'macd_signal', 'macd_diff', 'macd_signal_line',
        'bb_high', 'bb_mid', 'bb_low', 'bb_width', 'bb_position'
    ]
    
    return basic_features + advanced_features + technical_features