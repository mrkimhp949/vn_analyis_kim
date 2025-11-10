import pandas as pd
import numpy as np
import ta

def add_ml_features(df):
    """Thêm features cho ML models - BẢN NÂNG CẤP"""
    if df.empty:
        return df
        
    # Tạo bản copy để tránh warning
    df = df.copy()
    
    # ==================== CÁC FEATURES CƠ BẢN ====================
    # Technical Indicators với try-catch
    try:
        if len(df) < 50:
            print("⚠️ Không đủ dữ liệu để tính indicators")
            return df
            
        # Moving Averages
        df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['sma20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['rsi_signal'] = np.where(df['rsi'] < 30, 1, 
                                  np.where(df['rsi'] > 70, -1, 0))
        
        # ATR
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close']
            ).average_true_range()
        else:
            df['atr'] = df['close'].rolling(14).std().fillna(0)
            
    except Exception as e:
        print(f"⚠️ Lỗi tính indicators cơ bản: {e}")
    
    # ==================== FEATURES NÂNG CAO ====================
    try:
        # 1. PRICE MOMENTUM FEATURES
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10) 
        df['momentum_20'] = df['close'].pct_change(20)
        
        # 52-week high/low
        df['price_vs_52w_high'] = df['close'] / df['high'].rolling(252).max()
        df['price_vs_52w_low'] = df['close'] / df['low'].rolling(252).min()
        
        # 2. VOLUME-BASED FEATURES  
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
        
        # Volume surge detection
        df['volume_surge'] = (df['volume'] > df['volume_ma20'] * 1.5).astype(int)
        
        # 3. SUPPORT/RESISTANCE FEATURES
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        df['distance_to_support'] = (df['close'] - df['support_level']) / df['support_level']
        df['distance_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
        
        # 4. VOLATILITY FEATURES
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        df['volatility_50'] = df['close'].pct_change().rolling(50).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']
        
        # 5. TREND STRENGTH FEATURES
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        else:
            df['adx'] = 0
            
        # 6. CANDLESTICK FEATURES
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 0.001)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']).replace(0, 0.001)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low']).replace(0, 0.001)
        
        # 7. RELATIVE STRENGTH vs MARKET (giả lập)
        # Trong thực tế, cần so sánh với VNINDEX
        df['relative_strength'] = df['close'].pct_change(10) - df['close'].pct_change(10).mean()
        
    except Exception as e:
        print(f"⚠️ Lỗi tính features nâng cao: {e}")
    
    # ==================== MACD & BOLLINGER BANDS ====================
    try:
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_signal_line'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
    except Exception as e:
        print(f"⚠️ Lỗi tính MACD & Bollinger: {e}")
    
    # Target: Price direction next day (1 = up, 0 = down)
    try:
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    except Exception as e:
        print(f"⚠️ Lỗi tính target: {e}")
        df['target'] = 0
    
    # Remove NaN - thay thế bằng giá trị trung bình hoặc 0
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'target':  # Giữ nguyên target
                df[col] = df[col].fillna(df[col].mean() if df[col].notna().any() else 0)
    except Exception as e:
        print(f"⚠️ Lỗi xử lý NaN: {e}")
    
    print(f"✅ Đã thêm {len([col for col in df.columns if col not in ['time', 'open', 'high', 'low', 'close', 'volume']])} features")
    
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