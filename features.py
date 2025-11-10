import pandas as pd
import numpy as np
import ta

def add_ml_features(df):
    """Thêm features cho ML models"""
    if df.empty:
        return df
        
    # Tạo bản copy để tránh warning
    df = df.copy()
    
    # Technical Indicators với try-catch
    try:
        # Đảm bảo có đủ dữ liệu
        if len(df) < 50:
            print("⚠️ Không đủ dữ liệu để tính indicators")
            return df
            
        df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # ATR cần high, low, close
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'], 
                low=df['low'], 
                close=df['close']
            ).average_true_range()
        else:
            # Fallback: dùng volatility của close price
            df['atr'] = df['close'].rolling(14).std().fillna(0)
            
    except Exception as e:
        print(f"⚠️ Lỗi tính indicators: {e}")
        # Set default values
        df['ema20'] = df['close']
        df['ema50'] = df['close'] 
        df['rsi'] = 50
        df['atr'] = 0
    
    # MACD với try-catch
    try:
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
    except Exception as e:
        print(f"⚠️ Lỗi tính MACD: {e}")
        df['macd'] = 0
        df['macd_signal'] = 0
        df['macd_diff'] = 0
    
    # Bollinger Bands với try-catch
    try:
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    except Exception as e:
        print(f"⚠️ Lỗi tính Bollinger Bands: {e}")
        df['bb_high'] = df['close']
        df['bb_mid'] = df['close']
        df['bb_low'] = df['close']
        df['bb_width'] = 0
    
    # Momentum Features với try-catch
    try:
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
    except Exception as e:
        print(f"⚠️ Lỗi tính momentum: {e}")
        df['momentum_5'] = 0
        df['momentum_10'] = 0
        df['momentum_20'] = 0
    
    # Volume Features với try-catch
    try:
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1).fillna(1)
    except Exception as e:
        print(f"⚠️ Lỗi tính volume features: {e}")
        df['volume_ma20'] = df['volume']
        df['volume_ratio'] = 1
    
    # Price Position với try-catch
    try:
        high_low_range = df['high'] - df['low']
        high_low_range = high_low_range.replace(0, 1)  # Tránh chia cho 0
        df['price_position'] = (df['close'] - df['low']) / high_low_range
    except Exception as e:
        print(f"⚠️ Lỗi tính price position: {e}")
        df['price_position'] = 0.5
    
    # Volatility với try-catch
    try:
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['volatility'] = df['volatility'].fillna(0)
    except Exception as e:
        print(f"⚠️ Lỗi tính volatility: {e}")
        df['volatility'] = 0
    
    # Trend Strength (ADX) với try-catch
    try:
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        else:
            df['adx'] = 0
    except Exception as e:
        print(f"⚠️ Lỗi tính ADX: {e}")
        df['adx'] = 0
    
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
    
    return df


def get_feature_columns():
    """Danh sách features cho ML"""
    return [
        'ema20', 'ema50', 'rsi', 'atr',
        'macd', 'macd_signal', 'macd_diff',
        'bb_high', 'bb_mid', 'bb_low', 'bb_width',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volume_ratio', 'price_position', 'volatility', 'adx'
    ]