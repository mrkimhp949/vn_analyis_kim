import pandas as pd
import numpy as np
import ta

def add_ml_features(df):
    """Thêm features cho ML models"""
    
    # Technical Indicators đã có
    df['ema20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    
    # Momentum Features
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)
    
    # Volume Features
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # Price Position
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    
    # Trend Strength (ADX)
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    
    # Target: Price direction next day (1 = up, 0 = down)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Remove NaN
    df = df.dropna()
    
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