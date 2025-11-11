# market_regime_proxy.py
import sys
import os

# Fix encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except:
        pass

def safe_print(message):
    """Print an toàn"""
    try:
        print(message)
    except UnicodeEncodeError:
        clean_message = ''.join(char for char in message if ord(char) < 128)
        print(clean_message)

import pandas as pd
from data_loader import load_data
from ml_signals import MLSignalGenerator

class ProxyMarketRegimeAnalyzer:
    def __init__(self):
        from config import TICKERS
        # ✅ DÙNG TẤT CẢ MÃ TỪ CONFIG, không hardcode
        self.proxy_stocks = TICKERS
        self.ml_generator = MLSignalGenerator()
        safe_print(f"Khoi tao Market Regime Analyzer voi {len(self.proxy_stocks)} ma tu config")
    
    def analyze_market_regime(self):
        from config import TICKERS
        safe_print(f"PHAN TICH THI TRUONG: Bat dau phan tich {len(self.proxy_stocks)} ma")
        
        analyzed = 0
        buy_signals = 0
        total_confidence = 0
        
        # ✅ PHÂN TÍCH TẤT CẢ MÃ TRONG CONFIG
        for symbol in self.proxy_stocks:
            try:
                safe_print(f"  Dang phan tich {symbol}...")
                
                df = load_data(symbol, lookback=100)
                if df.empty or len(df) < 50:
                    safe_print(f"  {symbol}: Khong du du lieu")
                    continue
                    
                ml_signal = self.ml_generator.analyze(df)
                analyzed += 1
                
                if ml_signal['signal'] == 'BUY':
                    buy_signals += 1
                    signal_info = "BUY"
                else:
                    signal_info = ml_signal['signal']
                
                total_confidence += ml_signal['confidence']
                
                safe_print(f"  {symbol}: {signal_info} ({ml_signal['confidence']}%)")
                
            except Exception as e:
                safe_print(f"  LOI {symbol}: {e}")
                continue
        
        safe_print(f"DA PHAN TICH: {analyzed}/{len(self.proxy_stocks)} ma")
        
        if analyzed == 0:
            return {
                'regime': 'UNKNOWN', 
                'tradeable': False, 
                'confidence': 0, 
                'message': 'Khong phan tich duoc thi truong'
            }
        
        buy_rate = (buy_signals / analyzed) * 100
        avg_confidence = total_confidence / analyzed
        
        # Xac dinh market regime
        if buy_rate >= 60:
            regime = 'BULL'
            tradeable = True
        elif buy_rate >= 40:
            regime = 'SIDEWAYS' 
            tradeable = True
        else:
            regime = 'BEAR'
            tradeable = avg_confidence > 30
        
        message = f"{regime} market - {buy_rate:.1f}% ma BUY ({analyzed}/{len(self.proxy_stocks)} ma)"
        
        safe_print(f"KET QUA: {message}")
        
        return {
            'regime': regime,
            'tradeable': tradeable,
            'confidence': avg_confidence,
            'message': message,
            'analyzed_stocks': analyzed,
            'total_stocks': len(self.proxy_stocks),
            'buy_rate': buy_rate
        }
