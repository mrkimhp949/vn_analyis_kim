"""
Multi-timeframe Analysis
Phân tích đa khung thời gian để tăng độ tin cậy tín hiệu
"""

import pandas as pd
import numpy as np
from data_loader import load_data
from ml_signals import MLSignalGenerator

class MultiTimeframeAnalyzer:
    def __init__(self):
        self.ml_generator = MLSignalGenerator()
        
    def analyze_multi_timeframe(self, symbol):
        """
        Phân tích đa khung thời gian: Daily + Weekly
        
        Returns:
            dict: Tín hiệu tổng hợp từ nhiều khung thời gian
        """
        print(f"📊 Đang phân tích đa khung thời gian cho {symbol}...")
        
        signals = {}
        
        try:
            # 1. DAILY ANALYSIS (Khung ngày - chính)
            df_daily = load_data(symbol, lookback=200)
            daily_signal = self.ml_generator.analyze(df_daily)
            signals['daily'] = {
                'signal': daily_signal['signal'],
                'confidence': daily_signal['confidence'],
                'reason': daily_signal['reason']
            }
            print(f"  📅 Daily: {daily_signal['signal']} ({daily_signal['confidence']}%)")
            
        except Exception as e:
            print(f"  ❌ Lỗi Daily analysis: {e}")
            signals['daily'] = {'signal': 'HOLD', 'confidence': 0, 'reason': 'Lỗi phân tích'}
        
        try:
            # 2. WEEKLY ANALYSIS (Khung tuần - xu hướng dài hạn)
            # Giả lập weekly data từ daily (thực tế nên load weekly data thật)
            df_weekly = self._create_weekly_data(symbol)
            weekly_signal = self.ml_generator.analyze(df_weekly)
            signals['weekly'] = {
                'signal': weekly_signal['signal'], 
                'confidence': weekly_signal['confidence'],
                'reason': weekly_signal['reason']
            }
            print(f"  📆 Weekly: {weekly_signal['signal']} ({weekly_signal['confidence']}%)")
            
        except Exception as e:
            print(f"  ❌ Lỗi Weekly analysis: {e}")
            signals['weekly'] = {'signal': 'HOLD', 'confidence': 0, 'reason': 'Lỗi phân tích'}
        
        # COMBINE SIGNALS
        final_signal, final_confidence, reason = self._combine_signals(signals)
        
        print(f"  🎯 Final: {final_signal} ({final_confidence}%) - {reason}")
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'timeframe_signals': signals,
            'reason': reason
        }
    
    def _create_weekly_data(self, symbol):
        """Tạo weekly data từ daily data (tạm thời)"""
        df_daily = load_data(symbol, lookback=400)  # Load nhiều data hơn
        
        # Resample daily to weekly
        df_weekly = df_daily.copy()
        df_weekly['time'] = pd.to_datetime(df_weekly['time'])
        df_weekly = df_weekly.set_index('time')
        
        # Weekly aggregation
        weekly_df = df_weekly.resample('W').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        weekly_df = weekly_df.reset_index()
        return weekly_df.tail(100)  # Lấy 100 tuần gần nhất
    
    def _combine_signals(self, signals):
        """Kết hợp tín hiệu từ nhiều khung thời gian"""
        daily = signals.get('daily', {})
        weekly = signals.get('weekly', {})
        
        daily_signal = daily.get('signal', 'HOLD')
        daily_conf = daily.get('confidence', 0)
        weekly_signal = weekly.get('signal', 'HOLD') 
        weekly_conf = weekly.get('confidence', 0)
        
        # Weight: Daily 60%, Weekly 40%
        weights = {'daily': 0.6, 'weekly': 0.4}
        
        # Signal scoring
        signal_scores = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        
        daily_score = signal_scores.get(daily_signal, 0) * daily_conf / 100
        weekly_score = signal_scores.get(weekly_signal, 0) * weekly_conf / 100
        
        combined_score = (daily_score * weights['daily'] + 
                         weekly_score * weights['weekly'])
        
        # Confidence calculation
        if daily_signal == weekly_signal:
            # Cùng hướng -> confidence cao
            confidence = (daily_conf * weights['daily'] + 
                         weekly_conf * weights['weekly'])
            reason = f"Cùng hướng: Daily {daily_signal}, Weekly {weekly_signal}"
        else:
            # Ngược hướng -> confidence thấp
            confidence = abs(daily_conf - weekly_conf) * 0.5
            reason = f"Xung đột: Daily {daily_signal}, Weekly {weekly_signal}"
        
        # Final decision
        if combined_score >= 0.3:
            final_signal = 'BUY'
        elif combined_score <= -0.3:
            final_signal = 'SELL' 
        else:
            final_signal = 'HOLD'
            confidence *= 0.7  # Giảm confidence khi HOLD
            
        return final_signal, min(confidence, 100), reason