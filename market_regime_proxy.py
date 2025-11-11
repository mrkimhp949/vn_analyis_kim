# -*- coding: utf-8 -*-
"""
market_regime_proxy.py - Market Regime Detection Using Proxy Stocks
Sử dụng các cổ phiếu blue-chip thay vì VNINDEX để phát hiện market regime
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from data_loader import load_data
import logging

logger = logging.getLogger(__name__)


class ProxyMarketRegimeAnalyzer:
    """
    Phân tích market regime sử dụng proxy stocks
    (VCB, VNM, HPG, VHM, GAS - các cổ phiếu đại diện thị trường)
    """
    
    # Blue-chip stocks làm proxy cho VNINDEX
    PROXY_STOCKS = ['VCB', 'VNM', 'HPG', 'VHM', 'GAS']
    
    def __init__(self,
                 bear_threshold=-0.03,
                 high_volatility_threshold=0.03,
                 trend_period=50):
        self.bear_threshold = bear_threshold
        self.high_volatility_threshold = high_volatility_threshold
        self.trend_period = trend_period
    
    def analyze_market_regime(self) -> Dict:
        """
        Phân tích market regime từ proxy stocks
        
        Returns:
            dict: Market regime info
        """
        print("📊 Phân tích thị trường từ blue-chip stocks...")
        
        proxy_data = []
        successful_loads = 0
        
        # Load data từ proxy stocks
        for symbol in self.PROXY_STOCKS:
            try:
                df = load_data(symbol, lookback=100)
                if df.empty or len(df) < 50:
                    continue
                
                proxy_data.append({
                    'symbol': symbol,
                    'df': df,
                    'latest': df.iloc[-1]
                })
                successful_loads += 1
                
            except Exception as e:
                logger.debug(f"Không load được {symbol}: {e}")
        
        if successful_loads < 2:
            print("⚠️ Không đủ data từ proxy stocks, dùng fallback")
            return self._fallback_regime()
        
        print(f"✅ Đã load {successful_loads}/{len(self.PROXY_STOCKS)} proxy stocks")
        
        # Aggregate metrics từ proxy stocks
        weekly_changes = []
        volatilities = []
        trends = []
        
        for proxy in proxy_data:
            df = proxy['df']
            
            # Weekly change
            if len(df) >= 6:
                weekly_change = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1)
                weekly_changes.append(weekly_change)
            
            # Volatility
            if len(df) >= 20:
                returns = df['close'].pct_change()
                volatility = returns.std()
                volatilities.append(volatility)
            
            # Trend (SMA20 vs SMA50)
            if len(df) >= 50:
                sma20 = df['close'].rolling(20).mean().iloc[-1]
                sma50 = df['close'].rolling(50).mean().iloc[-1]
                trend = 1 if sma20 > sma50 else -1
                trends.append(trend)
        
        # Aggregate
        avg_weekly_change = np.mean(weekly_changes) if weekly_changes else 0
        avg_volatility = np.mean(volatilities) if volatilities else 0.02
        avg_trend = np.mean(trends) if trends else 0
        
        # Determine regime
        regime = self._determine_regime(
            avg_weekly_change,
            avg_trend,
            avg_volatility
        )
        
        # Tradeable decision
        tradeable = self._is_tradeable(regime, avg_volatility, avg_weekly_change)
        
        # Confidence
        confidence = self._calculate_confidence(regime, avg_trend, avg_volatility)
        
        details = {
            'avg_weekly_change': avg_weekly_change * 100,  # %
            'avg_volatility': avg_volatility * 100,  # %
            'avg_trend': avg_trend,
            'num_stocks_analyzed': successful_loads,
            'proxy_stocks': [p['symbol'] for p in proxy_data]
        }
        
        result = {
            'regime': regime,
            'tradeable': tradeable,
            'confidence': confidence,
            'details': details,
            'message': self._generate_message(regime, tradeable, details)
        }
        
        logger.info(f"Market Regime (Proxy): {regime} | Tradeable: {tradeable}")
        return result
    
    def _determine_regime(self,
                         weekly_change: float,
                         avg_trend: float,
                         avg_volatility: float) -> str:
        """Xác định regime từ aggregate metrics"""
        
        # High volatility override
        if avg_volatility > self.high_volatility_threshold:
            return 'HIGH_VOLATILITY'
        
        # Bear market
        if weekly_change < self.bear_threshold:
            return 'BEAR'
        
        # Bull market
        if avg_trend > 0.5 and weekly_change > 0.01:
            return 'BULL'
        
        # Strong bear
        if avg_trend < -0.5 and weekly_change < 0:
            return 'BEAR'
        
        # Sideways
        return 'SIDEWAYS'
    
    def _is_tradeable(self,
                     regime: str,
                     volatility: float,
                     weekly_change: float) -> bool:
        """Quyết định có trade hay không"""
        
        if regime == 'BEAR':
            return False
        
        if regime == 'HIGH_VOLATILITY':
            return False
        
        if weekly_change < -0.05:  # -5% trong tuần
            return False
        
        if regime in ['BULL', 'SIDEWAYS']:
            return True
        
        return False
    
    def _calculate_confidence(self,
                             regime: str,
                             avg_trend: float,
                             volatility: float) -> int:
        """Tính confidence score"""
        
        base_confidence = {
            'BULL': 80,
            'SIDEWAYS': 50,
            'BEAR': 20,
            'HIGH_VOLATILITY': 10
        }
        
        confidence = base_confidence.get(regime, 50)
        
        # Adjust by trend strength
        if regime == 'BULL':
            confidence += int(abs(avg_trend) * 10)
        
        # Penalize high volatility
        if volatility > 0.025:
            confidence -= 20
        
        return int(max(0, min(confidence, 100)))
    
    def _generate_message(self, regime: str, tradeable: bool, details: Dict) -> str:
        """Generate message"""
        
        if not tradeable:
            if regime == 'BEAR':
                return f"⛔ THỊ TRƯỜNG GIẢM ĐIỂM - KHÔNG TRADE\n" \
                       f"📉 Tuần này: {details['avg_weekly_change']:+.2f}%"
            
            elif regime == 'HIGH_VOLATILITY':
                return f"⚠️ BIẾN ĐỘNG MẠNH - RỦI RO CAO\n" \
                       f"📊 Volatility: {details['avg_volatility']:.2f}%"
            
            else:
                return f"⏸️ THỊ TRƯỜNG KHÔNG RÕ HƯỚNG\n" \
                       f"📊 Regime: {regime}"
        
        else:
            if regime == 'BULL':
                return f"✅ THỊ TRƯỜNG TÍCH CỰC - CÓ THỂ TRADE\n" \
                       f"📈 Tuần này: {details['avg_weekly_change']:+.2f}%\n" \
                       f"🎯 Trend: {'UP' if details['avg_trend'] > 0 else 'DOWN'}"
            
            else:  # SIDEWAYS
                return f"⚡ THỊ TRƯỜNG DAO ĐỘNG - TRADE THẬN TRỌNG\n" \
                       f"📊 Tuần này: {details['avg_weekly_change']:+.2f}%\n" \
                       f"💡 Chọn mã tốt, position nhỏ"
    
    def _fallback_regime(self) -> Dict:
        """Fallback khi không có data"""
        return {
            'regime': 'SIDEWAYS',
            'tradeable': True,
            'confidence': 40,
            'details': {
                'note': 'Fallback regime - insufficient proxy data'
            },
            'message': '⚠️ Không đủ data\n💡 Giả định SIDEWAYS - trade thận trọng'
        }
    
    def get_position_multiplier(self) -> float:
        """Position multiplier dựa trên regime"""
        regime_info = self.analyze_market_regime()
        
        if not regime_info['tradeable']:
            return 0.0
        
        regime = regime_info['regime']
        confidence = regime_info['confidence']
        
        if regime == 'BULL':
            if confidence >= 80:
                return 1.2
            else:
                return 1.0
        
        elif regime == 'SIDEWAYS':
            return 0.7
        
        else:
            return 0.5


# ============================================================================
# HELPERS
# ============================================================================

def check_market_before_trading() -> Tuple[bool, str]:
    """Quick check trước khi trade"""
    analyzer = ProxyMarketRegimeAnalyzer()
    result = analyzer.analyze_market_regime()
    return result['tradeable'], result['message']


def get_market_position_adjustment() -> float:
    """Get position multiplier"""
    analyzer = ProxyMarketRegimeAnalyzer()
    return analyzer.get_position_multiplier()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING PROXY MARKET REGIME ANALYZER")
    print("="*70 + "\n")
    
    analyzer = ProxyMarketRegimeAnalyzer()
    result = analyzer.analyze_market_regime()
    
    print(f"\n📊 Regime: {result['regime']}")
    print(f"✅ Tradeable: {result['tradeable']}")
    print(f"🎯 Confidence: {result['confidence']}%")
    print(f"\n{result['message']}")
    print(f"\n📈 Details:")
    for key, value in result['details'].items():
        print(f"  • {key}: {value}")
    
    print(f"\n💰 Position Multiplier: {analyzer.get_position_multiplier():.2f}x")
    print("\n" + "="*70)
    
    print("\n✅ Proxy approach works even without VNINDEX!")