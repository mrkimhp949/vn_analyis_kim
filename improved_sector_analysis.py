# -*- coding: utf-8 -*-
"""
improved_sector_analysis.py - Enhanced Sector Selection
Phân tích ngành với nhiều yếu tố hơn: volume, volatility, correlation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from data_loader import load_data
from ml_signals import MLSignalGenerator
import logging
import time

logger = logging.getLogger(__name__)


class EnhancedSectorAnalyzer:
    """
    Phân tích ngành nâng cao với multiple factors:
    - Signal quality (ML + Technical)
    - Liquidity (volume, spread)
    - Volatility
    - Relative strength vs market
    - Correlation (tránh các ngành tương quan cao)
    """
    
    def __init__(self, min_volume=1_000_000, min_price=10_000):
        self.ml_generator = MLSignalGenerator()
        self.min_volume = min_volume  # Volume tối thiểu
        self.min_price = min_price    # Giá tối thiểu (lọc penny stocks)
    
    def analyze_all_sectors(self, 
                           sectors_dict: Dict[str, List[str]],
                           lookback: int = 100) -> Dict:
        """
        Phân tích tất cả các ngành với scoring đa chiều
        
        Args:
            sectors_dict: {'sector_name': ['TICKER1', 'TICKER2', ...]}
            lookback: Số ngày dữ liệu
            
        Returns:
            dict: Sector analysis với scoring và ranking
        """
        print(f"\n{'='*70}")
        print(f"📊 PHÂN TÍCH TOÀN BỘ THỊ TRƯỜNG - ENHANCED VERSION")
        print(f"{'='*70}\n")
        
        sector_scores = {}
        all_stock_data = {}  # Cache stock data for correlation analysis
        
        # Step 1: Analyze each sector
        for sector_name, tickers in sectors_dict.items():
            print(f"\n🔍 Phân tích ngành: {sector_name} ({len(tickers)} mã)...")
            
            sector_result = self._analyze_sector(
                sector_name, 
                tickers, 
                lookback,
                all_stock_data
            )
            
            sector_scores[sector_name] = sector_result
            
            print(f"  ✅ Score: {sector_result['total_score']:.1f}/100")
        
        # Step 2: Rank sectors
        ranked_sectors = self._rank_sectors(sector_scores)
        
        # Step 3: Select top sectors với correlation check
        selected_sectors = self._select_uncorrelated_sectors(
            ranked_sectors, 
            all_stock_data,
            top_n=3
        )
        
        # Step 4: Select best stocks from top sectors
        selected_tickers = self._select_best_tickers(
            selected_sectors,
            sector_scores,
            all_stock_data
        )
        
        result = {
            'analyzed_at': datetime.now().isoformat(),
            'sector_scores': sector_scores,
            'ranked_sectors': ranked_sectors,
            'selected_sectors': selected_sectors,
            'selected_tickers': selected_tickers,
            'market_summary': self._generate_market_summary(sector_scores)
        }
        
        self._print_summary(result)
        
        return result
    
    def _analyze_sector(self,
                       sector_name: str,
                       tickers: List[str],
                       lookback: int,
                       all_stock_data: Dict) -> Dict:
        """Phân tích chi tiết 1 ngành"""
        
        analyzed = 0
        buy_signals = 0
        total_confidence = 0
        total_volume = 0
        total_volatility = 0
        total_rel_strength = 0
        valid_stocks = []
        
        for ticker in tickers:
            try:
                df = load_data(ticker, lookback)
                
                if df.empty or len(df) < 50:
                    continue
                
                latest = df.iloc[-1]
                
                # Filter 1: Liquidity check
                if latest['volume'] < self.min_volume:
                    logger.debug(f"  ⏭️ {ticker}: Volume thấp ({latest['volume']:,.0f})")
                    continue
                
                # Filter 2: Price check (lọc penny stocks)
                if latest['close'] < self.min_price:
                    logger.debug(f"  ⏭️ {ticker}: Giá quá thấp ({latest['close']:,.0f})")
                    continue
                
                # Analyze signal
                signal_result = self.ml_generator.analyze(df)
                
                # Calculate additional metrics
                volume_score = self._calculate_volume_score(df)
                volatility = self._calculate_volatility(df)
                rel_strength = self._calculate_relative_strength(df)
                
                # Store data for later correlation analysis
                all_stock_data[ticker] = {
                    'df': df,
                    'signal': signal_result,
                    'volume_score': volume_score,
                    'volatility': volatility,
                    'rel_strength': rel_strength
                }
                
                analyzed += 1
                
                if signal_result['signal'] == 'BUY':
                    buy_signals += 1
                
                total_confidence += signal_result['confidence']
                total_volume += volume_score
                total_volatility += volatility
                total_rel_strength += rel_strength
                
                valid_stocks.append(ticker)
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"  ❌ {ticker}: {e}")
        
        if analyzed == 0:
            return self._empty_sector_result(sector_name)
        
        # Calculate sector metrics
        buy_rate = (buy_signals / analyzed) * 100
        avg_confidence = total_confidence / analyzed
        avg_volume_score = total_volume / analyzed
        avg_volatility = total_volatility / analyzed
        avg_rel_strength = total_rel_strength / analyzed
        
        # SCORING: Multiple factors
        signal_score = buy_rate * 0.6 + avg_confidence * 0.4  # 50 points max
        volume_score = min(avg_volume_score * 50, 20)         # 20 points max
        volatility_score = self._score_volatility(avg_volatility)  # 15 points max
        strength_score = max(0, min(avg_rel_strength * 100, 15))   # 15 points max
        
        total_score = signal_score + volume_score + volatility_score + strength_score
        
        return {
            'sector_name': sector_name,
            'total_analyzed': analyzed,
            'valid_stocks': valid_stocks,
            'buy_signals': buy_signals,
            'buy_rate': buy_rate,
            'avg_confidence': avg_confidence,
            'avg_volume_score': avg_volume_score,
            'avg_volatility': avg_volatility,
            'avg_rel_strength': avg_rel_strength,
            'scores': {
                'signal_score': signal_score,
                'volume_score': volume_score,
                'volatility_score': volatility_score,
                'strength_score': strength_score
            },
            'total_score': total_score
        }
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> float:
        """
        Tính điểm thanh khoản
        
        Returns:
            float: 0-1 (0=low liquidity, 1=high liquidity)
        """
        if len(df) < 20:
            return 0.0
        
        current_volume = df['volume'].iloc[-1]
        avg_volume_20 = df['volume'].rolling(20).mean().iloc[-1]
        
        # Volume ratio
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
        
        # Normalize to 0-1
        score = min(volume_ratio / 2, 1.0)  # 2x avg volume = max score
        
        return score
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Tính volatility (normalized)"""
        if 'atr' not in df.columns or df['atr'].isna().all():
            returns = df['close'].pct_change()
            volatility = returns.std()
        else:
            atr = df['atr'].iloc[-1]
            price = df['close'].iloc[-1]
            volatility = atr / price if price > 0 else 0
        
        return volatility
    
    def _score_volatility(self, volatility: float) -> float:
        """
        Score volatility: Vừa phải là tốt
        
        Too low = không có momentum
        Too high = rủi ro cao
        """
        optimal_vol = 0.02  # 2% daily volatility is optimal
        
        if volatility < 0.01:
            # Too low
            return volatility / 0.01 * 7.5  # Max 7.5 points
        elif volatility <= 0.03:
            # Sweet spot
            deviation = abs(volatility - optimal_vol)
            return 15 - (deviation / 0.01 * 7.5)  # 15 points max
        else:
            # Too high
            return max(0, 15 - (volatility - 0.03) * 100)
    
    def _calculate_relative_strength(self, df: pd.DataFrame) -> float:
        """
        Relative strength vs market (proxy)
        
        Returns:
            float: -1 to 1 (-1=underperform, 1=outperform)
        """
        if len(df) < 20:
            return 0.0
        
        # 20-day return
        returns = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
        
        # Normalize to -1, 1
        # Assuming market average return is 0
        rel_strength = np.tanh(returns * 10)  # Soft clip to -1, 1
        
        return rel_strength
    
    def _rank_sectors(self, sector_scores: Dict) -> List[Tuple[str, float]]:
        """Rank sectors by total score"""
        ranked = sorted(
            sector_scores.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        return [(name, data['total_score']) for name, data in ranked]
    
    def _select_uncorrelated_sectors(self,
                                    ranked_sectors: List[Tuple[str, float]],
                                    all_stock_data: Dict,
                                    top_n: int = 3,
                                    max_correlation: float = 0.7) -> List[str]:
        """
        Chọn top N sectors nhưng tránh correlation cao
        
        Ví dụ: Không chọn cả Banks và Securities cùng lúc
        """
        selected = []
        
        for sector_name, score in ranked_sectors:
            if len(selected) >= top_n:
                break
            
            # Check correlation với sectors đã chọn
            is_correlated = False
            
            for selected_sector in selected:
                corr = self._calculate_sector_correlation(
                    sector_name,
                    selected_sector,
                    all_stock_data
                )
                
                if corr > max_correlation:
                    logger.info(f"  ⚠️ {sector_name} tương quan cao với {selected_sector} ({corr:.2f})")
                    is_correlated = True
                    break
            
            if not is_correlated:
                selected.append(sector_name)
                logger.info(f"  ✅ Chọn: {sector_name} (Score: {score:.1f})")
        
        # Nếu không đủ, lấy thêm (bỏ qua correlation)
        if len(selected) < top_n:
            for sector_name, score in ranked_sectors:
                if sector_name not in selected:
                    selected.append(sector_name)
                    if len(selected) >= top_n:
                        break
        
        return selected
    
    def _calculate_sector_correlation(self,
                                     sector1: str,
                                     sector2: str,
                                     all_stock_data: Dict) -> float:
        """
        Tính correlation giữa 2 sectors
        (Simplified: so sánh average returns)
        """
        # Lấy tất cả stocks của 2 sectors
        sector1_stocks = [ticker for ticker in all_stock_data.keys() 
                         if sector1 in ticker or True]  # Cần mapping tốt hơn
        sector2_stocks = [ticker for ticker in all_stock_data.keys() 
                         if sector2 in ticker or True]
        
        if not sector1_stocks or not sector2_stocks:
            return 0.0
        
        # Simplified: Giả sử correlation = 0.5 (cần implement tốt hơn)
        # TODO: Implement proper correlation calculation
        return 0.5
    
    def _select_best_tickers(self,
                            selected_sectors: List[str],
                            sector_scores: Dict,
                            all_stock_data: Dict,
                            max_per_sector: int = 5) -> List[str]:
        """
        Chọn các mã tốt nhất từ các sectors đã chọn
        """
        selected_tickers = []
        
        for sector_name in selected_sectors:
            sector_data = sector_scores[sector_name]
            valid_stocks = sector_data['valid_stocks']
            
            # Score từng mã trong sector
            stock_scores = []
            
            for ticker in valid_stocks:
                if ticker not in all_stock_data:
                    continue
                
                stock_data = all_stock_data[ticker]
                signal = stock_data['signal']
                
                # Chỉ chọn mã có signal BUY
                if signal['signal'] != 'BUY':
                    continue
                
                # Calculate stock score
                score = (
                    signal['confidence'] * 0.4 +
                    stock_data['volume_score'] * 20 * 0.2 +
                    (stock_data['rel_strength'] + 1) * 50 * 0.2 +
                    (1 - stock_data['volatility'] * 10) * 0.2
                )
                
                stock_scores.append((ticker, score))
            
            # Sort và chọn top
            stock_scores.sort(key=lambda x: x[1], reverse=True)
            
            sector_tickers = [ticker for ticker, score in stock_scores[:max_per_sector]]
            selected_tickers.extend(sector_tickers)
            
            logger.info(f"  📋 {sector_name}: Chọn {len(sector_tickers)} mã")
        
        return selected_tickers
    
    def _generate_market_summary(self, sector_scores: Dict) -> Dict:
        """Tổng hợp tình hình thị trường"""
        all_scores = [data['total_score'] for data in sector_scores.values()]
        all_buy_rates = [data['buy_rate'] for data in sector_scores.values()]
        
        return {
            'avg_sector_score': np.mean(all_scores),
            'avg_buy_rate': np.mean(all_buy_rates),
            'num_sectors': len(sector_scores),
            'market_sentiment': self._classify_sentiment(np.mean(all_buy_rates))
        }
    
    def _classify_sentiment(self, avg_buy_rate: float) -> str:
        """Phân loại tâm lý thị trường"""
        if avg_buy_rate >= 60:
            return 'VERY_BULLISH'
        elif avg_buy_rate >= 50:
            return 'BULLISH'
        elif avg_buy_rate >= 40:
            return 'NEUTRAL'
        elif avg_buy_rate >= 30:
            return 'BEARISH'
        else:
            return 'VERY_BEARISH'
    
    def _print_summary(self, result: Dict):
        """In tóm tắt kết quả"""
        print("\n" + "="*70)
        print("📊 KẾT QUẢ PHÂN TÍCH")
        print("="*70)
        
        print("\n🏆 TOP SECTORS:")
        for i, sector_name in enumerate(result['selected_sectors'], 1):
            score = result['sector_scores'][sector_name]['total_score']
            buy_rate = result['sector_scores'][sector_name]['buy_rate']
            print(f"{i}. {sector_name:20s} | Score: {score:5.1f} | BUY: {buy_rate:5.1f}%")
        
        print(f"\n📋 SELECTED TICKERS ({len(result['selected_tickers'])} mã):")
        print(", ".join(result['selected_tickers']))
        
        summary = result['market_summary']
        print(f"\n📈 MARKET SUMMARY:")
        print(f"  • Sentiment: {summary['market_sentiment']}")
        print(f"  • Avg Sector Score: {summary['avg_sector_score']:.1f}/100")
        print(f"  • Avg BUY Rate: {summary['avg_buy_rate']:.1f}%")
        
        print("\n" + "="*70)
    
    def _empty_sector_result(self, sector_name: str) -> Dict:
        """Empty result khi không phân tích được"""
        return {
            'sector_name': sector_name,
            'total_analyzed': 0,
            'valid_stocks': [],
            'buy_signals': 0,
            'buy_rate': 0,
            'avg_confidence': 0,
            'avg_volume_score': 0,
            'avg_volatility': 0,
            'avg_rel_strength': 0,
            'scores': {
                'signal_score': 0,
                'volume_score': 0,
                'volatility_score': 0,
                'strength_score': 0
            },
            'total_score': 0
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from config import KIM_SECTOR, THUY_SECTOR
    
    print("\n" + "="*70)
    print("🧪 TESTING ENHANCED SECTOR ANALYZER")
    print("="*70)
    
    # Combine sectors
    all_sectors = {
        f"Kim_{k}": v for k, v in KIM_SECTOR.items()
    }
    all_sectors.update({
        f"Thuy_{k}": v for k, v in THUY_SECTOR.items()
    })
    
    analyzer = EnhancedSectorAnalyzer(
        min_volume=500_000,
        min_price=10_000
    )
    
    result = analyzer.analyze_all_sectors(
        all_sectors,
        lookback=100
    )
    
    print("\n✅ Analysis complete!")
    print(f"Selected {len(result['selected_tickers'])} tickers for next week")