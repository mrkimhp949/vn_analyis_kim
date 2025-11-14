# -*- coding: utf-8 -*-
"""
Parallel Scanner
Quét nhiều mã cổ phiếu đồng thời để tăng tốc độ
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Kết quả scan một mã"""
    symbol: str
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    duration: float = 0.0


class ParallelScanner:
    """
    Scanner song song với ThreadPoolExecutor
    
    Features:
    - Scan nhiều mã đồng thời
    - Error handling cho từng mã
    - Progress tracking
    - Timeout protection
    """
    
    def __init__(self, max_workers: int = 5, timeout: int = 30):
        """
        Args:
            max_workers: Số threads tối đa
            timeout: Timeout cho mỗi task (seconds)
        """
        self.max_workers = max_workers
        self.timeout = timeout
    
    def scan_symbols(
        self,
        symbols: List[str],
        scan_function: Callable[[str], Dict],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ScanResult]:
        """
        Scan nhiều symbols song song
        
        Args:
            symbols: List các mã cần scan
            scan_function: Function để scan 1 mã, nhận symbol và trả về Dict
            progress_callback: Callback để báo tiến độ (current, total)
            
        Returns:
            List of ScanResult
        """
        results = []
        completed = 0
        total = len(symbols)
        
        print(f"⚡ Parallel scanning {total} symbols với {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._scan_single, symbol, scan_function): symbol
                for symbol in symbols
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                
                try:
                    result = future.result(timeout=self.timeout)
                    results.append(result)
                except TimeoutError:
                    logger.error(f"Timeout scanning {symbol}")
                    results.append(ScanResult(
                        symbol=symbol,
                        success=False,
                        error="Timeout"
                    ))
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    results.append(ScanResult(
                        symbol=symbol,
                        success=False,
                        error=str(e)
                    ))
                
                # Update progress
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                
                # Print progress
                if completed % 10 == 0 or completed == total:
                    success_count = sum(1 for r in results if r.success)
                    print(f"   Progress: {completed}/{total} ({success_count} success)")
        
        return results
    
    def _scan_single(self, symbol: str, scan_function: Callable) -> ScanResult:
        """Scan một mã với error handling"""
        start_time = time.time()
        
        try:
            data = scan_function(symbol)
            duration = time.time() - start_time
            
            return ScanResult(
                symbol=symbol,
                success=True,
                data=data,
                duration=duration
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error scanning {symbol}: {e}")
            
            return ScanResult(
                symbol=symbol,
                success=False,
                error=str(e),
                duration=duration
            )
    
    def get_summary(self, results: List[ScanResult]) -> Dict:
        """Tổng kết kết quả scan"""
        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success
        
        total_duration = sum(r.duration for r in results)
        avg_duration = total_duration / total if total > 0 else 0
        
        # Get error types
        error_types = {}
        for r in results:
            if not r.success and r.error:
                error_type = r.error.split(':')[0] if ':' in r.error else r.error
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total': total,
            'success': success,
            'failed': failed,
            'success_rate': (success / total * 100) if total > 0 else 0,
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'error_types': error_types
        }


# Specialized scanners
class MLSignalScanner(ParallelScanner):
    """Scanner chuyên cho ML signals"""
    
    def __init__(self, ml_generator, max_workers: int = 5):
        super().__init__(max_workers=max_workers)
        self.ml_generator = ml_generator
    
    def scan_for_signals(
        self,
        symbols: List[str],
        min_confidence: float = 60.0
    ) -> List[Dict]:
        """
        Scan ML signals cho nhiều symbols
        
        Returns:
            List of signals với confidence >= min_confidence
        """
        from data_loader import load_data
        from config import LOOKBACK
        
        def scan_function(symbol: str) -> Dict:
            # Load data
            df = load_data(symbol, lookback=LOOKBACK)
            
            if df.empty or len(df) < 50:
                raise ValueError("Insufficient data")
            
            # Analyze
            signal = self.ml_generator.analyze(df)
            
            return {
                'symbol': symbol,
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'price': signal['price'],
                'reason': signal['reason']
            }
        
        # Scan
        results = self.scan_symbols(symbols, scan_function)
        
        # Filter by confidence
        signals = [
            r.data for r in results
            if r.success and r.data and r.data['confidence'] >= min_confidence
        ]
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals


class EntrySignalScanner(ParallelScanner):
    """Scanner chuyên cho entry signals"""
    
    def __init__(
        self,
        ml_generator,
        entry_logic,
        market_regime,
        max_workers: int = 5
    ):
        super().__init__(max_workers=max_workers)
        self.ml_generator = ml_generator
        self.entry_logic = entry_logic
        self.market_regime = market_regime
    
    def scan_for_entries(
        self,
        symbols: List[str],
        existing_symbols: set = None
    ) -> List[Dict]:
        """
        Scan entry signals cho nhiều symbols
        
        Returns:
            List of entry opportunities
        """
        from data_loader import load_data
        from config import LOOKBACK
        
        existing_symbols = existing_symbols or set()
        
        def scan_function(symbol: str) -> Dict:
            # Skip if already have position
            if symbol in existing_symbols:
                raise ValueError("Already have position")
            
            # Load data
            df = load_data(symbol, lookback=LOOKBACK)
            
            if df.empty or len(df) < 50:
                raise ValueError("Insufficient data")
            
            # Get ML signal
            ml_signal = self.ml_generator.analyze(df)
            
            # Check entry
            entry_signal = self.entry_logic.analyze_entry(
                df=df,
                ml_signal=ml_signal,
                market_regime=self.market_regime
            )
            
            if not entry_signal.should_enter:
                raise ValueError("No entry signal")
            
            return {
                'symbol': symbol,
                'confidence': entry_signal.confidence,
                'strength': entry_signal.strength.name,
                'entry_price': entry_signal.entry_price,
                'stop_loss': entry_signal.stop_loss,
                'take_profit_targets': entry_signal.take_profit_targets,
                'reasons': entry_signal.reasons
            }
        
        # Scan
        results = self.scan_symbols(symbols, scan_function)
        
        # Get successful entries
        entries = [r.data for r in results if r.success and r.data]
        
        # Sort by confidence
        entries.sort(key=lambda x: x['confidence'], reverse=True)
        
        return entries


# Test
if __name__ == "__main__":
    print("Testing Parallel Scanner...")
    
    # Test basic scanner
    def test_scan_function(symbol: str) -> Dict:
        time.sleep(0.1)  # Simulate work
        return {'symbol': symbol, 'value': len(symbol)}
    
    scanner = ParallelScanner(max_workers=3)
    test_symbols = ['VCB', 'FPT', 'VNM', 'HPG', 'SSI', 'ACB', 'VIC', 'VHM']
    
    results = scanner.scan_symbols(test_symbols, test_scan_function)
    
    summary = scanner.get_summary(results)
    print(f"Summary: {summary}")
    
    print("\n✅ Test completed!")
