# -*- coding: utf-8 -*-
"""
Smart Caching System
Cache thông minh với TTL và invalidation
"""
import time
import pickle
import hashlib
import os
from datetime import datetime, timedelta
from typing import Any, Optional, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = 'smart_cache'
os.makedirs(CACHE_DIR, exist_ok=True)


class SmartCache:
    """
    Smart caching system với:
    - TTL (Time To Live)
    - Memory + Disk cache
    - Automatic cleanup
    - Cache statistics
    """
    
    def __init__(self):
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0
        }
    
    def get(self, key: str, ttl: int = 3600) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            Cached value or None if expired/not found
        """
        # Check memory cache first
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            if time.time() - timestamp < ttl:
                self.cache_stats['hits'] += 1
                return value
            else:
                # Expired
                del self.memory_cache[key]
        
        # Check disk cache
        cache_file = self._get_cache_file(key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    value, timestamp = pickle.load(f)
                
                if time.time() - timestamp < ttl:
                    # Load to memory
                    self.memory_cache[key] = (value, timestamp)
                    self.cache_stats['hits'] += 1
                    return value
                else:
                    # Expired, delete
                    os.remove(cache_file)
            except Exception as e:
                logger.error(f"Error loading cache {key}: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, save_to_disk: bool = True):
        """
        Set value to cache
        
        Args:
            key: Cache key
            value: Value to cache
            save_to_disk: Whether to save to disk
        """
        timestamp = time.time()
        
        # Save to memory
        self.memory_cache[key] = (value, timestamp)
        
        # Save to disk
        if save_to_disk:
            try:
                cache_file = self._get_cache_file(key)
                with open(cache_file, 'wb') as f:
                    pickle.dump((value, timestamp), f, protocol=pickle.HIGHEST_PROTOCOL)
                self.cache_stats['saves'] += 1
            except Exception as e:
                logger.error(f"Error saving cache {key}: {e}")
    
    def get_or_compute(self, key: str, compute_fn: Callable, ttl: int = 3600, 
                       save_to_disk: bool = True) -> Any:
        """
        Get from cache or compute if not found
        
        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: Time to live in seconds
            save_to_disk: Whether to save to disk
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache
        value = self.get(key, ttl)
        
        if value is not None:
            return value
        
        # Compute
        value = compute_fn()
        
        # Save to cache
        self.set(key, value, save_to_disk)
        
        return value
    
    def invalidate(self, key: str):
        """Invalidate cache entry"""
        # Remove from memory
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Remove from disk
        cache_file = self._get_cache_file(key)
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception as e:
                logger.error(f"Error removing cache {key}: {e}")
    
    def clear_all(self):
        """Clear all cache"""
        # Clear memory
        self.memory_cache.clear()
        
        # Clear disk
        try:
            for filename in os.listdir(CACHE_DIR):
                filepath = os.path.join(CACHE_DIR, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def cleanup_expired(self, ttl: int = 86400):
        """
        Cleanup expired cache entries
        
        Args:
            ttl: Time to live in seconds (default 24h)
        """
        current_time = time.time()
        
        # Cleanup memory
        expired_keys = [
            key for key, (_, timestamp) in self.memory_cache.items()
            if current_time - timestamp > ttl
        ]
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Cleanup disk
        try:
            for filename in os.listdir(CACHE_DIR):
                filepath = os.path.join(CACHE_DIR, filename)
                if os.path.isfile(filepath):
                    # Check file age
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > ttl:
                        os.remove(filepath)
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'saves': self.cache_stats['saves'],
            'hit_rate': hit_rate,
            'memory_entries': len(self.memory_cache)
        }
    
    def _get_cache_file(self, key: str) -> str:
        """Get cache file path for key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(CACHE_DIR, f"{key_hash}.cache")


# Singleton instance
_cache = None

def get_cache() -> SmartCache:
    """Get cache singleton"""
    global _cache
    if _cache is None:
        _cache = SmartCache()
    return _cache


# Decorator for caching function results
def cached(ttl: int = 3600, key_prefix: str = ""):
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        
    Example:
        @cached(ttl=1800, key_prefix="ml_signal")
        def analyze_signal(symbol):
            # expensive computation
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(cache_key, ttl)
            
            if result is not None:
                return result
            
            # Compute
            result = func(*args, **kwargs)
            
            # Save to cache
            cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


# Specialized caches for different data types
class MarketRegimeCache:
    """Cache for market regime (2 hours TTL)"""
    
    def __init__(self):
        self.cache = get_cache()
        self.ttl = 7200  # 2 hours
    
    def get_regime(self) -> Optional[dict]:
        """Get cached market regime"""
        return self.cache.get('market_regime', self.ttl)
    
    def set_regime(self, regime: dict):
        """Set market regime"""
        self.cache.set('market_regime', regime)
    
    def invalidate(self):
        """Invalidate market regime cache"""
        self.cache.invalidate('market_regime')


class MLSignalCache:
    """Cache for ML signals (1 hour TTL)"""
    
    def __init__(self):
        self.cache = get_cache()
        self.ttl = 3600  # 1 hour
    
    def get_signal(self, symbol: str) -> Optional[dict]:
        """Get cached ML signal"""
        return self.cache.get(f'ml_signal:{symbol}', self.ttl)
    
    def set_signal(self, symbol: str, signal: dict):
        """Set ML signal"""
        self.cache.set(f'ml_signal:{symbol}', signal)
    
    def invalidate(self, symbol: str):
        """Invalidate ML signal cache"""
        self.cache.invalidate(f'ml_signal:{symbol}')


class NewsCache:
    """Cache for news (30 minutes TTL)"""
    
    def __init__(self):
        self.cache = get_cache()
        self.ttl = 1800  # 30 minutes
    
    def get_news(self, symbol: str) -> Optional[dict]:
        """Get cached news"""
        return self.cache.get(f'news:{symbol}', self.ttl)
    
    def set_news(self, symbol: str, news: dict):
        """Set news"""
        self.cache.set(f'news:{symbol}', news)
    
    def invalidate(self, symbol: str):
        """Invalidate news cache"""
        self.cache.invalidate(f'news:{symbol}')


# Test
if __name__ == "__main__":
    print("Testing Smart Cache...")
    
    cache = SmartCache()
    
    # Test basic get/set
    cache.set('test_key', {'value': 123})
    result = cache.get('test_key')
    print(f"Get result: {result}")
    
    # Test get_or_compute
    def expensive_computation():
        print("Computing...")
        time.sleep(1)
        return {'computed': True}
    
    result1 = cache.get_or_compute('compute_key', expensive_computation, ttl=10)
    print(f"First call: {result1}")
    
    result2 = cache.get_or_compute('compute_key', expensive_computation, ttl=10)
    print(f"Second call (cached): {result2}")
    
    # Test stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Test decorator
    @cached(ttl=5, key_prefix="test")
    def test_function(x, y):
        print(f"Computing {x} + {y}")
        return x + y
    
    print(f"Result 1: {test_function(1, 2)}")
    print(f"Result 2 (cached): {test_function(1, 2)}")
    
    print("\n✅ Cache test completed!")
