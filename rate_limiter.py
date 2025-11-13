"""
Rate Limiter for API calls
Prevents hitting API rate limits
"""
import time
from threading import Lock
from functools import wraps


class RateLimiter:
    """
    Simple rate limiter using token bucket algorithm
    
    Usage:
        limiter = RateLimiter(calls_per_second=10)
        
        @limiter.limit
        def api_call():
            return requests.get(url)
    """
    
    def __init__(self, calls_per_second=10):
        """
        Args:
            calls_per_second: Maximum number of calls per second
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self.lock = Lock()
    
    def wait(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_call = time.time()
    
    def limit(self, func):
        """Decorator to rate limit a function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait()
            return func(*args, **kwargs)
        return wrapper


# Global rate limiters for different APIs
tcbs_limiter = RateLimiter(calls_per_second=10)  # TCBS API: 10 calls/sec
yahoo_limiter = RateLimiter(calls_per_second=5)  # Yahoo Finance: 5 calls/sec


# Test
if __name__ == "__main__":
    import requests
    
    print("Testing rate limiter...")
    
    @tcbs_limiter.limit
    def test_call(i):
        print(f"Call {i} at {time.time():.2f}")
        return i
    
    start = time.time()
    for i in range(15):
        test_call(i)
    
    elapsed = time.time() - start
    print(f"\n15 calls took {elapsed:.2f}s")
    print(f"Expected: ~1.5s (10 calls/sec)")
    print(f"Rate: {15/elapsed:.1f} calls/sec")
