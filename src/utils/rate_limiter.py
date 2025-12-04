"""
Rate Limiter for API calls
Prevents hitting API rate limits
"""

import random
import time
from functools import wraps
from threading import Lock


class RateLimiter:
    """
    Simple rate limiter using token bucket algorithm

    Usage:
        limiter = RateLimiter(calls_per_second=10)

        @limiter.limit
        def api_call():
            return requests.get(url)
    """

    def __init__(self, calls_per_second=10, jitter=0.2):
        """
        Args:
            calls_per_second: Maximum number of calls per second
            jitter: Random delay factor (0-1) to avoid pattern detection
        """
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.jitter = jitter
        self.last_call = 0
        self.lock = Lock()

    def wait(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call

            # Add random jitter to avoid pattern detection
            jitter_delay = random.uniform(0, self.min_interval * self.jitter)

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last + jitter_delay
                time.sleep(sleep_time)
            elif jitter_delay > 0:
                time.sleep(jitter_delay)

            self.last_call = time.time()

    def limit(self, func):
        """Decorator to rate limit a function"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait()
            return func(*args, **kwargs)

        return wrapper


# Global rate limiters for different APIs
# TCBS API: 1 call/sec (rất conservative để tránh 429)
tcbs_limiter = RateLimiter(calls_per_second=1)
yahoo_limiter = RateLimiter(calls_per_second=5)  # Yahoo Finance: 5 calls/sec


# Test
if __name__ == "__main__":
    pass

    print("Testing rate limiter...")

    @tcbs_limiter.limit
    def test_call(i):
        print(f"Call {i} at {time.time():.2f}")
        return i

    start = time.time()
    for i in range(15):
        test_call(i)

    elapsed = time.time() - start
    print("\n15 calls took {elapsed:.2f}s")
    print("Expected: ~1.5s (10 calls/sec)")
    print("Rate: {15/elapsed:.1f} calls/sec")
