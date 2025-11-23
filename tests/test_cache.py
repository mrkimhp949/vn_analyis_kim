"""
Unit tests for src/data/cache.py - Smart Caching System
"""

import os
import pytest
import pickle
import time
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
from src.data.cache import (
    SmartCache,
    get_cache,
    cached,
    MarketRegimeCache,
    MLSignalCache,
    NewsCache,
    CACHE_DIR,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory for testing"""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()

    # Patch CACHE_DIR to use temp directory
    with patch("src.data.cache.CACHE_DIR", str(cache_dir)):
        yield str(cache_dir)

    # Cleanup
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


@pytest.fixture
def smart_cache(temp_cache_dir):
    """Create SmartCache instance with temp directory"""
    cache = SmartCache()
    yield cache
    # Cleanup
    cache.clear_all()


@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return {"value": 123, "name": "test", "nested": {"data": [1, 2, 3]}}


# ============================================================================
# SMARTCACHE INITIALIZATION TESTS
# ============================================================================


def test_smart_cache_init():
    """Test SmartCache initialization"""
    cache = SmartCache()

    assert cache.memory_cache == {}
    assert cache.cache_stats == {"hits": 0, "misses": 0, "saves": 0}


def test_smart_cache_init_multiple_instances():
    """Test that multiple instances don't share state"""
    cache1 = SmartCache()
    cache2 = SmartCache()

    cache1.set("key1", "value1", save_to_disk=False)

    # cache2 should not have key1 in memory
    assert "key1" in cache1.memory_cache
    assert "key1" not in cache2.memory_cache


# ============================================================================
# GET/SET TESTS
# ============================================================================


def test_set_and_get_memory_only(smart_cache, sample_data):
    """Test set and get with memory cache only"""
    smart_cache.set("test_key", sample_data, save_to_disk=False)

    result = smart_cache.get("test_key")

    assert result == sample_data
    assert smart_cache.cache_stats["hits"] == 1
    assert smart_cache.cache_stats["misses"] == 0


def test_set_and_get_with_disk(smart_cache, sample_data, temp_cache_dir):
    """Test set and get with disk cache"""
    smart_cache.set("test_key", sample_data, save_to_disk=True)

    # Check that file was created
    cache_file = smart_cache._get_cache_file("test_key")
    assert os.path.exists(cache_file)

    result = smart_cache.get("test_key")

    assert result == sample_data
    assert smart_cache.cache_stats["saves"] == 1


def test_get_nonexistent_key(smart_cache):
    """Test get with nonexistent key"""
    result = smart_cache.get("nonexistent_key")

    assert result is None
    assert smart_cache.cache_stats["misses"] == 1


def test_get_expired_memory_cache(smart_cache, sample_data):
    """Test get with expired memory cache"""
    with patch("time.time") as mock_time:
        # Set cache at time 0
        mock_time.return_value = 0
        smart_cache.set("test_key", sample_data, save_to_disk=False)

        # Try to get at time 7200 (2 hours later) with TTL of 3600 (1 hour)
        mock_time.return_value = 7200
        result = smart_cache.get("test_key", ttl=3600)

        assert result is None
        assert "test_key" not in smart_cache.memory_cache
        assert smart_cache.cache_stats["misses"] == 1


def test_get_expired_disk_cache(smart_cache, sample_data, temp_cache_dir):
    """Test get with expired disk cache"""
    with patch("time.time") as mock_time:
        # Set cache at time 0
        mock_time.return_value = 0
        smart_cache.set("test_key", sample_data, save_to_disk=True)
        cache_file = smart_cache._get_cache_file("test_key")

        # Clear memory cache
        smart_cache.memory_cache.clear()

        # Try to get at time 7200 (2 hours later) with TTL of 3600 (1 hour)
        mock_time.return_value = 7200
        result = smart_cache.get("test_key", ttl=3600)

        assert result is None
        assert not os.path.exists(cache_file)  # File should be deleted


def test_get_from_disk_loads_to_memory(smart_cache, sample_data, temp_cache_dir):
    """Test that getting from disk loads value to memory"""
    smart_cache.set("test_key", sample_data, save_to_disk=True)

    # Clear memory cache
    smart_cache.memory_cache.clear()

    # Get should load from disk to memory
    result = smart_cache.get("test_key")

    assert result == sample_data
    assert "test_key" in smart_cache.memory_cache


def test_set_different_types(smart_cache):
    """Test set with different data types"""
    test_cases = [
        ("string", "hello world"),
        ("int", 42),
        ("float", 3.14),
        ("list", [1, 2, 3]),
        ("dict", {"a": 1, "b": 2}),
        ("tuple", (1, 2, 3)),
        ("none", None),
        ("bool", True),
    ]

    for key, value in test_cases:
        smart_cache.set(key, value, save_to_disk=False)
        result = smart_cache.get(key)
        assert result == value, f"Failed for type {type(value)}"


# ============================================================================
# GET_OR_COMPUTE TESTS
# ============================================================================


def test_get_or_compute_cache_miss(smart_cache):
    """Test get_or_compute when value not in cache"""
    compute_fn = Mock(return_value={"computed": True})

    result = smart_cache.get_or_compute("compute_key", compute_fn, ttl=3600)

    assert result == {"computed": True}
    compute_fn.assert_called_once()
    assert smart_cache.cache_stats["misses"] == 1


def test_get_or_compute_cache_hit(smart_cache, sample_data):
    """Test get_or_compute when value is in cache"""
    smart_cache.set("compute_key", sample_data, save_to_disk=False)

    compute_fn = Mock(return_value={"should_not_compute": True})

    result = smart_cache.get_or_compute("compute_key", compute_fn, ttl=3600)

    assert result == sample_data
    compute_fn.assert_not_called()
    assert smart_cache.cache_stats["hits"] == 1


def test_get_or_compute_with_disk(smart_cache, temp_cache_dir):
    """Test get_or_compute saves to disk"""
    compute_fn = Mock(return_value={"computed": True})

    result = smart_cache.get_or_compute("compute_key", compute_fn, save_to_disk=True)

    cache_file = smart_cache._get_cache_file("compute_key")
    assert os.path.exists(cache_file)


def test_get_or_compute_without_disk(smart_cache, temp_cache_dir):
    """Test get_or_compute without disk save"""
    compute_fn = Mock(return_value={"computed": True})

    result = smart_cache.get_or_compute("compute_key", compute_fn, save_to_disk=False)

    cache_file = smart_cache._get_cache_file("compute_key")
    assert not os.path.exists(cache_file)


# ============================================================================
# INVALIDATE TESTS
# ============================================================================


def test_invalidate_memory(smart_cache, sample_data):
    """Test invalidate removes from memory"""
    smart_cache.set("test_key", sample_data, save_to_disk=False)

    smart_cache.invalidate("test_key")

    assert "test_key" not in smart_cache.memory_cache
    assert smart_cache.get("test_key") is None


def test_invalidate_disk(smart_cache, sample_data, temp_cache_dir):
    """Test invalidate removes from disk"""
    smart_cache.set("test_key", sample_data, save_to_disk=True)
    cache_file = smart_cache._get_cache_file("test_key")

    smart_cache.invalidate("test_key")

    assert not os.path.exists(cache_file)


def test_invalidate_both(smart_cache, sample_data, temp_cache_dir):
    """Test invalidate removes from both memory and disk"""
    smart_cache.set("test_key", sample_data, save_to_disk=True)

    smart_cache.invalidate("test_key")

    assert "test_key" not in smart_cache.memory_cache
    cache_file = smart_cache._get_cache_file("test_key")
    assert not os.path.exists(cache_file)


def test_invalidate_nonexistent(smart_cache):
    """Test invalidate with nonexistent key doesn't crash"""
    # Should not raise exception
    smart_cache.invalidate("nonexistent_key")


# ============================================================================
# CLEAR_ALL TESTS
# ============================================================================


def test_clear_all_memory(smart_cache):
    """Test clear_all clears memory cache"""
    smart_cache.set("key1", "value1", save_to_disk=False)
    smart_cache.set("key2", "value2", save_to_disk=False)

    smart_cache.clear_all()

    assert len(smart_cache.memory_cache) == 0


def test_clear_all_disk(smart_cache, temp_cache_dir):
    """Test clear_all clears disk cache"""
    smart_cache.set("key1", "value1", save_to_disk=True)
    smart_cache.set("key2", "value2", save_to_disk=True)

    smart_cache.clear_all()

    # Check that cache directory is empty
    files = os.listdir(temp_cache_dir)
    assert len(files) == 0


def test_clear_all_both(smart_cache, temp_cache_dir):
    """Test clear_all clears both memory and disk"""
    smart_cache.set("key1", "value1", save_to_disk=True)
    smart_cache.set("key2", "value2", save_to_disk=True)

    smart_cache.clear_all()

    assert len(smart_cache.memory_cache) == 0
    files = os.listdir(temp_cache_dir)
    assert len(files) == 0


# ============================================================================
# CLEANUP_EXPIRED TESTS
# ============================================================================


def test_cleanup_expired_memory(smart_cache):
    """Test cleanup_expired removes expired entries from memory"""
    with patch("time.time") as mock_time:
        # Add entries at different times
        mock_time.return_value = 0
        smart_cache.set("old_key", "old_value", save_to_disk=False)

        mock_time.return_value = 100000
        smart_cache.set("new_key", "new_value", save_to_disk=False)

        # Cleanup with TTL of 50000
        mock_time.return_value = 100000
        smart_cache.cleanup_expired(ttl=50000)

        assert "old_key" not in smart_cache.memory_cache
        assert "new_key" in smart_cache.memory_cache


def test_cleanup_expired_disk(smart_cache, temp_cache_dir):
    """Test cleanup_expired removes expired files from disk"""
    with patch("time.time") as mock_time:
        # Create old file
        mock_time.return_value = 0
        smart_cache.set("old_key", "old_value", save_to_disk=True)
        old_file = smart_cache._get_cache_file("old_key")

        # Create new file
        mock_time.return_value = 100000
        smart_cache.set("new_key", "new_value", save_to_disk=True)
        new_file = smart_cache._get_cache_file("new_key")

        # Modify file times to simulate age
        os.utime(old_file, (0, 0))  # Set old timestamp
        os.utime(new_file, (100000, 100000))  # Set new timestamp

        # Cleanup
        mock_time.return_value = 100000
        smart_cache.cleanup_expired(ttl=50000)

        assert not os.path.exists(old_file)
        assert os.path.exists(new_file)


# ============================================================================
# GET_STATS TESTS
# ============================================================================


def test_get_stats_empty(smart_cache):
    """Test get_stats with empty cache"""
    stats = smart_cache.get_stats()

    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["saves"] == 0
    assert stats["hit_rate"] == 0
    assert stats["memory_entries"] == 0


def test_get_stats_with_data(smart_cache, sample_data):
    """Test get_stats with data"""
    # Generate some cache activity
    smart_cache.set("key1", sample_data)
    smart_cache.set("key2", sample_data)
    smart_cache.get("key1")  # hit
    smart_cache.get("key2")  # hit
    smart_cache.get("key3")  # miss

    stats = smart_cache.get_stats()

    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["saves"] == 2
    assert stats["hit_rate"] == pytest.approx(66.67, rel=0.1)
    assert stats["memory_entries"] == 2


# ============================================================================
# _GET_CACHE_FILE TESTS
# ============================================================================


def test_get_cache_file_creates_hash(smart_cache):
    """Test _get_cache_file creates consistent hash"""
    file1 = smart_cache._get_cache_file("test_key")
    file2 = smart_cache._get_cache_file("test_key")

    assert file1 == file2
    assert ".cache" in file1


def test_get_cache_file_different_keys(smart_cache):
    """Test _get_cache_file creates different hashes for different keys"""
    file1 = smart_cache._get_cache_file("key1")
    file2 = smart_cache._get_cache_file("key2")

    assert file1 != file2


# ============================================================================
# GET_CACHE SINGLETON TESTS
# ============================================================================


def test_get_cache_singleton():
    """Test get_cache returns singleton instance"""
    cache1 = get_cache()
    cache2 = get_cache()

    assert cache1 is cache2


def test_get_cache_shared_state():
    """Test get_cache instances share state"""
    cache1 = get_cache()
    cache1.set("test_key", "test_value", save_to_disk=False)

    cache2 = get_cache()
    result = cache2.get("test_key")

    assert result == "test_value"


# ============================================================================
# CACHED DECORATOR TESTS
# ============================================================================


def test_cached_decorator_basic():
    """Test @cached decorator basic functionality"""
    call_count = [0]

    @cached(ttl=3600, key_prefix="test")
    def expensive_function(x, y):
        call_count[0] += 1
        return x + y

    result1 = expensive_function(1, 2)
    result2 = expensive_function(1, 2)  # Should use cache

    assert result1 == 3
    assert result2 == 3
    assert call_count[0] == 1  # Function called only once


def test_cached_decorator_different_args():
    """Test @cached decorator with different arguments"""
    call_count = [0]

    @cached(ttl=3600, key_prefix="test")
    def expensive_function(x, y):
        call_count[0] += 1
        return x + y

    result1 = expensive_function(1, 2)
    result2 = expensive_function(3, 4)  # Different args, should recompute

    assert result1 == 3
    assert result2 == 7
    assert call_count[0] == 2  # Function called twice


def test_cached_decorator_with_kwargs():
    """Test @cached decorator with keyword arguments"""
    call_count = [0]

    @cached(ttl=3600, key_prefix="test")
    def expensive_function(x, y=10):
        call_count[0] += 1
        return x + y

    result1 = expensive_function(5, y=10)
    result2 = expensive_function(5, y=10)  # Should use cache
    result3 = expensive_function(5, y=20)  # Different kwargs, should recompute

    assert result1 == 15
    assert result2 == 15
    assert result3 == 25
    assert call_count[0] == 2


def test_cached_decorator_ttl_expiration():
    """Test @cached decorator respects TTL"""
    call_count = [0]

    @cached(ttl=10, key_prefix="test")
    def expensive_function(x):
        call_count[0] += 1
        return x * 2

    with patch("time.time") as mock_time:
        mock_time.return_value = 0
        result1 = expensive_function(5)

        mock_time.return_value = 5  # Within TTL
        result2 = expensive_function(5)

        mock_time.return_value = 20  # Beyond TTL
        result3 = expensive_function(5)

        assert result1 == 10
        assert result2 == 10
        assert result3 == 10
        assert call_count[0] == 2  # Called at time 0 and time 20


# ============================================================================
# SPECIALIZED CACHE TESTS
# ============================================================================


def test_market_regime_cache(temp_cache_dir):
    """Test MarketRegimeCache"""
    regime_cache = MarketRegimeCache()

    regime_data = {"regime": "BULL", "confidence": 85}
    regime_cache.set_regime(regime_data)

    result = regime_cache.get_regime()

    assert result == regime_data


def test_market_regime_cache_invalidate(temp_cache_dir):
    """Test MarketRegimeCache invalidation"""
    regime_cache = MarketRegimeCache()

    regime_data = {"regime": "BULL", "confidence": 85}
    regime_cache.set_regime(regime_data)

    regime_cache.invalidate()

    result = regime_cache.get_regime()
    assert result is None


def test_ml_signal_cache(temp_cache_dir):
    """Test MLSignalCache"""
    signal_cache = MLSignalCache()

    signal_data = {"signal": "BUY", "confidence": 0.75}
    signal_cache.set_signal("VNM", signal_data)

    result = signal_cache.get_signal("VNM")

    assert result == signal_data


def test_ml_signal_cache_different_symbols(temp_cache_dir):
    """Test MLSignalCache with different symbols"""
    signal_cache = MLSignalCache()

    signal_cache.set_signal("VNM", {"signal": "BUY"})
    signal_cache.set_signal("VCB", {"signal": "SELL"})

    result_vnm = signal_cache.get_signal("VNM")
    result_vcb = signal_cache.get_signal("VCB")

    assert result_vnm == {"signal": "BUY"}
    assert result_vcb == {"signal": "SELL"}


def test_ml_signal_cache_invalidate(temp_cache_dir):
    """Test MLSignalCache invalidation"""
    signal_cache = MLSignalCache()

    signal_cache.set_signal("VNM", {"signal": "BUY"})
    signal_cache.invalidate("VNM")

    result = signal_cache.get_signal("VNM")
    assert result is None


def test_news_cache(temp_cache_dir):
    """Test NewsCache"""
    news_cache = NewsCache()

    news_data = {"title": "Stock news", "sentiment": "positive"}
    news_cache.set_news("VNM", news_data)

    result = news_cache.get_news("VNM")

    assert result == news_data


def test_news_cache_invalidate(temp_cache_dir):
    """Test NewsCache invalidation"""
    news_cache = NewsCache()

    news_cache.set_news("VNM", {"title": "News"})
    news_cache.invalidate("VNM")

    result = news_cache.get_news("VNM")
    assert result is None


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


def test_corrupted_cache_file(smart_cache, temp_cache_dir):
    """Test handling of corrupted cache file"""
    # Create corrupted cache file
    cache_file = smart_cache._get_cache_file("corrupted_key")
    with open(cache_file, "w") as f:
        f.write("corrupted data")

    # Should return None and not crash
    result = smart_cache.get("corrupted_key")
    assert result is None


def test_disk_write_error(smart_cache):
    """Test handling of disk write errors"""
    with patch("builtins.open", side_effect=IOError("Disk full")):
        # Should not crash
        smart_cache.set("test_key", "test_value", save_to_disk=True)


def test_disk_read_error(smart_cache, temp_cache_dir):
    """Test handling of disk read errors"""
    smart_cache.set("test_key", "test_value", save_to_disk=True)

    with patch("builtins.open", side_effect=IOError("Permission denied")):
        # Should return None and not crash
        smart_cache.memory_cache.clear()
        result = smart_cache.get("test_key")
        assert result is None


def test_large_data_storage(smart_cache):
    """Test caching large data structures"""
    large_data = {"data": list(range(10000))}

    smart_cache.set("large_key", large_data, save_to_disk=False)
    result = smart_cache.get("large_key")

    assert result == large_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
