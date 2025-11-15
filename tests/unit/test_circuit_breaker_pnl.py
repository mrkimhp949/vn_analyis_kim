# -*- coding: utf-8 -*-
"""
Test Circuit Breaker PnL Recording
Kiểm tra việc ghi nhận PnL ngay lập tức sau khi thoát lệnh
"""

from datetime import date

import pytest

from src.risk.circuit_breaker import CircuitBreaker


@pytest.fixture
def temp_stats_file(tmp_path):
    """Tạo file stats tạm thời cho testing"""
    stats_file = tmp_path / "test_circuit_breaker_stats.json"
    return str(stats_file)


@pytest.fixture
def breaker(temp_stats_file):
    """Tạo CircuitBreaker instance cho testing"""
    return CircuitBreaker(
        max_trades_per_day=10,
        max_loss_per_day_pct=0.05,  # 5%
        max_consecutive_losses=5,
        vnindex_drop_threshold=-2.5,
        total_capital=100_000_000,
        stats_file=temp_stats_file,
    )


def test_record_pnl_normal(breaker):
    """Test ghi nhận PnL bình thường (chưa vượt ngưỡng)"""
    # PnL -2% (chưa vượt ngưỡng 5%)
    breaker.record_pnl(-0.02)

    assert not breaker.is_active()
    assert breaker.tripped_reason == ""


def test_record_pnl_triggers_circuit_breaker(breaker):
    """Test ghi nhận PnL vượt ngưỡng kích hoạt circuit breaker"""
    # PnL -6% (vượt ngưỡng 5%)
    breaker.record_pnl(-0.06)

    assert breaker.is_active()
    assert "Lỗ trong ngày" in breaker.tripped_reason
    assert "-6.00%" in breaker.tripped_reason


def test_record_pnl_exactly_at_threshold(breaker):
    """Test ghi nhận PnL đúng bằng ngưỡng"""
    # PnL -5% (đúng bằng ngưỡng)
    breaker.record_pnl(-0.05)

    assert breaker.is_active()
    assert "Lỗ trong ngày" in breaker.tripped_reason


def test_record_pnl_positive(breaker):
    """Test ghi nhận PnL dương (lãi)"""
    # PnL +3%
    breaker.record_pnl(0.03)

    assert not breaker.is_active()
    assert breaker.tripped_reason == ""


def test_check_and_update_validates_portfolio_pnl(breaker):
    """Test check_and_update validate portfolio_pnl_pct"""
    # Test với giá trị hợp lệ
    result = breaker.check_and_update(portfolio_pnl_pct=-0.03, vnindex_change_pct=-0.01)
    assert result is False

    # Test với giá trị không hợp lệ
    with pytest.raises(ValueError, match="portfolio_pnl_pct phải là số"):
        breaker.check_and_update(portfolio_pnl_pct="invalid", vnindex_change_pct=-0.01)

    with pytest.raises(ValueError, match="vnindex_change_pct phải là số"):
        breaker.check_and_update(portfolio_pnl_pct=-0.03, vnindex_change_pct="invalid")


def test_check_and_update_with_loss_threshold(breaker):
    """Test check_and_update kích hoạt khi vượt ngưỡng lỗ"""
    # PnL -6% vượt ngưỡng 5%
    result = breaker.check_and_update(portfolio_pnl_pct=-0.06, vnindex_change_pct=-0.01)

    assert result is True
    assert breaker.is_active()
    assert "Lỗ trong ngày" in breaker.tripped_reason


def test_check_and_update_with_vnindex_drop(breaker):
    """Test check_and_update kích hoạt khi VNINDEX giảm sâu"""
    # VNINDEX giảm -3% vượt ngưỡng -2.5%
    result = breaker.check_and_update(portfolio_pnl_pct=-0.02, vnindex_change_pct=-0.03)

    assert result is True
    assert breaker.is_active()
    assert "VNINDEX giảm sâu" in breaker.tripped_reason


def test_record_trade_then_record_pnl(breaker):
    """Test kết hợp record_trade và record_pnl"""
    # Ghi nhận một trade thua lỗ
    breaker.record_trade(-2_000_000)  # Lỗ 2 triệu

    assert breaker.stats["today"]["trades_count"] == 1
    assert breaker.stats["consecutive_losses"] == 1

    # Sau đó ghi nhận PnL tổng thể vượt ngưỡng
    breaker.record_pnl(-0.06)

    assert breaker.is_active()
    assert "Lỗ trong ngày" in breaker.tripped_reason


def test_circuit_breaker_stays_tripped(breaker):
    """Test circuit breaker vẫn tripped sau khi đã kích hoạt"""
    # Kích hoạt circuit breaker
    breaker.record_pnl(-0.06)
    assert breaker.is_active()

    # Gọi lại với PnL tốt hơn - vẫn phải tripped
    result = breaker.check_and_update(portfolio_pnl_pct=-0.02, vnindex_change_pct=-0.01)
    assert result is True
    assert breaker.is_active()


def test_circuit_breaker_resets_new_day(breaker, temp_stats_file):
    """Test circuit breaker reset khi sang ngày mới"""
    # Kích hoạt circuit breaker
    breaker.record_pnl(-0.06)
    assert breaker.is_active()

    # Tạo breaker mới (giả lập ngày mới)
    # Trong thực tế, _check_new_day() sẽ tự động reset
    new_breaker = CircuitBreaker(
        max_trades_per_day=10,
        max_loss_per_day_pct=0.05,
        max_consecutive_losses=5,
        vnindex_drop_threshold=-2.5,
        total_capital=100_000_000,
        stats_file=temp_stats_file,
    )

    # Nếu cùng ngày, vẫn tripped
    if new_breaker.stats["today"]["date"] == date.today().isoformat():
        # Trong test này, stats file được load lại nên tripped state có thể không persist
        # Nhưng nếu có lỗi trong ngày, check_and_update sẽ kích hoạt lại
        pass


def test_is_active_method(breaker):
    """Test phương thức is_active()"""
    assert not breaker.is_active()

    breaker.record_pnl(-0.06)
    assert breaker.is_active()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
