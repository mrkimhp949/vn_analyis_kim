# [file name]: logging_config.py
# [file content begin]

import io
import logging
import os
import sys

# Suppress warnings first (before any imports)
from src.utils import suppress_warnings  # noqa: F401


def setup_logging():
    """Cấu hình logging toàn hệ thống"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Đảm bảo stdout/stderr dùng UTF-8 (tránh UnicodeEncodeError trên Windows)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # Fallback: bọc stream với UTF-8 nếu cần
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Handlers với UTF-8
    file_handler = logging.FileHandler(os.path.join(log_dir, "trading_bot.log"), encoding="utf-8")
    console_handler = logging.StreamHandler(stream=sys.stdout)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler],
    )

    # Giảm log level cho một số thư viện (HTTP clients)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(
        logging.WARNING
    )  # Suppress httpx INFO logs (Telegram API calls)
    logging.getLogger("httpcore").setLevel(logging.WARNING)  # httpx uses httpcore underneath

    # Tensorflow logging
    try:
        import tensorflow as tf

        tf.get_logger().setLevel(logging.ERROR)
    except ImportError:
        pass

    print("✅ Logging system initialized")


# [file content end]
