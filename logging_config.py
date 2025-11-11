import logging
import os

def setup_logging():
    """Cấu hình logging toàn hệ thống"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'trading_bot.log')),
            logging.StreamHandler()
        ]
    )
    
    # Giảm log level cho một số thư viện
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
