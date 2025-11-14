"""
Module để suppress các warnings không cần thiết
Import module này ở đầu các file chính để tắt warnings

Usage:
    import suppress_warnings  # noqa: F401

Suppresses:
    - FutureWarning từ transformers/huggingface_hub
    - DeprecationWarning từ các thư viện cũ
    - TensorFlow oneDNN info messages
    - TensorFlow logging (chỉ hiện ERROR)
"""

import os
import warnings

# ===== TensorFlow Settings =====
# PHẢI set TRƯỚC KHI import tensorflow/keras
# Tắt oneDNN custom operations info message
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Tắt TensorFlow logging (chỉ hiện ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR

# Tắt TensorFlow deprecation warnings
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

# Disable TensorFlow warnings về deprecated APIs
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").propagate = False

# Try to disable TensorFlow v1 compatibility warnings
try:
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except (ImportError, AttributeError):
    pass

# ===== Python Warnings =====
# Suppress FutureWarning từ transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Suppress DeprecationWarning từ các thư viện cũ
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress UserWarning không quan trọng
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")

# Suppress Keras warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="keras")

# Suppress specific TensorFlow messages
warnings.filterwarnings("ignore", message=".*tf.losses.sparse_softmax_cross_entropy.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Có thể thêm các warnings khác cần suppress ở đây
