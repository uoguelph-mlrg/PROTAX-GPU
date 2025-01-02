from . import cpu_ops
try:
    from . import gpu_ops
    is_gpu_available = True
except ImportError:
    is_gpu_available = False

from .knn_register import knn, knn_v2

__all__ = [
    "knn",
    "knn_v2"
]
