# =======================================================
# This file defines the entire knn_utils public interface
# =======================================================
__all__ = ["__version__", "knn", "knn_v2"]

from .knn_ops import knn, knn_v2    # refers to knn_ops.py in same dir
