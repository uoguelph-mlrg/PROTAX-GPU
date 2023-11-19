# =======================================================
# This file defines the entire knn_utils public interface
# =======================================================
__all__ = ["__version__", "knn", "knn_v2"]

from .knn_jax import knn, knn_v2    # refers to local knn_jax.py
