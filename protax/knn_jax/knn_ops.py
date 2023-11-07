# =======================================================
# This file defines the interface between JAX and the
# custom KNN functions that are implemented in C++/CUDA.
# =======================================================

__all__ = ["knn, knn_v2"]

from knn_jax import gpu_ops                    # refers to knn_jax installed in site-packages
from jax.lib import xla_client
from jax import core
from jax.core import ShapedArray
from jax.interpreters import xla, mlir, batching
from jaxlib.hlo_helpers import custom_call            # NOTE: change to mhlo_helpers in older versions of JAX
from functools import partial

import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse

# =======================================================
#                      Helper Functions
# =======================================================
def default_layouts(*shapes):
    """
    Helper to specify default memory layout for custom call
    """
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

# =======================================================
#                     Primitives
# =======================================================

# Expose primitive to user code
def knn(indptr, indices, matdat, N):
    """
    Return row-wise k nearest neighbors for a sparse CSR matrix
    """

    res = jnp.zeros((N, 2))
    return _knn_prim.bind(indptr, indices, matdat, res)


def knn_v2(indptr, indices, matdat, N):
    """
    Return row-wise k nearest neighbors for a sparse CSR matrix
    """

    res = jnp.zeros((N, 2))
    return _knn_v2_prim.bind(indptr, indices, matdat, res)


# =======================================================
#             JIT Support for KNN Primitive
# =======================================================
def _knn_abstract_eval(indptr, indices, matdat, res):
    """
    Abstract evaluation for knn primitive where k=2

    mat: input CSR matrix

    NOTE: signature should match that of _knn_prim.bind() call
    """
    return ShapedArray(res.shape, matdat.dtype)


def _knn_lowering(ctx, indptr, indices, matdat, res):
    """
    MLIR lowering for knn primitive where k=2.
    i.e. lowering primitive to MLIR custom call (knn kernel dispatch)

    ctx: mlir.LoweringRuleContext
    """

    mat_dtype = ctx.avals_in[2].dtype
    mat_type = mlir.ir.RankedTensorType(matdat.type)
    res_type = mlir.ir.RankedTensorType(res.type)
    ip_type = mlir.ir.RankedTensorType(indptr.type)
    idx_type = mlir.ir.RankedTensorType(indices.type)
    md_type = mlir.ir.RankedTensorType(matdat.type)

    N = res_type.shape[0]

    # dispatch for float32 only
    if mat_dtype != jnp.float32:
        raise NotImplementedError(f"unsupported dtype: {mat_dtype}")

    if gpu_ops is None:
        raise ValueError("gpu_ops not compiled")
    
    # create opaque descriptor for problem size
    opaque = gpu_ops.build_knn_descriptor(N)

    out = custom_call(
        b"gpu_knn_f32",                                   # call target name
        out_types=[res_type],
        operands=[indptr, indices, matdat], 
        operand_layouts=default_layouts(ip_type.shape,
                                         idx_type.shape, 
                                         md_type.shape),
        result_layouts=default_layouts(res_type.shape),   # memory layout
        backend_config=opaque,                            # opaque descriptor
    )

    # output must be iterable
    return [out]


# =======================================================
#            JIT Support for KNN v2 Primitive
# =======================================================
def _knn_v2_abstract_eval(indptr, indices, matdat, res):
    """
    Abstract evaluation for knn primitive where k=2

    mat: input CSR matrix

    NOTE: signature should match that of _knn_prim.bind() call
    """
    return ShapedArray(res.shape, matdat.dtype)


def _knn_v2_lowering(ctx, indptr, indices, matdat, res):
    """
    MLIR lowering for knn primitive where k=2.
    i.e. lowering primitive to MLIR custom call (knn kernel dispatch)

    ctx: mlir.LoweringRuleContext
    """

    mat_dtype = ctx.avals_in[2].dtype
    mat_type = mlir.ir.RankedTensorType(matdat.type)
    res_type = mlir.ir.RankedTensorType(res.type)
    ip_type = mlir.ir.RankedTensorType(indptr.type)
    idx_type = mlir.ir.RankedTensorType(indices.type)
    md_type = mlir.ir.RankedTensorType(matdat.type)

    N = res_type.shape[0]

    # dispatch for float32 only
    if mat_dtype != jnp.float32:
        raise NotImplementedError(f"unsupported dtype: {mat_dtype}")

    if gpu_ops is None:
        raise ValueError("gpu_ops not compiled")
    
    # create opaque descriptor for problem size
    opaque = gpu_ops.build_knn_descriptor(N)

    out = custom_call(
        b"gpu_knn_v2_f32",                                   # call target name
        out_types=[res_type],
        operands=[indptr, indices, matdat], 
        operand_layouts=default_layouts(ip_type.shape,
                                         idx_type.shape, 
                                         md_type.shape),
        result_layouts=default_layouts(res_type.shape),   # memory layout
        backend_config=opaque,                            # opaque descriptor
    )

    # output must be iterable
    return [out]


# =======================================================
#             Registering KNN Primitive
# =======================================================

# register GPU XLA custom calls
for _name, _value in gpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


# defining KNN primitive for JAX
_knn_prim = core.Primitive("knn")
_knn_prim.def_impl(partial(xla.apply_primitive, _knn_prim))
_knn_prim.def_abstract_eval(_knn_abstract_eval)

# connect XLA translation rules for JIT compilation
mlir.register_lowering(_knn_prim, _knn_lowering, platform="gpu")


# defining KNN v2 primitive
_knn_v2_prim = core.Primitive("knn_v2")
_knn_v2_prim.def_impl(partial(xla.apply_primitive, _knn_v2_prim))
_knn_v2_prim.def_abstract_eval(_knn_v2_abstract_eval)
mlir.register_lowering(_knn_v2_prim, _knn_v2_lowering, platform="gpu")


# testing on a simple example
# TODO: make proper tests for this
if __name__ == "__main__":
    foo = jnp.array([
        [0.012, 0.3, 0.2],
        [0.01, 0, 0.3],
        [0.4, 0.1, 0.5],
    ], dtype=jnp.float32)

    foo = sparse.bcsr_fromdense(foo)
    x = knn(foo.indptr, foo.indices, foo.data, 3)

    h = 3
