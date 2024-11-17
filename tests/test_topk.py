import jax
import jax.numpy as jnp
from jax.experimental import sparse
import time
from protax.taxonomy import CSRWrapper
import numpy as np
from scipy.sparse import csr_matrix
from knn_jax import knn
import shutil
import logging

jax_key = jax.random.PRNGKey(0)

# ========= Helper functions ==========
def generate_random_csr_matrix(nrows, ncols, density):

    # Generate random values
    nnz = int(nrows * ncols * density)
    values = np.random.random(nnz)
    row_indices = np.random.randint(0, nrows, nnz)
    col_indices = np.random.randint(0, ncols, nnz)

    # Create the CSR matrix
    csr = csr_matrix((values, (row_indices, col_indices)), shape=(nrows, ncols))
    return _csr2wrapper(csr)


def _csr2wrapper(csr):
    """
    Turns a scipy csr matrix into a CSRWrapper for JAX
    """

    ndat, nind, nindptr = map(jnp.asarray, (csr.data, csr.indices, csr.indptr))
    return CSRWrapper(data=ndat, indices=nind, indptr=nindptr, shape=csr.shape)

# =========== Useful Constants ===========
N2S_SMALL = _csr2wrapper(csr_matrix(np.array([
                                [0.012, 0, 0],
                                [0.01, 0, 0.3],
                                [0.4, 0.1, 0.5],
                            ])))
N2S_1K = generate_random_csr_matrix(1000, 1000, 0.001)
N2S_40K = generate_random_csr_matrix(40000, 40000, 0.0004)


# ======================================
#             Benchmarks
# ======================================
def bench_knn():
    """
    Benchmark speed of KNN on sparse arrays of different sizes.
    NOTE: You might OOM on large matrices ~7M
    """

    for i in range(1, 235):
        curr_size = i*30000
        
        curr_mat = generate_random_csr_matrix(curr_size, curr_size, 1/curr_size)

        # constant sparsity version
        # curr_mat = generate_random_csr_matrix(curr_size, curr_size, 0.0001)

        # compile first
        N = curr_mat.shape[0]
        knn(curr_mat.indptr, curr_mat.indices, curr_mat.data, N)

        # run benchmark
        start_time = time.time()
        knn(curr_mat.indptr, curr_mat.indices, curr_mat.data, N).block_until_ready()
        end_time = time.time()
        print(curr_size, end_time - start_time)

# ======================================
#                Unit Tests
# ======================================

# # Function to check if CUDA is available

def cuda_available():
    return shutil.which("nvcc") is not None

def test_topk_platforms():
    cpu_knn = jax.jit(knn, backend="cpu", static_argnums=(3,))

    N = N2S_SMALL.shape[0]
    cpu_res = cpu_knn(N2S_SMALL.indptr, N2S_SMALL.indices, N2S_SMALL.data, N).block_until_ready()

    if(cuda_available()):
        gpu_knn = jax.jit(knn, backend="gpu", static_argnums=(3,))
        gpu_res = gpu_knn(N2S_SMALL.indptr, N2S_SMALL.indices, N2S_SMALL.data, N).block_until_ready()
        cpu_res = jax.device_put(cpu_res, jax.devices("gpu")[0])
        assert jnp.all(cpu_res == gpu_res)
    else:   
        logging.warning("CUDA not available, skipping GPU tests")
        return
    

def test_topk_small():

    N = N2S_SMALL.shape[0]
    start_time = time.time()
    res = knn(N2S_SMALL.indptr, N2S_SMALL.indices, N2S_SMALL.data, N).block_until_ready()
    end_time = time.time()

    print(f"test_topk_small: {end_time - start_time}s")

  
def test_topk_1k():
    N = N2S_1K.shape[0]
    start_time = time.time()
    res = knn(N2S_1K.indptr, N2S_1K.indices, N2S_1K.data, N).block_until_ready()
    end_time = time.time()

    print(f"test_topk_1K: {end_time - start_time}s")

    # testing shape and order is correct
    assert res.shape == (1000, 2)
    assert jnp.all(res.at[:, 0].get() <= res.at[:, 1].get())


def test_topk_40k():
    N = N2S_40K.shape[0]
    start_time = time.time()
    res = knn(N2S_40K.indptr, N2S_40K.indices, N2S_40K.data, N).block_until_ready()
    end_time = time.time()

    print(f"test_topk_40K: {end_time - start_time}s")

    # testing shape and order is correct
    assert res.shape == (40000, 2)
    assert jnp.all(res.at[:, 0].get() <= res.at[:, 1].get())


# put in function to debug manually
# TODO: remove this when vscode detects pytests properly
if __name__ == '__main__':
    test_topk_small()
    # Get the names of all functions in the global scope
    function_names = [name for name, value in globals().items() if callable(value) and value.__module__ == '__main__' and name.startswith("test")]

    # Iterate over the function names and call the functions
    for name in function_names:
        try:
            globals()[name]()
            print(f"passed {name}")
        except AssertionError as e:
            print(f"failed {name}: {e}")
