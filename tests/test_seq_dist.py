from protax.protax_utils import read_query, str2batch_query
from protax.model import seq_dist
import jax
import jax.numpy as jnp
import random
import time
import numpy as np


SEQ_LEN = 15
dummy = ""

# ======================================
#                Helpers
# ======================================
def gen_seq(d):
    """
    Generate random sequence of length <d>
    """
    return ''.join(random.choices("ATGC-", k=d))


# ======================================
#                Unit Tests
# ======================================

def test_no_match():
    """
    Test two sequence distances with no matching base pairs
    """
    q = ''.join(['A' for i in range(SEQ_LEN)])
    r = ''.join(['T' for i in range(SEQ_LEN)])

    q, q_ok = read_query(q)
    r, r_ok = read_query(r)

    r = jnp.array([r])
    r_ok = jnp.array([r_ok])

    assert seq_dist(q, r, r_ok, q_ok) == 1.0


def test_no_match_batched():

    # 2 references, 4 queries
    queries = [''.join([c for i in range(SEQ_LEN)]) for c in "AGC-"]
    r = ''.join(['T' for i in range(SEQ_LEN)])

    queries, q_ok = str2batch_query(queries)
    r, r_ok = read_query(r)
    r = jnp.array([r, r])
    r_ok = jnp.array([r_ok, r_ok])

    batch_sd = jax.vmap(seq_dist, (0, None, None, 0), 0)
    c = batch_sd(queries, r, r_ok, q_ok)
    
    assert jnp.all(c == 1.0)


# ======================================
#              Benchmarks
# ======================================

def test_1seq_scaling():
    """
    Test sequence distance with 1 query sequence as reference database size increases
    """

    q = ''.join(['A' for i in range(700)])
    r = ''.join(['T' for i in range(700)])

    q, q_ok = read_query(q)
    r, r_ok = read_query(r)

    r = jnp.tile(jnp.array(r), (13000000, 1))
    r_ok = jnp.tile(jnp.array(r_ok), (13000000, 1))

    # warmup
    seq_dist(q, r, r_ok, q_ok).block_until_ready()
    print("finished warmup")

    start = time.time()
    seq_dist(q, r, r_ok, q_ok).block_until_ready()
    end = time.time()

    print("Took", end - start, "seconds")

if __name__ == "__main__":
    test_1seq_scaling()