import jax
import jax.numpy as jnp
from protax_utils import read_baseline, read_query
import numpy as np
import time
import pandas as pd


@jax.jit
def seq_dist(q, seqs, ok, ok_query):
    """
    Computes sequence distance between the query and 
    an array of reference sequences
    """

    # count matches and valid positions
    ok = jnp.bitwise_and(ok_query, ok)
    ok = jnp.sum(jax.lax.population_count(ok), axis=1)
    match = jnp.bitwise_and(q, seqs)

    match_tots = jnp.sum(jax.lax.population_count(match), axis=1)
    return jnp.argmax(1 - (match_tots / ok))

def nearest_classifier(q, seqs, ok, ok_query, n2s):
    closest_r = int(seq_dist(q, seqs, ok, ok_query))
    return n2s[:, closest_r].indices[-1]


def classify_file(qdir, verbose=False):
    """
    Process a batch of queries using baseline classifier
    """

    refs, ok_pos, n2s, paths = read_baseline(r"/home/roy/Documents/PROTAX-dsets/30k_small")
    f = open(qdir)

    tot_time = 0
    res = []

    while True:
        curr = f.readline().strip('\n').split('\t')[0]
        seqs = f.readline().strip('\n')
        q, ok = read_query(seqs)

        if not seqs:
            break  # EOF

        start = time.time()
        species_id = nearest_classifier(q, refs, ok_pos, ok, n2s)
        end = time.time()
        res.append(paths[species_id])
        tot_time += end-start


    # saving results
    df = pd.DataFrame(np.array(res))
    df.to_csv("dist_baseline_results.csv")
    print(f"finished in {tot_time}s")

if __name__ == "__main__":
    classify_file(r"/home/roy/Documents/PROTAX-dsets/30k_small/refs.aln")