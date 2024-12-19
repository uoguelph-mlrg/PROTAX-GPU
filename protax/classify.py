import os

from .protax_utils import read_model_jax, read_query, read_baseline
from .model import get_probs, get_log_probs
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
import time
from pathlib import Path


def load_layer(tdir):
    tax_dir = Path(tdir)

    tax = np.load(tax_dir.resolve())

    lvl = tax['node_layer']

    tax = np.load(tax_dir.resolve())

    return lvl

def read_names(tdir):
    """
    Read names of each taxon from file
    """
    f = open(tdir)
    node_dat = f.readlines()

    names = []
    for l in node_dat:
        l = l.strip("\n")

        # collecting taxon data
        nid, pid, lvl, name, prior, _ = l.split("\t")
        nid, pid, lvl, prior = (int(nid), int(pid), int(lvl), float(prior))
        names.append(name)

    return names

def validate_taxonomy_query(tree, query, ok_query):
    """
    Validate that the taxonomy and query dimensions match.

    Args:
        tree: TaxTree object containing taxonomy information.
        query: Query sequence array.
        ok_query: Boolean array indicating valid positions in the query.

    Raises:
        ValueError: If dimensions are incompatible.
    """
    # Check the number of positions in references and query
    if tree.refs.shape[1] != query.shape[0]:
        raise ValueError(f"Mismatch in sequence lengths: "
                         f"taxonomy={tree.refs.shape[1]} vs query={query.shape[0]}")

    # Check ok_pos and ok_query have the same length
    if tree.ok_pos.shape[1] != ok_query.shape[0]:
        raise ValueError(f"Mismatch in valid positions: "
                         f"taxonomy={tree.ok_pos.shape[1]} vs query={ok_query.shape[0]}")

def classify_file(qdir, par_dir, tax_dir, verbose=False):
    """
    Process a batch of queries given a model and taxonomy directory
    """

    tree, params, N, segnum = read_model_jax(par_dir, tax_dir)

    n2s_clone = csr_matrix((tree.node2seq.data, tree.node2seq.indices, tree.node2seq.indptr), shape=(N, tree.refs.shape[0]))
    n2s_clone
    
    f = open(qdir)

    tot_time = 0
    res = []

    while True:
        curr = f.readline().strip('\n')
        curr = curr.replace('|', '\t').split('\t')
        seqs = f.readline().strip('\n')
        q, ok = read_query(seqs)

        validate_taxonomy_query(tree, q, ok)

        curr_name = ''.join(curr[1:])
        if not seqs:
            break  # EOF
            
        start = time.time()
        probs = get_probs(q, ok, tree, params, segnum, N).block_until_ready()
        end = time.time()

        probs = jnp.take(probs, tree.paths, fill_value=-1)


        # TODO argmax at leaf level?
        classified_layer = jnp.argmax(probs, axis=0)
        res.append(classified_layer)

    
        if verbose:
            pass
            # sel_name = names[classified_layer.at[-1].get()]
            # print(f"{curr}: {sel_name}: {end - start}s")
        tot_time += end-start

    # saving results
    df = pd.DataFrame(np.array(res))
    df.to_csv("pyprotax_results.csv")
    print(f"finished in {tot_time}s")



def classify(q, ok, tree, params, segnum, N):
    pass



if __name__ == "__main__":

    # protax_args = sys.argv
    # if len(protax_args) < 4:
    #     print("Usage: python3 classify.py [PATH_TO_TAXONOMY_FILE] [PATH_TO_PARAMETERS] [PATH_TO_QUERY_SEQUENCES]")
    
    # tax_dir, model_dir, query_dir = protax_args[1:4]

    # testing for now
    
    query_dir = r"FinPROTAX/FinPROTAX/modelCOIfull/refs.aln"
    classify_file(query_dir, "models/params/model.npz", "models/ref_db/taxonomy37k.npz") 
    compute_perplexity(query_dir, "models/params/model.npz", "models/ref_db/taxonomy37k.npz")

