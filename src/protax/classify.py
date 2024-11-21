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

def getMapping(tax_map):
    """
    Generate a mapping from node numbers to taxonomy labels dynamically
    identifying the label column as the first column with non-numeric values.
    """
    taxonomy_dict = {}

    with open(tax_map, "r") as file:
        for line in file:
            parts = line.strip().split("\t")  # Split line into parts using tab as the delimiter
            
            # Determine the label column dynamically
            for i, part in enumerate(parts):
                if not part.isdigit():  # Check if the column contains non-numeric data
                    label_column = i
                    break
            
            node_number = int(parts[0])  # Assume the first column is always the node number
            label = parts[label_column]  # Dynamically select the label column
            taxonomy_dict[node_number] = label

    return taxonomy_dict


def classify_file(qdir, par_dir, tax_dir, tax_map, verbose=False):
    """
    Process a batch of queries given a model and taxonomy directory
    """

    tree, params, N, segnum = read_model_jax(par_dir, tax_dir)

    n2s_clone = csr_matrix((tree.node2seq.data, tree.node2seq.indices, tree.node2seq.indptr), shape=(N, tree.refs.shape[0]))
    n2s_clone
    
    f = open(qdir)

    tot_time = 0
    res = []
    node2label = getMapping(tax_map)
    while True:
        curr = f.readline().strip('\n')
        curr = curr.replace('|', '\t').split('\t')
        seqs = f.readline().strip('\n')
        q, ok = read_query(seqs)

        curr_name = ''.join(curr[1:])
        if not seqs:
            break  # EOF
            
        start = time.time()
        probs = get_probs(q, ok, tree, params, segnum, N).block_until_ready()
        end = time.time()

        probs = jnp.take(probs, tree.paths, fill_value=-1)


        # TODO argmax at leaf level?
        probabilities = np.array(jnp.max(probs, axis=0))
        classified_layer = jnp.argmax(probs, axis=0)
        classified_tax_labels = [node2label[int(i)] for i in classified_layer]
        out = []
        out.extend(list(classified_tax_labels[-1].split(',')))
        out.extend(probabilities[1:])
        res.append(out)
        if verbose:
            pass
            # sel_name = names[classified_layer.at[-1].get()]
            # print(f"{curr}: {sel_name}: {end - start}s")
        tot_time += end-start

    # saving results
    df = pd.DataFrame(np.array(res))
    df.columns = ['tax_label1',  'tax_label2', 'tax_label3', 'tax_label4', 'tax_label5', 'tax_label6', 'tax_label7',
                  'prob_level1', 'prob_level2', 'prob_level3', 'prob_level4', 'prob_level5', 'prob_level6', 'prob_level7']
    df.to_csv("pyprotax_results.csv")
    print(f"finished in {tot_time}s")


def compute_perplexity(qdir, model_dir, tax_dir, verbose=False):
    """
    compute perplexity on dataset
    """

    tree, params, N, segnum = read_model_jax(model_dir, tax_dir)
    f = open(qdir)
    tot_time = 0

    layers = load_layer(tax_dir)
    num_layers = jax.ops.segment_sum(jnp.ones(layers.shape[0], dtype=int), layers)
    
    n = 0
    perplexity = jnp.zeros(8)
    while True:
        curr = f.readline().strip('\n').split('\t')[0]
        seqs = f.readline().strip('\n')
        q, ok = read_query(seqs)

        if not seqs:
            break  # EOF
        
        start = time.time()
        probs = get_log_probs(q, ok, tree, params, segnum, N).block_until_ready()
        end = time.time()


        probs = probs *jnp.exp(probs)
        # mask out paths which end short
        probs = jnp.take(probs, tree.paths, fill_value=-1)*(tree.paths != N)

        # aggregate entropy
        curr_p = -jnp.sum(probs, axis=0) 
        perplexity += curr_p
        n += 1

        tot_time += end-start

    # saving results
    print(jnp.exp(perplexity / n))
    print(f"finished in {tot_time}s")



if __name__ == "__main__":

    # protax_args = sys.argv
    # if len(protax_args) < 4:
    #     print("Usage: python3 classify.py [PATH_TO_TAXONOMY_FILE] [PATH_TO_PARAMETERS] [PATH_TO_QUERY_SEQUENCES]")
    
    # tax_dir, model_dir, query_dir = protax_args[1:4]

    # testing for now
    
    query_dir = r"FinPROTAX/FinPROTAX/modelCOIfull/refs.aln"
    classify_file(query_dir, "models/params/model.npz", "models/ref_db/taxonomy37k.npz", "FinPROTAX/FinPROTAX/modelCOIfull/taxonomy.priors") 
    compute_perplexity(query_dir, "models/params/model.npz", "models/ref_db/taxonomy37k.npz")