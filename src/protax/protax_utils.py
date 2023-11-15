"""
Functions for reading from files used by PROTAX
"""
import jax
import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse
from .taxonomy import CSRWrapper, TaxTree, ProtaxModel
from scipy.sparse import csr_matrix

from functools import reduce
import pandas as pd

import sys
from pathlib import Path
import time


def read_params(pdir):
    """
    Read parameters from file
    """
    print("reading parameters")
    
    with open(pdir, 'r') as f:
        res = []
        for l in f.readlines():
            res.append(jnp.fromstring(l, sep=" "))
        return res


def read_scalings(pdir):
    """
    Read scalings from file
    """
    print("reading scalings")
    f = open(pdir)
    res = []
    for l in f.readlines():
        res.append(l.split(" "))

    res = np.array(res)[:, 1::2].astype("float32")
    return res


def read_taxonomy(tdir):
    """
    Read taxonomy tree from file
    """
    print("reading taxonomy file")
    f = open(tdir)
    node_dat = f.readlines()

    # making result arrays
    N = len(node_dat)
    parents = []  # parent of node at nid index belongs to (segments)
    child_col = []  # child nid for parent
    unks = np.zeros(N, dtype=bool)
    priors = np.zeros(N)
    layers = np.zeros(N, dtype=int)
    
    for l in node_dat:
        l = l.strip("\n")

        # collecting taxon data
        nid, pid, lvl, name, prior, _ = l.split("\t")
        nid, pid, lvl, prior = (int(nid), int(pid), int(lvl), float(prior))
        name = name.split(",")[-1]

        # assign children
        parents.append(pid)
        child_col.append(nid)
        
        # information about node
        unks[nid] = name=="unk"
        layers[nid] = lvl
        priors[nid] = prior
    
    # making segments
    child_col = np.array(child_col)
    parents = np.array(parents)
    parents[0] = -1
    unq = np.unique(parents)
    segments = np.searchsorted(unq, parents)

    descendants = get_descendants(parents, N, layers)

    # convert rest to cupy
    return segments, unks, layers, priors, descendants, parents


def get_descendants(nodes, N, layers):
    """
    Get the descendant nodes for each entry in a node-parent vector
    """

    # TODO fix this
    res = (np.ones((nodes.shape[0], 8))*N).astype(int)
    for n in range(1, nodes.shape[0]):
        
        curr_nid = nodes[n]  # parent of node n
        res[n][layers[curr_nid]] = curr_nid

        # while curr isn't root
        while curr_nid != 0:
            curr_nid = nodes[curr_nid]  # parent of parent
            res[n][layers[curr_nid]] = curr_nid
            
        res[n][layers[n]] = n
    
    res[0][0] = 0
    return res


def read_refs(ref_dir):
    """
    Read reference sequences from file
    """
    print("reading reference sequences")
    f = open(ref_dir)
    ref_list = []
    ok_pos = []

    i = 1
    while True:
        name = f.readline().strip('\n').split('\t')[0]
        seqs = f.readline().strip('\n')
        seq_bits = get_seq_bits(seqs)

        if not seqs:
            break  # EOF
        ref_list.append(np.packbits(seq_bits[:4], axis=None))
        ok_pos.append(np.packbits(seq_bits[4], axis=None))
        print('\r' + str(i), end='\n')
        i += 1
    return np.array(ref_list), np.array(ok_pos)


def assign_refs(seq2tax_dir):
    """
    Assign reference sequences to nodes from file
    """
    # TODO make a processing script for easier reading on next runs
    print("\nassigning reference sequences to taxa")
    f = open(seq2tax_dir)

    seqs = np.array([], dtype=int)
    nids = np.array([], dtype=int)

    # assigning ref seq indices
    for n, l in enumerate(f.readlines()):
        nid, num_refs, ref_idx = l.split('\t')
        nid = int(nid)

        seq_ids = np.fromstring(ref_idx, sep=" ").astype(int)
        seqs = np.concatenate([seqs, seq_ids])
        nids = np.concatenate([nids, np.full(seq_ids.shape, nid)])

    return nids, seqs

def get_seq_bits(seq_str):
    """
    Convert seqence string to bit representation
    """
    seq_chars = np.frombuffer(seq_str.encode('ascii'), np.int8)
    a = seq_chars == 65
    t = seq_chars == 84
    g = seq_chars == 71
    c = seq_chars == 67
    ok = np.logical_or.reduce([a, t, g, c])

    seq_bits = np.array([a, t, g, c, ok])
    return seq_bits


def assign_params(beta, sc, lvl):
    """
    Assign parameters to each node given the levels each node is in
    """
    return np.take(beta, lvl-1, axis=0), np.take(sc, lvl-1, axis=0)


def convert_model(model_dir, savedir="models/params"):
    """
    Read and convert model files stored in model_dir, convert and save them to npz
    """
    mdir = Path(model_dir)
    savedir = Path(savedir)

    # reading model info
    beta = read_params(mdir.joinpath("model.pars"))
    scalings = read_scalings(mdir.joinpath("model.scs"))

    print("model converted, saving...")
    model_file = savedir.joinpath("model.npz")

    i = 1
    while model_file.exists():
        model_file = savedir.joinpath(f"model_{i}.npz")
        i += 1
    
    np.savez_compressed(
        model_file.resolve(),
        beta=beta,
        scalings=scalings,
    )

    print(f"saved model at {model_file.resolve()}")


def convert_taxonomy(treedir, savedir="models/ref_db"):
    """
    Read model files stored in treedir, convert and save them to a npz
    """

    treedir = Path(treedir)
    savedir = Path(savedir)

    seg, unk, layer, prior, paths, parents = read_taxonomy(treedir.joinpath("taxonomy.priors"))
    refs, ok_pos = read_refs(treedir.joinpath("refs.aln"))

    ref_rows, ref_cols = assign_refs(treedir.joinpath("model.rseqs.numeric"))
    
    # save state of each node [empty but known, has_refs]
    N = seg.shape[0]
    R = refs.shape[0]

    unk = np.expand_dims(unk, axis=1)

    no_refs = np.ones((N,1), dtype=bool)
    no_refs[ref_rows] = 0

    node_state = no_refs*np.logical_not(unk)
    node_state = np.concatenate((node_state, np.logical_not(no_refs)), axis=1)

    tax_file = savedir.joinpath("taxonomy.npz")
    i = 1
    while tax_file.exists():
        tax_file = savedir.joinpath(f"taxonomy_{i}.npz")
        i += 1

    print("taxonomy converted, saving...")

    # saving model
    np.savez_compressed(tax_file.resolve(),
                        segments=seg,
                        unk=unk,
                        node_layer=layer,
                        priors=prior,
                        paths=paths,
                        refs=refs,
                        ok_pos=ok_pos,
                        ref_rows=ref_rows,
                        ref_cols=ref_cols,
                        node_state=node_state,
                        parents=parents
                        )
    print(f"saved taxonomy at {tax_file.resolve()}")


def read_baseline(model_dir=r"/h/royga/Documents/PROTAX-dsets/30k_small"):
    loaded = np.load(model_dir + r'/model.npz')
    refs = jnp.array(loaded['refs'])
    seg = loaded['segments']
    paths = loaded['paths']
    ok_pos = jnp.array(loaded['ok_pos'])

    N = seg.shape[0]
    nids, seqs = (loaded['ref_rows'], loaded['ref_cols'])
    n2s = csr_matrix((np.ones(seqs.shape), (nids, seqs)), shape=(N, refs.shape[0])).tocsc()
    
    return refs, ok_pos, n2s, paths


def read_model_jax(par_dir, tax_dir):
    """
    Read model npz representation
    """
    par_dir = Path(par_dir)
    tax_dir = Path(tax_dir)

    tax = np.load(tax_dir.resolve())
    par = np.load(par_dir.resolve())
    refs = jnp.array(tax['refs'])
    
    ok_pos = jnp.array(tax['ok_pos'])
    prior = jnp.array(tax['priors'])

    seg = jnp.array(tax['segments'])
    paths = jnp.array(tax['paths'])
    node_state = jnp.array(tax['node_state'])

    N = seg.shape[0]
    nids, seqs = (tax['ref_rows'], tax['ref_cols'])

    # TODO dumb way of converting to JAX BCSR
    n2s = csr_matrix((np.ones(seqs.shape), (nids, seqs)), shape=(N, refs.shape[0]))
    # node2seq = sparse.BCSR((n2s.data, n2s.indices, n2s.indptr), shape=(N, refs.shape[0]))
    node2seq = CSRWrapper(data=jnp.array(n2s.data),
                          indices=jnp.array(n2s.indices), 
                          indptr=jnp.array(n2s.indptr),
                          shape=(N, refs.shape[0]))

    segnum = int(jnp.max(seg) + 1)

    # dataclass to contain taxonomy and sequences
    tree = TaxTree(
        refs=refs,
        ok_pos=ok_pos,
        segments=seg,
        node2seq=node2seq,
        paths=paths,
        node_state=node_state,
        prior=prior
    )

    # model parameters
    beta = par['beta']
    scalings = par['scalings']
    layer = tax['node_layer']
    beta, scalings = assign_params(beta, scalings, layer)

    # beta = jnp.array(beta)
    sc_mean = jnp.array(scalings[:, [0, 2]])
    sc_var = jnp.array(scalings[:, [1, 3]])

    params = ProtaxModel(
        beta=beta,
        sc_mean=sc_mean,
        sc_var=sc_var
    )

    return tree, params, N, segnum

    

def get_train_targets(seq2tax_dir, layers, R):
    """
    Save target NIDs for each reference sequence contained in R. Save as CSV
    """
    f = open(seq2tax_dir)

    # 8 x N result array
    res = np.zeros((7, R), dtype=int) - 1

    # assigning ref seq indices
    for n, l in enumerate(f.readlines()):
        nid, num_refs, ref_idx = l.split('\t')
        nid = int(nid)

        curr_layer = layers[nid]-1
        ref_idx = np.array(ref_idx.split(' ')).astype(int)
        res[curr_layer][ref_idx] = nid
    
    df = pd.DataFrame(res)
    df.to_csv("30k-targets.csv")


def read_query(q):
    s = get_seq_bits(q)
    return jnp.array(np.packbits(s[:4], axis=None)), jnp.array(np.packbits(s[4], axis=None))


def str2batch_query(q):

    queries = []
    ok_pos = []
    for i in q:
        curr = get_seq_bits(i)
        queries.append(np.packbits(curr[:4], axis=None))
        ok_pos.append(np.packbits(curr[4], axis=None))
    
    return jnp.array(queries), jnp.array(ok_pos)

if __name__ == "__main__":
    convert_model(r"/home/roy/Documents/PROTAX-dsets/30k_small")
    convert_taxonomy(r"/home/roy/Documents/PROTAX-dsets/30k_small")