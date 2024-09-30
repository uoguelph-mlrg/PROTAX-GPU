import protax.model as model
from protax import protax_utils
from protax.taxonomy import CSRWrapper

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse

import scipy.sparse as sp
import time
import pandas as pd

from pathlib import Path
import random
import matplotlib.pyplot as plt
from functools import partial
import argparse


def CE_loss(log_probs, y_ind):
    """
    Computes the cross-entropy loss between the log_probs and
    labels y_ind.

    Args:
        log_probs: Log probabilities returned by the model, shape (N, D)
        y_ind: Integer array of true class indices, shape (N,)
    """
    return -jnp.sum(jnp.take(log_probs, y_ind, axis=0))


def forward(q, ok, tree, beta, sc_mean, sc_var, N, segnum, y_ind, lvl):
    beta = jnp.take(beta, lvl, axis=0)
    X = model.get_X(q, ok, tree, N, sc_mean, sc_var)
    log_probs = model.fill_log_bprob(X, beta, tree, segnum)
    return CE_loss(log_probs, y_ind)


f_grad = jax.jit(jax.grad(forward, argnums=(3)), static_argnums=(6, 7))
forward_jit = jax.jit(forward, static_argnums=(6, 7))


def get_targ(target_dir):
    """
    Get node id for each reference sequence at lowest level
    """
    targ = pd.read_csv(target_dir)
    targ = targ.to_numpy()[:, 1:].T

    res = np.zeros((targ.shape[0],), dtype=np.int32)

    for i in range(len(targ)):
        old = -1
        for j in range(targ.shape[1]):
            if targ[i][j] == -1:
                res[i] = old
            elif j == targ.shape[1] - 1:
                res[i] = targ[i][j]
            old = targ[i][j]

    return jnp.array(res)


def mask_n2s(n2s, node_state, i):
    """
    Remove a column in node2seq
    """
    ref_mask = np.ones((n2s.shape[1],), dtype=np.int32)
    ref_mask[i] = 0
    n2s = n2s @ sp.diags(ref_mask)

    has_refs = np.array(n2s.sum(axis=1)) > 0
    empty = np.logical_not(has_refs)

    # update empty but known entries
    node_state = np.logical_or(node_state, empty)
    node_state = np.concatenate((node_state, has_refs), axis=1)

    n2s = CSRWrapper(
        data=jnp.array(n2s.data),
        indices=jnp.array(n2s.indices),
        indptr=jnp.array(n2s.indptr),
        shape=n2s.shape,
    )

    return n2s, jnp.array(node_state)


def load_params(pdir, tdir):
    par_dir = Path(pdir)
    tax_dir = Path(tdir)

    tax = np.load(tax_dir.resolve())
    par = np.load(par_dir.resolve())

    beta = par["beta"]
    sc = par["scalings"]
    lvl = tax["node_layer"]

    tax = np.load(tax_dir.resolve())
    par = np.load(par_dir.resolve())

    return beta, lvl, sc


def train(train_config, train_dir, targ_dir):
    tree, params, N, segnum = protax_utils.read_model_jax(
        "models/params/model.npz", "models/ref_db/taxonomy37k.npz"
    )
    pkey = jax.random.PRNGKey(0)
    lr = train_config["learning_rate"]

    beta = jax.random.uniform(pkey, (7, 4))
    n2s = sp.csr_matrix(
        (tree.node2seq.data, tree.node2seq.indices, tree.node2seq.indptr),
        shape=tree.node2seq.shape,
    )
    targ = get_targ(targ_dir)
    seq_list, ok_list = protax_utils.read_refs(train_dir)
    node_state = np.expand_dims(np.array(tree.node_state)[:, 0], 1)

    # params and node lvl
    _, lvl, sc = load_params("models/params/model.npz", "models/ref_db/taxonomy37k.npz")
    loss_hist = []
    for e in range(train_config["num_epochs"]):
        print(f"epoch {e}")

        beta_grad = 0
        loss_sum = 0
        batch_loss = 0

        traversal = list(range(seq_list.shape[0]))
        random.shuffle(traversal)

        # minibatch
        for i in traversal:
            # mask out tree
            q = seq_list[i]
            ok = ok_list[i]

            # tree.node_state = mask_design_mat(tree, num_refs, targ, i)

            # masks out a node2seq column given reference index (~1.2 ms)
            tree.node2seq, tree.node_state = mask_n2s(n2s, node_state, i)
            beta_grad += f_grad(
                q,
                ok,
                tree,
                beta,
                params.sc_mean,
                params.sc_var,
                N,
                segnum,
                targ.at[i].get(),
                lvl,
            )
            batch_loss += forward_jit(
                q,
                ok,
                tree,
                beta,
                params.sc_mean,
                params.sc_var,
                N,
                segnum,
                targ.at[i].get(),
                lvl,
            )

            if i % train_config["batch_size"] == 0:
                beta = beta - lr * beta_grad
                loss_sum += batch_loss
                curr_loss = batch_loss / train_config["batch_size"]
                print("batch_loss: ", curr_loss)
                loss_hist.append(curr_loss)
                batch_loss = 0

                # grad norm
                bflat = beta_grad.reshape(beta_grad.shape[0] * beta_grad.shape[1])

        print("loss: ", loss_sum / seq_list.shape[0])

        # save checkpoint
        mf = Path("models/params/m2.npz")
        np.savez_compressed(mf.resolve(), beta=np.array(beta), scalings=sc)
    plt.plot(loss_hist)
    plt.show()


if __name__ == "__main__":
    # parse config from command line
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--train_dir", type=str, help="Path to training data")
    parser.add_argument("--targ_dir", type=str, help="Path to target data")
    args = parser.parse_args()
    train_dir = Path(args.train_dir)
    targ_dir = Path(args.targ_dir)

    # train_dir = r"/home/roy/Documents/PROTAX-dsets/30k_small/refs.aln"
    # targ_dir = "/home/roy/Documents/PROTAX-dsets/30k_small/30k-targets.csv"

    # training config
    tc = {
        "learning_rate": 0.001,
        "batch_size": 500,
        "num_epochs": 30,
    }

    train(tc, train_dir, targ_dir)  # train the model
