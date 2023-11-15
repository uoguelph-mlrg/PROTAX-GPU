import jax
import jax.numpy as jnp
from jax.experimental import sparse
from functools import partial
import numpy as np
from functools import partial
from knn_jax import knn, knn_v2

# @jax.jit
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
    return 1 - (match_tots / ok)



def get_X(q, ok_q, tree, N, sc_mean, sc_var):
    """
    KNN-based method for computing design matrix
    """

    node2seq = tree.node2seq
    dists = seq_dist(q, tree.refs, tree.ok_pos, ok_q)

    # TODO maybe use custom BCSR multiplication kernel
    # or can also be replaced with a take operation
    new_dat = jnp.take(dists, node2seq.indices)
    node2seq.data = new_dat

    X = knn(node2seq.indptr, node2seq.indices, node2seq.data, N)
    X = (((X - sc_mean) / sc_var).T*(tree.node_state[:, 1])).T
    X = jnp.concatenate((tree.node_state, X), axis=1)
    return X


def get_z(X, params):
    """
    Compute weighted sum for each node
    """
    z = jnp.sum(jnp.multiply(X, params.beta), axis=1)
    return z


def get_bprobs(z, segments, segnum):
    """
    Compute branch probabilities of each node
    """
    norm_factors = jax.ops.segment_sum(z, segments, num_segments=segnum, indices_are_sorted=True)
    norm_factors = jnp.take(norm_factors, segments, indices_are_sorted=True)
    branch_probs =  jnp.nan_to_num(z / norm_factors)
    branch_probs = branch_probs.at[0].set(1)

    return branch_probs



def get_log_bprobs(z, segments, segnum):
    """
    Compute log branch probabilities of each node
    For training PROTAX
    """

    exp_z = jnp.exp(z)
    norm_factors = jnp.log(jax.ops.segment_sum(exp_z, segments, num_segments=segnum, indices_are_sorted=True))
    norm_factors = jnp.take(norm_factors, segments, indices_are_sorted=True)
    branch_probs =  jnp.nan_to_num(z - norm_factors)
    branch_probs = branch_probs.at[0].set(0)

    return branch_probs


def fill_bprob(X, beta, tree, segnum):
    """
    Compute branch probability of entire taxonomy, filled in relevant paths
    X: design matrix of shape [N, M]
    beta: param matrix of shape [N, M]
    parents: array with shape [N]

    N = # nodes
    M = # features
    """
    z = jnp.sum(jnp.multiply(X, beta), axis=1)
    max_z = jax.ops.segment_max(z, tree.segments, num_segments=segnum, indices_are_sorted=True)
    max_z = jnp.take(max_z, tree.segments, indices_are_sorted=True)
    exp_z = jnp.exp(z - max_z)*tree.prior
    branch_probs = get_bprobs(exp_z, tree.segments, segnum)

    filled_paths = jnp.take(branch_probs, tree.paths, indices_are_sorted=True,
                          fill_value=1, unique_indices=True)
    return filled_paths


def fill_log_bprob(X, beta, tree, segnum):
    """
    Compute log probabilities over entire taxonomy
    Used for training PROTAX
    """
    z = jnp.sum(jnp.multiply(X, beta), axis=1)
    max_z = jax.ops.segment_max(z, tree.segments, num_segments=segnum, indices_are_sorted=True)
    max_z = jnp.take(max_z, tree.segments, indices_are_sorted=True)
    z -= max_z
    branch_probs = get_log_bprobs(z, tree.segments, segnum)

    # total probability computation
    filled_paths = jnp.take(branch_probs, tree.paths, indices_are_sorted=True,
                          fill_value=0, unique_indices=True)
    return filled_paths


# @partial(jax.jit, static_argnums=(4, 5))
def get_log_probs(q, ok, tree, params, segnum, N):
    """
    Compute log probabilities for each node
    """
    X = get_X(q, ok, tree, N, params.sc_mean, params.sc_var)
    bprobs = fill_log_bprob(X, params.beta, tree, segnum)
    return jnp.sum(bprobs, axis=1)

@partial(jax.jit, static_argnums=(4, 5))
def get_probs(q, ok, tree, params, segnum, N):
    X = get_X(q, ok, tree, N, params.sc_mean, params.sc_var)
    bprobs = fill_bprob(X, params.beta, tree, segnum)
    return jnp.prod(bprobs, axis=1)


if __name__ == '__main__':
    print(jax.devices())
