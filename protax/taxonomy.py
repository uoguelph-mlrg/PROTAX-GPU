import jax
from typing import NamedTuple


class CSRWrapper(NamedTuple):
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    shape: tuple

class TaxTree(NamedTuple):
    """
    State of the taxonomic tree

    N = total number of nodes
    L = Number of non-species nodes (i.e nodes with depth < 7)
    R = total number of reference sequences

    refs: All reference sequences
    ok_pos: positions which contain a, t, c, g
    node_refs: reference sequences belong to node at the same index
    layer: boundary indices of each layer
    prior: prior probability of each node
    prob: predicted probability of each node
    children: adjacency matrix of each node
    descendants: descendants of each node
    unk: Whether the node at this index represents an unknown species or not
    """
    refs: jax.Array                  # [R, 5]
    ok_pos: jax.Array                # [R]
    segments: jax.Array              # [N]
    node2seq: CSRWrapper             # [N]
    paths: jax.Array                 # [N]
    node_state: jax.Array            # [N, 2]
    prior: jax.Array


class ProtaxModel(NamedTuple):
    """
    Contains parameters for PROTAX model
    """
    
    beta: jax.Array
    sc_mean: jax.Array
    sc_var: jax.Array
