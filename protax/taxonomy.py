import chex
from jax.experimental import sparse


@chex.dataclass
class CSRWrapper():
    """
    Simple dataclass to wrap CSR data
    """
    data: chex.ArrayDevice
    indices: chex.ArrayDevice
    indptr: chex.ArrayDevice
    shape: tuple


@chex.dataclass
class TaxTree():
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
    refs: chex.ArrayDevice           # [R, 5]
    ok_pos: chex.ArrayDevice         # [R]
    segments: chex.ArrayDevice       # [N]
    node2seq: CSRWrapper             # [N]
    paths: chex.ArrayDevice          # [N]
    node_state: chex.ArrayDevice     # [N, 2]
    prior: chex.ArrayDevice


@chex.dataclass
class ProtaxModel():
    """
    Contains parameters for PROTAX model
    """
    
    beta: chex.ArrayDevice
    sc_mean: chex.ArrayDevice
    sc_var: chex.ArrayDevice
