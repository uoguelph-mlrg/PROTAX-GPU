import pandas as pd

import jax.numpy as jnp
from protax import protax_utils, model
import numpy as np

debug_state_f = r"tests/debug-aln-state.csv"
debug_f = r"/home/roy/Documents/PROTAX-dsets/30k_small/debug.aln"
debug_x = r"tests/debug_x.csv"
targ_state = pd.read_csv(debug_state_f)
debug_x_df = pd.read_csv(debug_x)

def test_z_and_bprob():
    """
    Test the correctness on the 30k small set over entire design matrix
    """
    tree, params, N, segnum = protax_utils.read_model_jax("models/params/model.npz", "models/ref_db/taxonomy.npz")
    f = open(debug_f)

    name = f.readline().strip('\n').split('\t')[0]
    seqs = f.readline().strip('\n')
    q, ok = protax_utils.read_query(seqs)

    # testing design matrix
    X = np.array(model.get_X(q, ok, tree, N, params.sc_mean, params.sc_var))
    X_targ = debug_x_df.to_numpy()
    print("binary variables correct: ", np.sum(X[:, :2] == X_targ[:, :2])/(2*N))

    for t in [0.001/(10**i) for i in range(0, 4)]:
        print(f"distances correct (threshold {t}): ", np.sum(jnp.abs(X[:, 2:] - X_targ[:, 2:]) < t)/(2*N))

    # testing log probabilities
    bprobs = jnp.prod(model.fill_bprob(X, params.beta, tree, segnum), axis=1)
    log_bprobs = jnp.sum(model.fill_log_bprob(X, params.beta, tree, segnum), axis=1)

    print("log probs correct (threshold: 1e-5)", jnp.sum(jnp.abs(bprobs - jnp.exp(log_bprobs)) < 0.00001)/N)


if __name__ == "__main__":
    test_z_and_bprob()