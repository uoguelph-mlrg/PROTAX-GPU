import pandas as pd
import numpy as np
import jax.numpy as jnp

targ_dir = "/home/roy/Documents/PROTAX-dsets/30k_small/30k-targets.csv"
res_dir_c = "cprotax_results.csv"
res_dir_py = "pyprotax_results.csv"


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

def read_c_results(res_dir):
    # processing C csv output6
    pred_df = pd.read_csv(res_dir)
    pred_df = pred_df.iloc[:, 1:]
    pred =  pred_df.to_numpy().T

    return pred

def read_py_results(res_dir):

    pred_df_py = pd.read_csv(res_dir)
    pred_df_py = pred_df_py.iloc[:, 2:]

    pred = pred_df_py.to_numpy().T
    return pred


def read_targ(tdir):
    targ_df = pd.read_csv(targ_dir)
    targ = targ_df.to_numpy()[:, 1:]

    return targ

# =========== Evaluation Functions =============
def eval_layer_acc(pred_c, pred_py, targ, N):

    print("-----Evaluating Layer Accuracy-----")

    if pred_py.shape[1] != pred_c.shape[1]:
        print("different number of predictions made, doublecheck csv")
        return

    TP_c = np.sum(np.equal(targ, pred_c), axis=1)
    TP_py = np.sum(np.equal(targ, pred_py), axis=1)

    print("Python Accuracy: ", *(TP_py / N))
    print("C Accuracy: ", *(TP_c / N))


def eval_path_acc(pred_c, pred_py, targ, N):

    print("-----Evaluating Path Accuracy-----")

    if pred_py.shape[1] != pred_c.shape[1]:
        print("different number of predictions made, doublecheck csv")
        return

    TP_c = np.sum(np.equal(targ, pred_c))/N
    TP_py = np.sum(np.equal(targ, pred_py))/N

    print("Python Accuracy: ", TP_py)
    print("C Accuracy: ", TP_c)


def evaluate_all():
    """
    Run all evaluation metrics
    """
    print(res_dir_py)
    pred_c = read_c_results(res_dir_c)
    pred_py = read_py_results(res_dir_py)
    targ = read_targ(targ_dir)

    # mask out invalid indices
    targ_mask = targ != -1
    pred_py *= targ_mask
    pred_c *= targ_mask

    eval_layer_acc(pred_c, pred_py, targ, np.sum(targ_mask, axis=1))
    eval_path_acc(pred_c, pred_py, targ, np.sum(targ_mask))


if __name__ == "__main__":
    evaluate_all()