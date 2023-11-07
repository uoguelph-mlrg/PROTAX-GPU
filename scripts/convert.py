# This script converts the files used in PROTAX to a readable format for PROTAX-GPU
# Files:
#     - Taxonomy: refs.aln, taxonomy.priors, model.rseqs.numeric
#     - Model: model.pars, model.scs

from protax.protax_utils import *
import argparse
import protax.model as model

from pathlib import Path
import pandas as pd
import scipy.sparse as sparse

TEST_QUERY = '---------------------------GCTGGTATAGTAGGAACATCTTTA---AGAATTTTAATTCGTGCAGAATTAGGTCATCCAGGTGCTTTAATTGGTGAT---GATCAAATTTATAATGTAATTGTTACAGCACATGCTTTTGTAATAATTTTTTTTATAGTAATACCTATTATAATTGGAGGTTTTGGAAATTGATTAGTTCCTTTAATA---TTAGGAGCTCCTGATATAGCATTTCCTCGAATAAATAATATAAGTTTTTGATTATTACCTCCTTCATTAACATTATTACTAGTAAGTAGTATAGTAGAAAATGGAGCTGGGACAGGATGAACAGTATATCCTCCACTTTCTTCTAGCATTGCTCATGGAGGAGCTTCAGTAGATTTA---GCTATTTTTTCTTTACACTTAGCTGGTATATCTTCTATTTTAGGTGCAGTAAATTTTATTACAACAGTTATTAATATACGATCTTCTGGAATTTCTTACGATCGAATACCTTTATTTGTATGATCAGTTGTTATTACTGCTTTATTACTTCTTTTATCATTACTTGTATTAGCAGGA---GCAATTACTATACTTTTAACAGATCGTAATTTAAATACT----------------------------------------------------------------------------------'

# ===== Helper functions =====
def valid_path(dir):
    dir = Path(dir)
    if not dir.exists():
        raise argparse.ArgumentTypeError(f"{dir} is not a valid path")
    
    return dir


def lil_to_csr(lil, shape):
    """
    Converts a list of lists to a csr matrix
    """

    num_nonzero = 0
    indices = []
    indptr = [0]

    for row in lil:
        num_nonzero += len(row)
        indptr.append(num_nonzero)
        indices.extend(row)
    
    data = np.ones(num_nonzero, dtype=bool)
    csr = sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=bool)
    return csr
    


# ===== Main functions ====


def add_unknown_nodes(df):
    """
    Adds unknown nodes to the taxonomy
    """
    parents = df["parentid"].unique() 
    nid = df.index.max() + 1

    unk_df = pd.DataFrame({
        "id": range(nid, nid + len(parents)),
        "parentid": parents,
        "rank": df["rank"][parents] + 1,
        "taxon_name": "unk"
    }).set_index("id")

    df = pd.concat([df, unk_df])
    print("finished adding unknown nodes")
    return df


def add_prior(df):
    """
    Adds uniform prior over leaves to the taxonomy.
    """
    print("start adding prior")
    parents = df["parentid"].unique() 
    leaf_prob = 1/ (len(df) - len(parents))

    leaf_rank = df["rank"].max()
    # assume reverse topological order (children before parents)
    priors = {k: 0 for k in df.index}

    finished = 0

    for i, r in df[::-1].iterrows():
        if i not in parents:
            priors[i] = leaf_prob
        
        if r["parentid"] in priors:
            priors[r["parentid"]] += priors[i]
        finished += 1
        print(f"finished {finished} out of {len(df)}\r", end="")
    
    df["prior"] = df.index.map(priors)
    print("\nfinished adding prior")


def trim_subtaxa(df, keep):
    """ 
    Remove subtaxa from taxonomy specified in <keep>.
    parents inherit subtaxa's children
    """
    print("start trimming subtaxa")
    ranks = dict(zip(df.index, df["rank"]))
    to_drop = []

    rows_processed = 0

    # check if parent's rank is in keep
    # TODO doesn't work yet
    # df_todo = df[~df[df["parentid"]]["rank"].isin(keep)]

    for i, r in df.iterrows():
        if ranks[i] in keep:      # row's rank is in keep
            curr_par = r.iloc[0]
            par_rank = ranks[curr_par]

            while par_rank not in keep:
                curr_par = df["parentid"][curr_par]
                par_rank = ranks[curr_par]

            df.at[i, "parentid"] = curr_par
        
        else:
            to_drop.append(i)
        
        rows_processed += 1
        print(f"finished {rows_processed} out of {len(df)}\r", end="") 
    df.drop(to_drop, inplace=True)
    
    # map ranks to start from 0
    mapping = dict(zip(df["rank"].unique(), range(len(df["rank"].unique()))))
    df["rank"] = df["rank"].map(mapping)
    print("\nfinished trimming subtaxa")

def convert_tsv(t_dir):

    # TODO change get_descendants to work with root being excluded
    df = pd.read_csv(t_dir, sep='\t')
    df.at[0, "parentid"] = 1
    df["parentid"] = df["parentid"].astype(int)
    df.set_index("id", inplace=True)

    # keep only specific ranks
    trim_subtaxa(df, [1,2,5,8,11,12,14,17])

    # add unknown nodes
    df = add_unknown_nodes(df)

    df.at[1, "parentid"] = 0
    # add prior to df
    add_prior(df)

    df = df.reset_index()
    df = df.set_index("parentid").sort_index()

    # assign new ids based on sorted segments
    mapping = dict(zip(df["id"], range(len(df["id"]))))
    df["id"] = df["id"].map(mapping)
    df.index = df.index.map(mapping)

    # convert to protax format
    # Note parents and segments are the same
    segments = df.index.to_numpy() 
    prior = df["prior"].to_numpy()
    ranks = df["rank"].to_numpy()
    unk = (df["taxon_name"] == "unk").to_numpy()
    segments[0] = 0
    segments = segments.astype(np.int32)
    paths = get_descendants(segments, len(df), ranks)
    segments[0] = -1

    # temp
    np.savez("temp_tax.npz", 
             segments=segments, 
             unk=unk, ranks=ranks, 
             prior=prior, 
             paths=paths,
             names=df["taxon_name"].to_numpy())
    print("saved npz")
    return segments, unk, ranks, prior, paths


def assign_tax(ref_dir):
    # TODO: temporary hardcoded funct to prevent processing again when debugging
    tax = np.load("temp_tax.npz", allow_pickle=True)
    names = tax["names"]
    N = len(names)
    R = 0

    name2id = dict(zip(names, range(N)))
    no_refs = np.ones((N, 1), dtype=bool)
    has_refs = set()

    # temporarily stored in LIL format for easy appending 
    node2seq = [[] for _ in range(N)]

    print("start assigning sequences to taxa")
    with open(ref_dir, 'r') as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            R += 1
            # rank at all levels, convert to nid via mapping
            line = line.strip('\n').split('\t')[1:-1]
            curr_nids = []
            for taxa in line:
                if taxa in name2id:
                    curr_nids.append(name2id[taxa])

            for nid in curr_nids:
                has_refs.add(nid)
                node2seq[nid].append(i)
            
            print(f"\rassigned {i} seqs", end='')
    
    no_refs[list(has_refs)] = False

    print("\nfinished assigning sequences to taxa") 
    # turn into csr matrix
    node2seq = lil_to_csr(node2seq, (N, R))

    ref_npz = np.load("refs.npz", allow_pickle=True)
    refs = ref_npz["refs"]
    ok_pos = ref_npz["ok_pos"]
    n2s_indices = node2seq.indices
    n2s_indptr = node2seq.indptr

    # save full taxonomy 
    segments = tax["segments"]
    unk = tax["unk"]
    ranks = tax["ranks"]
    prior = tax["prior"]
    paths = tax["paths"]

    # compute node state
    unk = np.expand_dims(unk, axis=1)

    node_state = no_refs*np.logical_not(unk)
    node_state = np.concatenate([node_state, np.logical_not(no_refs)], axis=1)

    np.savez("8M_tax.npz",
             segments=segments,
             unk=unk,
             ranks=ranks,
             priors=prior,
             refs=refs,
             ok_pos=ok_pos,
             paths=paths,
             n2s_indices=n2s_indices,
             n2s_indptr=n2s_indptr,
             node_state=node_state,
             parents=segments
             )

    print("saved npz") 
    return 0


def convert_sequences(ref_dir):
    """
    converts reference sequences in tsv format to protax format
    """

    # TODO remove hardcoded values
    file_len = 0
    seq_len = 0 

    with open(ref_dir, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            line = line.strip('\n').split('\t')

            if seq_len == 0:
                seq_len = len(line[-1])

            file_len += 1

    dummy_seq = np.zeros((5, seq_len), dtype=bool)
    dummy_seq_packed = np.packbits(dummy_seq[:4], axis=None)
    dummy_ok = np.packbits(dummy_seq[4], axis=None)

    # Note: width is with packed bits
    ref_list = np.zeros((file_len, dummy_seq_packed.shape[0]), dtype=np.uint8) 
    ok_pos = np.zeros((file_len, dummy_ok.shape[0]), dtype=np.uint8)

    with open(ref_dir, 'r') as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            line = line.strip('\n').split('\t')

            seq = line[-1]
            seq_bits = get_seq_bits(seq)

            ref_list[i] = np.packbits(seq_bits[:4], axis=None)
            ok_pos[i] = np.packbits(seq_bits[4], axis=None)
            print(f"\rfinished {i} seqs", end='')
        print("\nfinished reading seqs")

    return ref_list, ok_pos


def read_jax_model(model_dir):
    """
    Reads model files from JAX implementation of PROTAX
    """
    tax_npz = np.load("8M_tax.npz", allow_pickle=True)
    
    refs = jnp.array(tax_npz["refs"])
    ok_pos = jnp.array(tax_npz["ok_pos"])
    segments = jnp.array(tax_npz["segments"])
    paths = jnp.array(tax_npz["paths"])
    node_state = jnp.array(tax_npz["node_state"])
    prior = jnp.array(tax_npz["priors"])

    N = len(segments)
    R = len(refs)

    n2s_indices = tax_npz["n2s_indices"]
    n2s_indptr = tax_npz["n2s_indptr"]
    node2seq = sparse.csr_matrix((np.ones(n2s_indices.shape[0], dtype=float), n2s_indices, n2s_indptr), shape=(N, R))
    node2seq = CSRWrapper(data=jnp.array(node2seq.data), 
                          indices=jnp.array(node2seq.indices), 
                          indptr=jnp.array(node2seq.indptr),
                          shape=node2seq.shape)
    
    segnum = int(jnp.max(segments) + 1)
    tree = TaxTree(
        refs=refs,
        ok_pos=ok_pos,
        segments=segments,
        node2seq=node2seq,
        paths=paths,
        node_state=node_state,
        prior=prior
    )

    beta = jnp.ones((N, 4))
    sc_mean = jnp.ones((N, 2))
    sc_var = jnp.ones((N, 2))
    params = ProtaxModel(
        beta=beta,
        sc_mean=sc_mean,
        sc_var=sc_var
    )

    # simple classification test
    # TODO: remove this

    q, ok = read_query(TEST_QUERY)

    print("knn test")
    knn_time = 0
    jit_knn = jax.jit(model.knn_v2, static_argnums=(3))
    jit_knn(node2seq.indptr, node2seq.indices, node2seq.data, N).block_until_ready()
    
    for i in range(20):
        start = time.time()
        jit_knn(node2seq.indptr, node2seq.indices, node2seq.data, N).block_until_ready()
        end = time.time()
        knn_time += end - start
    
    print(f"knn avg time: {knn_time}", knn_time/20)
    print("test seqdist")

    # jit_dist = jax.jit(model.seq_dist)
    # jit_getx = jax.jit(model.get_X, static_argnums=(3))

    # # warmup
    # jit_dist(q, tree.refs, tree.ok_pos, ok).block_until_ready()
    # jit_getx(q, ok, tree, N, params.sc_mean, params.sc_var).block_until_ready()
    
    # x_time = 0
    # seq_time = 0
    # for i in range(100):
    #     start = time.time()
    #     jit_dist(q, tree.refs, tree.ok_pos, ok).block_until_ready()
    #     end = time.time()
    #     seq_time += end - start

    #     start = time.time()
    #     jit_getx(q, ok, tree, N, params.sc_mean, params.sc_var).block_until_ready()
    #     end = time.time()
    #     x_time += end - start
    
    # print(f"avg x time: {x_time/100}s")
    # print(f"avg seqdist time: {seq_time/100}s")

    # print("start classification")
    # # jit warmup
    # probs = model.get_probs(q, ok, tree, params, segnum, N).block_until_ready()
    # tot_time = 0
    # # actual run
    # for i in range(100):
    #     start = time.time()
    #     model.get_probs(q, ok, tree, params, segnum, N).block_until_ready()
    #     end = time.time()
    #     tot_time += end - start
    
    # print(f"avg time: {tot_time/100}s")
    return tree, params, N, R, segnum

test = Path("/home/roy/Downloads/taxonomy.tsv")
test_ref = Path("/home/roy/Downloads/sequences.tsv")
# df = convert_sequences(test_ref)
# df = convert_tsv(test)
# aasdf = assign_tax(test_ref)

read_jax_model(test_ref)

cc = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="converts the files used in PROTAX to a readable format for PROTAX-GPU")

    parser.add_argument('-t', '--taxonomy', type=valid_path, help='path to folder containing taxonomy files')
    parser.add_argument('-m', '--model', type=valid_path, help='path to folder containing model files')
    parser.add_argument('-o', '--output', type=valid_path, default= '', help='path to output folder')
    args = parser.parse_args()

    # call the relevant conversion functions
    output_dir = args.output
    if args.taxonomy:
        print('converting taxonomy...')
        print(type(args.taxonomy))

    if args.model:
        print('converting model...', args.model)