from protax.classify import classify_file
import sys

if __name__ == "__main__":
    protax_args = sys.argv
    if len(protax_args) < 4:
        print(
            "Usage: python3 classify.py [PATH_TO_TAXONOMY_FILE] [PATH_TO_PARAMETERS] [PATH_TO_QUERY_SEQUENCES]"
        )

    query_dir, model_dir, tax_dir = protax_args[1:4]
    classify_file(query_dir, model_dir, tax_dir)

    # testing
    # query_dir = r"/home/roy/Documents/PROTAX-dsets/30k_small/refs.aln"
    # classify_file(query_dir, "models/params/model.npz", "models/ref_db/taxonomy37k.npz")
