# PROTAX-GPU

![alt text](img/Block_diagram_upd.png?raw=true)

A GPU-accelerated JAX-based implementation of [PROTAX](https://pubmed.ncbi.nlm.nih.gov/27296980/). Contains all code and experiments for [PROTAX-GPU](https://royalsocietypublishing.org/doi/10.1098/rstb.2023.0124)

To reproduce the BOLD 7.8M dataset experiments, PROTAX-GPU requires a NVIDIA GPU with at least 8GB VRAM and CUDA compute capability 6.0 or later. This corresponds to GPUs in the NVIDIA Pascal, NVIDIA Volta™, NVIDIA Turing™, NVIDIA Ampere architecture, and NVIDIA Hopper™ architecture families.

### Contents:

[Functionality](https://github.com/uoguelph-mlrg/PROTAX-GPU#Functionality)<br>
[Features](https://github.com/uoguelph-mlrg/PROTAX-GPU#Features)<br>
[Organization](https://github.com/uoguelph-mlrg/PROTAX-GPU#Organization)<br>
[Compatibility](https://github.com/uoguelph-mlrg/PROTAX-GPU#Compatibility)<br>
[Installation](https://github.com/uoguelph-mlrg/PROTAX-GPU#Installation)<br>
[Usage](https://github.com/uoguelph-mlrg/PROTAX-GPU#Usage)
- [Inference](https://github.com/uoguelph-mlrg/PROTAX-GPU#Inference)
- [Training](https://github.com/uoguelph-mlrg/PROTAX-GPU#Training)

# Functionality
Estimates the probability of each outcome in a taxonomic tree given a query barcode sequence compared to reference sequences.

- Uses JAX to accelerate sequence distance and probability decomposition calculations
- Uses custom CUDA kernels to accelerate the calculation of the top-k most similar reference sequences

**Features:**
- CPU and GPU inference
- Gradient-based optimization of model parameters
- Compatible with TSV and PROTAX input format 
- Full computation of all probabilities in the taxonomic tree

**Organization:**
```
experiments/        All experiments for the paper
lib/                C++/CUDA code for PROTAX-GPU
scripts/            Scripts for training and inference
├-- train.py        Trains a model with gradient descent 
└-- convert.py      Converts .TSV to PROTAX-GPU format
tests/              Unit tests 
src/                
├-- knn_jax/        JAX bindings for CUDA kernels
└-- protax/         Main PROTAX-GPU code

```

# Compatibility:

| **System** | **CPU** | **NVIDIA GPU** | **Apple GPU** |
| --- | --- | --- | --- |
| **Linux** | yes | yes | n/a |
| **Mac X86_64** | yes | n/a | no |
| **MAC (ARM)** | yes | n/a | no |
| **Windows** | experimental | experimental | n/a |

# Installation:

These instructions are for Linux and MacOS. Windows support is experimental.

## 1. Set up CUDA (required for GPU support)

**IMPORTANT:** PROTAX-GPU requires a full local installation of CUDA, including development headers and tools, due to its use of custom CUDA kernels.

- Install CUDA 12.2 from the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-downloads)
- Install cuDNN 8.9 following the [official cuDNN installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

Ensure that your system environment variables are correctly set up to point to your CUDA installation.

**NOTE:** While JAX offers an easier CUDA installation via pip wheels for some platforms, this method does not provide the full CUDA toolkit required by PROTAX-GPU. You must perform a local CUDA installation as described above.

## 2. Create & activate new environment (recommended)

Using Conda:
```
conda create -n [name] python=3.12
conda activate [name]
```

## 3. Install dependencies

Install CMake:
```
conda install -c conda-forge cmake
```

## 4. Install JAX and jaxlib

| **System** | **Type** | **Command** |
| --- | --- | --- |
| **Linux** | CPU | `pip install "jax[cpu]"`|
| **Linux** | GPU | `pip install "jax[cuda122]"`
| **MacOS** | CPU | `pip uninstall jax jaxlib && conda install -c conda-forge jax=0.4.26 jaxlib=0.4.23`|
| **MacOS** | GPU | **Not yet supported**|

**NOTE:** The GPU installation command assumes you have already installed CUDA 12.2 as per step 1.

## 5. Install PROTAX-GPU

Clone this repository:
```
git clone https://github.com/uoguelph-mlrg/PROTAX-GPU.git
cd PROTAX-GPU
```

Install requirements:
```
pip install -r requirements.txt
```

Finally, install PROTAX-GPU:
```
pip install .
```

This will install a package called `protax` in your environment.

**NOTE:** CUDA 11.2 is also supported by JAX, but support for it will be dropped in the future. As long as the JAX version supports CUDA 11.2 and is greater than or equal to 0.4.14, this should work. However, ensure that your local CUDA installation matches the version you're using with JAX.

**TODO**: Add Windows support instructions and check macOS GPU support

If you are on MacOS and facing installation issues, run the following commands
```
chmod +x ./scripts/fix_librhash.sh
./scripts/fix_librhash.sh
```


# Usage
Instructions for running PROTAX-GPU for inference and training.

## Inference
Once you have a trained model, you can use the classify_file function to classify query sequences.

Run the sequence classification script:
```
python scripts/process_seqs.py [PATH_TO_QUERY_SEQUENCES] [PATH_TO_MODEL] [PATH_TO_TAXONOMY] [PATH_TO_TAXONOMY_MAPPING]
```
Example:

```
python scripts/process_seqs.py data/refs.aln models/params/model.npz models/ref_db/taxonomy37k.npz data/tax_mapping.priors
```
<!-- python scripts/process_seqs.py FinPROTAX/FinPROTAX/modelCOIfull/refs.aln models/params/model.npz models/ref_db/taxonomy37k.npz -->

Arguments:

- `PATH_TO_QUERY_SEQUENCES`: File containing the sequences to classify (e.g., FASTA or alignment file)(Can use refs.aln from [FinPROTAX](https://github.com/psomervuo/FinPROTAX/tree/main) for experiment)
- `PATH_TO_MODEL`: Path to the model. (Base Model is available in `models/params/model.npz`)
- `PATH_TO_TAXONOMY`: Path to the taxonomy .npz file. (taxonomy file is available in `models/ref_db/taxonomy37k.npz`)
- `PATH_TO_TAXONOMY_MAPPING`: Path for taxonomy mapping (node to label mapping) (Can use taxonomy.prior from [FinPROTAX](https://github.com/psomervuo/FinPROTAX/tree/main) for experiment)

Results are saved to `pyprotax_results.csv`

<details>
<summary> <b>More details regarding input file formats </b></summary>

#### Structure of Query File (`PATH_TO_QUERY_SEQUENCES`)

The query file (`qdir`) contains data in the following structure:

1. **Header Line (Taxonomic Metadata)**  
   - Starts with `>` followed by a unique identifier for the query sequence.  
   - Contains the full taxonomic lineage associated with the query, separated by commas.  
   - Example:  
     ```
     >COLFA029-10	Insecta,Coleoptera,Ptiliidae,Acrotrichinae,Acrotrichini,Acrotrichis
     ```

2. **Sequence Line**  
   - Contains the DNA sequence corresponding to the query.  
   - The sequence can include standard nucleotide codes (e.g., A, T, C, G).  

   #### Structure of `PATH_TO_MODEL` (`.npz` file)

The `par_dir` file is a compressed `.npz` archive that contains the trained parameters of the PROTAX model. The file should include the following arrays:

1. **`beta`**  
   - Shape: `(M, R)`  
   - Description: Coefficients for the regression model, where `M` is the number of features and `R` is the number of ranks in the taxonomy.  

2. **`scalings`**  
   - Shape: `(R, 4)`  
   - Description: Scaling parameters for the regression model. Each row corresponds to a rank in the taxonomy and contains four values:
     - Mean scaling (columns 0 and 2).
     - Variance scaling (columns 1 and 3).  

3. **`node_layer`**  
   - Shape: `(N,)`  
   - Description: Indicates the layer (rank) in the taxonomy tree to which each node belongs. 

#### Structure of the `PATH_TO_TAXONOMY` `.npz` File

The `tax_dir` file is a serialized representation of the taxonomy and sequence data required by the PROTAX-GPU model. Below is the description of the required structure for the `.npz` file:

1. **`refs`**: A 2D array of shape `(R, L)`, where `R` is the number of reference sequences and `L` is the sequence length. Each row represents a reference sequence.

2. **`ok_pos`**: A binary 2D array of shape `(R, L)`, indicating valid positions (non-missing data) for each reference sequence.

3. **`priors`**: A 1D array of length `N`, where `N` is the number of nodes in the taxonomy. It specifies prior probabilities for each node.

4. **`segments`**: A 1D array of length `N`, containing segment identifiers for each node in the taxonomy.

5. **`paths`**: A 2D array of shape `(N, D)`, where `D` is the maximum depth of the taxonomy. Each row represents the path from the root node to a specific taxon.

6. **`node_state`**: A 2D array of shape `(N, S)`, where `S` is the state size (typically 2). Contains state information for each node in the taxonomy.

7. **`ref_rows` and `ref_cols`**: Two 1D arrays defining the row and column indices for mapping reference sequences to nodes in the taxonomy. These are used to construct a sparse binary matrix (`node2seq`).

8. **`node_layer`**: A 1D array of length `N`, defining the taxonomic layer (or rank) for each node.



#### Taxonomy Mapping File (`PATH_TO_TAXONOMY_MAPPING`) Format

The `tax_map` file defines the mapping between node numbers and their corresponding taxonomy labels in the taxonomy tree. Each line in the file represents a single node and its associated data.

The `tax_map` file should be a tab-separated text file , where each line contains information for one node. It must adhere to the following format:

| Column Name    | Description                                                               |
|----------------|---------------------------------------------------------------------------|
| **Node Number** | A unique integer ID representing a node in the taxonomy tree.            |
| **Other Fields**| Possible Additional metadata or numeric attributes (optional).                    |
| **Taxonomy Label** | A string representing the taxonomy label for the node (e.g., "Insecta"). |


</details>


## Output 
The `pyprotax_results.csv` file contains the taxonomic labels and their associated probabilities for each query, organized based on the hierarchical traversal of the taxonomy. The structure of the output aligns with the traversal paths of the taxonomy from the root to the leaves, ensuring consistency in representation across runs. Here’s a detailed explanation of the output:

- **Columns 1–7**: These columns represent the taxonomic labels at each level of the hierarchy. Each row corresponds to a single query, and these columns specify the path from the root node to the assigned taxon. For instance, a row might represent the path `[Insecta, Coleoptera, Ptiliidae, Acrotrichinae Acrotrichini Acrotrichis Acrotrichis_rugulosa]`, reflecting the taxonomic lineage of a species.

- **Columns 8–14**: These columns contain the probabilities associated with each taxonomic label at every level of the hierarchy for the given query. These probabilities are computed as the product of branch probabilities along the path, as determined by the traversal of the taxonomy.


## Training
Run the script from the command line: You need to specify the paths to your training data and target data using the `--train_dir` and `--targ_dir` arguments, respectively.
```
python scripts/train_model.py --train_dir [PATH_TO_TRAINING_DATA] --targ_dir [PATH_TO_TARGET_DATA]
```

Command Line Arguments
- `--train_dir`: Path to the training data (e.g., refs.aln).
- `--targ_dir`: Path to the target data (e.g., targets.csv).

Training Configuration
The script uses the following training parameters:

- Learning rate: `0.001`
- Batch size: `500`
- Number of epochs: `30`
These parameters are predefined in the script and can be modified if needed by editing the dictionary `tc` in the code.

The script uses `models/params/model.npz` as baseline and saves the trained model at `models/params/m2.npz`

# Hardware
To reproduce the BOLD dataset experiments, PROTAX-GPU requires an NVIDIA GPU with at least 16GB VRAM and CUDA compute capability 6.0 or later. This corresponds to GPUs in the NVIDIA Pascal, NVIDIA Volta™, NVIDIA Turing™, NVIDIA Ampere architecture, and NVIDIA Hopper™ architecture families.

# Datasets
The BOLD 7.8M dataset is available [here](https://www.boldsystems.org/index.php/datarelease). The dataset is not included in this repository due to its size.

The smaller FinPROTAX dataset is included in the `models` directory, sourced from [here](https://github.com/psomervuo/FinPROTAX).

# License
This project is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License - see the [LICENSE](LICENSE) file for details.
