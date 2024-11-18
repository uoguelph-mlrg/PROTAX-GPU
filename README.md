# PROTAX-GPU

![alt text](img/Block_diagram_upd.png?raw=true)

A GPU-accelerated JAX-based implementation of [PROTAX](https://pubmed.ncbi.nlm.nih.gov/27296980/). Contains all code and experiments for PROTAX-GPU

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
python scripts/process_seqs.py [PATH_TO_QUERY_SEQUENCES] [PATH_TO_MODEL] [PATH_TO_TAXONOMY]
```
Example:

```
python scripts/process_seqs.py data/refs.aln models/params/model.npz models/ref_db/taxonomy37k.npz
```
<!-- python scripts/process_seqs.py FinPROTAX/FinPROTAX/modelCOIfull/refs.aln models/params/model.npz models/ref_db/taxonomy37k.npz -->

Arguments:

- `PATH_TO_QUERY_SEQUENCES`: File containing the sequences to classify (e.g., FASTA or alignment file)(Can use refs.aln from [FinPROTAX](https://github.com/psomervuo/FinPROTAX/tree/main) for experiment)
- `PATH_TO_MODEL`: Path to the model. (Base Model is available in `models/params/model.npz`)
- `PATH_TO_TAXONOMY`: Path to the taxonomy .npz file. (taxonomy file is available in `models/ref_db/taxonomy37k.npz`)


Results are saved to `pyprotax_results.csv`

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
