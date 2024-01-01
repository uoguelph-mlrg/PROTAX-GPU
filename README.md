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
| **MAC (ARM)** | experimental | n/a | no |
| **Windows** | experimental | experimental | n/a |

# Installation:

See `requirements.txt` for a list of what is required to run PROTAX-GPU. These instructions are for Linux and MacOS. Windows support is experimental.

**1. Create & activate new environment (recommended)**

Conda:
```
conda create -n [name] python=3.10
conda activate [name]
```
**2. Install dependencies**

Cmake:
```
conda install -c conda-forge cmake
```

CUDA 12 + cuDNN:
```
conda install -c nvidia cudatoolkit=12.2 cudnn=8.9
```

**NOTE:** CUDA 11.2 is also supported by jax, but support for it will be dropped in the future. As long as the JAX version supports CUDA 11.2, and is greater than or equal to 0.4.14, this should work. 

**3. Install JAX and jaxlib**

| **System** | **Type** | **Command** |
| --- | --- | --- |
| **Linux** | CPU | `pip install "jax[cpu]"`|
| **Linux** | GPU | `pip install "jax[cuda122]"`
| **MacOS** | CPU | `pip install "jax[cpu]" "jaxlib[cpu]"`|
| **MacOS** | GPU | **TODO**|

**TODO**: Windows support, check macOS GPU support

**4. Install PROTAX-GPU**

Clone this repository.
```
git clone https://github.com/uoguelph-mlrg/PROTAX-GPU.git
```

Install `requirements.txt`
```
pip install -r requirements.txt
```

Finally, install PROTAX-GPU. This will install a package called `protax` in your environment.
```
pip install .
```

# Usage
Instructions for running PROTAX-GPU for inference and training.

## Inference
**TODO**

## Training
**TODO**

# Hardware
To reproduce the BOLD dataset experiments, PROTAX-GPU requires an NVIDIA GPU with at least 16GB VRAM and CUDA compute capability 6.0 or later. This corresponds to GPUs in the NVIDIA Pascal, NVIDIA Volta™, NVIDIA Turing™, NVIDIA Ampere architecture, and NVIDIA Hopper™ architecture families.

# License
This project is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License - see the [LICENSE](LICENSE) file for details.
