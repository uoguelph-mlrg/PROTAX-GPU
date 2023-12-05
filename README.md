# PROTAX-GPU

![alt text](img/Block_diagram_upd.pdf?raw=true)

Code and experiments for PROTAX-GPU.
A GPU-accelerated JAX-based implementation of [PROTAX](https://pubmed.ncbi.nlm.nih.gov/27296980/) 

# Functionality
Estimates the probability of each outcome in a taxonomic tree given a query barcode sequence compared to reference sequences.

Requirements:
- Python >=3.9
- JAX 0.4.14
- JAX 0.4.14+cuda12.cudnn89
- NumPy 1.23.1
- chex 0.1.82 

Built & Tested on:
- linux
- CMake 3.27.5
- CUDA 12.2


# Installation:

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
