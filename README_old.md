# PROTAX-py
Python implementation of [PROTAX](https://pubmed.ncbi.nlm.nih.gov/27296980/) 

# Functionality
Estimates the probability of each outcome in a taxonomic tree given a query sequence compared to reference sequences.

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
1. Navigate to the root folder
2. run `pip install .` to build custom CUDA kernels and create python bindings with CMake
3. test the functionality of `knn-utils` by running `tests/test_topk.py`

Everything is now ready for model inference. Classification function in `classify.py`
should work now