#include "knn_gpu.cuh"


// parallel reduction version of min k
template<typename T>
__device__ void min_k_v2_impl(const float* data, int start, int end, int row, float* result){
    if (end-start <= 0){
        return;
    }
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float sdata[];
    sdata[i] = data[i];

    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if (i < s){
            if (sdata[i] > sdata[i+s]){
                sdata[i] = sdata[i+s];
            }
        }
        __syncthreads();
    }

    if (i == 0){
        result[row*2] = sdata[0];
        result[row*2 + 1] = sdata[1];
    }
}

// --------------------------------------------------

// naive version of min k
template<typename T>
__device__ void min_k_impl(const int N, const int* indptr, const int* indices,
                 const T* data, T* result) {
    // get the row index, i.e. node index
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N){
        return;
    }
    
    // start-end segment for row
    int start = indptr[i];
    int end = indptr[i + 1];

    T m1 = 1;
    T m2 = 1;

    if (end-start <= 0){
        m2 = 0;
        m1 = 0;
    }
    else{
        m2 = data[start];
        m1 = data[start];
        start++;
    }

    // nonzero elements in row
    for (int j = start; j < end; j++) {
        T val = data[j];
        if (val < m1) {
            m2 = m1;
            m1 = val;
        }
        else if (val < m2){
          m2 = val;
        }
    }

    // store the minimum value in the result array
    result[i*2] = m1;
    result[i*2 + 1] = m2;
}


// min k variant used by finprotax
// stores top 1, and diff with 2nd
template<typename T>
__device__ void min_k_finprotax_impl(const int N, const int* indptr, const int* indices,
                 const T* data, T* result) {
    // get the row index, i.e. node index
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N){
        return;
    }
    
    // start-end segment for row
    int start = indptr[i];
    int end = indptr[i + 1];

    T m1 = 1;
    T m2 = 1;

    if (end-start <= 0){
        m2 = 0;
        m1 = 0;
    }
    else{
        m2 = data[start];
        m1 = data[start];
        start++;
    }

    // nonzero values in this row
    for (int j = start; j < end; j++) {
        T val = data[j];
        if (val < m1) {
            m2 = m1;
            m1 = val;
        }
        else if (val < m2){
          m2 = val;
        }
    }
    m2 = m2-m1;

    // store the minimum value in the result array
    result[i*2] = m1;
    result[i*2 + 1] = m2;
}


// define kernels specified in knn_gpu.cuh
namespace knn_gpu{


__global__ void min_k_finprotax_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result){
    // this will be inlined by nvcc
    min_k_finprotax_impl<float>(N, indptr, indices, data, result);
}

__global__ void min_k_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result){
    min_k_impl<float>(N, indptr, indices, data, result);
}


__global__ void min_k_v2_fp32(const float* data, int start, int end, int row, float* result){
    min_k_v2_impl<float>(data, start, end, row, result);
}


}  // namespace knn_gpu

