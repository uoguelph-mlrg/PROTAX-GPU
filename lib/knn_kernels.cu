// This file contains the CUDA kernels for computing
// the K-nearest neighbors of a sparse array

#include "kernels.h"
#include "kernel_helpers.h"

#define THREADS_PER_BLOCK 128

namespace knn{

__global__ void reduce_min_k(const float* data, int start, int end, int row, float* result){
    if (end-start <= 0){
        return;
    }
    // parallel reduction to find min 2 elements in a segment of data
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

__global__ void min_k(const int N, const int* indptr, const int* indices,
                 const float* data, float* result) {
    // get the row index, i.e. node index
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= N){
        return;
    }

    // get the start and end index of nonzero values in this row
    int start = indptr[i];
    int end = indptr[i + 1];

    // initialize the minimum value to 1.0
    float min_val = 1;
    float m2 = 1;

    if (end-start <= 0){
        m2 = 0;
        min_val = 0;
    }
    else{
        m2 = data[start];
        min_val = data[start];
        start++;
    }

    // loop over the nonzero values in this row
    for (int j = start; j < end; j++) {
        float val = data[j];
        // update the minimum value if needed
        if (val < min_val) {
            m2 = min_val;
            min_val = val;
        }
        else if (val < m2){
          m2 = val;
        }
    }

    // we actually store the diff
    m2 = m2-min_val;
    // store the minimum value in the result array
    result[i*2] = min_val;
    result[i*2 + 1] = m2;
}


template <typename T>
inline void dispatch_min_k(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len) {
    // get problem size
    const KNNDescriptor &d = *UnpackDescriptor<KNNDescriptor>(opaque, opaque_len);
    const int N = d.rows;

    const int *indptr = reinterpret_cast<const int *>(buffers[0]);
    const int *indices = reinterpret_cast<const int *>(buffers[1]);
    T *data = reinterpret_cast<T *>(buffers[2]);
    T *result = reinterpret_cast<T *>(buffers[3]);

    // ceil(N/THREADS_PER_BLOCK)
    min_k<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(N, indptr, indices, data, result);
}

template <typename T>
inline void dispatch_min_k_v2(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len) {
    // get problem size
    const KNNDescriptor &d = *UnpackDescriptor<KNNDescriptor>(opaque, opaque_len);
    const int N = d.rows;

    const int *indptr = reinterpret_cast<const int *>(buffers[0]);
    const int *indices = reinterpret_cast<const int *>(buffers[1]);
    T *data = reinterpret_cast<T *>(buffers[2]);
    T *result = reinterpret_cast<T *>(buffers[3]);

    for (int i = 0; i < N; i++){
        int start = indptr[i];
        int end = indptr[i + 1];
        int size = end-start;
        int num_blocks = (size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        int shared_mem_size = THREADS_PER_BLOCK*sizeof(T);
        reduce_min_k<<<num_blocks, THREADS_PER_BLOCK, shared_mem_size, stream>>>(data+start, 0, size, i, result);
    }
}

/**
 * XLA custom call target which calls kernel on correct type
*/
void gpu_knn_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len) {
    dispatch_min_k<float>(stream, buffers, opaque, opaque_len);
}

void gpu_knn_v2_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len) {
    dispatch_min_k_v2<float>(stream, buffers, opaque, opaque_len);
}

} // namespace knn 
