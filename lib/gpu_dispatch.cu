// implements gpu_ops.h

#include "knn_gpu.cuh"
#include "gpu_ops.h"
#include "kernel_helpers.h"

// TODO: query device for this
#define THREADS_PER_BLOCK 64

namespace knn{

// --------------------------------------------------
//                   Dispatchers
// --------------------------------------------------

template <typename T, typename K>
void dispatch_min_k(K kern, cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
    // get problem size
    const KNNDescriptor &d = *xla_helpers::UnpackDescriptor<KNNDescriptor>(opaque, opaque_len);
    const int N = d.rows;

    const int *indptr = reinterpret_cast<const int *>(buffers[0]);
    const int *indices = reinterpret_cast<const int *>(buffers[1]);
    T *data = reinterpret_cast<T *>(buffers[2]);
    T *result = reinterpret_cast<T *>(buffers[3]);

    // ceil(N/THREADS_PER_BLOCK)
    kern<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(N, indptr, indices, data, result);
}

// --------------------------------------------------

template <typename T, typename K>
void dispatch_min_k_v2(K kern, cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
    // get problem size
    const KNNDescriptor &d = *xla_helpers::UnpackDescriptor<KNNDescriptor>(opaque, opaque_len);
    const int N = d.rows;

    const int *indptr = reinterpret_cast<const int *>(buffers[0]);
    T *data = reinterpret_cast<T *>(buffers[2]);
    T *result = reinterpret_cast<T *>(buffers[3]);

    for (int i = 0; i < N; i++){
        int start = indptr[i];
        int end = indptr[i + 1];
        int size = end-start;
        int num_blocks = (size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        int shared_mem_size = THREADS_PER_BLOCK*sizeof(T);
        kern<<<num_blocks, THREADS_PER_BLOCK, shared_mem_size, stream>>>(data+start, 0, size, i, result);
    }
}

// --------------------------------------------------
//              XLA custom call targets
// --------------------------------------------------
void gpu_knn_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
    // handled by template deduction
    dispatch_min_k<float>(knn_gpu::min_k_finprotax_fp32, stream, buffers, opaque, opaque_len);
}

void gpu_knn_finprotax_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
    dispatch_min_k<float>(knn_gpu::min_k_fp32, stream, buffers, opaque, opaque_len);
}

void gpu_knn_v2_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
    dispatch_min_k_v2<float>(knn_gpu::min_k_v2_fp32, stream, buffers, opaque, opaque_len);
}

} // namespace knn 
