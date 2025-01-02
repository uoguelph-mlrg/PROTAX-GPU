#ifndef _GPU_OPS_H
#define _GPU_OPS_H

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace knn {

struct KNNDescriptor {
  int rows;   // number of rows in sparse array
  int k;      // number of neighbors
  int B;      // batch size
};

void gpu_knn_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

void gpu_knn_finprotax_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

void gpu_knn_v2_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

}  // namespace knn

#endif
