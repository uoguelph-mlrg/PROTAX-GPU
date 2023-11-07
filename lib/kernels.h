#ifndef _KNN_JAX_KERNELS_H_
#define _KNN_JAX_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace gpu_ops {

struct KNNDescriptor {
  int rows;   // number of rows in sparse array
  int k;      // number of neighbors
  int B;      // batch size
};

// TODO make a templated version of this function
void gpu_knn_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

void gpu_knn_v2_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

}  // namespace gpu_ops
#endif