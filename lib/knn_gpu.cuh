#ifndef _KNN_GPU_H
#define _KNN_GPU_H


// exposes cuda implementations of knn
namespace knn_gpu{

__global__ void min_k_v2_fp32(const float* data, int start, int end, int row, float* result);

__global__ void min_k_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result);

__global__ void min_k_finprotax_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result);
}  // knn_gpu

#endif
