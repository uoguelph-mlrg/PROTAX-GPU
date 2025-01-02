#ifndef _CPU_OPS_H
#define _CPU_OPS_H

namespace cpu_ops {

void cpu_min_k_row_f32(const int N, const int row, const int *indptr, const int *indices, const float *data, float *result);

}  // namespace cpu_ops

#endif
