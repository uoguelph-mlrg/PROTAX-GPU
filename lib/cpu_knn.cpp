#include "cpu_ops.h"


template <typename T>
void min_k_impl(const int N, const int row, const int* indptr, const int* indices,
                 const T* data, T* result){ 
    // get the start and end indices for this row
    int start = indptr[row];
    int end = indptr[row + 1];

    // init result
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

    // loop over the row
    for (int i = start; i < end; i++) {
        const T val = data[i];
        if (val < m1) {
            m2 = m1;
            m1 = val;
        } else if (val < m2) {
            m2 = val;
        }
    }

    m2-=m1;
    // write out the result
    result[row*2] = m1;
    result[row*2+1] = m2;
}

// --------------------------------------------------

namespace cpu_ops {


void cpu_min_k_row_f32(const int N, const int row, const int *indptr,
                       const int *indices, const float *data, float *result){
    min_k_impl(N, row, indptr, indices, data, result);
}


} // namespace cpu_ops
