#include "pybind11_kernel_helpers.h"

using namespace knn;

namespace {

template <typename T>
void cpu_min_k_row(const int N, const int row, const int* indptr, const int* indices,
                 const float* data, float* result){ 
    // get the start and end indices for this row
    int start = indptr[row];
    int end = indptr[row + 1];

    // init result
    float m1 = 1;
    float m2 = 1;

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
        const float val = data[i];
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

template <typename T>
void cpu_knn_f32(void *out, const void **in){
    const int N = *static_cast<const int*>(in[0]);
    const int* indptr = static_cast<const int*>(in[1]);
    const int* indices = static_cast<const int*>(in[2]);
    const T* data = static_cast<const float*>(in[3]);
    T* result = static_cast<float*>(out);

    // loop over the rows
    for (int row = 0; row < N; row++) {
        cpu_min_k_row<float>(N, row, indptr, indices, data, result);
    }
}

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["cpu_knn_f32"] = EncapsulateFunction(cpu_knn_f32<float>);
    return dict;
}

PYBIND11_MODULE(cpu_ops, m){
    m.def("registrations", &Registrations);
}

} // namespace