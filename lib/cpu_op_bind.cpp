#include "pybind11_kernel_helpers.h"
#include "cpu_ops.h"

namespace cpu_ops{

// --------------------------------------------------
//                    Dispatchers
// --------------------------------------------------
template <typename T>
void cpu_knn_f32(void *out, const void **in){
    const int N = *static_cast<const int*>(in[0]);
    const int* indptr = static_cast<const int*>(in[1]);
    const int* indices = static_cast<const int*>(in[2]);
    const T* data = static_cast<const float*>(in[3]);
    T* result = static_cast<float*>(out);

    // loop over the rows
    for (int row = 0; row < N; row++) {
        cpu_min_k_row_f32(N, row, indptr, indices, data, result);
    }
}

// --------------------------------------------------
//                     Bindings
// --------------------------------------------------
pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["cpu_knn_f32"] = xla_helpers::EncapsulateFunction(cpu_knn_f32<float>);
    return dict;
}

PYBIND11_MODULE(cpu_ops, m){
    m.def("registrations", &Registrations);
}

} // namespace
