#include "pybind11_kernel_helpers.h"

using namespace knn;

namespace {


template <typename T>
void cpu_min_k(const int N, const int* indptr, const int* indices,
                 const float* data, float* result){

    
}
} // namespace

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_knn_f32"] = EncapsulateFunction(cpu_min_k<float>);