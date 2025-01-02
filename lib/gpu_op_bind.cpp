// creates bindings for gpu ops
#include "gpu_ops.h"
#include "pybind11_kernel_helpers.h"

namespace {

// python dict holding function pointers
pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["gpu_knn_f32"] = xla_helpers::EncapsulateFunction(knn::gpu_knn_f32);
  dict["gpu_knn_v2_f32"] = xla_helpers::EncapsulateFunction(knn::gpu_knn_v2_f32);
  return dict;
}

// define python module: gpu ops
// expose registrations dict and build_knn_descriptor()
PYBIND11_MODULE(gpu_ops, m){
    m.def("registrations", &Registrations);
    m.def("build_knn_descriptor",
          [](int rows) { return xla_helpers::PackDescriptor(knn::KNNDescriptor{rows, 2, 1}); });
    m.def("build_knn_descriptor_batched",
          [](int rows, int batch_size) { return xla_helpers::PackDescriptor(knn::KNNDescriptor{rows, 2, batch_size}); });
}
} 

