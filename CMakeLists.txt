cmake_minimum_required(VERSION 3.10)
project(knn_jax LANGUAGES CXX CUDA)

message(STATUS "Using CMake version " ${CMAKE_VERSION})
message(${CMAKE_SOURCE_DIR})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

pybind11_add_module(
  gpu_ops
  ${CMAKE_CURRENT_LIST_DIR}/lib/knn_kernels.cu
  ${CMAKE_CURRENT_LIST_DIR}/lib/knn_ops.cpp)
install(TARGETS gpu_ops DESTINATION knn_jax)