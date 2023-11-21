cmake_minimum_required(VERSION 3.10)
project(knn_jax LANGUAGES CXX CUDA)

message(STATUS "Using CMake version " ${CMAKE_VERSION})
message(${CMAKE_SOURCE_DIR})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDA)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# cpu op library
pybind11_add_module(
  cpu_ops
  ${CMAKE_CURRENT_LIST_DIR}/lib/knn_cpu.cpp
)
install(TARGETS cpu_ops DESTINATION knn_jax)

if(NOT CUDA_FOUND)
  message(STATUS "CUDA not found, skipping GPU op compilation")
  return()
endif()

# gpu op library
pybind11_add_module(
  gpu_ops
  ${CMAKE_CURRENT_LIST_DIR}/lib/knn_kernels.cu
  ${CMAKE_CURRENT_LIST_DIR}/lib/knn_dispatch.cpp
)
install(TARGETS gpu_ops DESTINATION knn_jax)