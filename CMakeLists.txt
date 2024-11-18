cmake_minimum_required(VERSION 3.10)
project(knn_jax LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})
message(${CMAKE_SOURCE_DIR})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

# if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" AND CMAKE_SYSTEM_NAME MATCHES "Darwin")
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "Mac detected. Skipping CUDA configuration.")
    set(CUDA_FOUND OFF)
else()
    find_package(CUDA)
    if (CUDA_FOUND)
        enable_language(CUDA)
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    else()
        message(WARNING "CUDA not found or not supported. Skipping GPU ops.")
        set(CUDA_FOUND OFF)
    endif()
endif()

# CPU op library
pybind11_add_module(
  cpu_ops
  ${CMAKE_CURRENT_LIST_DIR}/lib/knn_cpu.cpp
)
install(TARGETS cpu_ops DESTINATION knn_jax)

# GPU op library (only if CUDA is found)
if(CUDA_FOUND)
  pybind11_add_module(
    gpu_ops
    ${CMAKE_CURRENT_LIST_DIR}/lib/knn_kernels.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/knn_dispatch.cpp
  )
  install(TARGETS gpu_ops DESTINATION knn_jax)
else()
  message(STATUS "Skipping GPU ops build since CUDA is not available.")
endif()
