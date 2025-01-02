# cmake boilerplate
cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
message(STATUS "Using CMake version " ${CMAKE_VERSION})

# libs
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# --------------------------------------------------
#               Platform detection 
# --------------------------------------------------
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "Mac detected. Skipping CUDA configuration.")
    set(CUDA_FOUND OFF)
else()
    find_package(CUDA)
    if (CUDA_FOUND)
        message (STATUS "CUDA found at" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        enable_language(CUDA)
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    else()
        message(WARNING "CUDA not found/supported")
    endif()
endif()

# --------------------------------------------------
#                    Binaries
# --------------------------------------------------
# debug - toggle in pyproject.toml
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Building in debug mode.")
  # install(TARGETS debug DESTINATION ${SKBUILD_PROJECT_NAME}/debug)
endif()

# CPU ops
# exposed to python as protax.ops.gpu_ops
pybind11_add_module(                                              # wraps cmake's add_library()
  cpu_ops
  ${CMAKE_SOURCE_DIR}/lib/cpu_op_bind.cpp
  ${CMAKE_SOURCE_DIR}/lib/cpu_knn.cpp
  ${CMAKE_SOURCE_DIR}/lib/cpu_ops.h
  ${CMAKE_SOURCE_DIR}/lib/pybind11_kernel_helpers.h
)
install(TARGETS cpu_ops DESTINATION ${SKBUILD_PROJECT_NAME}/ops)  # add shared lib to site-packages

# GPU ops
if(CUDA_FOUND)
  # exposed to python as protax.ops.gpu_ops
  pybind11_add_module(
    gpu_ops
    ${CMAKE_SOURCE_DIR}/lib/gpu_op_bind.cpp
    ${CMAKE_SOURCE_DIR}/lib/gpu_ops.h
    ${CMAKE_SOURCE_DIR}/lib/gpu_dispatch.cu
    ${CMAKE_SOURCE_DIR}/lib/knn_gpu.cuh
    ${CMAKE_SOURCE_DIR}/lib/knn_gpu.cu
    ${CMAKE_SOURCE_DIR}/lib/kernel_helpers.h
    ${CMAKE_SOURCE_DIR}/lib/pybind11_kernel_helpers.h
  )
  install(TARGETS gpu_ops DESTINATION ${SKBUILD_PROJECT_NAME}/ops)
else()
  message(STATUS "Skipping GPU ops since CUDA is not available.")
endif()

