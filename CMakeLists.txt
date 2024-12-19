# cmake boilerplate
cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
message(STATUS "Using CMake version " ${CMAKE_VERSION})

# libs
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/lib)

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
pybind11_add_module(                                              # wraps cmake's add_library()
  cpu_ops      # exposed to python as protax.ops.gpu_ops
  ${CMAKE_SOURCE_DIR}/lib/knn_cpu.cpp
)
# add shared lib to site-packages
install(TARGETS cpu_ops DESTINATION ${SKBUILD_PROJECT_NAME}/ops)

# GPU ops
if(CUDA_FOUND)
  pybind11_add_module(
    gpu_ops    # exposed to python as protax.ops.gpu_ops
    ${CMAKE_SOURCE_DIR}/lib/knn_kernels.cu
    ${CMAKE_SOURCE_DIR}/lib/knn_dispatch.cpp
  )
  install(TARGETS gpu_ops DESTINATION ${SKBUILD_PROJECT_NAME}/ops)
else()
  message(STATUS "Skipping GPU ops since CUDA is not available.")
endif()

