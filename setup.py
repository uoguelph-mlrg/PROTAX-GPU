import os
import subprocess

import pybind11
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import platform

HERE = os.path.dirname(os.path.realpath(__file__))
pb11_dir = pybind11.get_include()
import shutil
# Function to check if CUDA is available
def cuda_available():
    return shutil.which("nvcc") is not None



class CMakeBuildExt(build_ext):
    def build_extensions(self):
        """
        Build all extensions with CMake.
        """
        # First: configure CMake build
        import sys
        import sysconfig
        print("cuda_available")
        print(cuda_available())
        # Work out the relevant Python paths to pass to CMake, adapted from the
        # PyTorch build system
        install_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath("dummy")))
        os.makedirs(install_dir, exist_ok=True)

        # Determine if we're on an ARM-based Mac to disable CUDA
        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
            "-DPython_EXECUTABLE={}".format(sys.executable),
            "-DCMAKE_BUILD_TYPE={}".format("Debug" if self.debug else "Release"),
            "-DCMAKE_PREFIX_PATH={}".format(pybind11.get_cmake_dir()),
        ]

        # Specific handling for Windows platform
        if platform.system() == "Windows":
            cmake_python_library = "{}/libs/python{}.lib".format(
                sysconfig.get_config_var("prefix"),
                sysconfig.get_config_var("VERSION"),
            )
            if not os.path.exists(cmake_python_library):
                cmake_python_library = "{}/libs/python{}.lib".format(
                    sys.base_prefix,
                    sysconfig.get_config_var("VERSION"),
                )
            cmake_args.append("-DPython_LIBRARIES={}".format(cmake_python_library))
        else:
            cmake_python_library = "{}/{}".format(
                sysconfig.get_config_var("LIBDIR"),
                sysconfig.get_config_var("INSTSONAME"),
            )
        cmake_python_include_dir = sysconfig.get_paths()["include"]
        cmake_args.extend(
            [
                "-DPython_LIBRARIES={}".format(cmake_python_library),
                "-DPython_INCLUDE_DIRS={}".format(cmake_python_include_dir),
            ]
        )

        # Add condition for ARM-based Macs to disable CUDA
        # if platform.system() == "Darwin" and platform.machine() == "x86_64":
        if not cuda_available():
            cmake_args.append("-DUSE_CUDA=OFF")

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(["cmake", HERE] + cmake_args, cwd=self.build_temp)

        # Call CMake build for all extensions
        super().build_extensions()

        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"], cwd=self.build_temp
        )

    def build_extension(self, ext):
        """
        Define the build step for a single extension with CMake.
        """
        target_name = ext.name.split(".")[-1]
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target_name],
            cwd=self.build_temp,
        )


extensions = [
    Extension(
        name="knn_jax.cpu_ops",
        sources=["lib/knn_cpu.cpp"],
        include_dirs=[pb11_dir],
    ),
]

# Add GPU ops only if not on ARM-based Mac
# if platform.system() != "Darwin":
if cuda_available():
    extensions.append(
        Extension(
            name="knn_jax.gpu_ops",
            sources=["lib/knn_dispatch.cpp", "lib/knn_kernels.cu"],
            include_dirs=[pb11_dir],
        )
    )

pdirs = {"protax": "src/protax", "knn_jax": "src/knn_jax"}


# https://setuptools.pypa.io/en/latest/references/keywords.html#keywords
setup(
    name="protax",  # name of package
    author="Roy Li",
    author_email="roymy.li12@gmail.com",
    url="github.com/uoguelph-mlrg/protax-gpu",
    # license="MIT",
    description=("GPU accelerated DNA barcoding"),
    long_description_content_type="text/markdown",
    # packages to include in final knn_jax package
    packages=["protax", "knn_jax"],
    package_dir=pdirs,  # root directory for included python packages
    include_package_data=True,
    install_requires=["jax", "jaxlib"],
    extras_require={"test": "pytest"},
    ext_modules=extensions,
    cmdclass={"build_ext": CMakeBuildExt},
)
