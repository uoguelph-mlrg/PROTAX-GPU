import os, subprocess

import pybind11
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.realpath(__file__))
pb11_dir = pybind11.get_include()

class CMakeBuildExt(build_ext):
    def build_extensions(self):
        """
        Build all extensions with CMake.
        """
        # First: configure CMake build
        import platform
        import sys
        import sysconfig


        # Work out the relevant Python paths to pass to CMake, adapted from the
        # PyTorch build system
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
        else:
            cmake_python_library = "{}/{}".format(
                sysconfig.get_config_var("LIBDIR"),
                sysconfig.get_config_var("INSTSONAME"),
            )
        cmake_python_include_dir = sysconfig.get_paths()['include']

        install_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath("dummy"))
        )
        os.makedirs(install_dir, exist_ok=True)
        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
            "-DPython_EXECUTABLE={}".format(sys.executable),
            "-DPython_LIBRARIES={}".format(cmake_python_library),
            "-DPython_INCLUDE_DIRS={}".format(cmake_python_include_dir),
            "-DCMAKE_BUILD_TYPE={}".format(
                "Debug" if self.debug else "Release"
            ),
            "-DCMAKE_PREFIX_PATH={}".format(pybind11.get_cmake_dir()),
        ]

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", HERE] + cmake_args, cwd=self.build_temp
        )

        # Build all the extensions
        super().build_extensions()

        # Finally run install
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"],
            cwd=self.build_temp,
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

# TODO: check if gpu support is available

extensions = [
    Extension(
        name="knn_jax.cpu_ops",
        sources=["lib/knn_cpu.cpp"]
    ),
    Extension(
        name="knn_jax.gpu_ops",
        sources=['lib/knn_dispatch.cpp', 'lib/knn_kernels.cu'],
        include_dirs=[pb11_dir],
    ),
]

pdirs = {
    "protax": "src/protax",
    "knn_jax": "src/knn_jax"
}


# https://setuptools.pypa.io/en/latest/references/keywords.html#keywords
setup(
    name="protax",                                  # name of package
    author="Roy Li",
    author_email="roymy.li12@gmail.com",
    url="github.com/uoguelph-mlrg/protax-gpu",
    # license="MIT",
    description=(
        "GPU accelerated DNA barcoding"
    ),
    long_description_content_type="text/markdown",
    packages=["protax", "knn_jax"],                    # packages to include in final knn_jax package
    package_dir=pdirs,                          # root directory for included python packages
    include_package_data=True,
    install_requires=["jax", "jaxlib"],
    extras_require={"test": "pytest"},
    ext_modules=extensions,
    cmdclass={"build_ext": CMakeBuildExt},
)