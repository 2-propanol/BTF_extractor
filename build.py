import platform
from setuptools import Extension

import numpy
from Cython.Build import cythonize

compile_args = []
link_args = []

pf = platform.system()
if pf == "Windows":
    # for MSVC
    compile_args = ["/std:c++14", "/DNOMINMAX", "/O2", "/openmp"]
elif pf == "Darwin":
    # for clang
    compile_args = ["-std=c++14", "-O2", "-march=native", "-Xpreprocessor", "-fopenmp"]
    link_args = ["-lomp"]
elif pf == "Linux":
    # for gcc
    compile_args = ["-std=c++14", "-Ofast", "-march=native", "-fopenmp"]
    link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        name="ubo2014_cy",
        sources=["btf_extractor/ubo2014.pyx"],
        include_dirs=[numpy.get_include(), "btf_extractor/c_ext"],
        define_macros=[("BTF_IMPLEMENTATION", "1"), ("NPY_NO_DEPRECATED_API", "1")],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++",
    )
]


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update(
        {"ext_modules": cythonize(ext_modules)}
    )
    return setup_kwargs


if __name__ == "__main__":
    build({})
