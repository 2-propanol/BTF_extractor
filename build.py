import platform
from distutils.core import Extension
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

import numpy
from Cython.Distutils import build_ext

compile_args = []

pf = platform.system()
if pf == "Windows":
    # for MSVC
    compile_args = ["/std:c++14", "/DNOMINMAX", "/O2"]
elif pf == "Darwin":
    # for clang
    compile_args = ["-std=c++14", "-O2", "-march=native"]
elif pf == "Linux":
    # for gcc
    compile_args = ["-std=c++14", "-Ofast", "-march=native"]

ext_modules = [
    Extension(
        "ubo2014_cy",
        sources=["btf_extractor/ubo2014.pyx"],
        include_dirs=[numpy.get_include(), "btf_extractor/c_ext"],
        define_macros=[("BTF_IMPLEMENTATION", "1")],
        extra_compile_args=compile_args,
        language='c++'
    )
]

class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):
    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed("File not found. Could not compile C extension.")

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed("Could not compile C extension.")


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update(
        {"ext_modules": ext_modules, "cmdclass": {"build_ext": ExtBuilder}}
    )
    return setup_kwargs


if __name__ == "__main__":
    build({})
