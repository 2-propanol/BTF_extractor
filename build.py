import os
from distutils.command.build_ext import build_ext
from distutils.core import Extension
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError


if os.name == "nt":
    ext_modules = [
        Extension(
            "ubo2014_cpp",
            extra_compile_args=["/O2", "/arch:AVX", "/DNOMINMAX"],
            sources=["btf_extractor/c_ext/ubo2014.cc"],
        )
    ]
else:
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"

    ext_modules = [
        Extension(
            "ubo2014_cpp",
            extra_compile_args=["-mavx", "-Ofast", "-march=native"],
            sources=["btf_extractor/c_ext/ubo2014.cc"],
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
