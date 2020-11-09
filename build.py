import os
from setuptools import setup, Extension

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

ubo2014_module = Extension(
    "ubo2014_cpp",
    extra_compile_args=["-mavx", "-Ofast", "-march=native"],
    sources=["btf_extractor/c_ext/ubo2014.cc"],
)

setup(
    name="btf_extractor",
    version="1.0",
    description="This is a demo package",
    ext_modules=[ubo2014_module],
)
