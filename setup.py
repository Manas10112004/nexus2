from setuptools import setup, Extension
import sys

# This allows the installer to query metadata without crashing 
# if pybind11 isn't fully linked yet.
try:
    import pybind11
    include_dirs = [pybind11.get_include()]
except ImportError:
    include_dirs = []

cpp_module = Extension(
    'nexus_cpp',
    sources=['cpp_src/optimizer.cpp'],
    include_dirs=include_dirs,
    language='c++',
    extra_compile_args=['-O3', '-std=c++11'],
)

setup(
    name='nexus_cpp',
    version='1.0',
    ext_modules=[cpp_module],
    zip_safe=False,
)
