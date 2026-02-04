from setuptools import setup, Extension
import pybind11

# Define the extension module
cpp_module = Extension(
    'nexus_cpp',  # The name you will use in Python (import nexus_cpp)
    sources=['cpp_src/optimizer.cpp'], # Path to your C++ file
    include_dirs=[pybind11.get_include()],
    language='c++'
)

setup(
    name='nexus_cpp',
    version='1.0',
    description='C++ Accelerator for Nexus AI',
    ext_modules=[cpp_module],
    zip_safe=False,
)
