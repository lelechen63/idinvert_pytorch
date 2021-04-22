from setuptools import setup, Extension
from torch.utils import cpp_extension

from sys import platform

if platform == "linux":
    cpp_args = [ '-std=c++11', '-fopenmp' ]
    ext_modules = [
        cpp_extension.CppExtension(
            name = 'texUnwarp',
            sources = [
                'texUnwarp.cpp'
            ],
            include_dirs = [
                'pybind11/include/',
                '/mnt/home/chen/dependency/eigen_3.3.7/',
                '/mnt/home/chen/dependency/opencv-4.2.0/local/include/opencv4/',
            ],
            library_dirs = [
                '/mnt/home/chen/dependency/opencv-4.2.0/local/lib/',
            ],
            libraries = [
                'opencv_core',
                'opencv_imgproc'
            ],
            language = 'c++',
            extra_compile_args = cpp_args,
        )
    ]

setup( name = 'texUnwarp',
       ext_modules = ext_modules,
       cmdclass = { 'build_ext': cpp_extension.BuildExtension } )