from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    include_dirs=[numpy.get_include()] ,cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cython_bbox", ["cython_bbox.pyx"])]
)
