from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    name = 'bhfunctions',
    ext_modules=cythonize("bhfunctions.pyx", annotate=True, language_level = "3"),
    include_dirs=[numpy.get_include()]
)
