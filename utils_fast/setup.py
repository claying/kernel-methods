from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [Extension("utils_fast", 
	['utils_fast.pyx'],
	libraries=["m"],
	extra_compile_args = ["-ffast-math"],
	include_dirs = [numpy.get_include()])
]

setup(
  name = 'utils_fast',
  ext_modules = cythonize(extensions),
)