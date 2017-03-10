from distutils.extension import Extension

import numpy
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration

	config = Configuration('utils_fast', parent_package, top_path)

	extensions = [
		Extension("svm_solver.svm_solvers", 
				['svm_solver/svm_solvers.pyx'],
				libraries=["m"],
				extra_compile_args = ["-ffast-math"],
				include_dirs = [numpy.get_include()]
				)
	]
	

	config.ext_modules += extensions

	return config


if __name__ == '__main__':
	from numpy.distutils.core import setup

	setup(**configuration(top_path='').todict())