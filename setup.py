import numpy
from distutils.core import setup
from distutils.extension import Extension
from distutils.command import build_ext

gistmodule = Extension(
	'gist',
	sources=['gistmodule.c'],
	include_dirs=[numpy.get_include()],
	extra_objects=['lear_gist-1.2/gist.o', 'lear_gist-1.2/standalone_image.o'],
	library_dirs=['/Users/tsuchiya/local/lib'],
	libraries=['fftw3f'])

setup(name='gist',
	version='0.1',
	description='A wrapper package of lear_gist',
	ext_modules=[gistmodule])