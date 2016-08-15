import numpy
from distutils.core import setup
from distutils.extension import Extension
from distutils.command import build_ext

gistmodule = Extension(
	'gist',
	sources=['lear_gist-1.2/gist.c', 'lear_gist-1.2/standalone_image.c', 'gistmodule.c'],
	extra_compile_args=['-DUSE_GIST', '-DSTANDALONE_GIST', '--std=gnu99', '-lm'],
	include_dirs=[numpy.get_include()],
	libraries=['fftw3f'])

setup(name='gist',
	version='0.1',
	description='A wrapper package of lear_gist',
	ext_modules=[gistmodule])
