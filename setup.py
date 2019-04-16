import sys
from setuptools import setup, find_packages
from distutils.core import Extension
import numpy as np
import argparse
import os


if '--path' in sys.argv:
    index = sys.argv.index('--path')
    sys.argv.pop(index)  # Removes the '--path'
    nbispath = sys.argv.pop(index)  # Returns the element after the '--path'
else:
	nbispath = "/Users/MacD/Documents/Libraries/NBIS"

# Set up the NBIS extension
static_libraries = [
	'pca', 'pcautil', 'util', 'image', 'ioutil', 'ihead', # sgmnt
	'fft', # enhnc
	'an2k', 'mindtct', # mindtct
	]
static_lib_dir = os.path.join(nbispath, 'lib')
libraries = []
library_dirs = []

if sys.platform == 'win32':
	libraries.extend(static_libraries)
	library_dirs.append(static_lib_dir)
	extra_objects = []
else: # POSIX
	extra_objects = [os.path.join(static_lib_dir, 'lib{}.a'.format(l)) for l in static_libraries]

nbis_ext = Extension(
	'pynger.fingerprint.nbis',
	sources=[
		'pynger/fingerprint/nbismodule/nbismodule.c', 
		'pynger/fingerprint/nbismodule/sgmnt.c',
		'pynger/fingerprint/nbismodule/enhnc.c',
		'pynger/fingerprint/nbismodule/rors.c',
		'pynger/fingerprint/nbismodule/utils.c',
		'pynger/fingerprint/nbismodule/mindtct.c'],
	include_dirs=[os.path.join(nbispath, 'include'),
		np.get_include()],
	libraries=libraries,
	library_dirs=library_dirs,
	extra_objects=extra_objects
	)
	
# Set up the anigauss extension
pani_ext = Extension(
	'pynger.signal.pani',
	sources=[
		'pynger/signal/panimodule/panimodule.c', 
		'pynger/signal/panimodule/panigauss.c'],
	include_dirs=[np.get_include()],
	)

# Load README file
with open("README.md", "r") as fh:
	long_description = fh.read()

# Install the package
setup(
	name="pynger-DottD",
	version="0.0.1",
	author="Filippo Santarelli",
	author_email="filippo2.santarelli@gmail.com",
	description="A suite of utilities for Fingerprint Analysis",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/pypa/sampleproject",
	packages=find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	ext_modules=[nbis_ext, pani_ext]
)