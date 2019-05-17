import sys
import re
import itertools
from setuptools import setup, find_packages
from distutils.core import Extension
import numpy as np
import argparse
import os
from typing import Iterable


# Args parsing, gets the path to the library dirs
if '--path' in sys.argv:
    index = sys.argv.index('--path')
    sys.argv.pop(index)  # Removes the '--path'
    lib_dir = sys.argv.pop(index)  # Returns the element after the '--path'
else:
	lib_dir = "/Users/MacD/Documents/Libraries/fp-libs"
armadir = os.path.join(lib_dir, 'armadillo-install')
blasdir = os.path.join(lib_dir, 'openblas-install')
lapackdir = os.path.join(lib_dir, 'lp-install')
cvdir = os.path.join(lib_dir, 'cv-install')
nbisdir = os.path.join(lib_dir, 'nbis-install')

# Find all the libraries
def find_libs(root: str, libs: dict):
	""" Library linking helper.

	Finds the given libraries and return their path in a format compatible with the distutils.core.Extension class.

	Args:
		root: path containing all the libraries to be loaded
		libs: dictionary where the keys are (relative or absolute) path from root and values are lists of library names

	Return:
		The dictionary {libraries: ..., library_dirs: ..., extra_objects: ...}, to be used as
		>>> Extension(..., **find_libs('root', ['path1', 'path2'], ['lib1', 'lib2']))
	"""
	ret = {
		'libraries': [],
		'library_dirs': [],
		'extra_objects': []}
	for d, pair in libs.items():
		static_libraries, formats = pair
		if os.path.isabs(d):
			static_lib_dir = d
		else:
			static_lib_dir = os.path.join(root, d)
		if sys.platform == 'win32':
			ret['libraries'].extend(static_libraries)
			ret['library_dirs'].append(static_lib_dir)
		else: # POSIX
			ret['extra_objects'].extend([os.path.join(static_lib_dir, fmt.format(lib)) for lib, fmt in zip(static_libraries, formats)])
	return ret

def find_all_libs(root: str):
	def get_all_static_libs_in_path(path):
		lib_patt = re.compile('lib(\\w+)\\.(?:lib|a)')
		for dir, _, files in os.walk(path):
			matches = filter(None, map(lib_patt.match, files))
			files = list(set(map(lambda x: x[0], matches))) # use set to produce unique strings
			if len(files) > 0:
				yield list(map(lambda x: os.path.join(dir, x), files))
	extra_objects = get_all_static_libs_in_path(root)
	extra_objects = list(itertools.chain(*extra_objects))
	return extra_objects

# NBIS Extension
extra_objects = find_all_libs(os.path.join(nbisdir, 'lib'))
extra_objects = ['--start-group'] + extra_objects + ['--end-group']
print('NBIS extra_objects:', extra_objects)
nbis_ext = Extension(
	'pynger.fingerprint.nbis',
	sources=[
		'pynger/fingerprint/nbismodule/nbismodule.c', 
		'pynger/fingerprint/nbismodule/sgmnt.c',
		'pynger/fingerprint/nbismodule/enhnc.c',
		'pynger/fingerprint/nbismodule/rors.c',
		'pynger/fingerprint/nbismodule/utils.c',
		'pynger/fingerprint/nbismodule/mindtct.c'],
	include_dirs=[
		os.path.join(nbisdir, 'include'),
		np.get_include()],
	extra_objects=extra_objects
	# **find_libs(nbisdir, {
	# 	'lib': (
	# 			[
	# 			'pca', 'pcautil', 'util', 'image', 'ioutil', 'ihead', 'z', # sgmnt
	# 			'fft', # enhnc
	# 			'an2k', 'mindtct', # mindtct
	# 			],
	# 			['lib{}.a' for _ in range(9)]
	# 		)
	# 	})
	)
	
pani_ext = Extension(
	'pynger.signal.pani',
	sources=[
		'pynger/signal/panimodule/panimodule.c', 
		'pynger/signal/panimodule/panigauss.c'],
	include_dirs=[np.get_include()],
	)

# OpenCV 
ang_seg_args = ['-std=gnu++14', '-Wextra']
ang_seg_link_args = ['-fPIC', '-Wl,--verbose']
extra_objects = find_all_libs(os.path.join(cvdir, 'lib'))
# As the linking order matters, put the corelib at the end and the adelib right before
cv_corelib = [lib for lib in extra_objects if 'core' in lib][0]
cv_adelib = [lib for lib in extra_objects if 'libade' in lib][0]
cv_corelib_idx = extra_objects.index(cv_corelib)
cv_adelib_idx = extra_objects.index(cv_adelib)
indices = [cv_adelib_idx, cv_corelib_idx]
indices = [k for k in range(len(extra_objects)) if k not in indices] + indices
extra_objects = [extra_objects[k] for k in indices]
extra_objects = ['--start-group'] + extra_objects + ['--end-group']
print('OpenCV extra_compile_args:', ang_seg_args)
print('OpenCV extra_link_args:', ang_seg_link_args)
print('OpenCV extra_objects:', extra_objects)
ang_seg_ext = Extension(
	'pynger.fingerprint.cangafris',
	sources=[
		'pynger/fingerprint/angafris_segmentation/Sources/AdaptiveThreshold.cpp',
		'pynger/fingerprint/angafris_segmentation/Sources/ImageCropping.cpp',
		'pynger/fingerprint/angafris_segmentation/Sources/ImageMaskSimplify.cpp',
		'pynger/fingerprint/angafris_segmentation/Sources/ImageNormalization.cpp',
		'pynger/fingerprint/angafris_segmentation/Sources/ImageRescale.cpp',
		'pynger/fingerprint/angafris_segmentation/Sources/ImageSignificantMask.cpp',
		'pynger/fingerprint/angafris_segmentation/Sources/myMathFunc.cpp',
		'pynger/fingerprint/angafris_segmentation/Sources/TypesTraits.cpp',
		'pynger/fingerprint/angafris_segmentation/Sources/ang_seg_wrapper.cpp',
		'pynger/fingerprint/angafris_segmentation/ang_seg_module.cpp',
		],
	include_dirs=[
		np.get_include(),
		os.path.join(armadir, 'include'),
		os.path.join(cvdir, 'include/opencv4'),
		],
	extra_compile_args=ang_seg_args,
	extra_link_args=ang_seg_link_args,
	extra_objects=extra_objects,
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
	ext_modules=[nbis_ext, pani_ext, ang_seg_ext]
)