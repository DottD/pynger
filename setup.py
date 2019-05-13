import sys
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
	for d, l in libs.items():
		if os.path.isabs(d):
			static_lib_dir = d
		else:
			static_lib_dir = os.path.join(root, d)
		static_libraries = l
		if sys.platform == 'win32':
			ret['libraries'].extend(static_libraries)
			ret['library_dirs'].append(static_lib_dir)
		else: # POSIX
			ret['extra_objects'].extend([os.path.join(static_lib_dir, 'lib{}.a'.format(l)) for l in static_libraries])
	return ret
		

# Set up the extensions
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
		os.path.join(lib_dir, 'include'),
		np.get_include()],
	**find_libs(nbisdir, {
		'lib': [
			'pca', 'pcautil', 'util', 'image', 'ioutil', 'ihead', # sgmnt
			'fft', # enhnc
			'an2k', 'mindtct', # mindtct
			],
		})
	)
	
pani_ext = Extension(
	'pynger.signal.pani',
	sources=[
		'pynger/signal/panimodule/panimodule.c', 
		'pynger/signal/panimodule/panigauss.c'],
	include_dirs=[np.get_include()],
	)

ang_seg_libspecs = {
	os.path.join(cvdir, 'lib'): ['opencv_core', 'opencv_features2d', 'opencv_imgcodecs', 'opencv_imgproc'],
}
ang_seg_args = ['-std=gnu++14']
if sys.platform == 'darwin':
	ang_seg_args += ['-F/System/Library/Frameworks -lAccelerate']
else:
	ang_seg_libspecs.update({
		os.path.join(blasdir, 'lib'): ['openblas'],
		os.path.join(lapackdir, 'lib'): ['lapack'],
	})
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
		'pynger/fingerprint/angafris_segmentation/ang_seg_module.cpp',
		],
	include_dirs=[
		np.get_include(),
		os.path.join(armadir, 'include'),
		os.path.join(cvdir, 'include/opencv4'),
		],
	**find_libs(
		lib_dir,
		ang_seg_libspecs
	),
	extra_compile_args=ang_seg_args,
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