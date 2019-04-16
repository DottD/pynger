import os
import re
from PIL import Image
from subprocess import Popen, PIPE
import numpy as np
from time import time
from pathlib import Path
from pynger.field.manipulation import cart2polar
from scipy.ndimage import gaussian_filter
import pandas as pd


def nist_mkoas(image, **kwargs):
	""" Compute the LRO using the NIST-MKOAS method.
	
	Args:
		- image, the image containing the fingerprint (numpy array)
		- "mkoas_exe", path to the executable
		- support for every parameter of MKAOS (see below)
	"""
	exedir = os.path.dirname(__file__)
	try:
		exe_path = kwargs.pop('mkoas_exe')
	except KeyError:
		exe_path = os.path.join(exedir, 'mkoas')
	# Get the current timestamp (it will univoquely identify this function call)
	timestamp = str(int(time()))
	# Save the image in the current directory
	currdir = str(Path.home())
	imagefile = os.path.join(currdir, '{}.png'.format(timestamp))
	Image.fromarray(image).convert('L').save(imagefile)
	# Set up the other files
	file_oas_path = os.path.join(currdir, "{}_oas.txt".format(timestamp))
	file_list_path = os.path.join(currdir, "{}_list.txt".format(timestamp))
	with open(file_list_path, 'w') as f:
		f.write("{}\n".format(imagefile))
	# Write prs file 
	pars = {
		# Segmentation
		"sgmnt_fac_n": 5,
		"sgmnt_min_fg": 2000,
		"sgmnt_max_fg": 8000,
		"sgmnt_nerode": 3,
		"sgmnt_rsblobs": 1,
		"sgmnt_fill": 1,
		"sgmnt_min_n": 25,
		"sgmnt_hist_thresh": 20,
		"sgmnt_origras_wmax": 2000,
		"sgmnt_origras_hmax": 2000,
		"sgmnt_fac_min": 0.75,
		"sgmnt_fac_del": 0.05,
		"sgmnt_slope_thresh": 0.90,
		# Enhancement
		"enhnc_rr1": 150,
		"enhnc_rr2": 449,
		"enhnc_pow": 0.3,
		# Ridge-valley orientation finder
		"rors_slit_range_thresh": 10,
		# r92 registration program wrapper
		"r92a_discard_thresh": 0.01,
		# Registering pixelwise-orientations-reaverager
		"rgar_std_corepixel_x": 245,
		"rgar_std_corepixel_y": 189
	}
	pars.update(kwargs)
	pars.update({
		# Mkoas specific pars
		"ascii_oas": "y",  
		"update_freq": 1,
		"clobber_oas_file": "y",
		"proc_images_list": "{}".format(os.path.abspath(file_list_path)),
		"oas_file": "{}".format(os.path.abspath(file_oas_path))
	})
	file_prs_path = os.path.join(currdir, "{}.prs".format(timestamp))
	with open(file_prs_path, 'w') as f:
		for key, val in pars.items():
			f.write("{} {}\n".format(key, str(val)))
	# Run NBIS-MKOAS
	command = "\"{}\" \"{}\"".format(exe_path, file_prs_path)
	with Popen(command, cwd=currdir, shell=True, universal_newlines=True, stdout=PIPE, stderr=PIPE) as proc:
		err = proc.stderr.read()
		if err != "":
			raise RuntimeError(err)
	# Reformat output file
	with open(file_oas_path, 'r') as oas_file:
		lines = (line for n, line in enumerate(oas_file) if n >= 4)
		content = ",".join( map(lambda line: ",".join(line[:-1].split()), lines) )
	# Remove no-longer-needed files
	os.remove(file_list_path)
	os.remove(file_prs_path)
	os.remove(file_oas_path)
	os.remove(imagefile)
	# Load array from string
	lro = np.fromstring(content, sep=',')
	lro = lro.reshape((2, 28, 30))
	lro = lro.transpose(1, 2, 0)
	return lro
	
def nist_nfseg(image, **kwargs):
	""" Compute the LRO using the NIST-MKOAS method.
		
	Args:
		- image, the image containing the fingerprint (numpy array)
		- "nfseg_exe", path to the executable
		- support for every parameter of MKAOS (see below)
	"""
	exedir = os.path.dirname(__file__)
	try:
		exe_path = kwargs.pop('nfseg_exe')
	except KeyError:
		exe_path = os.path.join(exedir, 'nfseg')
	# Get the current timestamp (it will univoquely identify this function call)
	timestamp = str(int(time()))
	# Save the image in the current directory
	currdir = str(Path.home())
	imagefile = os.path.join(currdir, '{}.png'.format(timestamp))
	Image.fromarray(image).convert('L').save(imagefile)
	# Run NBIS-NFSEG
	bb = {} # list of bounding boxes
	command = "\"{}\" 11 0 0 0 0 \"{}\"".format(exe_path, imagefile)
	with Popen(command, cwd=currdir, shell=True, universal_newlines=True, stdout=PIPE, stderr=PIPE) as proc:
		err = proc.stderr.read()
		if err != "":
			raise RuntimeError(err)
		else:
			msg = proc.stdout.read()
			regex = re.compile("FILE (?P<file>[\\S]+) -> e \\d+ sw (?P<w>\\d+) sh (?P<h>\\d+) sx (?P<x>\\d+) sy (?P<y>\\d+) th [\\d.+-]+")
			match = regex.match(msg)
			if match:
				groups = match.groupdict()
				tmp_output_path = os.path.join(currdir, groups.pop('file'))
				os.remove(tmp_output_path)
				bb = {k:int(v) for k, v in groups.items()}
			else:
				raise RuntimeError("Cannot read parse data from NBIS-NFSEG output")
	os.remove(imagefile)
	return bb

def nbis_idx2angle(idx, N=8, mode='mindtct'):
	""" Converts indexed direction map to angular information. """
	idx = np.array(idx)
	if mode == 'pca':
		strength = (idx == 0)
		idx -= 1 # convert to 0, ..., N-1
	q = 0.5 - idx / N
	theta = np.pi * (q - np.floor(q))
	if mode == 'pca':
		theta[strength] = 0.0
	return theta

def nbis_angle2idx(theta, N=8, mode='mindtct'):
	""" Converts angular direction map to indexed information. """
	beta = ((theta + np.pi/2) % np.pi) - np.pi/2
	idx = np.round(N/2 * (1 - (2 * beta)/np.pi))
	idx[idx == N] = 0 # replace N with 0
	if mode == 'pca':
		idx += 1 # convert to 1, ..., N
	return idx.astype(int)

def nbis_bozorth3(left, right, **kwargs):
	""" Run the NBIS Bozorth3 matcher on the two input sets of minutiae.

	Args:
		left (minutiae list): First list of minutiae
		right (minutiae list): Second list of minutiae

    Keyword Args:
        verbose (bool): Whether some information should be returned in stdout (defaults to False)
				bozorth3_exe (str): Path to the bozorth3 executable
	"""
	# Get the full path to the executable
	exe_path = os.path.join(os.path.dirname(__file__), 'bozorth3')
	exe_path = kwargs.get('bozorth3_exe', exe_path)
	# Save minutiae information to the current directory
	currdir = str(Path.home())
	left_min_path = os.path.join(currdir, 'left-{}-{}.xyt'.format(id(left), id(right)))
	left = np.array([(m['x'], m['y'], m['direction'], m['reliability']) for m in left])
	left[:,2] = np.round(np.rad2deg(nbis_idx2angle(left[:,2], N=16)))
	left[:,3] = np.round(left[:,3] * 100.0)
	left = left.astype(int)
	left = left[left[:,3] > np.percentile(left[:,3], 5), :]
	to_csv_options = {'sep': ' ', 'header': False, 'index': False}
	pd.DataFrame(left).to_csv(left_min_path, **to_csv_options)
	right_min_path = os.path.join(currdir, 'right-{}-{}.xyt'.format(id(left), id(right)))
	right = np.array([(m['x'], m['y'], m['direction'], m['reliability']) for m in right])
	right[:,2] = np.round(np.rad2deg(nbis_idx2angle(right[:,2], N=16)))
	right[:,3] = np.round(right[:,3] * 100.0)
	right = right.astype(int)
	right = right[right[:,3] > np.percentile(right[:,3], 5), :]
	pd.DataFrame(right).to_csv(right_min_path, **to_csv_options)
	# Run matcher
	command = "{} \"{}\" \"{}\"".format(exe_path, left_min_path, right_min_path)
	with Popen(command, cwd=currdir, shell=True, universal_newlines=True, stdout=PIPE, stderr=PIPE) as proc:
		err = proc.stderr.read()
		if err != "":
			raise RuntimeError(err)
		if kwargs.get('verbose', False):
			score = int(proc.stdout.read())
	os.remove(left_min_path)
	os.remove(right_min_path)
	return left, right, score
	
	
if __name__ == '__main__':
	# from pynger.misc import vis_orient
	# import matplotlib.pyplot as plt
	# # Load the image
	# path = "/Users/MacD/Downloads/oas/f0007_09.png"
	# image = np.array(Image.open(path).convert('L'))
	# # Compute the orientation field
	# lro = nist_mkoas(image, mkoas_exe=os.path.join(os.path.dirname(__file__), 'mkoas_noreg_sd04'))
	# # Compute the bounding box
	# bb = nist_nfseg(image, mkoas_exe=os.path.join(os.path.dirname(__file__), 'nfseg'))
	# # Cut the image at the bounding box
	# x, y = (bb['x']-bb['w']//2, bb['y']-bb['h']//2)
	# image = image[-y-bb['h']:-y, x:x+bb['w']]
	# # Visualization
	# plt.imshow(image)
	# theta, strength = cart2polar(lro, keepDims=False)
	# vis_orient(theta=theta/2, strength=strength, dest_shape=image.shape)
	# plt.show()
	
	idx = range(1,9)
	angle = nbis_idx2angle(idx, mode='pca')
	new_idx = nbis_angle2idx(angle, mode='pca')
	print(list(idx))
	print(list(new_idx))
	print((idx == new_idx).all())