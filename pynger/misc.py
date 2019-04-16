import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from pynger.field.manipulation import cart2polar, dprod_2array

	
def _get_locations_of_subsampled_vf(sampled_vf_shape, original_vf_shape):
	""" Get the locations of the subsampled points in the original vf.
	
	:return: x, y
	"""
	try:
		original_vf_shape = np.broadcast_to(original_vf_shape, (2,))
		sampled_vf_shape = np.broadcast_to(sampled_vf_shape, (2,))
	except ValueError as err:
		print(err)
		raise RuntimeError("Input shapes ({} and {}) must be broadcastable to shape (1, 2)".format(sampled_vf_shape, original_vf_shape))
	step = np.array(original_vf_shape) / np.array(sampled_vf_shape)
	if (step < 1).any():
		raise RuntimeError("Sampling factor in at least one dimension is too low ({})".format(step))
	x, y = np.meshgrid(np.arange(sampled_vf_shape[1]).astype(float), np.arange(sampled_vf_shape[0]).astype(float))
	x *= step[1]
	y *= step[0]
	return x, y

def _subsample_vector_field(theta, strength, step=None):
	""" Subsample a vector field given in polar coordinate.
	
	:return: x, y, theta, strength of the new field.
	"""
	if np.array(theta.shape != strength.shape).any():
		raise RuntimeError("Theta {} and strength {} must have the same shape".format(theta.shape, strength.shape))
	rows, cols = theta.shape
	x, y = np.meshgrid(np.arange(cols), np.arange(rows))
	if step: S = step
	else: S = 6
	x = x[::S,::S]
	y = y[::S,::S]
	theta = theta[::S,::S]
	strength = strength[::S,::S]
	return x, y, theta, strength

def vis_orient_field(field, **kwargs):
	return vis_orient(*cart2polar(field, keepDims=False), **kwargs)

def vis_orient(theta, strength=1.0, mask=True, step=None, dest_shape=None, appl_trasl=None, **quiver_args):
	try:
		theta, strength, mask = np.broadcast_arrays(theta, strength, mask)
	except ValueError as err:
		print(err)
		raise RuntimeError("Theta and strength must be broadcastable to each other")
	strength *= mask.astype(float)
	if dest_shape:
		x, y = _get_locations_of_subsampled_vf(theta.shape, dest_shape)
	else:
		x, y, theta, strength = _subsample_vector_field(theta, strength, step)
	if appl_trasl:
		x += appl_trasl[0]
		y += appl_trasl[1]
	kwargs = {
		'units': 'xy',
		'pivot': 'mid',
		'angles': 'uv',
		'headaxislength': 0,
		'headlength': 0
		}
	kwargs.update(quiver_args)
	if 'axes' in quiver_args:
		del kwargs['axes']
		quiver_args['axes'].quiver(x, y, strength*np.cos(theta), strength*np.sin(theta), **kwargs)
	else:
		plt.quiver(x, y, strength*np.cos(theta), strength*np.sin(theta), **kwargs)

def vis_stream_orient_field(field, **kwargs):
	return vis_stream_orient(*cart2polar(field, keepDims=False), **kwargs)
		
def vis_stream_orient(theta, strength=1.0, mask=True, step=None, dest_shape=None, appl_trasl=None, **stream_args):
	try:
		theta, strength, mask = np.broadcast_arrays(theta, strength, mask)
	except ValueError as err:
		print(err)
		raise RuntimeError("Theta and strength must be broadcastable to each other")
	strength *= mask.astype(float)
	if not dest_shape is None:
		x, y = _get_locations_of_subsampled_vf(theta.shape, dest_shape)
	else:
		x, y, theta, strength = _subsample_vector_field(theta, strength, step)
	if appl_trasl:
		x += appl_trasl[0]
		y += appl_trasl[1]
	kwargs = {}
	kwargs.update(stream_args)
	if 'axes' in stream_args:
		del kwargs['axes']
		return stream_args['axes'].streamplot(x, y, strength*np.cos(theta), -strength*np.sin(theta), **kwargs)
	else:
		return plt.streamplot(x, y, strength*np.cos(theta), -strength*np.sin(theta), **kwargs)
		
def testTime(fun, rep=10, msg="Function executed in {} seconds on average"):
	t = time()
	for _ in range(rep-1): fun()
	out = fun()
	print(msg.format((time()-t)/rep))
	return out
	
class VisualizeFieldInterpolation:
	"""
	Visualizer for field and coordinate. Ad-hoc made...
	"""
	def __init__(self, field, coord, fac, normal, half_fac):
		self.field = field
		self.coord = coord
		self.fac = fac
		self.normal = normal
		self.half_fac = half_fac
		self.i, self.j = (0,0)
		self.f, (self.ax1, self.ax2) = plt.subplots(1, 2)
		
	def onclick(self, event):
		if event.inaxes:
			self.j = int(event.xdata)
			self.i = int(event.ydata)
			
	def redraw(self):
		field = self.field
		coord = self.coord
		fac = self.fac
		normal = self.normal
		half_fac = self.half_fac
		dot_prod = dprod_2array(half_fac, normal, axis=2, keepdims=True)
		# 1st
		self.ax1.clear()
		self.ax1.imshow(np.ones((field.shape[0], field.shape[1])), cmap='gray')
		angle, magnitude = cart2polar(*np.dsplit(field, 2))
		vis_orient(angle.squeeze(), step=10, strength=magnitude.squeeze(), color='red', axes=self.ax1)
		self.ax1.plot(*np.flipud(coord[self.i, self.j, :, :].squeeze()), color='green')
		# 2nd
		self.ax2.clear()
		line = fac[self.i, self.j, :, :].squeeze()
		stripe = np.vstack((
			np.zeros(line.shape), 
			dot_prod[self.i, self.j, :, :].squeeze(), 
			np.sign(dot_prod[self.i, self.j, :, :].squeeze())
			))
		self.ax2.imshow(stripe, cmap='gray')
		angle, magnitude = cart2polar(line[0], line[1])
		vis_orient(angle[None,:], step=1, strength=magnitude[None,:], color='red', axes=self.ax2)
		half_line = half_fac[self.i, self.j, :, :].squeeze()
		angle, magnitude = cart2polar(half_line[0], half_line[1])
		vis_orient(angle[None,:], step=1, strength=magnitude[None,:], color='green', axes=self.ax2)
		
	def offclick(self, event=None):
		self.redraw()
		self.f.canvas.draw()
		
	def __call__(self, pixel=(0,0)):
		self.i = pixel[0]
		self.j = pixel[1]
		self.offclick()
		self.f.canvas.mpl_connect('button_press_event', lambda event: self.onclick(event))
		self.f.canvas.mpl_connect('button_release_event', lambda event: self.offclick(event))
		plt.show()


def recursively_scan_dir(path, ending):
	"""
	Recursively scan the folder.
	
	Args:
		- path, path to recursevily analyze;
		- ending, list of possible extensions.
		
	Returns: a tuple containing the list of directories and the list of files (no clue on folder tree structure).
	"""
	file_list = []
	dir_list = []
	for curr_dir, _, local_files in os.walk(path):
		# filter local files
		local_files = [os.path.join(curr_dir, x) for x in local_files if any(map(lambda ext: x.endswith(ext), ending))]
		# append to global list
		file_list += local_files
		if local_files:
			dir_list.append(curr_dir)
	return dir_list, file_list