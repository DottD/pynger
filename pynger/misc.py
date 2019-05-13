import os
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from pynger.field.manipulation import cart2polar, dprod_2array
# Only for the visualizer class
from pynger.fingerprint.operators import _get_circle_coord, _fast_interp_over_circles, _interpolate_over_circles
from pynger.field.calculus import rot_2d
from pynger.field.manipulation import halve_angle, normalize, magnitude

	
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
	Visualizer for field and coordinate. See pynger.fingerprint.operators.generic_operator for an explanation of the parameters. 
	
	Note:
		Drifter still not supported.
	"""
	def __init__(self, field, concordance, emphasis, radius, num, relax=0.9, figure=None):
		self.field = field
		self.rows = field.shape[0]
		self.cols = field.shape[1]
		# Compute the coordinates of the field values
		i, j = np.meshgrid(range(self.rows), range(self.cols), indexing='ij')
		self.coord = _get_circle_coord(j, i, radius, num, indexing='xy')
		# Halve and normalize the input field
		self.hfield = halve_angle(field)
		self.nhfield = normalize(self.hfield)
		# Compute concordance vector (cvec)
		self.hf = _fast_interp_over_circles(self.hfield, radius, num=num)
		self.cvec = _get_circle_coord(0, 0, 1, num=num, indexing='xy')
		if concordance == 'center_real_field':
			self.cvec = self.hfield[:,:,:,None]
		elif concordance == 'tangent':
			self.cvec = rot_2d(self.cvec)
		elif concordance == 'normal':
			pass
		else:
			raise NotImplementedError('Concordance not supported!')
		# Compute signum
		d2a = dprod_2array(self.hf, self.cvec, axis=2, keepDims=True)
		d2a[np.isclose(np.abs(d2a), 0)].fill(1)
		self.signum = np.sign(d2a)
		# Compute emphasis vector (evec)
		self.nhf = _fast_interp_over_circles(self.nhfield, radius, num=num)
		self.evec = _get_circle_coord(0, 0, 1, num=num, indexing='xy')
		if emphasis == 'none':
			self.evec = self.nhf # not optimized ...
		if emphasis == 'tangent':
			self.evec = rot_2d(self.evec)
		if emphasis == 'normal':
			pass
		# Compute emphasis
		self.weight = dprod_2array(self.nhf, self.evec, axis=2, keepDims=True) **2
		# Compute total weight
		W = self.weight.sum(axis=-1)
		W[np.isclose(W, 0)].fill(1)
		# Compute the smoother field (put together the parts of the integral)
		self.vecs = self.signum * self.weight * self.hf
		self.sfield = self.vecs.sum(axis=-1) / W
		# Apply the relaxation
		if isinstance(relax, np.ndarray):
			self.sfield = (1 - relax)[:,:,None] * self.field + relax[:,:,None] * self.sfield
		else:
			self.sfield = (1 - relax) * self.field + relax * self.sfield
		# Get the position of the center
		self.i, self.j, _ = tuple(n//2 for n in field.shape)
		self.x = self.j
		self.y = self.i
		# Create the figure and its axes
		if figure is None:
			self.fig = plt.figure()
			self.ax1 = plt.axes()
		else:
			self.fig = figure
			if len(self.fig.axes) > 0:
				self.ax1 = self.fig.get_axes()[0]
			else:
				plt.figure(self.fig.number)
				self.ax1 = plt.axes()
		
	def onclick(self, event):
		if event.inaxes:
			self.x = int(event.xdata)
			self.j = self.x
			self.y = int(event.ydata)
			self.i = self.y
			
	def redraw(self):
		###### 1st plot
		self.ax1.clear()
		# Draw the orientation field
		vis_orient_field(self.hfield, step=10, color='black', axes=self.ax1)
		# Draw the current circle
		circlexy = self.coord[self.i, self.j, :, :].squeeze()
		self.ax1.plot(*circlexy, color='red')
		# Draw the field in the center of the circle
		self.ax1.quiver(self.x, self.y, 
			self.hfield[self.i, self.j, 0],
			self.hfield[self.i, self.j, 1],
			color='red', headlength=0, headaxislength=0, pivot='mid')
		# Draw the results in the center of the circle
		self.ax1.quiver(self.x, self.y, 
			self.sfield[self.i, self.j, 0],
			self.sfield[self.i, self.j, 1],
			color='blue', headlength=0, headaxislength=0, pivot='mid')
		# Get the vectors along the current circle
		vecs = self.vecs[self.i, self.j, :, :].T
		# Draw the vectors over the circle
		mag = magnitude(vecs[None, :, :], keepDims=False)
		self.ax1.quiver(circlexy[0,:], circlexy[1,:], vecs[:,0], vecs[:,1], mag, cmap='YlGn')
		# Set aspect ratio
		self.ax1.set_aspect('equal', 'box')
		
	def offclick(self, event=None):
		self.redraw()
		self.fig.canvas.draw()
		
	def __call__(self):
		self.offclick()
		self.fig.canvas.mpl_connect('button_press_event', lambda event: self.onclick(event))
		self.fig.canvas.mpl_connect('button_release_event', lambda event: self.offclick(event))
		plt.show()


def recursively_scan_dir(path, ending):
	"""
	Recursively scan the folder.
	
	Args:
		- path, path to recursevily analyze;
		- ending, list of possible extensions.
		
	Returns: a pair with the list of directories and the list of files (no clue on folder tree structure).
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


def recursively_scan_dir_gen(path, ending):
	"""
	Recursively scan the folder, but returns a generator.

	Return:
		This function yields the absolute path of the files.
	
	See Also:
		`.recursively_scan_dir'
	"""
	for curr_dir, _, local_files in os.walk(path):
		for x in filter(lambda x: x.endswith(ending), local_files):
			yield os.path.join(curr_dir, x)