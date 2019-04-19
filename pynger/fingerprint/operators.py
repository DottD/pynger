import numpy as np
from scipy.ndimage import map_coordinates
from pynger.signal.windows import circleWin
from pynger.field.manipulation import halve_angle, double_angle, normalize, magnitude, dprod_2array, reflection
from pynger.field.calculus import rot_2d
from pynger.types import Mask, Field

def _get_circle_coord(x, y, radius, num=100, indexing='xy'):
	""" Returns the coordinates of a circle with given parameters.

	Args:
		x (numpy.array): First coordinate of the center of the circle
		y (numpy.array): Second coordinate of the center of the circle
		radius (float): Radius of the circle
		num (int): Number of sampling points over the circle
	
	Return: 
		numpy.array: An array C with shape ``(field.shape[0], field.shape[1], 2, kwargs['num'])`` , such that:
		
		- ``C[i,j,0,:]`` is the x-coordinate of the points on the circle centered at pixel (i,j)
		- ``C[i,j,1,:]`` is the y-coordinate of the points on the circle centered at pixel (i,j)
		
	Note:
		The x and y arguments must be broadcastable to each other.
	"""
	x = np.array(x, ndmin=2)
	y = np.array(y, ndmin=2)
	if not x.shape == y.shape:
		raise ValueError('x and y must have the same shape')
	t = np.linspace(0, 2*np.pi, num, endpoint=False)
	if indexing=='xy':
		return np.dstack((
			x[:,:,None,None] + radius * np.cos(t)[None,None,None,:],
			y[:,:,None,None] + radius * np.sin(t)[None,None,None,:]))
	elif indexing=='ij':
		return np.dstack((
			x[:,:,None,None] - radius * np.sin(t)[None,None,None,:],
			y[:,:,None,None] + radius * np.cos(t)[None,None,None,:]))
	else:
		raise ValueError("Indexing must be either 'xy' or 'ij'")

def _interpolate_over_circles(field: int, radius, **kwargs):
	""" Returns the field values along circles centered at each element, through interpolation.

	Args:
		field: Field whose values shall be interpolated (with shape ``(:,:,2)``)
	
	Keyword Args:
		radius (int): radius of the circles
		num (int): number of interpolation points for each circle
		
	Return: 
		numpy.array: An array C with shape ``(field.shape[0], field.shape[1], 2, kwargs['num'])`` such that:
		
			- ``C[i,j,0,h]`` is the x-component of the field obtained by interpolation at the h-th point of the circle centered at pixel ``(i,j)``
			- ``C[i,j,1,h]`` its y-component
	"""
	# Read parameters
	if 'num' in kwargs: num = kwargs['num']
	else: num = 100
	# Create coordinates array 
	i, j = np.meshgrid(range(field.shape[0]), range(field.shape[1]), indexing='ij')
	coordinates = _get_circle_coord(i, j, radius, num, indexing='ij')
	# Swap dim-0 with dim-2 so that along the first dimension there are always (x,y) pairs
	# A flip is mandatory since map_coordinates require (i,j) pairs along the first axis
	coordinates = np.moveaxis(coordinates, 2, 0).reshape(2,-1)
	# Through interpolation recover field along circles (fac)
	fx, fy = np.dsplit(field, 2)
	facx = map_coordinates(fx.squeeze(), coordinates, order=1, mode='nearest', prefilter=False)
	facy = map_coordinates(fy.squeeze(), coordinates, order=1, mode='nearest', prefilter=False)
	# Reconstruct the array shape
	fac = np.stack((
		facx.reshape((field.shape[0], field.shape[1], num)), 
		facy.reshape((field.shape[0], field.shape[1], num))
		), axis=2)
	return fac

def generic_operator(field, **kwargs):
	""" Computes a field operator according to the specification.
	
	Args:
		field (numpy.array): Input field in doubled-angles format (with shape ``(:,:,2)``)
	
	Keyword Args:
		radius (int|float): Radius of the operator (defaults to 15)
		sample_dist (int|float): Distance (in pixels) between two consecutive sampling points of the integration path (defaults to 1)
		relaxation (float|array): Relaxation parameter, ranging in [0,1], or an array with the same shape of the field (except for depth) (defaults to 0.9)
		emphasis (str): Allows to specify which orientations should be emphasized. Can assume one of the following values:

			- none
			- tangent
			- normal
			
		concordance (str): Determine the signum of integrand. Can assume one of the following values:

			- none
			- center_real_field
			- adjuster
			- drifter 

		norm_fix (bool): Whether the norm of the final field should be fixed (i.e. if True, it is not allowed that an element gets a norm lower than the initial one)
		
	Return:
		numpy.array: The operator's output in doubled-angles format (same shape of input field).

	Important:
		The parameters can be combined at will. However, the concordance modes 'adjuster' and 'drifter' are special cases, hence some parameters will be forced to predefined values.

	See Also:
		:func:`drifterN`, :func:`drifterT`, :func:`smoother`, :func:`adjuster`
	"""
	# Handle keyword arguments
	radius = kwargs.get('radius', 15)
	step = kwargs.get('sample_dist', 1)
	num = int(2 * np.pi * radius / step)
	relax = kwargs.get('relaxation', 0.9)
	if isinstance(relax, np.ndarray):
		if len(relax.shape) != 2:
			raise ValueError("The relaxation matrix must be a 2D array")
		if relax.min() < 0 or relax.max() > 1:
			raise ValueError("The relaxation parameter shall be in [0,1]")
	else:
		if relax < 0 or relax > 1:
			raise ValueError("The relaxation parameter shall be in [0,1]")
	emphasis = kwargs.get("emphasis", 'normal')
	if not emphasis in ['none', 'normal', 'tangent']:
		raise ValueError("Selected emphasis strategy {} not allowed".format(emphasis))
	concordance = kwargs.get('concordance', 'center_real_field')
	if not concordance in ['none', 'center_real_field', 'adjuster', 'drifter']:
		raise ValueError("Selected concordance strategy {} not allowed".format(concordance))
	norm_fix = kwargs.get('norm_fix', True)
	# Apply some constraints
	if concordance == 'adjuster' and emphasis != 'tangent':
		emphasis = 'tangent'
		print('Adjuster concordance is forcing a tangent emphasis')
	elif concordance == 'drifter' and emphasis == 'none':
		raise ValueError("The emphasis {} is not suitable for the drifter operator".format(emphasis))
	elif concordance == 'drifter' and any([relax != 1.0, norm_fix == True]):
		relax = 1.0
		print('Drifter concordance is forcing the relaxation parameter to {}'.format(relax))
		norm_fix = False
		print('Drifter concordance is disabling the norm fix')
	# Halve and normalize the input field
	hfield = halve_angle(field)
	nhfield = normalize(hfield)
	if concordance in ['center_real_field', 'adjuster']:
		f = _interpolate_over_circles(hfield, radius, num=num)
	elif concordance == 'drifter':
		f = _interpolate_over_circles(nhfield, radius, num=num)
	else:
		f = _interpolate_over_circles(field, radius, num=num)
	# Compute the signum, if required (1st part of the integral)
	if concordance == 'center_real_field':
		signum = np.sign(dprod_2array(f, hfield[:,:,:,None], axis=2, keepDims=True))
	elif concordance == 'adjuster':
		nvec = _get_circle_coord(0, 0, 1, num=num)
		signum = np.sign(dprod_2array(f, nvec, axis=2, keepDims=True))
	else:
		signum = 1
	# Compute the proper dot product (2nd part of the integral)
	if emphasis == 'none':
		weight = 1
	else:
		nvec = _get_circle_coord(0, 0, 1, num=num)
		if emphasis == 'tangent': # rotate the circle points
			nvec = rot_2d(nvec)
		# Prevent recomputing the interpolation in case the drifter is required
		if concordance == 'drifter':
			nhf = f
		else:
			nhf = _interpolate_over_circles(nhfield, radius, num=num)
		weight = dprod_2array(nhf, nvec, axis=2, keepDims=True)
		# Square the dot product if required
		if concordance != 'drifter':
			weight **= 2
	# Compute the smoother field (put together the parts of the integral)
	sfield = (signum * weight * f).sum(axis=-1) / (2*radius)
	# Eventually double the field's angle
	if concordance in ['center_real_field', 'adjuster', 'drifter']:
		sfield = double_angle(sfield)
	# Apply the relaxation
	if isinstance(relax, np.ndarray):
		sfield = (1 - relax)[:,:,None] * field + relax[:,:,None] * sfield
	else:
		sfield = (1 - relax) * field + relax * sfield
	# Prevent the element-wise norm to decrease
	if norm_fix:
		sfield = normalize(sfield) * np.maximum(magnitude(sfield), magnitude(field))
	return sfield
	
def drifterT(field, radius=15, sample_dist=1):
	""" Computes the tangent drifter.
	
	Args:
		field (numpy.array): Input field in doubled-angles format (with shape ``(:,:,2)``)
		radius (int|float): Radius of the operator (defaults to 15)
		sample_dist (int|float): Distance (in pixels) between two consecutive sampling points of the integration path (defaults to 1)

	Note:
		This function simply calls :func:`generic_operator` with suitable parameters. See its documentation for better understanding.

	See Also:
		:func:`drifterN`, :func:`smoother`, :func:`adjuster`, :func:`generic_operator`
	"""
	kwargs = {
		'radius': radius,
		'sample_dist': sample_dist,
		'relaxation': 1,
		'emphasis': 'tangent',
		'concordance': 'drifter',
		'norm_fix': False
	}
	return generic_operator(field, **kwargs)
	
def drifterN(field, radius=15, sample_dist=1):
	""" Computes the normal drifter.
	
	Args:
		field (numpy.array): Input field in doubled-angles format (with shape ``(:,:,2)``)
		radius (int|float): Radius of the operator (defaults to 15)
		sample_dist (int|float): Distance (in pixels) between two consecutive sampling points of the integration path (defaults to 1)

	Note:
		This function simply calls :func:`generic_operator` with suitable parameters. See its documentation for better understanding.

	See Also:
		:func:`.drifterT`, :func:`smoother`, :func:`adjuster`, :func:`generic_operator`
	"""
	kwargs = {
		'radius': radius,
		'sample_dist': sample_dist,
		'relaxation': 1,
		'emphasis': 'normal',
		'concordance': 'drifter',
		'norm_fix': False
	}
	return generic_operator(field, **kwargs)

def drifter_mask(field: Field, threshold: float, radius: int = 15, sample_dist: float = 1.0, markLoops: bool = True, markDeltas: bool = True) -> Mask:
	""" Computes a mask of loops and deltas from the drifters responses.

	Args:
		field: Input field in doubled-angles format (with shape ``(:,:,2)``)
		threshold: Threshold value used to locate loops and deltas (must be in [0,1])
		radius: Radius of the operator (defaults to 15)
		sample_dist: Distance (in pixels) between two consecutive sampling points of the integration path (defaults to 1.0)
		markLoops: Whether the resulting mask should highlight loops (defaults to True)
		markDeltas: Whether the resulting mask should highlight loops (defaults to True)

	Return:
		A mask with loops and/or deltas marked.

	Note:
		The tangent and normal drifters gives very similar results. We combine them for improved robustness.
	"""
	cfield = reflection(field)
	mag = magnitude(drifterT(field, radius, sample_dist)) + magnitude(drifterT(field, radius, sample_dist)) - magnitude(drifterT(cfield, radius, sample_dist)) - magnitude(drifterT(cfield, radius, sample_dist))
	mag = mag.squeeze()
	mask = np.zeros(field.shape[:-1], dtype=bool)
	if markLoops:
		mask = np.logical_or(mask, mag > 2 * threshold)
	if markDeltas:
		mask = np.logical_or(mask, mag < - 2 * threshold)
	return mask
	
def smoother(field, radius=15, sample_dist=1, relax=0.9):
	""" Computes the smoother.
	
	Args:
		field (numpy.array): Input field in doubled-angles format (with shape ``(:,:,2)``)
		radius (int|float): Radius of the operator (defaults to 15)
		sample_dist (int|float): Distance (in pixels) between two consecutive sampling points of the integration path (defaults to 1)
		relax (float): Relaxation parameter, ranging in [0,1] (defaults to 0.9)

	Note:
		This function simply calls :func:`generic_operator` with suitable parameters. See its documentation for better understanding.

	See Also:
		:func:`drifterN`, :func:`drifterT`, :func:`adjuster`, :func:`generic_operator`
	"""
	kwargs = {
		'radius': radius,
		'sample_dist': sample_dist,
		'relaxation': relax,
		'emphasis': 'normal',
		'concordance': 'center_real_field',
		'norm_fix': True
	}
	return generic_operator(field, **kwargs)
	
def adjuster(field, radius=15, sample_dist=1, relax=0.9):
	""" Computes the adjuster.
	
	Args:
		field (numpy.array): Input field in doubled-angles format (with shape ``(:,:,2)``)
		radius (int|float): Radius of the operator (defaults to 15)
		sample_dist (int|float): Distance (in pixels) between two consecutive sampling points of the integration path (defaults to 1)
		relax (float): Relaxation parameter, ranging in [0,1] (defaults to 0.9)

	Note:
		This function simply calls :func:`generic_operator` with suitable parameters. See its documentation for better understanding.

	See Also:
		:func:`drifterN`, :func:`drifterT`, :func:`smoother`, :func:`generic_operator`
	"""
	kwargs = {
		'radius': radius,
		'sample_dist': sample_dist,
		'relaxation': relax,
		'emphasis': 'tangent',
		'concordance': 'adjuster',
		'norm_fix': True
	}
	return generic_operator(field, **kwargs)
	