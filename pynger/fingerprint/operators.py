import numpy as np
from scipy.ndimage import map_coordinates
from pynger.signal.windows import circleWin
from pynger.field.manipulation import halve_angle, double_angle, normalize, magnitude, dprod_2array, reflection
from pynger.field.calculus import rot_2d
from pynger.types import Mask, Field
from warnings import warn

def _get_circle_coord(x, y, radius, num=100, indexing='xy'):
	""" Returns the coordinates of a circle with given parameters.

	Args:
		x (numpy.array): First coordinate of the center of the circle
		y (numpy.array): Second coordinate of the center of the circle
		radius (float): Radius of the circle
		num (int): Number of sampling points over the circle
		indexing (str): if 'xy' then x and y are treated as such, otherwise, if 'ij', they are treated as i and j. Furthermore, the returned coordinates will be xy if 'xy' and ij if 'ij'.
	
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
	num = kwargs.get('num', 100)
	axis = kwargs.get('axis', 2)
	# Create coordinates array 
	i, j = np.meshgrid(range(field.shape[0]), range(field.shape[1]), indexing='ij')
	coordinates = _get_circle_coord(i, j, radius, num, indexing='ij')
	# Swap dim-0 with dim-2 so that along the first dimension there are always (x,y) pairs
	coordinates = np.moveaxis(coordinates, 2, 0).reshape(2,-1)
	# A flip is mandatory since map_coordinates require (i,j) pairs along the first axis
	# coordinates = np.flipud(coordinates)
	# Take the specified axis as the first one, in order to loop over it
	field = np.moveaxis(field, axis, 0)
	# For each element along the given dimension, compute the interpolation over circles
	r = radius
	si = field.shape[1]
	sj = field.shape[2]
	O = []
	for M in field:
		M = map_coordinates(M, coordinates, order=1, mode='nearest', prefilter=False)
		O.append( M.reshape((si, sj, num)) )
	return np.stack(O, axis=2)

def _fast_interp_over_circles(field, radius, num=100, axis=2):
	num = int(num)
	radius = int(radius)
	if not (num > 0 and radius > 0):
		raise RuntimeError("Radius and num must be positive!")
	if len(field.shape) > 3:
		raise NotImplementedError('Number of field\'s dimensions shall not exceed 3')
	# Create coordinates array in (i,j) format
	coordinates = _get_circle_coord(0, 0, radius, num, indexing='ij')
	# Kind of NN-interpolation
	coordinates = np.round(coordinates.squeeze()).astype(int)
	# A flip is mandatory since below (i,j) pairs along the first axis are required
	# coordinates = np.flipud(coordinates)
	# Take the specified axis as the first one, in order to loop over it
	field = np.moveaxis(field, axis, 0)
	# For each element along the given dimension, compute the interpolation over circles
	r = radius
	si = field.shape[1]
	sj = field.shape[2]
	O = []
	for M in field:
		M = np.pad(M, r, mode='edge')
		O.append( np.dstack( [M[r+mi:r+mi+si, r+mj:r+mj+sj] for mi, mj in coordinates.T] ) )
	return np.stack(O, axis=2)

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
			- center
			- tangent
			- normal
			
		concordance (str): Determine the signum of integrand. Can assume one of the following values:

			- none
			- center
			- tangent
			- normal

		norm_fix (bool): Whether the norm of the final field should be fixed (i.e. if True, it is not allowed that an element gets a norm lower than the initial one)
		use_fast (bool): Whether the fast interpolation should be used
		weight_power (float): Power to which the weight is raised
		norm_factor (str): the strategy to use for the normalization factor outside the integral - either 'estimated' for 2r or 'true' for the sum of weights used inside the integral
		field_mod (str): modification that will be applied to the field in the integral

			- none
			- half_angle
			- normalized_half_angle
		
	Return:
		numpy.array: The operator's output in doubled-angles format (same shape of input field).

	Important:
		The parameters can be combined at will, and refers to the generic operator:
			$$ \mathcal{O}[\mathcal{F}](x_0) 1/W \int_{\mathcal{C}(x_0)} c(x_0,x) \cdot w(x_0,x)^p \cdot f(x) dx $$
		where $W=\sum w^p$ is the total weight and acts as a normalization factor, $R$ is the image domain, $c(x_0,x)$ is a function that assumes only values in ${-1,1}$ and determines the sign of the field $f(x)$ at each point of the integration path, and finally $w(x_0,x)$ is a weighting function.

	See Also:
		:func:`drifterN`, :func:`drifterT`, :func:`smoother`, :func:`adjuster`
	"""
	# Handle keyword arguments
	radius = kwargs.get('radius', 15)
	if radius < 1:
		warn("Generic operator: radius must be positive - the operator returns the input as is")
		return field
	step = kwargs.get('sample_dist', 4)
	num = int(2 * np.pi * radius / step)
	if num < 1:
		warn("Generic operator: num must be positive - the operator returns the input as is")
		return field
	relax = kwargs.get('relaxation', 0.9)
	if isinstance(relax, np.ndarray):
		if len(relax.shape) != 2:
			raise ValueError("The relaxation matrix must be a 2D array")
		if (relax < 0).any() or (relax > 1).any():
			raise ValueError("The relaxation parameter shall be in [0,1]")
	else:
		if relax < 0 or relax > 1:
			raise ValueError("The relaxation parameter shall be in [0,1]")
	emphasis = kwargs.get("emphasis") # mandatory
	if emphasis not in ['none', 'normal', 'tangent']:
		raise ValueError("Selected emphasis strategy {} not allowed".format(emphasis))
	concordance = kwargs.get('concordance') # mandatory
	if not concordance in ['none', 'center', 'normal', 'tangent']:
		raise ValueError("Selected concordance strategy {} not allowed".format(concordance))
	norm_fix = kwargs.get('norm_fix', True)
	use_fast = kwargs.get('use_fast', False)
	if use_fast:
		circular_interpolation = _fast_interp_over_circles
	else:
		circular_interpolation = _interpolate_over_circles
	weight_power = kwargs.get('weight_power', 2)
	norm_factor = kwargs.get('norm_factor', 'true')
	field_mod = kwargs.get('field_mod', 'half_angle')

	# Eventually compute some modified versions of the input field
	if concordance != 'none' or emphasis != 'none':
		# Halve and normalize the input field
		nhfield = normalize(halve_angle(field))
		# Perform interpolation over circles
		nhf = circular_interpolation(nhfield, radius, num=num)
	else:
		nhf = None
	# Define a function that computes the projection of a field onto a reference vector
	# (among the possible choices)
	def project(F, ref_vec_descr):
		if ref_vec_descr == 'none':
			proj = np.array(1)
		else:
			# Reference vector (rvec)
			if ref_vec_descr == 'center':
				rvec = nhfield[:,:,:,None]
			elif ref_vec_descr in ['tangent', 'normal']:
				rvec = _get_circle_coord(0, 0, 1, num=num, indexing='xy')
				if ref_vec_descr == 'tangent':
					rvec = rot_2d(rvec)
			else:
				raise NotImplementedError('Reference vector not supported!')
			proj = dprod_2array(F, rvec, axis=2, keepDims=True)
		return proj
	# Compute the signum
	signum = project(nhf, concordance)
	# When two vectors are almost (or fully) orthogonal, the sign should be preserved, otherwise it may lead to frequent oscillations depending on floating point errors
	signum[np.isclose(np.abs(signum), 0)].fill(1)
	signum = np.sign(signum)
	# Compute the emphasis
	weight = project(nhf, emphasis)
	# Compute the absolute value to enforce them to be consistent weights
	weight = np.abs(weight) ** weight_power
	# Choose the normalization factor
	if norm_factor == 'estimated':
		W = 2 * radius
	elif norm_factor == 'true':
		W = weight.sum(axis=-1)
		W[np.isclose(W, 0)] = 1 # prevents div by 0 errors
	else:
		raise NotImplementedError('Normalization factor not supported!')
	# Compute the resulting field (put together the parts of the integral)
	if field_mod == 'none':
		mf = circular_interpolation(field, radius, num=num)
	elif field_mod == 'half_angle':
		if concordance == 'none' and emphasis == 'none':
			hfield = halve_angle(field)
			mf = circular_interpolation(hfield, radius, num=num)
		else:
			# If nhf is already available, exploit it in combination with the field norm
			norm = magnitude(field)
			hf_norm = circular_interpolation(norm, radius, num=num)
			mf = hf_norm * nhf
	elif field_mod == 'normalized_half_angle':
		if concordance == 'none' and emphasis == 'none':
			nhfield = normalize(halve_angle(field))
			nhf = circular_interpolation(nhfield, radius, num=num)
		mf = nhf
	rfield = (signum * weight * mf).sum(axis=-1) / W
	if field_mod != 'none':
		rfield = double_angle(rfield)
	# Apply the relaxation
	if isinstance(relax, np.ndarray):
		rfield = (1 - relax)[:,:,None] * field + relax[:,:,None] * rfield
	else:
		rfield = (1 - relax) * field + relax * rfield
	# Prevent the element-wise norm to decrease
	if norm_fix:
		rfield = normalize(rfield) * np.maximum(magnitude(rfield), magnitude(field))
	return rfield
	##### FieldAdjust funziona solamente con 0.9, altrimenti appiattisce tutto
	##### TODO: capire perché i drifter sbombano se metto il vero peso a dividere
	##### TODO: si può reimplentare, così come molte cose, con i generatori e lazy evaluations, oltre che con la classe memory di joblib per non calcolare più volte le stesse cose
	
def drifterT(field, radius=15, sample_dist=4):
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
		'concordance': 'tangent',
		'weight_power': 1, # abs val of weights
		'norm_factor': 'true',
		'norm_fix': False,
		'field_mod': 'normalized_half_angle'
	}
	return generic_operator(field, **kwargs)
	
def drifterN(field, radius=15, sample_dist=4):
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
		'concordance': 'normal',
		'weight_power': 1, # abs val of weights
		'norm_factor': 'true',
		'norm_fix': False,
		'field_mod': 'normalized_half_angle'
	}
	return generic_operator(field, **kwargs)

def drifter_mask(field: Field, threshold: float, radius: int = 15, sample_dist: float = 4.0, markLoops: bool = True, markDeltas: bool = True) -> Mask:
	""" Computes a mask of loops and deltas from the drifters responses.

	Args:
		field: Input field in doubled-angles format (with shape ``(:,:,2)``)
		threshold: Threshold value used to locate loops and deltas (must be in [0,1])
		radius: Radius of the operator (defaults to 15)
		sample_dist: Distance (in pixels) between two consecutive sampling points of the integration path (defaults to 1.0)
		markLoops: Whether the resulting mask should highlight loops (defaults to True)
		markDeltas: Whether the resulting mask should highlight loops (defaults to True)

	Return:
		A mask with loops and/or deltas marked. If maskLoops and markDeltas are both false, the drifter image is returned.

	Note:
		The tangent and normal drifters gives very similar results. We combine them for improved robustness.
	"""
	cfield = reflection(field)
	mag = magnitude(drifterT(field, radius, sample_dist)) + magnitude(drifterN(field, radius, sample_dist)) - magnitude(drifterT(cfield, radius, sample_dist)) - magnitude(drifterN(cfield, radius, sample_dist))
	mag = mag.squeeze()
	mag -= mag.min() # should be 
	mag /= mag.max() # mandatory...
	markL = lambda: mag > 2 * threshold
	markD = lambda: mag < - 2 * threshold
	if markLoops and not markDeltas:
		return markL()
	elif not markLoops and markDeltas:
		return markD()
	elif markLoops and markDeltas:
		return np.logical_or(markL(), markD())
	else:
		return mag
	
def smoother(field, radius=15, sample_dist=4, relax=1):
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
		'concordance': 'center',
		'weight_power': 2, # abs val of weights
		'norm_factor': 'estimated', # of true they go crazy
		'norm_fix': True,
		'field_mod': 'half_angle'
	}
	return generic_operator(field, **kwargs)
	
def adjuster(field, radius=15, sample_dist=4, relax=0.9):
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
		'concordance': 'tangent',
		'weight_power': 2, # abs val of weights
		'norm_factor': 'estimated', # of true they go crazy
		'norm_fix': True,
		'field_mod': 'half_angle'
	}
	return generic_operator(field, **kwargs)
	