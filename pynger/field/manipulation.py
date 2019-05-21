import numpy as np
from functools import reduce
from pynger.types import Field


def _decode_args(*args):
	""" Transform args into (x,y) if possible """
	if len(args)==1: return np.dsplit(args[0], 2)
	elif len(args)==2: return (args[0], args[1])
	else: raise ValueError("Wrong number of input args")

def magnitude(*args, keepDims=True):
	""" Returns magnitude of input field """
	x, y = _decode_args(*args)
	magnitude = np.round(np.sqrt(x*x+y*y), decimals=8)
	if keepDims: return magnitude
	else: return magnitude.squeeze()

def angle(*args, keepDims=True, unoriented=False):
	""" Returns phase angle of input field """
	x, y = _decode_args(*args)
	angle = np.arctan2(y, x)
	if unoriented:
		angle = np.remainder(angle, np.pi)
	if keepDims: return angle
	else: return angle.squeeze()

def cart2polar(*args, keepDims=True):
	""" Convert cartesian coordinates to polar. """
	return angle(*args, keepDims=keepDims), magnitude(*args, keepDims=keepDims)
	
def polar2cart(angle, magnitude, retField=False):
	""" Returns the field from the input phase angle and magnitude. 
	
	Args:
		- angle, phase angle of the field (2 dims);
		- magnitude, magnitude of the field (2 dims);
		- retField, if True returns a field, otherwise returns (x,y).
	"""
	if retField:
		if np.isscalar(magnitude):
			return np.dstack((np.cos(angle), np.sin(angle)))*magnitude
		else:
			return np.dstack((np.cos(angle), np.sin(angle)))*magnitude[:,:,None]
	else: 
		return np.cos(angle)*magnitude, np.sin(angle)*magnitude
	
def halve_angle(field):
	""" 
	Halves the phase angle of the input field. 
	Args:
		- field, the input field as an image with depth 2 (rows, cols, uv). Actually it could be like (rows, cols, uv, ...).
	"""
	return polar2cart(angle(field, keepDims=False)/2, magnitude(field, keepDims=False), retField=True)
	
def double_angle(field):
	""" 
	Doubles the phase angle of the input field. 
	Args:
		- field, the input field as an image with depth 2 (rows, cols, uv). Actually it could be like (rows, cols, uv, ...).
	"""
	return polar2cart(angle(field, keepDims=False)*2, magnitude(field, keepDims=False), retField=True)
	
def reflection(field: Field) -> Field:
	""" Computes the pixel-wise reflection around the horizontal axis. 

	Args:
		field: the input field as an image with depth 2 (rows, cols, uv). Actually it could be like (rows, cols, uv, ...)

	Return:
		The pixel-wise reflection around the horizontal axis of the input field.
	"""
	return polar2cart(-angle(field, keepDims=False), magnitude(field, keepDims=False), retField=True)
	
def set_magnitude(field, magnitude):
	""" 
	Set the magnitude of the input field to the desired values.
	Args:
		- field, the input field as an image with depth 2 (rows, cols, uv). Actually it could be like (rows, cols, uv, ...).
		- magnitude, broadcastable to field.
	Returns: resulting field with magnitude set to desired values.
	"""
	return polar2cart(angle(field), magnitude=magnitude, retField=True)
	
def normalize(field, safe: bool = True):
	""" Normalizes the input field.
	
	Args:
		safe: when True, prevents null elements from being normalized
	"""
	if safe:
		mag = magnitude(field, keepDims=False) > 0
	else:
		mag = 1
	return set_magnitude(field, mag)
	
def reduce_2array(A, B, fun, axis=-1, initializer=0.0, keepDims=False):
	"""
	Compute the dot product along the given axis. 
	
	a and b must have exactly the same shape, as the axis dimension will be reduced by element-wise multiplication and summation along that dimension.
	Args:
		- a, input N-D array_like;
		- b, input N-D array_like;
		- axis, axis along which the dot product should be performed (defaults to -1);
		- fun, function that takes three arguments (previously accumulated value, element of A, element of B) and return a value that will be progressively accumulated throughout the arrays (see functools.reduce for similar requirements); for instance, `lambda accum, a, b: accum + a*b` makes the function compute the dot product along the specified axes.
		- initializer, initial value of accumulated value.
	Returns an N-D array with the given axis reduced.
	"""
	AA, BB = np.broadcast_arrays(A, B)
	out = reduce(
		lambda accum, tup: fun(accum, tup[0], tup[1]), 
		zip(np.moveaxis(AA,axis,0), np.moveaxis(BB,axis,0)),
		initializer)
	if keepDims: return np.expand_dims(out, axis)
	else: return out
	
def dprod_2array(A, B, axis=-1, keepDims=False):
	return reduce_2array(A, B, axis=axis, fun=lambda accum, a, b: accum + a*b, keepDims=keepDims)
	
def angle_diff(t1, t2):
	""" Compute the element-wise difference between the two given array of angles.
	
	The law of cosines is used to take into account circularity.
	Input angles are also doubled, considering them in the range [-pi,pi]
	"""
	u = np.exp(2j*t1)
	v = np.exp(2j*t2)
	prod = u*np.conj(v)
	re = np.real(prod)
	im = np.imag(prod)
	re[np.where(re > 1.)] = 1. # solves problems with float precision...
	re[np.where(re < -1.)] = -1. # solves problems with float precision...
	return - 0.5 * np.sign(im) * np.arccos(re)

def angle_div(t1, t2):
	""" Compute the element-wise division between the two given array of angles.
	"""
	mask = np.where(t2 == 0.)
	t2[mask], t1[mask] = 1., 0.
	return np.angle(np.exp(1j*t1/t2))
