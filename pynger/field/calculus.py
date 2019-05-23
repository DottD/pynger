import numpy as np
from pynger.signal.pani import fast_ani_gauss_filter
	
def z_curl(field=None, sigma=1, dfydx=None, dfxdy=None):
	"""
	Computes z component of the curl of input field.

	Args:
		- field, input field from which the curl has to be computed (ignored is dfydx and dfxdy are provided);
		- sigma, gaussian smoothing std dev;
		- dfydx, partial derivative of the y-component of field along the x-axis (ignored if field is provided);
		- dfxdy, partial derivative of the x-component of field along the y-axis (ignored if field is provided).
		
	Return a scalar, that is the z component of the curl of the input vector field.
	"""
	if (dfydx is None) and (dfxdy is None) and not (field is None):
		return ddx(field[:,:,1], sigma) - ddy(field[:,:,0], sigma)
	elif not (dfydx is None) and not (dfxdy is None) and (field is None):
		return dfydx - dfxdy
	else:
		raise ValueError("Either the field or both dfydx and dfxdy must be provided")
	
def ddx(scalar, sigma):
	""" Computes derivatives along x-axis """
	return fast_ani_gauss_filter(scalar, 1, sigma, 0, 0, 1)
	
def ddy(scalar, sigma):
	""" Computes derivatives along y-axis """
	return fast_ani_gauss_filter(scalar, sigma, 1, 0, 1, 0)
	
def grad(scalar=None, sigma=1, partial_derx=None, partial_dery=None):
	"""
	Computes the gradient of input scalar function.
	
	Args:
		- scalar, input matrix from which the gradient has to be computed (ignored is ddx and ddy are provided);
		- sigma, gaussian smoothing std dev;
		- partial_derx, partial derivative along the x-axis (ignored if scalar is provided);
		- partial_dery, partial derivative along the y-axis (ignored if scalar is provided).
		
	Returns a vector field, that is the gradient of the input function.
	"""
	if ((partial_derx is None) or (partial_dery is None)) and not (scalar is None):
		return np.dstack((ddx(scalar, sigma), ddy(scalar, sigma)))
	elif not (partial_derx is None) and not (partial_dery is None) and (scalar is None):
		return np.dstack((partial_derx, partial_dery))
	else:
		raise ValueError("Either the scalar or both partial_derx and partial_dery must be provided")
	
def z_vprod(a, b):
	""" Computes the z component of the vector product of the input vector fields """
	return a[:,:,0]*b[:,:,1] - a[:,:,1]*b[:,:,0]
	
def rot_2d(field):
	""" Returns the input field rotated counterclockwise by 90° """
	fx, fy = np.dsplit(field, 2)
	return np.dstack((-fy, fx))
	
def irot_2d(field):
	""" Returns the input field rotated counterclockwise by 90° """
	fx, fy = np.dsplit(field, 2)
	return np.dstack((fy, -fx))
	