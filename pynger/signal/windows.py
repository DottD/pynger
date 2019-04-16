import numpy as np
from scipy.signal.windows import gaussian


def gaussWin2D(shape, sigma=None):
	"""
	Create a 2D Gaussian window
	The shape must have 2 components, namely the vertical and horizontal,
	in this order.
	"""
	# Check input
	if len(shape) == 1:
		shape = [shape[0] for _ in range(2)]
	elif len(shape) > 2:
		shape = shape[:2]
	shape = [max([1, int(x)]) for x in shape]
	if not sigma:
		sigma = [x/2.0 for x in shape]
	else:
		if len(sigma) == 1:
			sigma = [sigma[0] for _ in range(2)]
		elif len(sigma) > 2:
			sigma = sigma[:2]
		sigma = [np.finfo(np.float32).eps if x <= 0 else x for x in sigma]
	# Create vertical and horizontal components
	v = gaussian(shape[0], sigma[0])
	v = np.reshape(v, (-1, 1)) # column
	h = gaussian(shape[1], sigma[1])
	h = np.reshape(h, (1, -1)) # row
	return np.dot(v, h)
		
def circleWin(rad, retCoordinate=False):
	rad_int = int(np.ceil(rad))
	rad_span = np.arange(-rad_int, rad_int+1)
	x, y = np.meshgrid(rad_span, rad_span, indexing='xy')
	kernel = x*x + y*y <= rad*rad
	if retCoordinate:
		return kernel.astype(int), x, np.flipud(y)
	else:
		return kernel.astype(int)