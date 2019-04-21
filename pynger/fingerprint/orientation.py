from warnings import warn
import numpy as np
from scipy.signal import argrelmax
from scipy.ndimage import minimum_filter, map_coordinates, gaussian_filter1d
from scipy.fftpack import rfft
from pynger.signal.pani import fast_ani_gauss_filter
from pynger.signal.windows import circleWin


def LRO(image, **kwargs):
	""" Compute the local ridge orientation (LRO) for the input fingerprint.

	Args:
		image (numpy.array): The image to analyze
	
	Keyword Args:
		ridge_dist (int): estimated average distance between ridges (defaults to 10)
		number_angles (int): number of angles to be tested (defaults to 36)
		along_sigma_ratio (float): width of filter along the orientation (relative to ridge distance) (defaults to 0.85)
		ortho_sigma (float): width of filter orthogonal to the orientation (relative to ridge distance) (defaults to 0.05)
		mask (numpy.array or None): Mask indicating the region of interest (same shape of image)
		
	Returns:
		tuple of numpy.array: The LRO and its reliability.
	"""
	# Read parameters
	ridge_dist = kwargs.get('ridge_dist', 15)
	num = kwargs.get('number_angles', 36)
	along_sigma = kwargs.get('along_sigma_ratio', 0.3) * ridge_dist
	ortho_sigma = kwargs.get('ortho_sigma', 0.05) * ridge_dist
	# LRO Estimation
	lro_x = np.zeros(image.shape)
	lro_y = np.zeros_like(lro_x)
	resp_inf_norm = np.zeros(image.shape)
	resp_2_norm = np.zeros(image.shape)
	for _, theta in enumerate(np.linspace(0., np.pi, num, endpoint=False)):
		response = fast_ani_gauss_filter(image, ortho_sigma, along_sigma, np.rad2deg(theta), 1, 0)
		response = np.fabs(response)
		response = fast_ani_gauss_filter(response, ridge_dist, ridge_dist, 0, 0, 0)
		lro_x += response * np.cos(2*theta)
		lro_y += response * np.sin(2*theta)
		# Reliability 
		resp_inf_norm = np.maximum(response, resp_inf_norm)
		resp_2_norm += response*response
	lro = np.arctan2(lro_y, lro_x) / 2
	# Normalize with infinite norm (maximum)
	# Reliability as the L2-norm inversely rescaled from [1,sqrt(num)] to [0,1]
	safe_mask = resp_inf_norm != 0
	rel = np.zeros(resp_2_norm.shape)
	rel[safe_mask] = np.sqrt(resp_2_norm[safe_mask])/resp_inf_norm[safe_mask]
	rel = 1. - (rel-1.)/(np.sqrt(num)-1.)
	# Eventually apply the input mask
	mask = kwargs.get('mask', None)
	if not (mask is None):
		rel *= mask
	return lro, rel
	
def LRF(image, lro, rel, **kwargs):
	""" Computes the local ridge frequency of the given image.

	Args:
		image (numpy.array): The fingerprint image
		lro (numpy.array): The local ridge orientation
		rel (numpy.array): The reliability of each orientation

	Note:
		Each input parameter must have the same shape.

	Keyword Args:
		min_disk_size (int): size of the disk neighborhood of each point for the reliability check
		rel_check_grid_step (int): step of the grid used for the reliability check
		rel_check_threshold (float): threshold value used for the reliability check (expressed as a percentile of the input rel matrix) (defaults to 30)
		segment_n_points (int): number of points for each segment
		segment_length (int): length of each segment (in pixels)
		gaussian_smooth_std (float): standard deviation of gaussian filter used to smooth the signal on each segment (with respect to segment's length)

	Returns:
		float: The average ridge distance.
	"""
	min_disk_size = kwargs.get('min_disk_size', 10)
	grid_step = kwargs.get('rel_check_grid_step', 10)
	grid_thres = np.percentile(rel, kwargs.get('rel_check_threshold', 30))
	sp_num = kwargs.get('segment_n_points', 15)
	sp_len = kwargs.get('segment_length', 30)
	sigma = kwargs.get('gaussian_smooth_std', 0.1)
	# Parameters
	sp_len = int((sp_len//2)+1)
	# Reliability check
	min_rel = minimum_filter(rel, footprint=circleWin(min_disk_size))
	checked = np.zeros(min_rel.shape, dtype=bool)
	checked[sp_len:-sp_len:grid_step, sp_len:-sp_len:grid_step] = min_rel[sp_len:-sp_len:grid_step, sp_len:-sp_len:grid_step] >= grid_thres
	del min_rel
	I, J = np.nonzero(checked)
	del checked
	if I.size==0 or J.size==0:
		warn('No point passed the reliability check', RuntimeWarning)
		return None
	# Get segments by interpolation
	t = np.linspace(-sp_len, sp_len, sp_num) # endpoint included
	segments_xy = np.array([[j-np.sin(lro[i,j])*t, i+np.cos(lro[i,j])*t] for i,j in zip(I,J)])
	del t
	segments = map_coordinates(image, [segments_xy[:,0,:], segments_xy[:,1,:]], order=1, prefilter=False)
	del segments_xy
	# Compute spectrum of each segment
	dft = rfft(segments, axis=1)**2
	del segments
	spectrum = np.zeros((dft.shape[0], dft.shape[1]//2+1))
	spectrum[:,0] = 0 # null DC component (zero mean)
	if sp_num % 2 == 0: # even
		spectrum[:,1:-1] = dft[:,1:-1:2]+dft[:,2:-1:2]
		spectrum[:,-1] = dft[:,-1]
	else: # odd
		spectrum[:,1:] = dft[:,1::2]+dft[:,2::2]
	del dft
	smoothed = gaussian_filter1d(spectrum, sigma*sp_len, axis=1)
	del spectrum
	# Find the first peak, then the frequency
	max_i, max_j = argrelmax(smoothed, axis=1, order=2)
	if max_i.size==0 or max_j.size==0:
		warn('No peak found in spectrum segments', RuntimeWarning)
		return None
	# fs = 0.0
	# last = -1
	# for i,j in zip(max_i, max_j):
	# 	if i != last:
	# 		fs += j
	# 		last = i
	# fs *= sp_len / (2*smoothed.shape[0])
	# return fs
	### 
	mi, mi_idx = np.unique(max_i, return_index=True)
	mj = max_j[mi_idx]
	freq = np.fft.fftfreq(sp_num, d=(2*sp_len)/(sp_num-1))
	return 1/np.interp(mj.mean(), np.arange(freq.size), freq)

if __name__ == '__main__':	
	# import argparse
	from PIL import Image
	from scipy.ndimage.morphology import binary_closing, binary_opening
	from scipy.ndimage.measurements import label as binary_label
	# Parsing arguments
	# parser = argparse.ArgumentParser(description="Compute the LRO of the input fingerprint.")
	# parser.add_argument("in", help="Path to the fingerprint image")
	# parser.add_argument("-s", "--show", action='store_true', help="Whether to show images step by step")
	# parser.add_argument("-d", "--ridge-distance", default=10, type=int, help="A priori estimated distance between ridges")
	# parser.add_argument("-n", "--number-angles", default=16, type=int, help="Size of the circular smoothing window")
	# args = vars(parser.parse_args())
	# # Parameters
	# filename = args['in']
	# show = args['show']
	# ridge_dist = args['ridge_distance']
	# num = args['number_angles']
	filename = "/Users/MacD/Documents/Databases/FOESamples/Good/113.bmp"
	show = False
	ridge_dist = 15
	num = 32
	# Read image
	image = np.array(Image.open(filename).convert('L'))
	# Segmentation (rough)
	mask = binary_closing(image < np.percentile(image, 30), structure=circleWin(10), iterations=1)
	label, tot = binary_label(mask)
	idx = np.array([np.count_nonzero(label==n) for n in range(tot)]).argmax()
	mask = label == 4
	mask = binary_opening(mask, structure=circleWin(10), iterations=3)
	mask = label == 4
	i,j = mask.nonzero()
	l, r, t, b = (j.min(), j.max(), i.min(), i.max())
	image = image[t:b,l:r]
	
	from pynger.misc import testTime, vis_orient
	fun = lambda: LRO(image,
		ridge_dist=ridge_dist, number_angles=num, 
		along_sigma_ratio=0.85, ortho_sigma=1.0)
	lro, rel = testTime(fun, rep=1, msg="LRO time: {} seconds")
	fun = lambda: LRF(image, lro, rel, **{'min_disk_size': 15})
	Ts = testTime(fun, rep=1, msg="LRF time: {} seconds")
	print("Estimated ridge distance: ", Ts)
	if show:
		import matplotlib.pyplot as plt
		from matplotlib import cm
		from matplotlib.colors import from_levels_and_colors
		plt.subplot(121)
		plt.imshow(image, cmap='gray')
		vis_orient(lro, step=15, color='r')
		plt.subplot(122)
		rel = np.round(rel*5.)/5.
		cmap, norm = from_levels_and_colors(levels=[0.0,0.2,0.4,0.6,0.8,1.0], colors=['0.0','0.25','0.5','0.75','1.0'])
		plt.imshow(rel, cmap=cmap, norm=norm)
		plt.show()
		