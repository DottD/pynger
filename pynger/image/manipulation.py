import numpy as np


def scale_to_range(image, dest_range=(0,1)):
	""" Scale an image to the given range.
	"""
	return np.interp(image, xp=(image.min(), image.max()), fp=dest_range)
	
	
if __name__ == '__main__':
	image = np.random.random_integers(0, 255, (25, 30))
	print(image.min(), image.max())
	scaled = scale_to_range(image, (0,1))
	print(scaled.min(), scaled.max())
	scaled = scale_to_range(image, (4,67))
	print(scaled.min(), scaled.max())