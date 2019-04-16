import PIL
import matplotlib.pyplot as plt
import numpy as np
from pynger.misc import vis_orient, vis_stream_orient
from pynger.field.manipulation import _decode_args, cart2polar, magnitude
from pynger.image.manipulation import scale_to_range


def draw_image(image):
	""" Returns a PIL Image that can be surely visualized in IPython """
	return PIL.Image.fromarray(scale_to_range(image, (0,255)).astype('uint8'))
	
def draw_images(images):
	""" Returns a PIL Image originated from a list of images. """
	pimgs = list(map(draw_image, images))
	widths, heights = zip(*(im.size for im in pimgs))
	
	total_width = sum(widths)
	max_height = max(heights)

	new_im = PIL.Image.new('L', (total_width, max_height))

	x_offset = 0
	for im in pimgs:
		new_im.paste(im, (x_offset,0))
		x_offset += im.size[0]
	return new_im
	
def draw_field_magnitude(field):
	""" Returns a PIL Image with the field's magnitude """
	return draw_image(magnitude(field, keepDims=False))
	
def draw_fields_magnitude(fields):
	""" Returns a PIL Image with the field's magnitude """
	return draw_images(map(lambda field: magnitude(field, keepDims=False), fields))

def plot_vector_field(image, *args, **kwargs):
	""" Returns an IPython Image representation of the given vector field.
	
	Args:
		image: The vector field will be drawn over this image. Either a PIL.Image or a numpy.array with proper shape and dtype (see PIL documentation).
		args: List of vector fields to be plotted, either as two lists with the x and y components respectively, or as list of fields ((n,m,2)-shaped arrays). If the keyword argument `polar` is set to True, the function expects the field components to be polar.
		kwargs: See pynger.misc.vis_orient for all the possible key-value pairs. 
		
	Returns:
		A PIL.Image.Image showing the input image with each specified field overlaid.
		
	"""
	_vector_visualization(vis_orient, image, *args, **kwargs)

def stream_vector_field(image, *args, **kwargs):
	""" Returns an IPython Image representation of the given vector field.
	
	Args:
		image: The vector field will be drawn over this image. Either a PIL.Image or a numpy.array with proper shape and dtype (see PIL documentation).
		args: List of vector fields to be plotted, either as two lists with the x and y components respectively, or as list of fields ((n,m,2)-shaped arrays). If the keyword argument `polar` is set to True, the function expects the field components to be polar.
		kwargs: See pynger.misc.vis_orient for all the possible key-value pairs. 
		
	Returns:
		A PIL.Image.Image showing the input image with each specified field overlaid.
		
	"""
	_vector_visualization(vis_stream_orient, image, *args, **kwargs)
	
def _vector_visualization(fun, image, *args, **kwargs):
	fun_specific_kwargs = ['polar', 'scale_factor']
		
	if len(args)==1:
		if isinstance(args[0], list):
			am = [cart2polar(field, keepDims=False) for field in args[0]]
		else:
			am = [cart2polar(*args, keepDims=False)]
	elif len(args)==2:
		if isinstance(args[0], list) and isinstance(args[1], list):
			if 'polar' in kwargs and kwargs['polar']:
				am = list(zip(*args))
			else:
				am = [cart2polar(x, y, keepDims=False) for x, y in zip(*args)]
		else:
			if 'polar' in kwargs and kwargs['polar']:
				am = [args]
			else:
				am = [cart2polar(*args, keepDims=False)]
	else:
		raise ValueError("Wrong number of positional args")
	
	if isinstance(image, PIL.Image.Image):
		_image = np.array(image)
	else:
		_image = image
		
	if 'scale_factor' in kwargs:
		scale_factor = kwargs['scale_factor']
	else:
		scale_factor = 1
		
	fig = plt.figure(figsize=tuple(scale_factor*s/100 for s in _image.shape), dpi=100)
	plt.imshow(_image, cmap='gray')
	fun_specific_kwargs += ['color'] # prevent drawing with one single color
	if len(am) > 1:
		if 'color' in kwargs and isinstance(kwargs['color'], list) and len(kwargs['color']) == len(am):
			colors = kwargs['color']
		else:
			base_colors = 'rbgymc'
			colors = [base_colors[n % len(base_colors)] for n in range(len(am))]
	else:
		if 'color' in kwargs:
			colors = kwargs['color']
		else:
			colors = 'r'
	new_kwargs = dict(k for k in kwargs.items() if not k[0] in fun_specific_kwargs)
	for field, c in zip(am, colors):
		angle, magnitude = field
		fun(theta=angle, strength=magnitude, color=c, **new_kwargs)
	fig.canvas.draw()
	_data = np.array(fig.canvas.renderer._renderer)
	return PIL.Image.fromarray(_data)