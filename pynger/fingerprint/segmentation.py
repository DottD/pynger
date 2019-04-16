import os
import numpy as np
from scipy.ndimage.measurements import label as binary_label
from scipy.ndimage.morphology import binary_closing, binary_opening
from pynger.types import Image
from pynger.signal.windows import circleWin
from subprocess import Popen, PIPE
from pathlib import Path
import PIL


def rough_segmentation(image: Image, **kwargs):
    """ Performs a rough segmentation on the image.

    Args:
        image: Image to be segmented.

    Keyword Args:
        bin_level (int): number in the range [0,100] as a percentile used to binarize the image (defaults to 30)
        bin_close_radius (int): radius of binary closing performed on ridges to create a full mask (defaults to 10)
        bin_open_radius (int): radius of binary opening performed to smooth the full mask (defaults to 10)
    
    Return:
        A pair with the image and the foreground mask cropped to the relevant part of the mask, or (None, None) in case of an error.
        
    """
    # Foreground mask
    ridges = image < np.percentile(image, kwargs.get('bin_level', 30))
    struct_el = circleWin(kwargs.get('bin_close_radius', 10))
    mask = binary_closing(ridges, structure=struct_el, iterations=1)
    label, tot = binary_label(mask)
    label_areas = np.array([np.count_nonzero(label==n) for n in range(1,tot)])
    idx = label_areas.argmax() + 1
    mask = (label == idx)
    struct_el = circleWin(kwargs.get('bin_open_radius', 10))
    mask = binary_opening(mask, structure=struct_el, iterations=1)
    # Crop image and mask
    if np.count_nonzero(mask) == 0:
        return None, None
    i,j = mask.nonzero()
    l, r, t, b = (j.min(), j.max(), i.min(), i.max())
    return image[t:b,l:r], mask[t:b,l:r]


def angafis_preprocessing(image: Image, **kwargs):
    """ Apply the preprocessing technique that AnGaFIS-OF uses.

    Args:
        image: The image which must be pre-processed.

    Keyword Args:
        verbose (bool): Whether some information should be returned in stdout (defaults to False)
    """
    # Get the full path to the executable
    exe_path = os.path.join(os.path.dirname(__file__), 'angafis_preProcessing')
    exe_path = kwargs.get('exe_path', exe_path)
    # Save the input image as file
    currdir = str(Path.home())
    imagefile = os.path.join(currdir, 'tmp.png')
    PIL.Image.fromarray(image).convert('L').save(imagefile)
    outdir = os.path.join(currdir, 'angafis_tmp')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Run AnGaFIS preprocessing 
    command = "\"{}\" \"{}\" \"{}\"".format(exe_path, imagefile, outdir)
    with Popen(command, cwd=outdir, shell=True, universal_newlines=True, stdout=PIPE, stderr=PIPE) as proc:
    	err = proc.stderr.read()
    	if err != "":
    		raise RuntimeError(err)
    	if kwargs.get('verbose', False):
    		print(proc.stdout.read())
    # Get the results
    outimage = os.path.join(outdir, 'adjusted.png')
    image = np.array(PIL.Image.open(outimage).convert('L'))
    os.remove(outimage)
    os.remove(imagefile)
    os.removedirs(outdir)
    return image


if __name__ == '__main__':
    path = "/Users/MacD/Documents/Databases/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0003_10.png"
    image = np.array(PIL.Image.open(path).convert('L'))
    enhimage = angafis_preprocessing(image, verbose=True)
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(enhimage)
    plt.show()