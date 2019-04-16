import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.measurements import label, center_of_mass
from pynger.types import Mask


def distance_erosion(mask: Mask, threshold: float) -> Mask:
    """ Erodes the mask, based on the distance transform.

    The distance transform of the input mask is computed, then the given threshold is applied to shrink the mask.

    Args:
        mask: Mask to be shrinked
        threshold: Minimum distance from background allowed for a pixel to survive

    Return:
        The shrinked mask.
    """
    return distance_transform_edt(mask) > threshold

def distance_dilation(mask: Mask, threshold: float) -> Mask:
    """ Dilates the mask, based on the distance transform.

    The input mask is first inverted, then its distance transform is computed and the threshold is applied to shrink the mask. Finally an inversion is performed.

    Args:
        mask: Mask to be dilated
        threshold: Minimum distance from foreground allowed for a background pixel to keep its status

    Return:
        The dilated mask.
    """
    return np.logical_not(distance_transform_edt(np.logical_not(mask)) > threshold)

def gaussian_dilation(mask: Mask, radius: int) -> Mask:
    """ Smoothly dilates the mask.

    The input mask is first blurred, with a gaussian filter, then binarized through global thresholding.

    Args:
        mask: Mask to be dilated
        radius: Radius of the gaussian window used to perform the dilation

    Return:
        The dilated mask.
    """
    fuzzy_mask = gaussian_filter(mask, radius/4.0)
    return fuzzy_mask > 0.05

def gaussian_erosion(mask: Mask, radius: int) -> Mask:
    """ Smoothly erodes the mask.

    The input mask is first blurred, with a gaussian filter, then binarized through global thresholding.

    Args:
        mask: Mask to be eroded
        radius: Radius of the gaussian window used to perform the erosion

    Return:
        The eroded mask.
    """
    fuzzy_mask = gaussian_filter(mask, radius/4.0)
    return fuzzy_mask > 0.95

def ccomp_select_cmass(mask: Mask, threshold: float) -> Mask:
    """ Selects the connected components of the mask that have maximum distance from the center of mass greater than the threshold.
    
    Args:
        mask: The mask to process
        threshold: Threshold value used to select the components
    
    Returns:
        A mask with only the selected connected components.
    """
    out_mask = np.ones(mask.shape, dtype=bool)
    labels, num_labels = label(mask)
    for n in range(1, num_labels+1):
        ccomp = (labels == n)
        cmass = center_of_mass(ccomp)
        coord = np.transpose(np.nonzero(ccomp)).astype(float)
        coord -= np.array(cmass)
        accepted = np.max(coord[:,0]**2 + coord[:,1]**2) >= threshold**2
        if accepted:
            out_mask = np.logical_or(out_mask, ccomp)
    return out_mask