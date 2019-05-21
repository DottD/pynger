from warnings import warn
import numpy as np
from scipy.ndimage import gaussian_filter
from pynger.fingerprint.operators import drifter_mask, smoother, adjuster
from pynger.fingerprint.orientation import LRO, LRF
from pynger.field.manipulation import cart2polar, polar2cart, normalize, magnitude, double_angle, halve_angle
from pynger.string.manipulation import remove_prefix
from pynger.types import Image, Mask, Field
from pynger.mask.manipulation import distance_erosion, gaussian_dilation, gaussian_erosion, ccomp_select_cmass
import re


def _filter_kwargs(kwargs, prefix):
    kwargs = dict(filter(lambda s: s[0].startswith(prefix), kwargs.items()))
    kwargs = dict(zip(map(lambda s: remove_prefix(s, prefix), kwargs.keys()), kwargs.values()))
    return kwargs

def _input_sanity_check(kwargs, fun):
    """ Check input keyword arguments and return the fixed arguments.

    Todo: 
        Boundary check should be implemented.
    """
    kwargs_classes = _get_all_kwargs(fun)
    for key in kwargs:
        if key in kwargs_classes:
            kwargs[key] = kwargs_classes[key](kwargs[key])
    return kwargs

def _get_all_kwargs(fun):
    """ Get all keyword arguments found in the function docstring.

    Note:
        Docstring must be formatted according to Google style.
    """
    doc = fun.__doc__
    list_start = 'Keyword Args:\n'
    index = doc.find(list_start) + len(list_start)
    doc = doc[index:]
    pattern = re.compile('(\\w+) \\((\\w+)\\):')
    kwargs = {val[0]: eval(val[1]) for val in re.findall(pattern, doc)}
    return kwargs

def reliable_iterative_smoothing(image: Image, mask: Mask, field: Field, **kwargs) -> Field:
    """ Gives the field obtained by applying the iterative smoothing over a mask reliably covering loops and deltas.

    Args:
        image (numpy.array): Image with the fingerprint to process
        mask (numpy.array or None): Mask with relevant information
        field (numpy.array): Field that will be refined (non-doubled-phase)

    Keyword Args:
		LRF_min_disk_size (int): size of the disk neighborhood of each point for the reliability check (defaults to 10)
		LRF_rel_check_grid_step (int): step of the grid used for the reliability check (defaults to 10)
		LRF_rel_check_threshold (float): threshold value used for the reliability check (expressed as a percentile of the input rel matrix) (defaults to 30)
		LRF_segment_n_points (int): number of points for each segment (defaults to 15)
		LRF_segment_length (int): length of each segment (in pixels) (defaults to 30)
		LRF_gaussian_smooth_std (float): standard deviation of gaussian filter used to smooth the signal on each segment (with respect to segment's length) (defaults to 0.1)

		LRO1_scale_factor (float): radius used for the first LRO computation, relative to the average ridge distance (ARD) (defaults to 1.5)
		LRO1_number_angles (int): number of angles to be tested (defaults to 36)
		LRO1_along_sigma_ratio (float): width of filter along the orientation, relative to ``ARD * LRO1_scale_factor`` (defaults to 0.85)
		LRO1_ortho_sigma (float): width of filter orthogonal to the orientation (absolute) (defaults to 1.0)

        SM1_radius_factor (float): Radius of smoother's integration path, relative to ARD (defaults to 1)
        SM1_sample_dist (float): Distance in pixels between two consecutive samples of the integration path (defaults to 1)
        SM1_relax (float): Smoother's relaxation parameter (defaults to 1)

        DM1_radius_factor (float): Radius of drifters' integration path, relative to ARD (defaults to 1)
        DM1_sample_dist (float): Distance in pixels between two consecutive samples of drifters' integration path (defaults to 1)
        DM1_drift_threshold (float): Threshold value used to derive a mask without deltas (defaults to 0.3)
        DM1_shrink_rad_factor (float): Radius of the erosion, relative to ARD (defaults to 3.5)
        DM1_blur_stddev (float): Standard deviation of the gaussian window for blurring, relative to ARD (defaults to 0.5)

		LRO2_scale_factor (float): radius used for the second LRO computation, relative to the average ridge distance (ARD) (defaults to 0.5)
		LRO2_number_angles (int): number of angles to be tested (defaults to 36)
		LRO2_along_sigma_ratio (float): width of filter along the orientation, relative to ``ARD * LRO2_scale_factor`` (defaults to 0.85)
		LRO2_ortho_sigma (float): width of filter orthogonal to the orientation (absolute) (defaults to 1.0)

        SM2_radius_factor (float): Radius of adjusters' integration path, relative to ARD (defaults to 0.7)
        SM2_sample_dist (float): Distance in pixels between two consecutive samples of adjusters' integration path (defaults to 1)
        SM2_relax (float): Adjusters' relaxation parameter that will scale the fuzzy mask with no deltas (defaults to 0.9)

        DMASK2_blur_stddev (float): Standard deviation of gaussian blurring for the difference mask, relative to ARD (defaults to 0.5)
        DMASK2_threshold (float): Binarization level for the difference mask (defaults to 0.3)
        DMASK2_ccomp_ext_thres (float): Minimum extent required for a connected component to survive, relative to ARD (defaults to 1)
        DMASK2_dil_rad_factor (float): Gaussian dilation radius for the difference mask, relative to ARD (defaults to 3)

        DM3_radius_factor (float): Radius of drifters' integration path, relative to ARD (defaults to 1)
        DM3_sample_dist (float): Distance in pixels between two consecutive samples of drifters' integration path (defaults to 1)
        DM3_drift_threshold (float): Threshold value used to derive a mask with both loops and deltas (defaults to 0.1)
        DM3_shrink_rad_factor (float): Radius of the erosion, relative to ARD (defaults to 3.5)
        DM3_ccomp_ext_thres (float): Minimum extent required for a connected component to survive, relative to ARD (defaults to 0.75)
        DM3_dil_rad_factor (float): Gaussian dilation radius, relative to ARD (defaults to 2)

        IS_max_iterations (int): Maximum number of iterations, where a negative number stands for infinity (defaults to 10)
        IS_binlev_step (float): Constant to be added to DMASK_bin_level at each iteration (defaults to 0.01)
        IS_SMOOTH_radius_factor (float): Radius of the smoother, relative to ARD (defaults to 1.5)
        IS_SMOOTH_sample_dist (float): Distance in pixels between two consecutive samples of the integration path (defaults to 1.0)
        IS_SMOOTH_relaxation (float): Relaxation parameter for the smoother operator (defaults to 1.0)
        IS_DMASK_bin_level (float): Binarization level for the difference mask (defaults to 0.1)
        IS_GMASK_erode_rad_factor (float): Radius of global mask erosion, relative to ARD (defaults to 0.1)
        IS_GMASK_dilate_rad_factor (float): Radius of global mask erosion, relative to ARD (defaults to 2.0)
        IS_GMASK_blur_stddev (float): Std. dev. of global mask gaussian blurring (defaults to 0.5)

    Return:
        The enhanced field.

    Note:
        Each keyword is preceded by an identifier, that refers to the part of the function that will use such an argument; the identifier is separated from the argument name by a `__` (double underscore). For instance, every keyword prefixed by `LRF__` will be used to tune the Local Ridge Frequency estimator. 
        
        The whole algorithm is divided into three steps, after the very first operations, i.e. the ``LRF__`` step. The number present in some identifiers, such as ``DM1__``, refers to such steps.
    """
    # kwargs = _input_sanity_check(kwargs, reliable_iterative_smoothing)

    # Estimate the average ridge distance (period)
    local_kwargs = _filter_kwargs(kwargs, 'LRF_')
    if local_kwargs:
        period = LRF(image, *cart2polar(field, keepDims=False), **local_kwargs)
    else:
        period = LRF(image, *cart2polar(field, keepDims=False))
    
    if period is None or period < 3:
        warn("Skipped iterative smoothing", RuntimeWarning)
        return field

    # Check the mask parameter
    if mask is None:
        mask = np.ones(image.shape, dtype=bool)
    else:
        mask = np.broadcast_to(mask, image.shape)

    # --------
    # 1st part
    # --------

    # Compute the orientation field with the first radius
    local_kwargs = {
        'ridge_dist': kwargs.get('LRO1_scale_factor', 1.5) * period,
        'number_angles': kwargs.get('LRO1_number_angles', 36),
        'along_sigma_ratio': kwargs.get('LRO1_along_sigma_ratio', 0.85),
        'ortho_sigma': kwargs.get('LRO1_ortho_sigma', 1.0),
        'mask': mask,
    }
    field1 = polar2cart(*LRO(image, **local_kwargs), retField=True)
    # Double the phase angle to comply with operators requirements
    field1 = double_angle(field1)

    # Perform the smoothing of the original field
    local_kwargs = {
        'radius': kwargs.get('SM1_radius_factor', 1) * period,
        'sample_dist': kwargs.get('SM1_sample_dist', 1),
        'relax': kwargs.get('SM1_relax', 1),
    }
    sfield1 = smoother(field1, **local_kwargs)

    # Through the drifter operator, get a mask that hides deltas
    local_kwargs = {
        'radius': kwargs.get('DM1_radius_factor', 1) * period,
        'sample_dist': kwargs.get('DM1_sample_dist', 1),
        'threshold': kwargs.get('DM1_drift_threshold', 0.3),
    }
    deltas_mask = drifter_mask(sfield1, markLoops=False, markDeltas=True, **local_kwargs)
    no_deltas_mask = np.logical_not(deltas_mask)

    # Erosion and gaussian blurring are applied to the mask.
    # This way we are ensured that deltas are fully excluded from the mask.
    no_deltas_mask = np.logical_and(no_deltas_mask, mask)
    no_deltas_mask = distance_erosion(no_deltas_mask, kwargs.get('DM1_shrink_rad_factor', 3.5) * period)
    no_deltas_fuzzy = gaussian_filter(no_deltas_mask, kwargs.get('DM1_blur_stddev', 0.5 * period))

    # --------
    # 2nd part
    # --------

    # Compute the orientation field with the second radius
    local_kwargs = {
        'ridge_dist': kwargs.get('LRO2_scale_factor', 0.5) * period,
        'number_angles': kwargs.get('LRO2_number_angles', 36),
        'along_sigma_ratio': kwargs.get('LRO2_along_sigma_ratio', 0.85),
        'ortho_sigma': kwargs.get('LRO2_ortho_sigma', 1.0),
        'mask': mask,
    }
    field2 = polar2cart(*LRO(image, **local_kwargs), retField=True)
    # Double the phase angle to comply with operators requirements
    field2 = double_angle(field2)

    # Perform the adjusting of the two fields
    local_kwargs = {
        'radius': kwargs.get('SM2_radius_factor', 0.7) * period,
        'sample_dist': kwargs.get('SM2_sample_dist', 1),
        'relax': kwargs.get('SM2_relax', 0.9) * no_deltas_fuzzy,
    }
    afield1 = adjuster(field1, **local_kwargs)
    afield2 = adjuster(field2, **local_kwargs)

    # Compute the fuzzy difference mask and intersect with the initial one, then blur the mask and apply a threshold to convert it into a binary mask.
    fuzzy_diff = magnitude(normalize(afield1) - normalize(afield2), keepDims=False) / 2
    fuzzy_diff[np.logical_not(mask)] = 0.0
    fuzzy_diff = gaussian_filter(fuzzy_diff, kwargs.get('DMASK2_blur_stddev', 0.5 * period))
    binary_diff = fuzzy_diff > kwargs.get('DMASK2_threshold', 0.3)

    # Suppress all the components with too low distance from their center of mass, then perform a dilation
    binary_diff = ccomp_select_cmass(binary_diff, kwargs.get('DMASK2_ccomp_ext_thres', 1) * period)
    binary_diff = gaussian_dilation(binary_diff, kwargs.get('DMASK2_dil_rad_factor', 3) * period)

    # --------
    # 3rd part
    # --------

    # Through the drifters, compute a mask with loops and deltas, and intersect it with the contraction of the initial mask
    local_kwargs = {
        'radius': kwargs.get('DM3_radius_factor', 1) * period,
        'sample_dist': kwargs.get('DM3_sample_dist', 1),
        'threshold': kwargs.get('DM3_drift_threshold', 0.1),
    }
    ldmask = drifter_mask(afield1, markLoops=True, markDeltas=True, **local_kwargs)
    ldmask = np.logical_and(ldmask, distance_erosion(mask, kwargs.get('DM3_shrink_rad_factor', 3.5) * period))

    # Suppress all the components with too low distance from their center of mass, then perform a dilation
    ldmask = ccomp_select_cmass(ldmask, kwargs.get('DM3_ccomp_ext_thres', 0.75) * period)

    # Remove the elements present in the previous difference mask
    ldmask = np.logical_and(ldmask, np.logical_not(binary_diff))
    ldmask = gaussian_dilation(ldmask, kwargs.get('DM3_dil_rad_factor', 2) * period)
    ldmask = np.logical_not(ldmask)

    # Iterative smoothing
    local_kwargs = _filter_kwargs(kwargs, 'IS_')
    if local_kwargs:
        is_field = iterative_smoothing(afield1, ldmask, period, **local_kwargs)
    else:
        is_field = iterative_smoothing(afield1, ldmask, period)

    # Halves the phase angle, to comply with the input format
    is_field = halve_angle(is_field)

    return is_field

def iterative_smoothing(field: Field, mask: Mask, period: float, **kwargs) -> Field:
    """ Iteratively enhances the input field through the smoother operator.

    Args:
        field: Field (doubled-phase) to be enhanced, shape ``(:,:,2)``
        mask: 
        period: Fingerprint's spatial period, i.e. the average ridge distance
    
    Keyword Arguments:
        max_iterations (int): Maximum number of iterations, where a negative number stands for infinity (defaults to 10)
        binlev_step (float): Constant to be added to DMASK_bin_level at each iteration (defaults to 0.01)
        SMOOTH_radius_factor (float): Radius of the smoother, relative to ARD (defaults to 1.5)
        SMOOTH_sample_dist (float): Distance in pixels between two consecutive samples of the integration path (defaults to 1.0)
        SMOOTH_relaxation (float): Relaxation parameter for the smoother operator (defaults to 1.0)
        DMASK_bin_level (float): Binarization level for the difference mask (defaults to 0.1)
        GMASK_erode_rad_factor (float): Radius of global mask erosion, relative to ARD (defaults to 0.1)
        GMASK_dilate_rad_factor (float): Radius of global mask erosion, relative to ARD (defaults to 2.0)
        GMASK_blur_stddev (float): Std. dev. of global mask gaussian blurring (defaults to 0.5)

    Return: 
        The enhanced field.
    """
    _mask = mask.copy() # this will be referred to as the global mask
    _field = field.copy()
    step = 0
    max_iter = kwargs.get('max_iterations', 10)
    bin_lev = kwargs.get('DMASK_bin_level', 0.1)
    while np.count_nonzero(_mask) > 0 and step != max_iter:
        # Perform the smoothing of the field
        local_kwargs = {
            'radius': kwargs.get('SMOOTH_radius_factor', 1.5) * period,
            'sample_dist': kwargs.get('SMOOTH_sample_dist', 1.0),
            'relax': kwargs.get('SMOOTH_relaxation', 1.0),
        }
        _loc_field = smoother(_field, **local_kwargs)
        # Compute the difference mask (also called the local mask)
        fuzzy_diff = magnitude(normalize(_loc_field) - normalize(_field), keepDims=False) / 2
        binary_diff = fuzzy_diff > bin_lev
        # Erode the global mask to ensure convergence
        _mask = gaussian_erosion(_mask, kwargs.get('GMASK_erode_rad_factor', 0.1) * period)
        # Combine it with the local mask
        _mask = np.logical_and(_mask, binary_diff)
        # Dilate and blur the global mask
        fuzzy_mask = gaussian_dilation(_mask, kwargs.get('GMASK_dilate_rad_factor', 2.0) * period)
        fuzzy_mask = gaussian_filter(fuzzy_mask, kwargs.get('GMASK_blur_stddev', 0.5 * period))
        _field = _loc_field * fuzzy_mask[:,:,None] + _field * (1 - fuzzy_mask)[:,:,None]
        # Increase the difference mask binarization level, to ensure convergence
        bin_lev += kwargs.get('binlev_step', 0.01)
        # Update step
        step += 1

    return _field


if __name__ == '__main__':
    import PIL
    from pynger.fingerprint.segmentation import rough_segmentation
    path = "/Users/MacD/Documents/Unicam/Dottorato/pytest/test_images/f0001_01.bmp"
    image = np.array(PIL.Image.open(path).convert('L'))
    image, mask = rough_segmentation(image)
    lro, _ = LRO(image)
    field = polar2cart(lro, 1, retField=True)
    x = np.random.randn(44)
    args = dict(zip(_get_all_kwargs(reliable_iterative_smoothing).keys(), x))
    field2 = reliable_iterative_smoothing(image, mask, field, **args)