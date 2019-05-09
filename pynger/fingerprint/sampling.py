import numpy as np
from pynger.types import Image, Mask, Field, Union
from pynger.field.manipulation import double_angle, halve_angle
from scipy import ndimage
from itertools import chain


def convert_to_full(image: Union[Image,Mask], **kwargs) -> Union[Image,Mask]:
    """ Convert a sampled image to its full version.
    
    Args:
        image: Sampled image
    
    Keyword Args:
        border_x (int): Width of the unused horizontal space (on the left)
        border_y (int): Width of the unused vertical space (above)
        step_x (int): Distance between two sampled points along the horizontal axis
        step_y (int): Distance between two sampled points along the vertical axis
        mode (str): Padding mode. See :func:`np.pad` for details. When `mode='constant`, only zero-padding is allowed.
        policy (str): Either 'nist' or 'fvc' (defaults to 'fvc').

    Returns:
        Full version of the input image. May be smaller than the original image; in this case, some elements must be padded on the right and below the image.
    """
    border_x = kwargs.get('border_x')
    border_y = kwargs.get('border_y')
    step_x = kwargs.get('step_x')
    step_y = kwargs.get('step_y')
    mode_kwargs = {
        'mode': kwargs.get('mode', 'constant'),
    }
    if mode_kwargs['mode'] == 'constant':
        mode_kwargs.update({
            'constant_values': (0,0)
        })
    
    _image = np.repeat(image, step_x, axis=1)
    _image = np.repeat(_image, step_y, axis=0)

    policy = kwargs.get('policy', 'fvc')
    if policy == 'fvc':
        _image = np.pad(_image, (
            (border_y-step_y//2, border_y-step_y//2),
            (border_x-step_x//2, border_x-step_x//2)
            ), **mode_kwargs)
    elif policy == 'nist':
        _image = np.pad(_image, (
            (border_y, border_y),
            (border_x, border_x)
            ), **mode_kwargs)
    else:
        raise NotImplementedError('Policy not recognized')

    return _image.astype(image.dtype)

def subsample(mat: Union[Image,Mask,Field], **kwargs) -> Union[Image,Mask,Field]:
    # Read parameters
    bx = kwargs.get('border_x', 14)
    by = kwargs.get('border_y', 14)
    sx = kwargs.get('step_x', 8)
    sy = kwargs.get('step_y', 8)
    is_field = kwargs.get('is_field', False)
    smooth = kwargs.get('smooth', False)
    # Averaging
    initial_type = mat.dtype
    initial_dims = len(mat.shape)
    ri = range(by, mat.shape[0]-by+1, sy)
    rj = range(bx, mat.shape[1]-bx+1, sx)
    sampling_criterion = lambda M: np.mean(M.ravel())
    policy = kwargs.get('policy', 'fvc')
    if policy == 'fvc':
        sampling_fun = lambda M: _apply_blockwise(sampling_criterion, M, ri, rj, (sy, sx), 'center_coords')
    elif policy == 'nist':
        if ri[-1] < mat.shape[0]-by:
            ri = list(ri) + [mat.shape[0]-by-sy]
        if rj[-1] < mat.shape[1]-bx:
            rj = list(rj) + [mat.shape[1]-bx-sx]
        sampling_fun = lambda M: _apply_blockwise(sampling_criterion, M, ri, rj, [], 'corner_coords')
    else:
        raise NotImplementedError('Policy not recognized')
        
    if is_field:
        preprocess = lambda M: double_angle(M)
        postprocess = lambda M: halve_angle(M)
    else:
        preprocess = lambda M: M
        postprocess = lambda M: M

    mat = preprocess(mat)
    if initial_dims > 2:
        mat = np.dsplit(mat, mat.shape[2])
    else:
        mat = [mat]
    mat = [M.squeeze() for M in mat]
    if smooth:
        mat = [ndimage.gaussian_filter(M, (sy//2, sx//2)) for M in mat]
    mat = [sampling_fun(M) for M in mat]
    if initial_dims > 2:
        mat = np.dstack(mat)
    else:
        mat = mat[0]
    mat = postprocess(mat)

    return mat.astype(initial_type)

def _apply_blockwise(fun, M, i, j, width, mode):
    if mode == 'center_coords':
        s = [w//2 for w in width]
        return np.array([
                [fun(M[ci-s[0]:ci+s[0], cj-s[1]:cj+s[1]]) for cj in j]
                for ci in i
            ])
    elif mode == 'corner_coords':
        return np.array([
                [fun(M[t:b, l:r]) for l, r in zip(j[:-1], j[1:])]
                for t, b in zip(i[:-1],i[1:])
            ])
    else:
        raise NotImplementedError('Mode not recognized')