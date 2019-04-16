import os
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

class Proxy:
    def write(self, path: str):
        raise NotImplementedError("Derived classes must reimplement this method")

    def read(self, path: str):
        raise NotImplementedError("Derived classes must reimplement this method")

class MaskProxy(Proxy):
    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], np.ndarray):
                self.mask = args[0]
            elif isinstance(args[0], str):
                self.read(args[0])
            else:
                raise TypeError("Arguments not recognized")
        else:
            self.mask = None

    def read(self, path: str, full: bool = True):
        """ Reads the mask, according to FVC-OnGoing specs.

        Args:
            path: The input file path (generally with .fg extension)
            full: Whether the full output should be returned (not implemented yet)

        Return:
            The boolean mask represented in the given file.
        """
        if not os.path.exists(path):
            raise RuntimeError("The input file does not exist")
        with open(path, 'r') as f:
            shape = tuple([int(n) for n in f.readline().split()])
            mask = np.empty(shape, dtype=bool)
            for row_n, line in enumerate(f):
                mask[row_n,:] = [bool(int(n)) for n in line.split()]
        self.mask = mask
        return mask

    def write(self, path: str):
        """ Writes the mask, according to FVC-OnGoing specs.

        Args:
            path: The output file path (generally with .fg extension)
        """
        with open(path, 'w') as f:
            print(self.mask.shape, file=f)
            for line in self.mask.astype(int):
                print(line, file=f)

class FieldProxy(Proxy):
    def __init__(self, *args):
        if len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
            self.angle, self.mask = args[0], args[1]
        elif len(args) == 1 and isinstance(args[0], str):
            self.read(args[0])
        else:
            self.angle, self.mask = None, None

    def read(self, path: str, full: bool = True):
        """ Reads the field, according to FVC-OnGoing specs.

        Args:
            path: The input file path (generally with .gt extension)
            full: Whether the full output should be returned

        Return:
            The field represented in the given file.
        """
        if not os.path.exists(path):
            raise RuntimeError("The input file does not exist")
        with open(path, 'rb') as f:
            # Read and discard the header. To visualize -> print(f.read(8).decode('ascii'))
            f.read(8)
            # Read the field specifications
            get_next_int = lambda: int.from_bytes(f.read(4), byteorder='little', signed=True)
            border_x = get_next_int()
            border_y = get_next_int()
            step_x = get_next_int()
            step_y = get_next_int()
            cols = get_next_int()
            rows = get_next_int()
            # Read the values
            get_next_uint8 = lambda: int.from_bytes(f.read(1), byteorder='little', signed=False)
            content = [(get_next_uint8(), get_next_uint8()) for _ in range(cols*rows)]
            angle, mask = zip(*content)
            angle = np.array(angle, dtype=float).reshape((rows, cols))
            angle *= np.pi / 255.0
            mask = np.array(mask, dtype=bool).reshape((rows, cols))
            # Optionally convert to full matrix
            if full:
                self.angle = convert_to_full(angle, border_x=border_x, border_y=border_y, step_x=step_x, step_y=step_y, mode='constant')
                self.mask = convert_to_full(mask, border_x=border_x, border_y=border_y, step_x=step_x, step_y=step_y, mode='constant')
            else:
                self.angle = angle
                self.mask = mask
            return self.angle, self.mask

    def write(self, path: str, **kwargs):
        """ Writes the field, according to FVC-OnGoing specs.

        Args:
            path: The output file path (generally with .gt extension)

        Keyword Args:
            border_x (int): Horizontal border used to sample the field (defaults to 14)
            border_y (int): Vertical border used to sample the field (defaults to 14)
            step_x (int): Horizontal distance between two conscutive sample points (defaults to 8)
            step_y (int): Vertical distance between two conscutive sample points (defaults to 8)

        Note:
            The field is subsampled in the process. To avoid this behaviour, set border parameters to 0 and step parameters to 1.
        """
        # Read parameters
        bx = kwargs.get('border_x', 14)
        by = kwargs.get('border_y', 14)
        sx = kwargs.get('step_x', 8)
        sy = kwargs.get('step_y', 8)
        # Sample the field
        angle = self.angle[by:-by:sy, bx:-bx:sx]
        mask = self.mask[by:-by:sy, bx:-bx:sx]
        with open(path, 'wb') as f:
            # Read and discard the header. To visualize -> print(f.read(8).decode('ascii'))
            f.write("DIRIMG00".encode('ascii'))
            # Read the field specifications
            put_int = lambda n: int(n).to_bytes(4, byteorder='little', signed=True)
            put_int(bx)
            put_int(by)
            put_int(sx)
            put_int(sy)
            rows, cols = angle.shape
            put_int(cols)
            put_int(rows)
            # Write the values
            angle *= 255.0 / np.pi
            put_uint8 = lambda n: int(n).to_bytes(1, byteorder='little', signed=False)
            for a, m in zip(angle.flatten(), mask.flatten()):
                put_uint8(a)
                put_uint8(m)


if __name__ == '__main__':
    angle, mask = FieldProxy().read("/Users/MacD/Databases/FOESamples/Good/110.gt")

    from PIL import Image
    image = np.array(Image.open("/Users/MacD/Databases/FOESamples/Good/110.bmp").convert('L'), dtype=float)
    image = image[:mask.shape[0], :mask.shape[1]]
    import matplotlib.pyplot as plt
    image[np.logical_not(mask)] /= 3
    plt.imshow(image, cmap='gray')
    from pynger.misc import vis_orient
    vis_orient(angle, color='r', step=10)
    plt.show()

