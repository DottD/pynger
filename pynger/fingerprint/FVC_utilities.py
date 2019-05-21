import os
import re
import io
import numpy as np
import PIL.Image
import typing
from pynger.types import Image, Mask, Field
from pynger.fingerprint.tuning_lro import LROEstimator
from pynger.fingerprint.sampling import convert_to_full
from pynger.field.manipulation import polar2cart
from pynger.misc import recursively_scan_dir_gen


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

def loadDataset(path: str, loadGT: bool = True):
    """ Loads the FVC-TEST dataset.

    Args:
        path: Directory with the FVC-TEST dataset.
        loadGT: whether to load the ground truth information or not.

    Return:
        A generator of pairs (X, y) where X has the original image, its mask and its border specifications, and y is the corresponding orientation field ground truth.
    """
    with open(path, 'r') as f:
        _ = int(f.readline())
        for line in f:
            name, step, bd = line.split()
            step = int(step)
            bd = int(bd)
            # Load image
            image_path = os.path.join(os.path.dirname(path), name)
            image = np.array(PIL.Image.open(image_path).convert('L')).astype(float)
            # Load mask
            mask_path = os.path.splitext(image_path)[0]+'.fg'
            mask = MaskProxy().read(mask_path)
            # Set specifications
            specs = [bd, bd, step, step]
            # Adjust image shape
            _mask = convert_to_full(mask, border_x=bd, border_y=bd, step_x=step, step_y=step, mode='constant')
            image = image[:_mask.shape[0], :_mask.shape[1]]
            # Load the ground truth field
            if loadGT:
                field_path = os.path.splitext(image_path)[0]+'.gt'
                lro, _ = FieldProxy().read(field_path, full=False)
                field = polar2cart(lro, 1, retField=True)
                # Serialize input data and append to X and the ground truth information
                yield (LROEstimator.serialize_Xrow(image, mask, specs), LROEstimator.serialize_yrow(field))
            else:
                yield (LROEstimator.serialize_Xrow(image, mask, specs), image_path)

def countDatasetElements(path):
    with open(path, 'r') as f:
        return int(f.readline())

def loadSegmentationDataset(sdir: str, odir: str):
    """ Loads the dataset for segmentation evaluation.

    Args:
        sdir: Path to the segmented images; all the images shall be direct children of this directory.
        odir: Path to the original images; this folder shall contain as direct children the folder of the databases FVC2000, FVC2002, FVC2004 (from DB1a, DB1b, to DB4a, DB4b) - e.g. the main root of the DVD shipped with Handbook of Fingerprint Recognition.

    Note:
        If some DB is not available a warning will be issued, but the other images will be loaded anyway.

    Return:
        A generator of pairs (X, y) where X is the original image, and y the corresponding ground truth segmentation image.
    """
    pattern = re.compile('(FVC\\d+)_(\\w+)_\\w+_(\\d+)_(\\d+)')
    sfiles = recursively_scan_dir_gen(sdir, '.png')
    for sfile in sfiles:
        basename = os.path.basename(sfile)
        match = pattern.match(basename)
        if match:
            ofile = os.path.join(
                odir,
                match[1], # FVCxxxx
                'Dbs',
                # converts DB1 to Db1, them appends an 'a' for the first 100 images, and a 'b' otherwise
                match[2].title() + '_' + ('a' if int(match[3])<=100 else 'b'),
                '{}_{}.tif'.format(match[3], match[4]) # append the filename
                )
            yield (ofile, sfile)
