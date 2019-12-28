import os
import re
import io
import numpy as np
import PIL.Image
import typing
from pynger.types import Image, Mask, Field
from pynger.fingerprint.tuning_lro import LROEstimator
from pynger.fingerprint.sampling import convert_to_full, subsample
from pynger.field.manipulation import polar2cart
from pynger.misc import recursively_scan_dir_gen, recursively_scan_dir, random_combination
from itertools import combinations, starmap


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
            self.angle, self.mask = args[0].copy(), args[1].copy()
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
            # Read the field specifications
            get_next_int = lambda: int.from_bytes(f.read(4), byteorder='little', signed=True)
            self.border_x = get_next_int()
            self.border_y = get_next_int()
            self.step_x = get_next_int()
            self.step_y = get_next_int()
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
                self.angle = convert_to_full(angle, border_x=self.border_x, border_y=self.border_y, step_x=self.step_x, step_y=self.step_y, mode='constant')
                self.mask = convert_to_full(mask, border_x=self.border_x, border_y=self.border_y, step_x=self.step_x, step_y=self.step_y, mode='constant')
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
            subsample (bool): Whether the input shall be sub-sampled before saving it

        Note:
            The field is subsampled in the process. To avoid this behaviour, set border parameters to 0 and step parameters to 1.
        """
        # Read parameters
        bx = kwargs.get('border_x', 14)
        by = kwargs.get('border_y', 14)
        sx = kwargs.get('step_x', 8)
        sy = kwargs.get('step_y', 8)
        needSubsample = kwargs.pop('subsample', True)
        # Sample the field
        if self.angle.shape != self.mask.shape:
            raise RuntimeError('angle and mask sizes mismatch')
        if needSubsample:
            angle = subsample(self.angle, is_field=False, smooth=False, **kwargs)
            mask = subsample(self.mask, is_field=False, smooth=False, **kwargs)
        else:
            angle = self.angle
            mask = self.mask
        with open(path, 'wb') as f:
            f.write("DIRIMG00".encode('ascii'))
            # Read the field specifications
            put_int = lambda n: f.write(int(n).to_bytes(4, byteorder='little', signed=True))
            put_int(bx)
            put_int(by)
            put_int(sx)
            put_int(sy)
            rows, cols = angle.shape
            put_int(cols)
            put_int(rows)
            # Values conversion
            angle *= 255.0 / np.pi
            angle = angle.astype(int)
            mask = mask.astype(int)
            mask *= int(255 / mask.max())
            # Write the values
            put_uint8 = lambda n: f.write(int(n).to_bytes(1, byteorder='little', signed=False))
            for a, m in zip(angle.ravel(), mask.ravel()):
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
            # Load image
            image_path = os.path.join(os.path.dirname(path), name)
            image = np.array(PIL.Image.open(image_path).convert('L')).astype(float)
            # Load mask
            mask_path = os.path.splitext(image_path)[0]+'.fg'
            mask = MaskProxy().read(mask_path)
            # Set specifications
            specs = [bd, bd, step, step]
            # Adjust image shape
            _mask = convert_to_full(mask, border_x=bd, border_y=bd, step_x=step, step_y=step, mode='constant')
            image = image[:_mask.shape[0], :_mask.shape[1]]
            # Load the ground truth field
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
                match[1], # FVCxxxx
                'Dbs',
                # converts DB1 to Db1, them appends an 'a' for the first 100 images, and a 'b' otherwise
                match[2].title() + '_' + ('a' if int(match[3])<=100 else 'b'),
                '{}_{}.tif'.format(match[3], match[4]) # append the filename
                )
            yield (ofile, sfile)

def loadMatchingDatasetFVC(path: str):
    """ Loads the FVC-TEST dataset.

    Args:
        path: Directory with the FVC-TEST dataset.

    Return:
        A dictionary whose keys are pairs of:
        - tuples containing a reference to the database and competition where the images belong, and values are lists of pairs (X, y) where X has the pair of image filenames, and y is the corresponding ground truth label, i.e. a 0 for reject or 1 for accept;
        - the list of all images found in the given folder.
    """
    _, all_image_files = recursively_scan_dir(path, '.tif')

    _, index_files = recursively_scan_dir(path, '.MFA')
    comp_pattern = re.compile('(FVC\\d+)')
    
    competitions = {}
    # Loop over the four possible databases
    for db_n in range(1, 5):
        for MFA in index_files:
            # Get index for false matches
            MFR = MFA[:-1]+'R'
            # Retrieve competition
            match = comp_pattern.search(MFA)
            if match:
                competition = match[1]
            else:
                competition = 'NULL'
            # Retrieve database type (a or b)
            db_type = MFA[-5].lower()
            # Create a new key for this competition
            comp_key = (competition, db_n, db_type)
            competitions[comp_key] = []
            # Generate database name
            db_name = 'Db{}_{}'.format(db_n, db_type)
            # Take the subset of images related to this dataset
            image_files = [name for name in all_image_files if os.path.basename(os.path.dirname(name)) == db_name]
            # Load all the pairs that will be matched
            challenge_pairs = []
            for ifile, gt in zip([MFA, MFR], [0, 1]):
                dir_ = os.path.dirname(ifile)
                with open(ifile, 'r') as file_:
                    for line in file_:
                        file1, file2 = line.split()
                        path1 = os.path.join(dir_, db_name, file1)
                        path2 = os.path.join(dir_, db_name, file2)
                        challenge_pairs.append( ((path1, path2), gt) )
            # Update the competition dictionary
            competitions[comp_key] = (challenge_pairs, image_files)
    return competitions

def loadMatchingDatasetNIST(path: str, ratio: float = 2.0, verbose: bool = True):
    """ Load NIST SD04 for matching.
    
    Args: 
        path: Path to the folder containing the images.
        ratio: Ratio between the number of impostor and genuine matches.
        verbose: whether to print some basic information about the dataset.

    Return:
        A tuple (X, y, lenX) where X yields pairs of images, y generates 0 for a non-match and 1 for a match, lenX is the total number of elements.
    """
    # Load all images
    _, image_files = recursively_scan_dir(path, ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])
    # Split between first and second impression
    f_image_files = list(filter(lambda s: os.path.basename(s)[0]=='f', image_files))

    # Collect the genuine matches
    challenge_pairs = []
    for ffile in f_image_files:
        basename = os.path.basename(ffile)
        basename = 's'+basename[1:]
        sfile = os.path.join( os.path.dirname(ffile), basename )
        challenge_pairs.append( ((ffile, sfile), 1) )

    # Get the total number of impostor and genuine matches
    genuine_matches = len(challenge_pairs)
    impostor_matches = int(genuine_matches * ratio)
    total_matches = genuine_matches + impostor_matches
    if verbose:
        print('{} genuine matches and {} impostor matches will be selected'.format(genuine_matches, impostor_matches))

    # Collect the impostor matches:
    while True:
        pair = random_combination(image_files, 2)
        left_bname = os.path.basename(pair[0])
        right_bname = os.path.basename(pair[1])
        if left_bname[1:] == right_bname[1:]:
            continue # genuine or the same image
        else:
            challenge_pairs.append( (pair, 0) )
        if len(challenge_pairs) >= total_matches:
            break

    competitions = {
        ('NIST', 'SD04', '_'): (challenge_pairs, image_files)
    }
    return competitions
