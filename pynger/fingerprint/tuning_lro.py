import pickle
import time
import datetime
import itertools

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from pynger.field.manipulation import angle, angle_diff, cart2polar, polar2cart
from pynger.fingerprint.sampling import convert_to_full, subsample
from pynger.fingerprint.orientation import LRO
from pynger.fingerprint.refinement import reliable_iterative_smoothing
from pynger.types import Field, Image, List, Mask, Union
from pynger.fingerprint.nbis import compute_lro as nbis_lro
from pynger.fingerprint.nbis_wrapper import nbis_angle2idx, nbis_idx2angle, nbis_bozorth3
from pynger.fingerprint.tuning_segmentation import AnGaFIS_Seg_Estimator

from joblib import Parallel, delayed


class LROEstimator(BaseEstimator, RegressorMixin):

    @classmethod
    def serialize_Xrow(cls, image: Image, mask: Mask, specs: List[int]):
        # specs is [bx, by, sx, sy]
        # return pickle.dumps((image, mask, specs))
        return (image, mask, specs)

    @classmethod
    def deserialize_Xrow(cls, byte_repr):
        # return pickle.loads(byte_repr)
        return byte_repr

    @classmethod
    def serialize_yrow(cls, field: Field):
        # return pickle.dumps(field)
        return (field,)

    @classmethod
    def deserialize_yrow(cls, byte_repr):
        # return pickle.loads(byte_repr)
        return byte_repr[0]

    
    def check_X_y(self, X, y):
        if len(X) != len(y):
            raise ValueError("Number of images mismatch")

        for n, lineX, liney in zip(range(len(X)), X, y):
            image, mask, specs = LROEstimator.deserialize_Xrow(lineX)
            gt = LROEstimator.deserialize_yrow(liney)
            bx, by, sx, sy = specs
            # Check parameter consistency
            required_y = (image.shape[0] - 2*by) // sy + 1
            if required_y != gt.shape[0] or required_y != mask.shape[0]:
                raise ValueError("At {}. The condition (rows - 2 by) // sy + 1 = sampled_points is not satisfied: ({} - 2*{}) // {} + 1 = {} ≠ {}(gt) or {}(mask)".format(n, image.shape[0], by, sy, required_y, gt.shape[0], mask.shape[0]))
            required_x = (image.shape[1] - 2*bx) // sx + 1
            if required_x != gt.shape[1] or required_x != mask.shape[1]:
                raise ValueError("At {}. The condition (cols - 2 bx) // sx + 1 = sampled_points is not satisfied: ({} - 2*{}) // {} + 1 = {} ≠ {}(gt) or {}(mask)".format(n, image.shape[1], bx, sx, required_x, gt.shape[1], mask.shape[1]))
        return X, y

    
    def check_X(self, X):
        for n, line in enumerate(X):
            image, mask, specs = LROEstimator.deserialize_Xrow(line)
            bx, by, sx, sy = specs
            # Check parameter consistency
            required_y = (image.shape[0] - 2*by) // sy + 1
            if required_y != mask.shape[0]:
                raise ValueError("At {}. The condition (rows - 2 by) // sy + 1 = sampled_points is not satisfied: ({} - 2*{}) // {} + 1 = {} ≠ {}".format(n, image.shape[0], by, sy, required_y, mask.shape[0]))
            required_x = (image.shape[1] - 2*bx) // sx + 1
            if required_x != mask.shape[1]:
                raise ValueError("At {}. The condition (cols - 2 bx) // sx + 1 = sampled_points is not satisfied: ({} - 2*{}) // {} + 1 = {} ≠ {}".format(n, image.shape[1], bx, sx, required_x, mask.shape[1]))
        return X

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        # Input validation
        # X = self.check_X(X)

        # Make prediction
        for lineX in X:
            image, mask, specs = LROEstimator.deserialize_Xrow(lineX)
            bd_specs = {
                'border_x': specs[0],
                'border_y': specs[1],
                'step_x': specs[2],
                'step_y': specs[3],
            }
            field = self.compute_of(image, mask, **bd_specs)
            yield LROEstimator.serialize_yrow(field)
    
    def score(self, X, y=None):
        if y is None:
            raise ValueError("A true y must be given")
        # Check that X and y have correct shape
        # X, y = self.check_X_y(X, y)
        # Split the iterator
        X1, X2 = itertools.tee(X, 2)
        # Get the predicted y
        pred_y = self.predict(X1)
        # Accumulate the average error
        avgerr = []
        for py, ty, x in zip(pred_y, y, X2):
            # Load predicted and ground truth fields
            field = LROEstimator.deserialize_yrow(py)
            gfield = LROEstimator.deserialize_yrow(ty)
            _, mask, _ = LROEstimator.deserialize_Xrow(x)
            # Compute the error
            loc_avgerr = self.compute_error(field, gfield, mask)
            # Append the results to avgerr accumulator
            avgerr.append(loc_avgerr)
        # Compute the mean average error, if possible
        if len(avgerr) > 0:
            return -np.array(avgerr).mean()
        else:
            return np.nan

    def train_on_data(self, X, y):
        """ Trains the estimator on data.
        
        Important:
            Derived class must reimplement this method.

        Hint:
            Take a look at the :func:`check_X_y` to check the consistency of input arguemnts.

        Todo:
            This function should take an image, a mask and ground truth field. The fitting logic should go in the parent class. 

        Example:
            As an example, see :class:`.AnGaFIS_OF_Estimator`.
        """
        raise NotImplementedError("Derived class must reimplement this method")

    def compute_of(self, image, mask, **bd_specs):
        """ Computes the orientation field.
        
        Important:
            Derived class must reimplement this method.

        Example:
            As an example, see :class:`.AnGaFIS_OF_Estimator`.
        """
        raise NotImplementedError("Derived class must reimplement this method")

    def compute_error(self, field1: Field, field2: Field, mask: Mask):
        """ Computes the error between field1 and field2.

        Args:
            field1: First input field
            field2: Second input field
            mask: Foreground mask
        
        Important:
            Derived class must reimplement this method.

        Example:
            As an example, see :class:`.ScoreAngleDiffRMSD`.
        """
        raise NotImplementedError("Derived class must reimplement this method")

class ScoreAngleDiffRMSD:
    
    def compute_error(self, field1: Field, field2: Field, mask: Mask):
        """ Computes the RMSD of the angle differences between field1 and field2. """
        # Difference matrix
        diff = angle_diff(angle(field1, keepDims=False), angle(field2, keepDims=False))
        # Compute Root Mean Square Deviation (RMSD) on foreground mask
        diff = np.rad2deg(diff) ** 2
        diff[np.logical_not(mask)] = 0.0
        return np.sqrt(diff.sum() / np.count_nonzero(mask))

class AnGaFIS_OF_Estimator(ScoreAngleDiffRMSD, LROEstimator):
    def __init__(self, ridge_dist: int = 10, number_angles: int = 36, along_sigma_ratio: float = 0.85, ortho_sigma: float = 1.0):
        """ Initializes and stores all the algorithm's parameters. """
        self.ridge_dist = ridge_dist
        self.number_angles = number_angles
        self.along_sigma_ratio = along_sigma_ratio
        self.ortho_sigma = ortho_sigma
        self.segmentor = AnGaFIS_Seg_Estimator(enhanceOnly=True)

    def train_on_data(self, X, y):
        """ This algorithm does not need any fitting. """
        pass

    def compute_of(self, image: Image, mask: Mask, **bd_specs) -> Field:
        """ Computes the orientation field.

        The orientation field returned by this function must be sampled according to the FVC-OnGoing specifications and the parameters specified in ``bd_specs`` keyword arguments.
        
        Args:
            image: Fingerprint original image
            mask: Foreground mask (sampled as described in FVC-OnGoing)

        Keyword Arguments:
            border_x: horizontal border (in pixels)
            border_y: vertical border (in pixels)
            step_x: horizontal distance between sample points (in pixels)
            step_y: vertical distance between sample points (in pixels)
        """
        preprocessing = bd_specs.get('preprocessing', True)
        if 'preprocessing' in bd_specs:
            del bd_specs['preprocessing']
        if preprocessing:
            image, _ = self.segmentor.segment(image)
            mask = convert_to_full(mask, **bd_specs)
        lro, rel = LRO(
            image, mask=mask, 
            ridge_dist=self.ridge_dist,
            number_angles=self.number_angles,
            along_sigma_ratio=self.along_sigma_ratio,
            ortho_sigma=self.ortho_sigma)
        field = polar2cart(lro, rel, retField=True)
        field = subsample(field, is_field=True, **bd_specs)
        return field

class AnGaFIS_OF_Estimator_Complete(AnGaFIS_OF_Estimator):
    def __init__(self,
        LRF_min_disk_size: int = 10,
        LRF_rel_check_grid_step: int = 10,
        LRF_rel_check_threshold: float = 30,
        LRF_segment_n_points: int = 15,
        LRF_segment_length: int = 30,
        LRF_gaussian_smooth_std: float = 0.1,
        LRO1_scale_factor: float = 1.5,
        SM1_radius_factor: float = 1,
        SM1_sample_dist: float = 3,
        SM1_relax: float = 1,
        DM1_radius_factor: float = 1,
        DM1_sample_dist: float = 3,
        DM1_drift_threshold: float = 0.3,
        DM1_shrink_rad_factor: float = 3.5,
        DM1_blur_stddev: float = 0.5,
        LRO2_scale_factor: float = 0.5,
        SM2_radius_factor: float = 0.7,
        SM2_sample_dist: float = 3,
        SM2_relax: float = 0.9,
        DMASK2_blur_stddev: float = 0.5,
        DMASK2_threshold: float = 0.3,
        DMASK2_ccomp_ext_thres: float = 1,
        DMASK2_dil_rad_factor: float = 3,
        DM3_radius_factor: float = 1,
        DM3_sample_dist: float = 3,
        DM3_drift_threshold: float = 0.1,
        DM3_shrink_rad_factor: float = 3.5,
        DM3_ccomp_ext_thres: float = 0.75,
        DM3_dil_rad_factor: float = 2,
        IS_max_iterations: int = 10,
        IS_binlev_step: float = 0.01,
        IS_SMOOTH_radius_factor: float = 1.5,
        IS_SMOOTH_sample_dist: float = 3,
        IS_SMOOTH_relaxation: float = 1.0,
        IS_DMASK_bin_level: float = 0.1,
        IS_GMASK_erode_rad_factor: float = 0.1,
        IS_GMASK_dilate_rad_factor: float = 2.0,
        IS_GMASK_blur_stddev: float = 0.5) :
        """ Initializes and stores all the algorithm's parameters. """
        pars = inspect.signature(AnGaFIS_Seg_Estimator.__init__)
        for par in pars.parameters.keys():
            if par != 'self':
                setattr(self, par, eval(par))
        self.segmentor = AnGaFIS_Seg_Estimator(enhanceOnly=True)
        self.lro_estimator = AnGaFIS_OF_Estimator()

    def compute_of(self, image: Image, mask: Mask, **bd_specs) -> Field:
        """ See documentation of super.compute_of """
        image, _ = self.segmentor.segment(image)
        mask = convert_to_full(mask, **bd_specs)
        field = self.lro_estimator.compute_of(image, mask, **bd_specs, preprocessing=False)
        field = reliable_iterative_smoothing(image, mask, field, 
            # Take some arguments from the lro_estimator
            LRO1__number_angles=self.lro_estimator.number_angles,
            LRO1__along_sigma_ratio=self.lro_estimator.along_sigma_ratio,
            LRO1__ortho_sigma=self.lro_estimator.ortho_sigma,
            LRO2__number_angles=self.lro_estimator.number_angles,
            LRO2__along_sigma_ratio=self.lro_estimator.along_sigma_ratio,
            LRO2__ortho_sigma=self.lro_estimator.ortho_sigma,
            # Pass all the arguments of the initializer
            **{par:eval('self.{}'.format(par), {'self':self}) for par in pars.parameters.keys() if par != 'self'}
        )
        field = subsample(field, is_field=True, **bd_specs)
        return field

class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield [], range(len(X))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


if __name__ == '__main__':
    import PIL
    import numpy as np
    import time, datetime
    filename1 = "/Users/MacD/Documents/Databases/temp/f0363_10.bmp"
    # filename2 = "/Users/MacD/Documents/Databases/temp/s0363_10.bmp"
    readimg = lambda ff: np.array(PIL.Image.open(ff).convert('L'))
    left = readimg(filename1)
    # right = readimg(filename2)
    # X = [(left, right)]
    # an_score = AnGaFISMatcher().match_scores(X)
    # nb_score = NBISMatcher().match_scores(X)

    # print(an_score, nb_score)
    estimator = AnGaFIS_OF_Estimator_Complete()
    X = [estimator.serialize_Xrow(left[:508, :508], np.ones((61, 61), dtype=bool), [14, 14, 8, 8])]
    y = np.ones_like(left)
    estimator.fit(X, y)
    start = time.time()
    estimator.predict(X)
    print('Done in', datetime.timedelta(seconds=int(time.time()-start)))
