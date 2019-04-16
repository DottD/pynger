import pickle
import time
import datetime

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from pynger.field.manipulation import angle, angle_diff, cart2polar, polar2cart
from pynger.fingerprint.FVC_utilities import convert_to_full, subsample
from pynger.fingerprint.orientation import LRO
from pynger.fingerprint.refinement import reliable_iterative_smoothing
from pynger.types import Field, Image, List, Mask, Union
from pynger.fingerprint.nbis import mindtct
from pynger.fingerprint.nbis import compute_lro as nbis_lro
from pynger.fingerprint.nbis_wrapper import nbis_angle2idx, nbis_idx2angle, nbis_bozorth3

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
        self.X_ = X
        self.y_ = y
        # Fit the model on training data
        self.train_on_data(X, y)
        # Return the classifier
        return self

    
    def predict(self, X):
        # Check is fit had been called
        if not all([hasattr(self, attr) for attr in ['X_', 'y_']]):
            raise RuntimeError("You must call `fit` before predicting data!")

        # Input validation
        X = self.check_X(X)

        # Make prediction
        pred_y = []
        for lineX in X:
            image, mask, specs = LROEstimator.deserialize_Xrow(lineX)
            bd_specs = {
                'border_x': specs[0],
                'border_y': specs[1],
                'step_x': specs[2],
                'step_y': specs[3],
            }
            field = self.compute_of(image, mask, **bd_specs)
            pred_y.append(LROEstimator.serialize_yrow(field))

        return pred_y

    
    def score(self, X, y=None):
        if y is None:
            raise ValueError("A true y must be given")
        # Check that X and y have correct shape
        X, y = self.check_X_y(X, y)
        # Get the predicted y
        pred_y = self.predict(X)
        # Accumulate the average error
        avgerr = []
        for py, ty, x in zip(pred_y, y, X):
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
        ridge_dist: int = 10,
        number_angles: int = 36,
        along_sigma_ratio: float = 0.85,
        ortho_sigma: float = 1.0,
        LRF_min_disk_size: int = 10,
        LRF_rel_check_grid_step: int = 10,
        LRF_rel_check_threshold: float = 0.5,
        LRF_segment_n_points: int = 15,
        LRF_segment_length: int = 30,
        LRF_gaussian_smooth_std: float = 0.1,
        LRO1_scale_factor: float = 1.5,
        LRO1_number_angles: int = 36,
        LRO1_along_sigma_ratio: float = 0.85,
        LRO1_ortho_sigma: float = 1.0,
        SM1_radius_factor: float = 1,
        SM1_sample_dist: float = 1,
        SM1_relax: float = 1,
        DM1_radius_factor: float = 1,
        DM1_sample_dist: float = 1,
        DM1_drift_threshold: float = 0.3,
        DM1_shrink_rad_factor: float = 3.5,
        DM1_blur_stddev: float = 0.5,
        LRO2_scale_factor: float = 0.5,
        LRO2_number_angles: int = 36,
        LRO2_along_sigma_ratio: float = 0.85,
        LRO2_ortho_sigma: float = 1.0,
        SM2_radius_factor: float = 0.7,
        SM2_sample_dist: float = 1,
        SM2_relax: float = 0.9,
        DMASK2_blur_stddev: float = 0.5,
        DMASK2_threshold: float = 0.3,
        DMASK2_ccomp_ext_thres: float = 1,
        DMASK2_dil_rad_factor: float = 3,
        DM3_radius_factor: float = 1,
        DM3_sample_dist: float = 1,
        DM3_drift_threshold: float = 0.1,
        DM3_shrink_rad_factor: float = 3.5,
        DM3_ccomp_ext_thres: float = 0.75,
        DM3_dil_rad_factor: float = 2,
        IS_max_iterations: int = 10,
        IS_binlev_step: float = 0.01,
        IS_SMOOTH_radius_factor: float = 1.5,
        IS_SMOOTH_sample_dist: float = 1.0,
        IS_SMOOTH_relaxation: float = 1.0,
        IS_DMASK_bin_level: float = 0.1,
        IS_GMASK_erode_rad_factor: float = 0.1,
        IS_GMASK_dilate_rad_factor: float = 2.0,
        IS_GMASK_blur_stddev: float = 0.5) :
        """ Initializes and stores all the algorithm's parameters. """
        self.ridge_dist = ridge_dist
        self.number_angles = number_angles
        self.along_sigma_ratio = along_sigma_ratio
        self.ortho_sigma = ortho_sigma
        self.LRF_min_disk_size = LRF_min_disk_size
        self.LRF_rel_check_grid_step = LRF_rel_check_grid_step
        self.LRF_rel_check_threshold = LRF_rel_check_threshold
        self.LRF_segment_n_points = LRF_segment_n_points
        self.LRF_segment_length = LRF_segment_length
        self.LRF_gaussian_smooth_std = LRF_gaussian_smooth_std
        self.LRO1_scale_factor = LRO1_scale_factor
        self.LRO1_number_angles = LRO1_number_angles
        self.LRO1_along_sigma_ratio = LRO1_along_sigma_ratio
        self.LRO1_ortho_sigma = LRO1_ortho_sigma
        self.SM1_radius_factor = SM1_radius_factor
        self.SM1_sample_dist = SM1_sample_dist
        self.SM1_relax = SM1_relax
        self.DM1_radius_factor = DM1_radius_factor
        self.DM1_sample_dist = DM1_sample_dist
        self.DM1_drift_threshold = DM1_drift_threshold
        self.DM1_shrink_rad_factor = DM1_shrink_rad_factor
        self.DM1_blur_stddev = DM1_blur_stddev
        self.LRO2_scale_factor = LRO2_scale_factor
        self.LRO2_number_angles = LRO2_number_angles
        self.LRO2_along_sigma_ratio = LRO2_along_sigma_ratio
        self.LRO2_ortho_sigma = LRO2_ortho_sigma
        self.SM2_radius_factor = SM2_radius_factor
        self.SM2_sample_dist = SM2_sample_dist
        self.SM2_relax = SM2_relax
        self.DMASK2_blur_stddev = DMASK2_blur_stddev
        self.DMASK2_threshold = DMASK2_threshold
        self.DMASK2_ccomp_ext_thres = DMASK2_ccomp_ext_thres
        self.DMASK2_dil_rad_factor = DMASK2_dil_rad_factor
        self.DM3_radius_factor = DM3_radius_factor
        self.DM3_sample_dist = DM3_sample_dist
        self.DM3_drift_threshold = DM3_drift_threshold
        self.DM3_shrink_rad_factor = DM3_shrink_rad_factor
        self.DM3_ccomp_ext_thres = DM3_ccomp_ext_thres
        self.DM3_dil_rad_factor = DM3_dil_rad_factor
        self.IS_max_iterations = IS_max_iterations
        self.IS_binlev_step = IS_binlev_step
        self.IS_SMOOTH_radius_factor = IS_SMOOTH_radius_factor
        self.IS_SMOOTH_sample_dist = IS_SMOOTH_sample_dist
        self.IS_SMOOTH_relaxation = IS_SMOOTH_relaxation
        self.IS_DMASK_bin_level = IS_DMASK_bin_level
        self.IS_GMASK_erode_rad_factor = IS_GMASK_erode_rad_factor
        self.IS_GMASK_dilate_rad_factor = IS_GMASK_dilate_rad_factor
        self.IS_GMASK_blur_stddev = IS_GMASK_blur_stddev

    def compute_of(self, image: Image, mask: Mask, **bd_specs) -> Field:
        """ See documentation of super.compute_of """
        mask = convert_to_full(mask, **bd_specs)
        lro, rel = LRO(
            image, mask=mask, 
            ridge_dist=self.ridge_dist,
            number_angles=self.number_angles,
            along_sigma_ratio=self.along_sigma_ratio,
            ortho_sigma=self.ortho_sigma)
        field = polar2cart(lro, rel, retField=True)
        field = reliable_iterative_smoothing(image, mask, field, 
            LRF__min_disk_size=self.LRF_min_disk_size,
            LRF__rel_check_grid_step=self.LRF_rel_check_grid_step,
            LRF__rel_check_threshold=self.LRF_rel_check_threshold,
            LRF__segment_n_points=self.LRF_segment_n_points,
            LRF__segment_length=self.LRF_segment_length,
            LRF__gaussian_smooth_std=self.LRF_gaussian_smooth_std,
            LRO1__scale_factor=self.LRO1_scale_factor,
            LRO1__number_angles=self.LRO1_number_angles,
            LRO1__along_sigma_ratio=self.LRO1_along_sigma_ratio,
            LRO1__ortho_sigma=self.LRO1_ortho_sigma,
            SM1__radius_factor=self.SM1_radius_factor,
            SM1__sample_dist=self.SM1_sample_dist,
            SM1__relax=self.SM1_relax,
            DM1__radius_factor=self.DM1_radius_factor,
            DM1__sample_dist=self.DM1_sample_dist,
            DM1__drift_threshold=self.DM1_drift_threshold,
            DM1__shrink_rad_factor=self.DM1_shrink_rad_factor,
            DM1__blur_stddev=self.DM1_blur_stddev,
            LRO2__scale_factor=self.LRO2_scale_factor,
            LRO2__number_angles=self.LRO2_number_angles,
            LRO2__along_sigma_ratio=self.LRO2_along_sigma_ratio,
            LRO2__ortho_sigma=self.LRO2_ortho_sigma,
            SM2__radius_factor=self.SM2_radius_factor,
            SM2__sample_dist=self.SM2_sample_dist,
            SM2__relax=self.SM2_relax,
            DMASK2__blur_stddev=self.DMASK2_blur_stddev,
            DMASK2__threshold=self.DMASK2_threshold,
            DMASK2__ccomp_ext_thres=self.DMASK2_ccomp_ext_thres,
            DMASK2__dil_rad_factor=self.DMASK2_dil_rad_factor,
            DM3__radius_factor=self.DM3_radius_factor,
            DM3__sample_dist=self.DM3_sample_dist,
            DM3__drift_threshold=self.DM3_drift_threshold,
            DM3__shrink_rad_factor=self.DM3_shrink_rad_factor,
            DM3__ccomp_ext_thres=self.DM3_ccomp_ext_thres,
            DM3__dil_rad_factor=self.DM3_dil_rad_factor,
            IS__max_iterations=self.IS_max_iterations,
            IS__binlev_step=self.IS_binlev_step,
            IS__SMOOTH__radius_factor=self.IS_SMOOTH_radius_factor,
            IS__SMOOTH__sample_dist=self.IS_SMOOTH_sample_dist,
            IS__SMOOTH__relaxation=self.IS_SMOOTH_relaxation,
            IS__DMASK__bin_level=self.IS_DMASK_bin_level,
            IS__GMASK__erode_rad_factor=self.IS_GMASK_erode_rad_factor,
            IS__GMASK__dilate_rad_factor=self.IS_GMASK_dilate_rad_factor,
            IS__GMASK__blur_stddev=self.IS_GMASK_blur_stddev
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

class FingerprintMatcher(BaseEstimator, ClassifierMixin):

    def compute_lro(self, img, blkoffs, num_dir, params):
        raise NotImplementedError("Derived class must reimplement this method")

    def fit(self, X, y):
        return self

    def match_scores(self, X):
        # Create a function with fixed paramters
        fun = lambda img, blkoffs, num_dir: self.compute_lro(img, blkoffs, num_dir, self.params) 

        def get_score(X_line):
            # Get the images
            image_l, image_r = X_line

            # Compute minutiae of the first image
            minutiae_l = mindtct(image_l, fun, contrast_boost=True)[-1]

            # Compute minutiae of the second image
            minutiae_r = mindtct(image_r, fun, contrast_boost=True)[-1]

            # Compute the match score
            minutiae_l, minutiae_r, score = nbis_bozorth3(minutiae_l, minutiae_r, verbose=True, bozorth3_exe='/Users/MacD/Documents/Libraries/NBIS/bin/bozorth3')

            return score

        if hasattr(self, 'verbose') and self.verbose:
            verbosity = 10
        else:
            verbosity = 0

        scores = Parallel(verbose=verbosity)(delayed(get_score)(X_line) for X_line in X)
        
        return scores

    def predict(self, X):
        return (self.match_scores(X) > self.threshold).astype(int)

class AnGaFISMatcher(FingerprintMatcher):
    def __init__(self, ridge_dist=15, number_angles=36, along_sigma_ratio=0.2, ortho_sigma=0.05):
        self.params = {
            'ridge_dist': ridge_dist,
            'number_angles': number_angles,
            'along_sigma_ratio': along_sigma_ratio,
            'ortho_sigma': ortho_sigma,
        }

    def compute_lro(self, padded_img, blkoffs, num_dir, params):
        params.update({'number_angles': num_dir})
        preprocess = False
        # Eventually preprocess the image and get the foreground mask
        if preprocess:
            enhanced = angafis_preprocessing(padded_img, verbose=False)
            mask = enhanced > 0
            mask = binary_closing(mask, structure=np.ones((4,4)), iterations=1)
            lro, _ = LRO(enhanced, **params)
        else:
            lro, _ = LRO(padded_img, **params)
        field = polar2cart(lro, 1, retField=True)
        # Get border and step information
        i, j = np.unravel_index(blkoffs, shape=padded_img.shape)
        border_x = j[0,0]
        border_y = i[0,0]
        step_x = j[0,1]-j[0,0]
        step_y = i[1,0]-i[0,0]
        # Averaging pooling on field
        field = subsample(
            field, is_field=True, 
            border_x=border_x, border_y=border_y,
            step_x=step_x, step_y=step_y, smooth=True, policy='nist')
        # Convert field to index
        lro = angle(field, keepDims=False)
        idx = nbis_angle2idx(lro, N=num_dir)
        # Eventually apply a mask
        if preprocess:
            mask = np.round(subsample(
                mask.astype(int), is_field=False, 
                border_x=border_x, border_y=border_y,
                step_x=step_x, step_y=step_y, smooth=False, policy='nist')).astype(bool)
            idx[np.logical_not(mask)] = -1
        return idx.astype('int32')

class NBISMatcher(FingerprintMatcher):
    def __init__(self, slit_range_thresh=10):
        self.params = {
            'slit_range_thresh': slit_range_thresh,
        }

    def compute_lro(self, padded_img, blkoffs, num_dir, params):
        preprocess = False
        # Eventually preprocess the image and get the foreground mask
        if preprocess:
            enhanced = angafis_preprocessing(padded_img, verbose=False)
            mask = enhanced > 0
            mask = binary_closing(mask, structure=np.ones((4,4)), iterations=1)
            lro_idx, _, _ = nbis_lro(enhanced, **params)
        else:
            lro_idx, _, _ = nbis_lro(padded_img, **params)
        lro = nbis_idx2angle(lro_idx, N=num_dir)
        field = polar2cart(lro, 1, retField=True)
        # Get border and step information
        i, j = np.unravel_index(blkoffs, shape=padded_img.shape)
        border_x = j[0,0]
        border_y = i[0,0]
        step_x = j[0,1]-j[0,0]
        step_y = i[1,0]-i[0,0]
        # Averaging pooling on field
        field = subsample(
            field, is_field=True, 
            border_x=border_x, border_y=border_y,
            step_x=step_x, step_y=step_y, smooth=True, policy='nist')
        # Convert field to index
        lro = angle(field, keepDims=False)
        idx = nbis_angle2idx(lro, N=num_dir)
        # Eventually apply a mask
        if preprocess:
            mask = np.round(subsample(
                mask.astype(int), is_field=False, 
                border_x=border_x, border_y=border_y,
                step_x=step_x, step_y=step_y, smooth=False, policy='nist')).astype(bool)
            idx[np.logical_not(mask)] = -1
        return idx.astype('int32')


if __name__ == '__main__':
    from pynger.fingerprint.tuning import AnGaFISMatcher, NBISMatcher
    import PIL
    import numpy as np
    filename1 = "/Users/MacD/Documents/Databases/temp/f0363_10.bmp"
    filename2 = "/Users/MacD/Documents/Databases/temp/s0363_10.bmp"
    readimg = lambda ff: np.array(PIL.Image.open(ff).convert('L'))
    left = readimg(filename1)
    right = readimg(filename2)
    X = [(left, right)]
    an_score = AnGaFISMatcher().match_scores(X)
    nb_score = NBISMatcher().match_scores(X)

    print(an_score, nb_score)