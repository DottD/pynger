import pickle
import time
import datetime

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from pynger.field.manipulation import angle, angle_diff, cart2polar, polar2cart, magnitude
from pynger.fingerprint.sampling import convert_to_full, subsample
from pynger.fingerprint.orientation import LRO
from pynger.fingerprint.refinement import reliable_iterative_smoothing
from pynger.types import Field, Image, List, Mask, Union
from pynger.fingerprint.nbis import mindtct
from pynger.fingerprint.nbis import compute_lro as nbis_lro
from pynger.fingerprint.nbis_wrapper import nbis_angle2idx, nbis_idx2angle, nbis_bozorth3
from pynger.fingerprint.tuning_lro import AnGaFIS_OF_Estimator_Complete, AnGaFIS_OF_Estimator

from joblib import Parallel, delayed


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
            minutiae_l, minutiae_r, score = nbis_bozorth3(minutiae_l, minutiae_r, verbose=True, bozorth3_exe='/Users/MacD/Documents/Libraries/fp-libs/nbis-install/bin/bozorth3')

            return score

        if hasattr(self, 'verbose') and self.verbose:
            verbosity = 10
        else:
            verbosity = 0

        scores = Parallel(verbose=verbosity)(delayed(get_score)(X_line) for X_line in X)
        
        return scores

    def predict(self, X):
        return (self.match_scores(X) > self.threshold).astype(int)

class AnGaFISMatcherRefinement(FingerprintMatcher):
    def __init__(self):
        self.params = {}
        self.lro_estimator = AnGaFIS_OF_Estimator_Complete()
        self.lro_estimator.segmentor.set_params(enhanceOnly=False)

    def compute_lro(self, padded_img, blkoffs, num_dir, params):
        # Update number of angles for LRO estimator
        self.lro_estimator.lro_estimator.set_params(number_angles=num_dir)
        # Compute the lro
        field, mask = self.lro_estimator.compute_of(padded_img, None)
        # Get border and step information
        i, j = np.unravel_index(blkoffs, shape=padded_img.shape)
        border_x = j[0,0]
        border_y = i[0,0]
        step_x = j[0,1]-j[0,0]
        step_y = i[1,0]-i[0,0]
        # Average pooling on field
        field = subsample(
            field, is_field=True, 
            border_x=border_x, border_y=border_y,
            step_x=step_x, step_y=step_y, smooth=True, policy='nist')
        # Convert field to index
        lro = angle(field, keepDims=False)
        idx = nbis_angle2idx(lro, N=num_dir)
        # Eventually apply a mask
        mask = np.round(subsample(
            mask.astype(int), is_field=False, 
            border_x=border_x, border_y=border_y,
            step_x=step_x, step_y=step_y, smooth=False, policy='nist')).astype(bool)
        idx[np.logical_not(mask)] = -1
        return idx.astype('int32')

class AnGaFISMatcherLRO(FingerprintMatcher):
    def __init__(self):
        self.params = {}
        self.lro_estimator = AnGaFIS_OF_Estimator()
        self.lro_estimator.segmentor.set_params(enhanceOnly=False)

    def compute_lro(self, padded_img, blkoffs, num_dir, params):
        # Update number of angles for LRO estimator
        self.lro_estimator.set_params(number_angles=num_dir)
        # Compute the lro
        field, mask = self.lro_estimator.compute_of(padded_img, None, onlyLRO=True)
        # Get border and step information
        i, j = np.unravel_index(blkoffs, shape=padded_img.shape)
        border_x = j[0,0]
        border_y = i[0,0]
        step_x = j[0,1]-j[0,0]
        step_y = i[1,0]-i[0,0]
        # Average pooling on field
        field = subsample(
            field, is_field=True, 
            border_x=border_x, border_y=border_y,
            step_x=step_x, step_y=step_y, smooth=True, policy='nist')
        # Convert field to index
        lro = angle(field, keepDims=False)
        idx = nbis_angle2idx(lro, N=num_dir)
        # Eventually apply a mask
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