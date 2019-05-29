import os
import time
from subprocess import Popen, PIPE
from itertools import chain

import numpy as np
import PIL
import pandas as pd
# from sklearn.base import BaseEstimator, ClassifierMixin

from pynger.config import __NBIS_LIB__
from pynger.field.manipulation import angle, polar2cart
from pynger.fingerprint.sampling import subsample
from pynger.types import Field, Image, Mask
from pynger.fingerprint.nbis import mindtct
from pynger.fingerprint.nbis import compute_lro as nbis_lro
from pynger.fingerprint.nbis_wrapper import nbis_angle2idx, nbis_idx2angle, nbis_bozorth3
from pynger.fingerprint.tuning_lro import AnGaFIS_OF_Estimator_Complete, AnGaFIS_OF_Estimator, GaborEstimator
from pynger.misc import grouper

from joblib import Parallel, delayed, Memory


def minutiae_selection(minutiae):
	""" Selects the subset of most reliable minutiae.
	"""
	M = np.array([(m['x'], m['y'], m['direction'], m['reliability']) for m in minutiae])
	M[:,2] = np.round(np.rad2deg(nbis_idx2angle(M[:,2], N=16)))
	M[:,3] = np.round(M[:,3] * 100.0)
	M = M.astype(int)
	M = M[M[:,3] > np.percentile(M[:,3], 5), :]
	return M


class FingerprintMatcher:
    def __init__(self, cache_dir='joblib_cache', verbose: int=10):
        cache_dir = os.path.abspath(cache_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.memory = Memory(cache_dir, verbose=0)
        # self.compute_lro = self.memory.cache(self.compute_lro) # useless by now
        self.verbose = verbose
        self.minutiaeLUT = {}

    def __del__(self):
        self.memory.clear(warn=False)
        if hasattr(self, 'minutiaeLUT'):
            for path in self.minutiaeLUT.values():
                os.remove(path)

    def compute_lro(self, image, bd_specs, num_dir):
        raise NotImplementedError("Derived class must reimplement this method")

    def fit(self, X, y):
        return self

    def precompute(self, X):
        """ Precomputes the minutiae for all the files involved in matching.

        Args:
            X (iterable): iterable where each element is an absolute file path.

        Note:
            The iterable X must contain each and every file that needs to be involved in the matching phase.
        """
        # Ensure that X is a list and not a generator
        X = list(X)
        # Create a function with fixed paramters
        def field_compute(padded_img, blkoffs, num_dir):
            # Get border and step information
            i, j = np.unravel_index(blkoffs, shape=padded_img.shape)
            bd_specs = {
                'border_x': j[0,0],
                'border_y': i[0,0],
                'step_x': j[0,1]-j[0,0],
                'step_y': i[1,0]-i[0,0],
            }
            field, mask = self.compute_lro(padded_img, bd_specs, num_dir)
            if field is None: # allows to not change the direction map
                return None
            # Average pooling on field
            field = subsample(field, is_field=True, **bd_specs, smooth=True, policy='nist')
            # Convert field to index
            lro = angle(field, keepDims=False)
            idx = nbis_angle2idx(lro, N=num_dir)
            # Eventually apply a mask
            mask = subsample(mask.astype(int), is_field=False, **bd_specs, smooth=False, policy='nist')
            mask = np.round(mask).astype(bool)
            idx[np.logical_not(mask)] = -1
            return idx.astype('int32')

        def compute_minutiae(path):
            try:
                image = np.array(PIL.Image.open(path).convert('L'))
                M = mindtct(image, field_compute, contrast_boost=True)[-1]
                M = minutiae_selection(M)
            except Exception as err:
                print('Warning: skipping image due to', err)
                return None
            return M

        minutiae = Parallel(verbose=self.verbose)(delayed(compute_minutiae)(x) for x in X)

        for x, M in zip(X, minutiae):
            if M is None:
                continue
            # Create a filename that hopefully is not taken by other objects
            filename = '{}{}{}.xyt'.format(id(self), id(M), time.time())
            # Save minutiae to file
            filepath = os.path.join(self.cache_dir, filename)
            to_csv_options = {'sep': ' ', 'header': False, 'index': False}
            pd.DataFrame(M).to_csv(filepath, **to_csv_options)
            # Record the filepath in a dictionary
            self.minutiaeLUT[x] = filepath

    def match_scores(self, X):
        """ Perform matching exploiting the previously computed minutiae.

        Args:
            X (iterable): each element is a file absolute path, and must correspond to one of the file paths passed to the pre-computation function.
        """
        def _scores_from_batch(batch):
            """ Computes the scores for a batch with couples of file paths. """
            # Filter out null elements, coming from the last batch
            batch = filter(None, batch)
            # Create the mates file
            mates_file = os.path.join(self.cache_dir, '{}{}{}.lis'.format(id(self), id(batch), time.time()))
            excluded = []
            with open(mates_file, 'w') as f:
                for n, pair in enumerate(batch):
                    if pair[0] in self.minutiaeLUT and pair[1] in self.minutiaeLUT:
                        f.write(self.minutiaeLUT[pair[0]]+'\n')
                        f.write(self.minutiaeLUT[pair[1]]+'\n')
                    else:
                        excluded.append(n)
            # Run matcher
            exe_path = os.path.join(__NBIS_LIB__, 'bin', 'bozorth3')
            command = "{} -M \"{}\"".format(exe_path, mates_file)
            with Popen(command, cwd=self.cache_dir, shell=True, universal_newlines=True, stdout=PIPE, stderr=PIPE) as proc:
                err = proc.stderr.read()
                if err != "":
                    raise RuntimeError(err)
                # Read the list of scores
                # Splits on newlines and remove empty strings
                scores = [int(k) for k in filter(None, proc.stdout.read().split('\n'))]
            # Put Nones where a matching couldn't be executed
            for n in excluded:
                scores.insert(n, None)
            return scores

        X = grouper(X, 256)
        scores = Parallel(verbose=self.verbose, batch_size=512)(delayed(_scores_from_batch)(x) for x in X)
        scores = list(chain(*scores))
        
        return scores

    def predict(self, X):
        return (self.match_scores(X) > self.threshold).astype(int)

class AnGaFISMatcherRefinement(FingerprintMatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = {}
        self.lro_estimator = AnGaFIS_OF_Estimator_Complete()
        self.lro_estimator.segmentor.set_params(enhanceOnly=False)

    def compute_lro(self, image, bd_specs, num_dir):
        # Update number of angles for LRO estimator
        self.lro_estimator.lro_estimator.set_params(number_angles=num_dir)
        # Compute the lro
        field, mask = self.lro_estimator.compute_of(image, None)
        return field, mask

class AnGaFISMatcherLRO(FingerprintMatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = {}
        self.lro_estimator = AnGaFIS_OF_Estimator()
        self.lro_estimator.segmentor.set_params(enhanceOnly=False)

    def compute_lro(self, image, bd_specs, num_dir):
        # Update number of angles for LRO estimator
        self.lro_estimator.set_params(number_angles=num_dir)
        # Compute the lro
        field, mask = self.lro_estimator.compute_of(image, None, onlyLRO=True)
        return field, mask

class GaborLROMatcher(FingerprintMatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = {}
        self.lro_estimator = GaborEstimator()
        self.lro_estimator.segmentor.set_params(enhanceOnly=False)

    def compute_lro(self, image, bd_specs, num_dir):
        # Update number of angles for LRO estimator
        self.lro_estimator.set_params(number_angles=num_dir)
        # Compute the lro
        field, mask = self.lro_estimator.compute_of(image, None, onlyLRO=True)
        return field, mask

class NBISMatcher(FingerprintMatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_lro(self, padded_img, bd_specs, num_dir):
        # Returning None does not modify the direction map computed by NBIS
        return None, None

        # preprocess = False
        # # Eventually preprocess the image and get the foreground mask
        # if preprocess:
        #     enhanced = angafis_preprocessing(padded_img, verbose=False)
        #     mask = enhanced > 0
        #     mask = binary_closing(mask, structure=np.ones((4,4)), iterations=1)
        #     lro_idx, _, _ = nbis_lro(enhanced, **params)
        # else:
        #     lro_idx, _, _ = nbis_lro(padded_img, **params)
        # lro = nbis_idx2angle(lro_idx, N=num_dir)
        # field = polar2cart(lro, 1, retField=True)
        # # Get border and step information
        # i, j = np.unravel_index(blkoffs, shape=padded_img.shape)
        # border_x = j[0,0]
        # border_y = i[0,0]
        # step_x = j[0,1]-j[0,0]
        # step_y = i[1,0]-i[0,0]
        # # Averaging pooling on field
        # field = subsample(
        #     field, is_field=True, 
        #     border_x=border_x, border_y=border_y,
        #     step_x=step_x, step_y=step_y, smooth=True, policy='nist')
        # # Convert field to index
        # lro = angle(field, keepDims=False)
        # idx = nbis_angle2idx(lro, N=num_dir)
        # # Eventually apply a mask
        # if preprocess:
        #     mask = np.round(subsample(
        #         mask.astype(int), is_field=False, 
        #         border_x=border_x, border_y=border_y,
        #         step_x=step_x, step_y=step_y, smooth=False, policy='nist')).astype(bool)
        #     idx[np.logical_not(mask)] = -1
        # return idx.astype('int32')