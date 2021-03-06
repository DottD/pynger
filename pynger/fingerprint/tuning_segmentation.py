import itertools
import inspect
from statistics import mean

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from scipy.ndimage.morphology import distance_transform_cdt

from pynger.types import Field, Image, List, Mask, Union, Tuple
from pynger.fingerprint.cangafris import segment_enhance


class SegmentationEstimator(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        # Make prediction
        for img in X:
            _, mask = self.segment(img)
            yield ~mask.astype(bool) # ~ needed for True values on the foreground

    def get_scores(self, X, y=None):
        " Get the similarity measure over all the dataset "
        if y is None:
            raise ValueError("A true y must be given")
        # Split the iterator
        X1, X2 = itertools.tee(X, 2)
        # Get the predicted y
        pred_y = self.predict(X1)
        # Compute the similarity measure
        similarity = (self.compute_error(ty, py) for ty, py in zip(y, pred_y))
        return similarity
    
    def score(self, X, y=None):
        # Compute the average error
        similarity = self.get_scores(X, y)
        return mean(similarity)

    def segment(self, image) -> Tuple[Image, Mask]:
        """ Segment the fingerprint image.

        Return:
            This method shall return the pair (img, mask), where mask is the foreground mask and img is the enhanced version of the original image (at least within the mask).
        
        Important:
            Derived class must reimplement this method.

        Example:
            As an example, see :class:`.AnGaFIS_Seg_Estimator`.
        """
        raise NotImplementedError("Derived class must reimplement this method")

    def compute_error(self, true_mask: Mask, pred_mask: Mask):
        """ Computes the error between mask1 and mask2.

        Args:
            true_mask: First mask
            pred_mask: Second mask
        
        Important:
            Derived class must reimplement this method.

        Example:
            As an example, see :class:`.ScoreOverlapMeasure`.
        """
        raise NotImplementedError("Derived class must reimplement this method")

class ScoreOverlapMeasure:
    def compute_error(self, true_mask: Mask, pred_mask: Mask):
        """ Compute the similarity of the two masks as the number of elements of their intersection over their union """
        union = np.count_nonzero(true_mask | pred_mask)
        similarity = np.count_nonzero(true_mask & pred_mask) / union if union > 0 else 0.0
        return similarity

class ScoreElementwiseAccuracy:
    def compute_error(self, true_mask: Mask, pred_mask: Mask):
        """ Compute the similarity of the two masks as the number of elements of their intersection over their union """
        # Ensure that masks have binary values
        true_mask = true_mask > 0
        pred_mask = pred_mask > 0
        return balanced_accuracy_score(true_mask.ravel(), pred_mask.ravel())

class ScoreBaddeleyDissimilarity:
    def compute_error(self, true_mask: Mask, pred_mask: Mask, c: int = 5, p: float = 2):
        """ Compute the Baddeley Error for binary images.
        
        Note:
            A.J. Baddeley - "An Error Metric for Binary Images"
        """
        # Ensure that masks have binary values
        true_mask = true_mask > 0
        pred_mask = pred_mask > 0
        # Handle masks filled with the same value
        xor_mask = true_mask ^ pred_mask
        if (~xor_mask).all(): # Masks equal
            return 1.0
        elif xor_mask.all(): # Masks completely different
            return 0.0
        # Compute metric
        true_edt = distance_transform_cdt(true_mask, metric='taxicab').astype(float)
        true_edt = np.minimum(true_edt, c)
        true_edt[true_edt < 0] = c # where a distance cannot be computed, set to maximum
        pred_edt = distance_transform_cdt(pred_mask, metric='taxicab').astype(float)
        pred_edt = np.minimum(pred_edt, c)
        pred_edt[pred_edt < 0] = c
        dist = np.abs(true_edt - pred_edt)
        dist /= c # c is the maximum possible distance
        dist = (dist**p).mean()**(1/p)
        return 1.0-dist

class AnGaFIS_Seg_Estimator(ScoreBaddeleyDissimilarity, SegmentationEstimator):
    def __init__(self,
        brightness: float = 0.35,
        leftCut: float = 0.25,
        rightCut: float = 0.5,
        histSmooth: int = 25,
        reparSmooth: int = 10,
        minVariation: float = 0.01,
        cropSimpleMarg: int = 5,
        scanAreaAmount: float = 0.1,
        gradientFilterWidth: float = 0.25,
        gaussianFilterSide: int = 5,
        binarizationLevel: float = 0.2,
        f: float = 3.5,
        slopeAngle: float = 1.5,
        lev: float = 0.95,
        topMaskMarg: int = 5,
        medFilterSide: int = 2,
        gaussFilterSide: int = 3,
        minFilterSide: int = 5,
        binLevVarMask: float = 0.45,
        dilate1RadiusVarMask: int = 1,
        erodeRadiusVarMask: int = 35,
        dilate2RadiusVarMask: int = 2,
        maxCompNumVarMask: int = 2,
        minCompThickVarMask: int = 75,
        maxHolesNumVarMask: int = -1,
        minHolesThickVarMask: int = 18,
        histeresisThreshold1Gmask: int = 1,
        histeresisThreshold2Gmask: int = 2,
        radiusGaussFilterGmask: int = 10,
        minMeanIntensityGmask: float = 0.2,
        dilate1RadiusGmask: int = 1,
        erodeRadiusGmask: int = 25,
        dilate2RadiusGmask: int = 2,
        maxCompNumGmask: int = 2,
        minCompThickGmask: int = 75,
        maxHolesNumGmask: int = -1,
        minHolesThickGmask: int = 15,
        histeresisThreshold3Gmask: int = 3,
        histeresisThreshold4Gmask: int = 4,
        dilate3RadiusGmask: int = 3,
        erode2RadiusGmask: int = 2,
        histeresisThreshold5Gmask: int = 5,
        histeresisThreshold6Gmask: int = 6,
        dilate4RadiusGmask: int = 4,
        radiusGaussFilterComp: int = 30,
        meanIntensityCompThreshold: float = 0.6,
        dilateFinalRadius: int = 10,
        erodeFinalRadius: int = 20,
        smoothFinalRadius: int = 10,
        maxCompNumFinal: int = 2,
        minCompThickFinal: int = 75,
        maxHolesNumFinal: int = 4,
        minHolesThickFinal: int = 30,
        fixedFrameWidth: int = 20,
        smooth2FinalRadius: int = 2,
        minMaxFilter: int = 5,
        mincp1: float = 0.75,
        mincp2: float = 0.9,
        maxcp1: float = 0.0,
        maxcp2: float = 0.25,
        enhanceOnly: bool = False,
    ):
        """ Initializes and stores all the algorithm's parameters """
        pars = inspect.signature(AnGaFIS_Seg_Estimator.__init__)
        for par in pars.parameters.keys():
            if par != 'self':
                setattr(self, par, eval(par))

    def segment(self, image):
        """ Segments the input fingerprint image """
        pars = inspect.signature(AnGaFIS_Seg_Estimator.__init__)
        try:
            ret = segment_enhance(image,
                **{par:eval('self.{}'.format(par), {'self':self}) for par in pars.parameters.keys() if par != 'self'},
            )
        except Exception as err:
            print("Error in segmentation:", err)
            ret = (image, np.ones_like(image, dtype=bool))
        return ret
