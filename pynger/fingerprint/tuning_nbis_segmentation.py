import inspect
from pynger.fingerprint.tuning_segmentation import ScoreOverlapMeasure, SegmentationEstimator
from pynger.fingerprint.nbis import segment


# PERHAPS IT DOES NOT WORK
class NBIS_Seg_Estimator(ScoreOverlapMeasure, SegmentationEstimator):
    def __init__(self,
        fac_n: int = 5,
        min_fg: int = 2000,
        max_fg: int = 8000,
        nerode: int = 3,
        rsblobs: int = 1,
        fill: int = 1,
        min_n: int = 25,
        hist_thresh: int = 20,
        origras_wmax: int = 2000,
        origras_hmax: int = 2000,
        fac_min: float = 0.75,
        fac_del: float = 0.05,
        slope_thresh: float = 0.90,
    ):
        """ Initializes and stores all the algorithm's parameters """
        pars = inspect.signature(NBIS_Seg_Estimator.__init__)
        for par in pars.parameters.keys():
            if par != 'self':
                setattr(self, par, eval(par))

    def segment(self, image):
        """ Segments the input fingerprint image """
        pars = inspect.signature(NBIS_Seg_Estimator.__init__)
        return segment(image,
            **{par:eval('self.{}'.format(par), {'self':self}) for par in pars.parameters.keys() if par != 'self'},
        )
