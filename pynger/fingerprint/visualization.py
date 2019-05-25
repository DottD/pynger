import numpy as np
import matplotlib.pyplot as plt
from pynger.misc import vis_stream_orient, vis_orient
from pynger.fingerprint.nbis_wrapper import minutiae_selection


def plot_minutiae(minutiae):
    """ Plots minutiae on current figure.

    Args:
        minutiae: Structure as the output of mindtct.
    """
    M = minutiae_selection(minutiae)
    x, y, theta, quality = M.T
    clim = (np.min(quality), np.max(quality))
    theta = np.deg2rad(theta)
    plt.quiver(x, y, np.cos(theta), np.sin(theta), quality, **{
        'units': 'xy',
        'pivot': 'tail',
        'angles': 'uv',
        'clim': clim,
        'cmap': 'autumn',
    })
    plt.scatter(x, y, c=quality, **{
        'clim': clim,
        'cmap': 'autumn',
    })
