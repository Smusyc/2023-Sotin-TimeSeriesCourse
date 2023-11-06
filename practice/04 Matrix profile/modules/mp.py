import numpy as np
import pandas as pd
import math

import stumpy
from stumpy import config
from stumpy import core

def compute_mp(ts1, m, ts2=None,exclusion_zone=None,  ignore_trivial=False):
    """
    Compute the matrix profile.

    Parameters
    ----------
    ts1 : numpy.ndarrray
        The first time series.

    m : int
        The subsequence length.

    exclusion_zone : int, default = None
        Exclusion zone.

    ts2 : numpy.ndarrray, default = None
        The second time series.

    Returns
    -------
    output : dict
        The matrix profile structure 
        (matrix profile, matrix profile index, subsequence length, 
        exclusion zone, the first and second time series).
    """
    
    # INSERT YOUR CODE
    if exclusion_zone is not None:
      config.STUMPY_EXCL_ZONE_DENOM = exclusion_zone
    else: config.STUMPY_EXCL_ZONE_DENOM = np.inf
    if ts2 != None: ts2 = ts2.astype(float)
    mp = stumpy.stump(ts1.astype(float),  m,  ts2, ignore_trivial = ignore_trivial)
    return {'mp': mp[:,0],
            'mpi': mp[:,1],
            'm' : m,
            'excl_zone': exclusion_zone,
            'data': {'ts1' : ts1, 'ts2' : ts2},
            'indices': {'left' : mp[:,-2], 'right' : mp[:,-1]}
            }

