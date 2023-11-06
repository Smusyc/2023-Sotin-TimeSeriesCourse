import numpy as np

from modules.utils import *
from stumpy import config
from stumpy import core


def top_k_discords(matrix_profile, top_k=3, exclusion_zone=None):
    """
    Find the top-k discords based on matrix profile.

    Parameters
    ---------
    matrix_profile : dict
        The matrix profile structure.

    top_k : int
        Number of discords.

    Returns
    --------
    discords : dict
        Top-k discords (indices, distances to its nearest neighbor 
        and the nearest neighbors indices).
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    # INSERT YOUR CODE


    P = matrix_profile['mp'].astype(np.float64)

    

    discords_idx = np.full(top_k, -1, dtype=np.int64)
    discords_dist = np.full(top_k, np.NINF, dtype=np.float64)
    discords_nn_idx = np.full(top_k, -1, dtype=np.int64)

    for i in range(top_k):
        if np.all(P == np.NINF):
            break
        mp_discord_idx = np.argmax(P)

        discords_idx[i] = mp_discord_idx
        discords_dist[i] = P[mp_discord_idx]
        nnl = matrix_profile['indices']['left'][mp_discord_idx]
        nnr = matrix_profile['indices']['right'][mp_discord_idx]
        discords_nn_idx[i] = nnl if P[mp_discord_idx] - nnl < P[mp_discord_idx] - nnr else nnr

        core.apply_exclusion_zone(P, discords_idx[i], exclusion_zone, val=np.NINF)
    return {'indices': discords_idx,
            'distances': discords_dist,
            'nn_indices': discords_nn_idx
            }
