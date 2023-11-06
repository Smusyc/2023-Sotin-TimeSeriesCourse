#from _typeshed import NoneType
import numpy as np

from modules.utils import *

from stumpy import config
from stumpy import core


def top_k_motifs(matrix_profile, top_k=3, exclusion_zone=None):
    """
    Find the top-k motifs based on matrix profile.

    Parameters
    ---------
    matrix_profile : dict
        The matrix profile structure.

    top_k : int
        Number of motifs.

    Returns
    --------
    motifs : dict
        Top-k motifs (left and right indices and distances).
    """

    motifs_idx = []
    motifs_dist = []

    # INSERT YOUR CODE
    ''''motifs_dist = np.sort(matrix_profile['mp'])[:top_k]
    motifs_idx = []
    motifs_idx_left = []
    motifs_idx_right = []
    for dist in motifs_dist:
      min_mp = np.where(matrix_profile['mp']==dist,matrix_profile['mpi'],None)
      idxs = min_mp[min_mp != np.array(None)]
      for i in idxs:
        if i not in motifs_idx:
          motifs_idx.append(i)
          idx_i = np.where(matrix_profile['mpi']==i)[0][0]
          motifs_idx_left.append(matrix_profile['indices']['left'][idx_i])
          motifs_idx_right.append(matrix_profile['indices']['right'][idx_i])
    return {
            "indices" : np.array([motifs_idx_left,motifs_idx_right]).T,
            "distances" : motifs_dist
            }'''

    
    motifs_idx_left = []
    motifs_idx_right = []

    P = matrix_profile['mp'].astype(np.float64)

    

    discords_idx = np.full(top_k, -1, dtype=np.int64)
    discords_dist = np.full(top_k, np.PINF, dtype=np.float64)
    discords_nn_idx = np.full(top_k, -1, dtype=np.int64)

    for i in range(top_k):
        if np.all(P == np.PINF):
            break
        mp_discord_idx = np.argmin(P)
        if P[mp_discord_idx] == np.NINF:
          top_k = top_k+1
        else:
          discords_idx[i] = mp_discord_idx
          discords_dist[i] = P[mp_discord_idx]
          motifs_idx_left.append(matrix_profile['indices']['left'][mp_discord_idx])
          motifs_idx_right.append(matrix_profile['indices']['right'][mp_discord_idx])
        
        core.apply_exclusion_zone(P, discords_idx[i], exclusion_zone, val=np.PINF)
    return {'indices': np.array([motifs_idx_left,motifs_idx_right]).T,
            'distances': discords_dist
            }
