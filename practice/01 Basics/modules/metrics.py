import numpy as np


def ED_distance(ts1, ts2):
    """
    Calculate the Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    ed_dist : float
        Euclidean distance between ts1 and ts2.
    """
    
    ed_dist = 0
    ed_dist=np.sqrt(sum((ts1 - ts2)**2))
    # INSERT YOUR CODE
    
    return ed_dist


def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE 
    m=ts1.shape[0] 
    m_std = m * np.std(ts1) * np.std(ts2)
    m_mean = m * np.mean(ts1) * np.mean(ts2)
    norm_ed_dist = np.sqrt( np.abs(2*m*(1 - ( np.dot(ts1, ts2)- m_mean) / m_std )))
    return norm_ed_dist


def DTW_distance(q, c, r=None):
    """
    Calculate DTW distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    r : float
        Warping window size.
    
    Returns
    -------
    dtw_dist : float
        DTW distance between ts1 and ts2.
    """
    n=len(q)
    m=len(c)
    dtw_matrix = np.zeros((n+1,m+1))
    dtw_matrix[0,:]=np.inf
    dtw_matrix[:,0]=np.inf
    dtw_matrix[0,0]=0
    
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost=np.square(q[i-1]-c[j-1])
            dtw_matrix[i,j] = cost+min(dtw_matrix[i-1,j],dtw_matrix[i,j-1],dtw_matrix[i-1,j-1])

    # INSERT YOUR CODE
    dtw_dist = dtw_matrix[n,m]
    
    return dtw_dist