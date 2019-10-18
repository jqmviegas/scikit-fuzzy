"""
gk.py : Gustafson-Kessel clustering algorithm.
"""
import numpy as np
from scipy.spatial.distance import cdist


def _gk0(data, u_old, c, m):
    # Normalizing, then eliminating any potential zero values.
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / (np.ones((data.shape[1],
                                    1)).dot(np.atleast_2d(um.sum(axis=1))).T)
        
    covs = np.zeros((c, data.shape[1], data.shape[1]))
    dists = np.zeros((c, data.shape[0]))
    for _c in range(0, c):
        dmc = data-cntr[_c]

        covs[_c] = np.multiply(np.atleast_2d(um[_c]).T, dmc).T.dot(dmc) / um[_c].sum()
        #covs[_c] = np.linalg.det(covs[_c])**(1/data.shape[1]) *  np.linalg.inv(covs[_c])
        
#        dists[_c] = cdist(data, cntr, metric="mahalanobis", VI=np.linalg.inv(covs[_c]))[:, _c]
        dists[_c] = cdist(data, cntr, metric="mahalanobis", VI=np.linalg.det(covs[_c])**(1/data.shape[1])*np.linalg.inv(covs[_c]))[:, _c]
        

    jm = (um * dists ** 2).sum()

    u = dists ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return cntr, u, jm, dists, covs

def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.

    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.

    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.

    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


def gk(data, c, m, error, maxiter, init=None, seed=None):
    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 /= np.ones(
            (c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [cntr, u, Jjm, d, covs] = _gk0(data, u2, c, m)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)
    
    return cntr, u, u0, d, jm, p, fpc, covs


