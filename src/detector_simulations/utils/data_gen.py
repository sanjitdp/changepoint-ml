import numpy as np
from typing import Tuple, Literal

TYPES_PARABOLA = ['uniform', 'normal']


def sample_ring_shaped(sample_size: int, scale: float = 0.1) -> np.ndarray:
    """
    Function for sampling data from the unit circle.
    (simply sample 2d-Gaussian and normalize it)

    Parameters
    ----------
        sample_size: int
            num of points to sample
        scale: float
            level of noise added to the circle radius
    Returns
    -------
        pts: array_like of size (n,2)
            array of resulting values
    """
    pts = np.random.multivariate_normal(
        mean=np.array([0, 0]), cov=np.eye(2), size=sample_size)
    pts /= np.linalg.norm(pts, axis=1).reshape(-1, 1)
    # add small perturbation
    pts *= (1+np.random.normal(0, scale, size=[sample_size, 1]))
    return pts


def sample_parabola(sample_size: int, underlying_dist: TYPES_PARABOLA = 'uniform', scale: float = 0.1) -> np.ndarray:
    """
    Function for sampling data from parabola and add some noise.

    Parameters
    ----------
        sample_size: int
            num of points to sample
        underlying_dist: ['uniform','normal']
            distribution of X
        scale: float
            level of noise added to the parabola
    Returns
    -------
        pts: array_like of size (n,2)
            array of resulting values
    """
    if underlying_dist == 'uniform':
        X = np.random.uniform(low=-1, high=1, size=[sample_size, 1])
    elif underlying_dist == 'normal':
        X = np.random.normal(size=[sample_size, 1])
    Y = X**2 + np.random.normal(0, scale, size=[sample_size, 1])
    return np.hstack([X, Y])


def sample_sphere(sample_size: int, dim: int = 10) -> np.ndarray:
    """
    Simulate data from the unit sphere and take first two coordinates as X and Y

    Parameters
    ----------
        sample_size: int
            num of points to sample
        dim: int
            dimension of a sphere

    """
    data = np.random.normal(size=[sample_size, dim])
    data /= np.linalg.norm(data, axis=1).reshape([sample_size, 1])
    X, Y = data[:, 0], data[:, 1]
    return np.hstack([X, Y])


def sample_gaussian(sample_size, num_nonzero=1, dim=10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate data from linear gaussian model (see the write-up)

    Parameters
    ----------
        sample_size: int
            num of points to sample
        num_nonzero: int
            number of nonzero coordinates for beta
        dim: int
            dimension of X (and beta)

    """
    beta = np.zeros(dim)
    for i in range(num_nonzero):
        beta[i] = (-1)**i
    X = np.random.normal(size=[sample_size, dim])
    eps = np.random.normal(size=sample_size)
    Y = X@beta+eps
    return X, Y
