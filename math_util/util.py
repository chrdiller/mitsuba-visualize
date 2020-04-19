
import numpy as np


def find_closest_orthogonal_matrix(A: np.ndarray) -> np.ndarray:
    """
    Find closest orthogonal matrix to *A* using iterative method.
    Based on the code from REMOVE_SOURCE_LEAKAGE function from OSL Matlab package.
     Reading:
        Colclough GL et al., A symmetric multivariate leakage correction for MEG connectomes.,
                    Neuroimage. 2015 Aug 15;117:439-48. doi: 10.1016/j.neuroimage.2015.03.071

    :param A: array shaped k, n, where k is number of channels, n - data points
    :return: Orthogonalized matrix with amplitudes preserved
    """
    # Code from https://gist.github.com/dokato/7a997b2a94a0ec6384a5fd0e91e45f8b
    MAX_ITER = 2000
    TOLERANCE = np.max((1, np.max(A.shape) * np.linalg.svd(A.T, False, False)[0])) * np.finfo(A.dtype).eps
    reldiff = lambda a, b: 2 * abs(a - b) / (abs(a) + abs(b))
    convergence = lambda rho, prev_rho: reldiff(rho, prev_rho) <= TOLERANCE

    A_b = A.conj()
    d = np.sqrt(np.sum(A * A_b, axis=1))

    rhos = np.zeros(MAX_ITER)

    for i in range(MAX_ITER):
        scA = A.T * d
        u, s, vh = np.linalg.svd(scA, False)
        V = np.dot(u, vh)
        d = np.sum(A_b * V.T, axis=1)

        L = (V * d).T
        E = A - L
        rhos[i] = np.sqrt(np.sum(E * E.conj()))
        if i > 0 and convergence(rhos[i], rhos[i - 1]):
            break

    return L


if __name__ == "__main__":
    raise NotImplementedError("Cannot call this math_util script directly")
