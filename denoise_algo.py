from scipy import sparse
from scipy.sparse import linalg as slin
import numpy as np


EPS = 1E-10                # smoothed penalty function
def psi(x): return np.sqrt(x**2 + EPS)



def cgtv(noisy_signal, alpha, beta, Nit, denoised_signal=None):
    """
    run the algorithm

    noisy_signal: original signal
    alpha: denoising parameter
    beta: denoising parameter
    Nit: number of iteration
    denoised_signal: a previously denoised signal
    """

    N = len(noisy_signal)
    e = np.ones(N)
    D1 = sparse.spdiags([-e, e], [0, 1], N-1, N)
    D3 = sparse.spdiags([-e, 3*e, -3*e, e], [0, 1, 2, 3], N-3, N)
    I = sparse.spdiags(e, 0, N, N)

    # if previously denoised signal exists, continue denosing on that signal
    if denoised_signal is None:
        x = noisy_signal
    else:
        x = denoised_signal

    # run algorithm (check paper for detail)
    for i in range(Nit):
        Lam1 = sparse.spdiags(alpha/psi(np.diff(x)), 0, N-1, N-1)
        Lam3 = sparse.spdiags(beta/psi(np.diff(x, 3)), 0, N-3, N-3)
        temp = I + ((D1.T).dot(Lam1)).dot(D1) + ((D3.T).dot(Lam3)).dot(D3)
        x = slin.spsolve(temp, noisy_signal)
    return x
