"""Randomized iterative methods for decompositions.
"""

import numpy as np
import scipy.linalg as sla

from ..gen.rand import randn
from ..accel import dot, njit
# from .base_linalg import norm


def dag(X):
    try:
        return X.H
    except AttributeError:
        return X.conj().T


def lu_orthog(X):
    return sla.lu(X, permute_l=True, overwrite_a=False, check_finite=False)[0]


def qr_orthog(X):
    return sla.qr(X, mode='economic', overwrite_a=False, check_finite=False)[0]


def rchol(A, k, q=2):
    """Randomized partial Cholesky decomposition.
    """
    m, n = A.shape
    Q = randn((n, k), dtype=A.dtype)

    Q = dot(A, Q)
    Q = lu_orthog(Q)

    for i in range(1, q + 1):
        Q = dot(dag(A), Q)
        Q = lu_orthog(Q)

        Q = dot(A, Q)
        Q = lu_orthog(Q) if i < q else qr_orthog(Q)

    B1 = dot(A, Q)
    B2 = dot(dag(Q), B1)
    C = sla.cholesky(B2, overwrite_a=True, check_finite=False)
    F = sla.solve_triangular(dag(C), dag(B1), lower=True,
                             overwrite_b=True, check_finite=False)
    return F


def QB_to_svd(Q, B):
    U, s, V = np.linalg.svd(B, full_matrices=False)
    U = dot(Q, U)
    return U, s, V


# def _rsvd_iterate(A, k, q=2, QB=None):
#     m, n = A.shape

#     Q = randn((n, k), dtype=A.dtype)

#     if QB is not None:
#         Q0, B0 = QB

#     # power iterations with stabilization
#     Q = dot(A, Q)
#     if QB is not None:
#         Q = Q - Q0.dot(dag(Q0).dot(Q))
#     Q = lu_orthog(Q)

#     for i in range(1, q + 1):

#         Q = dot(dag(A), Q)
#         if QB is not None:
#             Q = Q - Q0.dot(dag(Q0).dot(Q))
#         Q = lu_orthog(Q)

#         Q = dot(A, Q)
#         if QB is not None:
#             Q = Q - Q0.dot(dag(Q0).dot(Q))
#         Q = lu_orthog(Q) if i < q else qr_orthog(Q)

#     B = dag(dot(dag(A), Q))

#     if QB is not None:
#         normB1 = norm(B, 'fro')
#         Q = np.concatenate((Q, Q0), axis=1)
#         B = np.concatenate((B, B0), axis=0)
#         return Q, B, normB1**2

#     return Q, B


# def rsvd(A, eps_or_k, q=2, dk=10, max_k=None):
#     if isinstance(eps_or_k, int):
#         Q, B = _rsvd_iterate(A, eps_or_k, q=q)
#         return QB_to_svd(Q, B)

#     if max_k is None:
#         max_k = min(A.shape)

#     # First iteration
#     Q, B = _rsvd_iterate(A, dk, q=q)
#     normB = norm(B, 'fro')**2
#     err = 1.0
#     QB = (Q, B)
#     rnk = dk

#     while (err > eps_or_k) and (rnk < max_k):
#         new_k = min(dk, max_k - rnk)
#         Q, B, normB1 = _rsvd_iterate(A, new_k, q=q, QB=QB)
#         normB += normB1
#         err = normB1 / normB
#         rnk += new_k
#         QB = (Q, B)
#         print(new_k, normB1, err, rnk)

#     return QB_to_svd(Q, B)


def _rsvd_iterate(A, k, q=2, UsV=None, p=0):
    if UsV in (None, 'G'):
        G = randn((A.shape[1], k + p), dtype=A.dtype)

        def maybe_project(X):
            return X
    else:
        U0, s0, V0, G = UsV

        def maybe_project(X):
            return X - U0.dot(dag(U0).dot(X))

    G = maybe_project(G)

    # power iterations with stabilization
    Q = dot(A, G)
    Q = maybe_project(Q)
    Q = lu_orthog(Q)

    for i in range(1, q + 1):
        Q = dot(dag(A), Q)
        Q = maybe_project(Q)
        Q = lu_orthog(Q)

        Q = dot(A, Q)
        Q = maybe_project(Q)
        Q = lu_orthog(Q) if i < q else qr_orthog(Q)

    B = dag(dot(dag(A), Q))
    U, s, V = QB_to_svd(Q, B)

    if UsV not in (None, 'G'):
        snrm = (s**2).sum()

        U = qr_orthog(maybe_project(U))
        U, s, V = U[:, :k], s[:k], V[:k, :]

        U = np.concatenate((U0, U), axis=1)
        V = np.concatenate((V0, V), axis=0)
        s = np.concatenate((s0, s))

        return U, s, V, snrm, G

    U, s, V = U[:, :k], s[:k], V[:k, :]

    if UsV == 'G':
        return U, s, V, G

    return U, s, V


@njit
def is_sorted(x):
    for i in range(x.size - 1):
        if x[i + 1] < x[i]:
            return False
    return True


def rsvd(A, eps_or_k, q=2, dk=10, max_k=None):
    """Fast randomized SVD, due to Halko. This scales as ``log(k)``
    rather than ``k`` so can be more efficient.

    Parameters
    ----------
    A : matrix_like, shape (m, n)
        The operator to decompose.
    k : int
        The number of singular values to target.
    q : int, optional
        The number of power iterations, increase for accuracy at the expense
        of runtime.
    dk : int, optional
        Stepsize when increasing rank adaptively.

    Returns
    -------
    U, array shape (m, k)
        Left singular vectors.
    s, array shape (k,)
        Singular values.
    V, array shape (k, n)
        Right singular vectors.
    """

    if isinstance(eps_or_k, int):
        return _rsvd_iterate(A, eps_or_k, q=q)

    if max_k is None:
        max_k = min(A.shape)

    # First iteration
    U, s, V, G = _rsvd_iterate(A, dk, q=q, UsV='G')

    # initial norm, error and rank
    normf, err, rnk = (s**2).sum(), 1.0, dk

    while (err > eps_or_k) and (rnk < max_k):
        # only step k as far as max_k
        new_k = min(dk, max_k - rnk)
        rnk += new_k

        # set current U, s, V and random sampler G
        UsV = U, s, V, G

        # Concatenate new U, s, V orthogonal to current U, s, V
        U, s, V, normf1, G = _rsvd_iterate(A, new_k, q=q, UsV=UsV)

        # check if the added norm has begun to get small
        normf += normf1
        err = normf1 / normf

    # make sure singular values always sorted in decreasing order
    if is_sorted(s):
        so = np.argsort(s)[::-1]
        U, s, V = U[:, so], s[so], V[so, :]

    return U, s, V
