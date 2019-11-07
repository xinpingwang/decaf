import numpy as np
import typing
from scipy.linalg import blas


def _gemm_f_contiguous(alpha: float,
                       A: np.ndarray,
                       B: np.ndarray,
                       out: np.ndarray):
    """
    A gemm function that uses scipy fblas functions, avoiding matrix copy when the input is transposed.

    The returned matrix is designed to be  F_CONTIGUOUS
    """
    # scipy_gemm = linalg.get_blas_funcs('gemm', arrays=(A, B))
    if out.shape != (A.shape[0], B.shape[1]) or out.dtype != A.dtype or not out.flags.f_contiguous:
        raise ValueError('Incorrect output data type.')
    if A.dtype != B.dtype:
        raise TypeError('The data type of the matrices should be the same')
    if A.dtype == np.float32:
        gemm = blas.sgemm
    elif A.dtype == np.float64:
        gemm = blas.dgemm
    else:
        raise TypeError('Unfit data type.')
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrices are not aligned")
    if A.flags.c_contiguous and B.flags.c_contiguous:
        gemm(alpha, a=A.T, b=B.T, trans_a=True, trans_b=True, c=out, overwrite_c=True)
    elif A.flags.c_contiguous and B.flags.f_contiguous:
        gemm(alpha, a=A.T, b=B, trans_a=True, c=out, overwrite_c=True)
    elif A.flags.f_contiguous and B.flags.c_contiguous:
        gemm(alpha, a=A, b=B.T, trans_b=True, c=out, overwrite_c=True)
    elif A.flags.f_contiguous and B.flags.f_contiguous:
        gemm(alpha, a=A, b=B, c=out, overwrite_c=True)
    else:
        raise ValueError('Incorrect matrix flags.')
    return out


def _gemm_c_contiguous(alpha: float,
                       A: np.ndarray,
                       B: np.ndarray,
                       out: np.ndarray):
    """
    A wrapper that computes C_CONTIGUOUS gemm results.
    """
    _gemm_f_contiguous(alpha, B.T, A.T, out=out.T)
    return out


def dot(A: np.ndarray,
        B: np.ndarray,
        out: typing.Optional[np.ndarray] = None):
    """
    A simple wrapper that mimics np.dot (if A and B are both matrices!). This function solves the problem that np.dot
    copies matrices when working on transposed matrices.

    Input:
        A, B: two matrices. should be either c-contiguous or f-contiguous
        out: (optional) the output matrix. If it is passed, the matrix should have the tighe shape and should be
        C_CONTIGUOUS.
    Output:
        out: the output matrix
    Raises:
        TypeError: if the type of matrices is wrong.
    """
    if out is None:
        out = np.empty((A.shape[0], B.shape[1]), max(A.dtype, B.dtype))
    out = _gemm_c_contiguous(1.0, A, B, out=out)
    return out
