import numpy as np
import typing


def exp(mat: np.ndarray,
        out: typing.Optional[np.ndarray] = None):
    """
    A (hacky) safe exp that avoids overflowing

    Input:
        mat: the input ndarray
        out: (optional) the output ndarray. Could be in-place.
    Output:
        out: the output ndarray
    """
    if out is None:
        out = np.empty_like(mat)
    np.clip(mat, -np.inf, 100, out=out)
    np.exp(out, out=out)
    return out


def log(mat: np.ndarray,
        out: typing.Optional[np.ndarray] = None):
    """
    A (hacky) safe log that avoid nans

    Note that if there are negative values in the input, this function does not throw an error. Handle these cases with
    care.
    """
    if out is None:
        out = np.empty_like(mat)
    # finfo Machine limits for floating point types.
    np.clip(mat, np.finfo(mat.dtype).tiny, np.inf, out=out)
    np.log(mat, out=out)
    return out
