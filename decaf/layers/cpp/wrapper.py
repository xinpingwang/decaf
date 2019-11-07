"""This folder contains some c++ implementations that either make code run faster or handles some numpy tricky issues.
"""
import ctypes as ct
import numpy as np
import os

# first, let's import the library
try:
    _cpp_util = np.ctypeslib.load_library('libim2col',
                                          os.path.join(os.path.dirname(__file__)))
except Exception as e:
    raise RuntimeError('I cannot load libcpputil.so. Please compile first')

################################################################################
# im2col operation
################################################################################
_cpp_util.im2col_float.restype = None
_cpp_util.im2col_float.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   np.ctypeslib.ndpointer(dtype=np.float32, flags='C')]

_cpp_util.im2col_double.restype = None
_cpp_util.im2col_double.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
                                    ct.c_int,
                                    ct.c_int,
                                    ct.c_int,
                                    ct.c_int,
                                    ct.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64, flags='C')]


def im2col(*args):
    """A wrapper of the im2col function."""
    if args[0].dtype == np.float32:
        return _cpp_util.im2col_float(*args)
    elif args[0].dtype == np.float64:
        return _cpp_util.im2col_float(*args)
    else:
        raise TypeError('Unsupported type: {}'.format(args[0].dtype))


###############################################################################
# col2im operation
################################################################################
_cpp_util.col2im_float.restype = None
_cpp_util.col2im_float.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, flags='C'),
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   ct.c_int,
                                   np.ctypeslib.ndpointer(dtype=np.float32, flags='C')]

_cpp_util.col2im_double.restype = None
_cpp_util.col2im_double.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
                                    ct.c_int,
                                    ct.c_int,
                                    ct.c_int,
                                    ct.c_int,
                                    ct.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64, flags='C')]


def col2im(*args):
    """A wrapper of the im2col function."""
    if args[0].dtype == np.float32:
        return _cpp_util.col2im_float(*args)
    elif args[0].dtype == np.float64:
        return _cpp_util.col2im_float(*args)
    else:
        raise TypeError('Unsupported type: {}'.format(args[0].dtype))
