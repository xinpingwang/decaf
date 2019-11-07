"""Implements basic fillers."""
import numpy as np

from decaf.base import Filler


class ConstantFiller(Filler):
    """
    Fills the values with a constant value.

    kwargs:
        value: the constant value to fill
    """

    def fill(self,
             mat: np.ndarray):
        mat[:] = self.spec['value']


class RandFiller(Filler):
    """
    Fills the values with random numbers in [min, max]

    kwargs:
        min_val: the min value (default 0)
        max_val: the max value (default 1)
    """

    def fill(self,
             mat: np.ndarray):
        min_val = self.spec.get('min', 0)
        max_val = self.spec.get('max', 1)
        mat[:] = np.random.random_sample(mat.shape)
        mat *= max_val - min_val
        mat += min_val


class GaussianRandFiller(Filler):
    """
    Fill the values with random gaussian.

    kwargs:
        mean: the mean value (default 0).
        std: the standard deviation (default 1).
    """

    def fill(self,
             mat: np.ndarray):
        mean = self.spec.get('mean', 0.)
        std = self.spec.get('std', 1.)
        mat[:] = np.random.standard_normal(mat.shape)
        mat *= std
        mat += mean


class DropoutFiller(Filler):
    """
    Fill the values with boolean.

    kwargs:
        ratio: the ratio of 1 values when generating random binaries.
    """

    def fill(self,
             mat: np.ndarray):
        ratio = self.spec['ratio']
        mat[:] = np.random.random_sample(mat.shape) < ratio
