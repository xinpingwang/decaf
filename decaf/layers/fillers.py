"""Implements basic fillers."""
import numpy as np

from decaf.base import Filler


class ConstantFiller(Filler):
    """
    Fills the values with a constant value.

    specs: value
    """

    def fill(self, mat):
        mat[:] = self.spec['value']


class RandFiller(Filler):
    """
    Fills the values with random numbers in [min, max]

    specs: min, max
    """

    def fill(self, mat):
        min = self.spec.get('min', 0)
        max = self.spec.get('max', 1)
        mat[:] = np.random.random_sample(mat.shape)
        mat *= max - min
        mat += min


class GaussianRandFiller(Filler):
    """
    Fill the values with random gaussian.

    specs: mean, std.
    """

    def fill(self, mat):
        mean = self.spec.get('mean', 0.)
        std = self.spec.get('std', 1.)
        mat[:] = np.random.standard_normal(mat.shape)
        mat *= std
        mat += mean


class DropoutFiller(Filler):
    """
    Fill the values with boolean.

    specs: ratio
    """

    def fill(self, mat):
        ratio = self.spec['ratio']
        mat[:] = np.random.random_sample(mat.shape) < ratio
