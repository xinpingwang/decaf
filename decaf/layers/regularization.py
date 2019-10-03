"""Implements basic regularizers"""
from decaf.base import Regularizer
import numpy as np


class L1Regularizer(Regularizer):
    def reg(self, blob):
        data = blob.data()
        diff = blob.diff()
        diff += self._weight * np.sign(data)
        return np.abs(data)


class L2Regularizer(Regularizer):
    def reg(self, blob):
        data = blob.data()
        diff = blob.diff()
        diff += self._weight * 2. * data
        return np.dot(data.flat, data.flat) * self._weight
