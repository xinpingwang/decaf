"""Implements basic regularizers"""
from decaf.base import Regularizer
import numpy as np


class L1Regularizer(Regularizer):
    def reg(self, blob):
        blob.diff += self._weight * np.sign(blob.data)
        return np.abs(blob.data)


class L2Regularizer(Regularizer):
    def reg(self, blob):
        blob.diff += self._weight * 2. * blob.data
        return np.dot(blob.data.flat, blob.data.flat) * self._weight
