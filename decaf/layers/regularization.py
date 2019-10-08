"""Implements basic regularizers"""
from decaf.base import Regularizer
import numpy as np


class L1Regularizer(Regularizer):
    def reg(self, blob, num_data):
        data = blob.data()
        diff = blob.diff()
        diff += self._weight * num_data * np.sign(data)
        return np.abs(data).sum()


class L2Regularizer(Regularizer):
    def reg(self, blob, num_data):
        data = blob.data()
        diff = blob.diff()
        diff += self._weight * num_data * 2. * data
        return np.dot(data.flat, data.flat) * self._weight
