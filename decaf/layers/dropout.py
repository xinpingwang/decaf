"""Implements the dropout layer."""
import numpy as np

from decaf.base import Layer, Blob
from decaf.layers import fillers


class DropoutLayer(Layer):
    """A Layer that implements the dropout."""

    def __init__(self, **kwargs):
        """
        Initializes a Dropout layer
        """
        Layer.__init__(self, **kwargs)
        self._ratio = self.spec['ratio']
        filler = fillers.DropoutFiller(ratio=self._ratio)
        self._mask = Blob(filler=filler)

    def forward(self, bottom, top):
        """Computes the forward pass."""
        # Get features and ouput
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype)
        self._mask.init_data(features.shape, np.bool)
        output[:] = features
        output *= self._mask.data()

    def backward(self, bottom, top, propagate_down):
        """Computes the backward pass."""
        if not propagate_down:
            return 0.
        top_diff = top[0].diff()
        bottom_diff = bottom[0].init_diff()
        bottom_diff[:] = top_diff
        bottom_diff *= self._mask.data()
        return 0.

    def update(self):
        """Dropout has nothing to update."""
        pass
