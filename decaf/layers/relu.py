import numpy as np
from decaf import base


class ReLULayer(base.Layer):
    """
    A layer that implements the ReLU activate
    """

    def __init__(self, **kwargs):
        """
        Initializes a ReLU layer.
        """
        base.Layer.__init__(self, **kwargs)

    def forward(self, bottom, top):
        """
        Compute the forward pass.
        """
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype)
        output[:] = features
        output *= (features > 0)

    def backward(self, bottom, top, propagate_down):
        """
        Compute the backward pass.
        """
        if not propagate_down:
            return 0.
        top_diff = top[0].diff()
        features = bottom[0].data()
        bottom_diff = bottom[0].init_diff()
        bottom_diff[:] = top_diff
        bottom_diff *= (features > 0)
        return 0.

    def update(self):
        """
        ReLU has nothing to update
        """
        pass
