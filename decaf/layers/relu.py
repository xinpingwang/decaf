import numpy as np
import typing

from decaf import base
from decaf.base import Blob


class ReLULayer(base.Layer):
    """
    A layer that implements the ReLU activate
    """

    def __init__(self, **kwargs):
        """
        Initializes a ReLU layer.
        """
        base.Layer.__init__(self, **kwargs)

    def forward(self,
                bottom: typing.List[Blob],
                top: typing.List[Blob]):
        """
        Compute the forward pass.
        """
        features = bottom[0].data()
        output = top[0].init_data(features.shape, features.dtype)
        output[:] = features
        output *= (features > 0)

    def backward(self,
                 bottom: typing.List[Blob],
                 top: typing.List[Blob],
                 propagate_down: bool):
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
