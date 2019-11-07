"""Implements the im2col layer."""
import typing

from decaf.base import Layer, Blob
from decaf.layers.cpp import wrapper

import numpy as np


class Im2colLayer(Layer):
    """A layer that implements the im2col function."""

    def __init__(self, **kwargs):
        """
        Initializes an im2col layer

        kwargs:
            name: the name of the layer.
            psize: the patch size (patch will be a square).
            stride: the patch stride.

        If the input image has shape [height, width, nchannels], the output will have shape [(height-psize)/stride+1,
        (width-psize)/stride+1, nchannels*psize*psize].
        """
        Layer.__init__(self, **kwargs)
        self._psize: int = self.spec['psize']
        self._stride: int = self.spec['stride']
        if self._psize <= 1:
            raise ValueError('Padding should be larger than 1.')
        if self._stride < 1:
            raise ValueError('Stride should be larger than 0.')

    def _analyze_shape(self,
                       features: np.ndarray):
        num, height, width = features.shape[:3]
        channels = 1
        if features.ndim == 4:
            channels = features.shape[3]
        new_shape = (num,
                     (height - self._psize) / self._stride + 1,
                     (width - self._psize) / self._stride + 1,
                     channels * self._psize * self._psize)
        return num, height, width, channels, new_shape

    def forward(self,
                bottom: typing.List[Blob],
                top: typing.List[Blob]):
        """Computes the forward pass"""
        # Get features and output
        features = bottom[0].data()
        num, height, width, channels, new_shape = self._analyze_shape(features)
        output = top[0].init_data(new_shape, features.dtype)
        for i in range(num):
            wrapper.im2col(features[i], height, width, channels, self._psize, self._stride, output[i])

    def backward(self,
                 bottom: typing.List[Blob],
                 top: typing.List[Blob],
                 propagate_down: bool):
        """Compute the backward pass"""
        if not propagate_down:
            return 0.
        top_diff = top[0].diff()
        features = bottom[0].data()
        num, height, width, channels, new_shape = self._analyze_shape(features)
        bottom_diff = bottom[0].init_diff()
        for i in range(num):
            wrapper.col2im(bottom_diff[i], height, width, channels, self._psize, self._stride, top_diff[i])
        return 0.

    def update(self):
        """Im2col has nothing to update."""
        pass
