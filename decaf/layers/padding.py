"""Implements the padding layer."""
import typing

from decaf.base import Layer, Blob


class PaddingLayer(Layer):
    """A Layer that pads a matrix."""

    def __init__(self, **kwargs):
        """
        Initializes a padding layer.
        kwargs:
            'pad': the number of pixels to pad, Should be non-negative. If pad is 0, the layer will simply mirror the
            input.
            'value': the value inserted to the padded area. Default 0.
        """
        Layer.__init__(self, **kwargs)
        self._pad: int = self.spec['pad']
        self._value: float = self.spec.get('value', 0)
        if self._pad < 0:
            raise ValueError('Padding should be non-negative.')

    def forward(self,
                bottom: typing.List[Blob],
                top: typing.List[Blob]):
        """Computes the forward pass."""
        if self._pad == 0:
            top[0].mirror(bottom[0].data())
            return
        features = bottom[0].data()
        pad = self._pad
        new_shape = (features.shape[0],
                     features.shape[1] + pad * 2,
                     features.shape[2] + pad * 2) + features.shape[3:]
        output = top[0].init_data(new_shape, features.dtype)
        output[:] = self._value
        output[:, pad:-pad, pad:-pad] = features

    def backward(self,
                 bottom: typing.List[Blob],
                 top: typing.List[Blob],
                 propagate_down: bool):
        """Computes the backward pass."""
        if not propagate_down:
            return 0.
        if self._pad == 0:
            bottom[0].mirror_diff(top[0].diff())
        else:
            pad = self._pad
            top_diff = top[0].diff()
            bottom_diff = bottom[0].init_diff()
            bottom_diff[:] = top_diff[:, pad:-pad, pad:-pad]
        return 0.

    def update(self):
        """Padding has nothing to update."""
        pass
