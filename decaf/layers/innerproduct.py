from decaf.base import Layer, Blob
from decaf.base import InvalidSpecError
from decaf.util import blasdot
import numpy as np


class InnerProductLayer(Layer):
    def __init__(self, **kwargs):
        """
        Initializes an inner product layer. You need to specify the kwarg 'num_output' as the number of output nodes.
        """
        Layer.__init__(self, **kwargs)
        self._num_output = self.spec.get('num_output', 0)
        if self._num_output <= 0:
            raise InvalidSpecError('Incorrect ou unspecified num_output for {}'.format(self.name))
        self._weight = Blob()
        self._has_bias = self.spec.get('bias', True)
        if self._has_bias:
            self._bias = Blob()
            self._param = [self._weight, self._bias]
        else:
            self._param = [self._weight]

    def forward(self, bottom, top):
        """Computes the forward pass"""
        bottom_data = bottom[0].data.view()
        bottom_data.shape = (bottom_data.shape[0], np.prod(bottom_data.shape[1:]))
        top[0].init_data((bottom_data.shape[0], self._num_output), bottom_data.dtype)
        top_data = top[0].data
        if not self._weight.has_data():
            self._weight.init_data((bottom_data.shape[1], self._num_output), bottom_data.dtype)
        if self._has_bias and not self._bias.has_data():
            self._bias.init_data((bottom_data.shape[1], self._num_output), bottom_data.dtype)
        self._weight.resize((bottom_data.shape[1], self._num_output), bottom_data.dtype)
        blasdot.dot(bottom_data, self._weight.data, out=top_data)
        if self._has_bias:
            self._bias.resize(self._num_output, bottom_data.dtype)
            top_data += self._bias.data
        return 0.

    def backward(self, bottom, top, need_bottom_diff):
        """Computes the backward pass."""
        top_diff = top[0].diff.view()
        bottom_data = bottom[0].data.view()
        bottom_data.shape = (bottom_data.shape[0], np.prod(bottom_data.shape[1:]))

        self._weight.init_diff()
        blasdot.dot(bottom_data.T, top_diff, out=self._weight.diff)
        if self._has_bias:
            self._bias.init_diff()
            self._bias.diff[:] = top_diff.sum(0)
        if need_bottom_diff:
            bottom[0].init_diff()
            bottom_diff = bottom[0].diff.view()
            bottom_diff.shape = (bottom_diff.shape[0], np.prod(bottom_diff.shape[1:]))
            blasdot.dot(top_diff, self._weight.diff.T, out=bottom_diff)

    def update(self):
        """Updates the parameters"""
        self._weight.data += self._weight.diff
        if self._has_bias:
            self._bias.data += self._bias.diff
