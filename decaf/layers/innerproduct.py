from decaf.base import Layer, Blob
from decaf.base import InvalidSpecError
from decaf.util import blasdot
import numpy as np


class InnerProductLayer(Layer):
    def __init__(self, **kwargs):
        """
        Initializes an inner product layer. You need to specify the kwarg 'num_output' as the number of output nodes.
        Optionally, pass in a regularizer with keyword 'reg' will add regularization terms to the weight (but not bias).
        """
        Layer.__init__(self, **kwargs)
        self._num_output = self.spec.get('num_output', 0)
        if self._num_output <= 0:
            raise InvalidSpecError('Incorrect ou unspecified num_output for {}'.format(self.name))
        self._weight = Blob()
        self._reg = self.spec.get('reg', None)
        self._has_bias = self.spec.get('bias', True)
        if self._has_bias:
            self._bias = Blob()
            self._param = [self._weight, self._bias]
        else:
            self._param = [self._weight]

    def forward(self, bottom, top):
        """Computes the forward pass"""
        features = bottom[0].data()
        if features.ndim > 2:
            features.shape = (features.shape[0], np.prod(features.shape[1:]))
            
        output = top[0].init_data((features.shape[0], self._num_output), features.dtype)
        # initialize weights and bias
        if not self._weight.has_data():
            self._weight.init_data((features.shape[1], self._num_output), features.dtype)
        if self._has_bias and not self._bias.has_data():
            self._bias.init_data(self._num_output, features.dtype)
        # computation
        weight = self._weight.data()
        blasdot.dot(features, weight, out=output)
        if self._has_bias:
            output += self._bias.data()
        return 0.

    def backward(self, bottom, top, need_bottom_diff):
        """Computes the backward pass."""
        top_diff = top[0].diff()
        features = bottom[0].data()
        if features.ndim > 2:
            features.shape = (features.shape[0], np.prod(features.shape[1:]))

        # compute the gradient
        weight_diff = self._weight.init_diff()
        blasdot.dot(features.T, top_diff, out=weight_diff)
        if self._has_bias:
            bias_diff = self._bias.init_diff()
            bias_diff[:] = top_diff.sum(0)
        # if necessary, compute the bottom Blob gradient
        if need_bottom_diff:
            bottom_diff = bottom[0].init_diff()
            if bottom_diff.ndim > 2:
                bottom_diff.shape = (bottom_diff.shape[0], np.prod(bottom_diff.shape[1:]))
            blasdot.dot(top_diff, weight_diff.T, out=bottom_diff)
        if self._reg is not None:
            return self._reg.reg(self._weight)
        else:
            return 0.

    def update(self):
        """Updates the parameters"""
        self._weight.update()
        if self._has_bias:
            self._bias.update()
