import logging
import numpy as np


class DecafError(Exception):
    pass


class InvalidLayerError(DecafError):
    """The error when an invalid spec is passed to a layer."""
    pass


class InvalidNetError(DecafError):
    """The error raised when the network does not pass validation."""
    pass


class Blob(object):
    """
    Blob is the data structure that holds a piece of numpy array as well as its gradient so that we can accumulate and
    pass around data more easily.

    We define two numpy matrices: one is data, which stores the data in the current blob; the other is diff (short for
    difference): when a network runs its forward and backward pass, diff will store the gradient value; when a solver
    goes through the blobs, diff will then be replaced with the value to update.

    The diff matrix will not be created unless you explicitly run init_diff, as many Blobs do not need the gradients
    to be computed.
    """

    def __init__(self, shape=None, dtype=np.float64, filler=None):
        self._data = None
        self._diff = None
        self._filler = filler
        if shape is not None:
            self.init_data(shape, dtype)

    def mirror(self, input_array, shape=None):
        # Create the data as a view of the input array. This is useful to save space and avoid duplication for data
        # layers.
        self._data = input_array.view()
        if shape is not None:
            self._data.shape = shape

    def has_data(self):
        """Checks if the blob has data."""
        return self._data is not None

    def data(self):
        """Returns a view of the data."""
        return self._data.view()

    def has_diff(self):
        """Checks if the blob has diff."""
        return self._diff is not None

    def diff(self):
        """Return a view of the diff."""
        return self._diff.view()

    def update(self):
        self._data += self._diff

    def resize(self, shape, dtype):
        if self._data.shape == shape and self._data.dtype == dtype:
            pass
        else:
            # Blob resize should not happen often. If it happens, we log it
            # so the user can know if multiple resize take place.
            logging.info('Blob resized to {0} dtype {1}'.format(str(shape), str(dtype)))
            self._data = np.zeros(shape, dtype=dtype)

    def init_data(self, shape, dtype=np.float64):
        """
        Initialize the data matrix if necessary. The filler will be always called even if no reallocation of data takes
        place.
        """
        if self.has_data() and self._data.shape == shape and self._data.dtype == dtype:
            self._data[:] = 0
        else:
            self._data = np.zeros(shape, dtype)
        if self._filler is not None:
            self._filler.fill(self._data)
        return self.data()

    def init_diff(self):
        """
        Initialize the diff in the same format as data

        Returns diff for easy access.
        """
        if not self.has_data():
            raise ValueError('The data should be initialized first!')
        if self.has_diff() and self._diff.shape == self._data.shape and self._diff.dtype == self._data.dtype:
            self._diff[:] = 0
        else:
            self._diff = np.zeros(self._data.shape, self._data.dtype)
        return self.diff()


class Layer(object):
    """
    A layer is the most basic component in decaf, It takes multiple blobs as its input, and produces its outputs as
    multiple blobs.

    When designing layers, always make sure that your code deals with mini-batches
    """

    def __init__(self, **kwargs):
        """
        Create a Layer.

        Necessary argument:
            name: the name of the layer.
        """
        self.spec = kwargs
        self.name = self.spec['name']
        self._param = []

    def forward(self, bottom, top):
        """
        Computes the forward pass.

        Input:
            bottom: the data at the bottom.
            top: the top-layer output.
        """
        raise NotImplementedError

    def backward(self, bottom, top, propagate_down):
        """
        Computes the backward pass.

        Input:
            bottom: the data at the bottom
            top: the data at top
            propagate_down: if set False, the gradient w.r.t. the bottom blobs does not need to be computed.
        Output:
            loss: the loss being generated in this layer. Note that if your layer does not generate any loss, you
            should still return 0.
        """
        raise NotImplementedError

    def update(self):
        """
        Updates my parameters based on the diff value given in the param blob.
        """
        raise NotImplementedError

    def param(self):
        """
        Returns the parameters in this layer. It should be a list of Blob objects.

        In out layer, either collect all your parameters into self._param list, or implement you own param() function.
        """
        return self._param


class DataLayer(Layer):
    """
    A Layer that generates data.
    """

    def forward(self, bottom, top):
        """
        Generates the data.

        Your data layer should override this function.
        """
        raise NotImplementedError

    def backward(self, bottom, top, propagate_down):
        """
        No gradient needs to be computed for data.

        You should not override this function.
        """
        raise DecafError("You should not reach this.")

    def update(self):
        """
        The data layer has no parameter, and the update() function should not be called.
        """
        pass


class LossLayer(Layer):
    """
    A Layer that implements loss. Usually, the forward pas of the loss does the actual computation of both the loss and
    the gradients, and the backward pass will simply return the loss value. The loss layer should not accept any blobs
    on its top.
    """

    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        self._loss = 0.

    def forward(self, bottom, top):
        raise NotImplementedError

    def backward(self, bottom, top, propagate_down):
        return self._loss

    def update(self):
        pass


class Solver(object):
    """
    This is the very basic form of the solver.
    """

    def __init__(self, **kwargs):
        self.spec = kwargs

    def solve(self, net):
        """
        The solve function takes a net as an input, and optimizes its parameters.
        """
        raise NotImplementedError


class Regularizer(object):
    """
    This is the class that implements the regularization terms.
    """

    def __init__(self, **kwargs):
        """
        Initializes a regularizer. A regularizer need a necessary keyword 'weight'
        """
        self.spec = kwargs
        self._weight = self.spec['weight']

    def reg(self, blob, num_data):
        """
        Compute the regularization term from the blob's data field, and add the regularization term to its diff directly
        """
        raise NotImplementedError


class Filler(object):
    """
    This is the class that implements util functions to fill a blob.

    A filler implements the fill() function that takes a blob as the input, and fills the blob's data() field.
    """

    def __init__(self, **kwargs):
        """
        simply get the spec.
        """
        self.spec = kwargs

    def fill(self, mat):
        raise NotImplementedError
