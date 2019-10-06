import numpy as np
import os

from decaf.layers.ndarraydatalayer import NdArrayDataLayer


class MNISTDataLayer(NdArrayDataLayer):
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    IMAGE_DIM = (28, 28)

    def __init__(self, **kwargs):
        """
        Initialize the mnist dataset
        """
        is_training = kwargs.get('is_training', True)
        root_folder = kwargs['root_folder']
        dtype = kwargs.get('dtype', np.float64)
        self._load_mnist(root_folder, is_training, dtype)
        # normalize data.
        # self._data /= 255.
        NdArrayDataLayer.__init__(self, sources=[self._data, self._label], **kwargs)

    def _load_mnist(self, root_folder, is_training, dtype):
        if is_training:
            self._data = self._read_byte_data(
                os.path.join(root_folder, 'train-images-idx3-ubyte'),
                16, (MNISTDataLayer.NUM_TRAIN,) + MNISTDataLayer.IMAGE_DIM
            ).astype(dtype)
            self._label = self._read_byte_data(
                os.path.join(root_folder, 'train-labels-idx1-ubyte'),
                8, [MNISTDataLayer.NUM_TRAIN]
            ).astype(np.int)
        else:
            self._data = self._read_byte_data(
                os.path.join(root_folder, 't10k-images-idx3-ubyte'),
                16, (MNISTDataLayer.NUM_TEST,) + MNISTDataLayer.IMAGE_DIM
            ).astype(dtype)
            self._label = self._read_byte_data(
                os.path.join(root_folder, 't10k-labels-idx1-ubyte'),
                8, [MNISTDataLayer.NUM_TEST]
            ).astype(np.int)

    def _read_byte_data(self, filename, skip_bytes, shape):
        """
        the image data in mnist dataset start with 16 bytes description and the label data start with 8 bytes
        description.
        """
        nbytes = np.prod(shape)
        with open(filename, 'rb') as fid:
            fid.seek(skip_bytes)
            raw_data = fid.read(nbytes)
        data = np.zeros(nbytes)
        for i in range(nbytes):
            data[i] = raw_data[i]
        data.resize(shape)
        return data
