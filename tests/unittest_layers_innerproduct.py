import unittest
import numpy as np

from decaf.base import Blob
from decaf.layers import innerproduct


class TestLayerInnerproduct(unittest.TestCase):
    """
    Test the InnerProductLayer module
    """

    def setUp(self) -> None:
        self.test_sizes = [(10, 5), (1, 1), (10, 1), (1, 5)]
        self.test_output_sizes = [1, 5, 10]
        self.test_blobs = [Blob(size, np.float32) for size in self.test_sizes]
        self.test_blobs += [Blob(size, np.float64) for size in self.test_sizes]

    def testForwardBackwardSize(self):
        for blob in self.test_blobs:
            for num_output in self.test_output_sizes:
                # test forward
                top_blob = Blob()
                decaf_layer = innerproduct.InnerProductLayer(name='ip', num_output=num_output)
                decaf_layer.forward([blob], [top_blob])
                self.assertTrue(top_blob.has_data())
                self.assertEqual(top_blob.data().shape[0], blob.data().shape[0])
                self.assertEqual(top_blob.data().shape[1], num_output)
                # test backward
                top_diff = top_blob.init_diff()
                top_diff[:] = 1.
                decaf_layer.backward([blob], [top_blob], propagate_down=True)
                self.assertTrue(blob.has_diff())
                self.assertEqual(blob.diff().shape, blob.data().shape)


if __name__ == '__main__':
    unittest.main()
