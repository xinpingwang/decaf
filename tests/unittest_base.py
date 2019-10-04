import numpy as np
import numpy.testing as npt
import unittest

from decaf.base import Blob


class TestBlob(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def testBlobInit(self):
        blob = Blob()
        self.assertFalse(blob.has_data())
        self.assertFalse(blob.has_diff())
        blob = Blob((1, 1))
        self.assertTrue(blob.has_data())
        self.assertFalse(blob.has_diff())
        self.assertEqual(blob.data().shape, (1, 1))

    def testBlobUpdate(self):
        blob = Blob((4, 3))
        diff = blob.init_diff()
        diff[:] = 1.
        blob.update()
        npt.assert_almost_equal(blob.data(), blob.diff())

    def testUseBlob(self):
        blob_a = Blob((4, 3))
        blob_b = Blob((3, 4))
        output = np.dot(blob_a.data(), blob_b.data())
        self.assertEqual(output.shape, (4, 4))
        blob_c = Blob((4, 4))
        output = np.dot(blob_a.data().T, blob_c.data())
        self.assertEqual(output.shape, (3, 4))


if __name__ == '__main__':
    unittest.main()
