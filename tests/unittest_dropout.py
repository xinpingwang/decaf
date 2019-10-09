import numpy as np
import unittest

from decaf.base import Blob
from decaf.layers import dropout, fillers


class TestDropout(unittest.TestCase):
    def testdropoutlayer(self):
        layer = dropout.DropoutLayer(name='dropout', ratio=0.5)
        np.random.seed(1701)
        filler = fillers.RandFiller(min=1, max=2)
        bottom = Blob((100, 4), filler=filler)
        top = Blob()
        # run teh dropout layer
        layer.forward([bottom], [top])
        # simulate a diff
        fillers.RandFiller().fill(top.init_diff())
        layer.backward([bottom], [top], True)
        np.testing.assert_array_equal(top.data()[top.data() != 0],
                                      bottom.data()[top.data() != 0])
        np.testing.assert_array_equal(bottom.diff()[top.data() == 0],
                                      0)
        np.testing.assert_array_equal(bottom.diff()[top.data() != 0],
                                      top.diff()[top.data() != 0])


if __name__ == '__main__':
    unittest.main()
