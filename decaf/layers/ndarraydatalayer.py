import typing
import numpy as np

from decaf.base import DataLayer, Blob


class NdArrayDataLayer(DataLayer):
    """
    This layer takes a bunch of data as a dictionary, and then emits them as Blobs.
    """

    def __init__(self, **kwargs):
        """
        Initialize the data layer. The input matrices will be provided by keyword 'sources' as a list of NdArrays, like
            sources = [array_1, array_2]

        The number of arrays should be identical to the number of output blobs.
        """
        DataLayer.__init__(self, **kwargs)
        self._sources: typing.List[np.ndarray] = self.spec['sources']

    def forward(self,
                bottom: typing.List[Blob],
                top: typing.List[Blob]):
        """
        Generates the data.
        """
        if len(top) != len(self._sources):
            raise ValueError('The number of sources and output blobs should be the same')
        for top_blob, sources in zip(top, self._sources):
            top_blob.mirror(sources)
