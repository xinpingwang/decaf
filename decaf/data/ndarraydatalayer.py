from decaf.base import DataLayer


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
        self._sources = self.spec['sources']

    def forward(self, bottom, top):
        """
        Generates the data.
        """
        for top_blob, sources in zip(top, self._sources):
            top_blob.mirror(sources)
        return 0.
