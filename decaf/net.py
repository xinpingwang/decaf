from collections import defaultdict
from decaf.base import DecafError, Blob
import networkx as nx


class InvalidNetworkError(DecafError):
    """
    The error raised when the network does not pass validation
    """
    pass


class Net(object):
    """
    A Net is a directed graph with layer names and layer instances.
    """

    def __init__(self):
        self._graph = nx.DiGraph()
        self._blobs = defaultdict(Blob)
        self._layers = {}
        self._needs = {}
        self._provides = {}
        # The topological order to execute the layer.
        self._forward_order = None
        self._backward_order = None
        self_params = None
        self._finished = False

    def add_layer(self, layer, needs=[], provides=[]):
        """
        Add a layer to the current network.

        Input:
            layer: a decaf.base.Layer instance.
            needs: a tuple of strings, indicating the blobs that the layer needs as its input.
            provides: similar to needs, but the layer's output instead.
        """
        if type(needs) is str:
            needs = [needs]
        if type(provides) is str:
            provides = [provides]
        if self._finished:
            # Trying to modify an already finished network.
            raise DecafError('Modifying an already finished net.')
        # Add the layer
        if layer.name in self._layers:
            raise InvalidNetworkError('Duplicated layer found: {0}'.format(layer.name))
        if layer.name in self._blobs:
            raise InvalidNetworkError('Layer name found as a blob: {0}'.format(layer.name))
        self._layers[layer.name] = layer
        # Add the blobs
        for blob_name in needs:
            if blob_name in self._layers:
                raise InvalidNetworkError('Blob name found as a layer {0}'.format(blob_name))
        for blob_name in provides:
            if blob_name in self._layers:
                raise InvalidNetworkError('Blob name found as a layer {0}'.format(blob_name))
        self._needs[layer.name] = [self._blobs[blob_name] for blob_name in needs]
        self._provides[layer.name] = [self._blobs[blob_name] for blob_name in provides]
        # create the graph structure
        for blob_name in needs:
            self._graph.add_edge(blob_name, layer.name)
        for blob_name in provides:
            self._graph.add_edge(layer.name, blob_name)

    def finish(self):
        """
        Call this function when you finish the network construction.
        """
        # validate.
        self._validate()
        topological_order = nx.topological_sort(self._graph)
        # For efficiency reasons, we will see for each layer, whether the backward operation needs to be carried out.
        # This is stored in two parameters:
        #   need_backward: whether the backward pass needs to be carried out
        #   need_bottom_diff: whether the gradient w.r.t. to be bottom layer needs to be carried out.
        for name in topological_order:
            pred_need_backward = any(self._graph.nodes[p]['need_backward'] for p in self._graph.predecessors(name))

            if name in self._layers:
                # see if a layer needs backward operation. A layer needs backward operation if
                # (1) it has parameters, or
                # (2) any of its predecessors needs backward operation.
                if self._layers[name].param() or pred_need_backward:
                    self._graph.nodes[name]['need_backward'] = True
                else:
                    self._graph.nodes[name]['need_backward'] = False
                # See if a layer needs to compute its bottom diff. A layer need to compute its bottom diff if any of its
                # predecessors needs backward operation.
                if pred_need_backward:
                    self._graph.nodes[name]['need_bottom_diff'] = True
                else:
                    self._graph.nodes[name]['need_bottom_diff'] = False
            else:
                # see if a blob needs backward operation. This is only used so we can verify further layer.
                self._graph.nodes[name]['need_backward'] = pred_need_backward
        # create the order to run forward and backward passes
        # topological_order is a iterator, after the above loop, it should be reinitialize
        topological_order = nx.topological_sort(self._graph)
        layer_order = [layer_name for layer_name in topological_order if layer_name in self._layers]
        self._forward_order = [(n, self._layers[n], self._needs[n], self._provides[n]) for n in layer_order]
        self._backward_order = [(n, self._layers[n], self._needs[n], self._provides[n],
                                 self._graph.nodes[n]['need_bottom_diff'])
                                for n in layer_order[::-1] if self._graph.nodes[n]['need_backward']]
        # store all the parameters
        self._params = []
        for name in layer_order:
            self._params.extend(self._layers[name].param())
        # Note: Any further finishing code should be inserted here.
        self._finished = True

    def params(self):
        """
        Return a list of parameters used in the network.
        """
        return self._params

    def _validate(self):
        """
        Validated if a network is executable. A net word being executable means that every blob node has a layer as its
        predecessor, and no loop exists in the network.
        """
        if not nx.is_directed_acyclic_graph(self._graph):
            raise InvalidNetworkError('The network is not a DAG')
        for blob_name in self._blobs:
            # check if every blob has predecessors, and each predecessor is a valid layer.
            predecessors = list(self._graph.predecessors(blob_name))
            if len(predecessors) != 1:
                raise InvalidNetworkError('Blob {} has no source layer or multiple source layers.'.format(blob_name))
            if predecessors[0] not in self._layers:
                raise InvalidNetworkError('Blob {} has a source that is not a layer.'.format(blob_name))
            successors = list(self._graph.successors(blob_name))
            if len(successors) > 1:
                raise InvalidNetworkError('Blob {} has multiple successors.'.format(blob_name))

        return True

    def execute(self):
        """Execute one round of the networkx."""
        # the forward pass. we will also accumulate the loss function
        loss = 0.
        for _, layer, bottom, top in self._forward_order:
            loss += layer.forward(bottom, top)
        # the backward pass
        for _, layer, bottom, top, need_bottom_diff in self._backward_order:
            layer.backward(bottom, top, need_bottom_diff)
        return loss

    def update(self):
        """
        Update the parameters using the diff values provided in the parameters blob.
        """
        for _, layer, _, _ in self._forward_order:
            layer.update()
