"""Import commonly used layers"""

# Data Layers
from decaf.layers.ndarraydatalayer import NdArrayDataLayer

# Computation Layers
from decaf.layers.innerproduct import InnerProductLayer
from decaf.layers.loss import SquaredLossLayer, MultinomialLogisticLossLayer
