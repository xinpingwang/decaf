"""A code to perform logistic regression."""
import numpy as np

from decaf import net
from decaf.layers import core_layers, regularization
from decaf.optimization import core_solvers
from decaf.util import blasdot


def logistic_regression(features, target, reg_weight=0.):
    """
    Carry out a logistic regression given features and target value.

    If you actually want to do logistic regression, this is probably not what you want to use. It is here just for
    demonstration purpose
    """
    if target.ndim == 1:
        num_output = target.max() + 1
    else:
        num_output = target.shape[1]
    if features.shape[0] != target.shape[0]:
        raise ValueError('features and target should have the same number of data points!')
    # first, construct the network
    decaf_net = net.Net()
    # add data layer
    data_layer = core_layers.NdArrayDataLayer(name='data', sources=[features, target])
    decaf_net.add_layer(data_layer, provides=['features', 'target'])
    # add inner production layer
    ip_layer = core_layers.InnerProductLayer(name='ip', num_output=num_output,
                                             reg=regularization.L2Regularizer(weight=reg_weight))
    decaf_net.add_layer(ip_layer, needs=['features'], provides=['output'])
    # add loss layer
    loss_layer = core_layers.MultinomialLogisticLossLayer(name='loss')
    decaf_net.add_layer(loss_layer, needs=['output', 'target'])
    # finish
    decaf_net.finish()
    # now, try to solve it
    solver = core_solvers.LBFGSSolver(lbfgs_args={'iprint': 1})
    solver.solve(decaf_net)
    # We will violate the locality a little bit.
    param = ip_layer.param()
    return param[0].data().copy(), param[1].data().copy()


def main():
    """
    The main demo for the ridge regression problem.
    """
    np.random.seed(1701)
    data = np.random.randn(1000, 2)
    features = np.vstack((data + np.array([2, 2]), data - np.array([2, 2])))
    target = np.vstack(((np.ones((1000, 1))), np.zeros((1000, 1))))
    target = target.reshape(target.shape[0]).astype(np.int)
    weight, bias = logistic_regression(features, target, reg_weight=0.05)
    print('weight: {}'.format(weight))
    print('bias: {}'.format(bias))
    pred_out = blasdot.dot(features.reshape(features.shape[0], np.prod(features.shape[1:])), weight) + bias
    pred = pred_out.argmax(axis=1)
    print('Accuracy: {}'.format((pred == target).sum() / float(target.shape[0])))


if __name__ == '__main__':
    main()
