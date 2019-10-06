import numpy as np

from decaf.layers import mnist
from decaf.util import blasdot
from decaf.wraps import logistic_regression

ROOT_FOLDER = '../data/mnist'

dataset = mnist.MNISTDataLayer(name='mnist', root_folder=ROOT_FOLDER, is_training=True)
train_data = dataset._data
train_label = dataset._label

dataset = mnist.MNISTDataLayer(name='mnist', root_folder=ROOT_FOLDER, is_training=False)
test_data = dataset._data
test_label = dataset._label

weight, bias = logistic_regression.logistic_regression(train_data, train_label, reg_weight=0.01)

test_score = blasdot.dot(test_data.reshape(test_data.shape[0], np.prod(test_data.shape[1:])), weight)
test_score += bias

test_pred = test_score.argmax(axis=1)
print('Accuracy: {}'.format((test_pred == test_label).sum() / float(test_label.shape[0])))
