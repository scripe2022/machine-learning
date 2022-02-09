import numpy as np
from nnlib.propagation import propagation
from nnlib.sl import load_parameters
from nnlib.dataset import load_dataset, flatten, normalization
import matplotlib.pyplot as plt

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, label = load_dataset(
    "datasets/train.h5", "datasets/test.h5"
)
print(test_set_x_orig.shape)
train_set_x_flatten = flatten(train_set_x_orig)
train_set_x = normalization(train_set_x_flatten)
test_set_x_flatten = flatten(test_set_x_orig)
test_set_x = normalization(test_set_x_flatten)

L, w, b = load_parameters("parameters.h5")

a_train = propagation(train_set_x, train_set_y, w, b, returnCost=False)
train_set_y_hat = np.round(a_train[L])
m_train = train_set_y.shape[1]
index_train = np.arange(0, m_train)
train_accuracy = len(index_train[train_set_y[0] == train_set_y_hat[0]]) / m_train
print("Train set accuracy: {:.2%}".format(train_accuracy))

a_test = propagation(test_set_x, test_set_y, w, b, returnCost=False)
test_set_y_hat = np.round(a_test[L])
m_test = test_set_y.shape[1]
index_test = np.arange(0, m_test)
test_accuracy = len(index_test[test_set_y[0] == test_set_y_hat[0]]) / m_test
print("Test set accuracy: {:.2%}".format(test_accuracy))
