import h5py
import numpy as np
import matplotlib.pyplot as plt
from nnlib.dataset import load_dataset, flatten, normalization, add_x0
from nnlib.propagate import predict

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, label = load_dataset(
    "datasets/train.h5", "datasets/test.h5"
)

train_set_x_flatten = flatten(train_set_x_orig)
test_set_x_flatten = flatten(test_set_x_orig)

train_set_x = add_x0(normalization(train_set_x_flatten))
test_set_x = add_x0(normalization(test_set_x_flatten))

theta = np.loadtxt("theta.txt")
costs = np.loadtxt("costs.txt")
costs = costs.reshape(costs.shape[0], 1)
theta = theta.reshape(theta.shape[0], 1)

train_set_y_hat = predict(theta, train_set_x)
test_set_y_hat = predict(theta, test_set_x)

threshold = 0.5
train_set_error = 0
for i in range(train_set_y_hat.shape[1]):
    if abs(train_set_y_hat[0][i] - train_set_y[0][i]) > threshold:
        train_set_error += 1
test_set_error = 0
for i in range(test_set_y_hat.shape[1]):
    if abs(test_set_y_hat[0][i] - test_set_y[0][i]) > threshold:
        test_set_error += 1
print("train set: {:.2%}".format(1 - (train_set_error / train_set_y_hat.shape[1])))
print("test set: {:.2%}".format(1 - (test_set_error / test_set_y_hat.shape[1])))

plt.plot(np.array(range(costs.shape[0]))[::1000], costs[::1000])
plt.ylim((0, 0.3))
plt.show()
