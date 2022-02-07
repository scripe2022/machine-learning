import numpy as np
from nnlib.activation import sigmoid


def cost(y_hat, y):
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def predict(theta, x):
    return sigmoid(np.dot(theta.T, x))


def propagate(theta, x, y):
    m = x.shape[1]
    y_hat = predict(theta, x)
    J = np.sum(cost(y_hat, y)) / m

    dz = y_hat - y
    # dz (1, 393)
    # x  (49152, 393)
    dtheta = np.dot(x, dz.T) / m

    return dtheta, J
