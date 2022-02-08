import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def my_log(y):
    return np.log(y + 1e-9)


def J(y_hat, y):
    m = y_hat.shape[1]
    cost = np.multiply(y, my_log(y_hat)) + np.multiply(1 - y, my_log(1 - y_hat))
    return (-1 / m) * np.sum(cost)


def propagation(x, y, w, b, returnCost=True):
    L = len(w)
    a = []
    a.append(x)
    for i in range(1, L):
        a.append(relu(np.dot(w[i - 1], a[i - 1]) + b[i - 1]))
    a.append(sigmoid(np.dot(w[L - 1], a[L - 1]) + b[L - 1]))

    if returnCost:
        return (a, J(a[L], y))
    else:
        return a
