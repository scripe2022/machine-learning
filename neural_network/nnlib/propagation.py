import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def J(a3, y):
    m = a3.shape[1]
    cost = np.multiply(y, np.log(a3)) + np.multiply(1 - y, np.log(1 - a3))
    return (-1 / m) * np.sum(cost)


def propogation(x, y, w0, b0, w1, b1, w2, b2):
    z1 = np.dot(w0, x) + b0
    # z1: (3, 350)
    a1 = np.tanh(z1)

    z2 = np.dot(w1, a1) + b1
    # z2: (3, 350)
    a2 = np.tanh(z2)

    z3 = np.dot(w2, a2) + b2
    # z3: (1, 350)
    a3 = sigmoid(z3)

    return (a1, a2, a3, J(a3, y))
