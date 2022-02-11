import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def my_log(y):
    return np.log(y)


def J(w, y_hat, y, LAMBDA):
    m = y_hat.shape[1]
    cost = np.multiply(y, my_log(y_hat)) + np.multiply(1 - y, my_log(1 - y_hat))
    cost_sum = (-1 / m) * np.sum(cost)
    for i in range(len(w)):
        cost_sum += (LAMBDA / (2 * m)) * np.sum(w[i] ** 2)
    return cost_sum


def propagation(x, y, w, b, LAMBDA, ALIVE, returnCost=True):
    L = len(w)
    a = []
    d = []

    a.append(x)
    d.append(np.random.rand(a[0].shape[0], a[0].shape[1]))
    d[0] = d[0] <= ALIVE[0]
    a[0] = a[0] * d[0]
    a[0] = a[0] / ALIVE[0]
    for i in range(1, L + 1):
        if i < L:
            a.append(relu(np.dot(w[i - 1], a[i - 1]) + b[i - 1]))
        else:
            a.append(sigmoid(np.dot(w[i - 1], a[i - 1]) + b[i - 1]))
        d.append(np.random.rand(a[i].shape[0], a[i].shape[1]))
        d[i] = d[i] <= ALIVE[i]
        a[i] = a[i] * d[i]
        a[i] = a[i] / ALIVE[i]

    if returnCost:
        return (a, d, J(w, a[L], y, LAMBDA))
    else:
        return (a, d)
