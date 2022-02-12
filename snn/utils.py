import numpy as np
from hyperparameters import *


def initialize_parameters():
    def initialize_wb(currentLayer, nextLayer):
        wi = np.random.randn(nextLayer, currentLayer) / np.sqrt(currentLayer)
        bi = np.zeros(shape=(nextLayer, 1))
        return (wi, bi)

    w = [None] * L
    b = [None] * L
    for i in range(L):
        w[i], b[i] = initialize_wb(S[i], S[i + 1])
    return (w, b)


def load_datasets(train_set_x_file, train_set_y_file):
    train_set_x = np.loadtxt(train_set_x_file)
    train_set_y = np.loadtxt(train_set_y_file)
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    return (train_set_x, train_set_y)


def activation(z, func):
    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    def relu(t):
        return np.maximum(0, t)

    def tanh(t):
        return np.tanh(t)

    if func == "sigmoid":
        return sigmoid(z)
    elif func == "relu":
        return relu(z)
    elif func == "tanh":
        return tanh(z)


def cost(w, y_hat, y):
    def log(y):
        return np.log(y)

    m = y_hat.shape[1]
    lost = np.multiply(y, log(y_hat)) + np.multiply(1 - y, log(1 - y_hat))
    J = (-1 / m) * np.sum(lost)
    for i in range(len(w)):
        J += (LAMBDA / (2 * m)) * np.sum(w[i] ** 2)
    return J


def propagation(x, y, w, b):
    a = []
    d = []
    for i in range(0, L + 1):
        if i == 0:
            a.append(x)
        else:
            z = np.dot(w[i - 1], a[i - 1]) + b[i - 1]
            a.append(activation(z, ACT[i - 1]))
        d.append(np.random.rand(a[i].shape[0], a[i].shape[1]))
        d[i] = d[i] <= ALIVE[i]
        a[i] = a[i] * d[i]
        a[i] = a[i] / ALIVE[i]
    cache = {
        "cost": cost(w, a[L], y),
        "d": d,
    }
    return (a, cache)


def activation_prime(a, func):
    def sigmoid_prime(t):
        return t * (1 - t)

    def tanh_prime(t):
        return 1 - (t**2)

    def relu_prime(t):
        return (t > 0) * 1

    if func == "sigmoid":
        return sigmoid_prime(a)
    elif func == "tanh":
        return tanh_prime(a)
    elif func == "relu":
        return relu_prime(a)


def backpropagation(x, y, a, w, d):
    m = a[0].shape[1]
    dz = [None] * (L + 1)
    dw = [None] * L
    db = [None] * L
    da = [None] * L

    for i in range(L - 1, -1, -1):
        if i == (L - 1):
            dz[i + 1] = a[i + 1] - y
        else:
            dz[i + 1] = np.multiply(da[i + 1], activation_prime(a[i + 1], ACT[i]))
        dw[i] = (1 / m) * np.dot(dz[i + 1], a[i].T) + (LAMBDA / m) * w[i]
        db[i] = (1 / m) * np.sum(dz[i + 1], axis=1, keepdims=True)
        da[i] = np.dot(w[i].T, dz[i + 1])
        da[i] = da[i] * d[i]
        da[i] = da[i] / ALIVE[i]
    return (dw, db)


def gradient_check(x, y, w, b):
    dw = []
    db = []
    delta = 1e-7
    for i in range(L):
        dw.append(np.zeros((w[i].shape[0], w[i].shape[1])))
        db.append(np.zeros((b[i].shape[0], b[i].shape[1])))
    for layer in range(L):
        for i in range(w[layer].shape[0]):
            for j in range(w[layer].shape[1]):
                w[layer][i][j] -= delta
                _, cache0 = propagation(x, y, w, b)
                w[layer][i][j] += 2 * delta
                _, cache1 = propagation(x, y, w, b)
                w[layer][i][j] -= delta
                dw[layer][i][j] = (cache1["cost"] - cache0["cost"]) / (2 * delta)
    for layer in range(L):
        for i in range(b[layer].shape[0]):
            for j in range(b[layer].shape[1]):
                b[layer][i][j] -= delta
                _, cache0 = propagation(x, y, w, b)
                b[layer][i][j] += 2 * delta
                _, cache1 = propagation(x, y, w, b)
                b[layer][i][j] -= delta
                db[layer][i][j] = (cache1["cost"] - cache0["cost"]) / (2 * delta)
    return (dw, db)
