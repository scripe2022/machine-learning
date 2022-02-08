import numpy as np
from nnlib.propagation import propagation


def relu_prime(a):
    return (a > 0) * 1


def backpropagation(x, y, a, w):
    m = a[0].shape[1]
    L = len(w)
    dz = [None] * (L + 1)
    dw = [None] * L
    db = [None] * L

    dz[L] = a[L] - y
    dw[L - 1] = (1 / m) * np.dot(dz[L], a[L - 1].T)
    db[L - 1] = (1 / m) * np.sum(dz[L], axis=1, keepdims=True)
    for i in range(L - 2, -1, -1):
        dz[i + 1] = np.multiply(
            np.dot(w[i + 1].T, dz[i + 2]),
            (a[i + 1] > 0) * 1,
        )
        dw[i] = (1 / m) * np.dot(dz[i + 1], a[i].T)
        db[i] = (1 / m) * np.sum(dz[i + 1], axis=1, keepdims=True)
    return (dw, db)


def force(x, y, w, b):
    m = x.shape[1]
    L = len(w)
    dw = []
    db = []
    delta = 1e-9
    for i in range(L):
        dw.append(np.zeros((w[i].shape[0], w[i].shape[1])))
    for i in range(L):
        db.append(np.zeros((b[i].shape[0], b[i].shape[1])))
    for layer in range(L):
        for i in range(w[layer].shape[0]):
            for j in range(w[layer].shape[1]):
                _, J0 = propagation(x, y, w, b)
                w[layer][i][j] += delta
                _, J1 = propagation(x, y, w, b)
                w[layer][i][j] -= delta
                dw[layer][i][j] = (J1 - J0) / delta
    for layer in range(L):
        for i in range(b[layer].shape[0]):
            for j in range(b[layer].shape[1]):
                _, J0 = propagation(x, y, w, b)
                b[layer][i][j] += delta
                _, J1 = propagation(x, y, w, b)
                b[layer][i][j] -= delta
                db[layer][i][j] = (J1 - J0) / delta

    return dw, db
