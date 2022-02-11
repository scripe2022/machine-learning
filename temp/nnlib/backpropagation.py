import numpy as np
from nnlib.propagation import propagation


def relu_prime(a):
    return (a > 0) * 1


def backpropagation(x, y, a, w, d, LAMBDA, ALIVE):
    m = a[0].shape[1]
    L = len(w)
    dz = [None] * (L + 1)
    dw = [None] * L
    db = [None] * L
    da = [None] * L

    dz[L] = a[L] - y
    dw[L - 1] = (1 / m) * np.dot(dz[L], a[L - 1].T) + (LAMBDA / m) * w[L - 1]
    db[L - 1] = (1 / m) * np.sum(dz[L], axis=1, keepdims=True)
    da[L - 1] = np.dot(w[L - 1].T, dz[L])
    da[L - 1] = da[L - 1] * d[L - 1]
    da[L - 1] = da[L - 1] / ALIVE[L]
    for i in range(L - 2, -1, -1):
        dz[i + 1] = np.multiply(
            da[i + 1],
            # np.dot(w[i + 1].T, dz[i + 2]),
            (a[i + 1] > 0) * 1,
        )
        dw[i] = (1 / m) * np.dot(dz[i + 1], a[i].T) + (LAMBDA / m) * w[i]
        db[i] = (1 / m) * np.sum(dz[i + 1], axis=1, keepdims=True)
        da[i] = np.dot(w[i].T, dz[i + 1])
        da[i] = da[i] * d[i]
        da[i] = da[i] / ALIVE[i + 1]
    return (dw, db)


def force(x, y, w, b, LAMBDA, ALIVE):
    m = x.shape[1]
    L = len(w)
    dw = []
    db = []
    delta = 1e-7
    for i in range(L):
        dw.append(np.zeros((w[i].shape[0], w[i].shape[1])))
    for i in range(L):
        db.append(np.zeros((b[i].shape[0], b[i].shape[1])))
    for layer in range(L):
        for i in range(w[layer].shape[0]):
            for j in range(w[layer].shape[1]):
                _, _, J0 = propagation(x, y, w, b, LAMBDA, ALIVE)
                w[layer][i][j] += delta
                _, _, J1 = propagation(x, y, w, b, LAMBDA, ALIVE)
                w[layer][i][j] -= delta
                dw[layer][i][j] = (J1 - J0) / delta
    for layer in range(L):
        for i in range(b[layer].shape[0]):
            for j in range(b[layer].shape[1]):
                _, _, J0 = propagation(x, y, w, b, LAMBDA, ALIVE)
                b[layer][i][j] += delta
                _, _, J1 = propagation(x, y, w, b, LAMBDA, ALIVE)
                b[layer][i][j] -= delta
                db[layer][i][j] = (J1 - J0) / delta

    return dw, db
