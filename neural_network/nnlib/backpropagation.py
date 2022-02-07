import numpy as np
from nnlib.propagation import propagation


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
            np.dot(w[i + 1].T, dz[i + 2]), 1 - np.power(a[i + 1], 2)
        )
        dw[i] = (1 / m) * np.dot(dz[i + 1], a[i].T)
        db[i] = (1 / m) * np.sum(dz[i + 1], axis=1, keepdims=True)
    return (dw, db)
