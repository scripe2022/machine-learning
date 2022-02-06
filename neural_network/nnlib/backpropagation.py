import numpy as np
from nnlib.propagation import propagation


def backpropagation(a1, a2, a3, w0, w1, w2, x, y):
    m = a1.shape[1]
    # a1: (3, 350)
    # a2: (3, 350)
    # a3: (1, 350)
    dz3 = a3 - y
    # dz3: (1, 350)
    dw2 = (1 / m) * np.dot(dz3, a2.T)
    db2 = (1 / m) * np.sum(dz3, axis=1, keepdims=True)

    dz2 = np.multiply(np.dot(w2.T, dz3), 1 - np.power(a2, 2))
    dw1 = (1 / m) * np.dot(dz2, a1.T)
    db1 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.multiply(np.dot(w1.T, dz2), 1 - np.power(a1, 2))
    dw0 = (1 / m) * np.dot(dz1, x.T)
    db0 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    return {
        "dw0": dw0,
        "db0": db0,
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }
