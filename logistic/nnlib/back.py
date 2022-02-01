import math
import numpy as np


def differential(x, y_hat, y):
    m = len(y_hat)
    return np.dot(x, (y_hat - y)) / (m)
