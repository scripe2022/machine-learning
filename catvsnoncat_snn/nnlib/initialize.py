import numpy as np


def initialize_parameters(currentLayer, nextLayer):
    w = np.random.randn(nextLayer, currentLayer)
    b = np.zeros(shape=(nextLayer, 1))
    return (w, b)
