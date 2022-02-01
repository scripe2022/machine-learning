import numpy as np
import math


def forward(x, theta):
    def activation(t):
        if t >= 0:
            return 1.0 / (1 + math.exp(-t))
        else:
            return np.exp(t) / (1 + math.exp(t))

    return np.array(list(map(activation, np.dot(theta.T, x))))
