import math
import numpy as np

# def loss(y, y_hat):
# return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))


def loss(y_hat_list, y_list):
    def J(pairs):
        y_hat = pairs[0]
        y = pairs[1]
        return -(y * math.log(y_hat + 1e-6) + ((1 - y) * math.log(1 - y_hat + 1e-6)))

    return np.mean(np.array(list(map(J, np.array([y_hat_list, y_list]).T))))
