import numpy as np
from nnlib.propagate import propagate


def optimize(theta, x, y, steps, rate):
    J_list = []
    for i in range(steps):
        dtheta, J = propagate(theta, x, y)
        J_list.append(J)
        theta -= rate * dtheta
        if i % 100 == 0:
            print("steps %d, loss %f" % (i, J))
    return theta, J_list
