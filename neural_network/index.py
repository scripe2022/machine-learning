import h5py
import numpy as np
import matplotlib.pyplot as plt
from nnlib.initialize import initialize_parameters
from nnlib.propagation import propagation
from nnlib.backpropagation import backpropagation, force

np.random.seed(971228)
L = 3
S = [2, 3, 3, 1]
RATE = 0.1
STEPS = 50000
train_set_x = np.loadtxt("datasets/train_set_x.txt")
train_set_y = np.loadtxt("datasets/train_set_y.txt")
train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
w0, b0 = initialize_parameters(S[0], S[1])
w1, b1 = initialize_parameters(S[1], S[2])
w2, b2 = initialize_parameters(S[2], S[3])
# train_set_x: (2, 350) train_set_y: (350, 1)
# w0: (3, 2)    b0: (3, 1)
# w1: (3, 3)    b1: (3, 1)
# w2: (1, 3)    b2: (1, 1)

for i in range(STEPS):
    a1, a2, a3, J = propagation(train_set_x, train_set_y, w0, b0, w1, b1, w2, b2)
    diff = backpropagation(a1, a2, a3, w0, w1, w2, train_set_x, train_set_y)
    w0 -= RATE * diff["dw0"]
    b0 -= RATE * diff["db0"]
    w1 -= RATE * diff["dw1"]
    b1 -= RATE * diff["db1"]
    w2 -= RATE * diff["dw2"]
    b2 -= RATE * diff["db2"]
    if i % 1000 == 0:
        print("steps: %d, loss: %f" % (i, J))

parameters = {
    "w0": w0,
    "b0": b0,
    "w1": w1,
    "b1": b1,
    "w2": w2,
    "b2": b2,
}
np.savez("parameters.npz", w0=w0, b0=b0, w1=w1, b1=b1, w2=w2, b2=b2)
