import numpy as np
import matplotlib.pyplot as plt
from nnlib.initialize import initialize_parameters
from nnlib.propagation import propagation
from nnlib.sl import save_parameters
from nnlib.backpropagation import backpropagation

np.random.seed(971228)
S = [2, 3, 3, 1]
L = len(S) - 1
RATE = 0.1
STEPS = 50000
w = [None] * L
b = [None] * L
for i in range(len(S) - 1):
    w[i], b[i] = initialize_parameters(S[i], S[i + 1])
# train_set_x: (2, 350) train_set_y: (350, 1)
# w0: (3, 2)    b0: (3, 1)
# w1: (3, 3)    b1: (3, 1)
# w2: (1, 3)    b2: (1, 1)
train_set_x = np.loadtxt("datasets/train_set_x.txt")
train_set_y = np.loadtxt("datasets/train_set_y.txt")
train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))


for i in range(STEPS):
    a, J = propagation(train_set_x, train_set_y, w, b)
    dw, db = backpropagation(train_set_x, train_set_y, a, w)
    w = [i - j for i, j in zip(w, [item * RATE for item in dw])]
    b = [i - j for i, j in zip(b, [item * RATE for item in db])]
    if i % 1000 == 0:
        print("steps: %d, loss: %f" % (i, J))


save_parameters("parameters.h5", w, b)
