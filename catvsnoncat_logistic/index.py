import h5py
import numpy as np
import matplotlib.pyplot as plt
from nnlib.dataset import load_dataset, flatten, normalization, add_x0
from nnlib.initialize import initialize_theta
from nnlib.optimize import optimize

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, label = load_dataset(
    "datasets/train.h5", "datasets/test.h5"
)
# train_set_x_orig    (393, 128, 128, 3)
# train_set y         (1, 393)
# test_set_x_orig     (99, 128, 128, 3)
# test_set_y          (1, 99)

train_set_x_flatten = flatten(train_set_x_orig)
test_set_x_flatten = flatten(test_set_x_orig)
# train_set_x_flatten (49152, 393) 49152=128*128*3
# test_set_x_flatten  (49152, 99)

train_set_x = add_x0(normalization(train_set_x_flatten))
test_set_x = add_x0(normalization(test_set_x_flatten))
# normalization [0, 1]
# add x0 = ones

theta = initialize_theta(train_set_x_flatten.shape[0] + 1)
# init

steps = 100000
rate = 0.0034
theta, costs = optimize(theta, train_set_x, train_set_y, steps, rate)
np.savetxt("theta.txt", theta)
np.savetxt("costs.txt", costs)
