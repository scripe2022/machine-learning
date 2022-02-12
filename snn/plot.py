import numpy as np
import matplotlib.pyplot as plt
from utils import *
from hyperparameters import *

train_set_x, train_set_y = load_datasets(
    "datasets/train_set_x.txt", "datasets/train_set_y.txt"
)
test_set_x, test_set_y = load_datasets(
    "datasets/test_set_x.txt", "datasets/test_set_y.txt"
)


def plot(w, b):
    xx, yy = np.meshgrid(np.arange(-11, 11, 0.1), np.arange(-11, 11, 0.1))

    def vec(x, y):
        return [x, y]

    Z = np.array(list(map(vec, xx.reshape(-1), yy.reshape(-1)))).T
    a_contour, _ = propagation(Z, np.zeros((2, Z.shape[1])), w, b)
    zz = a_contour[L].reshape(xx.shape[0], xx.shape[1])

    plt.contourf(xx, yy, zz, cmap=plt.cm.hot)
    plt.scatter(test_set_x[0][:], test_set_x[1][:], c=test_set_y[0][:], cmap="coolwarm")
    plt.colorbar()
    plt.show()
