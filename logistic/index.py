import matplotlib.pyplot as plt
from nnlib.forward import forward
from nnlib.loss import loss
from nnlib.back import differential
from nnlib.getData import getData
import numpy as np

dataset = "./data/" + "square"
x_train, y_train = getData(dataset + "/train.txt")

theta = np.array([-50, 0.05, 0.1])
rate = 0.02
steps = 30000

for i in range(steps + 1):
    y_train_hat = forward(x_train, theta)
    theta = theta - (rate * differential(x_train, y_train_hat, y_train))
    if i % 1000 == 0:
        J = loss(y_train_hat, y_train)
        print("%sth step: loss=%f" % (str(i), J))


np.savetxt("theta.txt", theta)


def test():
    x_test, y_test = getData(dataset + "/test.txt")
    y_test_hat = forward(x_test, theta)
    plt.scatter(x_test[1], x_test[2], c=y_test_hat, cmap="coolwarm")
    plt.colorbar()
    plt.grid(True)
    if dataset == "./data/circle":
        plt.xlim((-1000, 1000))
        plt.ylim((-1000, 1000))
    plt.show()


test()
