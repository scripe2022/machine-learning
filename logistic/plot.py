import numpy as np
import matplotlib.pyplot as plt
from nnlib.forward import forward

data = [[], [], []]
for line in open("./data/square/test.txt"):
    dataLine = list(map(int, line.strip().split(" ")))
    for i in range(3):
        data[i].append(dataLine[i])

theta = np.loadtxt("theta.txt")

xx, yy = np.meshgrid(np.arange(-50, 1050, 1), np.arange(-50, 1050, 1))


def vec(x, y):
    return [1.0, x, y]


Z = np.array(list(map(vec, xx.reshape(-1), yy.reshape(-1)))).T
zz = forward(Z, theta).reshape(xx.shape[0], xx.shape[1])

plt.contourf(
    xx,
    yy,
    zz,
    cmap=plt.cm.hot,
)
plt.scatter(data[0], data[1], c=data[2], cmap="coolwarm")
plt.show()
