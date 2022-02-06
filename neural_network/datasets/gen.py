import random
import math
import numpy as np
import matplotlib.pyplot as plt


def getRandom(l, r):
    return (random.random() * (r - l)) + l


def getData():
    y = 0 if random.random() > 0.5 else 1
    radius = getRandom(l[y], r[y])
    theta = random.random() * 2 * math.pi
    dataX = radius * math.cos(theta)
    dataY = radius * math.sin(theta)
    return [dataX, dataY, y]


n = 1000
l = [0, 5]
r = [5, 10]

train_set_x = []
train_set_y = []
for i in range(int(n * 0.7)):
    data = getData()
    train_set_x.append(data[:2])
    train_set_y.append(data[2])
train_set_x = np.array(train_set_x).T
train_set_y = np.array(train_set_y)

test_set_x = []
test_set_y = []
for i in range(int(n * 0.3)):
    data = getData()
    test_set_x.append(data[:2])
    test_set_y.append(data[2])
test_set_x = np.array(test_set_x).T
test_set_y = np.array(test_set_y)


np.savetxt("train_set_x.txt", train_set_x)
np.savetxt("train_set_y.txt", train_set_y)

np.savetxt("test_set_x.txt", test_set_x)
np.savetxt("test_set_y.txt", test_set_y)


plt.scatter(train_set_x[0][:], train_set_x[1][:], c=train_set_y, cmap="coolwarm")
plt.colorbar()
plt.show()
