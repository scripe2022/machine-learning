import random
import math
import numpy as np
import matplotlib.pyplot as plt


def getRandom(l, r):
    return int((random.random() * (r - l)) + l)


def getData():
    y = 0 if random.random() > 0.5 else 1
    radius = getRandom(l[y], r[y])
    theta = random.random() * 2 * math.pi
    dataX = radius * math.cos(theta)
    dataY = radius * math.sin(theta)
    return [dataX, dataY, y]


n = 500
l = [0, 700]
r = [300, 1000]

train_set_x = []
train_set_y = []
for i in range(int(n * 0.7)):
    data = getData()
    train_set_x.append(data[:2])
    train_set_y.append(data[2])
train_set_x = np.array(train_set_x).T
train_set_y = np.array(train_set_y)
train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))

test_set_x = []
test_set_y = []
for i in range(int(n * 0.3)):
    data = getData()
    test_set_x.append(data[:2])
    test_set_y.append(data[2])
test_set_x = np.array(test_set_x).T
test_set_y = np.array(test_set_y)
test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))


np.savetxt("train_set_x.txt", train_set_x)
np.savetxt("train_set_y.txt", train_set_y)

np.savetxt("test_set_x.txt", test_set_x)
np.savetxt("test_set_y.txt", test_set_y)


plt.scatter(train_set_x[0][:], train_set_x[1][:], c=train_set_y, cmap="coolwarm")
plt.colorbar()
plt.show()
