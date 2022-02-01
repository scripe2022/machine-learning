import random
import math
import numpy as np


def getRandom(l, r):
    return int((random.random() * (r - l)) + l)


def getData():
    y = 0 if random.random() > 0.5 else 1
    return [getRandom(l[y], r[y]), getRandom(l[y], r[y]), y]


n = 500
l = [0, 500]
r = [500, 1000]

f_train = open("train.txt", "w")
f_test = open("test.txt", "w")
f_validation = open("validation.txt", "w")

for i in range(int(n * 0.5)):
    data = getData()
    f_train.write("%d %d %d\n" % (data[0], data[1], data[2]))
for i in range(int(n * 0.3)):
    data = getData()
    f_test.write("%d %d %d\n" % (data[0], data[1], data[2]))
for i in range(int(n * 0.2)):
    data = getData()
    f_validation.write("%d %d %d\n" % (data[0], data[1], data[2]))

f_train.close()
f_test.close()
f_validation.close()
