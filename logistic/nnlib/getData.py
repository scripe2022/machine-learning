import numpy as np


def getData(filename):
    data = [[], [], []]
    y = []
    for line in open(filename):
        dataLine = list(map(int, line.strip().split(" ")))
        for i in range(2):
            data[i + 1].append(float(dataLine[i]))
        data[0].append(1.0)
        y.append(float(dataLine[2]))
    return (np.array(data), np.array(y))
