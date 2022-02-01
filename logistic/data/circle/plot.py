import numpy as np
import matplotlib.pyplot as plt

data = [[], [], []]
for line in open("./train.txt"):
    dataLine = list(map(int, line.strip().split(" ")))
    for i in range(3):
        data[i].append(dataLine[i])

plt.scatter(np.array(data[0]), np.array(data[1]), c=data[2], cmap="coolwarm")
plt.xlim((-1000, 1000))
plt.ylim((-1000, 1000))
plt.show()
