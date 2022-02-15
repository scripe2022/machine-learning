import numpy as np
import matplotlib.pyplot as plt

n = 300
p = np.random.permutation(2 * n)
x = np.append(np.random.uniform(-1, 0, (n)), np.random.uniform(0, 1, (n)))[p]
y = np.append(np.random.uniform(-1, 0, (n)), np.random.uniform(0, 1, (n)))[p]
label = np.append(np.zeros(n), np.ones(n))[p]
data = np.append(x.reshape(1, -1), y.reshape(1, -1), axis=0)

plt.scatter(x, y, c=label, cmap="coolwarm")
plt.colorbar()
plt.show()

np.savez("data.npz", data=data, label=label)
