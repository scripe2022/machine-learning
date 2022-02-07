import numpy as np
from nnlib.propagation import propagation
from nnlib.sl import load_parameters
import matplotlib.pyplot as plt


train_set_x = np.loadtxt("datasets/train_set_x.txt")
train_set_y = np.loadtxt("datasets/train_set_y.txt")
train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
test_set_x = np.loadtxt("datasets/test_set_x.txt")
test_set_y = np.loadtxt("datasets/test_set_y.txt")
test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

L, w, b = load_parameters("parameters.h5")

a_train = propagation(train_set_x, train_set_y, w, b, returnCost=False)
a_test = propagation(test_set_x, test_set_y, w, b, returnCost=False)
y_hat_train = np.round(a_train[L])
y_hat_test = np.round(a_test[L])


m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
index_train = np.arange(0, m_train)
index_test = np.arange(0, m_test)

train_accuracy = len(index_train[train_set_y[0] == y_hat_train[0]]) / m_train
print("Train set accuracy: {:.2%}".format(train_accuracy))

test_accuracy = len(index_test[test_set_y[0] == y_hat_test[0]]) / m_test
print("Test set accuracy: {:.2%}".format(test_accuracy))


xx, yy = np.meshgrid(np.arange(-11, 11, 0.1), np.arange(-11, 11, 0.1))


def vec(x, y):
    return [x, y]


Z = np.array(list(map(vec, xx.reshape(-1), yy.reshape(-1)))).T
a_contour = propagation(Z, test_set_y, w, b, returnCost=False)
zz = a_contour[L].reshape(xx.shape[0], xx.shape[1])

plt.contourf(
    xx,
    yy,
    zz,
    cmap=plt.cm.hot,
)
plt.scatter(test_set_x[0][:], test_set_x[1][:], c=test_set_y[0][:], cmap="coolwarm")
plt.colorbar()
plt.show()
