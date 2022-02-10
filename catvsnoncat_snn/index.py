import numpy as np
import matplotlib.pyplot as plt
from nnlib.initialize import initialize_parameters
from nnlib.propagation import propagation
from nnlib.sl import save_parameters
from nnlib.backpropagation import backpropagation, force
from nnlib.dataset import load_dataset, flatten, normalization

# init
np.random.seed(1)
image_size = 128
S = [image_size * image_size * 3, 35, 7, 5, 1]
L = len(S) - 1
RATE = 0.0075
STEPS = 1000
w = [None] * L
b = [None] * L
for i in range(len(S) - 1):
    w[i], b[i] = initialize_parameters(S[i], S[i + 1])

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, label = load_dataset(
    "datasets/train.h5", "datasets/test.h5"
)
train_set_x_flatten = flatten(train_set_x_orig)
train_set_x = normalization(train_set_x_flatten)
test_set_x_flatten = flatten(test_set_x_orig)
test_set_x = normalization(test_set_x_flatten)

print("RATE: ", RATE)
print("S: ", S)
for i in range(STEPS):
    a, J = propagation(train_set_x, train_set_y, w, b)
    dw, db = backpropagation(train_set_x, train_set_y, a, w)
    w = [i - j for i, j in zip(w, [item * RATE for item in dw])]
    b = [i - j for i, j in zip(b, [item * RATE for item in db])]
    if i % 20 == 0:
        a_train = propagation(train_set_x, train_set_y, w, b, returnCost=False)
        train_set_y_hat = np.round(a_train[L])
        m_train = train_set_y.shape[1]
        index_train = np.arange(0, m_train)
        train_accuracy = (
            len(index_train[train_set_y[0] == train_set_y_hat[0]]) / m_train
        )

        a_test, J_test = propagation(test_set_x, test_set_y, w, b)
        test_set_y_hat = np.round(a_test[L])
        m_test = test_set_y.shape[1]
        index_test = np.arange(0, m_test)
        test_accuracy = len(index_test[test_set_y[0] == test_set_y_hat[0]]) / m_test

        print(
            "steps: {}, loss: {}, train: {:.2%}, test: {:.2%}, test_cost: {}".format(
                i, J, train_accuracy, test_accuracy, J_test
            )
        )


save_parameters("parameters.h5", w, b)
