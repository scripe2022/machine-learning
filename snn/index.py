import numpy as np
from utils import *
from hyperparameters import *
from plot import plot
import matplotlib.pyplot as plt

w, b = initialize_parameters()

train_set_x, train_set_y = load_datasets(
    "datasets/train_set_x.txt", "datasets/train_set_y.txt"
)
test_set_x, test_set_y = load_datasets(
    "datasets/test_set_x.txt", "datasets/test_set_y.txt"
)

for i in range(len(w)):
    print(w[i].shape)

for i in range(10):
    a, cache = propagation(train_set_x, train_set_y, w, b)
    a_test, cache_test = propagation(test_set_x, test_set_y, w, b)
    dw, db = backpropagation(train_set_x, train_set_y, a, w, cache["d"])

    # gradient check
    dw_check, db_check = gradient_check(train_set_x, train_set_y, w, b)
    print("{}th step:".format(i))
    for j in range(len(dw)):
        print("dW[{}]".format(j))
        print(dw[j] - dw_check[j])
        print("db[{}]".format(j))
        print(db[j] - db_check[j])
        print()

    w = [i - j for i, j in zip(w, [item * RATE for item in dw])]
    b = [i - j for i, j in zip(b, [item * RATE for item in db])]
    if i % 1000 == 0:
        print(
            "step: {}, train_loss: {}, test_loss: {}".format(
                i, cache["cost"], cache_test["cost"]
            )
        )

# plot(w, b)
