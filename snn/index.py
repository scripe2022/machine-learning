import numpy as np
from utils import *
from hyperparameters import *
from plot import plot

np.random.seed(1)

w, b = initialize_parameters()

train_set_x, train_set_y = load_datasets(
    "datasets/train_set_x.txt", "datasets/train_set_y.txt"
)

for i in range(STEPS):
    a, cache = propagation(train_set_x, train_set_y, w, b)
    dw, db = backpropagation(train_set_x, train_set_y, a, w, cache["d"])

    # gradient check
    # dw_check, db_check = gradient_check(train_set_x, train_set_y, w, b)
    # print("{}th step:".format(i))
    # for j in range(len(dw)):
    #     print("dW[{}]".format(j))
    #     print(dw[j] - dw_check[j])
    #     print("db[{}]".format(j))
    #     print(db[j] - db_check[j])
    #     print()

    w = [i - j for i, j in zip(w, [item * RATE for item in dw])]
    b = [i - j for i, j in zip(b, [item * RATE for item in db])]
    if i % 100 == 0:
        print("step: {}, loss: {}".format(i, cache["cost"]))

plot(w, b)
