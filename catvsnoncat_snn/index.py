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
S = [image_size * image_size * 3, 14, 7, 1]
# S = [2, 3, 3, 1]
L = len(S) - 1
RATE = 0.0075
STEPS = 3000
w = [None] * L
b = [None] * L
for i in range(len(S) - 1):
    w[i], b[i] = initialize_parameters(S[i], S[i + 1])

# train set
# train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, label = load_dataset(
#     "datasets/train_catvnoncat.h5", "datasets/test_catvnoncat.h5"
# )
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, label = load_dataset(
    "datasets/train.h5", "datasets/test.h5"
)
train_set_x_flatten = flatten(train_set_x_orig)
train_set_x = normalization(train_set_x_flatten)

# train_set_x = np.loadtxt("datasets/train_set_x.txt")
# train_set_y = np.loadtxt("datasets/train_set_y.txt")
# train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))


for i in range(STEPS):
    a, J = propagation(train_set_x, train_set_y, w, b)
    dw, db = backpropagation(train_set_x, train_set_y, a, w)
    # force_dw, force_db = force(train_set_x, train_set_y, w, b)
    # for j in range(len(dw)):
    #     print(dw[j] - force_dw[j])
    #     print()
    #     print(db[j] - force_db[j])
    #     print()
    #     print()
    w = [i - j for i, j in zip(w, [item * RATE for item in dw])]
    b = [i - j for i, j in zip(b, [item * RATE for item in db])]
    if i % 100 == 0:
        print("steps: %d, loss: %f" % (i, J))


save_parameters("parameters.h5", w, b)
