import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_dataset():
    train_dataset = h5py.File("datasets/train.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File("datasets/test.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_set_x, train_set_y, test_set_x, test_set_y, label = load_dataset()


index = 6

plt.title(
    "%dth image in train set, %s"
    % (
        index,
        label[train_set_y[0][index]].decode("utf-8"),
    )
)
plt.imshow(train_set_x[index])
plt.show()
