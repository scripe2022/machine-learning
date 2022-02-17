import numpy as np
import h5py


def load_train_set(train_set_file):
    train_dataset = h5py.File(train_set_file, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    label = np.array(train_dataset["list_classes"][:])
    return train_set_x_orig, train_set_y_orig, label


def load_test_set(test_set_file):
    test_dataset = h5py.File(test_set_file, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    return test_set_x_orig, test_set_y_orig
