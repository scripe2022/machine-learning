import h5py
import numpy as np


def load_dataset(train_dataset_file, test_dataset_file):
    train_dataset = h5py.File(train_dataset_file, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File(test_dataset_file, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def flatten(dataset_orig):
    dataset_flatten = dataset_orig.reshape(dataset_orig.shape[0], -1).T
    return dataset_flatten


def normalization(dataset_flatten):
    dataset = dataset_flatten / 255
    return dataset
