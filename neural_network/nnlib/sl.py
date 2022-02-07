import numpy as np
import h5py


def save_parameters(filename, w, b):
    L = len(w)
    with h5py.File(filename, "w") as h5file:
        h5file.create_dataset("L", data=L)
        for i in range(L):
            h5file.create_dataset("w{}".format(i), data=w[i])
        for i in range(L):
            h5file.create_dataset("b{}".format(i), data=b[i])


def load_parameters(filename):
    params = h5py.File(filename, "r")
    w = []
    b = []
    L = np.array(params["L"])
    for i in range(L):
        w.append(np.array(params["w{}".format(i)]))
    for i in range(L):
        b.append(np.array(params["b{}".format(i)]))

    return (L, w, b)
