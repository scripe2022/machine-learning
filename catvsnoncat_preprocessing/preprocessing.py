import random
import h5py
import glob
import cv2
import numpy as np

HEIGHT = 128
WIDTH = 128
DIM = (WIDTH, HEIGHT)

dataset = []
y = []
label = ["noncat", "cat"]

PATH = ["./images/noncat/*", "./images/cat/*"]

# load images
for i in range(2):
    # 0: noncat
    # 1: cat
    for filename in glob.glob(PATH[i]):
        print(filename)
        img = cv2.resize(cv2.imread(filename), DIM, interpolation=cv2.INTER_AREA)
        dataset.append(img)
        y.append(i)

# random shuffle
SEED = 971228
random.seed(SEED)
random.shuffle(dataset)
random.seed(SEED)
random.shuffle(y)

# store as hdf5
TRAIN = 0.8
n = len(dataset)

with h5py.File("./datasets/train.h5", "w") as train_h5:
    train_h5.create_dataset("train_set_x", data=dataset[: int(n * TRAIN)])
    train_h5.create_dataset("train_set_y", data=y[: int(n * TRAIN)])
    train_h5.create_dataset("list_classes", data=np.array(label, dtype="S"))
with h5py.File("./datasets/test.h5", "w") as test_h5:
    test_h5.create_dataset("test_set_x", data=dataset[int(n * TRAIN) :])
    test_h5.create_dataset("test_set_y", data=y[int(n * TRAIN) :])
    test_h5.create_dataset("list_classes", data=np.array(label, dtype="S"))
