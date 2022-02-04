import sys
import cv2
import numpy as np
from nnlib.dataset import flatten, normalization
import matplotlib.pyplot as plt
from nnlib.propagate import predict

filename = sys.argv[1]

img_file = cv2.resize(cv2.imread(filename), (128, 128), interpolation=cv2.INTER_AREA)
img_orig = np.array([img_file])

img_flatten = flatten(img_orig)
img = normalization(img_flatten)

theta = np.loadtxt("theta.txt")
y_hat = predict(theta, img)

plt.title("Probability of being a cat: {:.2%}".format(y_hat[0]))
plt.imshow(img_file[:, :, [2, 1, 0]])
plt.show()
