import tensorflow as tf
import numpy as np

data_orig = np.load("data.npz")
train_set_x = tf.constant(tf.transpose(data_orig["data"]), dtype="float64")
train_set_y = tf.constant(data_orig["label"].reshape((1, -1)), dtype="float64")

w = tf.Variable(tf.zeros((1, 2), tf.float64), dtype="float64")
b = tf.Variable(0, dtype="float64")


def cost(y_hat, y):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat)
    )


def propagation(x, w, b):
    return w @ x + b


y_hat = propagation(train_set_x, w, b)
J = cost(y_hat, train_set_y)
