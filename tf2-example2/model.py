from tkinter import W
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import *
import time
import datetime

train_set_x_orig, train_set_y_orig, label = load_train_set("datasets/train.h5")
test_set_x_orig, test_set_y_orig = load_test_set("datasets/test.h5")

train_set_x = tf.constant(train_set_x_orig / 255)
train_set_y = tf.constant(train_set_y_orig)
test_set_x = tf.constant(test_set_x_orig / 255)
test_set_y = tf.constant(test_set_y_orig)

LAMBDA = 0.001
DROP = {"hidden1": 0, "hidden2": 0.05, "hidden3": 0}
EPOCHS = 1000
model = tf.keras.models.Sequential(
    [
        # input
        keras.layers.InputLayer(input_shape=(128, 128, 3), name="input"),
        keras.layers.Flatten(name="flatten"),
        # layer1
        keras.layers.Dense(
            units=35,
            kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode="fan_in"),
            kernel_regularizer=keras.regularizers.l2(LAMBDA),
            name="hidden1",
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu", name="hidden1_relu"),
        keras.layers.Dropout(rate=DROP["hidden1"]),
        # layer2
        keras.layers.Dense(
            units=7,
            kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode="fan_in"),
            kernel_regularizer=keras.regularizers.l2(LAMBDA),
            name="hidden2",
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu", name="hidden2_relu"),
        keras.layers.Dropout(rate=DROP["hidden2"]),
        # layer3
        keras.layers.Dense(
            units=5,
            kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode="fan_in"),
            kernel_regularizer=keras.regularizers.l2(LAMBDA),
            name="hidden3",
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu", name="hidden3_relu"),
        keras.layers.Dropout(rate=DROP["hidden3"]),
        # output
        keras.layers.Dense(
            units=1,
            kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode="fan_in"),
            kernel_regularizer=keras.regularizers.l2(LAMBDA),
            name="output",
        ),
        keras.layers.Activation("sigmoid", name="output_sigmoid"),
    ]
)


ts = time.time()


class NBatchLogger(keras.callbacks.Callback):
    def __init__(self, N=10, logs=None):
        self.N = N

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.N == 0:
            global ts
            print(
                "{:5} epoch, loss: {:.4f}, accuracy: {:.2%}, val_loss: {:.4f}, val_accuracy: {:.2%}, time: {:.1f}ms".format(
                    epoch,
                    logs["loss"],
                    logs["binary_accuracy"],
                    logs["val_loss"],
                    logs["val_binary_accuracy"],
                    (time.time() - ts) * 1000,
                )
            )
            ts = time.time()


# keras.utils.plot_model(model, "model_info.png", show_shapes=True)
model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=tf.keras.metrics.BinaryAccuracy(),
)
N = 20
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#     log_dir=log_dir, histogram_freq=N
# )
model.fit(
    train_set_x,
    train_set_y,
    epochs=EPOCHS,
    batch_size=64,
    verbose=0,
    validation_data=(test_set_x, test_set_y),
    validation_freq=N,
    callbacks=[
        NBatchLogger(N=N),
        # tensorboard_callback,
    ],
)
