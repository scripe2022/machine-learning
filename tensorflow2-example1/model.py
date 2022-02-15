import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

data_orig = np.load("data.npz")
train_set_x = tf.constant(tf.transpose(data_orig["data"]), dtype="float64")
train_set_y = tf.constant(data_orig["label"].reshape((-1)), dtype="float64")

# print(train_set_x.shape)
# print(train_set_y.shape)
# input_x = keras.Input(shape=(2,), name="input")
# hidden1 = layers.Dense(7, activation="relu", name="layer1")(input_x)
# hidden2 = layers.Dense(5, activation="relu", name="layer2")(hidden1)
# pred = layers.Dense(1, name="output")(hidden2)
# model = keras.Model(inputs=input_x, outputs=pred)

model = tf.keras.models.Sequential(
    [
        layers.InputLayer(input_shape=(2,)),
        layers.Dense(7, activation="tanh"),
        layers.Dense(5, activation="tanh"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# model.summary()
# keras.utils.plot_model(model, "model_info.png", show_shapes=True)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=["accuracy"],
)
model.fit(train_set_x, train_set_y, epochs=50, batch_size=64)
