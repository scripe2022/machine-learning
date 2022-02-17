import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

data_orig = np.load("data.npz")
train_set_x = tf.constant(tf.transpose(data_orig["data"]))
train_set_y = tf.constant(data_orig["label"].reshape((-1)))

# print(train_set_x.shape)
# print(train_set_y.shape)
input_x = keras.Input(shape=(2,), name="input")
hidden1 = layers.Dense(7, activation="tanh", name="layer1")(input_x)
hidden2 = layers.Dense(5, activation="tanh", name="layer2")(hidden1)
pred = layers.Dense(1, name="output")(hidden2)
model = keras.Model(inputs=input_x, outputs=pred)

# model = tf.keras.models.Sequential(
#     [
#         layers.InputLayer(input_shape=(2,), name="input"),
#         layers.Dense(7, activation="tanh", name="layer1"),
#         layers.Dense(5, activation="tanh", name="layer2"),
#         layers.Dense(1, activation="sigmoid", name="output"),
#     ]
# )

# model.summary()
# keras.utils.plot_model(model, "model_info.png", show_shapes=True)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=tf.keras.metrics.BinaryAccuracy(),
)
model.fit(train_set_x, train_set_y, epochs=5, batch_size=64)
