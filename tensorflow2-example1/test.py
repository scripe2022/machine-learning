import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_set_x, train_set_y), (test_set_x, test_set_y) = mnist.load_data()
train_set_x, test_set_x = train_set_x / 100.0, test_set_x / 100.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(train_set_x, train_set_y, epochs=20, batch_size=128)
