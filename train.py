# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #stops agressive error message printing
import tensorflow as tf
from tensorflow import keras

def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(45, input_dim=784, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(35, activation='relu'))
    model.add(keras.layers.Dense(23, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model

def train():
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("Formatting data...")
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    print("Sampling data...")
    # Limit the train data to 10000 samples
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    # Limit test data to 1000 samples
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    print("Creating model...")
    model = get_model()

    print("Starting training...")
    model.fit(x_train, y_train, epochs=50)

    print("Evaluating model...")
    model.evaluate(x_test, y_test, batch_size=128)

    model.save('tf_model')

train()