from tensorflow import keras

#https://thomasdelatte.com/2020/04/quickdraw/

def get_model(output):
    model = keras.Sequential()
    model.add(keras.layers.Convolution2D(16, (3, 3),
                            padding="same",
                            input_shape=(28,28,1), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Convolution2D(32, (3, 3), padding="same", activation= "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Convolution2D(64, (3, 3), padding="same", activation= "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size =(2,2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(output))

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])

    return model
