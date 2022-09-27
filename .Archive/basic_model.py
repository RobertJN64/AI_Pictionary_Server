from tensorflow import keras

def get_model(output):
    model = keras.Sequential(
        [keras.layers.Dense(128, activation='relu'),
         keras.layers.Dense(output)])

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])

    return model