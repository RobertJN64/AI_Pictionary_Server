#https://dev.to/larswaechter/recognizing-hand-drawn-doodles-using-deep-learning-ki0

from matplotlib import pyplot
from PIL import Image
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #stops agressive error message printing
from tensorflow import keras

def process_image(img: Image.Image):
    img = img.resize((28,28))

     # noinspection PyTypeChecker
    arr = np.asarray(img)
    arr = arr[:,:,3]
    return arr

def run(imlist):
    model = keras.models.load_model("tf_model")


    fig, (ax_im, ax_gr) = pyplot.subplots(1, 2, figsize=(10, 5))
    pyplot.show(block=False)

    lastim = None

    while True:
        if len(imlist) > 0:
            if lastim != imlist[0]:
                lastim = imlist[0]
                ax_im.clear()
                img = process_image(imlist[0])
                ax_im.imshow(img, cmap='gray')

                X = img.reshape(-1, 784).astype("float32") / 255.0
                pred = model.predict(X)[0]
                ax_gr.clear()
                ax_gr.bar(list(map(str, range(0, 10))), pred)
                print(np.argmax(pred))

        pyplot.draw()
        pyplot.pause(0.01)


        if not pyplot.fignum_exists(fig.number):
            break  # handle exit

