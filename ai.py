from matplotlib import pyplot
from PIL import Image
import numpy as np
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #stops agressive error message printing
from tensorflow import keras

def process_image(img: Image.Image):
    img = img.resize((28,28), Image.NEAREST)

     # noinspection PyTypeChecker
    arr = np.asarray(img)
    arr = arr[:,:,3]
    return arr

def run(imlist):
    model_name = 'basic'
    print("Using model:", model_name)
    with open('models/' + model_name + '/qd.json') as f:
        labels = sorted(json.load(f))
    print("Labels:", labels)
    model = keras.models.load_model("models/" + model_name + "/qd_model")


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

                X = img.reshape(-1, 28, 28).astype("float32")
                pred = model.predict(X)[0]
                ax_gr.clear()
                ax_gr.bar(labels, pred)
                pyplot.setp(ax_gr.get_xticklabels(), fontsize=10, rotation=45)
                print(labels[np.argmax(pred)])

        pyplot.draw()
        pyplot.pause(0.01)


        if not pyplot.fignum_exists(fig.number):
            break  # handle exit

