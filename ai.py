from matplotlib import pyplot
from PIL import Image
import numpy as np
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #stops agressive error message printing
from tensorflow import keras

# CONFIG
CONFIG_HIDE_GUESS = False #hide guesses while drawing
CONFIG_USE_THRESH_DELTA = True
CONFIG_GUESS_THRESH = 5 #guess threshold (top choice)
CONFIG_SHOW_NUM = 5 #Number of guesses to show

CONFIG_USE_PROB = True
CONFIG_PROB_THRESH = 0.95

def process_image(img: Image.Image):
    img = img.resize((28,28), Image.NEAREST)

     # noinspection PyTypeChecker
    arr = np.asarray(img)
    arr = arr[:,:,3]
    return arr

def run(imlist):
    model_name = 'v3'
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

                pred_labels = sorted(zip(pred, labels), reverse=True)
                ax_gr.clear()

                probability_model = keras.Sequential([model, keras.layers.Softmax()])
                print()

                disp = 0
                if CONFIG_USE_PROB:
                    if max(probability_model.predict(X)[0]) > CONFIG_PROB_THRESH:
                        disp += 1
                else:
                    disp += 1

                if ((CONFIG_USE_THRESH_DELTA and pred_labels[0][0] - pred_labels[1][0] >= CONFIG_GUESS_THRESH) or
                        (not CONFIG_USE_THRESH_DELTA and pred_labels[0][0] >= CONFIG_GUESS_THRESH)):
                    disp += 1

                if disp == 2:
                    ax_gr.text(0.5, 0.5, pred_labels[0][1], fontsize=30, ha='center')

                else:
                    t_labels = [label[1] for label in pred_labels[0:CONFIG_SHOW_NUM]]
                    if CONFIG_HIDE_GUESS:
                        show_labels = []
                        for i in range(0, len(t_labels)):
                            show_labels.append(" " * i + "?" + " " * i)
                    else:
                        show_labels = t_labels
                    ax_gr.bar(show_labels, [pred[0] for pred in pred_labels][0:CONFIG_SHOW_NUM])
                    if not CONFIG_HIDE_GUESS:
                        pyplot.setp(ax_gr.get_xticklabels(), fontsize=10, rotation=45)

        pyplot.draw()
        pyplot.pause(0.01)


        if not pyplot.fignum_exists(fig.number):
            break  # handle exit

