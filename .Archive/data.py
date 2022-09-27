import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #stops agressive error message printing
import tensorflow as tf
from matplotlib import pyplot
import random

def showdata():
    print("Loading data...")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x = random.randint(0, 100)
    pyplot.imshow(x_train[x], cmap='gray')
    print(y_train[x])
    pyplot.show()

showdata()