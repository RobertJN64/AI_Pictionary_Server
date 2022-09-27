import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #stops agressive error message printing
import numpy as np
import json

from MLDashboard.MLDashboardBackend import createDashboard
from MLDashboard.MLCallbacksBackend import DashboardCallbacks, CallbackConfig
from MLDashboard.MLCommunicationBackend import Message, MessageMode

#https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap

SAMPLE_SIZE = 10000 #sample 10 thousand items from each group

def train():
    use_db = input("Use dashboard? y/n ") == "y"
    if use_db:
        dashboardProcess, updatelist, returnlist = createDashboard('dashboard.json')
    else:
        dashboardProcess = None
        updatelist = []
        returnlist = []

    with open('train.json') as f:
        labels = sorted(json.load(f))

    Y = np.empty(0)
    X = np.empty((0, 784))
    print(labels)

    print("Loading", end='')
    for index, item in enumerate(labels):
        X = np.append(X, np.load('qd_files/' + item + '.npy')[0:SAMPLE_SIZE], axis=0)
        yvals = np.empty(SAMPLE_SIZE)
        yvals.fill(index)
        Y = np.append(Y, yvals.copy())
        print('.', end='')
    print()

    print("X Shape:", X.shape, " Y Shape: ", Y.shape)

    #RANDOMIZE
    print("Shuffling data...")
    full = np.append(X, Y[...,None], axis=1)
    np.random.shuffle(full)
    X = full[:, 0:784]
    Y = full[:, 784]

    Y = Y.astype("int")
    X = X.reshape(-1, 28, 28)

    print("Splitting data...")
    splitpoint = int(len(X) * 0.7)
    trainX = X[:splitpoint]
    testX = X[splitpoint:]

    trainY = Y[:splitpoint]
    testY = Y[splitpoint:]

    print("Creating model...")
    from models import adv
    model = adv(len(labels))

    if use_db:
        config = CallbackConfig()
        # labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        callback = DashboardCallbacks(updatelist, returnlist, model, trainX, trainY, testX, testY, labels, config)
    else:
        callback = None


    print("Starting training...")
    model.fit(trainX, trainY, epochs=10, callbacks=callback)
    print("Evaluating model...")
    model.evaluate(testX, testY, batch_size=128, callbacks=callback)

    model.save('models/new/qd_model')
    with open("models/new/qd.json", 'w+') as f:
        json.dump(labels, f, indent=4)

    if use_db:
        updatelist.append(Message(MessageMode.End, {}))
        print("Exiting cleanly...")
        dashboardProcess.join()
        print("Dashboard exited.")
        #callback.HandleRemaingCommands()


if __name__  == '__main__':
    train()