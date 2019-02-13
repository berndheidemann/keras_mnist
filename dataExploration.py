import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

def plotMnist(x, y):
    ax = []
    columns = 10
    rows = 10
    w = 28
    h = 28
    fig = plt.figure(figsize=(9, 13))

    for j in range( columns*rows ):
        i = np.random.randint(0, len(x))
        image = x[i] * 255
        title= y[i]
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, j+1) )
        ax[-1].set_title(title)
        plt.imshow(image.astype('uint8'))
    plt.show()

trainPath="./train.csv"
testPath="./test.csv"

train=pd.read_csv(trainPath, sep=",", skiprows=1).astype(dtype=float).values
print(train.shape)

trainX = np.zeros((len(train), 28, 28))
trainY=[]
for i in range(len(train)):
    trainY.append(train[i][0])
    digitArr = np.zeros((28, 28))
    c = 0
    for j in range(0, 768-28, 28):
        digitArr[c] = train[i][j:j + 28]
        c += 1
    trainX[i] = digitArr

test=pd.read_csv(testPath, sep=",", skiprows=1).astype(dtype=float).values


testX = np.zeros((len(test), 28, 28))
testY=[]
for i in range(len(test)):
    testY.append(test[i][0])
    digitArr=np.zeros((28,28))
    c=0
    for j in range(0, 768, 28):
        digitArr[c]=test[i][j:j+28]
        c+=1
    testX[i]=digitArr

trainX = trainX/255.
testX = testX/255.

plotMnist(trainX, trainY)
plotMnist(testX, testY)


