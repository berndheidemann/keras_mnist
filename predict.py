from keras.models import load_model
import pickle
import cv2
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


testPath="./test.csv"

test=pd.read_csv(testPath, sep=",").astype(dtype=float).values

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

testX = testX/255.

testX=testX.reshape(len(testX), 28, 28, 1)


model = load_model("./saves/model-61-0.9993.h5")
preds = model.predict(testX)

submission=np.zeros((len(testX), 2), dtype=int)

testY=[]

for i in range(len(testX)):
    submission[i][0]=i+1
    submission[i][1]=np.argmax(preds[i],axis=0)
    testY.append(np.argmax(preds[i],axis=0))

print(submission)
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

testX=testX.reshape(len(testX), 28, 28)

plotMnist(testX, testY )
np.savetxt("submission.csv", X=submission, fmt="%s", delimiter=",", header="ImageId,Label", comments='')