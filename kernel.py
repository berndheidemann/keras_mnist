import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

trainPath="./train.csv"

data=pd.read_csv(trainPath, sep=",", skiprows=1).astype(dtype=float).values

(train, test) = train_test_split(data, test_size=0.1, random_state=42)

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

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        shear_range=0.2, #Scherwinkel
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images




def v1():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), strides=(2,2), padding='same', activation='relu'))
    # model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    return model

def v2():
    model = Sequential()
    model.add(Conv2D(128, (5, 5), input_shape=(28, 28, 1), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), strides=(2,2), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), strides=(2,2), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    return model

#test=pd.read_csv(testPath, sep=",", skiprows=1).astype(dtype=float).values

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

print("trainingdata: %i" % len(trainX))
print("validationdata: %i" % len(testX))

trainX=trainX.reshape(len(trainX), 28, 28, 1)
testX=testX.reshape(len(testX), 28, 28, 1)

trainY=to_categorical(trainY, num_classes=10)
testY=to_categorical(testY, num_classes=10)
model=v1()
checkpoint = ModelCheckpoint('./saves/model-{epoch:02d}-{val_acc:.4f}.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='auto')  

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size=256
datagen.fit(trainX)
model.fit_generator(datagen.flow(trainX, trainY, batch_size=batch_size), validation_data=(testX, testY),
          epochs=300, steps_per_epoch=trainX.shape[0]//batch_size, callbacks=[checkpoint, learning_rate_reduction])
print("hier")