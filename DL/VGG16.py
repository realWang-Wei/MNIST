import pandas as pd
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import time

def myModel(weights_path=None):
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, kernel_size=3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

PATH = '../data/'
OUT_PATH = '../output/'
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH +'test.csv')
sample = pd.read_csv(PATH + 'sample_submission.csv')

y = train.label
X = train.drop('label', axis=1)

VGG = myModel()
optimizer = SGD()
VGG.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

X_train = X.values/255
X_train = X_train.reshape(-1,28,28,1)     # 这地方一定要用（-1，28，28，1）
y_train = np_utils.to_categorical(y)


# 尝试flyai的一种方法是否可行
start = time.clock()
VGG.fit(X_train, y_train,epochs=32, batch_size=32, validation_split=0.2, verbose=1)
finish = time.clock()
print('Time Cost:', finish - start)

X_test = test.values / 255
X_test = X_test.reshape(-1, 28, 28, 1)

ret = VGG.predict(X_test)

results = np.argmax(ret,axis = 1)

df = {'ImageId':sample['ImageId'],
     'Label':results }

submission = pd.DataFrame(df)

submission.to_csv(OUT_PATH + 'mnist_vgg16.csv', index=False)